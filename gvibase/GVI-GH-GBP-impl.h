
#pragma once

#ifndef GVI_GH_IMPL_H
#define GVI_GH_IMPL_H

using namespace Eigen;

#include <stdexcept>
#include <optional>
#include <omp.h>
#include "helpers/CudaOperation.h"

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace gvi{

template <typename Factor>
void GVIGH<Factor>::switch_to_high_temperature(){
    
    #pragma omp parallel for
    for (auto& i_factor : _vec_factors) {
        i_factor->factor_switch_to_high_temperature();
    }
    this->_temperature = this->_high_temperature;
    // this->initilize_precision_matrix();
}


/**
 * @brief optimize with backtracking
 */ 
template <typename Factor>
void GVIGH<Factor>::optimize(std::optional<bool> verbose)
{
    // default verbose
    bool is_verbose = verbose.value_or(true);
    bool is_lowtemp = true;
    bool converged = false;

    for (int i_iter = 0; i_iter < _niters; i_iter++)
    {   

        if (converged){
            break;
        }

        // ============= High temperature phase =============
        if (i_iter == _niters_lowtemp && is_lowtemp){
            if (is_verbose){
                std::cout << "Switching to high temperature.." << std::endl;
            }
            if (is_verbose){
                std::cout << "Switching to high temperature.." << std::endl;
            }
            this->switch_to_high_temperature();
            is_lowtemp = false;
        }

        // ============= Cost at current iteration =============
        double cost_iter = this->cost_value(); // -> Base::cost_value(this->_mu, this->_precision);

        if (is_verbose){
            std::cout << "========= iteration " << i_iter << " ========= " << std::endl;
            std::cout << "--- cost_iter ---" << std::endl << cost_iter << std::endl;
        }

        // ============= Collect factor costs =============
        VectorXd fact_costs_iter = this->factor_cost_vector();
        // std::cout << "Factor Costs:" << fact_costs_iter.transpose() << std::endl;

        _res_recorder.update_data(_mu, _covariance, _precision, cost_iter, fact_costs_iter);

        // gradients
        std::tuple<VectorXd, SpMat> gradients = compute_gradients(); //Used calculate partial V here

        VectorXd dmu = std::get<0>(gradients);
        SpMat dprecision = std::get<1>(gradients);

        int cnt = 0;
        int B = 1;
        double step_size = _step_size_base;

        // backtracking 
        while (true)
        {   
            // new step size
            // step_size = pow(_step_size_base, B);
            step_size = step_size * 0.75;

            auto onestep_res = onestep_linesearch(step_size, dmu, dprecision);

            double new_cost = std::get<0>(onestep_res);
            VectorXd new_mu = std::get<1>(onestep_res);
            auto new_precision = std::get<2>(onestep_res);

            // accept new cost and update mu and precision matrix
            if (new_cost < cost_iter){
                // update mean and covariance
                this->update_proposal(new_mu, new_precision); // Update the mu of GVIGH
                break;
            }else{ 
                // shrinking the step size
                B += 1;
                cnt += 1;
            }

            if (cnt > _niters_backtrack)
            {
                if (is_verbose){
                    std::cout << "Reached the maximum backtracking steps." << std::endl;
                }

                if (is_lowtemp){
                    this->switch_to_high_temperature();
                    is_lowtemp = false;
                }else{
                    converged = true;
                }

                // update_proposal(new_mu, new_precision);
                break;
            }                
        }
    }

    std::cout << "=========== Saving Data ===========" << std::endl;
    save_data(is_verbose);

}


template <typename Factor>
void GVIGH<Factor>::time_test()
{
    // std::cout << "========== Optimization Start: ==========" << std::endl << std::endl;

    Timer timer;
    std::vector<double> times;
    times.reserve(_niters);

    for (int i=0; i < _niters+1; i++){
        timer.start();
        double cost_iter = this->cost_value();
        VectorXd fact_costs_iter = this->factor_cost_vector();
        std::tuple<VectorXd, SpMat> gradients = compute_gradients_time();
        double time = timer.end_sec();
        if (i != 0)
            times.push_back(time * 1000);  
    }

    double average_time = std::accumulate(times.begin(), times.end(), 0.0) / _niters;

    double min_time = *std::min_element(times.begin(), times.end());
    double max_time = *std::max_element(times.begin(), times.end());

    std::cout << "% CPU average: " << average_time << " ms" << std::endl;
    std::cout << "% CPU min: " << min_time << " ms" << std::endl;
    std::cout << "% CPU max: " << max_time << " ms" << std::endl;

    std::cout << "% [ " << times[0];
    for (int i = 1; i < times.size(); ++i) {
        std::cout << ", " << times[i];
    }
    std::cout << " ]" << std::endl;
    
}

template <typename Factor>
inline void GVIGH<Factor>::set_precision(const SpMat &new_precision)
{
    _precision = new_precision;
    _covariance = inverse_GBP(_precision);

    #pragma omp parallel
    {
        #pragma omp for nowait // Nowait allows threads to continue without waiting at the end of the loop
        for (auto &factor : _vec_factors)
        {
            factor->update_precision_from_joint(_covariance);
        }
    }
}

/**
 * @brief Compute the costs of all factors for a given mean and cov.
 */
template <typename Factor>
VectorXd GVIGH<Factor>::factor_cost_vector(const VectorXd& fill_joint_mean, SpMat& joint_precision)
{
    VectorXd fac_costs(_nfactors);
    fac_costs.setZero();
    int cnt = 0;

    SpMat joint_cov = inverse_GBP(joint_precision);
    int thread_cnt = 0;

    #pragma omp for
    for (int i = 0; i < _vec_factors.size(); ++i)
    {
        auto &opt_k = _vec_factors[i];
        fac_costs(thread_cnt) = opt_k->fact_cost_value(fill_joint_mean, joint_cov); 
        thread_cnt += 1;
    }

    #pragma omp critical
    {
        cnt += thread_cnt; // Safely update the global counter
    }
    
    return fac_costs;
}

/**
 * @brief Compute the total cost function value given a state.
 */
template <typename Factor>
double GVIGH<Factor>::cost_value(const VectorXd &mean, SpMat &Precision)
{

    SpMat Cov = inverse_GBP(Precision);

    double value = 0.0;

    // sum up the factor costs
    // #pragma omp parallel for reduction(+:value)
    for (int i = 0; i < _vec_factors.size(); ++i)
    {
        // Access the current element - ensure this is safe in a parallel context
        auto &opt_k = _vec_factors[i];
        value += opt_k->fact_cost_value(mean, Cov); 
    }

    SparseLDLT ldlt(Precision);
    VectorXd vec_D = ldlt.vectorD();

    // std::cout << "entropy cost" << std::endl << vec_D.array().log().sum() / 2 << std::endl;
    return value + vec_D.array().log().sum() / 2;
}


/**
 * @brief Compute the covariances using Gaussian Belief Propagation.
 */
template <typename Factor>
SpMat GVIGH<Factor>::inverse_GBP(const SpMat &Precision)
{
    std::vector<Message> factors(2*_num_states-1);
    std::vector<Message> joint_factors(_num_states-1);
    Message variable_message;
    MatrixXd covariance(_dim, _dim);
    covariance.setZero();

    // Extract the factors from the precision matrix
    // The variable in factors are 0, {0,1}, 1, {1,2}, 2, ..., {_num_states-1,_num_states}, _num_states
    for (int i = 0; i < 2*_num_states-1; i++) {
        int var = i / 2;
        if (i % 2 == 0) {
            VectorXd variable(1);
            variable << var;
            MatrixXd lambda = Precision.block(_dim_state * var, _dim_state * var, _dim_state, _dim_state);
            factors[i] = {variable, lambda};
        }
        else {
            VectorXd variable(2);
            variable << var, var + 1;
            MatrixXd lambda = MatrixXd::Zero(2 * _dim_state, 2 * _dim_state);
            lambda.block(0, _dim_state, _dim_state, _dim_state) = Precision.block(_dim_state * var, _dim_state * (var + 1), _dim_state, _dim_state);
            lambda.block(_dim_state, 0, _dim_state, _dim_state) = Precision.block(_dim_state * (var + 1), _dim_state * var, _dim_state, _dim_state);
            factors[i] = {variable, lambda};
            joint_factors[var] = {variable, Precision.block(_dim_state * var, _dim_state * var, 2 * _dim_state, 2 * _dim_state)};
        }
    }
    
    // Initialize message
    std::vector<Message> forward_messages(_num_states);
    std::vector<Message> backward_messages(_num_states);
    forward_messages[0] = {VectorXd::Zero(1), MatrixXd::Zero(_dim_state, _dim_state)};
    backward_messages.back() = {VectorXd::Constant(1, _num_states-1), MatrixXd::Zero(_dim_state, _dim_state)};

    // Calculate messages between factors and variables
    for (int i = 0; i < _num_states - 1; i++) {
        variable_message = calculate_variable_message(forward_messages[i], factors[2 * i]);
        forward_messages[i + 1] = calculate_factor_message(variable_message, i + 1, factors[2 * i + 1]);
        int index = _num_states - 1 - i;
        variable_message = calculate_variable_message(backward_messages[index], factors[2 * index]);
        backward_messages[index - 1] = calculate_factor_message(variable_message, index - 1, factors[2 * index - 1]);
    }

    if (_num_states == 1){
        MatrixXd lambda = forward_messages[0].second + backward_messages[0].second + factors[0].second;
        MatrixXd variance = lambda.inverse();
        covariance.block(0, 0, _dim_state, _dim_state) = variance;
    }

    for (int i = 0; i < _num_states - 1; ++i) {
        MatrixXd lambda_joint = joint_factors[i].second;
        lambda_joint.block(0, 0, _dim_state, _dim_state) += forward_messages[i].second;
        lambda_joint.block(_dim_state, _dim_state, _dim_state, _dim_state) += backward_messages[i + 1].second;
        MatrixXd variance_joint = lambda_joint.inverse();
        covariance.block(i*_dim_state, i*_dim_state, 2*_dim_state, 2*_dim_state) = variance_joint;
    }

    return covariance.sparseView();
}

/**
 * @brief Compute the message of factors in GBP.
 */
template <typename Factor>
Message GVIGH<Factor>::calculate_factor_message(const Message &input_message, int target, const Message &factor_potential) {
    Message message;
    message.first = VectorXd::Constant(1, target);

    int index_variable = -1;
    int index_target = -1;

    // Identify indices of the variable and target in the factor potential
    for (int i = 0; i < factor_potential.first.size(); i++) {
        if (factor_potential.first(i) == input_message.first(0)) {
            index_variable = i;
        }
        if (factor_potential.first(i) == target) {
            index_target = i;
        }
    }

    MatrixXd lambda = factor_potential.second;
    lambda.block(_dim_state * index_variable, _dim_state * index_variable, _dim_state, _dim_state) += input_message.second;
    
    // Reorganize the lambda matrix to bring the target to the top
    if (index_target != 0) {
        lambda.block(0, 0, _dim_state, lambda.cols()).swap(lambda.block(index_target * _dim_state, 0, _dim_state, lambda.cols()));
        lambda.block(0, 0, lambda.rows(), _dim_state).swap(lambda.block(0, index_target * _dim_state, lambda.rows(), _dim_state));
    }

    MatrixXd lam_inverse = lambda.bottomRightCorner(lambda.rows() - _dim_state, lambda.cols() - _dim_state).inverse();
    MatrixXd lam_message = lambda.topLeftCorner(_dim_state, _dim_state) - lambda.topRightCorner(_dim_state, lambda.cols() - _dim_state) * lam_inverse * lambda.bottomLeftCorner(lambda.rows() - _dim_state, _dim_state);

    message.second = lam_message;
    return message;
}

}


#endif // GVI_GH_IMPL_H
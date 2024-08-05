
#pragma once

#ifndef GVI_GH_IMPL_H
#define GVI_GH_IMPL_H

using namespace Eigen;

#include <stdexcept>
#include <optional>
#include <omp.h>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace gvi{

template <typename Factor>
void GVIGH<Factor>::switch_to_high_temperature(){
    std::cout << "Switching to high temperature.." << std::endl;
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
            this->switch_to_high_temperature();
            is_lowtemp = false;
        }

        // ============= Cost at current iteration =============
        double cost_iter = this->cost_value();

        if (is_verbose){
            std::cout << "========= iteration " << i_iter << " ========= " << std::endl;
            std::cout << "--- cost_iter ---" << std::endl << cost_iter << std::endl;
        }

        // ============= Collect factor costs =============
        VectorXd fact_costs_iter = this->factor_cost_vector();
        // std::cout << "Factor Costs:" << fact_costs_iter.transpose() << std::endl;

        _res_recorder.update_data(_mu, _covariance, _precision, cost_iter, fact_costs_iter);

        // gradients
        std::tuple<VectorXd, SpMat> gradients = compute_gradients();

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
                this->update_proposal(new_mu, new_precision);
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
inline void GVIGH<Factor>::set_precision(const SpMat &new_precision)
{
    _precision = new_precision;
    // sparse inverse
    inverse_inplace();

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

    // Use a private counter for each thread to avoid race conditions
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
    #pragma omp parallel for reduction(+:value)
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
    /*dim->_dim_state, num_states->_num_states, Lambda->Precision*/
    std::vector<Message> factor(2*_num_states-1);
    std::vector<Message> joint_factor(_num_states-1);
    Message variable_message;

    // Extract the factors from the precision matrix
    for (int i = 0; i < 2*_num_states-1; ++i) {
        int var = i / 2;
        if (i % 2 == 0) {
            VectorXd variable(1);
            variable << var;
            MatrixXd lam = Precision.block(_dim_state * var, _dim_state * var, _dim_state, _dim_state);
            factor[i] = {variable, lam};
        } else {
            VectorXd variable(2);
            variable << var, var + 1;
            MatrixXd lam = MatrixXd::Zero(2 * _dim_state, 2 * _dim_state);
            lam.block(0, _dim_state, _dim_state, _dim_state) = Precision.block(_dim_state * var, _dim_state * (var + 1), _dim_state, _dim_state);
            lam.block(_dim_state, 0, _dim_state, _dim_state) = Precision.block(_dim_state * (var + 1), _dim_state * var, _dim_state, _dim_state);
            joint_factor[var] = {variable, Precision.block(_dim_state * var, _dim_state * var, 2 * _dim_state, 2 * _dim_state)};
            factor[i] = {variable, lam};
        }
    }

    std::vector<Message> factor_message(_num_states);
    std::vector<Message> factor_message1(_num_states);
    factor_message[0] = {VectorXd::Zero(1), MatrixXd::Zero(_dim_state, _dim_state)};
    factor_message1.back() = {VectorXd::Zero(1), MatrixXd::Zero(_dim_state, _dim_state)};
    factor_message1.back().first(0) = _num_states-1; 

    // Calculate the message from the factors to the variables
    for (int i = 0; i < _num_states - 1; i++) {
        variable_message = calculate_variable_message(factor_message[i], factor[2 * i]);
        factor_message[i + 1] = calculate_factor_message(variable_message, i + 1, factor[2 * i + 1], _dim_state);
        int index = _num_states - 1 - i;
        variable_message = calculate_variable_message(factor_message1[index], factor[2 * index]);
        factor_message1[index - 1] = calculate_factor_message(variable_message, index - 1, factor[2 * index - 1], _dim_state);
    }

    MatrixXd sigma(_dim, _dim);

    if (_num_states == 1){
        MatrixXd lam = factor_message[0].second + factor_message1[0].second + factor[0].second;
        MatrixXd variance = lam.inverse();
        sigma.block(0, 0, _dim_state, _dim_state) = variance;
    }

    for (int i = 0; i < _num_states - 1; ++i) {
        MatrixXd lam_joint = joint_factor[i].second;
        lam_joint.block(0, 0, _dim_state, _dim_state) += factor_message[i].second;
        lam_joint.block(_dim_state, _dim_state, _dim_state, _dim_state) += factor_message1[i + 1].second;
        MatrixXd variance_joint = lam_joint.inverse();
        sigma.block(i*_dim_state, i*_dim_state, 2*_dim_state, 2*_dim_state) = variance_joint;
    }


    return sigma.sparseView();

}

/**
 * @brief Compute the message of factors in GBP.
 */
template <typename Factor>
Message GVIGH<Factor>::calculate_factor_message(const Message &input_message, int target, const Message &factor_potential, int dim) {
    Message message;
    VectorXd target_vector(1);
    target_vector(0) = target;
    message.first = target_vector;

    int index_variable = -1;
    int index_target = -1;

    for (int i = 0; i < factor_potential.first.size(); i++) {
        if (factor_potential.first(i) == input_message.first(0)) {
            index_variable = i;
        }
        if (factor_potential.first(i) == target) {
            index_target = i;
        }
    }

    MatrixXd lam = factor_potential.second;
    
    lam.block(dim * index_variable, dim * index_variable, dim, dim) += input_message.second;
    
    if (index_target != 0) {
        lam.block(0, 0, dim, lam.cols()).swap(lam.block(index_target * dim, 0, dim, lam.cols()));
        lam.block(0, 0, lam.rows(), dim).swap(lam.block(0, index_target * dim, lam.rows(), dim));
    }

    MatrixXd lam_inverse = lam.block(dim, dim, lam.rows() - dim, lam.cols() - dim).inverse();
    MatrixXd lam_message = lam.block(0, 0, dim, dim) - lam.block(0, dim, dim, lam.cols() - dim) * lam_inverse * lam.block(dim, 0, lam.rows() - dim, dim);

    message.second = lam_message;
    return message;
}


}


#endif // GVI_GH_IMPL_H
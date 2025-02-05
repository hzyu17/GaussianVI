
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
    std::cout << "Switching to high temperature.." << std::endl;
    #pragma omp parallel for
    for (auto& i_factor : _vec_factors) {
        i_factor->factor_switch_to_high_temperature();
    }
    this->_temperature = this->_high_temperature;
    // this->initilize_precision_matrix();
}

template <typename Factor>
void GVIGH<Factor>::classify_factors(){
    for (auto& i_factor : _vec_factors) {
        if (i_factor->linear_factor())
            _vec_linear_factors.push_back(i_factor);
        else if (!i_factor->linear_factor())
            _vec_nonlinear_factors.push_back(i_factor);
    }
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

    _vec_nonlinear_factors[0]->cuda_init();

    // VectorXd mu_EMA(_mu.size());
    // SpMat precision_EMA(_precision.rows(), _precision.cols());
    // mu_EMA.setZero();
    // precision_EMA.setZero();
    // mu_EMA = _mu;
    // precision_EMA = _precision;

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

        if (is_verbose){
            std::cout << "========= iteration " << i_iter << " ========= " << std::endl;
        }

        auto result_cuda = factor_cost_vector_cuda();
        double cost_iter = std::get<0>(result_cuda);
        VectorXd fact_costs_iter = std::get<1>(result_cuda);
        VectorXd dmu = std::get<2>(result_cuda);
        SpMat dprecision = std::get<3>(result_cuda);

        if (is_verbose){
            std::cout << "--- cost_iter ---" << std::endl << cost_iter << std::endl;
            // std::cout << "Factor Costs:" << fact_costs_iter.transpose() << std::endl;
            // std::cout << "--- dmu ---" << std::endl << dmu.norm() << std::endl;
            // std::cout << "--- dprecision ---" << std::endl << dprecision.norm() << std::endl;
        }

        _res_recorder.update_data(_mu, _covariance, _precision, cost_iter, fact_costs_iter);
        
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
                // mu_EMA = _alpha * new_mu + (1-_alpha) * mu_EMA;
                // precision_EMA = _alpha * new_precision + (1-_alpha) * precision_EMA;
                this->update_proposal(_alpha * new_mu + (1-_alpha) * this->_mu, _alpha * new_precision + (1-_alpha) * this->_precision); // Update using EMA
                // std::cout << "back tracking time: "<< cnt << std::endl;
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

    _vec_nonlinear_factors[0]->cuda_free();

    std::cout << "=========== Saving Data ===========" << std::endl;
    save_data(is_verbose);

}

template <typename Factor>
std::tuple<double, VectorXd, VectorXd, SpMat> GVIGH<Factor>::factor_cost_vector_cuda(const VectorXd& fill_joint_mean, SpMat& joint_precision)
{
    int n_nonlinear = _vec_nonlinear_factors.size();

    VectorXd fac_costs(_nfactors);
    VectorXd nonlinear_fac_cost(n_nonlinear);
    fac_costs.setZero();
    nonlinear_fac_cost.setZero();

    SpMat joint_cov = inverse_GBP(joint_precision);
    SpMat joint_cov_splash = chain_splash_GBP(joint_precision);

    double norm_joint_cov = joint_cov.toDense().norm();
    double norm_error = (joint_cov_splash - joint_cov).toDense().norm();

    std::cout << "Norm of joint covariance: " << norm_joint_cov << std::endl;
    std::cout << "Norm of error: " << norm_error << std::endl;
    std::cout << "Error: " << norm_error / norm_joint_cov * 100 << "%" << std::endl;

    std::vector<MatrixXd> sigmapts_vec(n_nonlinear);
    std::vector<VectorXd> mean_vec(n_nonlinear);

    omp_set_num_threads(20); 

    #pragma omp parallel for
    for (int i = 0; i < n_nonlinear; i++)
    {
        auto &opt_k = _vec_nonlinear_factors[i];
        opt_k->cuda_matrices(sigmapts_vec, mean_vec); 
    }

    int sigma_rows = sigmapts_vec[0].rows();
    int sigma_cols = sigmapts_vec[0].cols();
    int mean_size = mean_vec[0].size();

    MatrixXd sigmapts_mat(sigma_rows, sigmapts_vec.size()*sigma_cols);
    MatrixXd mean_mat(mean_size, mean_vec.size());

    VectorXd E_phi_mat(n_nonlinear);
    VectorXd dmu_mat(sigma_cols * n_nonlinear);
    MatrixXd ddmu_mat(sigma_cols, sigma_cols * n_nonlinear);

    #pragma omp parallel for
    for (int i = 0; i < n_nonlinear; i++)
    {
        sigmapts_mat.block(0, i * sigma_cols, sigma_rows, sigma_cols) = sigmapts_vec[i];
        mean_mat.col(i) = mean_vec[i];
    }

    // Compute the cost and derivatives of the nonlinear factors
    _vec_nonlinear_factors[0]->dmuIntegration(sigmapts_mat, mean_mat, nonlinear_fac_cost, dmu_mat, ddmu_mat, sigma_cols);
    E_phi_mat = nonlinear_fac_cost;

    nonlinear_fac_cost = nonlinear_fac_cost / this ->_temperature; 


    MatrixXd covariance_matrix(sigma_cols, n_nonlinear*sigma_cols);
    MatrixXd sigma(sigma_rows, sigma_cols);

    #pragma omp parallel for
    for (int i = 0; i < n_nonlinear; i++)
    {
        auto &opt_k = _vec_nonlinear_factors[i];
        covariance_matrix.block(0, i * sigma_cols, sigma_cols, sigma_cols) = joint_cov.block((i+1)*_dim_state, (i+1)*_dim_state, sigma_cols, sigma_cols);
    }
    // _vec_nonlinear_factors[0]->compute_sigmapts(fill_joint_mean, covariance_matrix, sigma_cols, n_nonlinear, sigma);
    // std::cout << "error of sigmapts: " << std::endl << (sigma-sigmapts_mat).norm() << std::endl;


    int cnt = 0;

    // Use a private counter for each thread to avoid race conditions
    int thread_cnt = 0;

    // #pragma omp for
    for (int i = 0; i < _vec_factors.size(); ++i)
    {
        auto &opt_k = _vec_factors[i];
        if (opt_k->linear_factor())
            fac_costs(thread_cnt) = opt_k->fact_cost_value(_mu, _covariance); 
        else
            fac_costs(thread_cnt) = nonlinear_fac_cost(opt_k->_start_index - 1);
        thread_cnt += 1;
    }

    #pragma omp critical
    {
        cnt += thread_cnt; // Safely update the global counter
    }

    double value = fac_costs.sum();
    SparseLDLT ldlt(joint_precision);
    VectorXd vec_D = ldlt.vectorD();

    double cost = value + vec_D.array().log().sum() / 2;

    // double entropy = vec_D.array().log().sum() / 2;
    // double collision_cost = nonlinear_fac_cost.cwiseAbs().sum();
    // double prior_cost = fac_costs.sum() - collision_cost;

    // std::cout << "Entropy: " << entropy << std::endl;
    // std::cout << "Collision Cost: " << collision_cost << std::endl;
    // std::cout << "Prior Cost: " << prior_cost << std::endl;

    _Vdmu.setZero();
    _Vddmu.setZero();

    VectorXd Vdmu_sum(_dim);
    SpMat Vddmu_sum(_dim, _dim);
    Vdmu_sum.setZero();
    Vddmu_sum.setZero();

    #pragma omp parallel 
    {
        // Thread-local storage to avoid race conditions
        VectorXd Vdmu_private(Vdmu_sum.size());
        SpMat Vddmu_private(Vddmu_sum.rows(), Vddmu_sum.cols());
        Vdmu_private.setZero();
        Vddmu_private.setZero();

        #pragma omp for nowait // Nowait allows threads to continue without waiting at the end of the loop
        for (auto &opt_k : _vec_factors) {
            if (opt_k->linear_factor()){
                opt_k->calculate_partial_V();
            }
            else{
                int index = opt_k->index()-1;
                MatrixXd ddmu_i = ddmu_mat.block(0, index*sigma_cols, sigma_cols, sigma_cols);;
                VectorXd dmu_i = dmu_mat.segment(index*sigma_cols, sigma_cols);

                opt_k->calculate_partial_V(ddmu_i, dmu_i, E_phi_mat(index));
            }

            Vdmu_private += opt_k->local2joint_dmu_insertion();
            Vddmu_private += opt_k->local2joint_dprecision_insertion();
        }

        #pragma omp critical
        {
            Vdmu_sum += Vdmu_private;
            Vddmu_sum += Vddmu_private;
        }
    }

    _Vdmu = Vdmu_sum;
    _Vddmu = Vddmu_sum;

    SpMat dprecision(_dim, _dim);
    dprecision.setZero();
    dprecision = _Vddmu - _precision;

    Eigen::ConjugateGradient<SpMat, Eigen::Upper> solver;
    // Eigen::ConjugateGradient<SpMat, Eigen::Upper, Eigen::IncompleteLUT<double>> solver;
    VectorXd dmu(_dim);
    dmu.setZero();
    dmu = solver.compute(_Vddmu).solve(-_Vdmu);

    return std::make_tuple(cost, fac_costs, dmu, dprecision);
}

template <typename Factor>
std::tuple<double, VectorXd, VectorXd, SpMat> GVIGH<Factor>::factor_cost_vector_cuda_time(const VectorXd& fill_joint_mean, SpMat& joint_precision)
{
    static int flag = 0;
    Timer timer;

    int n_nonlinear = _vec_nonlinear_factors.size();

    VectorXd fac_costs(_nfactors);
    VectorXd nonlinear_fac_cost(n_nonlinear);
    fac_costs.setZero();
    nonlinear_fac_cost.setZero();

    if (!flag)
        timer.start();

    SpMat joint_cov = inverse_GBP(joint_precision);

    if (!flag)
        std::cout << "GBP Inverse time: " << timer.end_mis() << " ms" << std::endl;


    // if (!flag)
    //     timer.start();

    // SpMat cov = inverse(joint_precision);

    // if (!flag)
    //     std::cout << "Inverse time: " << timer.end_mis() << " ms" << std::endl;


    // if (!flag)
    //     timer.start();
    
    std::vector<MatrixXd> sigmapts_vec(n_nonlinear);
    std::vector<VectorXd> mean_vec(n_nonlinear);

    omp_set_num_threads(20); 

    #pragma omp parallel for
    for (int i = 0; i < n_nonlinear; i++)
    {
        auto &opt_k = _vec_nonlinear_factors[i];
        opt_k->cuda_matrices(sigmapts_vec, mean_vec); 
    }

    int sigma_rows = sigmapts_vec[0].rows();
    int sigma_cols = sigmapts_vec[0].cols();
    int mean_size = mean_vec[0].size();

    MatrixXd sigmapts_mat(sigma_rows, sigmapts_vec.size()*sigma_cols);
    MatrixXd mean_mat(mean_size, mean_vec.size());

    VectorXd E_phi_mat(n_nonlinear);
    VectorXd dmu_mat(sigma_cols * n_nonlinear);
    MatrixXd ddmu_mat(sigma_cols, sigma_cols * n_nonlinear);

    #pragma omp parallel for
    for (int i = 0; i < n_nonlinear; i++)
    {
        sigmapts_mat.block(0, i * sigma_cols, sigma_rows, sigma_cols) = sigmapts_vec[i];
        mean_mat.col(i) = mean_vec[i];
    }

    // if (!flag)
        // std::cout << "Matrix Filling time: " << timer.end_mis() << " ms" << std::endl;

    if (!flag)
        timer.start();

    // Compute the cost of the nonlinear factors
    _vec_nonlinear_factors[0]->dmuIntegration(sigmapts_mat, mean_mat, nonlinear_fac_cost, dmu_mat, ddmu_mat, sigma_cols);
    E_phi_mat = nonlinear_fac_cost;
    nonlinear_fac_cost = nonlinear_fac_cost / this ->_temperature;     

    if (!flag)
        std::cout << "Cuda Computation time: " << timer.end_mis() << " ms" << std::endl;

    int cnt = 0;
    // Use a private counter for each thread to avoid race conditions
    int thread_cnt = 0;

    #pragma omp for
    for (int i = 0; i < _vec_factors.size(); ++i)
    {
        auto &opt_k = _vec_factors[i];
        if (opt_k->linear_factor())
            fac_costs(thread_cnt) = opt_k->fact_cost_value(_mu, _covariance); 
        else
            fac_costs(thread_cnt) = nonlinear_fac_cost(opt_k->_start_index - 1);
        thread_cnt += 1;
    }

    #pragma omp critical
    {
        cnt += thread_cnt; // Safely update the global counter
    }

    double value = fac_costs.sum();
    SparseLDLT ldlt(joint_precision);
    VectorXd vec_D = ldlt.vectorD();

    double cost = value + vec_D.array().log().sum() / 2;

    if (!flag)
        timer.start();

    _Vdmu.setZero();
    _Vddmu.setZero();

    VectorXd Vdmu_sum(_dim);
    SpMat Vddmu_sum(_dim, _dim);
    Vdmu_sum.setZero();
    Vddmu_sum.setZero();

    #pragma omp parallel 
    {
        // Thread-local storage to avoid race conditions
        VectorXd Vdmu_private(Vdmu_sum.size());
        SpMat Vddmu_private(Vddmu_sum.rows(), Vddmu_sum.cols());
        Vdmu_private.setZero();
        Vddmu_private.setZero();

        #pragma omp for nowait // Nowait allows threads to continue without waiting at the end of the loop
        for (auto &opt_k : _vec_factors) {
            if (opt_k->linear_factor()){
                opt_k->calculate_partial_V();
            }
            else{
                int index = opt_k->index()-1;
                MatrixXd ddmu_i = ddmu_mat.block(0, index*sigma_cols, sigma_cols, sigma_cols);;
                VectorXd dmu_i = dmu_mat.segment(index*sigma_cols, sigma_cols);

                opt_k->calculate_partial_V(ddmu_i, dmu_i, E_phi_mat(index));
            }

            Vdmu_private += opt_k->local2joint_dmu_insertion();
            Vddmu_private += opt_k->local2joint_dprecision_insertion();
        }

        #pragma omp critical
        {
            Vdmu_sum += Vdmu_private;
            Vddmu_sum += Vddmu_private;
        }
    }

    _Vdmu = Vdmu_sum;
    _Vddmu = Vddmu_sum;

    if (!flag)
        std::cout << "Derivative Mapping time: " << timer.end_mis() << " ms" << std::endl;

    if (!flag)
        timer.start();

    SpMat dprecision(_dim, _dim);
    dprecision.setZero();
    dprecision = _Vddmu - _precision;

    Eigen::ConjugateGradient<SpMat, Eigen::Upper> solver;
    // Eigen::ConjugateGradient<SpMat, Eigen::Upper, Eigen::IncompleteLUT<double>> solver;
    VectorXd dmu(_dim);
    dmu.setZero();
    dmu = solver.compute(_Vddmu).solve(-_Vdmu);

    if (!flag)
        std::cout << "Solver time: " << timer.end_mis() << " ms" << std::endl << std::endl;
    
    flag = 1;

    return std::make_tuple(cost, fac_costs, dmu, dprecision);
}


template <typename Factor>
void GVIGH<Factor>::time_test()
{
    // std::cout << "========== Optimization Start: ==========" << std::endl << std::endl;
    _vec_nonlinear_factors[0]->cuda_init();

    Timer timer;

    std::vector<double> times_GBP, times_inverse;
    times_GBP.reserve(_niters);
    times_inverse.reserve(_niters);

    std::cout << "Optimization Start" << std::endl;

    for (int i = 0; i < _niters + 1; i++){
        timer.start();
        auto result_cuda = factor_cost_vector_cuda_time();
        double time = timer.end_mis();
        if (i!=0)
            times_GBP.push_back(time);  
    }

    double average_time = std::accumulate(times_GBP.begin(), times_GBP.end(), 0.0) / _niters;

    double min_time = *std::min_element(times_GBP.begin(), times_GBP.end());
    double max_time = *std::max_element(times_GBP.begin(), times_GBP.end());

    std::cout << "% " << _vec_nonlinear_factors.size() + 2 << std::endl;
    std::cout << "% GPU average: " << average_time << " ms" << std::endl;
    std::cout << "% GPU min: " << min_time << " ms" << std::endl;
    std::cout << "% GPU max: " << max_time << " ms" << std::endl;

    // std::cout << "% [ " << times_GBP[0];
    // for (int i = 1; i < times_GBP.size(); ++i) {
    //     std::cout << ", " << times_GBP[i];
    // }
    // std::cout << " ]" << std::endl;


    // for (int i = 0; i < _niters + 1; i++){
    //     timer.start();
    //     SpMat joint_cov = inverse(this -> _precision);
    //     double time = timer.end_sec();
    //     if (i!=0)
    //         times_inverse.push_back(time * 1000);  
    // }

    // average_time = std::accumulate(times_inverse.begin(), times_inverse.end(), 0.0) / _niters;

    // min_time = *std::min_element(times_inverse.begin(), times_inverse.end());
    // max_time = *std::max_element(times_inverse.begin(), times_inverse.end());

    // std::cout << "% Inverse average: " << average_time << " ms" << std::endl;
    // std::cout << "% Inverse min: " << min_time << " ms" << std::endl;
    // std::cout << "% Inverse max: " << max_time << " ms" << std::endl;

    // std::cout << "% [ " << times_inverse[0];
    // for (int i = 1; i < times_inverse.size(); ++i) {
    //     std::cout << ", " << times_inverse[i];
    // }
    // std::cout << " ]" << std::endl;

    _vec_nonlinear_factors[0]->cuda_free();

}

template <typename Factor>
inline void GVIGH<Factor>::set_precision(const SpMat &new_precision)
{
    _precision = new_precision;
    // // sparse inverse
    // inverse_inplace();
    _covariance = inverse_GBP(_precision);
    // _covariance = chain_splash_GBP(_precision);

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

template <typename Factor>
double GVIGH<Factor>::cost_value_cuda(const VectorXd& fill_joint_mean, SpMat& joint_precision)
{
    int n_nonlinear = _vec_nonlinear_factors.size();
    VectorXd nonlinear_fac_cost(n_nonlinear);
    nonlinear_fac_cost.setZero();

    SpMat joint_cov = chain_splash_GBP(joint_precision);

    std::vector<MatrixXd> sigmapts_vec(n_nonlinear);
    std::vector<VectorXd> mean_vec(n_nonlinear);

    #pragma omp parallel for
    for (int i = 0; i < n_nonlinear; i++)
    {
        auto &opt_k = _vec_nonlinear_factors[i];
        opt_k->cuda_matrices(fill_joint_mean, joint_cov, sigmapts_vec, mean_vec); 
    }

    int sigma_rows = sigmapts_vec[0].rows();
    int sigma_cols = sigmapts_vec[0].cols();
    int mean_size = mean_vec[0].size();

    MatrixXd sigmapts_mat(sigma_rows, sigmapts_vec.size()*sigma_cols);

    for (int i = 0; i < n_nonlinear; i++)
        sigmapts_mat.block(0, i * sigma_cols, sigma_rows, sigma_cols) = sigmapts_vec[i];

    // Compute the cost of the nonlinear factors
    _vec_nonlinear_factors[0]->newCostIntegration(sigmapts_mat, nonlinear_fac_cost, sigma_cols);

    nonlinear_fac_cost = nonlinear_fac_cost / this ->_temperature;

    double value = 0.0;

    #pragma omp parallel for reduction(+:value)
    for (int i = 0; i < _vec_linear_factors.size(); ++i)
    {
        auto &opt_k = _vec_linear_factors[i];
        value += opt_k->fact_cost_value(fill_joint_mean, joint_cov); 
    }

    value += nonlinear_fac_cost.sum();

    SparseLDLT ldlt(joint_precision);
    VectorXd vec_D = ldlt.vectorD();

    return value + vec_D.array().log().sum() / 2;
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

    return value + vec_D.array().log().sum() / 2;
}

/**
 * @brief Compute the covariances using Residual Splash based Gaussian Belief Propagation.
 *
 * This implementation initializes all messages to zero (for the second component only).
 * Each message's residual is maintained as the maximum of the forward and backward residuals.
 * In each iteration, the algorithm selects the node with the largest residual and updates a splash
 * region centered at that node. The algorithm terminates when all residuals fall below the tolerance
 * or the maximum number of iterations is reached.
 */
template <typename Factor>
SpMat GVIGH<Factor>::residual_splash_GBP(const SpMat &Precision)
{
    std::vector<Message> factors(2 * _num_states - 1);
    std::vector<Message> joint_factors(_num_states - 1);
    Message variable_message;
    MatrixXd covariance(_dim, _dim);
    covariance.setZero();

    // Construct factors: variable factors and edge factors; and joint factors for marginal computation
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
    
    // Initialize messages and residuals
    std::vector<Message> forward_messages(_num_states);
    std::vector<Message> backward_messages(_num_states);
    for (int i = 0; i < _num_states; i++) {
        forward_messages[i].first = VectorXd::Constant(1, i);
        forward_messages[i].second = MatrixXd::Zero(_dim_state, _dim_state);
        backward_messages[i].first = VectorXd::Constant(1, i);
        backward_messages[i].second = MatrixXd::Zero(_dim_state, _dim_state);
    }
    
    // Initialize residuals to a large value to ensure initial updates.
    const double INF = std::numeric_limits<double>::max();
    std::vector<double> res(_num_states, INF);
    
    // Residual Splash Iterative Updates
    // Parameters: tolerance, maximum iterations, splash region radius
    const double tol = 1e-5;
    const int max_iter = 100;
    const int splash_radius = 7;
    int iter = 0;
    
    while (iter < max_iter)
    {
        // Find the node with the maximum residual over all nodes
        double max_residual = 0.0;
        int max_index = 0;
        for (int i = 0; i < _num_states; i++) {
            if (res[i] > max_residual) {
                max_residual = res[i];
                max_index = i;
            }
        }
        std::cout << "Iteration " << iter << ", max residual: " << max_residual << ", index = " << max_index << std::endl;
        
        // If the maximum residual is below tolerance, the algorithm has converged.
        if (max_residual < tol){
            std::cout << "Converged after " << iter << " iterations." << std::endl;
            break;
        }
        
        // Determine the splash region based on the max_index and splash_radius
        int region_start = std::max(0, max_index - splash_radius);
        int region_end   = std::min(_num_states - 1, max_index + splash_radius);
        
        // Save current messages in the splash region to compute residuals after updates
        std::vector<MatrixXd> old_forward(region_end - region_start + 1);
        std::vector<MatrixXd> old_backward(region_end - region_start + 1);
        for (int i = region_start; i <= region_end; i++) {
            old_forward[i - region_start] = forward_messages[i].second;
            old_backward[i - region_start] = backward_messages[i].second;
        }

        // std::cout << "region center: " << max_index << ", region start: " << region_start << ", region end: " << region_end << std::endl;
        
        // Update messages in the splash region:
        // Forward update: update forward messages from region_start to region_end
        for (int i = region_start; i < region_end; i++) {
            // Compute message from variable i to factor and then from factor to variable i+1
            Message var_msg = calculate_variable_message(forward_messages[i], factors[2 * i]);
            Message fac_msg = calculate_factor_message(var_msg, i + 1, factors[2 * i + 1]);
            forward_messages[i + 1] = fac_msg;
        }

        // Backward update: update backward messages from region_end to region_start
        for (int i = region_end; i > region_start; i--) {
            Message var_msg = calculate_variable_message(backward_messages[i], factors[2 * i]);
            Message fac_msg = calculate_factor_message(var_msg, i - 1, factors[2 * i - 1]);
            backward_messages[i - 1] = fac_msg;
        }
        
        // Recompute the residual for each node in the splash region as the maximum of the forward and backward residuals
        for (int i = region_start; i <= region_end; i++) {
            double diff_forward = (forward_messages[i].second - old_forward[i - region_start]).norm();
            double diff_backward = (backward_messages[i].second - old_backward[i - region_start]).norm();
            res[i] = std::max(diff_forward, diff_backward);
            // std::cout << "residual[" << i << "]: " << res[i] << "  ";
        }
        // std::cout << std::endl;
        
        iter++;
    }
    
    // Compute marginal covariance using the final messages
    if (_num_states == 1) {
        MatrixXd lambda = forward_messages[0].second + backward_messages[0].second + factors[0].second;
        MatrixXd variance = lambda.inverse();
        covariance.block(0, 0, _dim_state, _dim_state) = variance;
    }
    else {
        for (int i = 0; i < _num_states - 1; ++i) {
            MatrixXd lambda_joint = joint_factors[i].second;
            lambda_joint.block(0, 0, _dim_state, _dim_state) += forward_messages[i].second;
            lambda_joint.block(_dim_state, _dim_state, _dim_state, _dim_state) += backward_messages[i + 1].second;
            MatrixXd variance_joint = lambda_joint.inverse();
            covariance.block(i * _dim_state, i * _dim_state, 2 * _dim_state, 2 * _dim_state) = variance_joint;
        }
    }
    
    return covariance.sparseView();
}


// ChainSplash based Gaussian Belief Propagation with OpenMP parallelization.
// The number of threads is chosen as approximately sqrt(_num_states).
template <typename Factor>
SpMat GVIGH<Factor>::chain_splash_GBP(const SpMat &Precision)
{
    std::vector<Message> factors(2*_num_states-1);
    std::vector<Message> joint_factors(_num_states-1);
    Message variable_message;
    MatrixXd covariance(_dim, _dim);
    covariance.setZero();

    // Construct factors: variable factors and edge factors; and joint factors for marginal computation
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
    
    // Initialize messages and residuals
    std::vector<Message> forward_messages(_num_states);
    std::vector<Message> backward_messages(_num_states);
    for (int i = 0; i < _num_states; i++) {
        forward_messages[i].first = VectorXd::Constant(1, i);
        forward_messages[i].second = MatrixXd::Zero(_dim_state, _dim_state);
        backward_messages[i].first = VectorXd::Constant(1, i);
        backward_messages[i].second = MatrixXd::Zero(_dim_state, _dim_state);
    }
    
    // ChainSplash Iterative Updates with Block Boundary Exchange and Convergence Check
    const int max_iter = 100;  // maximum number of iterations
    const double tol = 1e-5;   // convergence tolerance
    // Set the number of threads approximately to sqrt(n)
    int num_threads = static_cast<int>(std::sqrt(static_cast<double>(_num_states)));
    // Partition the chain into blocks: block_size = ceil(n / num_threads)
    int block_size = (_num_states + num_threads - 1) / num_threads;
    
    // std::cout << "ChainSplash: Using " << num_threads << " threads, block size = " << block_size << std::endl;
    
    // Containers to store previous iteration messages for residual computation.
    std::vector<MatrixXd> old_forward(_num_states);
    std::vector<MatrixXd> old_backward(_num_states);
    
    double global_residual = std::numeric_limits<double>::max();
    for (int iter = 0; iter < max_iter; iter++) {
        // Save current messages for residual computation.
        for (int i = 0; i < _num_states; i++) {
            old_forward[i] = forward_messages[i].second;
            old_backward[i] = backward_messages[i].second;
        }

        #pragma omp parallel num_threads(num_threads) shared(forward_messages, backward_messages, factors)
        {
            // Each thread compute its own block（Assume thread b compute block [b * block_size, min((b+1)*block_size-1, _num_states-1)]）
            int b = omp_get_thread_num();
            int block_start = b * block_size;
            int block_end = std::min(_num_states - 1, block_start + block_size - 1);

            // --- Block internal forward update ---
            for (int i = block_start; i < block_end; i++) {
                Message var_msg = calculate_variable_message(forward_messages[i], factors[2 * i]);
                Message fac_msg = calculate_factor_message(var_msg, i + 1, factors[2 * i + 1]);
                forward_messages[i + 1] = fac_msg;
            }
            // --- Block internal backward update ---
            for (int i = block_end; i > block_start; i--) {
                Message var_msg = calculate_variable_message(backward_messages[i], factors[2 * i]);
                Message fac_msg = calculate_factor_message(var_msg, i - 1, factors[2 * i - 1]);
                backward_messages[i - 1] = fac_msg;
            }

            // --- Barrier: ensure all threads have finished block internal updates ---
            #pragma omp barrier

            if (b < num_threads - 1) {
                int block_end_current = std::min(_num_states - 1, (b + 1) * block_size - 1);
                int block_start_next = block_end_current + 1;
                
                // Forward message exchange: update the first message of the next block.
                Message var_msg_fwd = calculate_variable_message(forward_messages[block_end_current], factors[2 * block_end_current]);
                Message fac_msg_fwd = calculate_factor_message(var_msg_fwd, block_start_next, factors[2 * block_end_current + 1]);
                forward_messages[block_start_next] = fac_msg_fwd;
                
                // Backward message exchange: update the last message of the current block.
                Message var_msg_bwd = calculate_variable_message(backward_messages[block_start_next], factors[2 * block_start_next]);
                Message fac_msg_bwd = calculate_factor_message(var_msg_bwd, block_end_current, factors[2 * block_start_next - 1]);
                backward_messages[block_end_current] = fac_msg_bwd;
            }
            // Barrier: wait for all boundary exchanges to finish before leaving the parallel region.
            #pragma omp barrier
        } // end of parallel region

        
        // -------------------------------
        // Compute global residual (max difference over all messages)
        // -------------------------------
        global_residual = 0.0;
        for (int i = 0; i < _num_states; i++) {
            double diff_forward = (forward_messages[i].second - old_forward[i]).norm();
            double diff_backward = (backward_messages[i].second - old_backward[i]).norm();
            double local_res = std::max(diff_forward, diff_backward);
            if (local_res > global_residual)
                global_residual = local_res;
        }
        
        // std::cout << "Iteration " << iter << ", global residual = " << global_residual << std::endl;
        if (global_residual < tol) {
            // std::cout << "Converged after " << iter+1 << " iterations." << std::endl;
            break;
        }
    } 
    
    // Compute marginal covariance using final messages
    if (_num_states == 1) {
        MatrixXd lambda = forward_messages[0].second + backward_messages[0].second + factors[0].second;
        MatrixXd variance = lambda.inverse();
        covariance.block(0, 0, _dim_state, _dim_state) = variance;
    }
    else {
        for (int i = 0; i < _num_states - 1; ++i) {
            MatrixXd lambda_joint = joint_factors[i].second;
            lambda_joint.block(0, 0, _dim_state, _dim_state) += forward_messages[i].second;
            lambda_joint.block(_dim_state, _dim_state, _dim_state, _dim_state) += backward_messages[i + 1].second;
            MatrixXd variance_joint = lambda_joint.inverse();
            covariance.block(i * _dim_state, i * _dim_state, 2 * _dim_state, 2 * _dim_state) = variance_joint;
        }
    }
    
    return covariance.sparseView();
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
    for (int i = 0; i < _num_states; i++) {
        forward_messages[i].second.setZero();
        backward_messages[i].second.setZero();
    }
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

// MatrixXd cov_error(cov_GBP.rows(), cov_GBP.cols());
// cov_error.setZero();

// for (int i = 0; i < _num_states - 1; ++i) {
//     cov_error.block(i*_dim_state, i*_dim_state, 2*_dim_state, 2*_dim_state) = _covariance.block(i*_dim_state, i*_dim_state, 2*_dim_state, 2*_dim_state) - 
//                                                                                 cov_GBP.block(i*_dim_state, i*_dim_state, 2*_dim_state, 2*_dim_state);
// }

// std::cout << "Norm of joint_cov = " << cov_GBP.norm() << "  Sum = " << cov_GBP.cwiseAbs().sum() << std::endl;
// std::cout << "Norm of cov_error = " << cov_error.norm() << "  Sum = " << cov_error.cwiseAbs().sum() << std::endl;


#endif // GVI_GH_IMPL_H

#pragma once

#ifndef GVI_GH_IMPL_H
#define GVI_GH_IMPL_H

using namespace Eigen;

#include <stdexcept>
#include <optional>
#include <omp.h>
#include "helpers/CudaOperation.h"
#include <Eigen/IterativeLinearSolvers>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace gvi{

template <typename Factor, typename CudaClass>
void GVIGH<Factor, CudaClass>::switch_to_high_temperature(){
    std::cout << "Switching to high temperature.." << std::endl;
    #pragma omp parallel for
    for (auto& i_factor : _vec_factors) {
        i_factor->factor_switch_to_high_temperature();
    }
    this->_temperature = this->_high_temperature;
    // this->initilize_precision_matrix();
}

template <typename Factor, typename CudaClass>
void GVIGH<Factor, CudaClass>::classify_factors(){
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
template <typename Factor, typename CudaClass>
void GVIGH<Factor, CudaClass>::optimize(std::optional<bool> verbose)
{
    // default verbose
    bool is_verbose = verbose.value_or(true);
    bool is_lowtemp = true;
    bool converged = false;

    if (_save_data){
        _res_recorder.init_data(_save_covariance);
    }
    Timer timer;

    // Initialize the cuda and give value to _sigma_rows and _dim_conf
    timer.start();
    cuda_init(_vec_nonlinear_factors.size());
    std::cout << "Time for initializing cuda: " << timer.end_mis() << " ms" << std::endl;

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

        timer.start();

        // 135 ms when n_states = 500
        auto [cost_iter, fact_costs_iter, dmu, dprecision] = factor_cost_vector_cuda();

        if (is_verbose){
            std::cout << "--- cost_iter ---" << std::endl << cost_iter << std::endl;
            // std::cout << "Factor Costs:" << fact_costs_iter.transpose() << std::endl;
        }

        std::cout << "Time for computing factor costs: " << timer.end_mus_output() << " us" << std::endl;

        // This update_data takes about 80 ms each iteration
        if (_save_data){
            _res_recorder.update_data(_mu, _covariance, _precision, cost_iter, fact_costs_iter);
        }
        
        int cnt = 0;
        int B = 1;
        double step_size = _step_size_base;

        timer.start();
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

                break;
            }
        }
        std::cout << "backtracking time: " << B << std::endl;
        std::cout << "Time for backtracking: " << timer.end_mus_output() << " us" << std::endl;
    }

    cuda_free();

    if (_save_data){
        std::cout << "=========== Saving Data ===========" << std::endl;
        save_data(is_verbose);
    }

}

template <typename Factor, typename CudaClass>
void GVIGH<Factor, CudaClass>::optimize_time_test()
{
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

        auto result_cuda = factor_cost_vector_cuda_time();
        double cost_iter = std::get<0>(result_cuda);
        VectorXd fact_costs_iter = std::get<1>(result_cuda);
        VectorXd dmu = std::get<2>(result_cuda);
        SpMat dprecision = std::get<3>(result_cuda);

        int cnt = 0;
        int B = 1;
        double step_size = _step_size_base;

        // backtracking
        while (true)
        {   
            // new step size
            step_size = step_size * 0.75;

            auto onestep_res = onestep_linesearch(step_size, dmu, dprecision);

            double new_cost = std::get<0>(onestep_res);
            VectorXd new_mu = std::get<1>(onestep_res);
            auto new_precision = std::get<2>(onestep_res);

            if (new_cost < cost_iter){
                break;
            }else{
                cnt += 1;
            }

            if (cnt > _niters_backtrack)
            {
                if (is_lowtemp){
                    this->switch_to_high_temperature();
                    is_lowtemp = false;
                }else{
                    converged = true;
                }
                break;
            }
        }
    }
}

template <typename Factor, typename CudaClass>
std::tuple<double, VectorXd, VectorXd, SpMat> GVIGH<Factor, CudaClass>::factor_cost_vector_cuda(const VectorXd& fill_joint_mean, SpMat& joint_precision)
{
    int n_nonlinear = _vec_nonlinear_factors.size();

    VectorXd fac_costs(_nfactors);
    VectorXd nonlinear_fac_cost(n_nonlinear);
    fac_costs.setZero();
    nonlinear_fac_cost.setZero();

    omp_set_num_threads(20);

    MatrixXd sigmapts_mat(_sigma_rows, n_nonlinear*_dim_conf);
    MatrixXd mean_mat(_dim_conf, n_nonlinear);
    MatrixXd covariance_matrix(_dim_conf, n_nonlinear*_dim_conf);
    MatrixXd sigma(_sigma_rows, _dim_conf);

    VectorXd E_phi_mat(n_nonlinear);
    VectorXd dmu_mat(_dim_conf * n_nonlinear);
    MatrixXd ddmu_mat(_dim_conf, _dim_conf * n_nonlinear);

    #pragma omp parallel for
    for (int i = 0; i < n_nonlinear; i++)
    {
        auto &opt_k = _vec_nonlinear_factors[i];
        mean_mat.col(i) = opt_k->_mu;
        covariance_matrix.block(0, i * _dim_conf, _dim_conf, _dim_conf) = opt_k->covariance();
    }

    compute_sigmapts(mean_mat, covariance_matrix, _dim_conf, n_nonlinear, sigma);

    // Compute the cost and derivatives of the nonlinear factors
    dmuIntegration(sigmapts_mat, mean_mat, nonlinear_fac_cost, dmu_mat, ddmu_mat, _dim_conf);
    E_phi_mat = nonlinear_fac_cost;
    // nonlinear_fac_cost = nonlinear_fac_cost / this ->_temperature * _delta_t;
    nonlinear_fac_cost = nonlinear_fac_cost / this ->_temperature;

    #pragma omp parallel for
    for (int i = 0; i < _vec_factors.size(); i++)
    {
        auto &opt_k = _vec_factors[i];
        if (opt_k->linear_factor()) // matrix multiplication between dim_state x dim_state and dim_state * dim_state
            fac_costs(i) = opt_k->fact_cost_value(_mu, _covariance);
        else
            fac_costs(i) = nonlinear_fac_cost(opt_k->_start_index - 1);
    }

    double value = fac_costs.sum();
    SparseLDLT ldlt(joint_precision);
    VectorXd vec_D = ldlt.vectorD();

    double cost = value + vec_D.array().log().sum() / 2;

    double entropy = vec_D.array().log().sum() / 2;
    double collision_cost = nonlinear_fac_cost.sum();
    double prior_cost = fac_costs.sum() - collision_cost;

    std::cout << "Prior Cost: " << prior_cost << std::endl;
    std::cout << "Collision Cost: " << collision_cost << std::endl;
    std::cout << "Entropy: " << entropy << std::endl;
    if (prior_cost < 0 || collision_cost < 0 || entropy < 0)
        std::cout << "Negative cost: " << fac_costs.transpose() << std::endl;

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
                MatrixXd ddmu_i = ddmu_mat.block(0, index*_dim_conf, _dim_conf, _dim_conf);
                VectorXd dmu_i = dmu_mat.segment(index*_dim_conf, _dim_conf);

                opt_k->calculate_partial_V(ddmu_i, dmu_i, E_phi_mat(index));
            }

            Vdmu_private += opt_k->local2joint_dmu_insertion();
            Vddmu_private += opt_k->local2joint_dprecision_triplet();
        }

        #pragma omp critical
        {
            Vdmu_sum += Vdmu_private;
            Vddmu_sum += Vddmu_private;
        }
    }

    _Vdmu = Vdmu_sum;
    _Vddmu = Vddmu_sum;

    SpMat dprecision = _Vddmu - _precision;

    // BiCGSTAB solver is much faster than the QR solver while maintaining the same accuracy
    Eigen::BiCGSTAB<SpMat, IncompleteLUT<double>> solver;
    solver.setTolerance(1e-6);
    VectorXd dmu = solver.compute(_Vddmu).solve(-_Vdmu);

    double error_solver = (_Vddmu * dmu + _Vdmu).norm() / _Vdmu.norm();
    if (error_solver > 1e-10)
        std::cout << "Error of the solver: " << error_solver * 100 << "%" << std::endl;

    return std::make_tuple(cost, fac_costs, dmu, dprecision);
}

template <typename Factor, typename CudaClass>
std::tuple<double, VectorXd, VectorXd, SpMat> GVIGH<Factor, CudaClass>::factor_cost_vector_cuda_time(const VectorXd& fill_joint_mean, SpMat& joint_precision)
{
    static int flag = 0;
    Timer timer;

    int n_nonlinear = _vec_nonlinear_factors.size();

    VectorXd fac_costs(_nfactors);
    VectorXd nonlinear_fac_cost(n_nonlinear);
    fac_costs.setZero();
    nonlinear_fac_cost.setZero();

    omp_set_num_threads(20);

    MatrixXd sigmapts_mat(_sigma_rows, _dim_conf*n_nonlinear);
    MatrixXd mean_mat(_dim_conf, n_nonlinear);
    MatrixXd covariance_matrix(_dim_conf, _dim_conf*n_nonlinear);
    MatrixXd sigma(_sigma_rows, _dim_conf*n_nonlinear);

    VectorXd E_phi_mat(n_nonlinear);
    VectorXd dmu_mat(_dim_conf * n_nonlinear);
    MatrixXd ddmu_mat(_dim_conf, _dim_conf * n_nonlinear);

    // if (flag % 5 == 0)
    //     timer.start();
    
    // #pragma omp parallel for
    // for (int i = 0; i < n_nonlinear; i++)
    // {
    //     auto &opt_k = _vec_nonlinear_factors[i];
    //     sigmapts_mat.block(0, i*_dim_conf, _sigma_rows, _dim_conf) = opt_k->sigma_matrix();
    // }
    // copySigmaPoints(sigmapts_mat);

    // if (flag % 5 == 0)
    //     std::cout << "Sigma Points CPU time: " << timer.end_mus_output() << " us" << std::endl;

    if (flag % 5 == 0)
        timer.start();

    #pragma omp parallel for
    for (int i = 0; i < n_nonlinear; i++)
    {
        auto &opt_k = _vec_nonlinear_factors[i];
        mean_mat.col(i) = opt_k->_mu;
        covariance_matrix.block(0, i * _dim_conf, _dim_conf, _dim_conf) = opt_k->covariance();
    }
    
    compute_sigmapts(mean_mat, covariance_matrix, _dim_conf, n_nonlinear, sigma);

    if (flag % 5 == 0)
        std::cout << "Sigma Points computaiton time: " << timer.end_mus_output() << " us" << std::endl;
    
    if (flag % 5 == 0)
        timer.start();

    // Compute the cost of the nonlinear factors
    dmuIntegration(sigmapts_mat, mean_mat, nonlinear_fac_cost, dmu_mat, ddmu_mat, _dim_conf);
    E_phi_mat = nonlinear_fac_cost;
    nonlinear_fac_cost = nonlinear_fac_cost / this ->_temperature;

    if (flag % 5 == 0)
        std::cout << "Nonlinear Factors computation time: " << timer.end_mus_output() << " us" << std::endl;

    if (flag % 5 == 0)
        timer.start();

    #pragma omp parallel for
    for (int i = 0; i < _vec_factors.size(); i++)
    {
        auto &opt_k = _vec_factors[i];
        if (opt_k->linear_factor()) // matrix multiplication between dim_state x dim_state and dim_state * dim_state
            fac_costs(i) = opt_k->fact_cost_value(_mu, _covariance);
        else
            fac_costs(i) = nonlinear_fac_cost(opt_k->_start_index - 1);
    }

    double value = fac_costs.sum();
    SparseLDLT ldlt(joint_precision);
    VectorXd vec_D = ldlt.vectorD();

    double cost = value + vec_D.array().log().sum() / 2;

    if (flag % 5 == 0)
        std::cout << "Cost computation time: " << timer.end_mus_output() << " us" << std::endl;

    // double entropy = vec_D.array().log().sum() / 2;
    // double collision_cost = nonlinear_fac_cost.sum();
    // double prior_cost = fac_costs.sum() - collision_cost;
    
    // std::cout << "Prior Cost: " << prior_cost << std::endl;
    // std::cout << "Collision Cost: " << collision_cost << std::endl;
    // std::cout << "Entropy: " << entropy << std::endl;

    if (flag % 5 == 0)
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
                MatrixXd ddmu_i = ddmu_mat.block(0, index*_dim_conf, _dim_conf, _dim_conf);;
                VectorXd dmu_i = dmu_mat.segment(index*_dim_conf, _dim_conf);

                opt_k->calculate_partial_V(ddmu_i, dmu_i, E_phi_mat(index));
            }

            Vdmu_private += opt_k->local2joint_dmu_insertion();
            Vddmu_private += opt_k->local2joint_dprecision_triplet();
        }

        #pragma omp critical
        {
            Vdmu_sum += Vdmu_private;
            Vddmu_sum += Vddmu_private;
        }
    }

    _Vdmu = Vdmu_sum;
    _Vddmu = Vddmu_sum;

    if (flag % 5 == 0)
        std::cout << "Derivative computation time: " << timer.end_mus_output() << " us" << std::endl;

    if (flag % 5 == 0)
        timer.start();

    SpMat dprecision = _Vddmu - _precision;

    Eigen::BiCGSTAB<SpMat, IncompleteLUT<double>> solver;
    solver.setTolerance(1e-6);
    solver.compute(_Vddmu);
    VectorXd dmu = solver.solve(-_Vdmu);

    // The speed of BiCGSTAB is much faster then QR solver, while still maintaining the same accuracy
    // VectorXd dmu = solveWithCuSolverQR(_Vddmu, -_Vdmu);

    if (flag % 5 == 0)
        std::cout << "Solver time: " << timer.end_mus_output() << " us" << std::endl;


    // Compare the solvers
    // if (flag % 5 == 0){
    //     timer.start();
    //     Eigen::ConjugateGradient<SpMat, Eigen::Upper, Eigen::IncompleteLUT<double>> solver_lut;
    //     VectorXd dmu_lut = solver_lut.compute(_Vddmu).solve(-_Vdmu);
    //     std::cout << "Solver time lut: " << timer.end_mus_output() << " us" << std::endl;

    //     timer.start();
    //     Eigen::BiCGSTAB<SpMat, IncompleteLUT<double>> bicgstab_solver;
    //     bicgstab_solver.setTolerance(1e-6);
    //     bicgstab_solver.compute(_Vddmu);
    //     VectorXd dmu_bicgstab = bicgstab_solver.solve(-_Vdmu);
    //     std::cout << "Solver time bicgstab: " << timer.end_mus_output() << " us" << std::endl;

    //     double Vdmu_norm = _Vdmu.norm();
    //     std::cout << "Ratio of QR solver: " << (_Vdmu + _Vddmu * dmu).norm() / Vdmu_norm * 100 << "%" << std::endl;
    //     std::cout << "Ratio of Conjugate Gradient solver: " << (_Vdmu + _Vddmu * dmu_lut).norm() / Vdmu_norm * 100 << "%" << std::endl;
    //     std::cout << "Ratio of BiCGSTAB solver: " << (_Vdmu + _Vddmu * dmu_bicgstab).norm() / Vdmu_norm * 100 << "%" << std::endl << std::endl;
    // }
    
    flag++;

    return std::make_tuple(cost, fac_costs, dmu, dprecision);
}


template <typename Factor, typename CudaClass>
void GVIGH<Factor, CudaClass>::time_test()
{
    cuda_init(_vec_nonlinear_factors.size());

    Timer timer;
    int n_repeat = 20;

    // // Time Test for Nonlinear Cost Evaluation
    // std::vector<double> times_evaluation;
    // times_evaluation.reserve(n_repeat);

    // for (int i=0; i < n_repeat+1; i++){
    //     timer.start();
    //     auto result_cuda = factor_cost_vector_cuda_time();
    //     double time = timer.end_mis();
    //     if (i != 0)
    //     times_evaluation.push_back(time);  // The first time need to initialize
    // }

    // double average_time = std::accumulate(times_evaluation.begin(), times_evaluation.end(), 0.0) / n_repeat;
    // double min_time = *std::min_element(times_evaluation.begin(), times_evaluation.end());
    // double max_time = *std::max_element(times_evaluation.begin(), times_evaluation.end());

    // std::cout << "% " << _vec_nonlinear_factors.size() + 2 << std::endl;
    // std::cout << "% GPU average: " << average_time << " ms" << std::endl;
    // std::cout << "% GPU min: " << min_time << " ms" << std::endl;
    // std::cout << "% GPU max: " << max_time << " ms" << std::endl;

    // std::cout << "% [ " << times_evaluation[0];
    // for (int i = 1; i < times_evaluation.size(); ++i) {
    //     std::cout << ", " << times_evaluation[i];
    // }
    // std::cout << " ]" << std::endl;

    // n_repeat = 5;
    // std::vector<double> times_optimization;
    // times_optimization.reserve(n_repeat);

    // for (int i=0; i < n_repeat+1; i++){
    //     timer.start();
    //     optimize_time_test();
    //     double time = timer.end_mis();
    //     if (i != 0)
    //     times_optimization.push_back(time);  // The first time need to initialize
    // }

    // average_time = std::accumulate(times_optimization.begin(), times_optimization.end(), 0.0) / n_repeat;
    // min_time = *std::min_element(times_optimization.begin(), times_optimization.end());
    // max_time = *std::max_element(times_optimization.begin(), times_optimization.end());

    // std::cout << "% " << _vec_nonlinear_factors.size() + 2 << std::endl;
    // std::cout << "% GPU Optimize average: " << average_time << " ms" << std::endl;
    // std::cout << "% GPU Optimize min: " << min_time << " ms" << std::endl;
    // std::cout << "% GPU Optimize max: " << max_time << " ms" << std::endl;

    // std::cout << "% [ " << times_optimization[0];
    // for (int i = 1; i < times_optimization.size(); ++i) {
    //     std::cout << ", " << times_optimization[i];
    // }
    // std::cout << " ]" << std::endl;


    // // Inverse Time Comparison
    // std::vector<double> times_GBP, times_inverse;
    // times_GBP.reserve(n_repeat);
    // times_inverse.reserve(n_repeat);

    // for (int i=0; i < n_repeat+1; i++){
    //     timer.start();
    //     _covariance = inverse_GBP(_precision);
    //     double time = timer.end_mis();
    //     if (i != 0)
    //     times_GBP.push_back(time);  // The first time need to initialize
    // }

    // // for (int i=0; i < n_repeat+1; i++){
    // //     timer.start();
    // //     _covariance = inverse(_precision);
    // //     double time = timer.end_mis();
    // //     if (i != 0)
    // //     times_inverse.push_back(time);  // The first time need to initialize
    // // }

    // double average_time = std::accumulate(times_GBP.begin(), times_GBP.end(), 0.0) / n_repeat;

    // std::cout << "Dimension: " << _dim << std::endl;
    // std::cout << "% GBP average: " << average_time << " ms" << std::endl;

    // std::cout << "% [" << times_GBP[0];
    // for (int i = 1; i < times_GBP.size(); ++i) {
    //     std::cout << ", " << times_GBP[i];
    // }
    // std::cout << "]" << std::endl;

    // // average_time = std::accumulate(times_inverse.begin(), times_inverse.end(), 0.0) / n_repeat;
    // // std::cout << "% Inverse average: " << average_time << " ms" << std::endl;
    // // std::cout << "% [" << times_inverse[0];
    // // for (int i = 1; i < times_inverse.size(); ++i) {
    // //     std::cout << ", " << times_inverse[i];
    // // }
    // // std::cout << "]" << std::endl;

    cuda_free();

}

template <typename Factor, typename CudaClass>
inline void GVIGH<Factor, CudaClass>::set_precision(const SpMat &new_precision)
{
    _precision = new_precision;
    if (_GBP_inverse)
        _covariance = inverse_GBP(_precision);
    else
        inverse_inplace();
    // _covariance = chain_splash_GBP(_precision);

    #pragma omp parallel for
    for (auto &factor : _vec_factors)
    {
        factor->update_precision_from_joint(_covariance);
    }
}

/**
 * @brief Compute the total cost function value given a state.
 */
template <typename Factor, typename CudaClass>
double GVIGH<Factor, CudaClass>::cost_value_cuda(const VectorXd& fill_joint_mean, SpMat& joint_precision)
{
    Timer timer;
    static int flag = 0;

    int n_nonlinear = _vec_nonlinear_factors.size();
    VectorXd nonlinear_fac_cost(n_nonlinear);
    nonlinear_fac_cost.setZero();

    if (flag % 5 == 0)
        timer.start();

    SpMat joint_cov;
    if (_GBP_inverse)
        joint_cov = inverse_GBP(joint_precision);
    else
        joint_cov = inverse(joint_precision);

    if (flag % 5 == 0)
        std::cout << "GBP Inverse computation time: " << timer.end_mus_output() << " us" << std::endl;

    MatrixXd sigmapts_mat(_sigma_rows, n_nonlinear*_dim_conf);
    MatrixXd mean_mat(_dim_conf, n_nonlinear);
    MatrixXd covariance_matrix(_dim_conf, n_nonlinear*_dim_conf);
    MatrixXd sigma(_sigma_rows, _dim_conf);

    #pragma omp parallel for
    for (int i = 0; i < n_nonlinear; i++)
    {
        mean_mat.col(i) = fill_joint_mean.segment((i+1)*_dim_state, _dim_conf);
        covariance_matrix.block(0, i * _dim_conf, _dim_conf, _dim_conf) = joint_cov.block((i+1)*_dim_state, (i+1)*_dim_state, _dim_conf, _dim_conf);
    }

    compute_sigmapts(mean_mat, covariance_matrix, _dim_conf, n_nonlinear, sigma);

    // Compute the cost of the nonlinear factors
    newCostIntegration(sigmapts_mat, nonlinear_fac_cost, _dim_conf);
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

    flag++;

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
template <typename Factor, typename CudaClass>
SpMat GVIGH<Factor, CudaClass>::residual_splash_GBP(const SpMat &Precision)
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
template <typename Factor, typename CudaClass>
SpMat GVIGH<Factor, CudaClass>::chain_splash_GBP(const SpMat &Precision)
{
    std::vector<Message> factors(2*_num_states-1);
    std::vector<Message> joint_factors(_num_states-1);
    Message variable_message;
    MatrixXd covariance(_dim, _dim);
    covariance.setZero();
    Timer timer;

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

    std::vector<Message> forward_messages_GBP(_num_states);
    std::vector<Message> backward_messages_GBP(_num_states);
    for (int i = 0; i < _num_states; i++) {
        forward_messages[i].first = VectorXd::Constant(1, i);
        forward_messages[i].second = MatrixXd::Zero(_dim_state, _dim_state);
        backward_messages[i].first = VectorXd::Constant(1, i);
        backward_messages[i].second = MatrixXd::Zero(_dim_state, _dim_state);

        forward_messages_GBP[i].first = VectorXd::Constant(1, i);
        forward_messages_GBP[i].second = MatrixXd::Zero(_dim_state, _dim_state);
        backward_messages_GBP[i].first = VectorXd::Constant(1, i);
        backward_messages_GBP[i].second = MatrixXd::Zero(_dim_state, _dim_state);
    }
    
    // ChainSplash Iterative Updates with Block Boundary Exchange and Convergence Check
    const int max_iter = 100;  // maximum number of iterations
    const double tol = 1e-5;   // convergence tolerance
    // Set the number of threads approximately to sqrt(n)
    int num_threads = 4;
    // Partition the chain into blocks: block_size = ceil(n / num_threads)
    int block_size = (_num_states + num_threads - 1) / num_threads;
    
    // Containers to store previous iteration messages for residual computation.
    std::vector<MatrixXd> old_forward(_num_states);
    std::vector<MatrixXd> old_backward(_num_states);
    std::vector<double> residuals(_num_states, 0.0);
    double global_residual;
    bool converged = false;

    timer.start();

    // Begin a single parallel region for all iterations.
    #pragma omp parallel num_threads(num_threads) \
        shared(forward_messages, backward_messages, factors, old_forward, old_backward, residuals, global_residual, converged)
    {
        int iter = 0;
        while (iter < max_iter && !converged) {
            // --- Save current messages for residual computation in parallel ---
            #pragma omp for
            for (int i = 0; i < _num_states; i++) {
                old_forward[i] = forward_messages[i].second;
                old_backward[i] = backward_messages[i].second;
            }
            #pragma omp barrier  // Ensure all threads see the saved messages

            // --- Each thread processes its assigned block ---
            int b = omp_get_thread_num();
            int block_start = b * block_size;
            int block_end = std::min(_num_states - 1, block_start + block_size - 1);

            // Block internal forward update
            for (int i = block_start; i < block_end; i++) {
                Message var_msg = calculate_variable_message(forward_messages[i], factors[2 * i]);
                Message fac_msg = calculate_factor_message(var_msg, i + 1, factors[2 * i + 1]);
                forward_messages[i + 1] = fac_msg;
            }
            // Block internal backward update
            for (int i = block_end; i > block_start; i--) {
                Message var_msg = calculate_variable_message(backward_messages[i], factors[2 * i]);
                Message fac_msg = calculate_factor_message(var_msg, i - 1, factors[2 * i - 1]);
                backward_messages[i - 1] = fac_msg;
            }

            // --- Boundary exchange ---
            // Each thread updates the right boundary if not the last block.
            if (b < num_threads - 1) {
                Message var_msg_fwd = calculate_variable_message(forward_messages[block_end], factors[2 * block_end]);
                Message fac_msg_fwd = calculate_factor_message(var_msg_fwd, block_end + 1, factors[2 * block_end + 1]);
                forward_messages[block_end + 1] = fac_msg_fwd;
            }
            // Each thread updates the left boundary if not the first block.
            if (b > 0) {
                Message var_msg_bwd = calculate_variable_message(backward_messages[block_start], factors[2 * block_start]);
                Message fac_msg_bwd = calculate_factor_message(var_msg_bwd, block_start - 1, factors[2 * block_start - 1]);
                backward_messages[block_start - 1] = fac_msg_bwd;
            }
            #pragma omp barrier  // Ensure all block internal updates and boundary exchanges are completed

            // --- Compute global residual in parallel ---
            #pragma omp for
            for (int i = 0; i < _num_states; i++) {
                double diff_forward = (forward_messages[i].second - old_forward[i]).norm();
                double diff_backward = (backward_messages[i].second - old_backward[i]).norm();
                residuals[i] = std::max(diff_forward, diff_backward);
            }
            #pragma omp barrier  // Ensure all residuals are computed

            // Compute maximum residual (single thread)
            #pragma omp single
            {
                global_residual = 0.0;
                for (int i = 0; i < _num_states; i++) {
                    global_residual = std::max(global_residual, residuals[i]);
                }
                if (global_residual < tol) {
                    std::cout << "Converged after " << iter+1 << " iterations, residual = " << global_residual << std::endl;
                    converged = true;
                }
            }
            #pragma omp barrier  // Ensure all threads see the updated global_residual and converged flag

            iter++;
        }
    } // end parallel region

    std::cout << "ChainSplash: " << timer.end_mis() << " ms" << std::endl;
    timer.start();

    // Calculate messages between factors and variables
    for (int i = 0; i < _num_states - 1; i++) {
        variable_message = calculate_variable_message(forward_messages_GBP[i], factors[2 * i]);
        forward_messages_GBP[i + 1] = calculate_factor_message(variable_message, i + 1, factors[2 * i + 1]);
        int index = _num_states - 1 - i;
        variable_message = calculate_variable_message(backward_messages_GBP[index], factors[2 * index]);
        backward_messages_GBP[index - 1] = calculate_factor_message(variable_message, index - 1, factors[2 * index - 1]);
    }
    std::cout << "GBP: " << timer.end_mis() << " ms" << std::endl;

    // std::vector<double> forward_messages_error(_num_states);
    // std::vector<double> backward_messages_error(_num_states);
    // for (int i = 0; i < _num_states; i++) {
    //     forward_messages_error[i] = (forward_messages[i].second - forward_messages_GBP[i].second).norm();
    //     backward_messages_error[i] = (backward_messages[i].second - backward_messages_GBP[i].second).norm();
    // }
    // std::cout << "Sum of forward message errors: " << std::accumulate(forward_messages_error.begin(), forward_messages_error.end(), 0.0) << std::endl;
    // std::cout << "Sum of backward message errors: " << std::accumulate(backward_messages_error.begin(), backward_messages_error.end(), 0.0) << std::endl;
    
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
template <typename Factor, typename CudaClass>
SpMat GVIGH<Factor, CudaClass>::inverse_GBP(const SpMat &Precision)
{
    Timer timer;
    std::vector<Message> factors(2*_num_states-1);
    std::vector<Message> joint_factors(_num_states-1);
    Message variable_message;

    // Extract the factors from the precision matrix
    // The variable in factors are 0, {0,1}, 1, {1,2}, 2, ..., {_num_states-1,_num_states}, _num_states
    // timer.start();
    # pragma omp parallel for
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
    // std::cout << "Factor construction: " << timer.end_mus_output() << " us" << std::endl;
    
    // Initialize message
    // timer.start();
    std::vector<Message> forward_messages(_num_states);
    std::vector<Message> backward_messages(_num_states);
    for (int i = 0; i < _num_states; i++) {
        forward_messages[i].second.setZero();
        backward_messages[i].second.setZero();
    }
    forward_messages[0] = {VectorXd::Zero(1), MatrixXd::Zero(_dim_state, _dim_state)};
    backward_messages.back() = {VectorXd::Constant(1, _num_states-1), MatrixXd::Zero(_dim_state, _dim_state)};
    // std::cout << "Message initialization: " << timer.end_mus_output() << " us" << std::endl;

    // timer.start();
    // Calculate messages between factors and variables
    for (int i = 0; i < _num_states - 1; i++) {
        variable_message = calculate_variable_message(forward_messages[i], factors[2 * i]);
        forward_messages[i + 1] = calculate_factor_message(variable_message, i + 1, factors[2 * i + 1]);
        int index = _num_states - 1 - i;
        variable_message = calculate_variable_message(backward_messages[index], factors[2 * index]);
        backward_messages[index - 1] = calculate_factor_message(variable_message, index - 1, factors[2 * index - 1]);
    }
    // std::cout << "Message Passing: " << timer.end_mus_output() << " us" << std::endl;

    // timer.start();
    std::vector<Eigen::Triplet<double>> tripletList;
    if (_num_states == 1) {
        MatrixXd lambda = forward_messages[0].second + backward_messages[0].second + factors[0].second;
        MatrixXd variance = lambda.inverse();
        tripletList.reserve(_dim_state * _dim_state);
        for (int r = 0; r < _dim_state; ++r) {
            for (int c = 0; c < _dim_state; ++c) {
                tripletList.emplace_back(r, c, variance(r, c));
            }
        }
    }
    else if (_precise_inverse) {
        std::cout << "Precise Inverse" << std::endl;
        // Create a local container, where each iteration corresponds to a joint covariance block
        int nBlocks = 2 * _num_states - 1;
        std::vector<std::vector<Eigen::Triplet<double>>> localTripletVectors(nBlocks);

        #pragma omp parallel for
        for (int i = 0; i < nBlocks; ++i) {
            std::vector<Eigen::Triplet<double>> localTriplets;
            if (i % 2 == 0) {
                // Even index: diagonal block, compute the inverse using marginal precision
                int state = i / 2;
                localTriplets.reserve(_dim_state * _dim_state);

                MatrixXd lambda_marginal = factors[i].second;
                lambda_marginal += forward_messages[state].second;
                lambda_marginal += backward_messages[state].second;
                MatrixXd variance_marginal = lambda_marginal.inverse(); // Will not likely to introduce negative eigenvalues

                int base = state * _dim_state;
                for (int r = 0; r < _dim_state; ++r) {
                    for (int c = 0; c < _dim_state; ++c) {
                        localTriplets.emplace_back(base + r, base + c, variance_marginal(r, c));
                    }
                }
            } else {
                // Odd index: off-diagonal block (between adjacent states), computed similarly
                int state = (i - 1) / 2; // state corresponds to the index in joint_factors
                localTriplets.reserve(2 * _dim_state * _dim_state);

                MatrixXd lambda_joint = joint_factors[state].second;
                lambda_joint.block(0, 0, _dim_state, _dim_state) += forward_messages[state].second;
                lambda_joint.block(_dim_state, _dim_state, _dim_state, _dim_state) += backward_messages[state + 1].second;
                MatrixXd variance_joint = lambda_joint.inverse();

                // // Now the problem is that the lambda_joint is PD, but the variance_joint has negative eigenvalues

                // SelfAdjointEigenSolver<MatrixXd> solver(variance_joint);
                // // Retrieve eigenvalues
                // VectorXd eigenvalues = solver.eigenvalues();
                // if ((eigenvalues.array() < 0).any()) {
                //     std::cout << "Warning: Negative eigenvalues detected in joint covariance!" << std::endl;
                //     // for (int j = 0; j < eigenvalues.size(); ++j) {
                //     //     if (eigenvalues(j) < 0) {
                //     //         std::cout << "Eigenvalue " << j << ": " << eigenvalues(j) << std::endl;
                //     //     }
                //     // }
                // }

                // SelfAdjointEigenSolver<MatrixXd> solver_lambda(lambda_joint);
                // // Retrieve eigenvalues
                // eigenvalues = solver_lambda.eigenvalues();
                // if ((eigenvalues.array() < 0).any()) {
                //     std::cout << "Warning: Negative eigenvalues detected in joint lambda!" << std::endl;
                // }

                int base_row = state * _dim_state;
                int base_col = (state + 1) * _dim_state;
                for (int r = 0; r < _dim_state; ++r) {
                    for (int c = 0; c < _dim_state; ++c) {
                        // Fill the upper triangular block and the lower triangular block
                        localTriplets.emplace_back(base_row + r, base_col + c, variance_joint(r, _dim_state + c));
                        localTriplets.emplace_back(base_col + r, base_row + c, variance_joint(_dim_state + r, c));
                    }
                }
            }
            localTripletVectors[i] = std::move(localTriplets);
        }

        // Merge the local Triplet vectors from all threads
        int totalTriplets = (3 * _num_states - 2) * _dim_state * _dim_state;
        tripletList.reserve(totalTriplets);
        for (const auto &vec : localTripletVectors)
            tripletList.insert(tripletList.end(), vec.begin(), vec.end());
    }
    else{
        // Create a local container, where each iteration corresponds to a joint covariance block
        int nBlocks = _num_states - 1; // The number of iterations matches the number of joint_factors
        std::vector<std::vector<Eigen::Triplet<double>>> localTripletVectors(nBlocks);
    
        #pragma omp parallel for
        for (int i = 0; i < nBlocks; ++i) {
            std::vector<Eigen::Triplet<double>> localTriplets;
            // Each joint block contributes 3 _dim_state×_dim_state sub-blocks.
            // If it is the last iteration (i == _num_states - 2), an additional sub-block is added.
            localTriplets.reserve(3 * _dim_state * _dim_state + ((i == nBlocks - 1) ? _dim_state * _dim_state : 0));
    
            // Copy joint_factor and add messages
            MatrixXd lambda_joint = joint_factors[i].second;
            lambda_joint.block(0, 0, _dim_state, _dim_state) += forward_messages[i].second;
            lambda_joint.block(_dim_state, _dim_state, _dim_state, _dim_state) += backward_messages[i + 1].second;
            MatrixXd variance_joint = lambda_joint.inverse();
    
            // Base row and column indices
            int base_row = i * _dim_state;
            int base_col = i * _dim_state;
            int next_base = base_row + _dim_state; // Corresponds to i+1
    
            // Fill in the diagonal, upper triangular, and lower triangular blocks
            for (int r = 0; r < _dim_state; ++r) {
                for (int c = 0; c < _dim_state; ++c) {
                    localTriplets.emplace_back(base_row + r, base_col + c, variance_joint(r, c));
                    localTriplets.emplace_back(base_row + r, next_base + c, variance_joint(r, _dim_state + c));
                    localTriplets.emplace_back(next_base + r, base_col + c, variance_joint(_dim_state + r, c));
                    if (i == nBlocks - 1) {
                        localTriplets.emplace_back(next_base + r, next_base + c, variance_joint(_dim_state + r, _dim_state + c));
                    }
                }
            }
            localTripletVectors[i] = std::move(localTriplets);
        }
    
        // Merge the local Triplet vectors from all threads
        int totalTriplets = (3 * _num_states - 2) * _dim_state * _dim_state;
        tripletList.reserve(totalTriplets);
        for (const auto &vec : localTripletVectors)
            tripletList.insert(tripletList.end(), vec.begin(), vec.end());
    }
    
    SpMat covariance_sparse(_dim, _dim);
    covariance_sparse.setFromTriplets(tripletList.begin(), tripletList.end());
    // std::cout << "Conversion to sparse (Direct Triplets): " << timer.end_mus_output() << " us" << std::endl;
    
    return covariance_sparse;
}

/**
 * @brief Compute the message of factors in GBP.
 */
template <typename Factor, typename CudaClass>
Message GVIGH<Factor, CudaClass>::calculate_factor_message(const Message &input_message, int target, const Message &factor_potential) {
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



// This function solves the linear system A*x = b using sparse QR decomposition via cuSolver.
// Input:
//   Vddmu: Sparse matrix A (of type Eigen::SparseMatrix<double>)
//   Vdmu:  Dense vector b (of type Eigen::VectorXd)
// Output:
//   Returns the solution vector x.
template <typename Factor, typename CudaClass>
VectorXd GVIGH<Factor, CudaClass>::solveWithCuSolverQR(const SpMat& Vddmu, const VectorXd& Vdmu)
{
    // Convert the input sparse matrix to row-major format (CSR representation)
    Eigen::SparseMatrix<double, Eigen::RowMajor> A = Vddmu;
    int m = A.rows();
    int nnz = A.nonZeros();

    // Retrieve CSR arrays from Eigen
    // outerIndexPtr: row pointers, innerIndexPtr: column indices, valuePtr: nonzero values
    const int* eigenOuter = A.outerIndexPtr();
    const int* eigenInner = A.innerIndexPtr();
    const double* eigenValues = A.valuePtr();

    // Copy the Eigen CSR data into std::vector containers
    std::vector<int> h_csrRowPtr(eigenOuter, eigenOuter + m + 1);
    std::vector<int> h_csrColInd(eigenInner, eigenInner + nnz);
    std::vector<double> h_csrVal(eigenValues, eigenValues + nnz);

    // Prepare the right-hand side vector b = Vdmu
    VectorXd b = Vdmu;
    std::vector<double> h_b(b.data(), b.data() + b.size());

    // Create cuSolver and cuSparse handles
    cusolverSpHandle_t cusolverH;
    cusolverSpCreate(&cusolverH);

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    // Allocate device memory
    double* d_csrVal;
    int* d_csrRowPtr;
    int* d_csrColInd;
    double* d_b;
    double* d_x;
    cudaMalloc((void**)&d_csrVal, nnz * sizeof(double));
    cudaMalloc((void**)&d_csrRowPtr, (m + 1) * sizeof(int));
    cudaMalloc((void**)&d_csrColInd, nnz * sizeof(int));
    cudaMalloc((void**)&d_b, m * sizeof(double));
    cudaMalloc((void**)&d_x, m * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_csrVal, h_csrVal.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrRowPtr, h_csrRowPtr.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, h_csrColInd.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), m * sizeof(double), cudaMemcpyHostToDevice);

    // Set tolerance and reordering flag for the QR solver
    double tol = 1e-10;
    int reorder = 1;
    int singularity = 0;  // Output parameter: singularity info

    // Solve the system A*x = b using QR decomposition
    cusolverSpDcsrlsvqr(cusolverH, m, nnz, descrA, d_csrVal, d_csrRowPtr, d_csrColInd, d_b,
                          tol, reorder, d_x, &singularity);

    // Copy the solution from device to host
    std::vector<double> h_x(m);
    cudaMemcpy(h_x.data(), d_x, m * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory and destroy handles
    cudaFree(d_csrVal);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_b);
    cudaFree(d_x);
    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(cusolverH);

    // Map the solution to an Eigen vector and return
    VectorXd x = Eigen::Map<VectorXd>(h_x.data(), m);
    return x;
}

}


#endif // GVI_GH_IMPL_H





// /**
//  * @brief Compute the covariances using Gaussian Belief Propagation.
//  */
// template <typename Factor, typename CudaClass>
// SpMat GVIGH<Factor, CudaClass>::inverse_GBP(const SpMat &Precision)
// {
//     Timer timer;
//     std::vector<Message> factors(2*_num_states-1);
//     std::vector<Message> joint_factors(_num_states-1);
//     Message variable_message;
//     MatrixXd covariance(_dim, _dim);
//     MatrixXd covariance_joint(_dim, _dim);
//     covariance.setZero();
//     covariance_joint.setZero();

//     // Extract the factors from the precision matrix
//     // The variable in factors are 0, {0,1}, 1, {1,2}, 2, ..., {_num_states-1,_num_states}, _num_states
//     timer.start();
//     for (int i = 0; i < 2*_num_states-1; i++) {
//         int var = i / 2;
//         if (i % 2 == 0) {
//             VectorXd variable(1);
//             variable << var;
//             MatrixXd lambda = Precision.block(_dim_state * var, _dim_state * var, _dim_state, _dim_state);
//             factors[i] = {variable, lambda};
//         }
//         else {
//             VectorXd variable(2);
//             variable << var, var + 1;
//             MatrixXd lambda = MatrixXd::Zero(2 * _dim_state, 2 * _dim_state);
//             lambda.block(0, _dim_state, _dim_state, _dim_state) = Precision.block(_dim_state * var, _dim_state * (var + 1), _dim_state, _dim_state);
//             lambda.block(_dim_state, 0, _dim_state, _dim_state) = Precision.block(_dim_state * (var + 1), _dim_state * var, _dim_state, _dim_state);
//             factors[i] = {variable, lambda};
//             joint_factors[var] = {variable, Precision.block(_dim_state * var, _dim_state * var, 2 * _dim_state, 2 * _dim_state)};
//         }
//     }
//     std::cout << "Factor construction: " << timer.end_mus_output() << " us" << std::endl;
    
//     // Initialize message
//     std::vector<Message> forward_messages(_num_states);
//     std::vector<Message> backward_messages(_num_states);
//     for (int i = 0; i < _num_states; i++) {
//         forward_messages[i].second.setZero();
//         backward_messages[i].second.setZero();
//     }
//     forward_messages[0] = {VectorXd::Zero(1), MatrixXd::Zero(_dim_state, _dim_state)};
//     backward_messages.back() = {VectorXd::Constant(1, _num_states-1), MatrixXd::Zero(_dim_state, _dim_state)};

//     timer.start();
//     // Calculate messages between factors and variables
//     for (int i = 0; i < _num_states - 1; i++) {
//         variable_message = calculate_variable_message(forward_messages[i], factors[2 * i]);
//         forward_messages[i + 1] = calculate_factor_message(variable_message, i + 1, factors[2 * i + 1]);
//         int index = _num_states - 1 - i;
//         variable_message = calculate_variable_message(backward_messages[index], factors[2 * index]);
//         backward_messages[index - 1] = calculate_factor_message(variable_message, index - 1, factors[2 * index - 1]);
//     }
//     std::cout << "Message Passing: " << timer.end_mus_output() << " us" << std::endl;

//     if (_num_states == 1){
//         MatrixXd lambda = forward_messages[0].second + backward_messages[0].second + factors[0].second;
//         MatrixXd variance = lambda.inverse();
//         covariance.block(0, 0, _dim_state, _dim_state) = variance;
//     }

//     timer.start();
//     #pragma omp parallel for
//     for (int i = 0; i < _num_states - 1; ++i) {
//         MatrixXd lambda_joint = joint_factors[i].second;
//         lambda_joint.block(0, 0, _dim_state, _dim_state) += forward_messages[i].second;
//         lambda_joint.block(_dim_state, _dim_state, _dim_state, _dim_state) += backward_messages[i + 1].second;
//         MatrixXd variance_joint = lambda_joint.inverse();

//         // Update the covariance matrix
//         covariance.block(i * _dim_state, i * _dim_state, _dim_state, _dim_state) = variance_joint.block(0, 0, _dim_state, _dim_state);
//         covariance.block(i * _dim_state, (i + 1) * _dim_state, _dim_state, _dim_state) = variance_joint.block(0, _dim_state, _dim_state, _dim_state);
//         covariance.block((i + 1) * _dim_state, i * _dim_state, _dim_state, _dim_state) = variance_joint.block(_dim_state, 0, _dim_state, _dim_state);

//         // For the last block, update the bottom-right sub-block
//         if (i == _num_states - 2) {
//             covariance.block((i + 1) * _dim_state, (i + 1) * _dim_state, _dim_state, _dim_state) = variance_joint.block(_dim_state, _dim_state, _dim_state, _dim_state);
//         }
//     }
//     std::cout << "Marginal Covariance omp: " << timer.end_mus_output() << " us" << std::endl;

//     timer.start();
//     // Construct a sparse matrix using Triplets
//     std::vector<Eigen::Triplet<double>> tripletList;
//     int nonZeroCount = (3 * _num_states - 2) * _dim_state * _dim_state;
//     tripletList.reserve(nonZeroCount);

//     for (int i = 0; i < _num_states; ++i) {
//         int row_offset = i * _dim_state;
//         int col_offset = i * _dim_state;
//         MatrixXd diagBlock = covariance.block(row_offset, col_offset, _dim_state, _dim_state);
//         for (int r = 0; r < _dim_state; ++r) {
//             for (int c = 0; c < _dim_state; ++c) {
//                 tripletList.emplace_back(row_offset + r, col_offset + c, diagBlock(r, c));
//             }
//         }
//     }

//     // Add upper triangular off-diagonal blocks
//     for (int i = 0; i < _num_states - 1; ++i) {
//         int row_offset = i * _dim_state;
//         int col_offset = (i + 1) * _dim_state;
//         MatrixXd upperBlock = covariance.block(row_offset, col_offset, _dim_state, _dim_state);
//         for (int r = 0; r < _dim_state; ++r) {
//             for (int c = 0; c < _dim_state; ++c) {
//                 tripletList.emplace_back(row_offset + r, col_offset + c, upperBlock(r, c));
//             }
//         }
//     }

//     // Add lower triangular off-diagonal blocks
//     for (int i = 0; i < _num_states - 1; ++i) {
//         int row_offset = (i + 1) * _dim_state;
//         int col_offset = i * _dim_state;
//         MatrixXd lowerBlock = covariance.block(row_offset, col_offset, _dim_state, _dim_state);
//         for (int r = 0; r < _dim_state; ++r) {
//             for (int c = 0; c < _dim_state; ++c) {
//                 tripletList.emplace_back(row_offset + r, col_offset + c, lowerBlock(r, c));
//             }
//         }
//     }

//     SpMat covariance_sparse(covariance.rows(), covariance.cols());
//     covariance_sparse.setFromTriplets(tripletList.begin(), tripletList.end());
//     std::cout << "Conversion to sparse (Triplets): " << timer.end_mus_output() << " us" << std::endl;

//     return covariance_sparse;
// }

#pragma once

#ifndef PROXKL_GH_IMPL_H
#define PROXKL_GH_IMPL_H

using namespace Eigen;
#include <stdexcept>
#include <optional>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace gvi{

template <typename Factor>
std::tuple<double, VectorXd, SpMat> ProxKLGH<Factor>::onestep_linesearch(const double &step_size, 
                                                                            const VectorXd& dmu, 
                                                                            const SpMat& dprecision)
{
    SpMat K_inverse(this->_precision.rows(), this->_precision.cols());
    VectorXd mu(this->_mu.size());

    K_inverse.insert(0, 0) = 1.0 / 9.0;
    mu(0) = 20.0;

    SpMat new_precision; 
    VectorXd new_mu; 
    new_mu.setZero(); new_precision.setZero();

    // update mu and precision matrix
    Eigen::ConjugateGradient<SpMat> solver;
    new_mu = solver.compute(K_inverse + this->_precision / step_size).solve(-dmu + K_inverse * mu + this->_precision * this->_mu / step_size);
    new_precision = step_size / (step_size + 1) * (dprecision + K_inverse + this->_precision / step_size);

    // // Test the error of the solver
    // double K_inv_value = K_inverse.coeff(0, 0);
    // double precision_value = this->_precision.coeff(0, 0);
    // VectorXd new_mu1 = (-dmu + K_inverse * mu + this->_precision * this->_mu / step_size) / (K_inv_value + precision_value / step_size);
    // std:cout << "error: " << (new_mu1 - new_mu).norm() << std::endl;
    

    // std::cout << "mu: " << this->_mu << std::endl;
    // std::cout << "new mu: " << new_mu << std::endl;
    // std::cout << "precision: " << this->_precision.coeff(0, 0) << std::endl;
    // std::cout << "new precision: " << new_precision.coeff(0, 0) << std::endl;


    // new_mu = this->_mu - step_size * dmu;
    // new_precision = 1 / (step_size + 1) * this->_precision + step_size /(step_size + 1)  * dprecision;

    // new cost
    double new_cost = Base::cost_value(new_mu, new_precision);
    return std::make_tuple(new_cost, new_mu, new_precision);

}

/**
 * @brief One step of optimization.
 */
template <typename Factor>
std::tuple<VectorXd, SpMat> ProxKLGH<Factor>::compute_gradients(std::optional<double>step_size){
    _Vdmu.setZero();
    _Vddmu.setZero();

    VectorXd Vdmu_sum = VectorXd::Zero(_Vdmu.size());
    SpMat Vddmu_sum = SpMat(_Vddmu.rows(), _Vddmu.cols());

    /**
     * @brief OMP parallel on cpu.
     */
    omp_set_num_threads(20); 

    #pragma omp parallel
    {
        // Thread-local storage to avoid race conditions
        VectorXd Vdmu_private = VectorXd::Zero(_Vdmu.size());
        SpMat Vddmu_private = SpMat(_Vddmu.rows(), _Vddmu.cols());

        #pragma omp for nowait // Nowait allows threads to continue without waiting at the end of the loop
        for (auto &opt_k : Base::_vec_factors) {
            opt_k->calculate_partial_V();
            Vdmu_private += opt_k->local2joint_dmu_insertion();
            Vddmu_private += opt_k->local2joint_dprecision_insertion();
        }

        #pragma omp critical
        {
            Vdmu_sum += Vdmu_private;
            Vddmu_sum += Vddmu_private;
        }
    }

    // Update the member variables 
    _Vdmu = Vdmu_sum;
    _Vddmu = Vddmu_sum;

    SpMat dprecision = _Vddmu - Base::_precision;

    Eigen::ConjugateGradient<SpMat, Eigen::Upper> solver;
    VectorXd dmu = solver.compute(_Vddmu).solve(-_Vdmu);

    return std::make_tuple(dmu, dprecision);
}


/**
 * @brief One step of optimization.
 */
template <typename Factor>
std::tuple<VectorXd, SpMat> ProxKLGH<Factor>::compute_gradients_KL(std::optional<double>step_size){
    _Vdmu.setZero();
    _Vddmu.setZero();

    VectorXd Vdmu_sum = VectorXd::Zero(_Vdmu.size());
    SpMat Vddmu_sum = SpMat(_Vddmu.rows(), _Vddmu.cols());

    /**
     * @brief OMP parallel on cpu.
     */
    omp_set_num_threads(20); 

    #pragma omp parallel
    {
        // Thread-local storage to avoid race conditions
        VectorXd Vdmu_private = VectorXd::Zero(_Vdmu.size());
        SpMat Vddmu_private = SpMat(_Vddmu.rows(), _Vddmu.cols());

        #pragma omp for nowait // Nowait allows threads to continue without waiting at the end of the loop
        for (auto &opt_k : Base::_vec_factors) {
            opt_k->calculate_partial_V();
            Vdmu_private += opt_k->local2joint_dmu_insertion();
            Vddmu_private += opt_k->local2joint_dprecision_insertion();
        }

        #pragma omp critical
        {
            Vdmu_sum += Vdmu_private;
            Vddmu_sum += Vddmu_private;
        }
    }

    // Update the member variables 
    _Vdmu = Vdmu_sum;
    _Vddmu = Vddmu_sum;

    return std::make_tuple(_Vdmu, _Vddmu);
}


template <typename Factor>
void ProxKLGH<Factor>::optimize(std::optional<bool> verbose)
{
    // default verbose
    bool is_verbose = verbose.value_or(true);
    bool is_lowtemp = true;
    bool converged = false;
    
    for (int i_iter = 0; i_iter < Base::_niters; i_iter++)
    {   

        if (converged){
            break;
        }

        // ============= High temperature phase =============
        if (i_iter == Base::_niters_lowtemp && is_lowtemp){
            if (is_verbose){
                std::cout << "Switching to high temperature.." << std::endl;
            }
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

        Base::_res_recorder.update_data(this->_mu, this->_covariance, this->_precision, cost_iter, fact_costs_iter);

        // gradients
        // here change to update proposal
        std::tuple<VectorXd, SpMat> gradients = compute_gradients_KL();

        VectorXd dmu = std::get<0>(gradients);
        SpMat dprecision = std::get<1>(gradients);
        
        int cnt = 0;
        int B = 1;
        double step_size = Base::_step_size_base;

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

            if (cnt > Base::_niters_backtrack)
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
    Base::save_data(is_verbose);

}


template <typename Factor>
inline void ProxKLGH<Factor>::update_proposal(const VectorXd& new_mu, const SpMat& new_precision)
{
    Base::set_mu(new_mu);
    Base::set_precision(new_precision);
}

/**
 * @brief Compute the costs of all factors, using current values.
 */
template <typename Factor>
VectorXd ProxKLGH<Factor>::factor_cost_vector()
{   
    return Base::factor_cost_vector(this->_mu, this->_precision);
}

/**
 * @brief Compute the total cost function value given a state, using current values.
 */
template <typename Factor>
double ProxKLGH<Factor>::cost_value()
{
    return Base::cost_value(this->_mu, this->_precision);
}

template <typename Factor>
double ProxKLGH<Factor>::cost_value_no_entropy()
{
    
    SpMat Cov = this->inverse(this->_precision);
    
    double value = 0.0;
    for (auto &opt_k : this->_vec_factors)
    {
        value += opt_k->fact_cost_value(this->_mu, Cov);
    }
    return value; // / _temperature;
}

}


#endif // ProxKL_GH_IMPL_H
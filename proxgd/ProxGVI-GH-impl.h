
#pragma once

#ifndef PROXGVI_GH_IMPL_H
#define PROXGVI_GH_IMPL_H

using namespace Eigen;
#include <stdexcept>
#include <optional>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace gvi{

template <typename Factor>
void ProxGVIGH<Factor>:: compute_marginal_gradients(){
    for (auto &opt_k : Base::_vec_factors)
    {
        opt_k->compute_BW_grads();
    }
}

template <typename Factor>
std::tuple<double, VectorXd, SpMat> ProxGVIGH<Factor>::onestep_linesearch_prox(const double &step_size)
{

    SpMat new_precision(this->_dim, this->_dim), dprecision(this->_dim, this->_dim); 
    VectorXd new_mu(this->_dim), dmu(this->_dim); 
    new_mu.setZero(); new_precision.setZero();
    dmu.setZero(); dprecision.setZero();

    for (auto &opt_k : Base::_vec_factors)
    {
        std::tuple<Eigen::VectorXd, Eigen::MatrixXd> dmu_dprecision_local;
        dmu_dprecision_local = opt_k->compute_gradients_linesearch(step_size);
    
        dmu = dmu + opt_k->local2joint_dmu(std::get<0>(dmu_dprecision_local));
        dprecision = dprecision + opt_k->local2joint_dprecision(std::get<1>(dmu_dprecision_local));
    }

    // update mu and precision matrix
    new_mu = this->_mu + step_size * dmu;
    new_precision = this->_precision + step_size * dprecision;

    // new cost
    double new_cost = Base::cost_value(new_mu, new_precision);
    return std::make_tuple(new_cost, new_mu, new_precision);

}

/**
 * @brief optimize with backtracking
 */ 
template <typename Factor>
void ProxGVIGH<Factor>::optimize(std::optional<bool> verbose)
{
    // default verbose
    bool is_verbose = verbose.value_or(true);
    
    for (int i_iter = 0; i_iter < Base::_niters; i_iter++)
    {
        // ============= High temperature phase =============
        if (i_iter == Base::_niters_lowtemp){
            Base::switch_to_high_temperature();
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
        
        int cnt = 0;
        int B = 1;
        double step_size = 0.0;

        // compute base gradients at marginal levels
        this->compute_marginal_gradients();

        // backtracking 
        while (true)
        {   
            // new step size
            step_size = pow(Base::_step_size_base, B);

            auto onestep_res = this->onestep_linesearch_prox(step_size);

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
                this->update_proposal(new_mu, new_precision);
                break;
            }                
        }
    }

    std::cout << "=========== Saving Data ===========" << std::endl;
    Base::save_data(is_verbose);

}


template <typename Factor>
inline void ProxGVIGH<Factor>::update_proposal(const VectorXd& new_mu, const SpMat& new_precision)
{
    Base::set_mu(new_mu);
    Base::set_precision(new_precision);
}


/**
 * @brief Compute the costs of all factors, using current values.
 */
template <typename Factor>
VectorXd ProxGVIGH<Factor>::factor_cost_vector()
{   
    return Base::factor_cost_vector(this->_mu, this->_precision);
}

/**
 * @brief Compute the total cost function value given a state, using current values.
 */
template <typename Factor>
double ProxGVIGH<Factor>::cost_value()
{
    return Base::cost_value(this->_mu, this->_precision);
}

template <typename Factor>
double ProxGVIGH<Factor>::cost_value_no_entropy()
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


#endif // ProxGVI_GH_IMPL_H

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
std::tuple<double, VectorXd, SpMat> ProxGVIGH<Factor>::onestep_linesearch(const double &step_size, 
                                                                            const VectorXd& dmu, 
                                                                            const SpMat& dprecision)
{

    SpMat new_precision; 
    VectorXd new_mu; 
    new_mu.setZero(); new_precision.setZero();

    // update mu and precision matrix
    new_mu = this->_mu + step_size * dmu;
    new_precision = this->_precision + step_size * dprecision;

    // new cost
    double new_cost = Base::cost_value(new_mu, new_precision);
    return std::make_tuple(new_cost, new_mu, new_precision);

}

template <typename Factor>
std::tuple<VectorXd, SpMat> ProxGVIGH<Factor>::compute_gradients(std::optional<double>step_size)
{
    _dmu.setZero();
    _dprecision.setZero();

    for (auto &opt_k : Base::_vec_factors)
    {
        opt_k->calculate_partial_V(step_size);
        _dmu = _dmu + opt_k->local2joint_dmu();
        _dprecision = _dprecision + opt_k->local2joint_dprecision();

        // // For comparison
        // std::tuple<Eigen::VectorXd, Eigen::MatrixXd> dmu_dprecision_local;
        // dmu_dprecision_local = opt_k->compute_gradients_linesearch(this->_step_size_base);

        // MatrixXd dprecision_k;
        // VectorXd dmu_k;
        // dmu_k.setZero(); dprecision_k.setZero();
        // dmu_k = std::get<0>(dmu_dprecision_local);
        // dprecision_k = std::get<1>(dmu_dprecision_local);

        // VectorXd dmu_k_joint = opt_k->local2joint_dmu();
        // VectorXd dmu_k_joint_1 = opt_k->local2joint_dmu(dmu_k);

        // MatrixXd dprecision_k_joint = opt_k->local2joint_dprecision();
        // MatrixXd dprecision_k_joint_1 = opt_k->local2joint_dprecision(dprecision_k);

        // VectorXd diff_dmu = dmu_k_joint - dmu_k_joint_1;
        // MatrixXd diff_dprecision = dprecision_k_joint - dprecision_k_joint_1;

        // std::cout << "diff_dmu " << std::endl << diff_dmu.norm() << std::endl;
        // std::cout << "diff_dprecision " << std::endl << diff_dprecision.norm() << std::endl;

    }

    std::cout << "_dprecision " << std::endl << _dprecision << std::endl;
    bool is_dprecision_spd = isSymmetricPositiveDefinite(_dprecision);
    std::cout << "is_dprecision_spd: " << is_dprecision_spd << std::endl;

    return std::make_tuple(_dmu, _dprecision);
}

// template <typename Factor>
// // std::tuple<double, VectorXd, SpMat> ProxGVIGH<Factor>::onestep_linesearch_prox(const double &step_size)
// std::tuple<VectorXd, SpMat> ProxGVIGH<Factor>::onestep_linesearch_prox(const double &step_size)
// {
    
//     SpMat dprecision(this->_dim, this->_dim); 
//     VectorXd dmu(this->_dim); 
//     dmu.setZero(); dprecision.setZero();

//     for (auto &opt_k : Base::_vec_factors)
//     {
//         std::tuple<Eigen::VectorXd, Eigen::MatrixXd> dmu_dprecision_local;
//         dmu_dprecision_local = opt_k->compute_gradients_linesearch(0.9);

//         MatrixXd dprecision_k;
//         VectorXd dmu_k;
//         dmu_k.setZero(); dprecision_k.setZero();
//         dmu_k = std::get<0>(dmu_dprecision_local);
//         dprecision_k = std::get<1>(dmu_dprecision_local);

//         Eigen::MatrixXd dprecision_k_full{dprecision_k};
//         std::cout << "dprecision_k_full " << std::endl << dprecision_k_full << std::endl; 

//         dmu = dmu + opt_k->local2joint_dmu(dmu_k);
//         dprecision = dprecision + opt_k->local2joint_dprecision(dprecision_k);

//     }

    // Eigen::MatrixXd dprecision_full{dprecision};
    // std::cout << "dprecision_full " << std::endl << dprecision_full << std::endl;

    // return this->onestep_linesearch(step_size, dmu, dprecision);
//     return std::make_tuple(dmu, dprecision);

// }

template <typename Factor>
void ProxGVIGH<Factor>::optimize(std::optional<bool> verbose){
    // default verbose
    bool is_verbose = verbose.value_or(true);
    
    for (int i_iter = 0; i_iter < Base::_niters; i_iter++)
    {
        // ============= High temperature phase =============
        if (i_iter == Base::_niters_lowtemp){
            Base::switch_to_high_temperature();
        }

        // ============= Cost at current iteration =============
        double cost_iter = cost_value();

        if (is_verbose){
            std::cout << "========= iteration " << i_iter << " ========= " << std::endl;
            std::cout << "--- cost_iter ---" << std::endl << cost_iter << std::endl;
        }

        // ============= Collect factor costs =============
        VectorXd fact_costs_iter = factor_cost_vector();

        Base::_res_recorder.update_data(this->_mu, this->_covariance, this->_precision, cost_iter, fact_costs_iter);

        int cnt = 0;
        int B = 1;
        double step_size = pow(this->_step_size_base, B);

        // gradients
        std::tuple<VectorXd, SpMat> gradients = compute_gradients(step_size);

        VectorXd dmu = std::get<0>(gradients);
        SpMat dprecision = std::get<1>(gradients);

        // backtracking 
        while (true)
        {   
            // new step size
            step_size = pow(this->_step_size_base, B);

            auto onestep_res = onestep_linesearch(step_size, dmu, dprecision);
            // auto onestep_res_1 = onestep_linesearch_prox(step_size);

            // VectorXd dmu_1 = std::get<0>(onestep_res_1);
            // SpMat dprecision_1 = std::get<1>(onestep_res_1);

            // VectorXd diff_dmu = dmu - dmu_1;
            // MatrixXd diff_dprecision = dprecision - dprecision_1;

            // std::cout << "diff_dmu " << std::endl << diff_dmu.norm() << std::endl;
            // std::cout << "diff_dprecision " << std::endl << diff_dprecision.norm() << std::endl;

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
                update_proposal(new_mu, new_precision);
                break;
            }                
        }
    }

    std::cout << "=========== Saving Data ===========" << std::endl;
    Base::save_data(is_verbose);
}



// /**
//  * @brief optimize with backtracking
//  */ 
// template <typename Factor>
// void ProxGVIGH<Factor>::optimize(std::optional<bool> verbose)
// {
//     // default verbose
//     bool is_verbose = verbose.value_or(true);
    
//     for (int i_iter = 0; i_iter < Base::_niters; i_iter++)
//     {
//         // ============= High temperature phase =============
//         if (i_iter == Base::_niters_lowtemp){
//             Base::switch_to_high_temperature();
//         }

//         // ============= Cost at current iteration =============
//         double cost_iter = this->cost_value();

//         if (is_verbose){
//             std::cout << "========= iteration " << i_iter << " ========= " << std::endl;
//             std::cout << "--- cost_iter ---" << std::endl << cost_iter << std::endl;
//         }

//         // ============= Collect factor costs =============
//         VectorXd fact_costs_iter = this->factor_cost_vector();

//         Base::_res_recorder.update_data(this->_mu, this->_covariance, this->_precision, cost_iter, fact_costs_iter);
        
//         int cnt = 0;
//         int B = 1;
//         double step_size = 0.0;

//         // compute base gradients at marginal levels
//         this->compute_marginal_gradients();

//         // backtracking 
//         while (true)
//         {   
//             // new step size
//             step_size = pow(Base::_step_size_base, B);

//             auto onestep_res = this->onestep_linesearch_prox(step_size);

//             double new_cost = std::get<0>(onestep_res);
//             VectorXd new_mu = std::get<1>(onestep_res);
//             auto new_precision = std::get<2>(onestep_res);

//             // accept new cost and update mu and precision matrix
//             if (new_cost < cost_iter){
//                 // update mean and covariance
//                 this->update_proposal(new_mu, new_precision);
//                 break;
//             }else{ 
//                 // shrinking the step size
//                 B += 1;
//                 cnt += 1;
//             }

//             if (cnt > Base::_niters_backtrack)
//             {
//                 if (is_verbose){
//                     std::cout << "Reached the maximum backtracking steps." << std::endl;
//                 }
//                 this->update_proposal(new_mu, new_precision);
//                 break;
//             }                
//         }
//     }

//     std::cout << "=========== Saving Data ===========" << std::endl;
//     Base::save_data(is_verbose);

// }


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
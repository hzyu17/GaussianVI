
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

        // _dmu = _dmu + opt_k->local2joint_dmu();
        // _dprecision = _dprecision + opt_k->local2joint_dprecision();

        this->_Vdmu = this->_Vdmu + opt_k->local2joint_dmu();
        this->_Vddmu = this->_Vddmu + opt_k->local2joint_dprecision();

    }

    this->_bk = this->_Vdmu;
    this->_Sk = this->_Vddmu;

    Eigen::VectorXd muk_half = this->_mu - this->_step_size_base * this->_bk;

    Eigen::MatrixXd precision_full(this->_dim, this->_dim);
    precision_full.setZero();
    precision_full = this->_precision;

    Eigen::MatrixXd Sigk(this->_dim, this->_dim);
    Sigk.setZero();
    Sigk = precision_full.inverse();

    // Compute the BW gradient step
    Eigen::MatrixXd Mk(this->_dim, this->_dim);
    Mk.setZero();

    Eigen::MatrixXd Identity(this->_dim, this->_dim);
    Identity = Eigen::MatrixXd::Identity(this->_dim, this->_dim);    
    
    Mk = Identity - this->_step_size_base*this->_Sk;

    Eigen::MatrixXd Sigk_half(this->_dim, this->_dim);
    Sigk_half.setZero();
    Sigk_half = Mk*Sigk*Mk.transpose();

    SpMat precision_half = Sigk_half.inverse().sparseView();

    // Compute the proximal step
    Eigen::MatrixXd temp(this->_dim, this->_dim);
    temp.setZero();
    temp = gvi::sqrtm(Sigk_half*(Sigk_half + 4.0*this->_step_size_base*Identity));

    // Eigen::MatrixXd temp_2(this->_dim, this->_dim);
    // temp_2.setZero();
    // temp_2 = gvi::sqrtm(temp);

    Eigen::VectorXd mk_new = this->_mu - this->_step_size_base * _bk;
    Eigen::MatrixXd Sigk_new = 0.5*Sigk_half + this->_step_size_base*Identity + 0.5*temp;

    SpMat precision_new = Sigk_new.inverse().sparseView();

    return std::make_tuple(muk_half, precision_new);

    // std::cout << "_dprecision " << std::endl << _dprecision << std::endl;
    // bool is_dprecision_spd = isSymmetricPositiveDefinite(_dprecision);
    // std::cout << "is_dprecision_spd: " << is_dprecision_spd << std::endl;

    // return std::make_tuple(_dmu, _dprecision);
}


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
        std::tuple<VectorXd, SpMat> mukhalf_Precisionhalf = compute_gradients(step_size);

        VectorXd muk_half = std::get<0>(mukhalf_Precisionhalf);
        SpMat new_precision = std::get<1>(mukhalf_Precisionhalf);

        update_proposal(muk_half, new_precision);


        // update_proposal(new_mu, new_precision);

        // // backtracking 
        // while (true)
        // {   
        //     // new step size
        //     step_size = pow(this->_step_size_base, B);

        //     // VectorXd dmu_zero(this->_dim);
        //     // SpMat dprecision_zero(this->_dim, this->_dim);

        //     // dmu_zero.setZero();
        //     // dprecision_zero.setZero();

        //     auto onestep_res = onestep_linesearch(step_size, dmu, dprecision);
        //     // auto onestep_res_1 = onestep_linesearch_prox(step_size);

        //     // VectorXd dmu_1 = std::get<0>(onestep_res_1);
        //     // SpMat dprecision_1 = std::get<1>(onestep_res_1);

        //     // VectorXd diff_dmu = dmu - dmu_1;
        //     // MatrixXd diff_dprecision = dprecision - dprecision_1;

        //     // std::cout << "diff_dmu " << std::endl << diff_dmu.norm() << std::endl;
        //     // std::cout << "diff_dprecision " << std::endl << diff_dprecision.norm() << std::endl;

        //     double new_cost = std::get<0>(onestep_res);
        //     VectorXd new_mu = std::get<1>(onestep_res);
        //     auto new_precision = std::get<2>(onestep_res);

        //     // accept new cost and update mu and precision matrix
        //     if (new_cost < cost_iter){
        //         // update mean and covariance
        //         this->update_proposal(new_mu, new_precision);
        //         break;
        //     }else{ 
        //         // shrinking the step size
        //         B += 1;
        //         cnt += 1;
        //     }

        //     if (cnt > Base::_niters_backtrack)
        //     {
        //         if (is_verbose){
        //             std::cout << "Reached the maximum backtracking steps." << std::endl;
        //         }
        //         update_proposal(new_mu, new_precision);
        //         break;
        //     }                
        // }
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

#pragma once

#ifndef GVI_GH_IMPL_H
#define GVI_GH_IMPL_H

using namespace Eigen;

#include <stdexcept>
#include <optional>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace gvi{

template <typename Factor>
void GVIGH<Factor>::switch_to_high_temperature(){
    for (auto& i_factor:_vec_factors){
        i_factor->switch_to_high_temperature();
    }
    this->initilize_precision_matrix();
}

/**
 * @brief optimize with backtracking
 */ 
template <typename Factor>
void GVIGH<Factor>::optimize(std::optional<bool> verbose)
{
    // default verbose
    bool is_verbose = verbose.value_or(true);
    
    for (int i_iter = 0; i_iter < _niters; i_iter++)
    {
        // ============= High temperature phase =============
        if (i_iter == _niters_lowtemp){
            switch_to_high_temperature();
        }

        // ============= Cost at current iteration =============
        double cost_iter = cost_value();

        if (is_verbose){
            std::cout << "========= iteration " << i_iter << " ========= " << std::endl;
            std::cout << "--- cost_iter ---" << std::endl << cost_iter << std::endl;
        }

        // ============= Collect factor costs =============
        VectorXd fact_costs_iter = factor_cost_vector();

        _res_recorder.update_data(_mu, _covariance, _precision, cost_iter, fact_costs_iter);

        // gradients
        std::tuple<VectorXd, SpMat> gradients = compute_gradients();

        VectorXd dmu = std::get<0>(gradients);
        SpMat dprecision = std::get<1>(gradients);
        
        int cnt = 0;
        int B = 1;
        double step_size = 0.0;

        // backtracking 
        while (true)
        {   
            // new step size
            step_size = pow(_step_size_base, B);

            auto onestep_res = onestep_linesearch(step_size, dmu, dprecision);

            double new_cost = std::get<0>(onestep_res);
            VectorXd new_mu = std::get<1>(onestep_res);
            auto new_precision = std::get<2>(onestep_res);

            // accept new cost and update mu and precision matrix
            if (new_cost < cost_iter){
                /// update mean and covariance
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
                    std::cout << "Too many iterations in the backtracking ... Dead" << std::endl;
                }
                update_proposal(new_mu, new_precision);
                break;
            }                
        }
    }

    save_data(is_verbose);

}

template <typename Factor>
inline void GVIGH<Factor>::set_precision(const SpMat &new_precision)
{
    _precision = new_precision;
    // sparse inverse
    inverse_inplace();

    for (auto &factor : _vec_factors)
    {
        factor->update_precision_from_joint(_covariance);
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
    SpMat joint_cov = inverse(joint_precision);
    for (auto &opt_k : _vec_factors)
    {
        fac_costs(cnt) = opt_k->fact_cost_value(fill_joint_mean, joint_cov); // / _temperature;
        cnt += 1;
    }
    return fac_costs;
}

/**
 * @brief Compute the total cost function value given a state.
 */
template <typename Factor>
double GVIGH<Factor>::cost_value(const VectorXd &mean, SpMat &Precision)
{

    SpMat Cov = inverse(Precision);

    double value = 0.0;
    for (auto &opt_k : _vec_factors)
    {
        value += opt_k->fact_cost_value(mean, Cov); // / _temperature;
    }

    SparseLDLT ldlt(Precision);
    VectorXd vec_D = ldlt.vectorD();

    double logdet = 0;
    for (int i_diag=0; i_diag<vec_D.size(); i_diag++){
        logdet += log(vec_D(i_diag));
    }

    return value + logdet / 2;
}

}


#endif // GVI_GH_IMPL_H
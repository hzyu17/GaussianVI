/**
 * @file NGDFactorizedNonlinerGH.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief factorized optimizer which only takes one cost class. (templated)
 * @version 0.1
 * @date 2022-08-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once 

#ifndef NGDFactorizedNonlinerGH_H
#define NGDFactorizedNonlinerGH_H

#include "ngd/NGDFactorizedBase.h"

namespace gvi{
template <typename CostClass>
class NGDFactorizedNonlinerGH : public NGDFactorizedBase{
    using Base = NGDFactorizedBase;
    using GHFunction = std::function<MatrixXd(const VectorXd&)>;
    using CostFunction = std::function<double(const VectorXd&, const CostClass&)>;
    public:
        NGDFactorizedNonlinerGH(int dimension,
                                int dim_state,
                                const CostFunction& function, 
                                const CostClass& cost_class,
                                int num_states,
                                int start_indx,
                                double temperature, 
                                double high_temperature):
            Base(dimension, dim_state, num_states, start_indx, temperature, high_temperature){

            Base::_func_phi = [this, function, cost_class, temperature](const VectorXd& x){return MatrixXd{MatrixXd::Constant(1, 1, function(x, cost_class) / this->temperature())};};
            Base::_func_Vmu = [this, function, cost_class, temperature](const VectorXd& x){return (x-Base::_mu) * function(x, cost_class) / this->temperature() ;};
            Base::_func_Vmumu = [this, function, cost_class, temperature](const VectorXd& x){return MatrixXd{(x-Base::_mu) * (x-Base::_mu).transpose().eval() * function(x, cost_class) / this->temperature()};};
            
            using GH = SparseGaussHermite<GHFunction>;
            Base::_gh = std::make_shared<GH>(GH{6, dimension, Base::_mu, Base::_covariance});
            
        }

    double fact_cost_value(const VectorXd& fill_joint_mean, const SpMat& joint_cov) override{
        VectorXd mean_k = extract_mu_from_joint(fill_joint_mean);
        MatrixXd Cov_k = extract_cov_from_joint(joint_cov);

        updateGH(mean_k, Cov_k);

        return this->_gh->Integrate(this->_func_phi)(0, 0);
    }
};

} //namespace
#endif //NGDFactorizedNonlinerGH_H
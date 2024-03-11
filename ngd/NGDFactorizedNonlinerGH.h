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

#include "gvibase/GVIFactorizedNonLinearBase.h"

namespace gvi{
template <typename CostClass>
class NGDFactorizedNonlinerGH : public GVIFactorizedNonLinearBase{
    using Base = GVIFactorizedNonLinearBase;
    using GHFunction = std::function<MatrixXd(const VectorXd&)>;
    using CostFunction = std::function<double(const VectorXd&, const CostClass&)>;
    public:
        NGDFactorizedNonlinerGH(int dimension,
                                int dim_state,
                                int gh_degree,
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

            Base::_gh = std::make_shared<GH>(GH{gh_degree, dimension, Base::_mu, Base::_covariance});
            
        }

    /**
     * @brief Calculating phi * (partial V) / (partial mu), and 
     * phi * (partial V^2) / (partial mu * partial mu^T)
     */

    void calculate_partial_V() override{
        // update the mu and sigma inside the gauss-hermite integrator
        updateGH(this->_mu, this->_covariance);

        this->_Vdmu.setZero();
        this->_Vddmu.setZero();

        /// Integrate for E_q{_Vdmu} 
        this->_Vdmu = this->_precision * this->_gh->Integrate(this->_func_Vmu);
        // this->_Vdmu = this->_precision * this->_Vdmu;

        /// Integrate for E_q{phi(x)}
        _E_Phi = this->_gh->Integrate(this->_func_phi)(0, 0);
        
        /// Integrate for partial V^2 / ddmu_ 
        MatrixXd E_xxphi{this->_gh->Integrate(this->_func_Vmumu)};

        this->_Vddmu.triangularView<Upper>() = (this->_precision * E_xxphi * this->_precision - this->_precision * _E_Phi).triangularView<Upper>();
        this->_Vddmu.triangularView<StrictlyLower>() = this->_Vddmu.triangularView<StrictlyUpper>().transpose();

    }

    double fact_cost_value(const VectorXd& fill_joint_mean, const SpMat& joint_cov) override{
        if (_E_Phi == 0){
            VectorXd mean_k = extract_mu_from_joint(fill_joint_mean);
            MatrixXd Cov_k = extract_cov_from_joint(joint_cov);

            updateGH(mean_k, Cov_k);
            _E_Phi = this->_gh->Integrate(this->_func_phi)(0, 0);
        }
            
        return _E_Phi;
    }
};

} //namespace
#endif //NGDFactorizedNonlinerGH_H
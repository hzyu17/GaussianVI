/**
 * @file NGDFactorizedLinearGH.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief Factorized optimization steps for linear gaussian factors 
 * -log(p(x|z)) = (1/2)*||\Lambda X - \Psi \mu_t||_{\Sigma_t^{-1}},
 *  using GH quadrotors for comparison.
 * @version 0.1
 * @date 2023-01-17
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#ifndef NGDFactorizedLinearGH_H
#define NGDFactorizedLinearGH_H

#include "gvibase/GVIFactorizedBaseGH.h"

namespace gvi{
template <typename Factor = NoneType>
class NGDFactorizedLinearGH : public GVIFactorizedBaseGH{
    using Base = GVIFactorizedBaseGH;
    using CostFunction = std::function<double(const VectorXd&, const Factor&)>;
public:
    NGDFactorizedLinearGH(const int& dimension,
                            int dim_state, 
                            int gh_degree,
                            const CostFunction& function, 
                            const Factor& linear_factor,
                            int num_states,
                            int start_indx,
                            double temperature,
                            double high_temperature,
                            std::optional<std::shared_ptr<QuadratureWeightsMap>>  weight_sigpts_map_option=std::nullopt):
        Base(dimension, dim_state, num_states, start_indx, 
                        temperature, high_temperature, weight_sigpts_map_option),
        _linear_factor{linear_factor}
        {
            Base::_func_phi = [this, function, linear_factor](const VectorXd& x){return MatrixXd::Constant(1, 1, function(x, linear_factor));};

            /// Override of the Base classes.
            Base::_func_phi = [this, function, linear_factor](const VectorXd& x){return MatrixXd::Constant(1, 1, function(x, linear_factor));};
            Base::_func_Vmu = [this, function, linear_factor](const VectorXd& x){return (x-Base::_mu) * function(x, linear_factor);};
            Base::_func_Vmumu = [this, function, linear_factor](const VectorXd& x){return MatrixXd{(x-Base::_mu) * (x-Base::_mu).transpose().eval() * function(x, linear_factor)};};
            Base::_gh = std::make_shared<GH>(GH{gh_degree, Base::_dim, Base::_mu, Base::_covariance, weight_sigpts_map_option});

            _target_mean = linear_factor.get_mu();
            _target_precision = linear_factor.get_precision();
            _Lambda = linear_factor.get_Lambda();
            _Psi = linear_factor.get_Psi();
            _constant = linear_factor.get_Constant();
        }

protected:
    Factor _linear_factor;

    MatrixXd _target_mean, _target_precision, _Lambda, _Psi;

    double _constant;

public:
    double constant() const { return _constant; }

    /*Calculating phi * (partial V) / (partial mu), and 
        * phi * (partial V^2) / (partial mu * partial mu^T) for Gaussian posterior: closed-form expression:
        * (partial V) / (partial mu) = Sigma_t{-1} * (mu_k - mu_t)
        * (partial V^2) / (partial mu)(partial mu^T): higher order moments of a Gaussian.
    */

    inline VectorXd local2joint_dmu() override{ 
        VectorXd res(this->_joint_size);
        res.setZero();
        this->_block.fill_vector(res, this->_Vdmu);
        return res;
    }

    inline SpMat local2joint_dprecision() override{ 
        SpMat res(this->_joint_size, this->_joint_size);
        res.setZero();
        this->_block.fill(this->_Vddmu, res);
        return res;
    }

    void calculate_partial_V(std::optional<double> step_size=std::nullopt) override{
        // update the mu and sigma inside the gauss-hermite integrator
        updateGH(this->_mu, this->_covariance);

        this->_Vdmu.setZero();
        this->_Vddmu.setZero();

        /// Integrate for E_q{_Vdmu} 
        this->_Vdmu = this->_gh->Integrate(this->_func_Vmu);
        this->_Vdmu = this->_precision * this->_Vdmu;
        this->_Vdmu = this->_Vdmu / this->temperature();

        /// Integrate for E_q{phi(x)}
        double E_phi = this->_gh->Integrate(this->_func_phi)(0, 0);
        
        /// Integrate for partial V^2 / ddmu_ 
        MatrixXd E_xxphi{this->_gh->Integrate(this->_func_Vmumu)};

        this->_Vddmu.triangularView<Upper>() = (this->_precision * E_xxphi * this->_precision - this->_precision * E_phi).triangularView<Upper>();
        this->_Vddmu.triangularView<StrictlyLower>() = this->_Vddmu.triangularView<StrictlyUpper>().transpose();
        this->_Vddmu = this->_Vddmu / this->temperature();
    }

    double fact_cost_value(const VectorXd& fill_joint_mean, const SpMat& joint_cov) override {
        VectorXd mean_k = extract_mu_from_joint(fill_joint_mean);
        MatrixXd Cov_k = extract_cov_from_joint(joint_cov);

        updateGH(mean_k, Cov_k);

        return this->_gh->Integrate(this->_func_phi)(0, 0) / this->temperature();
    }

};

} //namespace
#endif //NGDFactorizedLinearGH_H
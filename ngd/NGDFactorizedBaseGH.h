/**
 * @file NGDFactorizedBaseGH.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief The marginal optimizer class expecting two functions, one is the cost function, f1(x, cost1); 
 * the other is the function for GH expectation, f2(x).
 * @version 0.1
 * @date 2022-03-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#ifndef NGDFactorizedBaseGH_H
#define NGDFactorizedBaseGH_H

// #include "ngd/NGDFactorizedBase.h"
#include "gvibase/GVIFactorizedBaseGH.h"
#include <memory>

using namespace Eigen;

namespace gvi{

template <typename CostClass = NoneType>
class NGDFactorizedBaseGH: public GVIFactorizedBaseGH{

    using GVIBase = GVIFactorizedBaseGH;
    using Function = std::function<double(const VectorXd&, const CostClass &)>;

public:
    ///@param dimension The dimension of the state
    ///@param function Template function class which calculate the cost
    // NGDFactorizedBaseGH(const int& dimension, const Function& function, const CostClass& cost_class_, const MatrixXd& Pk_):
    
    NGDFactorizedBaseGH(int dimension, int state_dim, int gh_degree, 
                        const Function& function, const CostClass& cost_class,
                        int num_states, int start_index, 
                        double temperature=1.0, double high_temperature=10.0,
                        std::optional<QuadratureWeightsMap> weight_sigpts_map_option=std::nullopt):
                GVIBase(dimension, state_dim, num_states, start_index, 
                        temperature, high_temperature, weight_sigpts_map_option)
            {
                /// Override of the GVIBase classes.
                GVIBase::_func_phi = [this, function, cost_class](const VectorXd& x){return MatrixXd::Constant(1, 1, function(x, cost_class));};
                GVIBase::_func_Vmu = [this, function, cost_class](const VectorXd& x){return (x-GVIBase::_mu) * function(x, cost_class);};
                GVIBase::_func_Vmumu = [this, function, cost_class](const VectorXd& x){return MatrixXd{(x-GVIBase::_mu) * (x-GVIBase::_mu).transpose().eval() * function(x, cost_class)};};
                GVIBase::_gh = std::make_shared<GH>(GH{gh_degree, GVIBase::_dim, GVIBase::_mu, GVIBase::_covariance, weight_sigpts_map_option});
            }
public:

void calculate_partial_V(std::optional<double> step_size=std::nullopt) override{
        // update the mu and sigma inside the gauss-hermite integrator
        updateGH(this->_mu, this->_covariance);

        this->_Vdmu.setZero();
        this->_Vddmu.setZero();

        /// Integrate for E_q{_Vdmu} 
        this->_Vdmu = std::move(this->_gh->Integrate(this->_func_Vmu));
        this->_Vdmu = std::move(this->_precision * this->_Vdmu);
        this->_Vdmu = std::move(this->_Vdmu / this->temperature());

        /// Integrate for E_q{phi(x)}
        double E_phi = std::move(this->_gh->Integrate(this->_func_phi)(0, 0));
        
        /// Integrate for partial V^2 / ddmu_ 
        MatrixXd E_xxphi = std::move(this->_gh->Integrate(this->_func_Vmumu));

        this->_Vddmu.triangularView<Upper>() = std::move((this->_precision * E_xxphi * this->_precision - this->_precision * E_phi).triangularView<Upper>());
        this->_Vddmu.triangularView<StrictlyLower>() = std::move(this->_Vddmu.triangularView<StrictlyUpper>().transpose());
        this->_Vddmu = std::move(this->_Vddmu / this->temperature());
    }
    

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


    /**
     * @brief returns the (x-mu)*Phi(x) 
     */
    inline MatrixXd xMu_negative_log_probability(const VectorXd& x) const{
        return _func_Vmu(x);
    }

    /**
     * @brief returns the (x-mu)(x-mu)^T*Phi(x) 
     */
    inline MatrixXd xMuxMuT_negative_log_probability(const VectorXd& x) const{
        return _func_Vmumu(x);
    }

    double fact_cost_value(const VectorXd& fill_joint_mean, const SpMat& joint_cov) override {
        VectorXd mean_k = extract_mu_from_joint(fill_joint_mean);
        MatrixXd Cov_k = extract_cov_from_joint(joint_cov);

        updateGH(mean_k, Cov_k);

        return this->_gh->Integrate(this->_func_phi)(0, 0) / this->temperature();
    }
    

};


}

#endif // NGDFactorizedBaseGH_H
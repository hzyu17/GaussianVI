/**
 * @file NGDFactorizedBase.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief The base class for Natural gradient descent factorized optimizer.
 * @version 0.1
 * @date 2022-03-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */

# pragma once

#include "gvibase/GVIFactorizedBase.h"

class NGDFactorizedBase : public GVIFactorizedBase{

public:

    NGDFactorizedBase(int dimension, int state_dim, int num_states, int start_index, 
                        double temperature=10.0, double high_temperature=100.0, bool is_linear=false):
            GVIFactorizedBase(dimension, state_dim, num_states, start_index),
            _Vdmu(_dim),
            _Vddmu(_dim, _dim)
            {}

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
        this->_Vdmu = this->_gh->Integrate(this->_func_Vmu);
        this->_Vdmu = this->_precision * this->_Vdmu;

        /// Integrate for E_q{phi(x)}
        double E_phi = this->_gh->Integrate(this->_func_phi)(0, 0);
        
        /// Integrate for partial V^2 / ddmu_ 
        MatrixXd E_xxphi{this->_gh->Integrate(this->_func_Vmumu)};

        this->_Vddmu.triangularView<Upper>() = (this->_precision * E_xxphi * this->_precision - this->_precision * E_phi).triangularView<Upper>();
        this->_Vddmu.triangularView<StrictlyLower>() = this->_Vddmu.triangularView<StrictlyUpper>().transpose();

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

    inline MatrixXd E_xMuPhi(){
        return _gh->Integrate(_func_Vmu);
    }

    inline MatrixXd E_xMuxMuTPhi(){
        return _gh->Integrate(_func_Vmumu);
    }

    double fact_cost_value(const VectorXd& fill_joint_mean, const SpMat& joint_cov) override {
        VectorXd mean_k = extract_mu_from_joint(fill_joint_mean);
        MatrixXd Cov_k = extract_cov_from_joint(joint_cov);

        updateGH(mean_k, Cov_k);

        return this->_gh->Integrate(this->_func_phi)(0, 0);
    }

protected:
    VectorXd _Vdmu;
    MatrixXd _Vddmu;

    GHFunction _func_Vmu;
    GHFunction _func_Vmumu;
};


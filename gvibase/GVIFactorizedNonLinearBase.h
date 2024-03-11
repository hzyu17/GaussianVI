/**
 * @file GVIFactorizedNonLinearBase.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief The base class for Natural gradient descent factorized optimizer for nonlinear \phi(x).
 * @version 0.1
 * @date 2022-03-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */

# pragma once

#ifndef GVIFactorizedNonLinearBase_H
#define GVIFactorizedNonLinearBase_H

#include "gvibase/GVIFactorizedBase.h"

namespace gvi{

class GVIFactorizedNonLinearBase : public GVIFactorizedBase{

public:

    GVIFactorizedNonLinearBase(int dimension, int state_dim, int num_states, int start_index, 
                        double temperature=10.0, double high_temperature=100.0):
            GVIFactorizedBase(dimension, state_dim, num_states, start_index)
            {}

    /// update the GH approximator
    void updateGH(const VectorXd& x, const MatrixXd& P){
        _gh->update_P(P); 
        _gh->update_mean(x);
        
    }


    /**
     * @brief returns the E_q{phi(x)} = E_q{-log(p(x,z))}
     */
    inline double E_Phi() {
        return _gh->Integrate(_func_phi)(0, 0);
    }

    inline MatrixXd E_xMuPhi(){
        return _gh->Integrate(_func_Vmu);
    }

    inline MatrixXd E_xMuxMuTPhi(){
        return _gh->Integrate(_func_Vmumu);
    }

    void set_GH_points(int p){
        _gh->set_polynomial_deg(p);
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

    // double fact_cost_value(const VectorXd& fill_joint_mean, const SpMat& joint_cov) override {
    //     VectorXd mean_k = extract_mu_from_joint(fill_joint_mean);
    //     MatrixXd Cov_k = extract_cov_from_joint(joint_cov);

    //     updateGH(mean_k, Cov_k);

    //     return this->_gh->Integrate(this->_func_phi)(0, 0);
    // }

public:
    /// Intermediate functions for Gauss-Hermite quadratures, default definition, needed to be overrided by the
    /// derived classes.

    GHFunction _func_Vmu;
    GHFunction _func_Vmumu;

    /// G-H quadrature class
    std::shared_ptr<GH> _gh;    

};

}


#endif // GVIFactorizedNonLinearBase_H
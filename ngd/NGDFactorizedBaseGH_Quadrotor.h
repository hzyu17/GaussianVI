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

#include "gvibase/GVIFactorizedBaseGH_Cuda.h"
#include <helpers/CudaOperation.h>
#include <memory>

using namespace Eigen;

namespace gvi{

class NGDFactorizedBaseGH_Quadrotor: public GVIFactorizedBaseGH_Cuda{

    using GVIBase = GVIFactorizedBaseGH_Cuda;

public:
    ///@param dimension The dimension of the state
    ///@param function Template function class which calculate the cost
    // NGDFactorizedBaseGH(const int& dimension, const Function& function, const CostClass& cost_class_, const MatrixXd& Pk_):
    
    NGDFactorizedBaseGH_Quadrotor(int dimension, int state_dim, int gh_degree, 
                        int num_states, int start_index, double cost_sigma, 
                        double epsilon, double radius, 
                        double temperature, double high_temperature,
                        QuadratureWeightsMap weight_sigpts_map_option, 
                        std::shared_ptr<CudaOperation_Quad> cuda_ptr):
                GVIBase(dimension, state_dim, num_states, start_index, 
                        temperature, high_temperature, weight_sigpts_map_option),
                _epsilon(epsilon),
                _sigma(cost_sigma),
                _radius(radius)
            {
                /// Override of the GVIBase classes. _func_phi-> Scalar, _func_Vmu -> Vector, _func_Vmumu -> Matrix
                GVIBase::_gh = std::make_shared<GH>(GH{gh_degree, GVIBase::_dim, GVIBase::_mu, GVIBase::_covariance, weight_sigpts_map_option});
                _cuda = std::make_shared<CudaOperation_Quad>(CudaOperation_Quad{cost_sigma, epsilon, radius});
                // _cuda = cuda_ptr;

            }
public:

void calculate_partial_V() override{
        // update the mu and sigma inside the gauss-hermite integrator
        updateGH(this->_mu, this->_covariance);

        this->_Vdmu.setZero();
        this->_Vddmu.setZero();

        /// Integrate for E_q{_Vdmu} 
        this->_Vdmu = Integrate_cuda(1);
        this->_Vdmu = this->_precision * this->_Vdmu;
        this->_Vdmu = this->_Vdmu / this->temperature();

        /// Integrate for E_q{phi(x)}
        double E_phi = Integrate_cuda(0)(0, 0);
        
        /// Integrate for partial V^2 / ddmu_ 
        MatrixXd E_xxphi{Integrate_cuda(2)};

        this->_Vddmu.triangularView<Upper>() = (this->_precision * E_xxphi * this->_precision - this->_precision * E_phi).triangularView<Upper>();
        this->_Vddmu.triangularView<StrictlyLower>() = this->_Vddmu.triangularView<StrictlyUpper>().transpose();
        this->_Vddmu = this->_Vddmu / this->temperature();

    }
    
    MatrixXd Integrate_cuda(int type){
        VectorXd mean_gh = this -> _gh -> mean();
        MatrixXd sigmapts_gh = this -> _gh -> sigmapts();
        MatrixXd weights_gh = this -> _gh -> weights();
        MatrixXd result;

        if (type == 0)
            result = MatrixXd::Zero(1,1);
        else if (type ==  1)
            result = MatrixXd::Zero(sigmapts_gh.cols(),1);
        else
           result = MatrixXd::Zero(sigmapts_gh.cols(),sigmapts_gh.cols());

        _cuda -> CudaIntegration(sigmapts_gh, weights_gh, result, mean_gh, type);                  

        return result;
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

    inline void update_cuda() override{
        _cuda = std::make_shared<CudaOperation_Quad>(CudaOperation_Quad{_sigma, _epsilon, _radius});
    }

    inline void cuda_init() override{
        _cuda -> Cuda_init(this -> _gh -> weights());
    }

    inline void cuda_free() override{
        _cuda -> Cuda_free();
    }

    double fact_cost_value(const VectorXd& fill_joint_mean, const SpMat& joint_cov) override {

        VectorXd mean_k = extract_mu_from_joint(fill_joint_mean);
        MatrixXd Cov_k = extract_cov_from_joint(joint_cov);

        updateGH(mean_k, Cov_k);

        return Integrate_cuda(0)(0, 0) / this->temperature();
    }

    std::shared_ptr<CudaOperation_Quad> _cuda;
    double _sigma, _epsilon, _radius;

};


}

#endif // NGDFactorizedBaseGH_H
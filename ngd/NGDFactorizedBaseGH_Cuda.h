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
#include "gvibase/GVIFactorizedBaseGH_Cuda.h"
#include <gpmp2/kinematics/PointRobotModel.h>
#include <gpmp2/obstacle/ObstaclePlanarSDFFactor.h>
// #include <gpmp2/obstacle/ObstacleSDFFactor.h>

#include <memory>

using namespace Eigen;

namespace gvi{

template <typename CostClass>
class NGDFactorizedBaseGH_Cuda: public GVIFactorizedBaseGH_Cuda{

    using GVIBase = GVIFactorizedBaseGH_Cuda;
    using Function = std::function<double(const VectorXd&, const CostClass &)>;

    // Robot = gpmp2::PointRobotModel
    // CostClass = gpmp2::ObstaclePlanarSDFFactor<gpmp2::PointRobotModel>
    // NGDClass = NGDFactorizedBaseGH_Cuda<gpmp2::ObstaclePlanarSDFFactor<gpmp2::PointRobotModel>>

public:
    ///@param dimension The dimension of the state
    ///@param function Template function class which calculate the cost
    // NGDFactorizedBaseGH(const int& dimension, const Function& function, const CostClass& cost_class_, const MatrixXd& Pk_):
    
    NGDFactorizedBaseGH_Cuda(int dimension, int state_dim, int gh_degree, 
                        const Function& function, const CostClass& cost_class,
                        int num_states, int start_index, 
                        double temperature, double high_temperature,
                        QuadratureWeightsMap weight_sigpts_map_option):
                _cost_class{cost_class}, 
                GVIBase(dimension, state_dim, num_states, start_index, 
                        temperature, high_temperature, weight_sigpts_map_option)
            {
                /// Override of the GVIBase classes. _func_phi-> Scalar, _func_Vmu -> Vector, _func_Vmumu -> Matrix
                GVIBase::_func_phi = [this, function, cost_class](const VectorXd& x){return MatrixXd::Constant(1, 1, function(x, cost_class));};
                GVIBase::_func_Vmu = [this, function, cost_class](const VectorXd& x){return (x-GVIBase::_mu) * function(x, cost_class);};
                GVIBase::_func_Vmumu = [this, function, cost_class](const VectorXd& x){return MatrixXd{(x-GVIBase::_mu) * (x-GVIBase::_mu).transpose().eval() * function(x, cost_class)};};
                GVIBase::_gh = std::make_shared<GH>(GH{gh_degree, GVIBase::_dim, GVIBase::_mu, GVIBase::_covariance, weight_sigpts_map_option});

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
        MatrixXd pts(result.rows(), sigmapts_gh.rows()*result.cols());
        if (type == 0)
            result = MatrixXd::Zero(1,1);
        else if (type ==  1)
            result = MatrixXd::Zero(sigmapts_gh.cols(),1);
        else
            result = MatrixXd::Zero(sigmapts_gh.cols(),sigmapts_gh.cols());
        CudaIntegration(sigmapts_gh, weights_gh, result, mean_gh, type, pts);
        // std::cout << "Sigma Points:" << std::endl << sigmapts_gh << std::endl;
        // std::cout << "Function Value:" << std::endl << pts << std::endl;
        return result;
    }

    void CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type, MatrixXd& pts);

    // __host__ __device__ double cost_obstacle_planar(const VectorXd& pose, const gpmp2::ObstaclePlanarSDFFactor<gpmp2::PointRobotModel>& obs_factor){

    //     VectorXd vec_err = obs_factor.evaluateError(pose);
    //     MatrixXd precision_obs{MatrixXd::Identity(vec_err.rows(), vec_err.rows())};
    //     printf("In the function");
    //     // precision_obs = precision_obs / obs_factor.get_noiseModel()->sigmas()[0]; //Sigma penalty
    //     precision_obs = precision_obs * 15.5;
    //     return vec_err.transpose().eval() * precision_obs * vec_err;
    // }

    __host__ __device__ double cost_obstacle_planar(const VectorXd& pose){
        printf("In the function ");
        return pose.norm();
        // VectorXd vec_err = obs_factor.evaluateError(pose);
        // MatrixXd precision_obs{MatrixXd::Identity(vec_err.rows(), vec_err.rows())};
        // printf("In the function");
        // precision_obs = precision_obs * 15.5;
        // return vec_err.transpose().eval() * precision_obs * vec_err;
        // return 
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

    CostClass _cost_class;
    

};


}

#endif // NGDFactorizedBaseGH_H
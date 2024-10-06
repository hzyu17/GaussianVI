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

class NGDFactorizedBaseGH_No_Template: public GVIFactorizedBaseGH_Cuda{

    using GVIBase = GVIFactorizedBaseGH_Cuda;

public:
    ///@param dimension The dimension of the state
    ///@param function Template function class which calculate the cost
    // NGDFactorizedBaseGH(const int& dimension, const Function& function, const CostClass& cost_class_, const MatrixXd& Pk_):
    
    NGDFactorizedBaseGH_No_Template(int dimension, int state_dim, int gh_degree, 
                        int num_states, int start_index, double cost_sigma, 
                        double epsilon, double radius, 
                        double temperature, double high_temperature,
                        QuadratureWeightsMap weight_sigpts_map_option,
                        std::shared_ptr<CudaOperation> cuda_ptr):
                GVIBase(dimension, state_dim, num_states, start_index, 
                        temperature, high_temperature, weight_sigpts_map_option),
                _epsilon(epsilon),
                _sigma(cost_sigma),
                _radius(radius)
            {
                GVIBase::_gh = std::make_shared<GH>(GH{gh_degree, GVIBase::_dim, GVIBase::_mu, GVIBase::_covariance, weight_sigpts_map_option});
                // _cuda = std::make_shared<CudaOperation>(CudaOperation{cost_sigma, epsilon, radius});
                _cuda = cuda_ptr;

            }
public:

    void calculate_partial_V() override{
        // update the mu and sigma inside the gauss-hermite integrator
        updateGH(this->_mu, this->_covariance);

        this->_Vdmu.setZero();
        this->_Vddmu.setZero();

        /// Integrate for E_q{_Vdmu} 
        this->_Vdmu = Integrate_cuda(1);
        // this->_Vdmu = this->_gh->Integrate(this->_func_Vmu);
        this->_Vdmu = this->_precision * this->_Vdmu;
        this->_Vdmu = this->_Vdmu / this->temperature();

        /// Integrate for E_q{phi(x)}
        double E_phi = Integrate_cuda(0)(0, 0);
        // double E_phi = this->_gh->Integrate(this->_func_phi)(0, 0);
        
        /// Integrate for partial V^2 / ddmu_ 
        MatrixXd E_xxphi{Integrate_cuda(2)};
        // MatrixXd E_xxphi{this->_gh->Integrate(this->_func_Vmumu)};

        this->_Vddmu.triangularView<Upper>() = (this->_precision * E_xxphi * this->_precision - this->_precision * E_phi).triangularView<Upper>();
        this->_Vddmu.triangularView<StrictlyLower>() = this->_Vddmu.triangularView<StrictlyUpper>().transpose();
        this->_Vddmu = this->_Vddmu / this->temperature();

    }


    std::tuple<double, VectorXd, MatrixXd> derivatives() override{
        updateGH(this->_mu, this->_covariance);

        this->_Vdmu.setZero();
        this->_Vddmu.setZero();

        this->_Vdmu = Integrate_cuda(1);
        this->_Vdmu = this->_precision * this->_Vdmu;
        this->_Vdmu = this->_Vdmu / this->temperature();

        double E_phi = Integrate_cuda(0)(0, 0);
        
        MatrixXd E_xxphi{Integrate_cuda(2)};
        this->_Vddmu.triangularView<Upper>() = (this->_precision * E_xxphi * this->_precision - this->_precision * E_phi).triangularView<Upper>();
        this->_Vddmu.triangularView<StrictlyLower>() = this->_Vddmu.triangularView<StrictlyUpper>().transpose();
        this->_Vddmu = this->_Vddmu / this->temperature();

        return std::make_tuple(E_phi, this->_Vdmu, this->_Vddmu);

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
           
        // MatrixXd pts(result.rows(), sigmapts_gh.rows()*result.cols());
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

    inline void update_cuda() override{
        _cuda = std::make_shared<CudaOperation>(CudaOperation{_sigma, _epsilon, _radius});
    }

    inline void cuda_init() override{
        _cuda -> Cuda_init(this -> _gh -> weights());
    }

    inline void cuda_free() override{
        _cuda -> Cuda_free();
    }

    inline bool linear_factor() override { return _isLinear; }

    inline void costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols) override{
        _cuda -> costIntegration(sigmapts, results, sigmapts_cols);
    }

    inline void dmuIntegration(const MatrixXd& sigmapts, const MatrixXd& mean, VectorXd& E_phi_mat, VectorXd& dmu_mat, MatrixXd& ddmu_mat, const int sigmapts_cols) override{
        std::cout << std::setprecision(16) << "Sigmapts norm = " << sigmapts.norm() << std::endl;
        std::cout << std::setprecision(16) << "Mean norm = " << mean.norm() << std::endl;
        _cuda -> funcValueIntegration(sigmapts, E_phi_mat, sigmapts_cols);
        _cuda -> dmuIntegration(sigmapts, mean, dmu_mat, sigmapts_cols);
        _cuda -> ddmuIntegration(ddmu_mat);
        // std::cout << "dmu_mat norm = " << dmu_mat.norm() << std::endl;
        // std::cout << "ddmu_mat norm = " << ddmu_mat.norm() << std::endl;
        _cuda -> Cuda_free_iter();
    }

    double fact_cost_value(const VectorXd& fill_joint_mean, const SpMat& joint_cov) override {

        VectorXd mean_k = extract_mu_from_joint(fill_joint_mean);
        MatrixXd Cov_k = extract_cov_from_joint(joint_cov);

        updateGH(mean_k, Cov_k);

        // Function Value becomes 0 after Exp2
        return Integrate_cuda(0)(0, 0) / this->temperature();
        // return this->_gh->Integrate(this->_func_phi)(0, 0) / this->temperature();
    }

    void cuda_matrices(const VectorXd& fill_joint_mean, const SpMat& joint_cov, std::vector<MatrixXd>& vec_sigmapts, std::vector<VectorXd>& vec_mean) override {

        // Extract only displacement without extracting velocity
        VectorXd mean_k = extract_mu_from_joint(fill_joint_mean);
        MatrixXd Cov_k = extract_cov_from_joint(joint_cov);        

        updateGH(mean_k, Cov_k);

        vec_mean[_start_index-1] = mean_k;
        vec_sigmapts[_start_index-1] = this -> _gh -> sigmapts();

    }

    void cuda_matrices(std::vector<MatrixXd>& vec_sigmapts, std::vector<VectorXd>& vec_mean) override {   

        updateGH(this->_mu, this->_covariance);

        vec_mean[_start_index-1] = this->_mu;
        vec_sigmapts[_start_index-1] = this -> _gh -> sigmapts();

    }
    
    VectorXd _mean;
    std::shared_ptr<CudaOperation> _cuda;
    double _sigma, _epsilon, _radius;
    bool _isLinear = false;

};


}

#endif // NGDFactorizedBaseGH_H
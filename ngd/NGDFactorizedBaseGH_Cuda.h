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
#include <memory>

using namespace Eigen;

namespace gvi{

template <typename CudaClass>
class NGDFactorizedBaseGH_Cuda: public GVIFactorizedBaseGH_Cuda{

    using GVIBase = GVIFactorizedBaseGH_Cuda;

public:
    ///@param dimension The dimension of the state
    ///@param function Template function class which calculate the cost
    // NGDFactorizedBaseGH(const int& dimension, const Function& function, const CostClass& cost_class_, const MatrixXd& Pk_):
    
    NGDFactorizedBaseGH_Cuda(int dimension, int state_dim, int gh_degree, 
                        int num_states, int start_index, double cost_sigma, 
                        double epsilon, double radius, 
                        double temperature, double high_temperature,
                        std::shared_ptr<QuadratureWeightsMap> weight_sigpts_map_option,
                        std::shared_ptr<CudaClass> cuda_ptr):
                GVIBase(dimension, state_dim, num_states, start_index, 
                        temperature, high_temperature, weight_sigpts_map_option),
                _epsilon(epsilon),
                _sigma(cost_sigma),
                _radius(radius)
            {
                GVIBase::_gh = std::make_shared<GH>(GH{gh_degree, GVIBase::_dim, GVIBase::_mu, GVIBase::_covariance, weight_sigpts_map_option});
                // _cuda = std::make_shared<CudaClass>(CudaClass{cost_sigma, epsilon, radius});
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

    void calculate_partial_V(const MatrixXd& ddmu_mat, const VectorXd& Vdmu, double E_Phi) override{

        this->_Vdmu.setZero();
        this->_Vddmu.setZero();

        /// Integrate for E_q{_Vdmu} 
        this->_Vdmu = Vdmu;
        this->_Vdmu = this->_precision * this->_Vdmu;
        this->_Vdmu = this->_Vdmu / this->temperature();
        
        /// Integrate for partial V^2 / ddmu_ 
        MatrixXd E_xxphi = ddmu_mat;

        this->_Vddmu.triangularView<Upper>() = (this->_precision * E_xxphi * this->_precision - this->_precision * E_Phi).triangularView<Upper>();
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

    inline VectorXd local2joint_dmu_insertion() override{ 
        VectorXd res(this->_joint_size);
        res.setZero();
        res.block(this->_state_dim * this->_start_index, 0, this->_dim, 1) = this->_Vdmu;
        return res;
    }

    inline SpMat local2joint_dprecision_insertion() override{ 
        SpMat res(this->_joint_size, this->_joint_size);

        for (int i = 0; i < this->_dim; ++i)
            for (int j = 0; j < this->_dim; ++j)
                res.insert(i + this->_state_dim * this->_start_index, j + this->_state_dim * this->_start_index) = this->_Vddmu(i, j);
        
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

    // inline void update_cuda() override{
    //     _cuda = std::make_shared<CudaClass>(CudaClass{_sigma, _epsilon, _radius});
    // }

    inline void cuda_init() override{
        _cuda -> Cuda_init(this -> _gh -> weights());
        _cuda -> zeromean_init(this -> _gh ->zeromeanpts());
        // _cuda -> initializeSigmaptsResources(2, 48, 89);
    }

    inline void cuda_free() override{
        _cuda -> Cuda_free();
        _cuda -> zeromean_free();
        // _cuda -> freeSigmaptsResources(48);
    }

    inline bool linear_factor() override { return _isLinear; }

    inline void newCostIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols) override{
        _cuda -> Cuda_init_iter(sigmapts, results, sigmapts_cols);
        _cuda -> costIntegration(sigmapts, results, sigmapts_cols);
        _cuda -> Cuda_free_iter();
    }

    inline void dmuIntegration(const MatrixXd& sigmapts, const MatrixXd& mean, VectorXd& E_phi_mat, VectorXd& dmu_mat, MatrixXd& ddmu_mat, const int sigmapts_cols) override{
        _cuda -> Cuda_init_iter(sigmapts, E_phi_mat, sigmapts_cols);
        _cuda -> costIntegration(sigmapts, E_phi_mat, sigmapts_cols);
        _cuda -> dmuIntegration(sigmapts, mean, dmu_mat, sigmapts_cols);
        _cuda -> ddmuIntegration(ddmu_mat);
        _cuda -> Cuda_free_iter();
    }

    inline void compute_sigmapts(const MatrixXd& mean, const MatrixXd& covariance, int dim_conf, int num_states, MatrixXd& sigmapts) override{
        // MatrixXd P_0 = covariance.block(0, 0, dim_conf, dim_conf);
        // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(P_0);
        // MatrixXd sqrtP = es.operatorSqrt();

        // MatrixXd eigen_value = es.eigenvalues();
        // std::cout << "Eigen values" << std::endl << eigen_value.transpose() << std::endl;

        _cuda->update_sigmapts(covariance, mean, dim_conf, num_states, sigmapts);
        // std::cout << "Cholesky result" << std::endl << sqrtP << std::endl;
    }

    double fact_cost_value(const VectorXd& fill_joint_mean, const SpMat& joint_cov) override {

        VectorXd mean_k = extract_mu_from_joint(fill_joint_mean);
        MatrixXd Cov_k = extract_cov_from_joint(joint_cov);

        updateGH(mean_k, Cov_k);

        return Integrate_cuda(0)(0, 0) / this->temperature();
    }

    void cuda_matrices(const VectorXd& fill_joint_mean, const SpMat& joint_cov, std::vector<MatrixXd>& vec_sigmapts, std::vector<VectorXd>& vec_mean) override {

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
    std::shared_ptr<CudaClass> _cuda;
    double _sigma, _epsilon, _radius;
    bool _isLinear = false;

};


}

#endif // NGDFactorizedBaseGH_H
/**
 * @file ProxKLFactorizedBaseGH_Cuda.h
 * @author Zinuo Chang (zchang40@gatech.edu)
 * @brief The marginal optimizer class expecting two functions, one is the cost function, f1(x, cost1); 
 * the other is the function for GH expectation, f2(x).
 * @version 0.1
 * @date 2024-12-20
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include "gvibase/GVIFactorizedBaseGH_Cuda.h"
#include <memory>

using namespace Eigen;

namespace gvi{

template <typename CudaClass>
class ProxKLFactorizedBaseGH_Cuda: public GVIFactorizedBaseGH_Cuda{

    using GVIBase = GVIFactorizedBaseGH_Cuda;

public:
    ///@param dimension The dimension of the state
    ///@param function_d Template function class which calculate the cost
    // ProxKLFactorizedBaseGH_Cuda(const int& dimension, const Function& function, const CostClass& cost_class_, const MatrixXd& Pk_):
    
    ProxKLFactorizedBaseGH_Cuda(int dimension, int state_dim, int gh_degree, 
                        int num_states, int start_index,
                        double temperature, double high_temperature,
                        std::shared_ptr<QuadratureWeightsMap> weight_sigpts_map_option,
                        std::shared_ptr<CudaClass> cuda_ptr):
                GVIBase(dimension, state_dim, num_states, start_index, 
                        temperature, high_temperature, weight_sigpts_map_option)
            {
                GVIBase::_gh = std::make_shared<GH>(GH{gh_degree, GVIBase::_dim, GVIBase::_mu, GVIBase::_covariance, weight_sigpts_map_option});
                _cuda = cuda_ptr;
            }
public:

    void calculate_partial_V(const MatrixXd& ddmu_mat, const VectorXd& Vdmu, double E_Phi) override{
        this->_Vdmu.setZero();
        this->_Vddmu.setZero();

        /// Integrate for E_q{_Vdmu} 
        this->_Vdmu = Vdmu;
        this->_Vdmu = this->_precision * this->_Vdmu;
        
        /// Integrate for partial V^2 / ddmu_ 
        MatrixXd E_xxphi = ddmu_mat;

        this->_Vddmu.triangularView<Upper>() = (this->_precision * E_xxphi * this->_precision - this->_precision * E_Phi).triangularView<Upper>();
        this->_Vddmu.triangularView<StrictlyLower>() = this->_Vddmu.triangularView<StrictlyUpper>().transpose();
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

    inline SpMat local2joint_dprecision_triplet() override{ 
        SpMat res(this->_joint_size, this->_joint_size);
        std::vector<Trip> triplets;
        triplets.reserve(this->_dim * this->_dim);  

        int offset = this->_state_dim * this->_start_index;
        for (int i = 0; i < this->_dim; ++i)
            for (int j = 0; j < this->_dim; ++j)
                triplets.emplace_back(i + offset, j + offset, this->_Vddmu(i, j));

        res.setFromTriplets(triplets.begin(), triplets.end());
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

    inline void cuda_init(const int n_states) override{
        _cuda -> Cuda_init(this -> _gh -> weights(), this -> _gh ->zeromeanpts(), n_states);
    }

    inline void cuda_free() override{
        _cuda -> Cuda_free();
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
    
    std::shared_ptr<CudaClass> _cuda;
    bool _isLinear = false;
};


}
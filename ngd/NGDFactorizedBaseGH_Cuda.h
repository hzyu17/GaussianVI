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

#ifndef NGDFactorizedBaseGH_CUDA_H
#define NGDFactorizedBaseGH_CUDA_H

#include <iostream>
#include <random>
#include <utility>
#include <assert.h>
#include <memory>
#include <type_traits>
#include <cuda_runtime.h>
#include <memory>

#include "quadrature/SparseGaussHermite.h"
// #include "helpers/CommonDefinitions.h"
#include "helpers/MatrixHelper.h"
// #include "helpers/CudaOperation.h"


using namespace Eigen;

namespace gvi{

    using GHFunction = std::function<MatrixXd(const VectorXd&)>;
    using GH = SparseGaussHermite<GHFunction>;

    struct NoneType {};

template <typename CostClass = NoneType>
class NGDFactorizedBaseGH{

    using Function = std::function<double(const VectorXd&, const CostClass &)>;
    // using Cuda = CudaOperation<CostClass>;

public:
    ///@param dimension The dimension of the state
    ///@param function Template function class which calculate the cost
    // NGDFactorizedBaseGH(const int& dimension, const Function& function, const CostClass& cost_class_, const MatrixXd& Pk_):
    
    NGDFactorizedBaseGH(int dimension, int state_dim, int gh_degree, 
                        const Function& function, const CostClass& cost_class,
                        int num_states, int start_index, 
                        double temperature=1.0, double high_temperature=10.0):
                _dim{dimension},
                _state_dim{state_dim},
                _num_states{num_states},
                _temperature{temperature},
                _high_temperature{high_temperature},
                _mu(_dim),
                _covariance{MatrixXd::Identity(_dim, _dim)},
                _precision{MatrixXd::Identity(_dim, _dim)},
                _dprecision(_dim, _dim),
                _dcovariance(_dim, _dim),
                _block{state_dim, num_states, start_index, dimension},
                _Pk(dimension, state_dim*num_states),
                _E_Phi(0.0),
                _Vdmu(_dim),
                _Vddmu(_dim, _dim),
                _function{function}, 
                _cost_class{cost_class}
                {   
                    _joint_size = state_dim * num_states;
                    _Pk.setZero();
                    _Pk.block(0, start_index*state_dim, dimension, dimension) = Eigen::MatrixXd::Identity(dimension, dimension);
                    _func_phi = [this, function, cost_class](const VectorXd& x){return MatrixXd::Constant(1, 1, function(x, cost_class));};
                    _func_Vmu = [this, function, cost_class](const VectorXd& x){return (x-_mu) * function(x, cost_class);};
                    _func_Vmumu = [this, function, cost_class](const VectorXd& x){return MatrixXd{(x-_mu) * (x-_mu).transpose().eval() * function(x, cost_class)};};
                    _gh = std::make_shared<GH>(GH{gh_degree, _dim, _mu, _covariance});
                }

public:
    /**
     * @brief Update the step size
     */
    inline void set_step_size(double step_size){
        _step_size = step_size;
    }

    /**
     * @brief Update mean
     */
    inline void update_mu(const VectorXd& new_mu){ 
        _mu = new_mu; 
    }

    /**
     * @brief Update covariance matrix
     */
    inline void update_covariance(const MatrixXd& new_cov){ 
        _covariance = new_cov; 
        _precision = _covariance.inverse();
    }

    inline MatrixXd Pk(){
        return _Pk;
    }

    /**
     * @brief Update the marginal mean.
     */
    inline void update_mu_from_joint(const VectorXd & fill_joint_mean) {
        _mu = _block.extract_vector(fill_joint_mean);
    }

    /**
     * @brief Update the marginal precision matrix.
     */
    inline void update_precision_from_joint(const SpMat& fill_joint_cov) {
        _covariance = extract_cov_from_joint(fill_joint_cov);
        _precision = _covariance.inverse();
    }

    inline VectorXd extract_mu_from_joint(const VectorXd & fill_joint_mean) {
        return _block.extract_vector(fill_joint_mean);
    }

    inline SpMat extract_cov_from_joint(const SpMat& fill_joint_cov) {
        return _block.extract(fill_joint_cov);
    }

    inline SpMat fill_joint_cov(){
        SpMat joint_cov(_joint_size, _joint_size);
        joint_cov.setZero();
        _block.fill(_covariance, joint_cov);
        return joint_cov;
    }

    inline VectorXd fill_joint_mean(){
        VectorXd joint_mean(_joint_size);
        joint_mean.setZero();
        _block.fill_vector(_mu, joint_mean);
        return joint_mean;
    }

    inline VectorXd mean() const{ return _mu; }

    inline MatrixXd precision() const {return _precision; }

    inline MatrixXd covariance() const {return _covariance; }

    /**
     * @brief returns the Phi(x) 
     */
    inline MatrixXd negative_log_probability(const VectorXd& x) const{
        return _func_phi(x);
    }    

    void factor_switch_to_high_temperature(){
        _temperature = _high_temperature;
    }

    double temperature(){
        return _temperature;
    }

    void calculate_partial_V(){
        // update the mu and sigma inside the gauss-hermite integrator
        updateGH(this->_mu, this->_covariance);

        this->_Vdmu.setZero();
        this->_Vddmu.setZero();

        /// Integrate for E_q{_Vdmu} 
        this->_Vdmu = Integrate_cuda(this->_func_Vmu, 1);
        this->_Vdmu = this->_precision * this->_Vdmu;
        this->_Vdmu = this->_Vdmu / this->temperature();

        /// Integrate for E_q{phi(x)}
        double E_phi = Integrate_cuda(this->_func_phi, 0)(0, 0);
        
        /// Integrate for partial V^2 / ddmu_ 
        MatrixXd E_xxphi{Integrate_cuda(this->_func_Vmumu, 2)};

        this->_Vddmu.triangularView<Upper>() = (this->_precision * E_xxphi * this->_precision - this->_precision * E_phi).triangularView<Upper>();
        this->_Vddmu.triangularView<StrictlyLower>() = this->_Vddmu.triangularView<StrictlyUpper>().transpose();
        this->_Vddmu = this->_Vddmu / this->temperature();
    }

    MatrixXd Integrate_cuda(GHFunction func, int type){
        VectorXd mean_gh = this -> _gh -> mean();
        MatrixXd sigmapts_gh = this -> _gh -> sigmapts();
        MatrixXd weights_gh = this -> _gh -> weights();
        MatrixXd result{func(mean_gh)};
        result.setZero();
        CudaIntegration(sigmapts_gh, weights_gh, result, mean_gh, type);
        return result;
    }

    void CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type);

    __host__ __device__ inline double cost_function1(const VectorXd& vec_x){
        double x = vec_x(0);
        double mu_p = 20, f = 400, b = 0.1, sig_r_sq = 0.09;
        double sig_p_sq = 9;

        // y should be sampled. for single trial just give it a value.
        double y = f*b/mu_p - 0.8;

        return ((x - mu_p)*(x - mu_p) / sig_p_sq / 2 + (y - f*b/x)*(y - f*b/x) / sig_r_sq / 2); 
    }

    inline VectorXd local2joint_dmu(){ 
        VectorXd res(this->_joint_size);
        res.setZero();
        this->_block.fill_vector(res, this->_Vdmu);
        return res;
    }

    inline SpMat local2joint_dprecision(){ 
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

    double fact_cost_value(const VectorXd& fill_joint_mean, const SpMat& joint_cov) {
        VectorXd mean_k = extract_mu_from_joint(fill_joint_mean);
        MatrixXd Cov_k = extract_cov_from_joint(joint_cov);

        updateGH(mean_k, Cov_k);

        return this->_gh->Integrate(this->_func_phi)(0, 0) / this->temperature();
    }

    /// update the GH approximator
    void updateGH(const VectorXd& x, const MatrixXd& P){
        // This order cannot be changed! Need to update P before updating mean.
        _gh->update_P(P); 
        _gh->update_mean(x);
        _gh->update_sigmapoints();
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

    public:

    /// dimension
    int _dim, _state_dim, _num_states, _joint_size;

    VectorXd _mu;
    
    GHFunction _func_phi;
    // The function (x-\mu)*\phi(x)
    GHFunction _func_Vmu;
    // The function [(x-\mu)@(x-\mu)^T]*\phi(x)
    GHFunction _func_Vmumu;

    const Function _function;
    const CostClass _cost_class;

    /// G-H quadrature class
    std::shared_ptr<GH> _gh;
    // std::shared_ptr<Cuda> _cuda;

protected:

    /// optimization variables
    MatrixXd _precision, _dprecision;
    MatrixXd _covariance, _dcovariance;

    /// step sizes
    double _step_size = 0.9;
    double _E_Phi = 0.0;

    double _temperature, _high_temperature;
    
    // sparse mapping to sub variables
    TrajectoryBlock _block;

    MatrixXd _Pk;

    VectorXd _Vdmu;
    MatrixXd _Vddmu;

};


}

#endif // NGDFactorizedBaseGH_H
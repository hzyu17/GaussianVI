/**
 * @file GVIFactorizedBase.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief The base class for one factor.
 * @version 1.1
 * @date 2024-01-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once


#include <iostream>
#include <random>
#include <utility>
#include <assert.h>
#include <memory>
#include <type_traits>

// #include "quadrature/GaussHermite.h"
#include "quadrature/SparseGaussHermite_Cuda.h"
#include "helpers/CommonDefinitions.h"
#include "helpers/MatrixHelper.h"

using namespace Eigen;

namespace gvi{
using GHFunction = std::function<MatrixXd(const VectorXd&)>;
using GH = SparseGaussHermite_Cuda<GHFunction>;
// using GH = gvi::GaussHermite<GHFunction>;

struct NoneType {};

class GVIFactorizedBase_Cuda{
public:
    virtual ~GVIFactorizedBase_Cuda(){}
    /**
     * @brief Default Constructor
     */
    GVIFactorizedBase_Cuda(){}

    /**
     * @brief Construct a new GVIFactorizedBase object
     * 
     * @param dimension The dimension of the state
     */
    GVIFactorizedBase_Cuda(int dimension, int state_dim, int num_states, int start_index, 
                        double temperature=10.0, double high_temperature=100.0):
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
            _Vddmu(_dim, _dim)
            {   
                _joint_size = state_dim * num_states;
                _Pk.setZero();
                _Pk.block(0, start_index*state_dim, dimension, dimension) = Eigen::MatrixXd::Identity(dimension, dimension);
            }        
    
/// public functions
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


    /**
     * @brief Compute the increment on mean and precision (or covariance) matrix on the factorized level.
    */
    virtual void calculate_partial_V(){}

    /**
     * @brief Compute the cost function. V(x) = E_q(\phi(x))
     */
    virtual double fact_cost_value(const VectorXd& fill_joint_mean, const SpMat& joint_cov) {}

    /**
     * @brief Place holder for computing BW gradients for Gaussian distributions.
     */
    virtual void compute_BW_grads(){}

    /**
     * @brief Place holder for proximal GVI algorithm line search.
     * 
     */
    virtual std::tuple<Eigen::VectorXd, Eigen::MatrixXd> compute_gradients_linesearch(const double & step_size){}

    // /**
    //  * @brief Compute the cost function. V(x) = E_q(\phi(x)) using the current values.
    //  */

    /**
     * @brief Get the joint intermediate variable (partial V / partial mu).
     */
    virtual inline VectorXd local2joint_dmu() {}

    virtual inline VectorXd local2joint_dmu(Eigen::VectorXd & dmu_lcl) {}

    /**
     * @brief Get the joint Pk.T * V^2 / dmu /dmu * Pk using block insertion
     */
    virtual inline SpMat local2joint_dprecision() {}

    virtual inline SpMat local2joint_dcovariance() {}

    virtual inline SpMat local2joint_dprecision(Eigen::MatrixXd & dprecision_lcl){}

    virtual inline void update_cuda(){}

    virtual inline void cuda_init(){}

    virtual inline void cuda_free(){}

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



    /**
     * @brief Get the mean 
     */
    inline VectorXd mean() const{ return _mu; }

    inline MatrixXd precision() const {return _precision; }

    inline MatrixXd covariance() const {return _covariance; }

    inline MatrixXd Vdmu() const {return _Vdmu; }

    inline MatrixXd Vddmu() const {return _Vddmu; }

    /********************************************************/
    /// Function interfaces

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

    /// Public members for the inherited classes access
public:

    /// dimension
    int _dim, _state_dim, _num_states, _joint_size;

    VectorXd _mu;
    
    GHFunction _func_phi;
    // GHFunction _func_Vmu;
    // GHFunction _func_Vmumu;

    // /// G-H quadrature class
    // std::shared_ptr<GH> _gh;

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

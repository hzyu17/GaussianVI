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

#include "quadrature/GaussHermite.h"
#include "CommonDefinitions.h"
#include "MatrixHelper.h"

// using namespace std;
using namespace Eigen;

namespace gvi{

class GVIFactorizedBase{
public:
    virtual ~GVIFactorizedBase(){}
    /**
     * @brief Default Constructor
     */
    GVIFactorizedBase(){}

    /**
     * @brief Construct a new GVIFactorizedBase object
     * 
     * @param dimension The dimension of the state
     */
    GVIFactorizedBase(int dimension, int state_dim, int num_states, int start_index, 
                        double temperature=10.0, double high_temperature=100.0, bool is_linear=false):
            _is_linear{is_linear},
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
            _Pk(dimension, state_dim*num_states)
            {   
                _joint_size = state_dim * num_states;
                _Pk.setZero();
                _Pk.block(0, start_index*state_dim, dimension, dimension) = Eigen::MatrixXd::Identity(dimension, dimension);
            }        
    
/// public functions
public:

    /// update the GH approximator
    void updateGH(const VectorXd& x, const MatrixXd& P){
        _gh->update_mean(x);
        _gh->update_P(P); 
    }

    /**
     * @brief Update the step size
     */
    inline void set_step_size(double ss_mean, double ss_precision){
        _step_size_mu = ss_mean;
        _step_size_Sigma = ss_precision; 
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
    virtual double fact_cost_value(const VectorXd& fill_joint_mean, const SpMat& joint_cov) {
        
        
    }

    // /**
    //  * @brief Compute the cost function. V(x) = E_q(\phi(x)) using the current values.
    //  */

    /**
     * @brief Get the joint intermediate variable (partial V / partial mu).
     */
    virtual inline VectorXd local2joint_dmu() {}

    /**
     * @brief Get the joint Pk.T * V^2 / dmu /dmu * Pk using block insertion
     */
    virtual inline SpMat local2joint_dprecision() {}

    virtual inline SpMat local2joint_dcovariance() {}

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


    /********************************************************/
    /// Function interfaces

    /**
     * @brief returns the Phi(x) 
     */
    inline MatrixXd negative_log_probability(const VectorXd& x) const{
        return _func_phi(x);
    }    

    /**
     * @brief returns the E_q{phi(x)} = E_q{-log(p(x,z))}
     */
    inline double E_Phi() {
        return _gh->Integrate(_func_phi)(0, 0);
    }

    void set_GH_points(int p){
        _gh->set_polynomial_deg(p);
    }

    void switch_to_high_temperature(){
        _temperature = _high_temperature;
    }

    double temperature(){
        return _temperature;
    }

    /// Public members for the inherited classes access
public:

    bool _is_linear;

    /// dimension
    int _dim, _state_dim, _num_states, _joint_size;

    VectorXd _mu;
    
    /// Intermediate functions for Gauss-Hermite quadratures, default definition, needed to be overrided by the
    /// derived classes.
    using GHFunction = std::function<MatrixXd(const VectorXd&)>;
    GHFunction _func_phi;

    /// G-H quadrature class
    using GH = GaussHermite<GHFunction> ;
    std::shared_ptr<GH> _gh;

protected:

    /// optimization variables
    MatrixXd _precision, _dprecision;
    MatrixXd _covariance, _dcovariance;

    /// step sizes
    double _step_size_mu = 0.9;
    double _step_size_Sigma = 0.9;

    double _temperature, _high_temperature;
    
    // sparse mapping to sub variables
    TrajectoryBlock _block;

    MatrixXd _Pk;
    
};


}

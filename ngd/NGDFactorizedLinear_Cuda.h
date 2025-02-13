/**
 * @file NGDFactorizedLinear.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief Factorized optimization steps for linear gaussian factors 
 * -log(p(x|z)) = (1/2)*||\Lambda X - \Psi \mu_t||_{\Sigma_t^{-1}},
 *  which has closed-form expression in computing gradients wrt variables.
 * @version 0.1
 * @date 2023-01-17
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#ifndef NGDFactorizedLinear_H
#define NGDFactorizedLinear_H

#include "gvibase/GVIFactorizedBase_Cuda.h"
// #include "gp/linear_factor.h"

namespace gvi{
template <typename Factor = NoneType>
class NGDFactorizedLinear_Cuda : public GVIFactorizedBase_Cuda{
    using Base = GVIFactorizedBase_Cuda;
    using CostFunction = std::function<double(const VectorXd&, const Factor&)>;
public:
    NGDFactorizedLinear_Cuda(const int& dimension,
                        int dim_state,
                        const CostFunction& function, 
                        const Factor& linear_factor,
                        int num_states,
                        int start_indx,
                        double temperature,
                        double high_temperature):
        Base(dimension, dim_state, num_states, start_indx, temperature, high_temperature),
        _linear_factor{linear_factor}
        {
            Base::_func_phi = [this, function, linear_factor](const VectorXd& x){return MatrixXd::Constant(1, 1, function(x, linear_factor));};
        
            _target_mean = linear_factor.get_mu();
            _target_precision = linear_factor.get_precision();
            _Lambda = linear_factor.get_Lambda();
            _Psi = linear_factor.get_Psi();
            _constant = linear_factor.get_Constant();
        }

protected:
    Factor _linear_factor;

    MatrixXd _target_mean, _target_precision, _Lambda, _Psi;

    double _constant;

    bool _isLinear = true;

public:
    double constant() const { return _constant; }

    /*Calculating phi * (partial V) / (partial mu), and 
        * phi * (partial V^2) / (partial mu * partial mu^T) for Gaussian posterior: closed-form expression:
        * (partial V) / (partial mu) = Sigma_t{-1} * (mu_k - mu_t)
        * (partial V^2) / (partial mu)(partial mu^T): higher order moments of a Gaussian.
    */

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
        this->_block.fill_vector(res, this->_Vdmu);
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

    void calculate_partial_V() override{
        // Timer timer;

        _Vdmu.setZero();
        _Vddmu.setZero();

        // helper vectors
        MatrixXd tmp{MatrixXd::Zero(_dim, _dim)};
        MatrixXd tmp_cuda{MatrixXd::Zero(_dim, _dim)};

        // partial V / partial mu           
        this->_Vdmu = (2 * _Lambda.transpose() * _target_precision * (_Lambda*_mu - _Psi*_target_mean)) * constant();
        this->_Vdmu = this->_Vdmu / this->temperature();

        MatrixXd AT_precision_A = _Lambda.transpose() * _target_precision * _Lambda;

        // timer.start();

        // partial V^2 / partial mu*mu^T
        // update tmp matrix

        for (int i=0; i<(_dim); i++){
            for (int j=0; j<(_dim); j++) {
                for (int k=0; k<(_dim); k++){
                    for (int l=0; l<(_dim); l++){
                        tmp(i, j) += (_covariance(i, j)*_covariance(k, l) + _covariance(i,k)*_covariance(j,l) + _covariance(i,l)*_covariance(j,k))*AT_precision_A(k,l);
                    }
                }
            }
        }

        // computeTmp_CUDA(tmp, _covariance, AT_precision_A);

        // MatrixXd tmp_error = tmp - tmp_cuda;
        // std::cout << "tmp norm = " << tmp.norm() << std::endl << "tmp error = " << std::endl << tmp_error << std::endl;

        // std::cout << std::endl << "tmp: " << timer.end_sec() * 1000 << "ms" << std::endl;

        this->_Vddmu = (_precision * tmp * _precision - _precision * (AT_precision_A*_covariance).trace()) * constant();
        this->_Vddmu = this->_Vddmu / this->temperature();
    }

    double fact_cost_value(const VectorXd& fill_joint_mean, const SpMat& joint_cov) override {
        VectorXd mean_k = Base::extract_mu_from_joint(fill_joint_mean);
        MatrixXd Cov_k = Base::extract_cov_from_joint(joint_cov);

        _E_Phi = ((_Lambda.transpose()*_target_precision*_Lambda * Cov_k).trace() + 
                    (_Lambda*mean_k-_Psi*_target_mean).transpose() * _target_precision * (_Lambda*mean_k-_Psi*_target_mean)) * constant();

        return _E_Phi / this->temperature();
    }

    inline bool linear_factor() override { return _isLinear; }
};

} //namespace
#endif //NGDFactorizedLinear_H
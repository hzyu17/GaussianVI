/**
 * @file ProxGVIFactorizedLinear.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief Factorized optimization steps for gaussian factors as follow:
 * -log(p(x|z)) = (1/2)*||\Lambda X - \Psi \mu_t||_{\Sigma_t^{-1}},
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#ifndef ProxGVIFactorizedLinear_H
#define ProxGVIFactorizedLinear_H

#include "gvibase/GVIFactorizedBase.h"

namespace gvi{
template <typename Factor = NoneType>
class ProxFactorizedLinear : public GVIFactorizedBase{
    using Base = GVIFactorizedBase;
    using CostFunction = std::function<double(const VectorXd&, const Factor&)>;
public:
    ProxFactorizedLinear(const int& dimension,
                        int dim_state,
                        const CostFunction& function, 
                        const Factor& linear_factor,
                        int num_states,
                        int start_indx,
                        double temperature,
                        double high_temperature):
        Base(dimension, dim_state, num_states, start_indx, temperature, high_temperature),
        _linear_factor{linear_factor},
        _bk(dimension),
        _Sk(dimension, dimension)
        {
            Base::_func_phi = [this, function, linear_factor](const VectorXd& x){return MatrixXd::Constant(1, 1, function(x, linear_factor) / this->temperature() );};
        
            _target_mean = linear_factor.get_mu();
            _target_precision = linear_factor.get_precision();
            _Lambda = linear_factor.get_Lambda();
            _Psi = linear_factor.get_Psi();
            _constant = linear_factor.get_Constant();

            _bk.setZero();
            _Sk.setZero();
        }

protected:
    Factor _linear_factor;

    MatrixXd _target_mean, _target_precision, _Lambda, _Psi;

    double _constant;

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

    // inline VectorXd local2joint_dmu(Eigen::VectorXd & dmu_lcl) override{ 
    //     VectorXd res(this->_joint_size);
    //     res.setZero();
    //     this->_block.fill_vector(res, dmu_lcl);
    //     return res;
    // }

    // inline SpMat local2joint_dprecision(Eigen::MatrixXd & dprecision_lcl) override{ 
    //     SpMat res(this->_joint_size, this->_joint_size);
    //     res.setZero();
    //     this->_block.fill(dprecision_lcl, res);
    //     return res;
    // }

    void compute_BW_grads() override{
        // Compute the BW gradients
        _bk.setZero();
        _Sk.setZero();

        _bk = _Lambda.transpose()*_target_precision*(_Lambda*this->_mu - _Psi*_target_mean);
        _Sk = _Lambda.transpose()*_target_precision*_Lambda;
    }

    void calculate_partial_V(std::optional<double> stepsize_option=std::nullopt) override{

        this->compute_BW_grads();
        std::tuple<Eigen::VectorXd, Eigen::MatrixXd> gradients;

        if (stepsize_option.has_value()){
            gradients = BW_JKO(stepsize_option.value());
        }else{
            std::cout << "no step size input, using the default value in the class!" << std::endl;
            gradients = BW_JKO(this->_step_size);
        }

        this->_Vdmu = std::get<0>(gradients);
        this->_Vddmu = std::get<1>(gradients);

    }


    std::tuple<Eigen::VectorXd, Eigen::MatrixXd> BW_JKO(double step_size){
        Eigen::MatrixXd Identity(this->_dim, this->_dim);
        Identity.setZero();
        Identity = Eigen::MatrixXd::Identity(this->_dim, this->_dim);
        
        // Compute the proximal step
        Eigen::MatrixXd Mk(this->_dim, this->_dim);
        Mk.setZero();
        Eigen::MatrixXd Sigk(this->_dim, this->_dim);
        Sigk.setZero();

        Mk = Identity - step_size*this->_Sk;
        Sigk = this->_covariance;

        Eigen::MatrixXd Sigk_half(this->_dim, this->_dim);
        Sigk_half.setZero();
        Sigk_half = Mk*Sigk*Mk.transpose();
        
        Eigen::MatrixXd temp(this->_dim, this->_dim);
        temp.setZero();
        temp = Identity;
        temp = 4.0*step_size*temp;
        temp = Sigk_half + temp;
        temp = Sigk_half*temp;
        temp = this->sqrtm(temp);

        Eigen::VectorXd mk_new = this->_mu - step_size * this->_bk;
        Eigen::MatrixXd inv_Sigk_new(this->_dim, this->_dim);
        inv_Sigk_new.setZero();
        inv_Sigk_new = (0.5*Sigk_half + step_size*Identity + 0.5*temp).inverse();

        Eigen::VectorXd Vdmu = (mk_new - this->_mu) / step_size;
        Eigen::MatrixXd Vddmu = (inv_Sigk_new - this->_precision) / step_size;
        
        return std::make_tuple(Vdmu, Vddmu);
    }

    double fact_cost_value(const VectorXd& fill_joint_mean, const SpMat& joint_cov) override {
        VectorXd mean_k = Base::extract_mu_from_joint(fill_joint_mean);
        MatrixXd Cov_k = Base::extract_cov_from_joint(joint_cov);

        _E_Phi = std::move(((_Lambda.transpose()*_target_precision*_Lambda * Cov_k).trace() + 
                            (_Lambda*mean_k-_Psi*_target_mean).transpose() * _target_precision * (_Lambda*mean_k-_Psi*_target_mean)) * constant() / this->temperature());
        return _E_Phi;
    }

    // Helper functions
    bool isSymmetric(const Eigen::MatrixXd& matrix, double precision = 1e-10) {
        return (matrix - matrix.transpose()).cwiseAbs().maxCoeff() <= precision;
    }

    // Function to check if a matrix is symmetric positive-definite
    bool isSymmetricPositiveDefinite(const Eigen::MatrixXd& matrix, double precision = 1e-10) {
        if (!isSymmetric(matrix, precision)) {
            return false;  // Matrix is not symmetric
        }

        // Compute the eigenvalues using the Eigen solver
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(matrix);
        if (eigenSolver.info() != Eigen::Success) {
            throw std::runtime_error("Eigenvalue computation did not converge");
        }

        // Check if all eigenvalues are non-negative
        for (int i = 0; i < matrix.rows(); ++i) {
            if (eigenSolver.eigenvalues()[i] <= precision) {
                return false;  // Found a negative eigenvalue
            }
        }
        return true;
    }

    Eigen::MatrixXd sqrtm(const Eigen::MatrixXd& mat){
        assert(mat.rows() == mat.cols());

        Eigen::RealSchur<Eigen::MatrixXd> schur(mat.rows());
        schur.compute(mat);

        const Eigen::MatrixXd& T = schur.matrixT();
        const Eigen::MatrixXd& U = schur.matrixU();

        Eigen::MatrixXd T_sqrt = T;
        
        int n = T.rows();
        for (int i = 0; i < n;) {
            if (i == n - 1 || T(i + 1, i) == 0) {
                // Diagonal block is 1x1
                T_sqrt(i, i) = std::sqrt(T(i, i));
                i++;
            } else {
                // Diagonal block is 2x2
                Eigen::Matrix2d block = T.block<2, 2>(i, i);
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(block);
                Eigen::Matrix2d D_sqrt = es.eigenvalues().cwiseSqrt().asDiagonal();
                Eigen::Matrix2d S = es.eigenvectors();
                Eigen::Matrix2d S_inv = S.inverse();
                T_sqrt.block<2, 2>(i, i) = S * D_sqrt * S_inv;
                i += 2;
            }
        }

        Eigen::MatrixXd mat_sqrt = U * T_sqrt * U.transpose();
        return mat_sqrt;
    }

protected:
    Eigen::VectorXd _bk;
    Eigen::MatrixXd _Sk;

};

} //namespace
#endif //ProxGVIFactorizedLinear_H
/**
 * @file ProxKL-GH-Cuda.h
 * @author Zinuo Chang (zchang40@gatech.edu)
 * @brief The joint optimizer class using Gauss-Hermite quadrature, base class for different algorithms.
 * @version 1.1
 * @date 2024-12-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <utility>
#include <memory>

#include "gvibase/GVI-GH-Cuda.h"

using namespace Eigen;
namespace gvi{

template <typename FactorizedOptimizer>
class ProxKLGH: public GVIGH<FactorizedOptimizer>{
    using Base = GVIGH<FactorizedOptimizer>;
public:

    ProxKLGH(){}

    /**
     * @brief Construct a new VIMPOptimizerGH object
     * 
     * @param _vec_fact_optimizers vector of marginal optimizers
     * @param niters number of iterations
     */
    ProxKLGH(const std::vector<std::shared_ptr<FactorizedOptimizer>>& vec_fact_optimizers,
            int dim_state,
            int num_states,
            int niterations = 5,
            double temperature = 1.0,
            double high_temperature = 100.0) :
            GVIGH<FactorizedOptimizer>(vec_fact_optimizers, dim_state, num_states, niterations, temperature, high_temperature),
            _Vdmu(VectorXd::Zero(Base::_dim)),
            _Vddmu(SpMat(Base::_dim, Base::_dim))
        {
            _Vdmu.setZero();
            _Vddmu.setZero();
        }

protected:
    VectorXd _Vdmu;
    SpMat _Vddmu;
    VectorXd _mu_prior;
    SpMat _precision_prior;


public:
/// ************************* Override functions for Prox-GVI algorithm *************************************
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

    inline void set_prior(const VectorXd& mu_prior, const SpMat& precision_prior){
        _mu_prior = mu_prior;
        _precision_prior = precision_prior;
    }

/// Optimizations related

    std::tuple<double, VectorXd, VectorXd, SpMat> factor_cost_vector_cuda(const VectorXd& fill_joint_mean, SpMat& joint_precision);

    virtual std::tuple<double, VectorXd, SpMat> onestep_linesearch(const double &step_size, const VectorXd& dmu, const SpMat& dprecision) override;

    std::tuple<double, VectorXd, SpMat> bisection_update(const VectorXd& dmu, const SpMat& dprecision);

    inline void update_proposal(const VectorXd& new_mu, const SpMat& new_precision) override;

    void optimize(std::optional<bool> verbose=std::nullopt) override;

    void optimize_linear(std::optional<bool> verbose=std::nullopt);

    double KL_Divergence(const VectorXd& mean_former, const VectorXd& mean_latter, const SpMat& precision_former, const SpMat& precision_latter);

    /**
     * @brief Compute the total cost function value given a state, using current values.
     */
    double cost_value() override;

    double cost_value_cuda(const VectorXd& fill_joint_mean, SpMat& joint_precision);

    double cost_value_linear(const VectorXd& fill_joint_mean, const SpMat& joint_precision);

    /**
     * @brief given a state, compute the total cost function value without the entropy term, using current values.
     */
    double cost_value_no_entropy() override;

    /**
     * @brief Compute the costs of all factors, using current values.
     */
    VectorXd factor_cost_vector() override;

}; //class

} //namespace gvi

#include "ProxKL-GH-Cuda-impl.h"
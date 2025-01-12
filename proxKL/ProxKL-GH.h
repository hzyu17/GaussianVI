/**
 * @file ProxKL-GH.h
 * @author Zinuo Chang (zchang40@gatech.edu)
 * @brief The joint optimizer class using Gauss-Hermite quadrature, base class for different algorithms.
 * @version 1.1
 * @date 2024-12-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#ifndef PROXKL_GH_H
#define PROXKL_GH_H

#include <utility>
#include <memory>

#include "gvibase/GVI-GH.h"

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

/// Optimizations related
    /**
     * @brief Compute gradient and Hessian from marginal distributions.
     */
    std::tuple<VectorXd, SpMat> compute_gradients(std::optional<double>step_size=std::nullopt) override;

    std::tuple<VectorXd, SpMat> compute_gradients_KL(std::optional<double>step_size=std::nullopt);

    virtual std::tuple<double, VectorXd, SpMat> onestep_linesearch(const double &step_size, const VectorXd& dmu, const SpMat& dprecision) override;

    // std::tuple<double, VectorXd, SpMat> onestep_linesearch_prox(const double &step_size);
    std::tuple<VectorXd, SpMat> onestep_linesearch_prox(const double &step_size);

    // void optimize(std::optional<bool> verbose=std::nullopt) override;

    inline void update_proposal(const VectorXd& new_mu, const SpMat& new_precision) override;


    void optimize(std::optional<bool> verbose=std::nullopt) override;

    /**
     * @brief Compute the total cost function value given a state, using current values.
     */
    double cost_value() override;

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

#include "ProxKL-GH-impl.h"

#endif //PROXGVI_GH_H
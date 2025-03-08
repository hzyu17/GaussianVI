/**
 * @file NGD-GH.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief The joint optimizer class using Gauss-Hermite quadrature. 
 * @version 0.1
 * @date 2022-03-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#ifndef NGD_GH_H
#define NGD_GH_H

#include <utility>
#include <memory>

#include "gvibase/GVI-GH-Cuda.h"

using namespace Eigen;
namespace gvi{

template <typename FactorizedOptimizer, typename CudaClass>
class NGDGH: public GVIGH<FactorizedOptimizer, CudaClass>{
    using Base = GVIGH<FactorizedOptimizer, CudaClass>;
public:
    /**
     * @brief Default Constructor
     */
    NGDGH(){}

    /**
     * @brief Construct a new VIMPOptimizerGH object
     * 
     * @param _vec_fact_optimizers vector of marginal optimizers
     * @param niters number of iterations
     */
    NGDGH(const std::vector<std::shared_ptr<FactorizedOptimizer>>& vec_fact_optimizers,
          int dim_state,
          int num_states,
          std::shared_ptr<CudaClass> cuda_ptr,
          std::shared_ptr<GH> gh_ptr,
          int niterations = 5,
          double temperature = 1.0,
          double high_temperature = 100.0) :
        GVIGH<FactorizedOptimizer, CudaClass>(vec_fact_optimizers, dim_state, num_states, cuda_ptr, gh_ptr, niterations, temperature, high_temperature)
    {}

public:
/// ************************* Override functions for NGD algorithm *************************************
/// Optimizations related
    /**
     * @brief Function which computes one step of update.
     */
    std::tuple<VectorXd, SpMat> compute_gradients(std::optional<double>step_size=std::nullopt) override;

    std::tuple<double, VectorXd, SpMat> onestep_linesearch(const double &step_size, const VectorXd& dmu, const SpMat& dprecision) override;

    double bisection_stepsize(const VectorXd& dmu, const SpMat& dprecision);

    inline void update_proposal(const VectorXd& new_mu, const SpMat& new_precision) override;

    /**
     * @brief given a state, compute the total cost function value without the entropy term, using current values.
     */
    double cost_value_no_entropy() override;
    double KL_Divergence(const VectorXd& mean_former, const VectorXd& mean_latter, const SpMat& precision_former, const SpMat& precision_latter);

    bool isPositiveDefinite(const SpMat& precision);


}; //class


} //namespace gvi

// function implementations

#include "NGD-GH-Cuda-impl.h"

#endif 
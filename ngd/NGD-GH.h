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

#include "gvibase/GVI-GH-GBP.h"

using namespace Eigen;
namespace gvi{

template <typename FactorizedOptimizer>
class NGDGH: public GVIGH<FactorizedOptimizer>{
    using Base = GVIGH<FactorizedOptimizer>;
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
/// ************************* Override functions for NGD algorithm *************************************
/// Optimizations related
    /**
     * @brief Function which computes one step of update.
     */
    std::tuple<VectorXd, SpMat> compute_gradients(std::optional<double>step_size=std::nullopt) override;


    std::tuple<double, VectorXd, SpMat> onestep_linesearch(const double &step_size, const VectorXd& dmu, const SpMat& dprecision) override;

    inline void update_proposal(const VectorXd& new_mu, const SpMat& new_precision) override;

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
    

    inline VectorXd Vdmu() const {return _Vdmu; }

    inline SpMat Vddmu() const { return _Vddmu; }
    

}; //class


} //namespace gvi

// function implementations

#include "NGD-GH-impl.h"

#endif //NGD-GH
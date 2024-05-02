/**
 * @file ProxGVI-GH.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief Proximal Gaussian VI using GH quadratures.
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#ifndef PROXGVI_GH_H
#define PROXGVI_GH_H

#include <utility>
#include <memory>

#include "gvibase/GVI-GH.h"

using namespace Eigen;
namespace gvi{

template <typename FactorizedOptimizer>
class ProxGVIGH: public GVIGH<FactorizedOptimizer>{
    using Base = GVIGH<FactorizedOptimizer>;
public:

    ProxGVIGH(){}

    /**
     * @brief Construct a new VIMPOptimizerGH object
     * 
     * @param _vec_fact_optimizers vector of marginal optimizers
     * @param niters number of iterations
     */
    ProxGVIGH(const std::vector<std::shared_ptr<FactorizedOptimizer>>& vec_fact_optimizers,
            int dim_state,
            int num_states,
            int niterations = 5,
            double temperature = 1.0,
            double high_temperature = 100.0) :
            GVIGH<FactorizedOptimizer>(vec_fact_optimizers, dim_state, num_states, niterations, temperature, high_temperature),
            _dmu(VectorXd::Zero(Base::_dim)),
            _dprecision(SpMat(Base::_dim, Base::_dim))
        {
            _dmu.setZero();
            _dprecision.setZero();
        }

protected:
    VectorXd _dmu;
    SpMat _dprecision;    

public:
/// ************************* Override functions for NGD algorithm *************************************
/// Optimizations related
    /**
     * @brief Function which computes one step of update.
     */
    std::tuple<VectorXd, SpMat> compute_gradients() override{
        _dmu.setZero();
        _dprecision.setZero();

        for (auto &opt_k : Base::_vec_factors)
        {
            opt_k->calculate_partial_V();
            _dmu = _dmu + opt_k->local2joint_dmu();
            _dprecision = _dprecision + opt_k->local2joint_dprecision();
        }

        return std::make_tuple(_dmu, _dprecision);
    }

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

    inline VectorXd dmu() const {return _dmu; }

    inline SpMat dprecision() const { return _dprecision; }

}; //class

} //namespace gvi

#include "ProxGVI-GH-impl.h"

#endif //PROXGVI_GH_H
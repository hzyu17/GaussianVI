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

#ifndef NGDFactorizedBaseGH_H
#define NGDFactorizedBaseGH_H

#include <type_traits>
#include "ngd/NGDFactorizedBase.h"
#include <memory>

using namespace Eigen;

namespace gvi{

struct NoneType {};

template <typename CostClass = NoneType>
class NGDFactorizedBaseGH: public NGDFactorizedBase{

    using NGDBase = NGDFactorizedBase;
    using Function = std::function<double(const VectorXd&, const CostClass &)>;

public:
    ///@param dimension The dimension of the state
    ///@param function Template function class which calculate the cost
    // NGDFactorizedBaseGH(const int& dimension, const Function& function, const CostClass& cost_class_, const MatrixXd& Pk_):
    
    NGDFactorizedBaseGH(int dimension, int state_dim, int num_states, int start_index, 
                          const Function& function, const CostClass& cost_class,
                            double temperature=1.0, double high_temperature=10.0):
                NGDBase(dimension, state_dim, num_states, start_index, temperature, high_temperature)
            {
                /// Override of the NGDBase classes.
                NGDBase::_func_phi = [this, function, cost_class](const VectorXd& x){return MatrixXd::Constant(1, 1, function(x, cost_class));};
                NGDBase::_func_Vmu = [this, function, cost_class](const VectorXd& x){return (x-NGDBase::_mu) * function(x, cost_class);};
                NGDBase::_func_Vmumu = [this, function, cost_class](const VectorXd& x){return MatrixXd{(x-NGDBase::_mu) * (x-NGDBase::_mu).transpose().eval() * function(x, cost_class)};};
                NGDBase::_gh = std::make_shared<GH>(GH{10, NGDBase::_dim, NGDBase::_mu, NGDBase::_covariance});
            }
public:
    typedef std::shared_ptr<NGDFactorizedBaseGH> shared_ptr;

};


}

#endif // NGDFactorizedBaseGH_H
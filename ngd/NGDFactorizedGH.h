/**
 * @file NGDFactorizedGH.h
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

#ifndef NGDFactorizedGH_H
#define NGDFactorizedGH_H

#include "gvibase/GVIFactorizedNonLinearBase.h"
#include <memory>

using namespace Eigen;

namespace gvi{

template <typename Function, typename CostClass>
class NGDFactorizedGH: public GVIFactorizedNonLinearBase{

    using OptBase = GVIFactorizedNonLinearBase;
    // using GHFunction = std::function<MatrixXd(const VectorXd&)>;

public:
    ///@param dimension The dimension of the state
    ///@param function Template function class which calculate the cost
    NGDFactorizedGH(const int& dimension, const Function& function, const CostClass& cost_class_, const MatrixXd& Pk_):
            OptBase(dimension, Pk_, dimension, 0, false)
            {
                /// Override of the base classes.
                OptBase::_func_phi = [this, function, cost_class_](const VectorXd& x){return MatrixXd{MatrixXd::Constant(1, 1, function(x, cost_class_))} / OptBase::_temperature;};
                OptBase::_func_Vmu = [this, function, cost_class_](const VectorXd& x){return (x-OptBase::_mu) * function(x, cost_class_) / OptBase::_temperature ;};
                OptBase::_func_Vmumu = [this, function, cost_class_](const VectorXd& x){return MatrixXd{(x-OptBase::_mu) * (x-OptBase::_mu).transpose().eval() * function(x, cost_class_)} / OptBase::_temperature;};
                
                OptBase::_func_phi_highT = [this](const VectorXd& x){ return OptBase::_func_phi(x) / OptBase::_high_temperature; }
                OptBase::_func_Vmu_highT = [this](const VectorXd& x){ return OptBase::_func_Vmu(x) / OptBase::_high_temperature; }
                OptBase::_func_Vmumu_highT = [this](const VectorXd& x){ return OptBase::_func_Vmumu(x) / OptBase::_high_temperature; }
                OptBase::_gh = std::make_shared<GH>(new GH{6, OptBase::_dim, OptBase::_mu, OptBase::_covariance});
            }
public:
    typedef std::shared_ptr<NGDFactorizedGH> shared_ptr;

};


}

#endif // NGDFactorizedGH_H
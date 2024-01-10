/**
 * @file cost_functions.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief Two cost functions related to gp.
 * @version 0.1
 * @date 2024-01-10
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "gp/fixed_prior.h"
#include "gp/minimum_acc_prior.h"

namespace gvi{

/**
 * @brief Fixed cost with a covariance
 * @param x input
 * @param fixed_gp 
 * @return double cost
 */
double cost_fixed_gp(const VectorXd& x, const gvi::FixedPriorGP& fixed_gp){
    return fixed_gp.fixed_factor_cost(x);
}


/**
 * @brief cost for the linear gaussian process.
 * @param pose : the combined two secutive states. [x1; v1; x2; v2]
 * @param gp_minacc : the linear gp object
 * @return double, the cost (-logporb)
 */
double cost_linear_gp(const VectorXd& pose_cmb, const gvi::MinimumAccGP& gp_minacc){
    int dim = gp_minacc.dim_posvel();
    return gp_minacc.cost(pose_cmb.segment(0, dim), pose_cmb.segment(dim, dim));
}

}
/**
 * @file fixed_prior.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief Fixed Gaussian prior -log(p(x|z)) = ||A*x - B*\mu_t||_{K^{-1}}, 
 * with A = I, B = I.
 * @version 0.1
 * @date 2022-07-31
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include "linear_factor.h"

namespace gvi{

class FixedPriorGP : public LinearFactor{
    public:
        FixedPriorGP(){}
        FixedPriorGP(const MatrixXd& Covariance, const VectorXd& mu):
        _mu{mu}, 
        _dim{mu.size()}, 
        _K{Covariance}, 
        _invK{Covariance.inverse()}{}

        double fixed_factor_cost(const VectorXd& x) const{ 
            return (x-_mu).transpose() * _invK * (x-_mu); 
        }

        VectorXd get_mu() const { return _mu; }

        MatrixXd get_precision() const{ return _invK; }

        MatrixXd get_covariance() const{ return _K; }

        MatrixXd get_Lambda() const { return MatrixXd::Identity(_dim, _dim);}

        MatrixXd get_Psi() const { return MatrixXd::Identity(_dim, _dim);}

        inline double get_Constant() const { return 1.0; }

    private:
        MatrixXd _K;
        MatrixXd _invK;
        int _dim;
        VectorXd _mu;

};


}




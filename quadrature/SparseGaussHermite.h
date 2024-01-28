/**
 * @file SparseGaussHermite.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief Sparse Gauss-Hermite approximation of integrations implemented as tabulated form.
 * @version 0.1
 * @date 2024-01-26
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#pragma once

#include "quadrature/generateSpGHWeights.h"

using namespace Eigen;

namespace gvi{
template <typename Function>
class SparseGaussHermite{
public:
    /**
     * @brief Default constructor.
     */
    SparseGaussHermite(){}

    /**
     * @brief Constructor
     * 
     * @param deg degree of GH polynomial
     * @param dim dimension of the integrand input
     * @param mean mean 
     * @param P covariance matrix
     * @param func the integrand function
     */
    SparseGaussHermite(
        const int& deg, 
        const int& dim, 
        const VectorXd& mean, 
        const MatrixXd& P,
        const Function& func): 
            _deg(deg),
            _dim(dim),
            _mean(mean),
            _P(P),
            _f(func){  
                computeSigmaPtsWeights();
            }

    /**
     * @brief Compute the Sigma Pts
     */
    void computeSigmaPtsWeights(){
        if (!mclInitializeApplication(nullptr, 0)) {
            std::cerr << "Could not initialize the application." << std::endl;
            return ;
        }
        double d_dim = _dim;
        double d_deg = _deg;

        if (!libSpGHInitialize()) {
            std::cerr << "Could not initialize the library properly" << std::endl;
            
        } else {
            PointsWeightsTuple pts_weights = sigmapts_weights(d_dim, d_deg);
            _sigmapts = std::get<0>(pts_weights);
            _Weights = std::get<1>(pts_weights);

            // Call the application and library termination routine
            libSpGHTerminate();
        }

        mclTerminateApplication();

        return ;
    }

    /**
     * @brief Compute the approximated integration using Gauss-Hermite.
     */
    MatrixXd Integrate(const Function& function){
        // compute matrix sqrt of P
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(_P);
        MatrixXd sqrtP{es.operatorSqrt()};

        VectorXd pts_shifted{(_sigmapts*sqrtP).rowwise() + _mean.transpose()};       

        MatrixXd res(1, 1);
        double integration = 0.0;
        for (int i=0; i<pts_shifted.rows(); i++){
            VectorXd pt_i = pts_shifted.row(i).transpose();
            integration += function(pt_i)(0,0)*_Weights(i);
        }
        res(0,0) = integration;

        return res;
        
    };

    void update_integrand(const Function& function){
        _f = function;
    };

    MatrixXd Integrate(){
        return Integrate(_f);
    };

    /**
     * Update member variables
     * */
    inline void update_mean(const VectorXd& mean){ _mean = mean; }

    inline void update_P(const MatrixXd& P){ _P = P; }

    inline void set_polynomial_deg(const int& deg){ _deg = deg; }

    inline void update_dimension(const int& dim){ _dim = dim; }


    inline VectorXd weights() const { return this->_W; }
    inline VectorXd sigmapts() const { return this->_sigmapts; }

protected:
    int _deg;
    int _dim;
    VectorXd _mean;
    MatrixXd _P;
    VectorXd _Weights;
    VectorXd _sigmapts;
    Function _f;
};


} // namespace gvi
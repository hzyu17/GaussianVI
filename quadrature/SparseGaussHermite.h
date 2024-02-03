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
        const MatrixXd& P): 
            _deg(deg),
            _dim(dim),
            _mean(mean),
            _P(P)
            {  
                computeSigmaPtsWeights();
            }

    /**
     * @brief Compute the Sigma Pts
     */
    void computeSigmaPtsWeights(){
        
        // if (!mclInitializeApplication(nullptr, 0)) {
        //     std::cerr << "Could not initialize the application." << std::endl;
        //     return ;
        // }

        double d_dim = _dim;
        double d_deg = _deg;

        if (!libSpGHInitialize()) {
            std::cerr << "Could not initialize the library properly" << std::endl;
            
        } else {
            PointsWeightsTuple pts_weights = sigmapts_weights(d_dim, d_deg);

            _zeromeanpts = std::get<0>(pts_weights);
            _Weights = std::get<1>(pts_weights);

            // compute matrix sqrt of P
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(_P);
            _sqrtP = es.operatorSqrt();

            _sigmapts = (_zeromeanpts*_sqrtP).rowwise() + _mean.transpose(); 

            // Call the application and library termination routine
            libSpGHTerminate();
        }

        // mclTerminateApplication();

        return ;
    }

    /**
     * @brief Compute the approximated integration using Gauss-Hermite.
     */
    MatrixXd Integrate(const Function& function){
              
        MatrixXd res{function(_mean)};
        res.setZero();

        for (int i=0; i<_sigmapts.rows(); i++){
            VectorXd pt_i{_sigmapts.row(i).transpose()};
            
            res += function(pt_i)*_Weights(i);

        }
        
        return res;
        
    };

    /**
     * Update member variables
     * */
    inline void update_mean(const VectorXd& mean){ 
        _mean = mean; 
        _sigmapts = (_zeromeanpts*_sqrtP).rowwise() + _mean.transpose(); 
    }

    inline void update_P(const MatrixXd& P){ 
        _P = P; 
        // compute matrix sqrt of P
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(_P);
        _sqrtP = es.operatorSqrt();
        _sigmapts = (_zeromeanpts*_sqrtP).rowwise() + _mean.transpose(); 
    }

    inline void set_polynomial_deg(const int& deg){ 
        _deg = deg; 
        computeSigmaPtsWeights();
    }

    inline void update_dimension(const int& dim){ 
        _dim = dim; 
        computeSigmaPtsWeights();
    }

    inline void update_parameters(const int& deg, const int& dim, const VectorXd& mean, const MatrixXd& P){ 
        _deg = deg;
        _dim = dim;
        _mean = mean;
        _P = P;
        computeSigmaPtsWeights();
    }


    inline VectorXd weights() const { return this->_W; }
    inline MatrixXd sigmapts() const { return this->_sigmapts; }

protected:
    int _deg;
    int _dim;
    VectorXd _mean;
    MatrixXd _P, _sqrtP;
    VectorXd _Weights;
    MatrixXd _sigmapts, _zeromeanpts;
};


} // namespace gvi
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

#include "quadrature/SparseGHQuadratureWeights.h"

#define STRING(x) #x
#define XSTRING(x) STRING(x)
std::string source_root{XSTRING(SOURCE_ROOT)};

namespace gvi{
template <typename Function>
class SparseGaussHermite{
public:

    /**
     * @brief Constructor
     * 
     * @param deg degree of GH polynomial
     * @param dim dimension of the integrand input
     * @param mean mean 
     * @param P covariance matrix
     */
    SparseGaussHermite(
        const int& deg, 
        const int& dim, 
        const Eigen::VectorXd& mean, 
        const Eigen::MatrixXd& P): 
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
        
        std::ifstream ifs(source_root+"/quadrature/SparseGHQuadratureWeights.bin", std::ios::binary);
        boost::archive::binary_iarchive ia(ifs);
        ia >> _nodes_weights_map;

        DimDegTuple dim_deg;
        dim_deg = std::make_tuple(_dim, _deg);;

        PointsWeightsTuple pts_weights;
        pts_weights = _nodes_weights_map[dim_deg];

        _zeromeanpts = std::get<0>(pts_weights);
        _Weights = std::get<1>(pts_weights);

        // compute matrix sqrt of P
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(_P);
        _sqrtP = es.operatorSqrt();

        _sigmapts = (_zeromeanpts*_sqrtP).rowwise() + _mean.transpose(); 

        return ;
    }

    /**
     * @brief Compute the approximated integration using Gauss-Hermite.
     */
    Eigen::MatrixXd Integrate(const Function& function){
              
        Eigen::MatrixXd res{function(_mean)};
        res.setZero();

        std::cout << "_sigmapts " << std::endl << _sigmapts << std::endl;
        std::cout << "_Weights " << std::endl << _Weights << std::endl;
        
        Eigen::VectorXd pt(_dim);

        for (int i=0; i<_sigmapts.rows(); i++){
            
            pt = _sigmapts.row(i);
            res += function(pt)*_Weights(i);
            
        }
        
        return res;
        
    };

    /**
     * Update member variables
     * */
    inline void update_mean(const Eigen::VectorXd& mean){ 
        _mean = mean; 
        _sigmapts = (_zeromeanpts*_sqrtP).rowwise() + _mean.transpose(); 
    }

    inline void update_P(const Eigen::MatrixXd& P){ 
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

    inline void update_parameters(const int& deg, const int& dim, const Eigen::VectorXd& mean, const Eigen::MatrixXd& P){ 
        _deg = deg;
        _dim = dim;
        _mean = mean;
        _P = P;
        computeSigmaPtsWeights();
    }


    inline Eigen::VectorXd weights() const { return this->_W; }
    inline Eigen::MatrixXd sigmapts() const { return this->_sigmapts; }

protected:
    int _deg;
    int _dim;
    Eigen::VectorXd _mean;
    Eigen::MatrixXd _P, _sqrtP;
    Eigen::VectorXd _Weights;
    Eigen::MatrixXd _sigmapts, _zeromeanpts;

    QuadratureWeightsMap _nodes_weights_map;
    
};


} // namespace gvi
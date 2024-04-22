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
#include "helpers/CommonDefinitions.h"

#ifdef GVI_SUBDUR_ENV
std::string map_file{source_root+"/GaussianVI/quadrature/SparseGHQuadratureWeights.bin"};
#else
std::string map_file{source_root+"/quadrature/SparseGHQuadratureWeights.bin"};
#endif

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
                try {
                    std::ifstream ifs(map_file, std::ios::binary);
                    if (!ifs.is_open()) {
                        throw std::runtime_error("Failed to open file for GH weights reading");
                    }

                    boost::archive::binary_iarchive ia(ifs);
                    ia >> _nodes_weights_map;

                } catch (const boost::archive::archive_exception& e) {
                    std::cerr << "Boost archive exception: " << e.what() << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Standard exception: " << e.what() << std::endl;
                }

                computeSigmaPtsWeights();
            }

    SparseGaussHermite(
        const int& deg, 
        const int& dim, 
        const Eigen::VectorXd& mean, 
        const Eigen::MatrixXd& P,
        const QuadratureWeightsMap& weights_map): 
            _deg(deg),
            _dim(dim),
            _mean(mean),
            _P(P)
            {  
                computeSigmaPtsWeights(weights_map);
            }
            

    /**
     * @brief Compute the Sigma Pts
     */
    void computeSigmaPtsWeights(){
        
        DimDegTuple dim_deg;
        dim_deg = std::make_tuple(_dim, _deg);;

        PointsWeightsTuple pts_weights;
        if (_nodes_weights_map.count(dim_deg) > 0) {
            pts_weights = _nodes_weights_map[dim_deg];

            _zeromeanpts = std::get<0>(pts_weights);
            _Weights = std::get<1>(pts_weights);
            
            // Eigenvalue decomposition
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(_P);
            if (eigensolver.info() != Eigen::Success) {
                std::cerr << "Eigenvalue decomposition failed!" << std::endl;
                return;
            }

            // Compute square roots of the eigenvalues
            Eigen::MatrixXd D = eigensolver.eigenvalues().asDiagonal();
            Eigen::MatrixXd sqrtD = D.unaryExpr([](double elem) { return std::sqrt(std::max(0.0, elem)); });

            // Compute the square root of the matrix
            Eigen::MatrixXd _sqrtP = eigensolver.eigenvectors() * sqrtD * eigensolver.eigenvectors().transpose();

            _sigmapts = (_zeromeanpts*_sqrtP.transpose()).rowwise() + _mean.transpose(); 

        } else {
            std::cout << "(dimension, degree) " << "(" << _dim << ", " << _deg << ") " <<
             "key does not exist in the GH weight map." << std::endl;
        }
        

        return ;
    }

    /**
     * @brief Compute the Sigma Pts
     */
    void computeSigmaPtsWeights(const QuadratureWeightsMap& weights_map){
        
        DimDegTuple dim_deg{std::make_tuple(_dim, _deg)};

        PointsWeightsTuple pts_weights;
        if (weights_map.count(dim_deg) > 0) {
            std::cout << "(dimension, degree) tuple: " << "(" << _dim << ", " << _deg << ") " <<
             "exists in the GH weight map." << std::endl;
            
            pts_weights = weights_map.at(dim_deg);

            _zeromeanpts = std::get<0>(pts_weights);
            _Weights = std::get<1>(pts_weights);

            // compute matrix sqrt of P
            Eigen::LLT<MatrixXd> lltP(_P);
            _sqrtP = lltP.matrixL();

            // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(_P);
            // _sqrtP = es.operatorSqrt();

            _sigmapts = (_zeromeanpts*_sqrtP).rowwise() + _mean.transpose(); 
        } else {
            std::cout << "(dimension, degree) " << "(" << _dim << ", " << _deg << ") " <<
             "key does not exist in the GH weight map." << std::endl;
        }
        
        return ;
    }

    /**
     * @brief Compute the approximated integration using Gauss-Hermite.
     */
    Eigen::MatrixXd Integrate(const Function& function){
        
        Eigen::MatrixXd res{function(_mean)};
        res.setZero();
        
        Eigen::VectorXd pt(_dim);

        for (int i=0; i<_sigmapts.rows(); i++){
            
            pt = _sigmapts.row(i);
            res += function(pt)*_Weights(i);

        }
        // std::cout << "========== Integration time" << std::endl;
        // timer.end_mus();
        
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
        Eigen::LLT<MatrixXd> lltP(_P);
        _sqrtP = lltP.matrixL();
        
        // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(_P);
        // _sqrtP = es.operatorSqrt();

        if (_sqrtP.hasNaN()) {
            std::cerr << "Error: sqrt Covariance matrix contains NaN values." << std::endl;
            // Handle the situation where _sqrtP contains NaN values
        }

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

        // Timer timer;
        // timer.start();
        computeSigmaPtsWeights();

        // std::cout << "========== Compute weight time" << std::endl;
        // timer.end_mus();
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
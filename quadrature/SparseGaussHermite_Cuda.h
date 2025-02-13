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

#ifndef SPARSEGAUSSHERMITE_H
#define SPARSEGAUSSHERMITE_H

#include <optional>
#include "quadrature/SparseGHQuadratureWeights.h"
#include "helpers/CommonDefinitions.h"
#include "cuda_runtime.h"

namespace gvi{
template <typename Function>
class SparseGaussHermite_Cuda{
    using GHFunction = std::function<MatrixXd(const VectorXd&)>;

public:

    /**
     * @brief Constructor
     * 
     * @param deg degree of GH polynomial
     * @param dim dimension of the integrand input
     * @param mean mean 
     * @param P covariance matrix
     */

    SparseGaussHermite_Cuda(
        const int& deg, 
        const int& dim, 
        const Eigen::VectorXd& mean, 
        const Eigen::MatrixXd& P): 
            _deg(deg),
            _dim(dim),
            _mean(mean),
            _P(P)
            {  
                std::string map_file_local{source_root+"/GaussianVI/quadrature/SparseGHQuadratureWeights_cereal.bin"};
                // Read map from file
                QuadratureWeightsMap nodes_weights_map;
                try {
                    std::ifstream ifs(map_file_local, std::ios::binary);
                    if (!ifs.is_open()) {
                        std::string error_msg = "Failed to open file for GH weights reading in file: " + map_file_local;
                        throw std::runtime_error(error_msg);
                    }

                    // Use cereal for deserialization
                    cereal::BinaryInputArchive archive(ifs);
                    archive(nodes_weights_map);

                } catch (const std::exception& e) {
                    std::cerr << "Standard exception: " << e.what() << std::endl;
                }

                _nodes_weights_map = std::make_shared<QuadratureWeightsMap>(nodes_weights_map);
                
                computeSigmaPtsWeights();
            }


    SparseGaussHermite_Cuda(
        const int& deg, 
        const int& dim, 
        const Eigen::VectorXd& mean, 
        const Eigen::MatrixXd& P,
        const std::shared_ptr<QuadratureWeightsMap>& weights_map_pointer): 
            _deg(deg),
            _dim(dim),
            _mean(mean),
            _P(P)
            {  
                _nodes_weights_map = weights_map_pointer;
                computeSigmaPtsWeights();
            }
    
    SparseGaussHermite_Cuda(
        const int& deg, 
        const int& dim, 
        const std::shared_ptr<QuadratureWeightsMap>& weights_map_pointer): 
            _deg(deg),
            _dim(dim)
            {  
                _nodes_weights_map = weights_map_pointer;
                computeWeights();
            }
            

    /**
     * @brief Compute the Sigma Pts
     */
    void computeSigmaPtsWeights(){
        
        DimDegTuple dim_deg;
        dim_deg = std::make_tuple(_dim, _deg);;

        PointsWeightsTuple pts_weights;
        if (_nodes_weights_map->count(dim_deg) > 0) {
            pts_weights = _nodes_weights_map->at(dim_deg);

            _zeromeanpts = std::get<0>(pts_weights);
            _Weights = std::get<1>(pts_weights);
            
            // Eigenvalue decomposition
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(_P);
            if (eigensolver.info() != Eigen::Success) {
                std::cerr << "Eigenvalue decomposition failed!" << std::endl;
                return;
            }

            update_sigmapoints();

        } else {
            std::cout << "(dimension, degree) " << "(" << _dim << ", " << _deg << ") " <<
             "key does not exist in the GH weight map." << std::endl;
        }

        
        return ;
    }


    /**
     * @brief Compute Weights without computing SigmaPts
     */
    void computeWeights(){
        
        DimDegTuple dim_deg;
        dim_deg = std::make_tuple(_dim, _deg);;

        PointsWeightsTuple pts_weights;
        if (_nodes_weights_map->count(dim_deg) > 0) {
            pts_weights = _nodes_weights_map->at(dim_deg);

            _zeromeanpts = std::get<0>(pts_weights);
            _Weights = std::get<1>(pts_weights);

        }
        else {
            std::cout << "(dimension, degree) " << "(" << _dim << ", " << _deg << ") " <<
             "key does not exist in the GH weight map." << std::endl;
        }

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

            update_sigmapoints();
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

        #pragma omp parallel
        {
            // Create a private copy of the res matrix for each thread
            Eigen::MatrixXd private_res = Eigen::MatrixXd::Zero(res.rows(), res.cols());
            Eigen::VectorXd pt(_dim);

            #pragma omp for nowait  // The 'nowait' clause can be used if there is no need for synchronization after the loop
            for (int i = 0; i < _sigmapts.rows(); i++) {
                pt = _sigmapts.row(i); // Row of the matrix
                private_res += function(pt) * _Weights(i);
            }

            // Use a critical section to sum up results from all threads
            #pragma omp critical
            res += private_res;
        }
        return res;
    };


    /**
     * Update member variables
     * */
    inline void update_mean(const Eigen::VectorXd& mean){ 
        _mean = mean; 
        
    }

    inline void update_sigmapoints(){
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(_P);
        _sqrtP = es.operatorSqrt();

        if (_sqrtP.hasNaN()) {
            Eigen::VectorXd eigenvalues = es.eigenvalues();
            std::cout << "eigenvalues" << std::endl << eigenvalues << std::endl;
            std::cerr << "Error: sqrt Covariance matrix contains NaN values." << std::endl;
            // Handle the situation where _sqrtP contains NaN values
        }

        _sigmapts = (_zeromeanpts*_sqrtP.transpose()).rowwise() + _mean.transpose(); 
    }

    inline void update_P(const Eigen::MatrixXd& P){ 
        _P = P;         
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

    inline Eigen::VectorXd weights() const { return this->_Weights; }

    inline Eigen::MatrixXd zeromeanpts() const { return this->_zeromeanpts; }

    inline Eigen::MatrixXd sigmapts() const { return this->_sigmapts; }

    inline Eigen::MatrixXd mean() const { return this->_mean; }

    inline Eigen::MatrixXd covariance() const { return this->_P; }

    Function _function;

protected:
    int _deg;
    int _dim;
    Eigen::VectorXd _mean;
    Eigen::MatrixXd _P, _sqrtP;
    Eigen::VectorXd _Weights;
    Eigen::MatrixXd _sigmapts, _zeromeanpts;

    std::shared_ptr<QuadratureWeightsMap> _nodes_weights_map;
    
};


} // namespace gvi

#endif

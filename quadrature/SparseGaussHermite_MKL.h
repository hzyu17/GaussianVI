/**
 * @file SparseGaussHermite_MKL.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief Sparse Gauss-Hermite approximation of integrations implemented as tabulated form.
 * @version 0.1
 * @date 2025-02-02
 * 
 * @copyright Copyright (c) 2025
 * 
 */


#pragma once

#include <optional>
#include "quadrature/SparseGHQuadratureWeights.h"
#include "helpers/CommonDefinitions.h"
#include "src_MKL/gvimp_mkl.h"
#include <omp.h>
#include "helpers/timer.h"


// #ifdef GVI_SUBDUR_ENV 
// std::string map_file{source_root+"/GaussianVI/quadrature/SparseGHQuadratureWeights_cereal.bin"};
// #else
// std::string map_file{source_root+"/quadrature/SparseGHQuadratureWeights_cereal.bin"};
// #endif

namespace gvi{

template <typename Function_MKL>
class SparseGaussHermite_MKL{
public:

    using FunctionType = std::function<std::vector<double>(const std::vector<double>&)>;

    virtual ~SparseGaussHermite_MKL(){}

    /**
     * @brief Constructor
     * 
     * @param deg degree of GH polynomial
     * @param dim dimension of the integrand input
     * @param mean mean 
     * @param P covariance matrix
     */
    SparseGaussHermite_MKL(
        int deg, 
        int dim, 
        const std::vector<double>& mean, 
        const std::vector<double>& P,
        std::optional<QuadratureWeightsMap> weight_sigpts_map_option=std::nullopt): 
            _deg(deg),
            _dim(dim),
            _mean(mean),
            _P(P){  
                // If input has a loaded map
                if (weight_sigpts_map_option.has_value()){
                    _nodes_weights_map = std::make_shared<QuadratureWeightsMap_MKL>(weight_sigpts_map_option.value());
                }
                // Read map from file
                else{
                    QuadratureWeightsMap_MKL nodes_weights_map;
                    try {
                        std::ifstream ifs(map_file, std::ios::binary);
                        if (!ifs.is_open()) {
                            std::string error_msg = "Failed to open file for GH weights reading in file: " + map_file;
                            throw std::runtime_error(error_msg);
                        }

                        // Use cereal for deserialization
                        cereal::BinaryInputArchive archive(ifs);
                        archive(nodes_weights_map);

                    } catch (const std::exception& e) {
                        std::cerr << "Standard exception: " << e.what() << std::endl;
                    }

                    _nodes_weights_map = std::make_shared<QuadratureWeightsMap_MKL>(nodes_weights_map);

                }
                
                computeSigmaPtsWeights();
            }

    SparseGaussHermite_MKL(
        const int& deg, 
        const int& dim, 
        const std::vector<double>& mean, 
        const std::vector<double>& P,
        std::optional<std::shared_ptr<QuadratureWeightsMap_MKL>> weight_sigpts_map_option=std::nullopt): 
            _deg(deg),
            _dim(dim),
            _mean(mean),
            _P(P)
            {  
                for (int ii=0; ii<dim; ii++){
                    _P[ii*dim+ii] = P[ii*dim+ii];
                    _mean[ii] = mean[ii];
                }

                if (weight_sigpts_map_option.has_value()){
                    std::cout << "weight_sigpts_map_option has value. Taking into the SparseGH class" << std::endl;
                    _nodes_weights_map = weight_sigpts_map_option.value();
                }
                // Read map from file
                else{
                    QuadratureWeightsMap_MKL nodes_weights_map;
                    try {

                        auto buffer = readFileToBuffer(map_mkl_file);
                        std::istringstream iss(std::string(buffer.data(), buffer.size()));
                        cereal::BinaryInputArchive archive(iss);
                        
                        archive(nodes_weights_map);

                    } catch (const std::exception& e) {
                        std::cerr << "Standard exception: " << e.what() << std::endl;
                    }

                    _nodes_weights_map = std::make_shared<QuadratureWeightsMap_MKL>(nodes_weights_map);

                }
                
                computeSigmaPtsWeights();
            }


    SparseGaussHermite_MKL(
        const int& deg, 
        const int& dim, 
        const std::vector<double>& mean, 
        const std::vector<double>& P,
        const QuadratureWeightsMap_MKL& weights_map): 
            _deg(deg),
            _dim(dim),
            _mean(mean),
            _P(P)
            {  
                for (int ii=0; ii<dim; ii++){
                    _P[ii*dim+ii] = P[ii*dim+ii];
                    _mean[ii] = mean[ii];
                }
                computeSigmaPtsWeights(weights_map);
            }
            

    /**
     * @brief Compute the Sigma Pts
     */
    void computeSigmaPtsWeights(){
        DimDegTuple dim_deg;
        dim_deg = std::make_tuple(_dim, _deg);;

        PointsWeightsTuple_MKL pts_weights;
        if (_nodes_weights_map->count(dim_deg) > 0) {
            pts_weights = _nodes_weights_map->at(dim_deg);

            _zeromeanpts = std::get<0>(pts_weights);
            _Weights = std::get<1>(pts_weights);

            _num_sigmapoints = _zeromeanpts.size() / _dim;
            
            update_sigmapoints();

        } else {
            std::cout << "(dimension, degree) " << "(" << _dim << ", " << _deg << ") " <<
             "key does not exist in the GH weight map." << std::endl;
        }

        // std::cout << "Finished the computing of sigma points and weights." << std::endl;

    }

    /**
     * @brief Compute the Sigma Pts
     */
    void computeSigmaPtsWeights(const QuadratureWeightsMap_MKL& weights_map){
        
        DimDegTuple dim_deg{std::make_tuple(_dim, _deg)};

        PointsWeightsTuple_MKL pts_weights;
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
    std::vector<double> Integrate(const Function_MKL& function, 
                                    const int& output_rows, 
                                    const int& output_cols){
       
        // std::cout << "Starting Integration... " << std::endl; 
        int size = output_rows*output_cols;
        std::vector<double> integration_result(size, 0.0);
        
        std::vector<double> pt(_dim, 0.0);
        std::vector<double> res_i(size, 0.0);
        double weight_i(0.0);

        for (int i = 0; i < _num_sigmapoints; i++) {
            
            get_row_i(_sigmapts, i, _dim, pt);
            
            weight_i = _Weights[i];

            res_i = function(pt);
            
            for (size_t j = 0; j < size; ++j) {
                res_i[j] = res_i[j]*weight_i;
                integration_result[j] += res_i[j];
            }

        }

        return integration_result;
        
    }

    /**
     * Update member variables
     * */
    inline void update_mean(const std::vector<double>& mean){ 
        _mean = mean; 
        
    }

    inline void update_sigmapoints(){
        // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(_P);
        // _sqrtP = es.operatorSqrt();

        std::vector<double> result(_dim * _dim, 0.0);
        _sqrtP = result;

        SqrtEigenSolverMKL(_P, _sqrtP, _dim);

        // _sigmapts = (_zeromeanpts*_sqrtP.transpose()).rowwise() + _mean.transpose(); 

        std::vector<double> temp(_num_sigmapoints*_dim, 0.0);
        _sigmapts = temp;

        AMultiplyB(_zeromeanpts, _sqrtP, _sigmapts, _num_sigmapoints, _dim, _dim);

        AddTransposeToRows(_sigmapts, _mean, _num_sigmapoints, _dim);

    }

    std::vector<char> readFileToBuffer(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
        if (!ifs.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        std::streamsize fileSize = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        
        std::vector<char> buffer(fileSize);
        if (!ifs.read(buffer.data(), fileSize)) {
            throw std::runtime_error("Failed to read file: " + filename);
        }
        return buffer;
    }

    inline void update_P(const std::vector<double>& P){ 
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

    inline void update_parameters(const int& deg, const int& dim, const std::vector<double>& mean, const std::vector<double>& P){ 
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

    inline std::vector<double> weights() const { return this->_Weights; }

    inline std::vector<double> sigmapts() const { return this->_sigmapts; }

    inline std::vector<double> mean() const { return this->_mean; }

protected:
    int _deg;
    int _dim;
    int _num_sigmapoints;

    std::vector<double> _mean;
    std::vector<double> _P, _sqrtP;
    std::vector<double> _Weights;
    std::vector<double> _sigmapts, _zeromeanpts;

    std::shared_ptr<QuadratureWeightsMap_MKL> _nodes_weights_map;

    FunctionType m_function;
    
};


} // namespace gvi


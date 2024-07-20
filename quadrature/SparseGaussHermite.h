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

#include <optional>
#include <functional>
#include "quadrature/SparseGHQuadratureWeights.h"
#include "helpers/CommonDefinitions.h"
#include "helpers/CudaOperation.h"
#include "cuda_runtime.h"


#ifdef GVI_SUBDUR_ENV 
std::string map_file{source_root+"/GaussianVI/quadrature/SparseGHQuadratureWeights.bin"};
#else
std::string map_file{source_root+"/quadrature/SparseGHQuadratureWeights.bin"};
#endif

typedef void (*FunctionPtr)(double* input, double* output);

// extern "C" void CudaIntegration(FunctionPtr func_ptr, double* d_sigmapts, double* d_weights, double* d_results, int sigma_rows, int sigma_cols, int res_rows, int res_cols);

namespace gvi{
template <typename Function>
class SparseGaussHermite{
    
    using CudaFunction = std::function<void(double*, double*)>;
    using GHFunction = std::function<MatrixXd(const VectorXd&)>;
    // using CostFunction = std::function<double(const VectorXd&, const CostClass &)>;
    using Cuda = CudaOperation<GHFunction>;

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
        const Eigen::MatrixXd& P,
        std::optional<QuadratureWeightsMap> weight_sigpts_map_option=std::nullopt): 
            _deg(deg),
            _dim(dim),
            _mean(mean),
            _P(P)
            {  
                // If input has a loaded map
                if (weight_sigpts_map_option.has_value()){
                    _nodes_weights_map = std::make_shared<QuadratureWeightsMap>(weight_sigpts_map_option.value());
                }
                // Read map from file
                else{
                    QuadratureWeightsMap nodes_weights_map;
                    try {
                        std::ifstream ifs(map_file, std::ios::binary);
                        if (!ifs.is_open()) {
                            std::string error_msg = "Failed to open file for GH weights reading in file: " + map_file;
                            throw std::runtime_error(error_msg);
                        }

                        std::cout << "Opening file for GH weights reading in file: " << map_file << std::endl;
                        boost::archive::binary_iarchive ia(ifs);
                        ia >> nodes_weights_map;

                    } catch (const boost::archive::archive_exception& e) {
                        std::cerr << "Boost archive exception: " << e.what() << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "Standard exception: " << e.what() << std::endl;
                    }

                    _nodes_weights_map = std::make_shared<QuadratureWeightsMap>(nodes_weights_map);

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

    // static void functionWrapper(double* input, double* output, int size, void* context) {
    //     auto* self = static_cast<SparseGaussHermite*>(context);
    //     Eigen::Map<const Eigen::VectorXd> x_vector(input, size);
    //     std::cout << x_vector.transpose() << std::endl << std::endl;
    //     Eigen::MatrixXd result = self->global_function(x_vector);
    //     int rows = result.rows();
    //     int cols = result.cols();
    //     double* result_array = new double[result.size()];
    //     Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(output, rows, cols) = result;
    //     return;
    // }

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

    Eigen::MatrixXd Integrate_cuda(const Function& function, const int& type){
        
        Eigen::MatrixXd res{function(_mean)};
        res.setZero();
        _mean_func = res;

        update_function(function);
        // FunctionPtr funcPtr = static_cast<FunctionPtr>(func_cuda.target<void(double*, double*)>());
        // FunctionPtr funcPtr = *static_cast<void (**)(double*, double*)>(func_cuda.target<void(*)(double*, double*)>());
        FunctionPtr funcPtr;

        // std::cout << sizeof(*this) <<std::endl;

        global_function = function;

        // Calculate the result of functions (Try to integrate it in cuda)
        Eigen::MatrixXd pts(res.rows(), _sigmapts.rows()*res.cols());

        #pragma omp parallel
        {
            #pragma omp for nowait  // The 'nowait' clause can be used if there is no need for synchronization after the loop
           
            for (int i = 0; i < _sigmapts.rows(); i++) {
                pts.block(0, i * res.cols(), res.cols(), res.rows()) = function(_sigmapts.row(i)).transpose();
            }

        }

        double* sigmapts_array = new double[_sigmapts.size()];
        double* pts_array = new double[pts.size()];
        double* pts_array1 = new double[pts.size()];
        double* weight_array = new double[_Weights.size()];
        double* res_array = new double[res.size()];
        double* mu_array = new double[_mean.size()];

        // std::cout << "pts:" << std::endl << pts << std::endl << std::endl;
        // std::cout << "Weight:" << std::endl << _Weights.transpose() << std::endl << std::endl;

        // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(sigmapts_array, _sigmapts.rows(), _sigmapts.cols()) = _sigmapts;
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(pts_array, pts.rows(), pts.cols()) = pts;
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(weight_array, _Weights.rows(), _Weights.cols()) = _Weights;
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(mu_array, _mean.rows(), _mean.cols()) = _mean;
        
        // std::cout<< "Sigma:" << _sigmapts.transpose() << std::endl;


        // std::cout << "Sigma rows and cols" << std::endl << _sigmapts.rows() << _sigmapts.cols() << std::endl << std::endl;

        this -> _cuda -> CudaIntegration(function, _sigmapts, _Weights, res, _mean, _sigmapts.rows(), _sigmapts.cols(), res.rows(), res.cols(), pts.data(), pts_array1, type);

        // this -> _cuda -> CudaIntegration1(pts, _Weights, res, _sigmapts.rows(), _sigmapts.cols(), res.rows(), res.cols());

        // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> res_cuda(res_array, res.rows(), res.cols());
        
        // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> pts_cuda(pts_array1, pts.rows(), pts.cols());
        // std::cout << "pts:" << std::endl << pts <<std::endl;
        // std::cout << "pts_cuda:" << std::endl << pts_cuda <<std::endl;

        // res = res_cuda;
        
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

    inline void update_function(const Function& function){
        func_cuda = [this, function](double* input, double* output){
            std::cout << "Comming in the function" << std::endl;
            double* result_array = new double[_mean_func.size()];
            Eigen::Map<const Eigen::VectorXd> x_vector(input, this->_sigmapts.rows());
            // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> x_vector(non_const_x, res.rows(), res.cols());
            Eigen::MatrixXd result = function(x_vector);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(result_array, result.rows(), result.cols()) = result;
            for (int i = 0; i<_mean_func.size(); i++){
                output[i] = result_array[i];
            }
            // output =  result_array;
        };
    }


    inline Eigen::VectorXd weights() const { return this->_Weights; }

    inline Eigen::MatrixXd sigmapts() const { return this->_sigmapts; }

    // The function to use in kernel function
    // CudaFunction _func_cuda;

    Function global_function;
    std::shared_ptr<Cuda> _cuda;

protected:
    int _deg;
    int _dim;
    Eigen::VectorXd _mean;
    Eigen::MatrixXd _P, _sqrtP, _mean_func;
    Eigen::VectorXd _Weights;
    Eigen::MatrixXd _sigmapts, _zeromeanpts;
    
    CudaFunction func_cuda;

    std::shared_ptr<QuadratureWeightsMap> _nodes_weights_map;
    
};


} // namespace gvi


// 

// func_cuda = [this, function, res](const double* x){
//     double* result_array = new double[res.size()];
//     double* non_const_x = const_cast<double*>(x);
//     Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> x_vector(non_const_x, res.rows(), res.cols());
//     Eigen::MatrixXd result = function(x_vector);

//     int rows = result.rows();
//     int cols = result.cols();

//     Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(result_array, rows, cols) = result;

//     return std::make_tuple(result_array, rows, cols);
// };


// std::cout << "sigma:(" << _sigmapts.rows() << "," << _sigmapts.cols() << ")" << std::endl;
// std::cout << "weight:(" << _Weights.rows() << "," << _Weights.cols() << ")" << std::endl;
// std::cout << "res:(" << res.rows() << "," << res.cols() << ")" << std::endl;
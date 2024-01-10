/**
 * @file GVI-GH.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief The joint optimizer class using Gauss-Hermite quadrature, base class for different algorithms.
 * @version 1.1
 * @date 2024-01-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#ifndef GVI_GH_H
#define GVI_GH_H

#include <utility>
#include <memory>

#include "helpers/EigenWrapper.h"
#include "helpers/DataRecorder.h"

using namespace Eigen;

namespace gvi{

template <typename FactorizedOptimizer>
class GVIGH{
public:
    /**
     * @brief Default Constructor
     */
    GVIGH(){}

    /**
     * @brief Construct a new VIMPOptimizerGH object
     * 
     * @param _vec_fact_optimizers vector of marginal optimizers
     * @param niters number of iterations
     */
    GVIGH(const std::vector<std::shared_ptr<FactorizedOptimizer>>& vec_fact_optimizers, 
          int dim_state, 
          int num_states, 
          int niterations=5,
          double temperature=1.0, 
          double high_temperature=100.0):
            _dim_state{dim_state},
            _num_states{num_states},
            _dim{dim_state*num_states},
            _niters{niterations},
            _niters_lowtemp{10},
            _niters_backtrack{10},
            _stop_err{1e-5},
            _temperature{temperature},
            _high_temperature{high_temperature},
            _nfactors{vec_fact_optimizers.size()},
            _vec_factors{vec_fact_optimizers},
            _mu{VectorXd::Zero(_dim)},
            _precision{SpMat(_dim, _dim)},
            _covariance{SpMat(_dim, _dim)},
            _res_recorder{niterations, dim_state, num_states, _nfactors}
    {
        construct_sparse_precision();
    }

protected:
    /// optimization variables
    int _dim, _niters, _niters_lowtemp, _niters_backtrack, _nfactors, _dim_state, _num_states;

    double _temperature, _high_temperature, _initial_precision_factor, _boundary_penalties;

    double _stop_err;

    /// @param _vec_factors Vector of marginal optimizers
    std::vector<std::shared_ptr<FactorizedOptimizer>> _vec_factors;

    VectorXd _mu;

    /// Data and result storage
    VIMPResults _res_recorder;
    MatrixIO _matrix_io;

    // sparse matrices
    SpMat _precision, _covariance;
    EigenWrapper _ei;
    VectorXi _Rows, _Cols; VectorXd _Vals;
    int _nnz = 0;
    SparseLDLT _ldlt;
    SpMat _L; VectorXd _D, _Dinv; // for computing the determinant

    /// step sizes by default
    double _step_size = 0.9;
    double _step_size_base = 0.55;

    /// filename for the perturbed costs
    std::string _file_perturbed_cost;


public:

// ************************* Optimizations related functions *************************************
// ******** Common functions for all algorithms ********
    /*
    * @brief The main optimization loop with line search algorithm.
    */
    void optimize(std::optional<bool> verbose=std::nullopt);

    /**
     * @brief Compute the total cost function value given a mean and covariace.
     */
    double cost_value(const VectorXd &mean, SpMat &Precision);

    /**
     * @brief Compute the costs of all factors for a given mean and cov.
     */
    VectorXd factor_cost_vector(const VectorXd& x, SpMat& Precision);


// ******** Functions that differs in different algorithms ********
    /**
     * @brief Function which computes one step of update.
     */
    virtual std::tuple<VectorXd, SpMat> compute_gradients(){};

    // /**
    //  * @brief The optimizing process.
    //  */
    // virtual void optimize(std::optional<bool> verbose= std::nullopt);

    virtual std::tuple<double, VectorXd, SpMat> onestep_linesearch(const double &step_size, const VectorXd& dmu, const SpMat& dprecision){};

    virtual inline void update_proposal(const VectorXd& new_mu, const SpMat& new_precision){};

    virtual double cost_value(){};

    /**
     * @brief given a state, compute the total cost function value without the entropy term, using current values.
     */
    virtual double cost_value_no_entropy(){};

    /**
     * @brief Default computation of the cost vector.
     */
    virtual VectorXd factor_cost_vector(){};


/// **************************************************************
/// Internal data IO
    inline VectorXd mean() const{ return _mu; }

    inline SpMat covariance() const { return _covariance; }

    inline void inverse_inplace(){
        ldlt_decompose();

        _ei.inv_sparse(_precision, _covariance, _Rows, _Cols, _Vals, _Dinv);
    }

    inline SpMat inverse(const SpMat & mat){
        SpMat res(_dim, _dim);
        _ei.inv_sparse(mat, res, _Rows, _Cols, _nnz);
        return res;
    }

    /// update the step sizes
    inline void set_step_size(double step_size){ _step_size = step_size; }

    inline void set_stop_err(double stop_err) { _stop_err = stop_err; }

    /// The base step size in backtracking
    inline void set_step_size_base(double step_size_base){ _step_size_base = step_size_base; }

    inline void set_max_iter_backtrack(double max_backtrack_iter){ _niters_backtrack = max_backtrack_iter; }

    inline void set_mu(const VectorXd& mean){
        _mu = mean; 
        for (std::shared_ptr<FactorizedOptimizer> & opt_fact : _vec_factors){
            opt_fact->update_mu_from_joint(_mu);
        }
    }

    inline void set_initial_precision_factor(double initial_precision_factor){
        _initial_precision_factor = initial_precision_factor;
    }

    inline void initilize_precision_matrix(){
        initilize_precision_matrix(_initial_precision_factor);
    }

    inline void initilize_precision_matrix(double initial_precision_factor){
        // boundaries
        set_initial_precision_factor(initial_precision_factor);

        MatrixXd init_precision(_dim, _dim);
        init_precision = MatrixXd::Identity(_dim, _dim)*initial_precision_factor;
        
        set_precision(init_precision.sparseView());
    }

    inline void set_precision(const SpMat& new_precision);

    inline void set_initial_values(const VectorXd& init_mean, const SpMat& init_precision){
        set_mu(init_mean);
        set_precision(init_precision);
    }

    void construct_sparse_precision(){
        _precision.setZero();
        // fill in the precision matrix to the known sparsity pattern
        if (_num_states == 1){
            Eigen::MatrixXd block = MatrixXd::Ones(_dim_state, _dim_state);
            _ei.block_insert_sparse(_precision, 0, 0, _dim_state, _dim_state, block);
        }else{
            Eigen::MatrixXd block = MatrixXd::Ones(2*_dim_state, 2*_dim_state);
            for (int i=0; i<_num_states-1; i++){
                _ei.block_insert_sparse(_precision, i*_dim_state, i*_dim_state, 2*_dim_state, 2*_dim_state, block);
            }

        }
        
        SpMat lower = _precision.triangularView<Eigen::Lower>();
        _nnz = _ei.find_nnz(lower, _Rows, _Cols, _Vals); // the Rows and Cols table are fixed since the initialization.
    }

    inline void set_GH_degree(const int deg){
        for (auto & opt_fact : _vec_factors){
            opt_fact->set_GH_points(deg);
        }
    }

    inline void set_niter_low_temperature(int iters_low_temp){
        _niters_lowtemp = iters_low_temp;
    }

    inline void set_temperature(double temperature){
        _temperature = temperature;
    }

    inline void set_high_temperature(double high_temp){
        _high_temperature = high_temp;
    }

    void ldlt_decompose(){
        _ldlt.compute(_precision);
        _L = _ldlt.matrixL();
        _Dinv = _ldlt.vectorD().real().cwiseInverse();
        _ei.find_nnz_known_ij(_L, _Rows, _Cols, _Vals);
        // _D = ldlt.vectorD().real();
    }

    
/// **************************************************************
/// Experiment data and result recordings
    /**
     * @brief update filenames
     * @param file_mean filename for the means
     * @param file_cov filename for the covariances
     */
    inline void update_file_names(const std::string& file_mean, 
                                  const std::string& file_cov,
                                  const std::string& file_joint_cov,
                                  const std::string& file_precision, 
                                  const std::string& file_joint_precision, 
                                  const std::string& file_cost,
                                  const std::string& file_fac_costs,
                                  const std::string& file_perturbed_costs,
                                  const std::string& file_zk_sdf,
                                  const std::string& file_Sk_sdf){
        _res_recorder.update_file_names(file_mean, file_cov, file_joint_cov, file_precision, 
                                        file_joint_precision, file_cost, file_fac_costs, 
                                        file_zk_sdf, file_Sk_sdf);
        _file_perturbed_cost = file_perturbed_costs;
    }

    inline void update_file_names(const std::string & prefix = "", const std::string & afterfix=""){
        std::vector<std::string> vec_filenames;
        vec_filenames.emplace_back("mean");
        vec_filenames.emplace_back("cov");
        vec_filenames.emplace_back("joint_cov");
        vec_filenames.emplace_back("precision");
        vec_filenames.emplace_back("joint_precision");
        vec_filenames.emplace_back("cost");
        vec_filenames.emplace_back("factor_costs");
        vec_filenames.emplace_back("perturbation_statistics");
        vec_filenames.emplace_back("zk_sdf");
        vec_filenames.emplace_back("Sk_sdf");

        std::string underscore{"_"};
        std::string file_type{".csv"};

        if (prefix != ""){
            for (std::string & i_file : vec_filenames){
                i_file = prefix + i_file;
            }
        }

        if (afterfix != ""){
            for (std::string & i_file : vec_filenames){
                i_file = i_file + underscore + afterfix;
            }
        }

        for (std::string & i_file : vec_filenames){
                i_file = i_file + file_type;
        }

        _res_recorder.update_file_names(vec_filenames[0], 
                                        vec_filenames[1], 
                                        vec_filenames[2], 
                                        vec_filenames[3], 
                                        vec_filenames[4], 
                                        vec_filenames[5], 
                                        vec_filenames[6],
                                        vec_filenames[8],
                                        vec_filenames[9]);
        _file_perturbed_cost = vec_filenames[7];
    }

    /**
     * @brief save process data into csv files.
     */
    inline void save_data(bool verbose=true) { _res_recorder.save_data(verbose);}

    /**
     * @brief save a matrix to a file. 
     */
    inline void save_matrix(const std::string& filename, const MatrixXd& m) const{
        _matrix_io.saveData<MatrixXd>(filename, m);
    }

    inline void save_vector(const std::string& filename, const VectorXd& vec) const{
        _matrix_io.saveData<VectorXd>(filename, vec);
    }

    void switch_to_high_temperature();

    /**
     * @brief calculate and return the E_q{phi(x)} s for each factorized entity.
     * @return vector<double> 
     */
    std::vector<double> E_Phis(){
        std::vector<double> res;
        for (auto & p_factor: _vec_factors){
            res.emplace_back(p_factor->E_Phi());
        }
        return res;
    }

    /**
     * @brief calculate and return the E_q{(x-mu).*phi(x)} s for each factorized entity.
     * @return vector<double> 
     */
    std::vector<MatrixXd> E_xMuPhis(){
        std::vector<MatrixXd> res;
        for (auto & p_factor: _vec_factors){
            res.emplace_back(p_factor->E_xMuPhi());
        }
        return res;
    }

    /**
     * @brief calculate and return the E_q{(x-mu).*phi(x)} s for each factorized entity.
     * @return vector<double> 
     */
    std::vector<MatrixXd> E_xMuxMuTPhis(){
        std::vector<MatrixXd> res;
        for (auto & p_factor: _vec_factors){
            res.emplace_back(p_factor->E_xMuxMuTPhi());
        }
        return res;
    }

    /**************************** ONLY FOR 1D CASE ***********************/
    /**
     * @brief Draw a heat map for cost function in 1d case
     * @return MatrixXd heatmap of size (nmesh, nmesh)
     */
    MatrixXd cost_map(const double& x_start, 
                      const double& x_end, const double& y_start, 
                      const double& y_end, const int& nmesh){
        double res_x = (x_end - x_start) / nmesh;
        double res_y = (y_end - y_start) / nmesh;
        MatrixXd Z = MatrixXd::Zero(nmesh, nmesh);

        for (int i=0; i<nmesh; i++){
            VectorXd mean{VectorXd::Constant(1, x_start + i*res_x)};
            for (int j=0; j<nmesh; j++){
                SpMat precision(1, 1);
                precision.coeffRef(0, 0) = (y_start + j*res_y);
                Z(j, i) = cost_value(mean, precision); /// the order of the matrix in cpp and in matlab
            }
        }
        return Z;
    }

    /**
     * @brief save the cost map
     */
    void save_costmap(std::string filename="costmap.csv"){
        MatrixXd cost_m = cost_map(18, 25, 0.05, 1, 40);
        std::ofstream file(filename);
        if (file.is_open()){
            file << cost_m.format(CSVFormat);
            file.close();}
    }

}; //class

}


#include "GVI-GH-impl.h"

#endif // GVI_GH_H
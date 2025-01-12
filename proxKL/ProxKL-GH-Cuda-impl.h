
#pragma once

using namespace Eigen;
#include <stdexcept>
#include <optional>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace gvi{

template <typename Factor>
std::tuple<double, VectorXd, SpMat> ProxKLGH<Factor>::onestep_linesearch(const double &step_size, 
                                                                            const VectorXd& dmu, 
                                                                            const SpMat& dprecision)
{
    SpMat new_precision; 
    VectorXd new_mu; 
    new_mu.setZero(); new_precision.setZero();

    // update mu and precision matrix
    Eigen::ConjugateGradient<SpMat> solver;
    // Eigen::ConjugateGradient<SpMat, Eigen::Upper> solver;

    // std::cout << "mu_prior" << _mu_prior.transpose() << std::endl << std::endl;
    // std::cout << "mu" << this->_mu.transpose() << std::endl << std::endl;

    std::cout << "step_size: " << step_size << std::endl;

    VectorXd prior_term = _precision_prior * _mu_prior;
    VectorXd dmu_term = dmu;
    double threshold = 1e-9;
    for (int i = 0; i < prior_term.size(); ++i) {
        if (std::abs(prior_term[i]) < threshold)
            prior_term[i] = 0;
        if (std::abs(dmu_term[i]) < threshold)
            dmu_term[i] = 0;
    }

    // std::cout << "dmu" << dmu_term.transpose() << std::endl << std::endl;
    // std::cout << "K_inv * mu" << prior_term.transpose() << std::endl << std::endl;
    // std::cout << "precision * mu" << (this->_precision * this->_mu / step_size).transpose() << std::endl << std::endl;

    // std::cout << "K-inv norm: " << _precision_prior.norm() << std::endl;
    // std::cout << "precision norm: " << this->_precision.norm() << std::endl;
    // std::cout << "dprecision norm: " << dprecision.norm() << std::endl;

    // Need to make the dmu larger to realize collision avoidance, add Temperature
    new_mu = solver.compute(_precision_prior + this->_precision / step_size).solve(-5*dmu + _precision_prior * _mu_prior + this->_precision * this->_mu / step_size);
    new_precision = step_size / (step_size + 1) * (dprecision + _precision_prior) + this->_precision / (step_size + 1);

    // std::cout << "New mu: " << new_mu.transpose() << std::endl;

    // The dprecision make the precision matrix not definite positive, which will cause the solver failed when updating GH.
    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_dp(dprecision);
    // if (solver_dp.info() == Eigen::Success)
    //     std::cout << "Eigenvaluse: \n" << solver_dp.eigenvalues().transpose() << std::endl;

    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_prior(_precision_prior);
    // if (solver_prior.info() == Eigen::Success)
    //     std::cout << "Eigenvaluse: \n" << solver_prior.eigenvalues().transpose() << std::endl;

    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_joint(this->_precision);
    // if (solver_joint.info() == Eigen::Success)
    //     std::cout << "Eigenvaluse: \n" << solver_joint.eigenvalues().transpose() << std::endl;

    // new cost
    double new_cost = cost_value_cuda(new_mu, new_precision);

    if (std::isnan(new_cost)) {
        std::cerr << "Error: Detected NaN in cost calculation. Exiting program." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::cout << "New cost = " << new_cost << std::endl << std::endl;
    return std::make_tuple(new_cost, new_mu, new_precision);
}


template <typename Factor>
void ProxKLGH<Factor>::optimize(std::optional<bool> verbose)
{
    // default verbose
    bool is_verbose = verbose.value_or(true);
    bool is_lowtemp = true;
    bool converged = false;

    Base::_vec_nonlinear_factors[0]->cuda_init();
    
    for (int i_iter = 0; i_iter < Base::_niters; i_iter++)
    {   

        if (converged){
            break;
        }

        // ============= High temperature phase =============
        if (i_iter == Base::_niters_lowtemp && is_lowtemp){
            if (is_verbose){
                std::cout << "Switching to high temperature.." << std::endl;
            }
            this->switch_to_high_temperature();
            is_lowtemp = false;
        }

        auto result_cuda = factor_cost_vector_cuda(this->_mu, this->_precision);
        double cost_iter = std::get<0>(result_cuda);
        VectorXd fact_costs_iter = std::get<1>(result_cuda);
        VectorXd dmu = std::get<2>(result_cuda);
        SpMat dprecision = std::get<3>(result_cuda);

        if (is_verbose){
            std::cout << "========= iteration " << i_iter << " ========= " << std::endl;
            std::cout << "--- cost_iter ---" << std::endl << cost_iter << std::endl << std::endl;
            // std::cout << "Factor Costs:" << fact_costs_iter.transpose() << std::endl;
        }

        Base::_res_recorder.update_data(this->_mu, this->_covariance, this->_precision, cost_iter, fact_costs_iter);
        
        int cnt = 0;
        int B = 1;
        double step_size = Base::_step_size_base;

        // backtracking 
        while (true)
        {   
            // new step size
            // step_size = pow(_step_size_base, B);
            step_size = step_size * 0.75;

            auto onestep_res = onestep_linesearch(step_size, dmu, dprecision);

            double new_cost = std::get<0>(onestep_res);
            VectorXd new_mu = std::get<1>(onestep_res);
            auto new_precision = std::get<2>(onestep_res);

            // accept new cost and update mu and precision matrix
            if (new_cost < cost_iter){
                // update mean and covariance
                this->update_proposal(new_mu, new_precision);
                break;
            }else{ 
                // shrinking the step size
                B += 1;
                cnt += 1;
            }

            if (cnt > Base::_niters_backtrack)
            {
                if (is_verbose){
                    std::cout << "Reached the maximum backtracking steps." << std::endl;
                }

                if (is_lowtemp){
                    this->switch_to_high_temperature();
                    is_lowtemp = false;
                }else{
                    converged = true;
                }

                // update_proposal(new_mu, new_precision);
                break;
            }                
        }
    }

    Base::_vec_nonlinear_factors[0]->cuda_free();

    std::cout << "=========== Saving Data ===========" << std::endl;
    Base::save_data(is_verbose);

    std::cout << "Optimization Finished" << std::endl;

}


template <typename Factor>
void ProxKLGH<Factor>::optimize_linear(std::optional<bool> verbose)
{
    // default verbose
    bool is_verbose = verbose.value_or(true);
    bool is_lowtemp = true;
    bool converged = false;
    
    for (int i_iter = 0; i_iter < Base::_niters; i_iter++)
    {   

        if (converged){
            break;
        }

        // ============= High temperature phase =============
        if (i_iter == Base::_niters_lowtemp && is_lowtemp){
            if (is_verbose){
                std::cout << "Switching to high temperature.." << std::endl;
            }
            this->switch_to_high_temperature();
            is_lowtemp = false;
        }

        double cost_iter = cost_value_linear(this->_mu, this->_precision);

        if (is_verbose){
            std::cout << "========= iteration " << i_iter << " ========= " << std::endl;
            std::cout << "--- cost_iter ---" << std::endl << cost_iter << std::endl << std::endl;
            // std::cout << "Factor Costs:" << fact_costs_iter.transpose() << std::endl;
        }
        
        int cnt = 0;
        int B = 1;
        double step_size = Base::_step_size_base;

        // backtracking 
        while (true)
        {   
            // new step size
            step_size = step_size * 0.75;

            SpMat new_precision; 
            VectorXd new_mu; 
            new_mu.setZero(); new_precision.setZero();

            Eigen::ConjugateGradient<SpMat> solver;

            new_mu = solver.compute(_precision_prior + this->_precision / step_size).solve(_precision_prior * _mu_prior + this->_precision * this->_mu / step_size);
            new_precision = step_size / (step_size + 1) * (_precision_prior + this->_precision / step_size);

            double new_cost = cost_value_linear(new_mu, new_precision);
            std::cout << "New cost = " << new_cost << std::endl << std::endl;

            // accept new cost and update mu and precision matrix
            if (new_cost < cost_iter){
                // update mean and covariance
                this->update_proposal(new_mu, new_precision);
                break;
            }else{ 
                // shrinking the step size
                B += 1;
                cnt += 1;
            }

            if (cnt > Base::_niters_backtrack)
            {
                if (is_verbose){
                    std::cout << "Reached the maximum backtracking steps." << std::endl;
                }

                if (is_lowtemp){
                    this->switch_to_high_temperature();
                    is_lowtemp = false;
                }else{
                    converged = true;
                }

                // update_proposal(new_mu, new_precision);
                break;
            }                
        }
    }

    std::cout << "=========== Saving Data ===========" << std::endl;
    Base::save_data(is_verbose);

    std::cout << "Optimization Finished" << std::endl;

}


template <typename Factor>
std::tuple<double, VectorXd, VectorXd, SpMat>ProxKLGH<Factor>::factor_cost_vector_cuda(const VectorXd& fill_joint_mean, SpMat& joint_precision)
{
    int n_nonlinear = Base::_vec_nonlinear_factors.size();

    VectorXd fac_costs(Base::_nfactors);
    VectorXd nonlinear_fac_cost(n_nonlinear);
    fac_costs.setZero();
    nonlinear_fac_cost.setZero();

    std::vector<MatrixXd> sigmapts_vec(n_nonlinear);
    std::vector<VectorXd> mean_vec(n_nonlinear);

    omp_set_num_threads(20); 

    #pragma omp parallel for
    for (int i = 0; i < n_nonlinear; i++)
    {
        auto &opt_k = Base::_vec_nonlinear_factors[i];
        opt_k->cuda_matrices(sigmapts_vec, mean_vec); 
    }

    int sigma_rows = sigmapts_vec[0].rows();
    int sigma_cols = sigmapts_vec[0].cols();
    int mean_size = mean_vec[0].size();

    MatrixXd sigmapts_mat(sigma_rows, sigmapts_vec.size()*sigma_cols);
    MatrixXd mean_mat(mean_size, mean_vec.size());

    VectorXd E_phi_mat(n_nonlinear);
    VectorXd E_Xphi_mat(sigma_cols * n_nonlinear);
    MatrixXd E_XXphi_mat(sigma_cols, sigma_cols * n_nonlinear);

    #pragma omp parallel for
    for (int i = 0; i < n_nonlinear; i++)
    {
        sigmapts_mat.block(0, i * sigma_cols, sigma_rows, sigma_cols) = sigmapts_vec[i];
        mean_mat.col(i) = mean_vec[i];
    }

    // Compute the cost and derivatives of the nonlinear factors
    Base::_vec_nonlinear_factors[0]->dmuIntegration(sigmapts_mat, mean_mat, nonlinear_fac_cost, E_Xphi_mat, E_XXphi_mat, sigma_cols);
    E_phi_mat = nonlinear_fac_cost;

    nonlinear_fac_cost = nonlinear_fac_cost / this ->_temperature; 

    int cnt = 0;

    // Use a private counter for each thread to avoid race conditions
    int thread_cnt = 0;

    // #pragma omp for
    for (int i = 0; i < Base::_vec_factors.size(); ++i)
    {
        auto &opt_k = Base::_vec_factors[i];
        if (opt_k->linear_factor())
        {
            double cost_value = opt_k->fact_cost_value(this->_mu, this->_covariance); 
            fac_costs(thread_cnt) = cost_value;
            // if (cost_value > 10) // only show the cost of the linear factors that are too large
            //     std::cout << "Linear Factor " << i << " Cost: " << cost_value << std::endl;
        }
        else
            fac_costs(thread_cnt) = nonlinear_fac_cost(opt_k->_start_index - 1);
        thread_cnt += 1;
    }

    #pragma omp critical
    {
        cnt += thread_cnt; // Safely update the global counter
    }

    double value = fac_costs.sum();
    SparseLDLT ldlt(joint_precision);
    VectorXd vec_D = ldlt.vectorD();

    double cost = value + vec_D.array().log().sum() / 2;

    double entropy = vec_D.array().log().sum() / 2;
    double collision_cost = nonlinear_fac_cost.sum();
    double prior_cost = fac_costs.sum() - collision_cost;

    std::cout << "Entropy: " << entropy << std::endl;
    std::cout << "Collision Cost: " << collision_cost << std::endl;
    std::cout << "Prior Cost: " << prior_cost << std::endl;


    // SparseLDLT ldlt_prior(_precision_prior);

    // SpMat precision_prior_times_Cov = _precision_prior * this->_covariance;
    // double trace_term = precision_prior_times_Cov.diagonal().sum();
    // double quadratic_term = (fill_joint_mean - _mu_prior).transpose() * _precision_prior * (fill_joint_mean - _mu_prior);

    // std::cout << "Joint linear cost: " << (trace_term + quadratic_term) / (2 * this->_temperature) << std::endl;
    // std::cout << "Error of the joint linear cost: " << (trace_term + quadratic_term) / (2 * this->_temperature) - prior_cost << std::endl;

    // double cost_joint = nonlinear_fac_cost.sum() + (trace_term + quadratic_term) / (2 * this->_temperature) + vec_D.array().log().sum() / 2;


    _Vdmu.setZero();
    _Vddmu.setZero();

    VectorXd Vdmu_sum(Base::_dim);
    SpMat Vddmu_sum(Base::_dim, Base::_dim);
    Vdmu_sum.setZero();
    Vddmu_sum.setZero();

    #pragma omp parallel 
    {
        // Thread-local storage to avoid race conditions
        VectorXd Vdmu_private(Vdmu_sum.size());
        SpMat Vddmu_private(Vddmu_sum.rows(), Vddmu_sum.cols());
        Vdmu_private.setZero();
        Vddmu_private.setZero();

        #pragma omp for nowait // Nowait allows threads to continue without waiting at the end of the loop
        for (auto &opt_k : Base::_vec_nonlinear_factors) {
            int index = opt_k->index()-1;
            MatrixXd ddmu_i = E_XXphi_mat.block(0, index*sigma_cols, sigma_cols, sigma_cols);
            VectorXd dmu_i = E_Xphi_mat.segment(index*sigma_cols, sigma_cols);

            opt_k->calculate_partial_V(ddmu_i, dmu_i, E_phi_mat(index));

            Vdmu_private += opt_k->local2joint_dmu_insertion();
            Vddmu_private += opt_k->local2joint_dprecision_insertion();
        }

        #pragma omp critical
        {
            Vdmu_sum += Vdmu_private;
            Vddmu_sum += Vddmu_private;
        }
    }

    _Vdmu = Vdmu_sum;
    _Vddmu = Vddmu_sum;

    return std::make_tuple(cost, fac_costs, _Vdmu, _Vddmu);
}


template <typename Factor>
double ProxKLGH<Factor>::cost_value_cuda(const VectorXd& fill_joint_mean, SpMat& joint_precision)
{
    int n_nonlinear = Base::_vec_nonlinear_factors.size();
    VectorXd nonlinear_fac_cost(n_nonlinear);
    nonlinear_fac_cost.setZero();

    SpMat joint_cov = Base::inverse_GBP(joint_precision);

    std::vector<MatrixXd> sigmapts_vec(n_nonlinear);
    std::vector<VectorXd> mean_vec(n_nonlinear);

    #pragma omp parallel for
    for (int i = 0; i < n_nonlinear; i++)
    {
        auto &opt_k = Base::_vec_nonlinear_factors[i];
        opt_k->cuda_matrices(fill_joint_mean, joint_cov, sigmapts_vec, mean_vec); 
    }

    int sigma_rows = sigmapts_vec[0].rows();
    int sigma_cols = sigmapts_vec[0].cols();
    int mean_size = mean_vec[0].size();

    MatrixXd sigmapts_mat(sigma_rows, sigmapts_vec.size()*sigma_cols);

    for (int i = 0; i < n_nonlinear; i++)
        sigmapts_mat.block(0, i * sigma_cols, sigma_rows, sigma_cols) = sigmapts_vec[i];

    // Compute the cost of the nonlinear factors
    Base::_vec_nonlinear_factors[0]->newCostIntegration(sigmapts_mat, nonlinear_fac_cost, sigma_cols);

    nonlinear_fac_cost = nonlinear_fac_cost / this ->_temperature;

    double value = 0.0;

    #pragma omp parallel for reduction(+:value)
    for (int i = 0; i < Base::_vec_linear_factors.size(); ++i)
    {
        auto &opt_k = Base::_vec_linear_factors[i];
        value += opt_k->fact_cost_value(fill_joint_mean, joint_cov); 
    }

    std::cout << "Collision Cost: " << nonlinear_fac_cost.sum() << std::endl;
    std::cout << "Prior Cost: " << value << std::endl;
    
    value += nonlinear_fac_cost.sum();

    SparseLDLT ldlt(joint_precision);
    VectorXd vec_D = ldlt.vectorD();

    std::cout << "Entropy: " << vec_D.array().log().sum() / 2 << std::endl;
    return value + vec_D.array().log().sum() / 2;


    // // Compute prior cost in joint level
    // SparseLDLT ldlt_prior(_precision_prior);

    // SpMat precision_prior_times_Cov = _precision_prior * joint_cov;
    // double trace_term = precision_prior_times_Cov.diagonal().sum();
    // double quadratic_term = (fill_joint_mean - _mu_prior).transpose() * _precision_prior * (fill_joint_mean - _mu_prior);

    // VectorXd vec_D_prior = ldlt_prior.vectorD();
    // VectorXd vec_D_joint = ldlt.vectorD();
    // double log_term = vec_D_prior.array().log().sum() - vec_D_joint.array().log().sum();

    // // std::cout << "Joint linear cost: " << (trace_term + quadratic_term) / (2 * this->_temperature) << std::endl << std::endl;

    // double cost_joint = (trace_term + quadratic_term) / (2 * this->_temperature);
    // cost_joint += nonlinear_fac_cost.sum();
    // return cost_joint + vec_D.array().log().sum() / 2;
}

template <typename Factor>
inline void ProxKLGH<Factor>::update_proposal(const VectorXd& new_mu, const SpMat& new_precision)
{
    Base::set_mu(new_mu);
    Base::set_precision(new_precision);
    // std::cout << "New mu: " << new_mu.transpose() << std::endl;
    // std::cout << "Updated proposal" << std::endl;
}

/**
 * @brief Compute the costs of all factors, using current values.
 */
template <typename Factor>
VectorXd ProxKLGH<Factor>::factor_cost_vector()
{   
    return Base::factor_cost_vector(this->_mu, this->_precision);
}

template <typename Factor>
double ProxKLGH<Factor>::cost_value_linear(const VectorXd& fill_joint_mean, const SpMat& joint_precision)
{   
    SpMat joint_cov = Base::inverse_GBP(joint_precision); // The result of matrix multiplication will keeps the same because of the sparse structure.

    // Compute the cost of the linear factors
    VectorXd fac_costs(Base::_vec_linear_factors.size());
    fac_costs.setZero();

    for (int i = 0; i < Base::_vec_linear_factors.size(); ++i)
    {
        auto &opt_k = Base::_vec_linear_factors[i];
        double cost_value = opt_k->fact_cost_value(fill_joint_mean, joint_cov); 
        fac_costs(i) = cost_value;
    }

    double value = fac_costs.sum();
    SparseLDLT ldlt(joint_precision);
    VectorXd vec_D = ldlt.vectorD();

    std::cout << "linear cost: " << value << std::endl;
    std::cout << "entropy: " << vec_D.array().log().sum() / 2 << std::endl;

    double cost = value + vec_D.array().log().sum() / 2;

    //return {cost, fac_costs};



    // Compute the cost and KL divergence in joint level (The time cost is actually lower than the factorization)
    SparseLDLT ldlt_prior(_precision_prior);

    SpMat precision_prior_times_Cov = _precision_prior * joint_cov;
    double trace_term = precision_prior_times_Cov.diagonal().sum();
    // std::cout << "trace_term: " << trace_term << std::endl;

    double quadratic_term = (fill_joint_mean - _mu_prior).transpose() * _precision_prior * (fill_joint_mean - _mu_prior);
    // std::cout << "quadratic_term: " << quadratic_term << std::endl;

    double cost_joint_linear = (trace_term + quadratic_term) / (2 * this->_temperature);

    std::cout << "Joint linear cost: " << cost_joint_linear << std::endl;
    std::cout << "Entropy: " << vec_D.array().log().sum() / 2 << std::endl;

    double cost_joint = cost_joint_linear + vec_D.array().log().sum() / 2;

    // VectorXd vec_D_prior = ldlt_prior.vectorD();
    // VectorXd vec_D_joint = ldlt.vectorD();
    // double log_term = vec_D_prior.array().log().sum() - vec_D_joint.array().log().sum();

    // std::cout << "log_term: " << log_term << std::endl;
    // std::cout << "prior_log_term: " << vec_D_prior.array().log().sum() << std::endl;
    // std::cout << "joint_log_term: " << vec_D_joint.array().log().sum() << std::endl;


    // double cost_joint = (trace_term + quadratic_term + log_term - fill_joint_mean.size()) / 2;
    // std::cout << "KL Divergence: " << cost_joint << std::endl << std::endl;    

    return cost_joint;
}


/**
 * @brief Compute the total cost function value given a state, using current values.
 */
template <typename Factor>
double ProxKLGH<Factor>::cost_value()
{
    return Base::cost_value(this->_mu, this->_precision);
}

template <typename Factor>
double ProxKLGH<Factor>::cost_value_no_entropy()
{
    
    SpMat Cov = this->inverse(this->_precision);
    
    double value = 0.0;
    for (auto &opt_k : this->_vec_factors)
    {
        value += opt_k->fact_cost_value(this->_mu, Cov);
    }
    return value; // / _temperature;
}

}

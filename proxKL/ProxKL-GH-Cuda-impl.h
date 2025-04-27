
#pragma once

using namespace Eigen;
#include <stdexcept>
#include <optional>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace gvi{

template <typename Factor, typename CudaClass>
std::tuple<double, VectorXd, SpMat> ProxKLGH<Factor, CudaClass>::onestep_linesearch(const double &step_size,
                                                                            const VectorXd& dmu,
                                                                            const SpMat& dprecision)
{
    SpMat new_precision;
    VectorXd new_mu;
    new_mu.setZero(); new_precision.setZero();
    double temperature = this->_temperature;

    // update mu and precision matrix
    Eigen::BiCGSTAB<SpMat, IncompleteLUT<double>> solver;
    solver.setTolerance(1e-6);

    // VectorXd prior_term = _precision_prior * _mu_prior / temperature;
    // VectorXd dmu_term = -dmu / temperature;
    // double threshold = 1e-9;
    // for (int i = 0; i < prior_term.size(); ++i) {
    //     if (std::abs(prior_term[i]) < threshold)
    //         prior_term[i] = 0;
    //     if (std::abs(dmu_term[i]) < threshold)
    //         dmu_term[i] = 0;
    // }
    // std::cout << "dmu" << dmu_term.transpose() << std::endl << std::endl;
    // std::cout << "K_inv * mu" << prior_term.transpose() << std::endl << std::endl;
    // std::cout << "precision * mu" << (this->_precision * this->_mu / step_size).transpose() << std::endl << std::endl;
    // std::cout << "K-inv norm: " << _precision_prior.norm() << std::endl;
    // std::cout << "precision norm: " << this->_precision.norm() << std::endl;
    // std::cout << "dprecision norm: " << dprecision.norm() << std::endl;
    VectorXd combined_mu = -dmu / temperature + _precision_prior * _mu_prior / temperature + this->_precision * this->_mu / step_size;

    new_mu = solver.compute(_precision_prior / temperature + this->_precision / step_size).solve(-dmu / temperature + _precision_prior * _mu_prior / temperature + this->_precision * this->_mu / step_size);
    new_precision = (dprecision / temperature + _precision_prior / temperature + this->_precision / step_size) * step_size / (step_size + 1);

    // std::cout << "Error of the solver: " << ((_precision_prior / temperature + this->_precision / step_size) * new_mu - combined_mu).norm() / combined_mu.norm() * 100 << "%" << std::endl;

    double new_cost;
    if (this->isPositiveDefinite(new_precision))
        new_cost = Base::cost_value_cuda(new_mu, new_precision);
    else
        new_cost = std::numeric_limits<double>::infinity();

    return std::make_tuple(new_cost, new_mu, new_precision);
}

template <typename Factor, typename CudaClass>
double ProxKLGH<Factor, CudaClass>::bisection_stepsize(const VectorXd& dmu, const SpMat& dprecision)
{
    Timer timer;
    SpMat new_precision;
    VectorXd new_mu;
    new_mu.setZero(); new_precision.setZero();
    double temperature = this->_temperature;

    double log_lower = -2;
    double log_upper = 2;
    double log_threshold = 0.02;
    double epsilon = this->_alpha;

    if (!_temp_switch && _narrow_range) {
        // std::cout << "Narrower range" << std::endl;
        log_lower = max(_step_size_last - 1, -2.0);
        log_upper = min(_step_size_last + 1, 2.0);
        log_threshold = 0.075;
    }

    // update mu and precision matrix
    Eigen::BiCGSTAB<SpMat, IncompleteLUT<double>> solver;
    solver.setTolerance(1e-6);

    // timer.start();
    VectorXd prior_precision_mu = _precision_prior * _mu_prior / temperature;
    VectorXd current_precision_mu = this->_precision * this->_mu;
    SpMat precision_term = _precision_prior / temperature;
    VectorXd dmu_term = -dmu / temperature;
    SpMat dprecision_term = dprecision / temperature;

    // std::cout << "dmu: " << dmu_term.norm() << std::endl;
    // std::cout << "dprecision: " << dprecision_term.norm() << std::endl;
    // std::cout << "Time for preparing the terms: " << timer.end_mus_output() << " us" << std::endl;

    VectorXd warm_start_guess = VectorXd::Zero(_mu_prior.size());

    while (log_upper - log_lower > log_threshold)
    {
        double log_mid = (log_lower + log_upper) / 2;
        double step_size = std::exp(log_mid);

        // timer.start();
        SpMat combined_precision = precision_term + this->_precision / step_size;
        VectorXd combined_mu = -dmu_term + prior_precision_mu + current_precision_mu / step_size;
        // std::cout << "Time for preparing the combined terms: " << timer.end_mus_output() << " us" << std::endl;

        // timer.start();
        new_mu = solver.compute(combined_precision).solveWithGuess(combined_mu, warm_start_guess);
        new_precision = (dprecision_term + precision_term + this->_precision / step_size) * step_size / (step_size + 1);
        warm_start_guess = new_mu;
        // std::cout << "Time for solving the linear system with warm start: " << timer.end_mus_output() << " us" << std::endl;

        // Compute KL divergence and check for PD indirectly
        // timer.start();
        double KL = KL_Divergence(this->_mu, this->_precision, this->_covariance, new_mu, new_precision);

        // std::cout << "Error of the solver: " << (combined_precision * new_mu - combined_mu).norm() / combined_mu.norm() * 100 << "%" << std::endl;
        // std::cout << "Time for computing KL divergence: " << timer.end_mus_output() << " us" << std::endl << std::endl;

        if (std::isnan(KL) || KL >= epsilon) {
            log_upper = log_mid;
        } else {
            log_lower = log_mid;
        }
    }

    double final_step_size = (log_lower + log_upper) / 2;
    // std::cout << "Finial Step Size: " << final_step_size << std::endl;
    // double diff_step = final_step_size - _step_size_last;
    // if (abs(diff_step) > 0.75)
    //     std::cout << "Step size difference: " << diff_step << std::endl;

    _step_size_last = final_step_size;

    return std::exp((log_lower + log_upper) / 2);
}

template <typename Factor, typename CudaClass>
std::tuple<double, VectorXd, SpMat> ProxKLGH<Factor, CudaClass>::bisection_update(const VectorXd& dmu, const SpMat& dprecision)
{
    SpMat new_precision;
    VectorXd new_mu;
    new_mu.setZero(); new_precision.setZero();
    double temperature = this->_temperature;

    double log_lower = -2;
    double log_upper = 2;
    double log_threshold = 0.01;
    double epsilon = this->_alpha;

    // update mu and precision matrix
    Eigen::BiCGSTAB<SpMat, IncompleteLUT<double>> solver;
    solver.setTolerance(1e-6);

    while (log_upper - log_lower > log_threshold)
    {
        double log_mid = (log_lower + log_upper) / 2;
        double step_size = std::exp(log_mid);

        solver.compute(_precision_prior / temperature + this->_precision / step_size);
        new_mu = solver.solve(-dmu / temperature + _precision_prior * _mu_prior / temperature + this->_precision * this->_mu / step_size);

        new_precision = (dprecision / temperature + _precision_prior / temperature + this->_precision / step_size) * step_size / (step_size + 1);

        // Compute KL divergence and check for PD indirectly
        double KL = KL_Divergence(this->_mu, this->_precision, this->_covariance, new_mu, new_precision);

        if (std::isnan(KL) || KL >= epsilon) {
            log_upper = log_mid;
        } else {
            log_lower = log_mid;
        }
    }

    double final_step_size = std::exp((log_lower + log_upper) / 2);

    new_mu = solver.compute(_precision_prior / temperature + this->_precision / final_step_size).solve(-dmu / temperature + _precision_prior * _mu_prior / temperature + this->_precision * this->_mu / final_step_size);
    new_precision = (dprecision / temperature + _precision_prior / temperature + this->_precision / final_step_size) * final_step_size / (final_step_size + 1);

    // new cost
    double new_cost = Base::cost_value_cuda(new_mu, new_precision);

    if (std::isnan(new_cost)) {
        std::cerr << "Error: Detected NaN in cost calculation. Exiting program." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // std::cout << "Final Step Size: " << final_step_size << std::endl;
    // std::cout << "New cost = " << new_cost << std::endl << std::endl;
    return std::make_tuple(new_cost, new_mu, new_precision);
}


template <typename Factor, typename CudaClass>
void ProxKLGH<Factor, CudaClass>::optimize(std::optional<bool> verbose)
{
    // default verbose
    bool is_verbose = verbose.value_or(true);
    bool is_lowtemp = true;
    bool converged = false;

    Timer timer, timer_total;

    timer.start();
    timer_total.start();
    Base::cuda_init(Base::_vec_nonlinear_factors.size());
    std::cout << "Time for initializing cuda: " << timer.end_mus_output() << " us" << std::endl;

    if (this->_save_data){
        // Base::_niters = 1;
        Base::_res_recorder.init_data(Base::_save_covariance);
    }

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
            _temp_switch = true;
        }

        if (is_verbose){
            std::cout << "========= iteration " << i_iter << " ========= " << std::endl;
        }

        timer.start();
        auto [cost_iter, fact_costs_iter, dmu, dprecision] = factor_cost_vector_cuda(this->_mu, this->_precision);
        std::cout << "Time for computing factor costs: " << timer.end_mus_output() << " us" << std::endl;

        if (is_verbose){
            std::cout << "--- cost_iter ---" << std::endl << cost_iter << std::endl;
            // std::cout << "Factor Costs:" << fact_costs_iter.transpose() << std::endl;
        }

        // // Check the shape of the prior
        // if (this->_save_data){
        //     SpMat prior_covariance = Base::inverse_GBP(this->_precision_prior);
        //     Base::_res_recorder.update_data(this->_mu_prior, prior_covariance, this->_precision_prior, cost_iter, fact_costs_iter);
        // }

        if (this->_save_data){
            Base::_res_recorder.update_data(this->_mu, this->_covariance, this->_precision, cost_iter, fact_costs_iter);
        }

        timer.start();
        int cnt = 0;
        double step_size = bisection_stepsize(dmu, dprecision);
        _temp_switch = false;
        std::cout << "Time for bisection step size: " << timer.end_mus_output() << " us" << std::endl;

        timer.start();
        // backtracking
        while (true)
        {
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
                    _temp_switch = true;
                }else{
                    converged = true;
                }

                break;
            }

            // new step size
            step_size = step_size * 0.75;
        }
        std::cout << "Time for backtracking: " << timer.end_mus_output() << " us" << std::endl;
    }

    Base::cuda_free();
    std::cout << "Time for optimization: " << timer_total.end_mus_output() << " us" << std::endl;

    if (this->_save_data){
        std::cout << "=========== Saving Data ===========" << std::endl;
        Base::save_data(is_verbose);
    }
    std::cout << "Optimization Finished" << std::endl;
}


template <typename Factor, typename CudaClass>
void ProxKLGH<Factor, CudaClass>::optimize_time_test()
{
    bool is_lowtemp = true;
    bool converged = false;
    _narrow_range = true;
    _temp_switch = false;

    for (int i_iter = 0; i_iter < Base::_niters; i_iter++)
    {

        if (converged){
            break;
        }

        // ============= High temperature phase =============
        if (i_iter == Base::_niters_lowtemp && is_lowtemp){
            this->switch_to_high_temperature();
            is_lowtemp = false;
        }

        auto [cost_iter, fact_costs_iter, dmu, dprecision] = factor_cost_vector_cuda(this->_mu, this->_precision);

        int cnt = 0;
        double step_size = bisection_stepsize(dmu, dprecision);

        // backtracking
        while (true)
        {
            auto onestep_res = onestep_linesearch(step_size, dmu, dprecision);
            double new_cost = std::get<0>(onestep_res);
            VectorXd new_mu = std::get<1>(onestep_res);
            SpMat new_precision = std::get<2>(onestep_res);

            // accept new cost and update mu and precision matrix
            if (new_cost < cost_iter){
                // update mean and covariance
                break;
            }else{
                // shrinking the step size
                cnt += 1;
            }

            if (cnt > Base::_niters_backtrack)
            {
                if (is_lowtemp){
                    this->switch_to_high_temperature();
                    is_lowtemp = false;
                }else{
                    converged = true;
                }
                break;
            }
        }
    }
}


template <typename Factor, typename CudaClass>
void ProxKLGH<Factor, CudaClass>::time_test()
{
    Base::cuda_init(Base::_vec_nonlinear_factors.size());

    Timer timer;
    int n_repeat = 5;

    std::vector<double> times_optimization;
    times_optimization.reserve(n_repeat);

    for (int i=0; i < n_repeat+1; i++){
        timer.start();
        optimize_time_test();
        double time = timer.end_mis();
        if (i != 0)
        times_optimization.push_back(time);  // The first time need to initialize
    }

    double average_time = std::accumulate(times_optimization.begin(), times_optimization.end(), 0.0) / n_repeat;

    std::cout << "% " << Base::_vec_nonlinear_factors.size() + 2 << std::endl;
    std::cout << "% GPU Optimize average: " << average_time << " ms" << std::endl;

    std::cout << "% [ " << times_optimization[0];
    for (int i = 1; i < times_optimization.size(); ++i) {
        std::cout << ", " << times_optimization[i];
    }
    std::cout << " ]" << std::endl;

    Base::cuda_free();
}


/**
 * @brief Compute the costs of all factors, using current values.
 */
template <typename Factor, typename CudaClass>
std::tuple<double, VectorXd, VectorXd, SpMat>ProxKLGH<Factor, CudaClass>::factor_cost_vector_cuda(const VectorXd& fill_joint_mean, SpMat& joint_precision)
{
    int n_nonlinear = Base::_vec_nonlinear_factors.size();

    VectorXd fac_costs(Base::_nfactors);
    VectorXd nonlinear_fac_cost(n_nonlinear);
    fac_costs.setZero();
    nonlinear_fac_cost.setZero();

    MatrixXd sigmapts_mat(_sigma_rows, n_nonlinear*_dim_conf);
    MatrixXd mean_mat(_dim_conf, n_nonlinear);
    MatrixXd covariance_matrix(_dim_conf, n_nonlinear*_dim_conf);
    MatrixXd sigma(_sigma_rows, _dim_conf);

    VectorXd E_phi_mat(n_nonlinear);
    VectorXd E_Xphi_mat(_dim_conf * n_nonlinear);
    MatrixXd E_XXphi_mat(_dim_conf, _dim_conf * n_nonlinear);

    #pragma omp parallel for
    for (int i = 0; i < n_nonlinear; i++)
    {
        auto &opt_k = Base::_vec_nonlinear_factors[i];
        mean_mat.col(i) = opt_k->_mu;
        covariance_matrix.block(0, i * _dim_conf, _dim_conf, _dim_conf) = opt_k->covariance();
    }

    Base::compute_sigmapts(mean_mat, covariance_matrix, _dim_conf, n_nonlinear, sigma);

    // Compute the cost and derivatives of the nonlinear factors
    Base::dmuIntegration(sigmapts_mat, mean_mat, nonlinear_fac_cost, E_Xphi_mat, E_XXphi_mat, _dim_conf);
    E_phi_mat = nonlinear_fac_cost;

    nonlinear_fac_cost = nonlinear_fac_cost / this ->_temperature;

    #pragma omp parallel for
    for (int i = 0; i < Base::_vec_factors.size(); i++)
    {
        auto &opt_k = Base::_vec_factors[i];
        if (opt_k->linear_factor()) // matrix multiplication between dim_state x dim_state and dim_state * dim_state
            fac_costs(i) = opt_k->fact_cost_value(this->_mu, this->_covariance);
        else
            fac_costs(i) = nonlinear_fac_cost(opt_k->_start_index - 1);
    }

    double value = fac_costs.sum();
    SparseLDLT ldlt(joint_precision);
    VectorXd vec_D = ldlt.vectorD();

    double cost = value + vec_D.array().log().sum() / 2;

    double entropy = vec_D.array().log().sum() / 2;
    double collision_cost = nonlinear_fac_cost.sum();
    double prior_cost = fac_costs.sum() - collision_cost;

    // if ((fac_costs.array() < 0).any()) {
    //     for (int i = 0; i < fac_costs.size(); ++i) {
    //         if (fac_costs[i] < 0) {
    //             std::cout << "Negative value at index " << i << ": " << fac_costs[i] << std::endl;
    //         }
    //     }
    //     // std::cout << "fac_costs contains negative values: " << fac_costs.transpose() << std::endl;
    // }

    std::cout << "Prior Cost: " << prior_cost << std::endl;
    std::cout << "Collision Cost: " << collision_cost << std::endl;
    std::cout << "Entropy: " << entropy << std::endl;

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
            MatrixXd ddmu_i = E_XXphi_mat.block(0, index*_dim_conf, _dim_conf, _dim_conf);
            VectorXd dmu_i = E_Xphi_mat.segment(index*_dim_conf, _dim_conf);

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


    // VectorXd theta = this->_precision * this->_mu;
    // for(int i = 0; i < n_nonlinear; i++){
    //     if (nonlinear_fac_cost(i) > nonlinear_fac_cost.maxCoeff() / 2){
    //         std::cout << "Factor " << i+1 << " cost: " << nonlinear_fac_cost(i) << ", dmu: " << -_Vdmu.segment((i+1)*this->_dim_state, this->_dim_state).transpose() / this->_temperature << ", current: " << theta.segment((i+1)*this->_dim_state, this->_dim_state).transpose()/1.35 << std::endl;
    //     }
    // }

    return std::make_tuple(cost, fac_costs, _Vdmu, _Vddmu);
}


template <typename Factor, typename CudaClass>
inline void ProxKLGH<Factor, CudaClass>::update_proposal(const VectorXd& new_mu, const SpMat& new_precision)
{
    Base::set_mu(new_mu);
    Base::set_precision(new_precision);
}


template <typename Factor, typename CudaClass>
double ProxKLGH<Factor, CudaClass>::cost_value_no_entropy()
{
    SpMat Cov = this->inverse(this->_precision);

    double value = 0.0;
    for (auto &opt_k : this->_vec_factors)
    {
        value += opt_k->fact_cost_value(this->_mu, Cov);
    }
    return value; // / _temperature;
}


template <typename Factor, typename CudaClass>
double ProxKLGH<Factor, CudaClass>::KL_Divergence(const VectorXd& mean_current, const SpMat& precision_current, const SpMat& covariance_current, const VectorXd& mean_new, const SpMat& precision_new)
{
    SparseLDLT ldlt_current(precision_current);
    SparseLDLT ldlt_new(precision_new);

    VectorXd vec_D_current = ldlt_current.vectorD();
    VectorXd vec_D_new = ldlt_new.vectorD();

    double trace_term = 0;
    for (int k = 0; k < precision_new.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(precision_new, k); it; ++it) {
            int i = it.row();
            int j = it.col();
            trace_term += it.value() * covariance_current.coeff(i, j);
        }
    }

    VectorXd diff = mean_new - mean_current;
    double quadratic_term = diff.transpose() * precision_new * diff;

    double log_term = vec_D_current.array().log().sum() - vec_D_new.array().log().sum();

    double KL = (trace_term + quadratic_term + log_term - mean_current.size()) / 2.0;

    // if (KL < 0){
    //     std::cout << "KL Divergence: " << KL << std::endl;
    //     std::cout << "trace_term: " << trace_term << std::endl;
    //     std::cout << "quadratic_term: " << quadratic_term << std::endl;
    //     std::cout << "log_term: " << log_term << std::endl;
    //     std::cout << "mean_current: " << -mean_current.size() << std::endl << std::endl;
    // }

    return KL;
}

// There might be some problem in the KL divergence calculation
template <typename Factor, typename CudaClass>
double ProxKLGH<Factor, CudaClass>::KL_Divergence_general(const VectorXd& mean_former, const VectorXd& mean_latter, const SpMat& precision_former, const SpMat& precision_latter)
{
    Timer timer;

    // Compute the KL divergence
    int dim_state = 6;
    timer.start();
    SparseLDLT ldlt_former(precision_former);
    SparseLDLT ldlt_latter(precision_latter);

    VectorXd vec_D_former = ldlt_former.vectorD();
    VectorXd vec_D_latter = ldlt_latter.vectorD();
    std::cout << "Time for computing LDLT: " << timer.end_mus_output() << " us" << std::endl;

    // There's some problem with GBP inverse here, find another method that works fast and accurately
    timer.start();
    SpMat covariance_former = this->inverse_GBP(precision_former);
    std::cout << "Time for computing GBP inverse: " << timer.end_mus_output() << " us" << std::endl;

    timer.start();
    double trace_term = 0;
    for (int k = 0; k < precision_latter.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(precision_latter, k); it; ++it) {
            int i = it.row();
            int j = it.col();
            trace_term += it.value() * covariance_former.coeff(i, j);
        }
    }
    std::cout << "Time for computing trace term: " << timer.end_mus_output() << " us" << std::endl;

    // SpMat covariance = this -> inverse(precision_former);

    // // These two methods are equivalent, but now the result is different
    // SpMat covariance = this -> inverse(precision_former);
    // SpMat precision_prior_times_Cov = precision_latter * covariance;
    // double trace_term_inverse = precision_prior_times_Cov.diagonal().sum();

    timer.start();
    double quadratic_term = (mean_latter - mean_former).transpose() * precision_latter * (mean_latter - mean_former);
    std::cout << "Time for computing quadratic term: " << timer.end_mus_output() << " us" << std::endl;

    timer.start();
    double log_term = vec_D_former.array().log().sum() - vec_D_latter.array().log().sum();
    std::cout << "Time for computing log term: " << timer.end_mus_output() << " us" << std::endl;

    // std::cout << "vec_D_former min: " << vec_D_former.minCoeff() << std::endl;
    // std::cout << "vec_D_former max: " << vec_D_former.maxCoeff() << std::endl;
    // std::cout << "new entropy: " << vec_D_former.array().log().sum()/2 << std::endl;
    // std::cout << "current entropy: " << vec_D_latter.array().log().sum()/2 << std::endl;

    double KL = (trace_term + quadratic_term + log_term - mean_former.size()) / 2;

    if (KL < 0){
        std::cout << "trace_term: " << trace_term << std::endl;
        // std::cout << "trace_term_inverse: " << trace_term_inverse << std::endl;
        std::cout << "quadratic_term: " << quadratic_term << std::endl;
        std::cout << "log_term: " << log_term << std::endl;
        std::cout << "mean_former: " << -mean_former.size() << std::endl;
        std::cout << "KL Divergence: " << KL << std::endl << std::endl;
        // for (int i = 0; i < 50; i++){
        //     MatrixXd Cov_i_GBP = covariance_former.block(i*dim_state, i*dim_state, dim_state, dim_state);
        //     MatrixXd Cov_i = covariance.block(i*dim_state, i*dim_state, dim_state, dim_state);
        //     std::cout << "Covariance " << i << " difference: " << (Cov_i - Cov_i_GBP).norm() / Cov_i.norm() << std::endl;
        // }
    }

    // std::cout << "Error of trace term: " << (trace_term - trace_term_1) / trace_term * 100 << "%" << std::endl;
    // std::cout << "Error of different inverse: " << (trace_term_1 - trace_term_2) / trace_term_1 * 100 << "%" << std::endl;

    return KL;
}


}

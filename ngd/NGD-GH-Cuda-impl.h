
#pragma once

#ifndef NGD_GH_IMPL_H
#define NGD_GH_IMPL_H

using namespace Eigen;
#include <stdexcept>
#include <optional>
#include <omp.h>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace gvi{

/**
 * @brief One step of optimization.
 */
template <typename Factor>
std::tuple<VectorXd, SpMat> NGDGH<Factor>::compute_gradients(std::optional<double>step_size){
    Base::_Vdmu.setZero();
    Base::_Vddmu.setZero();

    VectorXd Vdmu_sum = VectorXd::Zero(Base::_Vdmu.size());
    SpMat Vddmu_sum = SpMat(Base::_Vddmu.rows(), Base::_Vddmu.cols());

    /**
     * @brief OMP parallel on cpu.
     */
    omp_set_num_threads(20); 

    #pragma omp parallel
    {
        // Thread-local storage to avoid race conditions
        VectorXd Vdmu_private = VectorXd::Zero(Base::_Vdmu.size());
        SpMat Vddmu_private = SpMat(Base::_Vddmu.rows(), Base::_Vddmu.cols());

        #pragma omp for nowait // Nowait allows threads to continue without waiting at the end of the loop
        for (auto &opt_k : Base::_vec_factors) {
            opt_k->calculate_partial_V();
            Vdmu_private += opt_k->local2joint_dmu();
            Vddmu_private += opt_k->local2joint_dprecision();
        }

        #pragma omp critical
        {
            Vdmu_sum += Vdmu_private;
            Vddmu_sum += Vddmu_private;
        }
    }

    // Update the member variables 
    Base::_Vdmu = Vdmu_sum;
    Base::_Vddmu = Vddmu_sum;

    // std::cout << "Vdmu" << _Vdmu << std::endl;

    SpMat dprecision = Base::_Vddmu - Base::_precision;

    Eigen::ConjugateGradient<SpMat, Eigen::Upper> solver;
    VectorXd dmu = solver.compute(Base::_Vddmu).solve(-Base::_Vdmu);

    return std::make_tuple(dmu, dprecision);
}

template <typename Factor>
std::tuple<double, VectorXd, SpMat> NGDGH<Factor>::onestep_linesearch(const double &step_size, 
                                                                        const VectorXd& dmu, 
                                                                        const SpMat& dprecision
                                                                        )
{

    SpMat new_precision; 
    VectorXd new_mu; 
    new_mu.setZero(); new_precision.setZero();

    // update mu and precision matrix
    new_mu = this->_mu + step_size * dmu;
    new_precision = this->_precision + step_size * dprecision;

    double new_cost;
    if (isPositiveDefinite(new_precision))
        new_cost = Base::cost_value_cuda(new_mu, new_precision);
    else
        new_cost = std::numeric_limits<double>::infinity();
    
    // std::cout << "New cost = " << new_cost << std::endl;
    return std::make_tuple(new_cost, new_mu, new_precision);
}

template <typename Factor>
bool NGDGH<Factor>::isPositiveDefinite(const SpMat& precision)
{
    SparseLDLT ldlt(precision);
    VectorXd diag = ldlt.vectorD();
    return (diag.array() > 0).all();
}

template <typename Factor>
double NGDGH<Factor>::bisection_stepsize(const VectorXd& dmu, const SpMat& dprecision)
{   
    SpMat new_precision; 
    VectorXd new_mu; 
    new_mu.setZero(); new_precision.setZero();

    double log_lower = -2;
    double log_upper = 2;

    double log_threshold = 0.01; 
    double epsilon = 100;

    while (log_upper - log_lower > log_threshold) 
    {
        double log_mid = (log_lower + log_upper) / 2;
        double step_size = std::exp(log_mid);

        new_mu = this->_mu + step_size * dmu;
        new_precision = this->_precision + step_size * dprecision;

        // Compute KL divergence and check for PD indirectly
        double KL = KL_Divergence(new_mu, this->_mu, new_precision, this->_precision);
        std::cout << "KL Divergence: " << KL << std::endl;

        if (std::isnan(KL) || KL >= epsilon) {
            log_upper = log_mid;
        } else {
            log_lower = log_mid;
        }
    }

    double final_step_size = std::exp((log_lower + log_upper) / 2);

    return final_step_size;
}

template <typename Factor>
double NGDGH<Factor>::KL_Divergence(const VectorXd& mean_former, const VectorXd& mean_latter, const SpMat& precision_former, const SpMat& precision_latter)
{
    Timer timer;
    // Compute the KL divergence
    timer.start();
    SparseLDLT ldlt_former(precision_former);
    SparseLDLT ldlt_latter(precision_latter);

    VectorXd vec_D_former = ldlt_former.vectorD();
    VectorXd vec_D_latter = ldlt_latter.vectorD();
    std::cout << "LDLT time: " << timer.end_mus_output() << " us" << std::endl;

    timer.start();
    SpMat covariance_former = this->inverse_GBP(precision_former);    
    std::cout << "Inverse time: " << timer.end_mus_output() << " us" << std::endl;
    
    timer.start();
    double trace_term = 0;
    for (int k = 0; k < precision_latter.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(precision_latter, k); it; ++it) {
            int i = it.row();
            int j = it.col();
            trace_term += it.value() * covariance_former.coeff(i, j);
        }
    }
    std::cout << "Trace time: " << timer.end_mus_output() << " us" << std::endl;

    timer.start();
    double quadratic_term = (mean_latter - mean_former).transpose() * precision_latter * (mean_latter - mean_former);
    std::cout << "Quadratic time: " << timer.end_mus_output() << " us" << std::endl;

    timer.start();
    double log_term = vec_D_former.array().log().sum() - vec_D_latter.array().log().sum();
    std::cout << "Log time: " << timer.end_mus_output() << " us" << std::endl;

    double KL = (trace_term + quadratic_term + log_term - mean_former.size()) / 2;
    return KL;
}


template <typename Factor>
inline void NGDGH<Factor>::update_proposal(const VectorXd& new_mu, const SpMat& new_precision)
{
    Base::set_mu(new_mu);
    Base::set_precision(new_precision);
}


/**
 * @brief Compute the costs of all factors, using current values.
 */
template <typename Factor>
VectorXd NGDGH<Factor>::factor_cost_vector()
{   
    return Base::factor_cost_vector(this->_mu, this->_precision);
}

template <typename Factor>
std::tuple<double, VectorXd, VectorXd, SpMat> NGDGH<Factor>::factor_cost_vector_cuda()
{   
    return Base::factor_cost_vector_cuda(this->_mu, this->_precision);
}

template <typename Factor>
std::tuple<double, VectorXd, VectorXd, SpMat> NGDGH<Factor>::factor_cost_vector_cuda_time()
{   
    return Base::factor_cost_vector_cuda_time(this->_mu, this->_precision);
}

/**
 * @brief Compute the total cost function value given a state, using current values.
 */
template <typename Factor>
double NGDGH<Factor>::cost_value()
{
    return Base::cost_value(this->_mu, this->_precision);
}


template <typename Factor>
double NGDGH<Factor>::cost_value_cuda()
{
    return Base::cost_value_cuda(this->_mu, this->_precision);
}

/**
 * @brief given a state, compute the total cost function value without the entropy term, using current values.
 */
template <typename Factor>
double NGDGH<Factor>::cost_value_no_entropy()
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


#endif // NGD_GH_IMPL_H
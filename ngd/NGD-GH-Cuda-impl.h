
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
    _Vdmu.setZero();
    _Vddmu.setZero();

    VectorXd Vdmu_sum = VectorXd::Zero(_Vdmu.size());
    SpMat Vddmu_sum = SpMat(_Vddmu.rows(), _Vddmu.cols());

    /**
     * @brief OMP parallel on cpu.
     */
    omp_set_num_threads(20); 

    #pragma omp parallel
    {
        // Thread-local storage to avoid race conditions
        VectorXd Vdmu_private = VectorXd::Zero(_Vdmu.size());
        // VectorXd Vdmu_private_nonlinear = VectorXd::Zero(_Vdmu.size());
        SpMat Vddmu_private = SpMat(_Vddmu.rows(), _Vddmu.cols());
        // SpMat Vddmu_private_nonlinear = SpMat(_Vddmu.rows(), _Vddmu.cols());

        #pragma omp for nowait // Nowait allows threads to continue without waiting at the end of the loop
        for (auto &opt_k : Base::_vec_factors) {
            opt_k->calculate_partial_V();
            Vdmu_private += opt_k->local2joint_dmu();
            Vddmu_private += opt_k->local2joint_dprecision();
        }

        // #pragma omp for nowait // Nowait allows threads to continue without waiting at the end of the loop
        // for (auto &opt_k : Base::_vec_nonlinear_factors) {
        //     auto result = opt_k->derivatives();
        //     Vdmu_private_nonlinear += opt_k->local2joint_dmu();
        //     Vddmu_private_nonlinear += opt_k->local2joint_dprecision();
        // }

        #pragma omp critical
        {
            Vdmu_sum += Vdmu_private;
            Vddmu_sum += Vddmu_private;
            // Vdmu_sum += Vdmu_private_nonlinear;
            // Vddmu_sum += Vddmu_private_nonlinear;
        }
    }

    // Update the member variables 
    _Vdmu = Vdmu_sum;
    _Vddmu = Vddmu_sum;

    // std::cout << "Vdmu" << _Vdmu << std::endl;

    SpMat dprecision = _Vddmu - Base::_precision;

    Eigen::ConjugateGradient<SpMat, Eigen::Upper> solver;
    VectorXd dmu =  solver.compute(_Vddmu).solve(-_Vdmu);

    return std::make_tuple(dmu, dprecision);
}


template <typename Factor>
void NGDGH<Factor>::derivatives(VectorXd& dmu_mat, MatrixXd& ddmu_mat, int sigma_cols){
    _Vdmu.setZero();
    _Vddmu.setZero();

    VectorXd Vdmu(dmu_mat.size());
    MatrixXd Vddmu(ddmu_mat.rows(), ddmu_mat.cols());

    for (auto &opt_k : Base::_vec_factors) {
        opt_k->calculate_partial_V();
        Vdmu += opt_k->local2joint_dmu();
        Vddmu += opt_k->local2joint_dprecision();
    }

    // for (int i = 0; i < Base::_vec_factors.size(); i++){
    //     auto result = Base::_vec_factors[i]->derivatives();
    //     Vdmu.segment(i*sigma_cols, sigma_cols) = std::get<1>(result);
    //     Vddmu.block(0, i*sigma_cols, sigma_cols, sigma_cols) = std::get<2>(result);
    // }

    // std::cout << "Vdmu Error" << std::endl << (Vdmu - dmu_mat) << std::endl;
    // std::cout << "Vddmu Error" << std::endl << (Vddmu - ddmu_mat) << std::endl;
    // std::cout << "Vddmu" << std::endl << Vddmu.transpose() << std::endl;

    // dmu_mat = Vdmu;
    // ddmu_mat = Vddmu

    // SpMat dprecision = _Vddmu - Base::_precision;

    // Eigen::ConjugateGradient<SpMat, Eigen::Upper> solver;
    // VectorXd dmu =  solver.compute(_Vddmu).solve(-_Vdmu);

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

    // new cost
    double new_cost = Base::cost_value_cuda(new_mu, new_precision);
    return std::make_tuple(new_cost, new_mu, new_precision);

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
std::tuple<VectorXd, VectorXd, SpMat> NGDGH<Factor>::time_test_cuda()
{   
    return Base::time_test_cuda(this->_mu, this->_precision);
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
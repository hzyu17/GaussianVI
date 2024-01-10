
#pragma once

#ifndef NGD_GH_IMPL_H
#define NGD_GH_IMPL_H

using namespace Eigen;
using namespace std;
#include <stdexcept>
#include <optional>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace gvi{

/**
 * @brief One step of optimization.
 */
template <typename Factor>
std::tuple<VectorXd, SpMat> NGDGH<Factor>::compute_gradients(){
    _Vdmu.setZero();
    _Vddmu.setZero();

    for (auto &opt_k : Base::_vec_factors)
    {
        opt_k->calculate_partial_V();
        _Vdmu = _Vdmu + opt_k->local2joint_dmu();
        _Vddmu = _Vddmu + opt_k->local2joint_dprecision();
    }

    SpMat dprecision = _Vddmu - Base::_precision;

    Eigen::ConjugateGradient<SpMat, Eigen::Upper> solver;
    VectorXd dmu =  solver.compute(_Vddmu).solve(-_Vdmu);

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

    // new cost
    double new_cost = Base::cost_value(new_mu, new_precision);
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

/**
 * @brief Compute the total cost function value given a state, using current values.
 */
template <typename Factor>
double NGDGH<Factor>::cost_value()
{
    return Base::cost_value(this->_mu, this->_precision);
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
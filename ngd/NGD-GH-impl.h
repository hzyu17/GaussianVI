
#pragma once

#ifndef NGD_GH_IMPL_H
#define NGD_GH_IMPL_H

using namespace Eigen;
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

}


#endif // NGD_GH_IMPL_H
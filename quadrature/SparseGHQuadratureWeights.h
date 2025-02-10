/**
 * @file SparseGHQuadratureWeights.h
 * @author Hongzhe Yu
 * @date 02/03/2024
 * @brief Define two names related to the storage of the quadrature sigma points and weights.
*/

#pragma once

#include "helpers/SerializeEigenMaps.h"

namespace gvi{

using DimDegTuple = std::tuple<double, double>;
using PointsWeightsTuple = std::tuple<Eigen::MatrixXd, Eigen::VectorXd>;
using QuadratureWeightsMap = std::unordered_map<DimDegTuple, PointsWeightsTuple>;

using PointsWeightsTuple_MKL = std::tuple<std::vector<double>, std::vector<double>>;
using QuadratureWeightsMap_MKL = std::unordered_map<DimDegTuple, PointsWeightsTuple_MKL>;

}
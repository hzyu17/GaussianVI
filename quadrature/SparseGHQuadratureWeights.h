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
using NodesWeightsTuple = std::tuple<Eigen::MatrixXd, Eigen::VectorXd>;
using QuadratureWeightsMap = std::unordered_map<DimDegTuple, NodesWeightsTuple>;


}
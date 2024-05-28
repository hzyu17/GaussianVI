/**
 * @file ProxGVIFactorizedSimpleGH.h
 * @author Hongzhe Yu (hyu419@getach.edu)
 * @brief Simple optimizer just to verify the algorithm.
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "proxgd/ProxGVIFactorizedBaseGH.h"
#include <memory>

using namespace Eigen;

namespace gvi{

using ProxGVIFactorizedSimpleGH = ProxGVIFactorizedBaseGH<NoneType>;

} // namespace
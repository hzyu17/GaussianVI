/**
 * @file NGDFactorizedSimpleGH.h
 * @author Hongzhe Yu (hyu419@getach.edu)
 * @brief Simple optimizer just to verify the algorithm itself.
 * @version 0.1
 * @date 2022-07-25
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "ngd/NGDFactorizedBaseGH_Cuda.h"
#include <memory>

using namespace Eigen;

namespace gvi{

using NGDFactorizedSimpleGH_Cuda = NGDFactorizedBaseGH_Cuda<NoneType>;

} // namespace


#include "ngd/NGDFactorizedLinear.h"
#include "ngd/NGDFactorizedFixedGaussian.h"

namespace gvi{
    using FixedGpPrior = NGDFactorizedFixedGaussian<FixedPriorGP>;
    using LinearGpPrior = NGDFactorizedLinear<MinimumAccGP>;
}
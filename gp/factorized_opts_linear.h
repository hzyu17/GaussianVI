

#include "ngd/NGDFactorizedLinear.h"
#include "ngd/NGDFactorizedFixedGaussian.h"
#include "gp/fixed_prior.h"
#include "gp/minimum_acc_prior.h"

namespace gvi{
    using FixedGpPrior = NGDFactorizedFixedGaussian<FixedPriorGP>;
    using LinearGpPrior = NGDFactorizedLinear<MinimumAccGP>;
}
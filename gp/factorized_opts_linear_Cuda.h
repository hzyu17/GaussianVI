

#include "ngd/NGDFactorizedLinear_Cuda.h"
#include "ngd/NGDFactorizedLinearGH_Cuda.h"
#include "gp/fixed_prior.h"
#include "gp/minimum_acc_prior.h"

namespace gvi{
    using FixedGpPrior = NGDFactorizedLinear_Cuda<FixedPriorGP>;
    using LinearGpPrior = NGDFactorizedLinear_Cuda<MinimumAccGP>;

    // For comparison
    using FixedGpPriorGH = NGDFactorizedLinearGH_Cuda<FixedPriorGP>;
    using LinearGpPriorGH = NGDFactorizedLinearGH_Cuda<MinimumAccGP>;
}
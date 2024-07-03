

#include "ngd/NGDFactorizedLinear.h"
#include "ngd/NGDFactorizedLinearGH.h"
#include "gp/fixed_prior.h"
#include "gp/minimum_acc_prior.h"

namespace gvi{
    using FixedGpPrior = NGDFactorizedLinear<FixedPriorGP>;
    using LinearGpPrior = NGDFactorizedLinear<MinimumAccGP>;

    // For comparison
    using FixedGpPriorGH = NGDFactorizedLinearGH<FixedPriorGP>;
    using LinearGpPriorGH = NGDFactorizedLinearGH<MinimumAccGP>;
}
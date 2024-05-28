

#include "proxgd/ProxGVIFactorizedLinear.h"
#include "gp/fixed_prior.h"
#include "gp/minimum_acc_prior.h"

namespace gvi{
    using ProxFixedGpPrior = ProxFactorizedLinear<FixedPriorGP>;
    using ProxLinearGpPrior = ProxFactorizedLinear<MinimumAccGP>;
}
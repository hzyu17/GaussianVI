/**
 * @file ProxGVIFactorizedBase.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief The base class for proximal gradient descent factorized optimizer.
 * @version 0.1
 * @date 2024-01-09
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "gvibase/GVIFactorizedBase.h"

namespace gvi{
class ProxGVIFactorizedBase : public GVIFactorizedBase{

public:

    ProxGVIFactorizedBase(int dimension, int state_dim, int num_states, int start_index, 
                        double temperature=10.0, double high_temperature=100.0, bool is_linear=false):
            GVIFactorizedBase(dimension, state_dim, num_states, start_index)
            {}

    /**
     * @brief Calculating (dmu, d_covariance) in the factor level.
     * \hat b_k = K^{-1}(\mu_\theta - \mu) + \mE[h(x)*\Sigma_\theta(x-\mu_\theta)]
     * \hat S_k = K^{-1} + \mE[\Sigma_\theta(x-\mu_\theta)@(\nabla(h(x)).T)]
     */
    void calculate_partial_V() override{
        
    }
    
    inline VectorXd local2joint_dmu() override{ 
        VectorXd res(this->_joint_size);
        res.setZero();
        this->_block.fill_vector(res, this->_Vdmu);
        return res;
    }

    inline SpMat local2joint_dcovariance() override{ 
        SpMat res(this->_joint_size, this->_joint_size);
        res.setZero();
        this->_block.fill(this->_Vddmu, res);
        return res;
    }

};

} //namespace
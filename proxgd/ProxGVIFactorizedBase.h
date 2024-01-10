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

class ProxGVIFactorizedBase : public GVIFactorizedBase{

public:

    ProxGVIFactorizedBase(int dimension, int state_dim, int num_states, int start_index, 
                        double temperature=10.0, double high_temperature=100.0, bool is_linear=false):
            GVIFactorizedBase(dimension, state_dim, num_states, start_index),
            _Vdmu(_dim),
            _Vddmu(_dim, _dim)
            {}

    /**
     * @brief Calculating (dmu, d_covariance) in the factor level.
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

    inline MatrixXd E_xMuPhi(){
        return _gh->Integrate(_func_Vmu);
    }

    inline MatrixXd E_xMuxMuTPhi(){
        return _gh->Integrate(_func_Vmumu);
    }

protected:
    VectorXd _Vdmu;
    MatrixXd _Vddmu;

    GHFunction _func_Vmu;
    GHFunction _func_Vmumu;
};

/**
 * @file GVIFactorizedBaseGH.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief The base class for one factor with a GH-quadrature.
 * @version 1.1
 * @date 2024-04-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include "GVIFactorizedBase_Cuda.h"

using namespace Eigen;

namespace gvi{

class GVIFactorizedBaseGH_Cuda: public GVIFactorizedBase_Cuda{
using Base = GVIFactorizedBase_Cuda;

public:
    virtual ~GVIFactorizedBaseGH_Cuda(){}
    /**
     * @brief Default Constructor
     */
    GVIFactorizedBaseGH_Cuda(){}

    /**
     * @brief Construct a new GVIFactorizedBase object
     * 
     * @param dimension The dimension of the state
     */
    GVIFactorizedBaseGH_Cuda(int dimension, int state_dim, int num_states, int start_index, 
                        double temperature, double high_temperature, 
                        std::shared_ptr<QuadratureWeightsMap> weight_sigpts_map_option):
            Base(dimension, state_dim, num_states, start_index, temperature, high_temperature){
            }
            
public:

    /// update the GH approximator
    void updateGH(const VectorXd& x, const MatrixXd& P){
        // This order cannot be changed! Need to update P before updating mean.
        _gh->update_P(P); 
        _gh->update_mean(x);
        _gh->update_sigmapoints();
    }

    /**
     * @brief returns the E_q{phi(x)} = E_q{-log(p(x,z))}
     */
    inline double E_Phi() {
        return _gh->Integrate(_func_phi)(0, 0);
    }

    inline MatrixXd E_xMuPhi(){
        return _gh->Integrate(_func_Vmu);
    }

    inline MatrixXd E_xMuxMuTPhi(){
        return _gh->Integrate(_func_Vmumu);
    }

    void set_GH_points(int p){
        _gh->set_polynomial_deg(p);
    }
    
public:
    // The function (x-\mu)*\phi(x)
    GHFunction _func_Vmu;
    // The function [(x-\mu)@(x-\mu)^T]*\phi(x)
    GHFunction _func_Vmumu;

    /// G-H quadrature class
    std::shared_ptr<GH> _gh;

};

}

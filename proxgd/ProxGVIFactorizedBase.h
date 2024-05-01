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
     * \hat b_k = K^{-1}(\mu_\theta - \mu) + \Sigma_\theta\mE[h(x)*(x-\mu_\theta)]
     * \hat S_k = K^{-1} + \Sigma_\theta\mE[(x-\mu_\theta)@(\nabla(h(x)).T)]
     */
    void calculate_partial_V() override{
        // Compute the BW gradients
        Eigen::VectorXd bk = this->_precision * this->_gh->Integrate(this->_func_Vmu);
        Eigen::MatrixXd Sk = this->_precision * this->_gh->Integrate(this->_func_Vmumu) * this->_precision - this->_precision*this->_gh->Integrate(this->_func_phi);

        Eigen::MatrixXd Identity{Eigen::MatrixXd::Identity(this->_dim)};

        // Compute the proximal step
        Eigen::MatrixXd Mk{Identity - Sk};
        Eigen::Sigk{this->_covariance};

        Eigen::MatrixXd Sigk_half{Sigk - this->_step_size_mu* Mk*Sigk*Mk.transpose()};
        
        MatrixXd temp{Identity};
        temp = Sigk_half*(Sigk_half + 4*this->_step_size_mu*Identity)

        MatrixXd temp_2(this->_dim, this->_dim);
        temp_2.setZero();
        temp_2 = this->sqrtm(temp);

        Eigen::VectorXd mk_new = this->_mu - this->_step_size_mu * bk;
        Eigen::MatrixXd Sigk_new = 0.5*Sigk_half + this->_step_size_mu*Identity + 0.5*temp_2;

        this->_Vdmu = (mk_new - this->_mu) / this->_step_size_mu;
        this->_Vddmu = (Sigk_new - Sigk) / this->_step_size_mu;

        //TODO: the variations wrt the information matrix. 

    }

    Eigen::MatrixXd sqrtm(const Eigen::MatrixXd& mat){
        assert(mat.rows() == mat.cols());

        // Compute the Schur decomposition
        Eigen::RealSchur<Eigen::MatrixXd> schur(mat.rows());
        schur.compute(mat);

        const Eigen::MatrixXd& T = schur.matrixT();
        const Eigen::MatrixXd& U = schur.matrixU();

        Eigen::MatrixXd T_sqrt = T;
        
        // Since T is quasi-triangular, we can directly compute the square root of diagonal 2x2 blocks
        int n = T.rows();
        for (int i = 0; i < n;) {
            if (i == n - 1 || T(i + 1, i) == 0) {
                // Diagonal block is 1x1
                T_sqrt(i, i) = std::sqrt(T(i, i));
                i++;
            } else {
                // Diagonal block is 2x2
                Eigen::Matrix2d block = T.block<2, 2>(i, i);
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(block);
                Eigen::Matrix2d D_sqrt = es.eigenvalues().cwiseSqrt().asDiagonal();
                Eigen::Matrix2d S = es.eigenvectors();
                Eigen::Matrix2d S_inv = S.inverse();
                T_sqrt.block<2, 2>(i, i) = S * D_sqrt * S_inv;
                i += 2;
            }
        }

        // Compute the square root of the original matrix
        Eigen::MatrixXd mat_sqrt = U * T_sqrt * U.transpose();
        return mat_sqrt;
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
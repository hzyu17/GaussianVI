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

#include "gvibase/GVIFactorizedBaseGH.h"

namespace gvi{

template <typename CostClass = NoneType>
class ProxGVIFactorizedBaseGH : public GVIFactorizedBaseGH{

    using GVIBase = GVIFactorizedBaseGH;
    using Function = std::function<double(const VectorXd&, const CostClass &)>;

public:

    ProxGVIFactorizedBaseGH(int dimension, int state_dim, int gh_degree, 
                        const Function& function, const CostClass& cost_class,
                        int num_states, int start_index, 
                        double temperature=1.0, double high_temperature=10.0,
                        std::optional<std::shared_ptr<QuadratureWeightsMap>> weight_sigpts_map_option=std::nullopt):
                GVIBase(dimension, state_dim, num_states, start_index, temperature, high_temperature),
                _bk(dimension),
                _Sk(dimension, dimension)
            {
                /// Override of the GVIBase classes.
                GVIBase::_func_phi = [this, function, cost_class](const VectorXd& x){return MatrixXd::Constant(1, 1, function(x, cost_class));};
                GVIBase::_func_Vmu = [this, function, cost_class](const VectorXd& x){return (x-GVIBase::_mu) * function(x, cost_class);};
                GVIBase::_func_Vmumu = [this, function, cost_class](const VectorXd& x){return MatrixXd{(x-GVIBase::_mu) * (x-GVIBase::_mu).transpose().eval() * function(x, cost_class)};};
                GVIBase::_gh = std::make_shared<GH>(GH{gh_degree, GVIBase::_dim, GVIBase::_mu, GVIBase::_covariance, weight_sigpts_map_option});
                _bk.setZero();
                _Sk.setZero();
            }

    /**
     * @brief Calculating (dmu, d_covariance) in the factor level.
     * \hat b_k = K^{-1}(\mu_\theta - \mu) + \Sigma_\theta\mE[h(x)*(x-\mu_\theta)]
     * \hat S_k = K^{-1} + \Sigma_\theta\mE[(x-\mu_\theta)@(\nabla(h(x)).T)]
     */
    void calculate_partial_V(std::optional<double> stepsize_option=std::nullopt) override{

        // this->compute_BW_grads();
        // std::tuple<Eigen::VectorXd, Eigen::MatrixXd> gradients;

        // if (stepsize_option.has_value()){
        //     gradients = BW_JKO(stepsize_option.value());
        // }else{
        //     std::cout << "no step size input, using the default value in the class!" << std::endl;
        //     gradients = BW_JKO(this->_step_size);
        // }
        
        // this->_Vdmu = std::get<0>(gradients);
        // this->_Vddmu = std::get<1>(gradients);

        // update the mu and sigma inside the gauss-hermite integrator
        updateGH(this->_mu, this->_covariance);

        this->_Vdmu.setZero();
        this->_Vddmu.setZero();

        /// Integrate for E_q{_Vdmu} 
        this->_Vdmu = this->_precision * this->_gh->Integrate(this->_func_Vmu) / this->temperature();

        /// Integrate for E_q{phi(x)}
        double E_phi = this->_gh->Integrate(this->_func_phi)(0, 0);
        
        /// Integrate for partial V^2 / ddmu_ 
        MatrixXd E_xxphi{this->_gh->Integrate(this->_func_Vmumu)};

        this->_Vddmu.triangularView<Upper>() = (this->_precision * E_xxphi * this->_precision - this->_precision * E_phi).triangularView<Upper>();
        this->_Vddmu.triangularView<StrictlyLower>() = this->_Vddmu.triangularView<StrictlyUpper>().transpose();
        this->_Vddmu = this->_Vddmu / this->temperature();
        
    }


    std::tuple<Eigen::VectorXd, Eigen::MatrixXd> BW_JKO(double step_size){
        Eigen::MatrixXd Identity(this->_dim, this->_dim);
        Identity.setZero();
        Identity = Eigen::MatrixXd::Identity(this->_dim, this->_dim);
        
        // Compute the proximal step
        Eigen::MatrixXd Mk(this->_dim, this->_dim);
        Mk.setZero();
        Eigen::MatrixXd Sigk(this->_dim, this->_dim);
        Sigk.setZero();

        Mk = Identity - step_size*this->_Sk;
        Sigk = this->_covariance;

        bool Sk_symm = isSymmetric(this->_Sk);
        bool Mk_symm = isSymmetric(Mk);

        Eigen::MatrixXd Sigk_half(this->_dim, this->_dim);
        Sigk_half.setZero();
        Sigk_half = Mk*Sigk*Mk.transpose();
        
        // Eigen::MatrixXd temp(this->_dim, this->_dim);
        // temp.setZero();
        // temp = Sigk_half*(Sigk_half + 4.0*step_size*Identity);

        // bool is_temp_spd = isSymmetricPositiveDefinite(temp);
        // std::cout << "is_temp_spd: " << is_temp_spd << std::endl;

        // temp = this->sqrtm(temp);

        Eigen::VectorXd mk_new = this->_mu - step_size * this->_bk;
        Eigen::MatrixXd Sigk_new(this->_dim, this->_dim);
        Sigk_new.setZero();
        Sigk_new = Sigk_half;

        // Sigk_new = 0.5*Sigk_half + step_size*Identity + 0.5*temp;

        // bool is_Sigk_new_spd = isSymmetricPositiveDefinite(Sigk_new);
        // std::cout << "is_inv_Sigk_new_spd: " << is_Sigk_new_spd << std::endl;

        Eigen::MatrixXd Precision_new(this->_dim, this->_dim);
        Precision_new.setZero();
        Precision_new = Sigk_new.inverse();

        Eigen::VectorXd Vdmu = mk_new - this->_mu;
        Eigen::MatrixXd Vddmu = Precision_new - this->_precision;
        
        return std::make_tuple(Vdmu, Vddmu);
    }

    void compute_BW_grads(){
        // Compute the BW gradients
        _bk.setZero();
        _Sk.setZero();

        _bk = this->_precision * this->_gh->Integrate(this->_func_Vmu);
        _Sk = this->_precision * this->_gh->Integrate(this->_func_Vmumu) * this->_precision - this->_precision*this->_gh->Integrate(this->_func_phi)(0,0);

    }
    
    inline VectorXd local2joint_dmu() override{ 
        VectorXd res(this->_joint_size);
        res.setZero();
        this->_block.fill_vector(res, this->_Vdmu);
        return res;
    }

    inline SpMat local2joint_dprecision() override{ 
        SpMat res(this->_joint_size, this->_joint_size);
        res.setZero();
        this->_block.fill(this->_Vddmu, res);
        return res;
    }

    // inline VectorXd local2joint_dmu(Eigen::VectorXd & dmu_lcl){ 
    //     VectorXd res(this->_joint_size);
    //     res.setZero();
    //     this->_block.fill_vector(res, dmu_lcl);
    //     return res;
    // }

    // inline SpMat local2joint_dprecision(Eigen::MatrixXd & dprecision_lcl){ 
    //     SpMat res(this->_joint_size, this->_joint_size);
    //     res.setZero();
    //     this->_block.fill(dprecision_lcl, res);
    //     return res;
    // }

    // Helper functions
    bool isSymmetric(const Eigen::MatrixXd& matrix, double precision = 1e-10) {
        return (matrix - matrix.transpose()).cwiseAbs().maxCoeff() <= precision;
    }

    // Function to check if a matrix is symmetric positive-definite
    bool isSymmetricPositiveDefinite(const Eigen::MatrixXd& matrix, double precision = 1e-10) {
        if (!isSymmetric(matrix, precision)) {
            return false;  // Matrix is not symmetric
        }

        // Compute the eigenvalues using the Eigen solver
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(matrix);
        if (eigenSolver.info() != Eigen::Success) {
            throw std::runtime_error("Eigenvalue computation did not converge");
        }

        // Check if all eigenvalues are non-negative
        for (int i = 0; i < matrix.rows(); ++i) {
            if (eigenSolver.eigenvalues()[i] <= precision) {
                return false;  // Found a negative eigenvalue
            }
        }
        return true;
    }


    double fact_cost_value(const VectorXd& fill_joint_mean, const SpMat& joint_cov) override {
        VectorXd mean_k = extract_mu_from_joint(fill_joint_mean);
        MatrixXd Cov_k = extract_cov_from_joint(joint_cov);

        updateGH(mean_k, Cov_k);

        return this->_gh->Integrate(this->_func_phi)(0, 0);
    }

protected:
    Eigen::VectorXd _bk;
    Eigen::MatrixXd _Sk;

};

} //namespace
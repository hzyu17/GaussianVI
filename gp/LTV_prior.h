/**
 * @file LTV_prior.h
 * @author Zinuo Chang (zchang40@gatech.edu)
 * @brief Obtain the Transition Matrix and Controllability Gramian for the LTV system.
 * @version 0.1
 * @date 2024-09-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include "linear_factor.h"
#include "helpers/EigenWrapper.h"
// #include <boost/numeric/odeint.hpp>

#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>

namespace gvi{

// using namespace boost::numeric::odeint;  

int gramian_ode_gsl_helper(double t, const double Q_vec[], double dQ_dt[], void *params);
int system_ode_gsl_helper(double t, const double Phi_vec[], double dPhi_dt[], void *params);

class LTV_GP : public LinearFactor{
    public: 
        LTV_GP(){};
        /**
         * @brief the state is [x; v] where the dimension of x is _dim. 
         * The returned mean is the concatenation of the two consecutive [\mu_i, \mu_{i+1}].
         * The constant velocity linear factor has closed-forms in transition matrix \Phi,
         * the matrices in computing the quadratic costs, (\Lambda, \Psi).
         * Qc is in shape (_dim, _dim); 
         * _Phi and _Q are in shape(2*_dim, 2*_dim)
         * @param Qc 
         * @param delta_t 
         */

        LTV_GP(const MatrixXd& Qc, int start_index, const double& delta_t, const VectorXd& mu_0, int n_states, const std::vector<MatrixXd>& hA, const std::vector<MatrixXd>& hB, const std::vector<VectorXd>& target_mean): 
        LinearFactor(),
        _dim{Qc.cols()},
        _dim_state{2*_dim},
        _start_index{start_index},
        _target_mu{VectorXd::Zero(2*_dim_state)},
        _delta_t{delta_t}, 
        _Qc{Qc}, 
        _invQc{Qc.inverse()}, 
        _invQ{MatrixXd::Zero(_dim_state, _dim_state)},
        _Phi{MatrixXd::Zero(_dim_state, _dim_state)}{

            _A_vec.resize(5); 
            _B_vec.resize(5); 
            for (int i = 0; i < 5; i++) {
                _A_vec[i] = hA[4 * start_index + i];
                _B_vec[i] = hB[4 * start_index + i];
            }

            // _Phi = compute_Phi();
            _Phi = compute_Phi_gsl();

            // MatrixXd Phi_gsl = compute_Phi_gsl();
            // std::cout << "Phi: " << _Phi.norm() << std::endl;
            // std::cout << "Phi_gsl: " << Phi_gsl.norm() << std::endl;
            // std::cout << "Phi_gsl Error: " << (_Phi - Phi_gsl).norm() << std::endl << std::endl;

            // Obtain mi and mi_next
            VectorXd mi = target_mean[start_index];
            VectorXd mi_next = target_mean[start_index + 1];

            _target_mu.segment(0, _dim_state) = mi;
            _target_mu.segment(_dim_state, _dim_state) = mi_next;
            
            // _Q = compute_Q();
            _Q = compute_Q_gsl();

            // MatrixXd Q_gsl = compute_Q_gsl();
            // std::cout << "Q = " << _Q.norm() << std::endl; 
            // std::cout << "Q_gsl = " << Q_gsl.norm() << std::endl;
            // std::cout << "Q_gsl Error: " << (_Q - Q_gsl).norm() << std::endl << std::endl;

            compute_invQ();

            // \Lambda = [-\Phi, I]
            _Lambda = MatrixXd::Zero(_dim_state, 2*_dim_state);
            _Lambda.block(0, 0, _dim_state, _dim_state) = -_Phi;
            _Lambda.block(0, _dim_state, _dim_state, _dim_state) = MatrixXd::Identity(_dim_state, _dim_state);

            // \Psi = [\Phi, -I]. When a(t)=0, this part is eliminated.
            _Psi = MatrixXd::Zero(_dim_state, 2*_dim_state);
            _Psi.block(0, 0, _dim_state, _dim_state) = -_Phi;
            _Psi.block(0, _dim_state, _dim_state, _dim_state) = MatrixXd::Identity(_dim_state, _dim_state);
        } 

        // MatrixXd compute_Phi(){
        //     MatrixXd Phi0 = MatrixXd::Identity(_dim_state, _dim_state);
        //     std::vector<double> Phi_vec(Phi0.data(), Phi0.data() + _dim_state * _dim_state);

        //     double dt = _delta_t / 20;
        //     auto system_ode_bound = std::bind(&gvi::LTV_GP::system_ode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

        //     integrate_adaptive(_stepper, system_ode_bound, Phi_vec, 0.0, _delta_t, dt);
        //     MatrixXd Phi_result = Eigen::Map<const MatrixXd>(Phi_vec.data(), _dim_state, _dim_state);

        //     return Phi_result;
        // }

        // MatrixXd compute_Q(){
        //     MatrixXd Q_0 = MatrixXd::Zero(_dim_state, _dim_state);
        //     std::vector<double> Q_vec(Q_0.data(), Q_0.data() + _dim_state * _dim_state);

        //     double dt = _delta_t / 20;
        //     auto gramian_ode_bound = std::bind(&gvi::LTV_GP::gramian_ode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

        //     integrate_adaptive(_stepper, gramian_ode_bound, Q_vec, 0.0, _delta_t, dt);
        //     MatrixXd Gramian = Eigen::Map<const MatrixXd>(Q_vec.data(), _dim_state, _dim_state);

        //     return Gramian;
        // }

        MatrixXd compute_Phi_gsl() {
            gsl_odeiv2_system sys = { system_ode_gsl_helper, nullptr, _dim_state * _dim_state, this};
            gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45, 1e-12, 1e-12, 0.0);
            MatrixXd Phi0 = MatrixXd::Identity(_dim_state, _dim_state);
            std::vector<double> Phi_vec(Phi0.data(), Phi0.data() + _dim_state * _dim_state);
            double t = 0.0;
            
            // Perform integration
            double dt = _delta_t / 20;
            gsl_odeiv2_driver_apply(d, &t, _delta_t, Phi_vec.data());
            gsl_odeiv2_driver_free(d);

            // Map the result back to an Eigen matrix
            return Eigen::Map<const MatrixXd>(Phi_vec.data(), _dim_state, _dim_state);
        }

        MatrixXd compute_Q_gsl() {
            gsl_odeiv2_system sys = { gramian_ode_gsl_helper, nullptr, _dim_state * _dim_state, this };
            gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45, 1e-12, 1e-12, 0.0);
            std::vector<double> Q_vec(_dim_state * _dim_state, 0.0);
            double t = 0.0;
            
            // Perform integration
            double dt = _delta_t / 20;
            gsl_odeiv2_driver_apply(d, &t, _delta_t, Q_vec.data());
            gsl_odeiv2_driver_free(d);

            // Map the result back to an Eigen matrix
            return Eigen::Map<const MatrixXd>(Q_vec.data(), _dim_state, _dim_state);
        }


        // void system_ode(const std::vector<double>& Phi_vec, std::vector<double>& dPhi_dt, double t) {
        //     MatrixXd A = A_function(t);
        //     MatrixXd Phi = Eigen::Map<const MatrixXd>(Phi_vec.data(), _dim_state, _dim_state);
        //     MatrixXd dPhi = A * Phi;
        //     Eigen::Map<MatrixXd>(dPhi_dt.data(), _dim_state, _dim_state) = dPhi;
        // }

        // void gramian_ode(const std::vector<double>& Q_vec, std::vector<double>& dQ_dt, double t) {
        //     auto matrices = system_param(t);
        //     MatrixXd gramian = Eigen::Map<const MatrixXd>(Q_vec.data(), _dim_state, _dim_state);
        //     MatrixXd dQ = matrices.first * gramian + gramian * matrices.first.transpose() + matrices.second * matrices.second.transpose();
        //     Eigen::Map<MatrixXd>(dQ_dt.data(), _dim_state, _dim_state) = dQ;
        // }

        int system_ode_gsl(double t, const double Phi_vec[], double dPhi_dt[], void *params) {
            LTV_GP* obj = static_cast<LTV_GP*>(params);
            MatrixXd Phi = Eigen::Map<const MatrixXd>(Phi_vec, obj->_dim_state, obj->_dim_state);
            MatrixXd A = obj->A_function(t);
            MatrixXd dPhi = A * Phi;
            Eigen::Map<MatrixXd>(dPhi_dt, obj->_dim_state, obj->_dim_state) = dPhi;
            return GSL_SUCCESS;
        }

        int gramian_ode_gsl(double t, const double Q_vec[], double dQ_dt[], void *params) {
            LTV_GP* obj = static_cast<LTV_GP*>(params);  // Convert params back to the LTV_GP class
            MatrixXd gramian = Eigen::Map<const MatrixXd>(Q_vec, obj->_dim_state, obj->_dim_state);
            auto matrices = obj->system_param(t);
            MatrixXd dQ = matrices.first * gramian + gramian * matrices.first.transpose() + matrices.second * matrices.second.transpose();
            Eigen::Map<MatrixXd>(dQ_dt, obj->_dim_state, obj->_dim_state) = dQ;
            return GSL_SUCCESS;
        }

        MatrixXd A_function(double t) {
            int t_idx;
            t_idx = static_cast<int>(std::floor(4 * t / _delta_t));
            return _A_vec[t_idx];
        }

        std::pair <MatrixXd, MatrixXd> system_param(double t) {
            int t_idx;
            t_idx = static_cast<int>(std::floor(4 * t / _delta_t));
            return {_A_vec[t_idx], _B_vec[t_idx]};
        }


    private:
        int _dim, _start_index;
        int _dim_state;
        double _delta_t;
        MatrixXd _Qc, _invQc, _Q, _invQ, _Phi;
        MatrixXd _Lambda, _Psi;
        std::vector<MatrixXd> _A_vec, _B_vec;
        VectorXd _m0, _target_mu;
        EigenWrapper _ei;
        // runge_kutta_dopri5<std::vector<double>> _stepper;
        
    public:
        inline MatrixXd Q() const { return _Q; }
        
        inline MatrixXd Qc() const { return _Qc; }

        inline MatrixXd Phi() const { return _Phi; }

        /**
         * @brief the cost function
         * @param theta1 [x1; v1]
         * @param theta2 [x2; v2]
         */
        inline double cost(const VectorXd& theta1, const VectorXd& theta2) const{
            double cost = (_Phi*theta1-theta2).transpose()* _invQ * (_Phi*theta1-theta2);
            return cost / 2;
        }

        inline int dim_posvel() const { return 2*_dim; }

        inline void compute_invQ() {
            _invQ = MatrixXd::Zero(_dim_state, _dim_state);
            _invQ = _Q.inverse();
        }

        inline VectorXd get_mu() const { return _target_mu; }

        inline MatrixXd get_precision() const{ return _invQ; }

        inline MatrixXd get_covariance() const{ return _Q; }

        inline MatrixXd get_Lambda() const{ return _Lambda; }

        inline MatrixXd get_Psi() const{ return _Psi; }

        inline double get_Constant() const { return 0.5; }
        
};


int gramian_ode_gsl_helper(double t, const double Q_vec[], double dQ_dt[], void *params) {
    LTV_GP* obj = static_cast<LTV_GP*>(params);
    return obj->gramian_ode_gsl(t, Q_vec, dQ_dt, params);
}

int system_ode_gsl_helper(double t, const double Phi_vec[], double dPhi_dt[], void *params) {
    LTV_GP* obj = static_cast<LTV_GP*>(params);
    return obj->system_ode_gsl(t, Phi_vec, dPhi_dt, params);
}

}

// // Use Boole's Rule to approximate the integration
// MatrixXd gramian = (7 * _Phi_ode_results[0] * _B_vec[0] * _B_vec[0].transpose() * _Phi_ode_results[0].transpose() 
// + 32 * _Phi_ode_results[1] * _B_vec[1] * _B_vec[1].transpose() * _Phi_ode_results[1].transpose()
// + 4 * _Phi_ode_results[2] * _B_vec[2] * _B_vec[2].transpose() * _Phi_ode_results[2].transpose()
// + 32 * _Phi_ode_results[3] * _B_vec[3] * _B_vec[3].transpose() * _Phi_ode_results[3].transpose()
// + 7 * _B_vec[4] * _B_vec[4].transpose()) / 90 * _delta_t;
// return gramian;
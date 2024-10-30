#pragma once

#include "linear_factor.h"
#include "helpers/EigenWrapper.h"
#include <boost/numeric/odeint.hpp>

namespace gvi{

using namespace boost::numeric::odeint;  

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

            double dt = _delta_t / 20;
            auto system_ode_bound = std::bind(&gvi::LTV_GP::system_ode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

            MatrixXd Phi0 = MatrixXd::Identity(_dim_state, _dim_state);
            std::vector<double> Phi_vec(Phi0.data(), Phi0.data() + _dim_state * _dim_state);
            integrate_adaptive(_stepper, system_ode_bound, Phi_vec, 0.0, _delta_t, dt);
            MatrixXd Phi_result = Eigen::Map<const MatrixXd>(Phi_vec.data(), _dim_state, _dim_state);
            _Phi = Phi_result;

            // Obtain mi and mi_next
            VectorXd mi = target_mean[start_index];
            VectorXd mi_next = target_mean[start_index + 1];

            _target_mu.segment(0, _dim_state) = mi;
            _target_mu.segment(_dim_state, _dim_state) = mi_next;
            
            _Q = MatrixXd::Zero(_dim_state, _dim_state);
            _Q = compute_Q();

            compute_invQ();

            // \Lambda = [-\Phi, I]
            _Lambda = MatrixXd::Zero(_dim_state, 2*_dim_state);
            _Lambda.block(0, 0, _dim_state, _dim_state) = -_Phi;
            _Lambda.block(0, _dim_state, _dim_state, _dim_state) = MatrixXd::Identity(_dim_state, _dim_state);

            // \Psi = [\Phi, -I]. When a(t)=0, this part is eliminated.
            _Psi = MatrixXd::Zero(_dim_state, 2*_dim_state);
            _Psi.block(0, 0, _dim_state, _dim_state) = _Phi;
            _Psi.block(0, _dim_state, _dim_state, _dim_state) = -MatrixXd::Identity(_dim_state, _dim_state);
        } 


        MatrixXd compute_Q (){
            MatrixXd Q_0 = MatrixXd::Zero(_dim_state, _dim_state);
            std::vector<double> Q_vec(Q_0.data(), Q_0.data() + _dim_state * _dim_state);

            double dt = _delta_t / 20;
            auto gramian_ode_bound = std::bind(&gvi::LTV_GP::gramian_ode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

            integrate_adaptive(_stepper, gramian_ode_bound, Q_vec, 0.0, _delta_t, dt);
            MatrixXd Gramian = Eigen::Map<const MatrixXd>(Q_vec.data(), _dim_state, _dim_state);

            // // Use Boole's Rule to approximate the integration
            // MatrixXd gramian = (7 * _Phi_ode_results[0] * _B_vec[0] * _B_vec[0].transpose() * _Phi_ode_results[0].transpose() 
            // + 32 * _Phi_ode_results[1] * _B_vec[1] * _B_vec[1].transpose() * _Phi_ode_results[1].transpose()
            // + 4 * _Phi_ode_results[2] * _B_vec[2] * _B_vec[2].transpose() * _Phi_ode_results[2].transpose()
            // + 32 * _Phi_ode_results[3] * _B_vec[3] * _B_vec[3].transpose() * _Phi_ode_results[3].transpose()
            // + 7 * _B_vec[4] * _B_vec[4].transpose()) / 90 * _delta_t;

            return Gramian;
        }


        void system_ode(const std::vector<double>& Phi_vec, std::vector<double>& dPhi_dt, double t) {
            MatrixXd A = A_function(t);
            MatrixXd Phi = Eigen::Map<const MatrixXd>(Phi_vec.data(), _dim_state, _dim_state);
            MatrixXd dPhi = A * Phi;
            Eigen::Map<MatrixXd>(dPhi_dt.data(), _dim_state, _dim_state) = dPhi;
        }

        void gramian_ode(const std::vector<double>& Q_vec, std::vector<double>& dQ_dt, double t) {
            auto matrices = system_param(t);
            MatrixXd gramian = Eigen::Map<const MatrixXd>(Q_vec.data(), _dim_state, _dim_state);
            MatrixXd dQ = matrices.first * gramian + gramian * matrices.first.transpose() + matrices.second * matrices.second.transpose();
            Eigen::Map<MatrixXd>(dQ_dt.data(), _dim_state, _dim_state) = dQ;
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
        runge_kutta_dopri5<std::vector<double>> _stepper;
        
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

}
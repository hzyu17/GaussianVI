#pragma once

#include "linear_factor.h"
#include "helpers/EigenWrapper.h"
#include <unsupported/Eigen/MatrixFunctions>
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
        _m0{mu_0},
        _target_mu{VectorXd::Zero(2*_dim_state)},
        _delta_t{delta_t}, 
        _Qc{Qc}, 
        _invQc{Qc.inverse()}, 
        _invQ{MatrixXd::Zero(_dim_state, _dim_state)},
        _Phi{_dim_state, _dim_state}{
            _Phi.setZero();
            _Phi = (hA[4*start_index] * delta_t).exp();

            _Phi_left_quater = (hA[4*start_index+1] * delta_t * 3 / 4).exp();
            _Phi_mid = (hA[4*start_index+2] * delta_t / 2).exp();
            _Phi_right_quater = (hA[4*start_index+3] * delta_t / 4).exp();

            _A_vec.resize(5); 
            for (int i = 0; i < 5; i++) {
                _A_vec[i] = hA[4 * start_index + i];
            }

            // MatrixXd Phi_pred = MatrixXd::Identity(_dim_state, _dim_state) + _delta_t * hA[_start_index];
            // _Phi = MatrixXd::Identity(_dim_state, _dim_state) + (hA[_start_index] + hA[_start_index+1] * Phi_pred) / 2 * delta_t;

            std::vector<double> start_times = {0.0, _delta_t / 4, _delta_t / 2, 3 * _delta_t / 4};

            runge_kutta_dopri5<std::vector<double>> stepper;

            double dt = _delta_t / 20;
            auto system_ode_bound = std::bind(&gvi::LTV_GP::system_ode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

            for (double start_time : start_times) {
                MatrixXd Phi0 = MatrixXd::Identity(_dim_state, _dim_state);
                std::vector<double> Phi_vec(Phi0.data(), Phi0.data() + _dim_state * _dim_state);
                integrate_adaptive(stepper, system_ode_bound, Phi_vec, start_time, _delta_t, dt);
                MatrixXd Phi_result = Eigen::Map<const MatrixXd>(Phi_vec.data(), _dim_state, _dim_state);
                _Phi_results.push_back(Phi_result);
            }

            // std::cout << "Phi0  "  << std::endl << _Phi<< std::endl;
            // std::cout << "Error of Phi0 = "  << std::endl << _Phi - _Phi_results[0] << std::endl;
            // std::cout << "Error = " << (Phi_result - Phi_results[0] * Phi_results[1].inverse()).norm() << std::endl;
            // std::cout << "Norm of Phi0  " << _Phi.norm() << std::endl;
            // std::cout << "Norm of Error = "  << std::endl << (_Phi - _Phi_results[0]).norm() << std::endl << std::endl;
            // std::cout << "Error of Phi1  " << (_Phi_left_quater - Phi_results[1]).norm() << std::endl;
            // std::cout << "Error of Phi2  " << (_Phi_mid - Phi_results[2]).norm() << std::endl;
            // std::cout << "Error of Phi3  " << (_Phi_right_quater - Phi_results[3]).norm() << std::endl << std::endl << std::endl;

            // Obtain mi and mi_next
            VectorXd mi = target_mean[start_index];
            VectorXd mi_next = target_mean[start_index + 1];

            _target_mu.segment(0, _dim_state) = mi;
            _target_mu.segment(_dim_state, _dim_state) = mi_next;
            
            _Q = MatrixXd::Zero(_dim_state, _dim_state);
            _Q = compute_Q(hA, hB, n_states);

            compute_invQ();

            // \Lambda = [-\Phi, I]
            _Lambda = MatrixXd::Zero(_dim_state, 2*_dim_state);
            _Lambda.block(0, 0, _dim_state, _dim_state) = -_Phi_results[0];
            _Lambda.block(0, _dim_state, _dim_state, _dim_state) = MatrixXd::Identity(_dim_state, _dim_state);

            // \Psi = [\Phi, -I]. When a(t)=0, this part is eliminated.
            _Psi = MatrixXd::Zero(_dim_state, 2*_dim_state);
            _Psi.block(0, 0, _dim_state, _dim_state) = _Phi_results[0];
            _Psi.block(0, _dim_state, _dim_state, _dim_state) = -MatrixXd::Identity(_dim_state, _dim_state);
        } 

        MatrixXd compute_Q (const std::vector<MatrixXd>& hA, const std::vector<MatrixXd>& hB, int n_states){
            MatrixXd hA_next, hB_i, hB_mid, hB_next, hB_left_quater, hB_right_quater;
            hB_i = hB[4 * _start_index];
            hB_left_quater = hB[4 * _start_index + 1];
            hB_mid = hB[4 * _start_index + 2];
            hB_right_quater = hB[4 * _start_index + 3];
            hB_next = hB[4 * _start_index + 4];

            // Use Boole's Rule to approximate the integration
            MatrixXd gramian = (7 * _Phi_results[0] * hB_i * hB_i.transpose() * _Phi_results[0].transpose() 
            + 32 * _Phi_results[1] * hB_left_quater * hB_left_quater.transpose() * _Phi_results[1].transpose()
            + 4 * _Phi_results[2] * hB_mid * hB_mid.transpose() * _Phi_results[2].transpose()
            + 32 * _Phi_results[3] * hB_right_quater * hB_right_quater.transpose() * _Phi_results[3].transpose()
            + 7 * hB_next * hB_next.transpose()) / 90 * _delta_t;

            // MatrixXd gramian_pred = (_Phi * hB_i * hB_i.transpose() * _Phi.transpose() + hB_next * hB_next.transpose()) / 2 * _delta_t;
            // MatrixXd derivative_i = hB_i * hB_i.transpose();
            // MatrixXd derivative_next = hB_next * hB_next.transpose() + (hA_next + hA_next.transpose()) * gramian_pred;
            // MatrixXd gramian = gramian_pred - (derivative_next - derivative_i) / 12 * _delta_t * _delta_t;

            return gramian;

            // return (_Phi * hB[_start_index] * hB[_start_index].transpose() * _Phi.transpose() +
            // hB[_start_index + 1] * hB[_start_index + 1].transpose()) / 2 * _delta_t;

        }

        void system_ode(const std::vector<double>& Phi_vec, std::vector<double>& dPhi_dt, double t) {
            MatrixXd A = A_function(t);
            MatrixXd Phi = Eigen::Map<const MatrixXd>(Phi_vec.data(), _dim_state, _dim_state);
            MatrixXd dPhi = A * Phi;
            Eigen::Map<MatrixXd>(dPhi_dt.data(), _dim_state, _dim_state) = dPhi;
        }

        MatrixXd A_function(double t) {
            int t_idx;
            t_idx = static_cast<int>(std::floor(4 * t / _delta_t));
            return _A_vec[t_idx];
        }

    private:
        int _dim, _start_index;
        int _dim_state;
        double _delta_t;
        MatrixXd _Qc, _invQc, _Q, _invQ;
        MatrixXd _Phi, _Lambda, _Psi, _Phi_mid, _Phi_left_quater, _Phi_right_quater;
        std::vector<MatrixXd> _A_vec;
        std::vector<MatrixXd> _Phi_results;
        VectorXd _m0, _target_mu;
        EigenWrapper _ei;
        
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


// MatrixXd compute_phi_i (const std::vector<MatrixXd>& hA){
//     MatrixXd Phi = MatrixXd::Identity(_dim_state, _dim_state);
//     MatrixXd Phi_i(_dim_state, _dim_state);
//     // In each loop i, add the derivative and get Phi_i+1
//     for (int i = 0; i < _start_index; i++){
//         MatrixXd A_i = hA[i];
//         Phi = Phi + A_i * Phi * _delta_t;
//     }
//     Phi_i = Phi;
//     return Phi_i;
// }



// EigenSolver<MatrixXd> solver(_Q);
// VectorXd eigenvalues = solver.eigenvalues().real();

// EigenSolver<MatrixXd> solver_inv(_invQ);
// VectorXd eigenvalues_inv = solver_inv.eigenvalues().real();

// std::cout << "Matrix A= " << std::endl << hA[start_index] << std::endl;
// std::cout << "Matrix B= " << std::endl << hB[start_index] << std::endl;
// std::cout << "Transition Matrix = " << std::endl << _Phi << std::endl;
// std::cout << "Phi x hB= " << std::endl << _Phi * hB[start_index] << std::endl;

// std::cout << "Grammian = " << std::endl << _Q << std::endl;
// std::cout << "Precision = " << std::endl << _invQ << std::endl << std::endl;
// std::cout << "The eigenvalues of the Gramian:\n" << eigenvalues << std::endl;
// std::cout << "The eigenvalues of the Precision:\n" << eigenvalues_inv << std::endl;


// MatrixXd A_Mat(_dim_state, _dim_state);
// A_Mat << MatrixXd::Zero(_dim, _dim),  MatrixXd::Identity(_dim, _dim), MatrixXd::Zero(_dim, _dim_state);
// return A_Mat;


// MatrixXd Phi_LTI (_dim_state, _dim_state);
// Phi_LTI.setZero();
// Phi_LTI << MatrixXd::Identity(_dim, _dim), delta_t*MatrixXd::Identity(_dim, _dim), 
//             MatrixXd::Zero(_dim, _dim), MatrixXd::Identity(_dim, _dim);

// MatrixXd Err_LTI = _Phi_results[0] - Phi_LTI;
// std::cout << "Norm of LTI Error = "  << std::endl << Err_LTI.norm() << std::endl << std::endl;
/**
 * @file minimum_acc.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief minimum acceleration gp model, which is a linear model of the form
 * -log(p(x|z)) = ||A*x - B*\mu_t||_{\Sigma^{-1}}.
 * @version 0.1
 * @date 2022-07-31
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

/**
 * @brief The model A(t) = [0 I; 0 0], u(t)=0, F(t)=[0; I], 
 * Phi(t,s)=[I (t-s)I; 0 I], Q_{i,i+1}=[1/3*(dt)^3Qc 1/2*(dt)^2Qc; 1/2*(dt)^2Qc (dt)Qc]
 * x and v share one same Qc.
 */

#include "linear_factor.h"
#include "helpers/EigenWrapper.h"


namespace gvi{

class QuadGP : public LinearFactor{
    public: 
        QuadGP(){};
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

        QuadGP(const MatrixXd& Qc, int start_index, const double& delta_t, const VectorXd& mu_0, int n_states, const std::vector<MatrixXd>& hA, const std::vector<MatrixXd>& hb, const std::vector<MatrixXd>& Phi_vec): 
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
            _Phi = MatrixXd::Identity(_dim_state, _dim_state) + _delta_t * hA[_start_index];

            // compute the concatenation [\mu_i, \mu_{i+1}]
            MatrixXd Phi_i(_dim_state, _dim_state);
            Phi_i.setZero();
            Phi_i = Phi_vec[start_index];

            // Obtain mi and mi_next
            VectorXd mi = Phi_i * _m0;
            VectorXd mi_next = _Phi * mi;
            
            _Q = MatrixXd::Zero(_dim_state, _dim_state);
            _Q = compute_Q(hb, n_states);
            compute_invQ();

            // \Lambda = [-\Phi, I]
            _Lambda = MatrixXd::Zero(_dim_state, 2*_dim_state);
            _Lambda.block(0, 0, _dim_state, _dim_state) = -_Phi;
            _Lambda.block(0, _dim_state, _dim_state, _dim_state) = MatrixXd::Identity(_dim_state, _dim_state);

            // \Psi = [\Phi, -I]. When a(t)=0, this part is eliminated.
            _Psi = MatrixXd::Zero(_dim_state, 2*_dim_state);
            // _Psi.block(0, 0, _dim_state, _dim_state) = _Phi;
            // _Psi.block(0, _dim_state, _dim_state, _dim_state) = -MatrixXd::Identity(_dim_state, _dim_state);
        }

        MatrixXd compute_phi_i (std::vector<MatrixXd> hA){
            MatrixXd Phi = MatrixXd::Identity(_dim_state, _dim_state);
            MatrixXd Phi_i(_dim_state, _dim_state);
            // In each loop i, add the derivative and get Phi_i+1
            for (int i = 0; i < _start_index; i++){
                MatrixXd A_i = hA[i];
                Phi = Phi + A_i * Phi * _delta_t;
            }
            Phi_i = Phi;
            return Phi_i;
        }

        MatrixXd compute_Q (std::vector<MatrixXd> hb, int n_states){
            // if (_start_index < n_states - 1)
                return (_Phi * hb[_start_index] * _Qc * hb[_start_index].transpose() * _Phi.transpose() +
                hb[_start_index + 1] * _Qc * hb[_start_index + 1].transpose()) / 2 * _delta_t;
            // else
            //     return _Phi * hb[_start_index] * _Qc * hb[_start_index].transpose() * _Phi.transpose() * _delta_t;
        }

    private:
        int _dim, _start_index;
        int _dim_state;
        double _delta_t;
        MatrixXd _Qc, _invQc, _Q, _invQ;
        MatrixXd _Phi, _Lambda, _Psi;
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
            _invQ = MatrixXd::Zero(2*_dim, 2*_dim);
            _invQ = _Q.inverse();
            
        }

        inline VectorXd get_mu() const { return _target_mu; }

        inline MatrixXd get_precision() const{ return _invQ; }

        inline MatrixXd get_covariance() const{ return _invQ.inverse(); }

        inline MatrixXd get_Lambda() const{ return _Lambda; }

        inline MatrixXd get_Psi() const{ return _Psi; }

        inline double get_Constant() const { return 0.5; }
        
};

}

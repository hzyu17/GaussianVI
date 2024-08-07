/**
 * @file GaussHermite.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief Class to calculate the approximated integrations using Gauss-Hermite quadrature
 * @version 0.1
 * @date 2022-05-11
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include <eigen3/Eigen/Dense>
#include <iostream>
#include "helpers/CommonDefinitions.h"

using namespace Eigen;

namespace gvi{
template <typename Function>
class GaussHermite{
public:
    /**
     * @brief Default constructor.
     */
    GaussHermite(){}

    /**
     * @brief Constructor
     * 
     * @param deg degree of GH polynomial
     * @param dim dimension of the integrand input
     * @param mean mean 
     * @param P covariance matrix
    //  * @param func the integrand function
     */
    GaussHermite(
        const int& deg, 
        const int& dim, 
        const VectorXd& mean, 
        const MatrixXd& P
        ): 
            _deg{deg},
            _dim{dim},
            _mean{mean},
            _P{P},
            _W{VectorXd::Zero(_deg)},
            _sigmapts{VectorXd::Zero(_deg)}{
                this->computeWeights();
            }

    /**
     * @brief A helper function to compute all possible permutations given a dimension and a degree.
     *  for computing the integration using sigmapoints and weights. Returns all vector of length dimension,
     *  collected from the number degree. It is a permutation with replacement.
     * @param dimension 
     * @return std::vector<double>
     */
    void permute_replacing(const std::vector<int>& vec, 
                            const int& dimension, 
                            std::vector<int>& res, 
                            int index, 
                            std::vector<std::vector<int>>& v_res);

    /**
     * @brief Compute the Sigma Pts
     */
    void computeSigmaPts();

    /**
     * @brief Define the Hermite polynomial of degree deg, evaluate at x.
     * @param deg the degree to evaluate
     * @param x input
     */
    double HermitePolynomial(const int& deg, const double& x) const;

    /**
     * @brief Compute the weights in the Gauss-Hermite cubature method.
     */
    void computeWeights();

    /**
     * @brief Compute the approximated integration using Gauss-Hermite.
     */
    MatrixXd Integrate(const Function& function);

    /**
     * Update member variables
     * */
    inline void update_mean(const VectorXd& mean){ _mean = mean; }

    inline void update_P(const MatrixXd& P){ _P = P; }

    inline void set_polynomial_deg(const int& deg){ _deg = deg; }

    inline void update_dimension(const int& dim){ _dim = dim; }


    inline VectorXd weights() const { return this->_W; }
    inline VectorXd sigmapts() const { return this->_sigmapts; }

    protected:
    int _deg;
    int _dim;
    VectorXd _mean;
    MatrixXd _P;
    VectorXd _W;
    VectorXd _sigmapts;
};

} // namespace gvi

#include "GaussHermite-impl.h"
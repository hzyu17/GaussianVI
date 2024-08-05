/**
 * @file NGDFactorizedBaseGH.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief The marginal optimizer class expecting two functions, one is the cost function, f1(x, cost1); 
 * the other is the function for GH expectation, f2(x).
 * @version 0.1
 * @date 2022-03-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#ifndef NGDFactorizedBaseGH_H
#define NGDFactorizedBaseGH_H

// #include "ngd/NGDFactorizedBase.h"
#include "gvibase/GVIFactorizedBaseGH_Cuda.h"
#include <gpmp2/kinematics/PointRobotModel.h>
#include <gpmp2/obstacle/ObstaclePlanarSDFFactor.h>
// #include <gpmp2/obstacle/ObstacleSDFFactor.h>

#include <memory>
#include <chrono>
#include <thread>


using namespace Eigen;

namespace gvi{

class PlanarSDF {

public:
  // index and float_index is <row, col>
  typedef std::tuple<size_t, size_t> index;
//   typedef std::tuple<double, double> float_index;
  typedef Vector2d float_index;
  typedef std::shared_ptr<PlanarSDF> shared_ptr;
  double* data_array;
  Eigen::Vector2d origin_;

  // geometry setting of signed distance field
  size_t field_rows_, field_cols_;
  double cell_size_;
  Eigen::MatrixXd data_;

private:
  

public:
  /// constructor
  PlanarSDF() : field_rows_(0), field_cols_(0), cell_size_(0.0) {}

  /// constructor with data
  PlanarSDF(const Eigen::Vector2d& origin, double cell_size, const Eigen::MatrixXd& data) :
      origin_(origin), field_rows_(data.rows()), field_cols_(data.cols()),
      cell_size_(cell_size), data_(data){
        data_array = data_.data();
      }

  ~PlanarSDF() {}


  /// give a point, search for signed distance field and (optional) gradient
  /// return signed distance
  __host__ __device__ inline double getSignedDistance(const Eigen::Vector2d& point) const {
    const float_index pidx = convertPoint2toCell(point);
    return signed_distance(pidx);
  }

  __host__ __device__ inline Eigen::Vector2d getGradient(const Eigen::Vector2d& point) const {
    const float_index pidx = convertPoint2toCell(point);
    const Eigen::Vector2d g_idx = gradient(pidx);
    // convert gradient of index to gradient of metric unit
    return Eigen::Vector2d(g_idx(1), g_idx(0)) / cell_size_;
  }


  /// convert between point and cell corrdinate
  __host__ __device__ inline float_index convertPoint2toCell(const Eigen::Vector2d& point) const {
    // // check point range
    // if (point.x() < origin_.x() || point.x() > (origin_.x() + (field_cols_-1.0)*cell_size_) ||
    //     point.y() < origin_.y() || point.y() > (origin_.y() + (field_rows_-1.0)*cell_size_)) {
        
    //   // Convert the number to a string using std::to_string
    //   std::string origin_x_Str = std::to_string(point.x());
    //   std::string origin_y_Str = std::to_string(point.y());

    //   // Concatenate the string and the number
    //   std::string err_msg = "Index out of range. point.x: " + origin_x_Str + "; " + "point.y: " + origin_y_Str;
    //   throw std::out_of_range(err_msg);
    // }

    const double col = (point.x() - origin_.x()) / cell_size_;
    const double row = (point.y() - origin_.y()) / cell_size_;
    return Vector2d{row, col};
  }

  __host__ __device__ inline Eigen::Vector2d convertCelltoPoint2(const float_index& cell) const {
    return origin_ + Eigen::Vector2d(
        cell(1) * cell_size_,
        cell(0) * cell_size_);
  }


  /// bilinear interpolation
  __host__ __device__ inline double signed_distance(const float_index& idx) const {
    const double lr = floor(idx(0)), lc = floor(idx(1));
    const double hr = lr + 1.0, hc = lc + 1.0;
    const size_t lri = static_cast<size_t>(lr), lci = static_cast<size_t>(lc),
                 hri = static_cast<size_t>(hr), hci = static_cast<size_t>(hc);
    // printf("lr = %lf, lc = %lf, hr = %lf, hc = %lf, lri = %d, lci = %d, hri = %d, hci = %d\n", lr, lc, hr, hc, lri, lci, hri, hci);
    return
        (hr-idx(0))*(hc-idx(1))*signed_distance(lri, lci) +
        (idx(0)-lr)*(hc-idx(1))*signed_distance(hri, lci) +
        (hr-idx(0))*(idx(1)-lc)*signed_distance(lri, hci) +
        (idx(0)-lr)*(idx(1)-lc)*signed_distance(hri, hci);
  }

  /// gradient operator for bilinear interpolation
  /// gradient regrads to float_index
  /// not numerical differentiable at index point
  __host__ __device__ inline Eigen::Vector2d gradient(const float_index& idx) const {
    const double lr = floor(idx(0)), lc = floor(idx(1));
    const double hr = lr + 1.0, hc = lc + 1.0;
    const size_t lri = static_cast<size_t>(lr), lci = static_cast<size_t>(lc),
        hri = static_cast<size_t>(hr), hci = static_cast<size_t>(hc);
    return Eigen::Vector2d(
        (hc-idx(1)) * (signed_distance(hri, lci)-signed_distance(lri, lci)) +
        (idx(1)-lc) * (signed_distance(hri, hci)-signed_distance(lri, hci)),

        (hr-idx(0)) * (signed_distance(lri, hci)-signed_distance(lri, lci)) +
        (idx(0)-lr) * (signed_distance(hri, hci)-signed_distance(hri, lci)));
  }

  // access
  __host__ __device__ inline double signed_distance(size_t r, size_t c) const {
    // printf("data_array = %lf\n", data_array[r + c * field_rows_]);
    // printf("distance(%d, %d) = %lf\n", r, c, data_(r,c));
    return data_array[r + c * field_rows_];
    // return data_(r,c);
  }

  const Eigen::Vector2d& origin() const { return origin_; }
  size_t x_count() const { return field_cols_; }
  size_t y_count() const { return field_rows_; }
  double cell_size() const { return cell_size_; }
  const Eigen::MatrixXd& raw_data() const { return data_; }


};




template <typename CostClass>
class NGDFactorizedBaseGH_Cuda: public GVIFactorizedBaseGH_Cuda{

    using GVIBase = GVIFactorizedBaseGH_Cuda;
    using Function = std::function<double(const VectorXd&, const CostClass &)>;

    // Robot = gpmp2::PointRobotModel
    // CostClass = gpmp2::ObstaclePlanarSDFFactor<gpmp2::PointRobotModel>
    // NGDClass = NGDFactorizedBaseGH_Cuda<gpmp2::ObstaclePlanarSDFFactor<gpmp2::PointRobotModel>>

public:
    ///@param dimension The dimension of the state
    ///@param function Template function class which calculate the cost
    // NGDFactorizedBaseGH(const int& dimension, const Function& function, const CostClass& cost_class_, const MatrixXd& Pk_):
    
    NGDFactorizedBaseGH_Cuda(int dimension, int state_dim, int gh_degree, 
                        const Function& function, const CostClass& cost_class,
                        int num_states, int start_index, 
                        double temperature, double high_temperature,
                        QuadratureWeightsMap weight_sigpts_map_option):
                GVIBase(dimension, state_dim, num_states, start_index, 
                        temperature, high_temperature, weight_sigpts_map_option)
            {
                /// Override of the GVIBase classes. _func_phi-> Scalar, _func_Vmu -> Vector, _func_Vmumu -> Matrix
                GVIBase::_func_phi = [this, function, cost_class](const VectorXd& x){return MatrixXd::Constant(1, 1, function(x, cost_class));};
                GVIBase::_func_Vmu = [this, function, cost_class](const VectorXd& x){return (x-GVIBase::_mu) * function(x, cost_class);};
                GVIBase::_func_Vmumu = [this, function, cost_class](const VectorXd& x){return MatrixXd{(x-GVIBase::_mu) * (x-GVIBase::_mu).transpose().eval() * function(x, cost_class)};};
                GVIBase::_gh = std::make_shared<GH>(GH{gh_degree, GVIBase::_dim, GVIBase::_mu, GVIBase::_covariance, weight_sigpts_map_option});

                MatrixIO _m_io;
                std::string field_file = source_root + "/maps/2dpR/map2/field_multiobs_map2.csv";
                MatrixXd field = _m_io.load_csv(field_file);      

                Vector2d origin;
                origin.setZero();
                origin << -20.0, -10.0;

                double cell_size = 0.1;
                _sdf = PlanarSDF{origin, cell_size, field};
                // _sdf = std::make_shared<PlanarSDF>(PlanarSDF{origin, cell_size, field});

            }
public:

void calculate_partial_V() override{
        // update the mu and sigma inside the gauss-hermite integrator
        updateGH(this->_mu, this->_covariance);

        this->_Vdmu.setZero();
        this->_Vddmu.setZero();

        /// Integrate for E_q{_Vdmu} 
        this->_Vdmu = Integrate_cuda(1);
        // VectorXd Vdmu_cpu = this->_gh->Integrate(this->_func_Vmu);
        // std::cerr << "Vdmu Error: " << (_Vdmu - Vdmu_cpu).cwiseAbs().maxCoeff() << std::endl;
        this->_Vdmu = this->_precision * this->_Vdmu;
        this->_Vdmu = this->_Vdmu / this->temperature();
        

        /// Integrate for E_q{phi(x)}
        double E_phi = Integrate_cuda(0)(0, 0);
        // double E_phi = this->_gh->Integrate(this->_func_phi)(0, 0);
        // double E_phi_cpu = this->_gh->Integrate(this->_func_phi)(0, 0);
        // std::cerr << "Ephi Error: " << E_phi - E_phi_cpu << std::endl;
        
        /// Integrate for partial V^2 / ddmu_ 
        MatrixXd E_xxphi{Integrate_cuda(2)};
        // MatrixXd E_xxphi_cpu{this->_gh->Integrate(this->_func_Vmumu)};
        // std::cerr << "E_xxphi Error: " << (E_xxphi - E_xxphi_cpu).cwiseAbs().maxCoeff() << std::endl << std::endl;

        this->_Vddmu.triangularView<Upper>() = (this->_precision * E_xxphi * this->_precision - this->_precision * E_phi).triangularView<Upper>();
        this->_Vddmu.triangularView<StrictlyLower>() = this->_Vddmu.triangularView<StrictlyUpper>().transpose();
        this->_Vddmu = this->_Vddmu / this->temperature();
    }
    
    MatrixXd Integrate_cuda(int type){
        VectorXd mean_gh = this -> _gh -> mean();
        MatrixXd sigmapts_gh = this -> _gh -> sigmapts();
        MatrixXd weights_gh = this -> _gh -> weights();
        MatrixXd result;
        if (type == 0)
          result = MatrixXd::Zero(1,1);
        else if (type ==  1)
          result = MatrixXd::Zero(sigmapts_gh.cols(),1);
        else
          result = MatrixXd::Zero(sigmapts_gh.cols(),sigmapts_gh.cols());

        // MatrixXd function_value = this -> _gh -> Obtain_function_value(this->_func_Vmumu);
        // std::cerr << "Function Value:" << std::endl << function_value << std::endl << std::endl;

        // MatrixXd pts_cpu(result.rows(), sigmapts_gh.rows()*result.cols());
        // for (int i = 0; i < sigmapts_gh.rows(); i++) {
        //   // std::cout << "i = " << i << ", sigma.row = " << sigmapts_gh.row(i) << std::endl;
        //   pts_cpu(i) = cost_obstacle_planar(sigmapts_gh.row(i), _sdf);
        //   // std::cout << "i = " << i << ", value = " << pts_cpu(i) << std::endl;
        // }
        
        // std::cout << "Function Value CPU:" << std::endl << pts_cpu << std::endl << std::endl;
        // std::cout << "Function Value Error:" << std::endl << function_value - pts_cpu << std::endl << std::endl;

        
        MatrixXd pts(result.rows(), sigmapts_gh.rows()*result.cols());
        CudaIntegration(sigmapts_gh, weights_gh, result, mean_gh, type, pts);
        
        // std::cout << "Function Value Cuda Error:" << std::endl << function_value - pts << std::endl << std::endl;

        return result;
    }

    void CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type, MatrixXd& pts);

    __host__ __device__ double cost_obstacle_planar(const VectorXd& pose, const PlanarSDF& sdf){
      double radius = 1;
      double epsilon = 0.5;
      double sigma = 15.5;
      double total_eps = radius + epsilon;
      double err;

      double signed_distance = sdf.getSignedDistance(pose);

      if (signed_distance > total_eps)
        err =  0.0;
      else
        err =  total_eps - signed_distance;
      
      return err * err * sigma;
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


    /**
     * @brief returns the (x-mu)*Phi(x) 
     */
    inline MatrixXd xMu_negative_log_probability(const VectorXd& x) const{
        return _func_Vmu(x);
    }

    /**
     * @brief returns the (x-mu)(x-mu)^T*Phi(x) 
     */
    inline MatrixXd xMuxMuT_negative_log_probability(const VectorXd& x) const{
        return _func_Vmumu(x);
    }

    double fact_cost_value(const VectorXd& fill_joint_mean, const SpMat& joint_cov) override {
        VectorXd mean_k = extract_mu_from_joint(fill_joint_mean);
        MatrixXd Cov_k = extract_cov_from_joint(joint_cov);

        updateGH(mean_k, Cov_k);

        return Integrate_cuda(0)(0, 0) / this->temperature();
        // return this->_gh->Integrate(this->_func_phi)(0, 0) / this->temperature();
    }
    
    PlanarSDF _sdf;

};


}

#endif // NGDFactorizedBaseGH_H
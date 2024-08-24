#ifndef CUDA_OPERATION_H
#define CUDA_OPERATION_H

#include <cuda_runtime.h>
#include <helpers/MatrixHelper.h>
#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include <gpmp2/obstacle/ObstaclePlanarSDFFactor.h>
#include <gpmp2/kinematics/PointRobotModel.h>

using namespace Eigen;

namespace gvi{

// __host__ __device__ void invert_matrix(double* A, double* A_inv, int dim); 

class PlanarSDF {

public:
  // index and float_index is <row, col>
  typedef std::tuple<size_t, size_t> index;
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
  __host__ __device__ inline VectorXd getSignedDistance(const Eigen::MatrixXd& point) const {
    int n_balls = point.rows();
    VectorXd signed_dis(n_balls);
    for (int i = 0; i < n_balls; i++){
      const float_index pidx = convertPoint2toCell(point.row(i));
      signed_dis(i) = signed_distance(pidx);
    }
    return signed_dis;
  }

  /// convert between point and cell corrdinate
  __host__ __device__ inline float_index convertPoint2toCell(const Eigen::Vector2d& point) const {
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
    return data_array[r + c * field_rows_];
  }

  const Eigen::Vector2d& origin() const { return origin_; }
  size_t x_count() const { return field_cols_; }
  size_t y_count() const { return field_rows_; }
  double cell_size() const { return cell_size_; }
  const Eigen::MatrixXd& raw_data() const { return data_; }


};


template <typename CostClass>
class CudaOperation{

public:
    CudaOperation(double cost_sigma = 15.5, double epsilon = 0.5, double radius = 1):
    _sigma(cost_sigma), _epsilon(epsilon), _radius(radius)
    {
        MatrixIO _m_io;
        std::string field_file = source_root + "/maps/2dpR/map2/field_multiobs_map2.csv";
        MatrixXd field = _m_io.load_csv(field_file);      

        Vector2d origin;
        origin.setZero();
        origin << -20.0, -10.0;

        double cell_size = 0.1;
        _sdf = PlanarSDF{origin, cell_size, field};
    }

    void CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type, MatrixXd& pts);

    __host__ __device__ double cost_obstacle_planar(const VectorXd& pose, const PlanarSDF& sdf){
      MatrixXd checkpoints = vec_balls(pose, 1);
      VectorXd signed_distance = sdf.getSignedDistance(checkpoints);
      VectorXd err(signed_distance.size());
      // printf("dim of err = %d\n", err.size());
      MatrixXd sigma_matrix(err.size(), err.size());
      double cost = 0;
      sigma_matrix = _sigma * MatrixXd::Identity(err.size(), err.size());

      for (int i = 0; i < signed_distance.size(); i++){
        if (signed_distance(i) > _epsilon + _radius)
          err(i) =  0.0;
        else
          err(i) =  _epsilon + _radius - signed_distance(i);
        cost =+ err(i) * err(i) * _sigma;
      }
      
      // return err.transpose() * sigma_matrix * err;
      return cost;
    }

    __host__ __device__ Eigen::MatrixXd vec_balls(const Eigen::VectorXd& x, int n_balls) {
      Eigen::MatrixXd v_pts = Eigen::MatrixXd::Zero(n_balls, 2);

      double pos_x = x(0);
      double pos_z = x(1);
      // double phi = x(2);

      // double l_pt_x = pos_x - (L - _radius * 1.5) * std::cos(phi) / 2.0;
      // double l_pt_z = pos_z - (L - _radius * 1.5) * std::sin(phi) / 2.0;

      // for (int i = 0; i < n_balls; ++i) {
      //     double pt_xi = l_pt_x + L * std::cos(phi) / n_balls * i;
      //     double pt_zi = l_pt_z + L * std::sin(phi) / n_balls * i;
      //     v_pts(i, 0) = pt_xi;
      //     v_pts(i, 1) = pt_zi;
      // }

      for (int i = 0; i < n_balls; ++i) {
          v_pts(i, 0) = pos_x;
          v_pts(i, 1) = pos_z;
      }

      return v_pts;
    }

  double _epsilon, _radius, _sigma;
  PlanarSDF _sdf;

};



// class GBP_Cuda{
// public:
//     GBP_Cuda(){};

//     MatrixXd obtain_cov(std::vector<MatrixXd> joint_factor, std::vector<MatrixXd> factor_message, std::vector<MatrixXd> factor_message1);
// };

}


#endif // CUDA_OPERATION_H
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
class CudaOperation{

public:
    CudaOperation(){
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


    PlanarSDF _sdf;

};


}


#endif // CUDA_OPERATION_H
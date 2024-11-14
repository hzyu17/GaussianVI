#pragma once

#ifndef CUDA_OPERATION_H
#define CUDA_OPERATION_H

#include <cuda_runtime.h>
#include <helpers/MatrixHelper.h>
#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include <iomanip>
#include <math.h>

#include <gpmp2/obstacle/SignedDistanceField.h>
#include <gpmp2/kinematics/ArmModel.h>

using namespace Eigen;

namespace gvi{

class PlanarSDF {

public:
  // index and float_index is <row, col>
  typedef std::tuple<size_t, size_t> index;
  typedef Vector2d float_index;
  typedef std::shared_ptr<PlanarSDF> shared_ptr;
  double* data_array_;
  Eigen::Vector2d origin_;

  // geometry setting of signed distance field
  size_t field_rows_, field_cols_;
  double cell_size_;
  Eigen::MatrixXd data_;

public:
  /// constructor
  PlanarSDF() : field_rows_(0), field_cols_(0), cell_size_(0.0) {}

  /// constructor with data
  PlanarSDF(const Eigen::Vector2d& origin, double cell_size, const Eigen::MatrixXd& data) :
      origin_(origin), field_rows_(data.rows()), field_cols_(data.cols()),
      cell_size_(cell_size), data_(data){
        data_array_ = data_.data();
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
    // check point range
    double x_inrange, y_inrange;

    if (point.x() < origin_.x())
      x_inrange = origin_.x();
    else if (point.x() > (origin_.x() + (field_cols_-1.0)*cell_size_))
      x_inrange = origin_.x() + (field_cols_-1.0)*cell_size_;
    else
      x_inrange = point.x();

    if (point.y() < origin_.y())
      y_inrange = origin_.y();
    else if (point.y() > (origin_.y() + (field_rows_-1.0)*cell_size_))
      y_inrange = origin_.y() + (field_rows_-1.0)*cell_size_;
    else
      y_inrange = point.y();

    const double col = (x_inrange - origin_.x()) / cell_size_;
    const double row = (y_inrange - origin_.y()) / cell_size_;
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
    const int lri = static_cast<int>(lr), lci = static_cast<int>(lc),
              hri = static_cast<int>(hr), hci = static_cast<int>(hc);
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
  __host__ __device__ inline double signed_distance(int r, int c) const {
    return data_array_[r + c * field_rows_];
  }

  const Eigen::Vector2d& origin() const { return origin_; }
  double cell_size() const { return cell_size_; }
  const Eigen::MatrixXd& raw_data() const { return data_; }

};


class SignedDistanceField {

public:
  // index and float_index is <row, col>
  typedef std::tuple<size_t, size_t, size_t> index;
  typedef Vector3d float_index;
  typedef std::shared_ptr<SignedDistanceField> shared_ptr;
  Eigen::Vector3d origin_;

  // geometry setting of signed distance field
  int field_rows_, field_cols_, field_z_;
  double cell_size_;
  Eigen::MatrixXd data_;
  double* data_array_;

private:
  

public:
  /// constructor
  SignedDistanceField() : field_rows_(0), field_cols_(0), field_z_(0), cell_size_(0.0) {}

  /// constructor with data
  /// The data here need to change a type
  SignedDistanceField(const Eigen::Vector3d& origin, double cell_size, const std::vector<Eigen::MatrixXd>& data) :
      origin_(origin), field_rows_(data[0].rows()), field_cols_(data[0].cols()), field_z_(data.size()), cell_size_(cell_size)
      {
        // std::cout << "Raw data 1 in GPU :" << data[20](125, 100) << std::endl;
        // std::cout << "Raw data 2 in GPU :" << data[50](50, 50) << std::endl;
        // std::cout << "Raw data 3 in GPU :" << data[100](125, 100) << std::endl;
        // std::cout << "Raw data 4 in GPU :" << data[200](125, 100) << std::endl;

        // Check some points in the raw data
        // print out the indeces and sdf value
        data_.resize(field_rows_, field_cols_*field_z_);
        for (int i = 0; i < field_z_; i++){
          data_.block(0, i*field_cols_, field_rows_, field_cols_) = data[i];
        }
        data_array_ = data_.data();
        // std::cout << "Raw data 1 in GPU after filling :" << data_array_[125 + (100 + 20 * field_cols_) * field_rows_] << std::endl;
        // std::cout << "Raw data 2 in GPU after filling :" << data_array_[50 + (50 + 50 * field_cols_) * field_rows_] << std::endl;
        // std::cout << "Raw data 3 in GPU after filling :" << data_array_[125 + (100 + 100 * field_cols_) * field_rows_] << std::endl;
        // std::cout << "Raw data 4 in GPU after filling :" << data_array_[125 + (100 + 200 * field_cols_) * field_rows_] << std::endl;
      }

  ~SignedDistanceField() {}


  /// give a point, search for signed distance field and (optional) gradient
  /// return signed distance
  __host__ __device__ inline VectorXd getSignedDistance(const Eigen::MatrixXd& point) const {
    int n_balls = point.rows();
    VectorXd signed_dis(n_balls);
    // printf("point = (%lf, %lf, %lf)\n", point(0), point(1), point(2));
    for (int i = 0; i < n_balls; i++){
      const float_index pidx = convertPoint3toCell(point.row(i));
      signed_dis(i) = signed_distance(pidx);
    }
    return signed_dis;
  }

  /// convert between point and cell corrdinate
  __host__ __device__ inline float_index convertPoint3toCell(const Eigen::Vector3d& point) const {
    // check point range
    double x_inrange, y_inrange, z_inrange;

    if (point(0) < origin_(0))
      x_inrange = origin_(0);
    else if (point(0) > (origin_(0) + (field_cols_-1.0)*cell_size_))
      x_inrange = origin_(0) + (field_cols_-1.0)*cell_size_;
    else
      x_inrange = point(0);

    if (point(1) < origin_(1))
      y_inrange = origin_(1);
    else if (point(1) > (origin_(1) + (field_rows_-1.0)*cell_size_))
      y_inrange = origin_(1) + (field_rows_-1.0)*cell_size_;
    else
      y_inrange = point(1);

    if (point(2) < origin_(2))
      z_inrange = origin_(2);
    else if (point(2) > (origin_(2) + (field_z_-1.0)*cell_size_))
      z_inrange = origin_(2) + (field_z_-1.0)*cell_size_;
    else
      z_inrange = point(2);

    const double col = (x_inrange - origin_(0)) / cell_size_;
    const double row = (y_inrange - origin_(1)) / cell_size_;
    const double z   = (z_inrange - origin_(2)) / cell_size_;
    return Vector3d{row, col, z};
  }

  __host__ __device__ inline Eigen::Vector3d convertCelltoPoint2(const float_index& cell) const {
    return origin_ + Eigen::Vector3d(
        cell(1) * cell_size_,
        cell(0) * cell_size_,
        cell(2) * cell_size_);
  }


  /// bilinear interpolation
  __host__ __device__ inline double signed_distance(const float_index& idx) const {
    const double lr = floor(idx(0)), lc = floor(idx(1)), lz = floor(idx(2));
    const double hr = lr + 1.0, hc = lc + 1.0, hz = lz + 1.0;
    const int lri = static_cast<int>(lr), lci = static_cast<int>(lc), lzi = static_cast<int>(lz), 
              hri = static_cast<int>(hr), hci = static_cast<int>(hc), hzi = static_cast<int>(hz);
    // printf("lri = %d, lci = %d, lzi = %d, hri = %d, hci = %d, hzi = %d\n\n", lri, lci, lzi, hri, hci, hzi);
    return
        (hr-idx(0))*(hc-idx(1))*(hz-idx(2))*signed_distance(lri, lci, lzi) +
        (idx(0)-lr)*(hc-idx(1))*(hz-idx(2))*signed_distance(hri, lci, lzi) +
        (hr-idx(0))*(idx(1)-lc)*(hz-idx(2))*signed_distance(lri, hci, lzi) +
        (idx(0)-lr)*(idx(1)-lc)*(hz-idx(2))*signed_distance(hri, hci, lzi) +
        (hr-idx(0))*(hc-idx(1))*(idx(2)-lz)*signed_distance(lri, lci, hzi) +
        (idx(0)-lr)*(hc-idx(1))*(idx(2)-lz)*signed_distance(hri, lci, hzi) +
        (hr-idx(0))*(idx(1)-lc)*(idx(2)-lz)*signed_distance(lri, hci, hzi) +
        (idx(0)-lr)*(idx(1)-lc)*(idx(2)-lz)*signed_distance(hri, hci, hzi);
  }

  /// gradient operator for bilinear interpolation
  /// gradient regrads to float_index
  /// not numerical differentiable at index point

  // __host__ __device__ inline Eigen::Vector2d gradient(const float_index& idx) const {
  //   const double lr = floor(idx(0)), lc = floor(idx(1));
  //   const double hr = lr + 1.0, hc = lc + 1.0;
  //   const size_t lri = static_cast<size_t>(lr), lci = static_cast<size_t>(lc),
  //       hri = static_cast<size_t>(hr), hci = static_cast<size_t>(hc);
  //   return Eigen::Vector2d(
  //       (hc-idx(1)) * (signed_distance(hri, lci)-signed_distance(lri, lci)) +
  //       (idx(1)-lc) * (signed_distance(hri, hci)-signed_distance(lri, hci)),

  //       (hr-idx(0)) * (signed_distance(lri, hci)-signed_distance(lri, lci)) +
  //       (idx(0)-lr) * (signed_distance(hri, hci)-signed_distance(hri, lci)));
  // }

  // access
  __host__ __device__ inline double signed_distance(int r, int c, int z) const {
    return data_array_[r + c * field_rows_ + z * field_rows_ * field_cols_]; //Need to change a way to read
  }

  const Eigen::Vector3d& origin() const { return origin_; }
  double cell_size() const { return cell_size_; }
  const Eigen::MatrixXd& raw_data() const { return data_; }

};


class ForwardKinematics{

public:
  // Denavit-Hartenberg (DH) variables
  Eigen::VectorXd _a;
  Eigen::VectorXd _alpha;
  Eigen::VectorXd _d;
  Eigen::VectorXd _theta_bias;
  Eigen::VectorXi _frames;
  Eigen::VectorXd _radii;
  Eigen::MatrixXd _centers;
  // BodySpheres _body_spheres;

private:

public:
    // Constructors
    ForwardKinematics() {}

    ForwardKinematics(size_t dof, const Eigen::VectorXd& a, const Eigen::VectorXd& alpha, 
                      const Eigen::VectorXd& d, const Eigen::VectorXd theta_bias, 
                      Eigen::VectorXi frames, Eigen::VectorXd radii, Eigen::MatrixXd centers) :
      _a(a), _alpha(alpha), _d(d), _theta_bias(theta_bias), _frames(frames), _radii(radii), _centers(centers)
      {

      }

    __host__ __device__ inline Eigen::VectorXd compute_transformed_sphere_centers(const Eigen::VectorXd& theta) const {
        int num_spheres = _frames.size();
        Eigen::VectorXd pose(3*num_spheres);
        for(int i=0; i<num_spheres; ++i){
            Eigen::VectorXd center {{ _centers(0, i), _centers(1, i), _centers(2, i) }};
            pose.segment(3*i, 3) = forward_kinematics(theta, _frames[i], center);
        }
        return pose;
    }

    __host__ __device__ inline Eigen::Vector3d forward_kinematics(const Eigen::VectorXd& theta, int frame, const Eigen::VectorXd& center) const {
        Eigen::Vector3d pos;
        Eigen::MatrixXd T = Eigen::MatrixXd::Identity(4, 4);
        for(int i=0; i<=frame; ++i){
            T = T*dh_matrix(i, theta(i)+_theta_bias(i));
        }
        Eigen::VectorXd base_pos {{ T(0, 3), T(1, 3), T(2, 3) }};
        Eigen::MatrixXd base_rot {{ T(0, 0), T(0, 1), T(0, 2) },
                                  { T(1, 0), T(1, 0), T(1, 2) },
                                  { T(2, 0), T(2, 1), T(2, 2) }};
        pos = base_pos + base_rot*center;
        return pos;
    }

    __host__ __device__ inline Eigen::MatrixXd dh_matrix(int i, double theta) const {
        Eigen::MatrixXd mat {{ cos(theta), -sin(theta)*cos(_alpha(i)),  sin(theta)*sin(_alpha(i)), _a(i)*cos(theta) },
                             { sin(theta),  cos(theta)*cos(_alpha(i)), -cos(theta)*sin(_alpha(i)), _a(i)*sin(theta) },
                             {          0,             sin(_alpha(i)),             cos(_alpha(i)),            _d(i) },
                             {          0,                          0,                          0,                1 }};
        return mat;
    }

private:
};

class CudaOperation_PlanarPR{
public:
    CudaOperation_PlanarPR(double cost_sigma = 15.5, double epsilon = 0.5, double radius = 1):
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

    void Cuda_init(const MatrixXd& weights){
      cudaMalloc(&_weight_gpu, weights.size() * sizeof(double));
      cudaMalloc(&_data_gpu, _sdf.data_.size() * sizeof(double));
      cudaMalloc(&_class_gpu, sizeof(CudaOperation_PlanarPR));

      cudaMemcpy(_weight_gpu, weights.data(), weights.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_data_gpu, _sdf.data_.data(), _sdf.data_.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_class_gpu, this, sizeof(CudaOperation_PlanarPR), cudaMemcpyHostToDevice);
    }

    void Cuda_free(){
      cudaFree(_weight_gpu);
      cudaFree(_data_gpu);
      cudaFree(_class_gpu);
    }

    void Cuda_init_iter(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols){
      _sigmapts_rows = sigmapts.rows();
      _dim_state = sigmapts_cols;
      _n_states = results.size();

      cudaMalloc(&_sigmapts_gpu, sigmapts.size() * sizeof(double));
      cudaMalloc(&_func_value_gpu, _sigmapts_rows * _n_states * sizeof(double));
    }

    void Cuda_free_iter(){
      cudaFree(_sigmapts_gpu);
      cudaFree(_func_value_gpu); 
    }

    void CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type);

    void costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols);

    void dmuIntegration(const MatrixXd& sigmapts, const MatrixXd& mu, VectorXd& results, const int sigmapts_cols);

    void ddmuIntegration(MatrixXd& results);

    __host__ __device__ double cost_obstacle_planar(const VectorXd& pose, const PlanarSDF& sdf){
      int n_balls = 1;
      double slope = 1;
      MatrixXd checkpoints = vec_balls(pose, n_balls);
      VectorXd signed_distance = sdf.getSignedDistance(checkpoints);
      VectorXd err(signed_distance.size());

      double cost = 0;
      for (int i = 0; i < n_balls; i++){
        if (signed_distance(i) > _epsilon + _radius)
          err(i) =  0.0;
        else
          err(i) =  (_epsilon + _radius - signed_distance(i)) * slope;
        cost += err(i) * err(i) * _sigma;
      }
      
      return cost;
    }

    __host__ __device__ Eigen::MatrixXd vec_balls(const Eigen::VectorXd& x, int n_balls) {
      Eigen::MatrixXd v_pts = Eigen::MatrixXd::Zero(n_balls, 2);

      double pos_x = x(0);
      double pos_z = x(1);

      for (int i = 0; i < n_balls; i++) {
          v_pts(i, 0) = pos_x;
          v_pts(i, 1) = pos_z;
      }
      return v_pts;
    }

  double _epsilon, _radius, _sigma;
  PlanarSDF _sdf;

  int _sigmapts_rows, _dim_state, _n_states;
  double *_weight_gpu, *_data_gpu, *_func_value_gpu, *_sigmapts_gpu, *_mu_gpu;
  CudaOperation_PlanarPR* _class_gpu;

};


class CudaOperation_3dpR{
public:
    CudaOperation_3dpR(double cost_sigma = 15.5, double epsilon = 0.5, double radius = 1):
    _sigma(cost_sigma), _epsilon(epsilon), _radius(radius)
    {
        std::string sdf_file = source_root + "/maps/3dpR/pRSDF3D.bin";  
        gpmp2::SignedDistanceField sdf;
        sdf.loadSDF(sdf_file);

        Vector3d origin;
        origin.setZero();
        origin << -10.0, -10.0, -10.0;

        double cell_size = 0.1;
        _sdf = SignedDistanceField{origin, cell_size, sdf.raw_data()};
    }

    void Cuda_init(const MatrixXd& weights){
      cudaMalloc(&_weight_gpu, weights.size() * sizeof(double));
      cudaMalloc(&_data_gpu, _sdf.data_.size() * sizeof(double));
      cudaMalloc(&_class_gpu, sizeof(CudaOperation_3dpR));

      cudaMemcpy(_weight_gpu, weights.data(), weights.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_class_gpu, this, sizeof(CudaOperation_3dpR), cudaMemcpyHostToDevice);
      cudaMemcpy(_data_gpu, _sdf.data_.data(), _sdf.data_.size() * sizeof(double), cudaMemcpyHostToDevice);
    }

    void Cuda_free(){
      cudaFree(_weight_gpu);
      cudaFree(_data_gpu);
      cudaFree(_class_gpu);
    }

    void Cuda_init_iter(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols){
      _sigmapts_rows = sigmapts.rows();
      _dim_state = sigmapts_cols;
      _n_states = results.size();

      cudaMalloc(&_sigmapts_gpu, sigmapts.size() * sizeof(double));
      cudaMalloc(&_func_value_gpu, _sigmapts_rows * _n_states * sizeof(double));
    }

    void Cuda_free_iter(){
      cudaFree(_sigmapts_gpu);
      cudaFree(_func_value_gpu); 
    }

    void CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type);

    void costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols);

    void dmuIntegration(const MatrixXd& sigmapts, const MatrixXd& mu, VectorXd& results, const int sigmapts_cols);

    void ddmuIntegration(MatrixXd& results);

    __host__ __device__ double cost_obstacle_planar(const VectorXd& pose, const SignedDistanceField& sdf){
      int n_balls = 1;
      double slope = 1;
      MatrixXd checkpoints = vec_balls(pose, n_balls);
      VectorXd signed_distance = sdf.getSignedDistance(checkpoints);
      // printf("signed_distance of pt: (%lf, %lf, %lf) = %lf\n", pose(0), pose(1), pose(2), signed_distance(0));
      VectorXd err(signed_distance.size());

      double cost = 0;
      for (int i = 0; i < n_balls; i++){
        if (signed_distance(i) > _epsilon + _radius)
          err(i) =  0.0;
        else
          err(i) =  (_epsilon + _radius - signed_distance(i)) * slope;
        cost += err(i) * err(i) * _sigma;
      }
      
      return cost;
    }

    __host__ __device__ Eigen::MatrixXd vec_balls(const Eigen::VectorXd& x, int n_balls) {
      Eigen::MatrixXd v_pts = Eigen::MatrixXd::Zero(n_balls, 3);

      double pos_x = x(0);
      double pos_y = x(1);
      double pos_z = x(2);

      for (int i = 0; i < n_balls; i++) {
          v_pts(i, 0) = pos_x;
          v_pts(i, 1) = pos_y;
          v_pts(i, 2) = pos_z;
      }
      return v_pts;
    }

  double _epsilon, _radius, _sigma;
  SignedDistanceField _sdf;

  int _sigmapts_rows, _dim_state, _n_states;
  double *_weight_gpu, *_data_gpu, *_func_value_gpu, *_sigmapts_gpu, *_mu_gpu;
  CudaOperation_3dpR* _class_gpu;

};


class CudaOperation_Quad{
public:
    CudaOperation_Quad(double cost_sigma = 15.5, double epsilon = 0.5, double radius = 1):
    _sigma(cost_sigma), _epsilon(epsilon), _radius(radius)
    {
        MatrixIO _m_io;
        // std::string field_file = source_root + "/maps/2dQuad/field_multiobs.csv";
        std::string field_file = source_root + "/maps/2dQuad/SingleObstacleMap_field.csv";
        MatrixXd field = _m_io.load_csv(field_file);      

        Vector2d origin;
        origin.setZero();
        origin << -20.0, -20.0;

        double cell_size = 0.1;
        _sdf = PlanarSDF{origin, cell_size, field};
    }

    void Cuda_init(const MatrixXd& weights){
      cudaMalloc(&_weight_gpu, weights.size() * sizeof(double));
      cudaMalloc(&_data_gpu, _sdf.data_.size() * sizeof(double));
      cudaMalloc(&_class_gpu, sizeof(CudaOperation_Quad));

      cudaMemcpy(_weight_gpu, weights.data(), weights.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_data_gpu, _sdf.data_.data(), _sdf.data_.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_class_gpu, this, sizeof(CudaOperation_Quad), cudaMemcpyHostToDevice);
    }

    void Cuda_free(){
      cudaFree(_weight_gpu);
      cudaFree(_data_gpu);
      cudaFree(_class_gpu);
    }

    void Cuda_init_iter(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols){
      _sigmapts_rows = sigmapts.rows();
      _dim_state = sigmapts_cols;
      _n_states = results.size();

      cudaMalloc(&_sigmapts_gpu, sigmapts.size() * sizeof(double));
      cudaMalloc(&_func_value_gpu, _sigmapts_rows * _n_states * sizeof(double));
    }

    void Cuda_free_iter(){
      cudaFree(_sigmapts_gpu);
      cudaFree(_func_value_gpu); 
    }

    void CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type);

    void costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols);

    void dmuIntegration(const MatrixXd& sigmapts, const MatrixXd& mu, VectorXd& results, const int sigmapts_cols);

    void ddmuIntegration(MatrixXd& results);

    __host__ __device__ double cost_obstacle_planar(const VectorXd& pose, const PlanarSDF& sdf){
      int n_balls = 5;
      double slope = 5.0;

      MatrixXd checkpoints = vec_balls(pose, n_balls);
      VectorXd signed_distance = sdf.getSignedDistance(checkpoints);
      VectorXd err(signed_distance.size());

      double cost = 0;

      for (int i = 0; i < n_balls; i++){
        if (signed_distance(i) > _epsilon + _radius)
          err(i) =  0.0;
        else
          err(i) =  (_epsilon + _radius - signed_distance(i)) * slope;
        cost += err(i) * err(i) * _sigma;
      }
      
      return cost;
    }

    __host__ __device__ Eigen::MatrixXd vec_balls(const Eigen::VectorXd& x, int n_balls) {
      Eigen::MatrixXd v_pts = Eigen::MatrixXd::Zero(n_balls, 2);

      double L = 5.0;

      double pos_x = x(0);
      double pos_z = x(1);
      double phi = x(2);
      
      double l_pt_x = pos_x - (L - _radius * 1.5) * std::cos(phi) / 2.0;
      double l_pt_z = pos_z - (L - _radius * 1.5) * std::sin(phi) / 2.0;

      for (int i = 0; i < n_balls; i++) {
        double pt_xi = l_pt_x + L * std::cos(phi) / n_balls * i;
        double pt_zi = l_pt_z + L * std::sin(phi) / n_balls * i;
        v_pts(i, 0) = pt_xi;
        v_pts(i, 1) = pt_zi;
      }
      return v_pts;
    }

  double _epsilon, _radius, _sigma;
  PlanarSDF _sdf;

  int _sigmapts_rows, _dim_state, _n_states;
  double *_weight_gpu, *_data_gpu, *_func_value_gpu, *_sigmapts_gpu, *_mu_gpu;
  CudaOperation_Quad* _class_gpu;

};

class CudaOperation_3dArm{
public:
    CudaOperation_3dArm(const Eigen::VectorXd& a, const Eigen::VectorXd& alpha, const Eigen::VectorXd& d, const Eigen::VectorXd& theta_bias,
                        size_t dof, const Eigen::VectorXd& radii, const Eigen::VectorXi& frames, const Eigen::VectorXd centers,
                        double cost_sigma = 15.5, double epsilon = 0.5):
    _radii(radii), _sigma(cost_sigma), _epsilon(epsilon)
    {
        std::string sdf_file = source_root + "/maps/WAM/WAMDeskDataset.bin";  
        gpmp2::SignedDistanceField sdf;
        sdf.loadSDF(sdf_file);

        Vector3d origin;
        origin.setZero();
        origin << -10.0, -10.0, -10.0;

        double cell_size = 0.1;
        _sdf = SignedDistanceField{origin, cell_size, sdf.raw_data()};
        _fk = ForwardKinematics(dof, a, alpha, d, theta_bias, frames, radii, centers.transpose());
    }

    CudaOperation_3dArm(const Eigen::VectorXd& a, const Eigen::VectorXd& alpha, const Eigen::VectorXd& d, const Eigen::VectorXd& theta_bias,
                        size_t dof, const Eigen::VectorXd& radii, const Eigen::VectorXi& frames, const Eigen::MatrixXd centers,
                        double cost_sigma, double epsilon, std::string& sdf_file):
    _radii(radii), _sigma(cost_sigma), _epsilon(epsilon)
    {
        gpmp2::SignedDistanceField sdf;
        sdf.loadSDF(sdf_file);

        Vector3d origin;
        origin.setZero();
        origin << -10.0, -10.0, -10.0;

        double cell_size = 0.1;
        _sdf = SignedDistanceField{origin, cell_size, sdf.raw_data()};
        _fk = ForwardKinematics(dof, a, alpha, d, theta_bias, frames, radii, centers);
    }

    CudaOperation_3dArm(const Eigen::VectorXd& a, const Eigen::VectorXd& alpha, const Eigen::VectorXd& d, const Eigen::VectorXd& theta_bias,
                        size_t dof, const Eigen::VectorXd& radii, const Eigen::VectorXi& frames, const Eigen::MatrixXd centers,
                        double cost_sigma, double epsilon, gpmp2::SignedDistanceField sdf):
    _radii(radii), _sigma(cost_sigma), _epsilon(epsilon)
    {
        _sdf = SignedDistanceField{sdf.origin(), sdf.cell_size(), sdf.raw_data()};
        _fk = ForwardKinematics(dof, a, alpha, d, theta_bias, frames, radii, centers);
    }

    void Cuda_init(const MatrixXd& weights){
      cudaMalloc(&_weight_gpu, weights.size() * sizeof(double));
      cudaMalloc(&_data_gpu, _sdf.data_.size() * sizeof(double));
      cudaMalloc(&_class_gpu, sizeof(CudaOperation_3dArm));

      cudaMemcpy(_weight_gpu, weights.data(), weights.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_class_gpu, this, sizeof(CudaOperation_3dArm), cudaMemcpyHostToDevice);
      cudaMemcpy(_data_gpu, _sdf.data_.data(), _sdf.data_.size() * sizeof(double), cudaMemcpyHostToDevice);
    }

    void Cuda_free(){
      cudaFree(_weight_gpu);
      cudaFree(_data_gpu);
      cudaFree(_class_gpu);
    }

    void Cuda_init_iter(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols){
      _sigmapts_rows = sigmapts.rows();
      _dim_state = sigmapts_cols;
      _n_states = results.size();

      cudaMalloc(&_sigmapts_gpu, sigmapts.size() * sizeof(double));
      cudaMalloc(&_func_value_gpu, _sigmapts_rows * _n_states * sizeof(double));
    }

    void Cuda_free_iter(){
      cudaFree(_sigmapts_gpu);
      cudaFree(_func_value_gpu); 
    }

    void CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type);

    void costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols);

    void dmuIntegration(const MatrixXd& sigmapts, const MatrixXd& mu, VectorXd& results, const int sigmapts_cols);

    void ddmuIntegration(MatrixXd& results);

    __host__ __device__ double cost_obstacle(const Eigen::VectorXd& theta, const SignedDistanceField& sdf, const ForwardKinematics& fk){
      int n_balls = theta.size();
      double slope = 1;
      Eigen::VectorXd pose = fk.compute_transformed_sphere_centers(theta);
      // printf("pose: %f, %f, %f", pose[0], pose[1], pose[2]);
      MatrixXd checkpoints = vec_balls(pose, n_balls);
      VectorXd signed_distance = sdf.getSignedDistance(checkpoints);
      VectorXd err(signed_distance.size());

      double cost = 0;
      for (int i = 0; i < n_balls; i++){
        if (signed_distance(i) > _epsilon + _radii[i])
          err(i) =  0.0;
        else
          err(i) =  (_epsilon + _radii[i] - signed_distance(i)) * slope;
        cost += err(i) * err(i) * _sigma;
      }
      
      return cost;
    }

    __host__ __device__ Eigen::MatrixXd vec_balls(const Eigen::VectorXd& x, int n_balls) {
      Eigen::MatrixXd v_pts = Eigen::MatrixXd::Zero(n_balls, 3);

      double pos_x = x(0);
      double pos_y = x(1);
      double pos_z = x(2);

      for (int i = 0; i < n_balls; i++) {
          v_pts(i, 0) = pos_x;
          v_pts(i, 1) = pos_y;
          v_pts(i, 2) = pos_z;
      }
      return v_pts;
    }

  double _epsilon, _sigma;
  Eigen::VectorXd _radii;
  SignedDistanceField _sdf;
  ForwardKinematics _fk;

  int _sigmapts_rows, _dim_state, _n_states;
  double *_weight_gpu, *_data_gpu, *_func_value_gpu, *_sigmapts_gpu, *_mu_gpu;
  CudaOperation_3dArm* _class_gpu;

};

MatrixXd compute_AT_B_A(MatrixXd& _Lambda, MatrixXd& _target_precision);

void computeTmp_CUDA(Eigen::MatrixXd& tmp, const Eigen::MatrixXd& covariance, const Eigen::MatrixXd& AT_precision_A);


}


#endif // CUDA_OPERATION_H
#pragma once

#ifndef CUDA_OPERATION_H
#define CUDA_OPERATION_H

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <helpers/MatrixHelper.h>
#include <helpers/SerializeEigenMaps.h>
#include <iostream>
#include <memory>
#include <math.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cusolverSp.h>
#include <cusparse_v2.h>
#include "helpers/timer.h"

#include <gpmp2/obstacle/SignedDistanceField.h>
#include <gpmp2/kinematics/ArmModel.h>

using namespace Eigen;

namespace gvi{

struct Point2 {
  double x;
  double y;
};

struct FloatIndex {
  double row;
  double col;
};


class PlanarSDF {
public:
  // FloatIndex is <row, col>
  const double* data_array_;
  Point2 origin_;

  // geometry setting of signed distance field
  size_t field_rows_, field_cols_;
  double cell_size_;

public:
  /// constructor
  PlanarSDF() : field_rows_(0), field_cols_(0), cell_size_(0.0) {}

  /// constructor with data
  PlanarSDF(const Vector2d& origin, double cell_size, const MatrixXd& data) :
      origin_{origin(0), origin(1)}, field_rows_(data.rows()), 
      field_cols_(data.cols()), cell_size_(cell_size){
        data_array_ = data.data(); // Has no use when we pass the data to the GPU
      }

  ~PlanarSDF() {}

  /// return signed distance
  __device__ void getSignedDistance(const Point2* points, int n_points, double* out_signed_distance) const {
    for (int i = 0; i < n_points; i++) {
        FloatIndex fi = convertPoint2toCell(points[i]);
        out_signed_distance[i] = signed_distance(fi);
    }
  }


  /// convert between point and cell corrdinate
  __device__ FloatIndex convertPoint2toCell(const Point2& point) const {
    // check point range
    double x_inrange = point.x;
    double y_inrange = point.y;

    if (point.x < origin_.x)
      x_inrange = origin_.x;
    else if (point.x > (origin_.x + (field_cols_-1.0)*cell_size_))
      x_inrange = origin_.x + (field_cols_-1.0)*cell_size_;

    if (point.y < origin_.y)
      y_inrange = origin_.y;
    else if (point.y > (origin_.y + (field_rows_-1.0)*cell_size_))
      y_inrange = origin_.y + (field_rows_-1.0)*cell_size_;

    const double col = (x_inrange - origin_.x) / cell_size_;
    const double row = (y_inrange - origin_.y) / cell_size_;
    return {row, col};
  }

  // __device__ inline Eigen::Vector2d convertCelltoPoint2(const float_index& cell) const {
  //   return origin_ + Eigen::Vector2d(
  //       cell(1) * cell_size_,
  //       cell(0) * cell_size_);
  // }


  /// bilinear interpolation
  __device__ inline double signed_distance(const FloatIndex& idx) const {
    const double lr = floor(idx.row), lc = floor(idx.col);
    const double hr = lr + 1.0, hc = lc + 1.0;
    const int lri = static_cast<int>(lr), lci = static_cast<int>(lc),
              hri = static_cast<int>(hr), hci = static_cast<int>(hc);
    return
        (hr-idx.row)*(hc-idx.col)*signed_distance(lri, lci) +
        (idx.row-lr)*(hc-idx.col)*signed_distance(hri, lci) +
        (hr-idx.row)*(idx.col-lc)*signed_distance(lri, hci) +
        (idx.row-lr)*(idx.col-lc)*signed_distance(hri, hci);
  }

  /// gradient operator for bilinear interpolation
  /// gradient regrads to float_index
  /// not numerical differentiable at index point

  __device__ inline Point2 gradient(const FloatIndex& idx) const {
    const double lr = floor(idx.row), lc = floor(idx.col);
    const double hr = lr + 1.0, hc = lc + 1.0;
    const size_t lri = static_cast<size_t>(lr), lci = static_cast<size_t>(lc),
        hri = static_cast<size_t>(hr), hci = static_cast<size_t>(hc);
    return {
        (hc-idx.col) * (signed_distance(hri, lci)-signed_distance(lri, lci)) +
        (idx.col-lc) * (signed_distance(hri, hci)-signed_distance(lri, hci)),

        (hr-idx.row) * (signed_distance(lri, hci)-signed_distance(lri, lci)) +
        (idx.row-lr) * (signed_distance(hri, hci)-signed_distance(hri, lci))};
  }

  // access
  __host__ __device__ inline double signed_distance(int r, int c) const {
    return data_array_[r + c * field_rows_];
  }

  const Vector2d origin() const { return Vector2d(origin_.x, origin_.y); }
  double cell_size() const { return cell_size_; }

};


struct Point3 {
  double x;
  double y;
  double z;
};

struct FloatIndex3 {
  double row;
  double col;
  double z;
};

/////    Check Serilization in the maps/3dpR     /////
class SignedDistanceField {
public:
  // Members loaded from the bin file
  Vector3d origin_;
  int field_rows_, field_cols_, field_z_;
  double cell_size_;
  std::vector<Eigen::MatrixXd> data_;

  MatrixXd data_matrix_;

  // Members used in the GPU
  double* data_array_;
  Point3 origin_device_;

public:
  /// constructor
  SignedDistanceField() {}

  /// constructor with data
  SignedDistanceField(const Eigen::Vector3d& origin, double cell_size, const std::vector<Eigen::MatrixXd>& data) :
      origin_(origin), origin_device_{origin(0), origin(1), origin(2)}, field_rows_(data[0].rows()), field_cols_(data[0].cols()), 
      field_z_(data.size()), cell_size_(cell_size), data_(data), data_matrix_(field_rows_, field_cols_ * field_z_)
      {
        for (int i = 0; i < field_z_; i++){
          data_matrix_.block(0, i*field_cols_, field_rows_, field_cols_) = data_[i];
        }
        data_array_ = data_matrix_.data();
      }

  ~SignedDistanceField() {}


  /// give a point, search for signed distance field and (optional) gradient
  /// return signed distance
  __device__ void getSignedDistance(const Point3* points, int n_points, double* out_signed_distance) const {
    for (int i = 0; i < n_points; i++) {
      FloatIndex3 idx = convertPoint3toCell(points[i]);
      out_signed_distance[i] = signed_distance(idx);
    }
  }

  /// convert between point and cell corrdinate
  __device__ inline FloatIndex3 convertPoint3toCell(const Point3& point) const {
    // check point range
    double x_inrange = point.x;
    double y_inrange = point.y;
    double z_inrange = point.z;

    if (point.x < origin_device_.x)
      x_inrange = origin_device_.x;
    else if (point.x > (origin_device_.x + (field_cols_-1.0)*cell_size_))
      x_inrange = origin_device_.x + (field_cols_-1.0)*cell_size_;

    if (point.y < origin_device_.y)
      y_inrange = origin_device_.y;
    else if (point.y > (origin_device_.y + (field_rows_-1.0)*cell_size_))
      y_inrange = origin_device_.y + (field_rows_-1.0)*cell_size_;

    if (point.z < origin_device_.z)
      z_inrange = origin_device_.z;
    else if (point.z > (origin_device_.z + (field_z_-1.0)*cell_size_))
      z_inrange = origin_device_.z + (field_z_-1.0)*cell_size_;

    const double col = (x_inrange - origin_device_.x) / cell_size_;
    const double row = (y_inrange - origin_device_.y) / cell_size_;
    const double z   = (z_inrange - origin_device_.z) / cell_size_;
    return FloatIndex3{row, col, z};
  }

  // __host__ __device__ inline Eigen::Vector3d convertCelltoPoint2(const float_index& cell) const {
  //   return origin_ + Eigen::Vector3d(
  //       cell(1) * cell_size_,
  //       cell(0) * cell_size_,
  //       cell(2) * cell_size_);
  // }


  /// bilinear interpolation
  __device__ inline double signed_distance(const FloatIndex3& idx) const {
    const double lr = floor(idx.row), lc = floor(idx.col), lz = floor(idx.z);
    const double hr = lr + 1.0, hc = lc + 1.0, hz = lz + 1.0;
    const int lri = static_cast<int>(lr), lci = static_cast<int>(lc), lzi = static_cast<int>(lz), 
              hri = static_cast<int>(hr), hci = static_cast<int>(hc), hzi = static_cast<int>(hz);
    // printf("lri = %d, lci = %d, lzi = %d, hri = %d, hci = %d, hzi = %d\n\n", lri, lci, lzi, hri, hci, hzi);
    return
        (hr-idx.row)*(hc-idx.col)*(hz-idx.z)*signed_distance(lri, lci, lzi) +
        (idx.row-lr)*(hc-idx.col)*(hz-idx.z)*signed_distance(hri, lci, lzi) +
        (hr-idx.row)*(idx.col-lc)*(hz-idx.z)*signed_distance(lri, hci, lzi) +
        (idx.row-lr)*(idx.col-lc)*(hz-idx.z)*signed_distance(hri, hci, lzi) +
        (hr-idx.row)*(hc-idx.col)*(idx.z-lz)*signed_distance(lri, lci, hzi) +
        (idx.row-lr)*(hc-idx.col)*(idx.z-lz)*signed_distance(hri, lci, hzi) +
        (hr-idx.row)*(idx.col-lc)*(idx.z-lz)*signed_distance(lri, hci, hzi) +
        (idx.row-lr)*(idx.col-lc)*(idx.z-lz)*signed_distance(hri, hci, hzi);
  }

  __device__ inline double signed_distance(int r, int c, int z) const {
    return data_array_[r + c * field_rows_ + z * field_rows_ * field_cols_];
  }

  const Vector3d origin() const { return origin_; }
  double cell_size() const { return cell_size_; }
  const std::vector<Eigen::MatrixXd>& raw_data() const { return data_; }

  // Remember to check the loadSDF and serialization functions
  void loadSDF(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        std::cout << "File '" << filename << "' does not exist!" << std::endl;
        return;
    }

    std::string fext = filename.substr(filename.find_last_of(".") + 1);
    if (fext == "xml") {
        cereal::XMLInputArchive ia(ifs);
        ia(*this);
    }
    else if (fext == "bin") {
        cereal::BinaryInputArchive ia(ifs);
        ia(*this);
    }
    else {
        cereal::JSONInputArchive ia(ifs);
        ia(*this);
    }

    data_matrix_ = MatrixXd(field_rows_, field_cols_ * field_z_);
    for (int i = 0; i < field_z_; i++){
      data_matrix_.block(0, i*field_cols_, field_rows_, field_cols_) = data_[i];
    }

    data_array_ = data_matrix_.data();
    origin_device_ = {origin_(0), origin_(1), origin_(2)};
  }

  void saveSDF(const std::string filename) {
    std::ofstream ofs(filename.c_str());
    assert(ofs.good());
    std::string fext = filename.substr(filename.find_last_of(".") + 1);
    if (fext == "xml") {
      cereal::XMLOutputArchive archive(ofs);
      archive(CEREAL_NVP(*this));
    }
    else if (fext == "bin") {
      cereal::BinaryOutputArchive archive(ofs);
      archive(*this);
    }
    else {
      cereal::JSONOutputArchive archive(ofs);
      archive(CEREAL_NVP(*this));
    }
  }

  /** Serialization function */
  template<class Archive>
  void serialize(Archive& ar) {
    ar(CEREAL_NVP(origin_));
    ar(CEREAL_NVP(field_rows_));
    ar(CEREAL_NVP(field_cols_));
    ar(CEREAL_NVP(field_z_));
    ar(CEREAL_NVP(cell_size_));
    ar(CEREAL_NVP(data_));
  }
};

class ForwardKinematics{
public:
  // Denavit-Hartenberg (DH) variables
  Eigen::VectorXd _a;
  Eigen::VectorXd _alpha;
  Eigen::VectorXd _d;
  Eigen::VectorXd _theta_bias;

  double* _a_data;
  double* _alpha_data;
  double* _d_data;
  double* _theta_bias_data;

  // Body sphere variables
  int _num_spheres;
  Eigen::VectorXi _frames;
  Eigen::MatrixXd _centers; // Note: centers must be constructed in row-major form (each center belongs to a row)

  int* _frames_data;
  double* _centers_data;

private:

public:
    // Constructors/Destructor
    ForwardKinematics() {}

    ForwardKinematics(const Eigen::VectorXd& a, const Eigen::VectorXd& alpha, 
                      const Eigen::VectorXd& d, const Eigen::VectorXd& theta_bias, int num_spheres,
                      const Eigen::VectorXi& frames, const Eigen::MatrixXd& centers) :
      _a(a), _alpha(alpha), _d(d), _theta_bias(theta_bias), _num_spheres(num_spheres), _frames(frames), _centers(centers)
      {
          _a_data = _a.data();
          _alpha_data = _alpha.data();
          _d_data = _d.data();
          _theta_bias_data = _theta_bias.data();
          _frames_data = _frames.data();
          _centers_data = _centers.data();
      }

    ~ForwardKinematics() {}

    // Compute 3D pose of the center of each sphere on the arm
    __host__ __device__ inline Eigen::VectorXd compute_transformed_sphere_centers(const Eigen::VectorXd& theta) const {
        Eigen::VectorXd pose = Eigen::VectorXd::Zero(3*_num_spheres);
        for(int i=0; i<_num_spheres; ++i){
            Eigen::Vector3d center(centers(i, 0), centers(i, 1), centers(i, 2));
            pose.segment(3*i, 3) = forward_kinematics(theta, frames(i), center);
        }
        return pose;
    }

    // Forward kinematics computed by DH algorithm
    __host__ __device__ inline Eigen::Vector3d forward_kinematics(const Eigen::VectorXd& theta, int frame, const Eigen::Vector3d& center) const {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        for(int i=0; i<=frame; ++i){
            T = T*dh_matrix(i, theta(i)+theta_bias(i));
        }
        Eigen::Vector3d base_pos(T(0, 3), T(1, 3), T(2, 3));
        Eigen::Matrix3d base_rot;
        base_rot << T(0, 0), T(0, 1), T(0, 2),
                    T(1, 0), T(1, 1), T(1, 2),
                    T(2, 0), T(2, 1), T(2, 2);
        Eigen::Vector3d pos = base_pos + base_rot*center;
        return pos;
    }

    // Helper function for computing DH matrices
    __host__ __device__ inline Eigen::Matrix4d dh_matrix(int i, double theta) const {
        Eigen::Matrix4d mat;
        mat << cosf(theta), -sinf(theta)*cosf(alpha(i)),  sinf(theta)*sinf(alpha(i)), a(i)*cosf(theta),
               sinf(theta),  cosf(theta)*cosf(alpha(i)), -cosf(theta)*sinf(alpha(i)), a(i)*sinf(theta),
                         0,              sinf(alpha(i)),              cosf(alpha(i)),             d(i),
                         0,                           0,                           0,                1;
        return mat;
    }

    // access functions
    __host__ __device__ inline double a(int i) const { return _a_data[i]; }
    __host__ __device__ inline double alpha(int i) const { return _alpha_data[i]; }
    __host__ __device__ inline double d(int i) const { return _d_data[i]; }
    __host__ __device__ inline double theta_bias(int i) const { return _theta_bias_data[i]; }
    __host__ __device__ inline double frames(int i) const { return _frames_data[i]; }
    __host__ __device__ inline double centers(int row, int col) const { return _centers_data[3*row + col]; }
};


template <typename SDFType>
class CudaOperation_Base{

public:
    CudaOperation_Base(double cost_sigma, double epsilon, double radius = 1):
    _sigma(cost_sigma), _epsilon(epsilon), _radius(radius){}

    virtual void Cuda_init(const MatrixXd& weights, const MatrixXd& zeromean, const int n_states) = 0;

    virtual void Cuda_free() = 0;

    void GH_parameters_init(const MatrixXd& weights, const MatrixXd& zeromean, const int n_states){
      _sigmapts_rows = zeromean.rows();
      _dim_conf = zeromean.cols();
      _n_states = n_states;

      cudaMalloc(&_weight_gpu, weights.size() * sizeof(double));
      cudaMalloc(&_zeromean_gpu, zeromean.size() * sizeof(double));
      cudaMalloc(&_data_gpu, _data_matrix.size() * sizeof(double));
      cudaMalloc(&_sigmapts_gpu, _sigmapts_rows * _dim_conf * _n_states * sizeof(double));
      cudaMalloc(&_func_value_gpu, _sigmapts_rows * _n_states * sizeof(double));

      cudaMemcpy(_weight_gpu, weights.data(), weights.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_zeromean_gpu, zeromean.data(), zeromean.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_data_gpu, _data_matrix.data(), _data_matrix.size() * sizeof(double), cudaMemcpyHostToDevice);

      cusolverDnCreate(&_cusolverH);
      cublasCreate(&_cublasH);
      cusolverDnCreateSyevjInfo(&_syevj_params);

      size_t covarianceSize = _n_states * _dim_conf * _dim_conf * sizeof(double);
      size_t meanSize       = _n_states * _dim_conf * sizeof(double);

      cudaMalloc(&d_covariance, covarianceSize);
      cudaMalloc(&d_mean, meanSize);
      cudaMalloc(&d_eigenvalues, covarianceSize);
      cudaMalloc(&d_info, _n_states * sizeof(int));

      lwork = 0;
      cusolverDnDsyevjBatched_bufferSize(_cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                        _dim_conf, d_covariance, _dim_conf, d_eigenvalues, &lwork, _syevj_params, _n_states);
      cudaMalloc(&work, lwork * sizeof(double));

      cudaMalloc(&d_eigvec, covarianceSize);
      cudaMalloc(&d_scaledEigvec, covarianceSize);
      cudaMalloc(&d_sqrtP, covarianceSize);
    }

    void GH_parameters_free(){
      cudaFree(_weight_gpu);
      cudaFree(_zeromean_gpu);
      cudaFree(_data_gpu);
      cudaFree(_sigmapts_gpu);
      cudaFree(_func_value_gpu);

      cudaFree(d_covariance);
      cudaFree(d_mean);
      cudaFree(d_eigenvalues);
      cudaFree(d_info);
      cudaFree(work);
      cudaFree(d_eigvec);
      cudaFree(d_scaledEigvec);
      cudaFree(d_sqrtP);

      cusolverDnDestroySyevjInfo(_syevj_params);
      cusolverDnDestroy(_cusolverH);
      // cublasDestroy(_cublasH);
    }
    

    void Cuda_init_iter(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols){
      cudaMemcpy(_sigmapts_gpu, sigmapts.data(), sigmapts.size() * sizeof(double), cudaMemcpyHostToDevice);
    }

    void Cuda_free_iter(){
      // cudaFree(_sigmapts_gpu);
      // cudaFree(_func_value_gpu); 
    }

    void update_sigmapts(const MatrixXd& covariance, const MatrixXd& mean, int dim_state, int num_states, MatrixXd& sigmapts);

    virtual void CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type){}

    virtual void costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols){}

    void dmuIntegration(const MatrixXd& sigmapts, const MatrixXd& mu, VectorXd& results, const int sigmapts_cols);

    void ddmuIntegration(MatrixXd& results);

  double _epsilon, _radius, _sigma;
  SDFType _sdf; // define sdf in the derived class

  MatrixXd _data_matrix;

  int _sigmapts_rows, _dim_conf, _n_states, num_streams;
  double *_weight_gpu, *_data_gpu, *_func_value_gpu, *_sigmapts_gpu, *_mu_gpu, *_zeromean_gpu;

  double* covariance_gpu, *mean_gpu, *d_sigmapt_cuda;  // sigmapts size: _sigmapts_rows x (dim_conf * num_states)

  cusolverDnHandle_t _cusolverH = nullptr;
  cublasHandle_t _cublasH = nullptr;
  syevjInfo_t _syevj_params = nullptr;

  double* d_covariance  = nullptr;
  double* d_mean        = nullptr;
  double* d_eigenvalues = nullptr;
  int*    d_info        = nullptr;
  int     lwork         = 0;
  double* work          = nullptr;
  double* d_eigvec      = nullptr;
  double* d_scaledEigvec = nullptr;
  double* d_sqrtP        = nullptr;

};


class CudaOperation_PlanarPR : public CudaOperation_Base<PlanarSDF>{
public:
    CudaOperation_PlanarPR(double cost_sigma = 15.5, double epsilon = 0.5, double radius = 1):
    CudaOperation_Base(cost_sigma, epsilon, radius)
    {
        MatrixIO _m_io;
        std::string field_file = source_root + "/maps/2dpR/map2/field_multiobs_map2.csv";
        MatrixXd field = _m_io.load_csv(field_file);      

        Vector2d origin;
        origin.setZero();
        origin << -20.0, -10.0;

        double cell_size = 0.1;
        _sdf = PlanarSDF{origin, cell_size, field};

        hostCost._epsilon = _epsilon;
        hostCost._radius = _radius;
        hostCost._sigma = _sigma;
        hostCost._sdf = _sdf;

        _data_matrix = field;
    }

    void Cuda_init(const MatrixXd& weights, const MatrixXd& zeromean, const int n_states) override{
      GH_parameters_init(weights, zeromean, n_states);
      hostCost._sdf.data_array_ = _data_gpu;

      cudaMalloc(&d_cost, sizeof(ObstacleCost));
      cudaMemcpy(d_cost, &hostCost, sizeof(ObstacleCost), cudaMemcpyHostToDevice);
    }

    void Cuda_free(){
      cudaFree(d_cost);
      GH_parameters_free();
    }

    void costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols);

    struct ObstacleCost {
      double _epsilon;
      double _radius;
      double _sigma;

      PlanarSDF _sdf;

      __device__ double cost_obstacle_planar(const double* pose){
        int n_balls = 1;
        double slope = 1;

        Point2 checkpoint = {pose[0], pose[1]};
        double signed_distance = 0.0;

        _sdf.getSignedDistance(&checkpoint, n_balls, &signed_distance);

        double err = 0.0;
        if (signed_distance > _epsilon + _radius)
          err = 0.0;
        else
          err = (_epsilon + _radius - signed_distance) * slope;

        double cost = err * err * _sigma;
        return cost;
      }

    };

  ObstacleCost hostCost;
  ObstacleCost* d_cost;
};


class CudaOperation_Quad : public CudaOperation_Base<PlanarSDF>{
public:
    CudaOperation_Quad(double cost_sigma = 15.5, double epsilon = 0.5, double radius = 1):
    CudaOperation_Base(cost_sigma, epsilon, radius)
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

        hostCost._epsilon = _epsilon;
        hostCost._radius = _radius;
        hostCost._sigma = _sigma;
        hostCost._sdf = _sdf;

        _data_matrix = field;
    }

    void Cuda_init(const MatrixXd& weights, const MatrixXd& zeromean, const int n_states) override{
      GH_parameters_init(weights, zeromean, n_states);
      hostCost._sdf.data_array_ = _data_gpu;

      cudaMalloc(&d_cost, sizeof(ObstacleCost));
      cudaMemcpy(d_cost, &hostCost, sizeof(ObstacleCost), cudaMemcpyHostToDevice);
    }

    void Cuda_free() override{
      cudaFree(d_cost);
      GH_parameters_free();
    }

    void costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols);

    struct ObstacleCost {
      double _epsilon;
      double _radius;
      double _sigma;

      PlanarSDF _sdf;

      __device__ double cost_obstacle_planar(const double* pose){
        constexpr int n_balls = 5;
        double slope = 1.0; // I can use sigma to replace the slope

        Point2 checkpoints[n_balls];
        vec_balls(pose, n_balls, checkpoints);

        double signed_distance[n_balls];
        _sdf.getSignedDistance(checkpoints, n_balls, signed_distance);
  
        double cost = 0;
  
        for (int i = 0; i < n_balls; i++){
          double err = 0.0;
          if (signed_distance[i] > _epsilon + _radius)
            err =  0.0;
          else
            err = (_epsilon + _radius - signed_distance[i]) * slope;
          cost += err * err * _sigma;
        }
        
        return cost;
      }

      __device__ void vec_balls(const double* pose, int n_balls, Point2* v_pts) {  
        double L = 5.0;
        double pos_x = pose[0];
        double pos_z = pose[1];
        double phi = pose[2];
        
        double l_pt_x = pos_x - (L - _radius * 1.5) * std::cos(phi) / 2.0;
        double l_pt_z = pos_z - (L - _radius * 1.5) * std::sin(phi) / 2.0;
  
        for (int i = 0; i < n_balls; i++) {
          double pt_xi = l_pt_x + L * std::cos(phi) / n_balls * i;
          double pt_zi = l_pt_z + L * std::sin(phi) / n_balls * i;
          v_pts[i].x = pt_xi;
          v_pts[i].y = pt_zi;
        }
      }

    };

  ObstacleCost hostCost;
  ObstacleCost* d_cost;
};


class CudaOperation_3dpR : public CudaOperation_Base<SignedDistanceField>{
public:
    CudaOperation_3dpR(double cost_sigma = 15.5, double epsilon = 0.5, double radius = 1):
    CudaOperation_Base(cost_sigma, epsilon, radius)
    {
        std::string sdf_file = source_root + "/maps/3dpR/pRSDF3D_cereal.bin";
        _sdf.loadSDF(sdf_file);

        hostCost._epsilon = _epsilon;
        hostCost._radius = _radius;
        hostCost._sigma = _sigma;
        hostCost._sdf = _sdf;

        _data_matrix = _sdf.data_matrix_;
    }

    void Cuda_init(const MatrixXd& weights, const MatrixXd& zeromean, const int n_states) override{
      GH_parameters_init(weights, zeromean, n_states);
      hostCost._sdf.data_array_ = _data_gpu;

      cudaMalloc(&d_cost, sizeof(ObstacleCost));
      cudaMemcpy(d_cost, &hostCost, sizeof(ObstacleCost), cudaMemcpyHostToDevice);
    }

    void Cuda_free() override{
      cudaFree(d_cost);
      GH_parameters_free();
    }

    void costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols);

    struct ObstacleCost {
      double _epsilon;
      double _radius;
      double _sigma;

      SignedDistanceField _sdf;

      __device__ double cost_obstacle_planar(const double* pose){
        int n_balls = 1;
        double slope = 1;

        Point3 checkpoint = {pose[0], pose[1], pose[2]};
        double signed_distance = 0.0;

        _sdf.getSignedDistance(&checkpoint, n_balls, &signed_distance);
        // printf("signed_distance of pt: (%lf, %lf, %lf) = %lf\n", pose(0), pose(1), pose(2), signed_distance(0));
        
        double err = 0.0;
        if (signed_distance > _epsilon + _radius)
          err = 0.0;
        else
          err = (_epsilon + _radius - signed_distance) * slope;
  
        double cost = err * err * _sigma;
        return cost;
      }

    };

  ObstacleCost hostCost;
  ObstacleCost* d_cost;
};


class CudaOperation_3dArm : public CudaOperation_Base<SignedDistanceField>{
public:
    CudaOperation_3dArm(const Eigen::VectorXd& a, const Eigen::VectorXd& alpha, const Eigen::VectorXd& d, const Eigen::VectorXd& theta_bias,
                        const Eigen::VectorXd& radii, const Eigen::VectorXi& frames, const Eigen::VectorXd& centers,
                        double cost_sigma = 15.5, double epsilon = 0.5):
    _radii(radii), CudaOperation_Base(cost_sigma, epsilon)
    {
        std::string sdf_file = source_root + "/maps/WAM/WAMDeskDataset.bin";  
        _sdf.loadSDF(sdf_file);
        // gpmp2::SignedDistanceField sdf;
        // sdf.loadSDF(sdf_file);
        // _sdf = SignedDistanceField{sdf.origin(), sdf.cell_size(), sdf.raw_data()};

        _radii_data = _radii.data();
        const int num_spheres = frames.size();
        _fk = ForwardKinematics(a, alpha, d, theta_bias, num_spheres, frames, centers);

        _data_matrix = _sdf.data_matrix_;
    }

    CudaOperation_3dArm(const Eigen::VectorXd& a, const Eigen::VectorXd& alpha, const Eigen::VectorXd& d, const Eigen::VectorXd& theta_bias,
                        const Eigen::VectorXd& radii, const Eigen::VectorXi& frames, const Eigen::MatrixXd& centers,
                        double cost_sigma, double epsilon, gpmp2::SignedDistanceField sdf):
    _radii(radii), CudaOperation_Base(cost_sigma, epsilon) // we can replace the input with the sdf class we defined
    {
        _sdf = SignedDistanceField{sdf.origin(), sdf.cell_size(), sdf.raw_data()}; 

        _radii_data = _radii.data();
        const int num_spheres = frames.size();
        _fk = ForwardKinematics(a, alpha, d, theta_bias, num_spheres, frames, centers);

        _data_matrix = _sdf.data_matrix_;
    }

    void Cuda_init(const MatrixXd& weights, const MatrixXd& zeromean, const int n_states) override{
      cudaMalloc(&_class_gpu, sizeof(CudaOperation_3dArm));
      cudaMalloc(&_a_gpu, _fk._a.size() * sizeof(double));
      cudaMalloc(&_alpha_gpu, _fk._alpha.size() * sizeof(double));
      cudaMalloc(&_d_gpu, _fk._d.size() * sizeof(double));
      cudaMalloc(&_theta_gpu, _fk._theta_bias.size() * sizeof(double));
      cudaMalloc(&_rad_gpu, _radii.size() * sizeof(double));
      cudaMalloc(&_frames_gpu, _fk._frames.size() * sizeof(int));
      cudaMalloc(&_centers_gpu, _fk._centers.size() * sizeof(double));

      cudaMemcpy(_class_gpu, this, sizeof(CudaOperation_3dArm), cudaMemcpyHostToDevice);
      cudaMemcpy(_a_gpu, _fk._a.data(), _fk._a.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_alpha_gpu, _fk._alpha.data(), _fk._alpha.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_d_gpu, _fk._d.data(), _fk._d.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_theta_gpu, _fk._theta_bias.data(), _fk._theta_bias.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_rad_gpu, _radii.data(), _radii.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_frames_gpu, _fk._frames.data(), _fk._frames.size() * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(_centers_gpu, _fk._centers.data(), _fk._centers.size() * sizeof(double), cudaMemcpyHostToDevice);

      GH_parameters_init(weights, zeromean, n_states);
    }

    void Cuda_free() override{
      cudaFree(_class_gpu);
      cudaFree(_a_gpu);
      cudaFree(_alpha_gpu);
      cudaFree(_d_gpu);
      cudaFree(_theta_gpu);
      cudaFree(_rad_gpu);
      cudaFree(_frames_gpu);
      cudaFree(_centers_gpu);
      GH_parameters_free();
    }

    void CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type);

    void costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols);

    __device__ double cost_obstacle(const VectorXd& theta, const SignedDistanceField& sdf, const ForwardKinematics& fk){
      constexpr int MAX_BALLS = 128;
      int n_balls = theta.size();
      double slope = 1;
      VectorXd pose = fk.compute_transformed_sphere_centers(theta);

      Point3 checkpoints[MAX_BALLS];
      vec_balls(pose, n_balls, checkpoints);

      double signed_distance[MAX_BALLS] = {0};

      sdf.getSignedDistance(checkpoints, n_balls, signed_distance);

      double cost = 0;
      for (int i = 0; i < n_balls; i++){
        double err = 0.0;
        if (signed_distance[i] > _epsilon + radius(i))
          err =  0.0;
        else
          err =  (_epsilon + radius(i) - signed_distance[i]) * slope;
        cost += err * err * _sigma;
      }
      
      return cost;
    }

    // Reshape from vector to a matrix
    __device__ void vec_balls(const Eigen::VectorXd& x, int n_balls, Point3* pts) {
      for (int i = 0; i < n_balls; i++) {
        pts[i].x = x(3 * i);
        pts[i].y = x(3 * i + 1);
        pts[i].z = x(3 * i + 2);
      }
    }

    __host__ __device__ inline double radius(int i) const {
      return _radii_data[i];
    }

  VectorXd _radii;
  double* _radii_data;
  ForwardKinematics _fk;

  double *_a_gpu, *_alpha_gpu, *_d_gpu, *_theta_gpu, *_rad_gpu, *_centers_gpu;
  int *_frames_gpu;
  CudaOperation_3dArm* _class_gpu;

};


class CudaOperation_SLR{
public:
    CudaOperation_SLR(const MatrixXd& sigmapts, const VectorXd& weights, const MatrixXd& x_bar, int dim_states, int n_states) :
    _sigmapts_rows(sigmapts.rows()), _dim_state(dim_states), _n_states(n_states)
    {
      cudaMalloc(&_sigmapts_gpu, sigmapts.size() * sizeof(double));
      cudaMalloc(&_y_sigmapts_gpu, sigmapts.size() * sizeof(double));
      cudaMalloc(&_x_bar_gpu, x_bar.size() * sizeof(double));
      cudaMalloc(&_y_bar_gpu, x_bar.size() * sizeof(double));
      cudaMalloc(&_weights_gpu, weights.size() * sizeof(double));

      cudaMemcpy(_sigmapts_gpu, sigmapts.data(), sigmapts.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_x_bar_gpu, x_bar.data(), x_bar.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_weights_gpu, weights.data(), weights.size() * sizeof(double), cudaMemcpyHostToDevice);

      cudaMemset(_y_sigmapts_gpu, 0, sigmapts.size() * sizeof(double));
      cudaMemset(_y_bar_gpu, 0, x_bar.size() * sizeof(double));
    }

    ~CudaOperation_SLR(){
      cudaFree(_sigmapts_gpu);
      cudaFree(_y_sigmapts_gpu);
      cudaFree(_x_bar_gpu);
      cudaFree(_y_bar_gpu);
      cudaFree(_weights_gpu);
    }

    void expectationIntegration(MatrixXd& y_bar);

    void covarianceIntegration(MatrixXd& results);

    int _sigmapts_rows, _dim_state, _n_states;
    double *_sigmapts_gpu, *_y_sigmapts_gpu, *_x_bar_gpu, *_y_bar_gpu, *_weights_gpu;

};

MatrixXd compute_AT_B_A(MatrixXd& _Lambda, MatrixXd& _target_precision);

void computeTmp_CUDA(Eigen::MatrixXd& tmp, const Eigen::MatrixXd& covariance, const Eigen::MatrixXd& AT_precision_A);

}


#endif // CUDA_OPERATION_H
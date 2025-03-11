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
  VectorXd _a;
  VectorXd _alpha;
  VectorXd _d;
  VectorXd _theta_bias;

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
  int _num_joints;

private:

public:
    // Constructors/Destructor
    ForwardKinematics() {}

    ForwardKinematics(const Eigen::VectorXd& a, const Eigen::VectorXd& alpha,
                      const Eigen::VectorXd& d, const Eigen::VectorXd& theta_bias,
                      const Eigen::VectorXi& frames, const Eigen::MatrixXd& centers) :
      _a(a), _alpha(alpha), _d(d), _theta_bias(theta_bias), _num_spheres(frames.size()), _frames(frames), _centers(centers), _num_joints(a.size())
      {
          _a_data = _a.data();
          _alpha_data = _alpha.data();
          _d_data = _d.data();
          _theta_bias_data = _theta_bias.data();
          _frames_data = _frames.data();
          _centers_data = _centers.data();
      }

    ~ForwardKinematics() {}

    __device__ inline void compute_transformed_sphere_centers(const double* theta, Point3* pose) const {
        // Precompute DH matrices for all joints.
        constexpr int MATRIX_ELEMENTS = 16;
        constexpr int MAX_JOINTS = 10;
        double dh_mats[MATRIX_ELEMENTS * MAX_JOINTS];
        precompute_dh_matrices(theta, dh_mats, _num_joints);

        for (int i = 0; i < _num_spheres; ++i) {
            int frame = frames(i);
            double center[3] = {centers(i, 0), centers(i, 1), centers(i, 2)};
            double pos[3];
            forward_kinematics(dh_mats, frame, center, pos);
            pose[i].x = pos[0];
            pose[i].y = pos[1];
            pose[i].z = pos[2];
        }
    }

    __device__ inline void precompute_dh_matrices(const double* theta, double* dh_mats, int n) const {
        constexpr int matrix_element = 16;
        double T[matrix_element];
        double mul_result[matrix_element];
        identity(T); // Set T to the identity matrix in column-major order
        
        for (int i = 0; i < n; i++) {
          double th = theta[i] + theta_bias(i);
          double dh_mat[matrix_element];
          dh_matrix(i, th, dh_mat);
          mat_mul(T, dh_mat, mul_result, 4);
          for (int j = 0; j < matrix_element; j++) {
            T[j] = mul_result[j];
            dh_mats[i*matrix_element + j] = mul_result[j];
          }
        }
    }

    __device__ inline void forward_kinematics(const double* dh_mats, int frame, const double* center, double* pos) const {
        constexpr int matrix_element = 16;
        // Retrieve the cumulative transformation matrix for the given frame (column-major order)
        const double* T = &dh_mats[frame * matrix_element];
        // Extract the translation component from T (last column: indices 12, 13, 14)
        double base_pos[3] = { T[12], T[13], T[14] };
        // Compute the rotated center using the upper 3x3 rotation matrix from T
        double rotated[3];
        rotated[0] = T[0] * center[0] + T[4] * center[1] + T[8]  * center[2];
        rotated[1] = T[1] * center[0] + T[5] * center[1] + T[9]  * center[2];
        rotated[2] = T[2] * center[0] + T[6] * center[1] + T[10] * center[2];
        // Compute final position as the sum of translation and rotated center
        pos[0] = base_pos[0] + rotated[0];
        pos[1] = base_pos[1] + rotated[1];
        pos[2] = base_pos[2] + rotated[2];
    }

    __device__ inline void identity(double* M) const {
        M[0]  = 1; M[1]  = 0; M[2]  = 0; M[3]  = 0;
        M[4]  = 0; M[5]  = 1; M[6]  = 0; M[7]  = 0;
        M[8]  = 0; M[9]  = 0; M[10] = 1; M[11] = 0;
        M[12] = 0; M[13] = 0; M[14] = 0; M[15] = 1;
    }

    __device__ inline void mat_mul(const double* A, const double* B, double* C, const int dim) const {
      for (int col = 0; col < dim; col++) {
        for (int row = 0; row < dim; row++) {
            double sum = 0.0;
            for (int k = 0; k < dim; k++) {
                sum += A[row + k*dim] * B[k + col*dim];
            }
            C[row + col*dim] = sum;
              }
          }
    }

    __device__ inline void dh_matrix(int i, double theta, double* mat) const {
        double ct = cos(theta);
        double st = sin(theta);
        double ca = cos(alpha(i));
        double sa = sin(alpha(i));
        // Column 0
        mat[0] = ct;
        mat[1] = st;
        mat[2] = 0;
        mat[3] = 0;
        // Column 1
        mat[4] = -st * ca;
        mat[5] = ct * ca;
        mat[6] = sa;
        mat[7] = 0;
        // Column 2
        mat[8]  = st * sa;
        mat[9]  = -ct * sa;
        mat[10] = ca;
        mat[11] = 0;
        // Column 3
        mat[12] = a(i) * ct;
        mat[13] = a(i) * st;
        mat[14] = d(i);
        mat[15] = 1;
    }

    // access functions
    __device__ inline double a(int i) const { return _a_data[i]; }
    __device__ inline double alpha(int i) const { return _alpha_data[i]; }
    __device__ inline double d(int i) const { return _d_data[i]; }
    __device__ inline double theta_bias(int i) const { return _theta_bias_data[i]; }
    __device__ inline int frames(int i) const { return _frames_data[i]; }
    __device__ inline double centers(int row, int col) const { return _centers_data[3*row + col]; }
};

template <typename Derived>
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
    

    void copy_sigma(const MatrixXd& sigmapts){
      cudaMemcpy(_sigmapts_gpu, sigmapts.data(), sigmapts.size() * sizeof(double), cudaMemcpyHostToDevice);
    }

    void update_sigmapts(const MatrixXd& covariance, const MatrixXd& mean, int dim_state, int num_states, MatrixXd& sigmapts);

    virtual void CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type){}

    void costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols);

    void dmuIntegration(const MatrixXd& sigmapts, const MatrixXd& mu, VectorXd& results, const int sigmapts_cols);

    void ddmuIntegration(MatrixXd& results);

  double _epsilon, _radius, _sigma;

  MatrixXd _data_matrix;

  int _sigmapts_rows, _dim_conf, _n_states;
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


class CudaOperation_PlanarPR : public CudaOperation_Base<CudaOperation_PlanarPR>{
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

        hostCost._epsilon = _epsilon;
        hostCost._radius = _radius;
        hostCost._sigma = _sigma;
        hostCost._sdf = PlanarSDF{origin, cell_size, field};

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

    struct ObstacleCost {
      double _epsilon;
      double _radius;
      double _sigma;

      PlanarSDF _sdf;

      __device__ double cost_obstacle(const double* pose){
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


class CudaOperation_Quad : public CudaOperation_Base<CudaOperation_Quad>{
public:
    CudaOperation_Quad(double cost_sigma = 15.5, double epsilon = 0.5, double radius = 1, const std::string& map_name = ""):
    CudaOperation_Base(cost_sigma, epsilon, radius)
    {
        MatrixIO _m_io;
        std::string field_file = source_root + "/python/sdf_robot/map/planar/" + map_name + "_field.csv";
        std::cout << "field_file: " << field_file << std::endl;
        MatrixXd field = _m_io.load_csv(field_file);

        Vector2d origin;
        origin.setZero();
        if(map_name == "MultiObstacleLongRangeMap") {
          origin << -50.0, -50.0;
        } else {
          origin << -20.0, -20.0;
        }

        double cell_size = 0.1;

        hostCost._epsilon = _epsilon;
        hostCost._radius = _radius;
        hostCost._sigma = _sigma;
        hostCost._sdf = PlanarSDF{origin, cell_size, field};

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

    struct ObstacleCost {
      double _epsilon;
      double _radius;
      double _sigma;

      PlanarSDF _sdf;

      __device__ double cost_obstacle(const double* pose){
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


class CudaOperation_3dpR : public CudaOperation_Base<CudaOperation_3dpR>{
public:
    CudaOperation_3dpR(double cost_sigma = 15.5, double epsilon = 0.5, double radius = 1):
    CudaOperation_Base(cost_sigma, epsilon, radius)
    {
        std::string sdf_file = source_root + "/maps/3dpR/pRSDF3D_cereal.bin";

        hostCost._epsilon = _epsilon;
        hostCost._radius = _radius;
        hostCost._sigma = _sigma;
        hostCost._sdf.loadSDF(sdf_file);

        _data_matrix = hostCost._sdf.data_matrix_;
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

    struct ObstacleCost {
      double _epsilon;
      double _radius;
      double _sigma;

      SignedDistanceField _sdf;

      __device__ double cost_obstacle(const double* pose){
        int n_balls = 1;
        double slope = 1;

        Point3 checkpoint = {pose[0], pose[1], pose[2]};
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


class CudaOperation_3dArm : public CudaOperation_Base<CudaOperation_3dArm>{
public:
    CudaOperation_3dArm(const Eigen::VectorXd& a, const Eigen::VectorXd& alpha, const Eigen::VectorXd& d, const Eigen::VectorXd& theta_bias,
                        const Eigen::VectorXd& radii, const Eigen::VectorXi& frames, const Eigen::MatrixXd& centers,
                        double cost_sigma = 15.5, double epsilon = 0.5):
    _radii(radii), CudaOperation_Base(cost_sigma, epsilon)
    {
        std::string sdf_file = source_root + "/maps/WAM/WAMDeskDataset_cereal.bin";
        
        hostCost._epsilon = _epsilon;
        hostCost._sigma = _sigma;
        hostCost._sdf.loadSDF(sdf_file);
        hostCost._fk = ForwardKinematics(a, alpha, d, theta_bias, frames, centers);

        _data_matrix = hostCost._sdf.data_matrix_;
    }

    // CudaOperation_3dArm(const Eigen::VectorXd& a, const Eigen::VectorXd& alpha, const Eigen::VectorXd& d, const Eigen::VectorXd& theta_bias,
    //                     const Eigen::VectorXd& radii, const Eigen::VectorXi& frames, const Eigen::MatrixXd& centers,
    //                     double cost_sigma, double epsilon, gpmp2::SignedDistanceField sdf):
    // _radii(radii), CudaOperation_Base(cost_sigma, epsilon) // we can replace the input with the sdf class we defined
    // {
    //     _sdf = SignedDistanceField{sdf.origin(), sdf.cell_size(), sdf.raw_data()};

    //     _radii_data = _radii.data();
    //     const int num_spheres = frames.size();
    //     _fk = ForwardKinematics(a, alpha, d, theta_bias, num_spheres, frames, centers);

    //     _data_matrix = _sdf.data_matrix_;
    // }

    void Cuda_init(const MatrixXd& weights, const MatrixXd& zeromean, const int n_states) override{
      GH_parameters_init(weights, zeromean, n_states);

      cudaMalloc(&_a_gpu, hostCost._fk._a.size() * sizeof(double));
      cudaMalloc(&_alpha_gpu, hostCost._fk._alpha.size() * sizeof(double));
      cudaMalloc(&_d_gpu, hostCost._fk._d.size() * sizeof(double));
      cudaMalloc(&_theta_gpu, hostCost._fk._theta_bias.size() * sizeof(double));
      cudaMalloc(&_rad_gpu, _radii.size() * sizeof(double));
      cudaMalloc(&_frames_gpu, hostCost._fk._frames.size() * sizeof(int));
      cudaMalloc(&_centers_gpu, hostCost._fk._centers.size() * sizeof(double));

      cudaMemcpy(_a_gpu, hostCost._fk._a.data(), hostCost._fk._a.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_alpha_gpu, hostCost._fk._alpha.data(), hostCost._fk._alpha.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_d_gpu, hostCost._fk._d.data(), hostCost._fk._d.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_theta_gpu, hostCost._fk._theta_bias.data(), hostCost._fk._theta_bias.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_rad_gpu, _radii.data(), _radii.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_frames_gpu, hostCost._fk._frames.data(), hostCost._fk._frames.size() * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(_centers_gpu, hostCost._fk._centers.data(), hostCost._fk._centers.size() * sizeof(double), cudaMemcpyHostToDevice);

      hostCost._sdf.data_array_ = _data_gpu;

      hostCost._fk._a_data = _a_gpu;
      hostCost._fk._alpha_data = _alpha_gpu;
      hostCost._fk._d_data = _d_gpu;
      hostCost._fk._theta_bias_data = _theta_gpu;
      hostCost._fk._frames_data = _frames_gpu;
      hostCost._fk._centers_data = _centers_gpu;

      hostCost._radii_data = _rad_gpu;

      cudaMalloc(&d_cost, sizeof(ObstacleCost));
      cudaMemcpy(d_cost, &hostCost, sizeof(ObstacleCost), cudaMemcpyHostToDevice);
    }

    void Cuda_free() override{
      cudaFree(d_cost);
      cudaFree(_a_gpu);
      cudaFree(_alpha_gpu);
      cudaFree(_d_gpu);
      cudaFree(_theta_gpu);
      cudaFree(_rad_gpu);
      cudaFree(_frames_gpu);
      cudaFree(_centers_gpu);
      GH_parameters_free();
    }

    struct ObstacleCost {
      double _epsilon;
      double _sigma;
      double* _radii_data;

      SignedDistanceField _sdf;
      ForwardKinematics _fk;

      __device__ double cost_obstacle(const double* theta){
        constexpr int MAX_BALLS = 32;
        int n_balls = _fk._num_spheres;
        double slope = 1;

        Point3 checkpoints[MAX_BALLS];
        _fk.compute_transformed_sphere_centers(theta, checkpoints);
  
        double signed_distance[MAX_BALLS] = {0};
        _sdf.getSignedDistance(checkpoints, n_balls, signed_distance);
  
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

      __device__ inline double radius(int i) const {
        return _radii_data[i];
      }

    };

  ObstacleCost hostCost;
  ObstacleCost* d_cost;
  
  VectorXd _radii;

  double *_a_gpu, *_alpha_gpu, *_d_gpu, *_theta_gpu, *_rad_gpu, *_centers_gpu;
  int *_frames_gpu;
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
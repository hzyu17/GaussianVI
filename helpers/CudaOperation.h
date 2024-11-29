#pragma once

#ifndef CUDA_OPERATION_H
#define CUDA_OPERATION_H

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <helpers/MatrixHelper.h>
#include <helpers/SerializeEigenMaps.h>
#include <iostream>
#include <memory>

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
  std::vector<Eigen::MatrixXd> data_;
  Eigen::MatrixXd data_matrix_;
  double* data_array_;

public:
  /// constructor
  SignedDistanceField() {}

  /// constructor with data
  SignedDistanceField(const Eigen::Vector3d& origin, double cell_size, const std::vector<Eigen::MatrixXd>& data) :
      origin_(origin), field_rows_(data[0].rows()), field_cols_(data[0].cols()), 
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

    data_matrix_ = Eigen::MatrixXd(field_rows_, field_cols_ * field_z_);
    for (int i = 0; i < field_z_; i++){
      data_matrix_.block(0, i*field_cols_, field_rows_, field_cols_) = data_[i];
    }

    data_array_ = data_matrix_.data();
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


  // access
  __host__ __device__ inline double signed_distance(int r, int c, int z) const {
    return data_array_[r + c * field_rows_ + z * field_rows_ * field_cols_]; //Need to change a way to read
  }

  const Eigen::Vector3d& origin() const { return origin_; }
  double cell_size() const { return cell_size_; }
  const std::vector<Eigen::MatrixXd>& raw_data() const { return data_; }

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

template <typename SDFType>
class CudaOperation_Base{

public:
    CudaOperation_Base(double cost_sigma, double epsilon, double radius):
    _sigma(cost_sigma), _epsilon(epsilon), _radius(radius){}

    virtual void Cuda_init(const Eigen::MatrixXd& weights) = 0;

    virtual void Cuda_free() = 0;

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

    virtual void CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type){}

    virtual void costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols){}

    void dmuIntegration(const MatrixXd& sigmapts, const MatrixXd& mu, VectorXd& results, const int sigmapts_cols);

    void ddmuIntegration(MatrixXd& results);

  double _epsilon, _radius, _sigma;
  SDFType _sdf; // define sdf in the derived class

  int _sigmapts_rows, _dim_state, _n_states;
  double *_weight_gpu, *_data_gpu, *_func_value_gpu, *_sigmapts_gpu, *_mu_gpu;
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

    void CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type);

    void costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols);

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

  CudaOperation_PlanarPR* _class_gpu;
};


class CudaOperation_3dpR : public CudaOperation_Base<SignedDistanceField>{
public:
    CudaOperation_3dpR(double cost_sigma = 15.5, double epsilon = 0.5, double radius = 1):
    CudaOperation_Base(cost_sigma, epsilon, radius)
    {
        std::string sdf_file = source_root + "/maps/3dpR/pRSDF3D.bin";
        _sdf.loadSDF(sdf_file);
    }

    void Cuda_init(const MatrixXd& weights){
      cudaMalloc(&_weight_gpu, weights.size() * sizeof(double));
      cudaMalloc(&_data_gpu, _sdf.data_matrix_.size() * sizeof(double));
      cudaMalloc(&_class_gpu, sizeof(CudaOperation_3dpR));

      cudaMemcpy(_weight_gpu, weights.data(), weights.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_class_gpu, this, sizeof(CudaOperation_3dpR), cudaMemcpyHostToDevice);
      cudaMemcpy(_data_gpu, _sdf.data_matrix_.data(), _sdf.data_matrix_.size() * sizeof(double), cudaMemcpyHostToDevice);
    }

    void Cuda_free(){
      cudaFree(_weight_gpu);
      cudaFree(_data_gpu);
      cudaFree(_class_gpu);
    }

    void CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type);

    void costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols);

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

  CudaOperation_3dpR* _class_gpu;
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
    }

    void Cuda_init(const MatrixXd& weights) override{
      cudaMalloc(&_weight_gpu, weights.size() * sizeof(double));
      cudaMalloc(&_data_gpu, _sdf.data_.size() * sizeof(double));
      cudaMalloc(&_class_gpu, sizeof(CudaOperation_Quad));

      cudaMemcpy(_weight_gpu, weights.data(), weights.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_data_gpu, _sdf.data_.data(), _sdf.data_.size() * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(_class_gpu, this, sizeof(CudaOperation_Quad), cudaMemcpyHostToDevice);
    }

    void Cuda_free() override{
      cudaFree(_weight_gpu);
      cudaFree(_data_gpu);
      cudaFree(_class_gpu);
    }

    void CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type);

    void costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols);

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

  CudaOperation_Quad* _class_gpu;
};

MatrixXd compute_AT_B_A(MatrixXd& _Lambda, MatrixXd& _target_precision);

void computeTmp_CUDA(Eigen::MatrixXd& tmp, const Eigen::MatrixXd& covariance, const Eigen::MatrixXd& AT_precision_A);

}


#endif // CUDA_OPERATION_H
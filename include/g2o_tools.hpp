#ifndef G2O_TOOLS_HPP
#define G2O_TOOLS_HPP

#include "vmath.hpp"

#include <cstddef>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/se3quat.h>
#include <iostream>
#include <memory>
#include <vector>

namespace neves {

class VertexSO3 : public g2o::BaseVertex<3, SO3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = g2o::BaseVertex<3, SO3d>;
  using EstimateType = SO3d;
  using Tangent = Vec3d;

  void setToOriginImpl() override { this->_estimate = EstimateType(); }

  void oplusImpl(const double *update) override {
    Eigen::Map<const Tangent> delta(update);
    this->_estimate = EstimateType::exp(delta) * this->estimate();
  }

  bool read(std::istream & /*input*/) override { return false; }
  bool write(std::ostream & /*output*/) const override { return false; }
};

class EdgeSO3Point : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexSO3> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit EdgeSO3Point(const Eigen::Vector3d &point) : point_(point) {}

  void computeError() override {
    const auto *vertex = static_cast<const VertexSO3 *>(_vertices[0]);
    const Vec3d prediction = vertex->estimate() * point_;

    _error = prediction - _measurement;
  }

  void linearizeOplus() override {
    const auto *vertex = static_cast<const VertexSO3 *>(_vertices[0]);
    const Vec3d q = vertex->estimate() * point_;
    _jacobianOplusXi = -SO3d::hat(q);
  }

  bool read(std::istream &) override { return false; }
  bool write(std::ostream &) const override { return false; }

private:
  Eigen::Vector3d point_;
};

inline bool SolveSo3WithG2O(const SO3d &init_R,
                            const std::vector<Vec3d> &points,
                            const std::vector<Vec3d> &observations,
                            SO3d &estimated_R) {
  if (points.size() != observations.size()) {
    std::cerr << "points and observations size mismatch\n";
    return false;
  }

  if (points.size() < 2) {
    std::cerr << "At least two non-collinear point pairs are required\n";
    return false;
  }

  g2o::SparseOptimizer optimizer;
  using BlockSolverType = g2o::BlockSolverX;
  using LinearSolverType =
      g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

  auto linear_solver = std::make_unique<LinearSolverType>();
  auto block_solver =
      std::make_unique<BlockSolverType>(std::move(linear_solver));
  auto algorithm =
      new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

  optimizer.setAlgorithm(algorithm);
  optimizer.setVerbose(false);

  auto *vertex = new VertexSO3();
  vertex->setId(0);
  vertex->setEstimate(init_R);
  if (!optimizer.addVertex(vertex)) {
    std::cerr << "Failed to add SO3 vertex\n";
    return false;
  }

  for (std::size_t i = 0; i < points.size(); ++i) {
    auto *edge = new EdgeSO3Point(points[i]);
    edge->setVertex(0, vertex);
    edge->setMeasurement(observations[i]);
    edge->setInformation(Mat3d::Identity());

    if (!optimizer.addEdge(edge)) {
      std::cerr << "Failed to add edge " << i << '\n';
      return false;
    }
  }

  if (!optimizer.initializeOptimization()) {
    std::cerr << "Failed to initialize optimization\n";
    return false;
  }

  constexpr int max_iterations = 20;
  const int performed_iterations = optimizer.optimize(max_iterations);

  if (performed_iterations <= 0) {
    std::cerr << "g2o optimization failed\n";
    return false;
  }

  estimated_R = vertex->estimate();
  return true;
}

class VertexSE3 : public g2o::BaseVertex<6, SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void setToOriginImpl() override { _estimate = SE3d(); }

  void oplusImpl(const double *update) override {
    Eigen::Map<const Vec6d> delta(update);
    _estimate = SE3d::exp(delta) * _estimate;
  }

  bool read(std::istream &) override { return false; }
  bool write(std::ostream &) const override { return false; }
};

class EdgeSE3Point : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexSE3> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit EdgeSE3Point(const Eigen::Vector3d &point) : point_(point) {}

  void computeError() override {
    const auto *vertex = static_cast<const VertexSE3 *>(_vertices[0]);
    const Vec3d prediction = vertex->estimate() * point_;

    _error = prediction - _measurement;
  }

  void linearizeOplus() override {
    const auto *vertex = static_cast<const VertexSE3 *>(_vertices[0]);
    const Vec3d q = vertex->estimate() * point_;
    _jacobianOplusXi.setZero();
    _jacobianOplusXi.template leftCols<3>().setIdentity();
    _jacobianOplusXi.template rightCols<3>() = -SO3d::hat(q);
  }

  bool read(std::istream &) override { return false; }
  bool write(std::ostream &) const override { return false; }

private:
  Eigen::Vector3d point_;
};

inline bool SolveSe3WithG2O(const SE3d &init_T,
                            const std::vector<Vec3d> &points,
                            const std::vector<Vec3d> &observations,
                            SE3d &estimated_T) {
  if (points.size() != observations.size()) {
    std::cerr << "points and observations size mismatch\n";
    return false;
  }

  if (points.size() < 2) {
    std::cerr << "At least two non-collinear point pairs are required\n";
    return false;
  }

  g2o::SparseOptimizer optimizer;
  using BlockSolverType = g2o::BlockSolverX;
  using LinearSolverType =
      g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

  auto linear_solver = std::make_unique<LinearSolverType>();
  auto block_solver =
      std::make_unique<BlockSolverType>(std::move(linear_solver));
  auto algorithm =
      new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

  optimizer.setAlgorithm(algorithm);
  optimizer.setVerbose(false);

  auto *vertex = new VertexSE3();
  vertex->setId(0);
  vertex->setEstimate(init_T);
  if (!optimizer.addVertex(vertex)) {
    std::cerr << "Failed to add SE3 vertex\n";
    return false;
  }

  for (std::size_t i = 0; i < points.size(); ++i) {
    auto *edge = new EdgeSE3Point(points[i]);
    edge->setVertex(0, vertex);
    edge->setMeasurement(observations[i]);
    edge->setInformation(Mat3d::Identity());

    if (!optimizer.addEdge(edge)) {
      std::cerr << "Failed to add edge " << i << '\n';
      return false;
    }
  }

  if (!optimizer.initializeOptimization()) {
    std::cerr << "Failed to initialize optimization\n";
    return false;
  }

  constexpr int max_iterations = 20;
  const int performed_iterations = optimizer.optimize(max_iterations);

  if (performed_iterations <= 0) {
    std::cerr << "g2o optimization failed\n";
    return false;
  }

  estimated_T = vertex->estimate();
  return true;
}

class VertexSE3Expmap : public g2o::BaseVertex<6, g2o::SE3Quat> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void setToOriginImpl() override { this->_estimate = g2o::SE3Quat(); }

  void oplusImpl(const double *update) override {
    Eigen::Map<const Vec6d> update_(update);
    _estimate = g2o::SE3Quat::exp(update_) * estimate();
  }

  bool read(std::istream & /*input*/) override { return false; }
  bool write(std::ostream & /*output*/) const override { return false; }
};

class VertexSBAPointXYZ : public g2o::BaseVertex<3, Vec3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void setToOriginImpl() override { this->_estimate = Vec3d::Zero(); }

  void oplusImpl(const double *update) override {
    Eigen::Map<const Vec3d> update_(update);
    this->_estimate += update_;
  }

  bool read(std::istream & /*input*/) override { return false; }
  bool write(std::ostream & /*output*/) const override { return false; }
};

class CameraParametersXY : public g2o::Parameter {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CameraParametersXY() = default;

  CameraParametersXY(double fx, double fy, double cx, double cy)
      : fx(fx), fy(fy), cx(cx), cy(cy) {}

  bool read(std::istream &is) override {
    return static_cast<bool>(is >> fx >> fy >> cx >> cy);
  }

  bool write(std::ostream &os) const override {
    return static_cast<bool>(os << fx << " " << fy << " " << cx << " " << cy);
  }

  double fx = 1.0;
  double fy = 1.0;
  double cx = 0.0;
  double cy = 0.0;
};
class EdgeProjectXyz2UV
    : public g2o::BaseBinaryEdge<2, Vec2d, VertexSBAPointXYZ, VertexSE3Expmap> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeProjectXyz2UV() {
    resizeParameters(1);
    installParameter(camera_, 0);
  }

  explicit EdgeProjectXyz2UV(const Vec2d &measurement) : EdgeProjectXyz2UV() {
    setMeasurement(measurement);
  }

  void computeError() override {
    const auto *point_vertex =
        static_cast<const VertexSBAPointXYZ *>(_vertices[0]);

    const auto *pose_vertex =
        static_cast<const VertexSE3Expmap *>(_vertices[1]);

    const auto *cam = static_cast<const CameraParametersXY *>(parameter(0));

    assert(cam != nullptr);

    const Vec3d pc = pose_vertex->estimate().map(point_vertex->estimate());

    const double z = pc.z();
    const double inv_z = 1.0 / z;

    const Vec2d prediction(cam->fx * pc.x() * inv_z + cam->cx,
                           cam->fy * pc.y() * inv_z + cam->cy);

    _error = _measurement - prediction;
  }

  void linearizeOplus() override {
    VertexSE3Expmap *vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
    g2o::SE3Quat Tcw(vj->estimate());
    VertexSBAPointXYZ *vi = static_cast<VertexSBAPointXYZ *>(_vertices[0]);
    Vec3d xyz = vi->estimate();
    Vec3d xyz_t = Tcw.map(xyz);
    double x = xyz_t[0];
    double y = xyz_t[1];
    double z = xyz_t[2];

    const auto *cam = static_cast<const CameraParametersXY *>(parameter(0));
    const double inv_z = 1.0 / z;
    const double inv_z2 = inv_z * inv_z;

    // 这里就是你要单独取出的两个值
    const double fx_z = cam->fx * inv_z;
    const double fy_z = cam->fy * inv_z;

    Eigen::Matrix<double, 2, 3> tmp;

    tmp(0, 0) = fx_z;
    tmp(0, 1) = 0.0;
    tmp(0, 2) = -x / z * fx_z;

    tmp(1, 0) = 0.0;
    tmp(1, 1) = fy_z;
    tmp(1, 2) = -y / z * fy_z;
    Eigen::Matrix<double, 3, 6> J_point_pose;

    J_point_pose.rightCols<3>().setIdentity();
    J_point_pose.leftCols<3>() = -SO3d::hat(xyz_t);

    _jacobianOplusXi.setZero();
    _jacobianOplusXi = -tmp * Tcw.rotation().toRotationMatrix();

    _jacobianOplusXj = -tmp * J_point_pose;
  }

  bool read(std::istream &) override { return false; }
  bool write(std::ostream &) const override { return false; }

private:
  CameraParametersXY *camera_ = nullptr;
};

struct BundleAdjustmentResult {
  bool success = false;
  int iterations = 0;
  double initial_chi2 = 0.0;
  double final_chi2 = 0.0;
  g2o::SE3Quat pose;
};

inline BundleAdjustmentResult
bundleAdjusterment(const std::vector<Vec3d> &points3d,
                   const std::vector<Vec2d> &points2d, const cv::Mat &K,
                   const cv::Mat &R, const cv::Mat &t) {
  BundleAdjustmentResult result;

  if (points3d.size() != points2d.size()) {
    std::cerr << "points3d and points2d size mismatch\n";
    return result;
  }

  if (points3d.size() < 4) {
    std::cerr << "At least four 3D-2D correspondences are required\n";
    return result;
  }

  g2o::SparseOptimizer optimizer;
  using BlockSolverType = g2o::BlockSolverX;
  using LinearSolverType =
      g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

  auto linear_solver = std::make_unique<LinearSolverType>();
  auto block_solver =
      std::make_unique<BlockSolverType>(std::move(linear_solver));
  auto algorithm =
      new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

  optimizer.setAlgorithm(algorithm);
  optimizer.setVerbose(false);

  auto *camera = new CameraParametersXY(K.at<double>(0, 0), K.at<double>(1, 1),
                                        K.at<double>(0, 2), K.at<double>(1, 2));

  camera->setId(0);
  optimizer.addParameter(camera);

  Eigen::Matrix3d R_eigen = Mat2Eigen33(R);
  Eigen::Vector3d t_eigen(t.at<double>(0), t.at<double>(1), t.at<double>(2));

  auto *pose = new VertexSE3Expmap();
  pose->setId(0);
  pose->setEstimate(g2o::SE3Quat(R_eigen, t_eigen));
  optimizer.addVertex(pose);
  int index = 1;
  for (size_t i = 0; i < points3d.size(); ++i) {
    auto *point = new VertexSBAPointXYZ();
    point->setId(index++);
    point->setEstimate(points3d[i]);
    point->setFixed(true);
    optimizer.addVertex(point);

    auto *edge = new EdgeProjectXyz2UV(points2d[i]);
    edge->setVertex(0, point);
    edge->setVertex(1, pose);
    edge->setInformation(Eigen::Matrix2d::Identity());
    edge->setParameterId(0, 0);
    optimizer.addEdge(edge);
  }

  if (!optimizer.initializeOptimization()) {
    std::cerr << "Failed to initialize bundle adjustment\n";
    return result;
  }

  optimizer.computeActiveErrors();
  result.initial_chi2 = optimizer.activeChi2();

  result.iterations = optimizer.optimize(20);
  if (result.iterations <= 0) {
    std::cerr << "bundle adjustment failed\n";
    return result;
  }

  optimizer.computeActiveErrors();
  result.final_chi2 = optimizer.activeChi2();
  result.pose = pose->estimate();
  result.success = true;

  return result;
}

// 直接法
class EdgeSE3ProjectDirect
    : public g2o::BaseUnaryEdge<1, double, VertexSE3Expmap> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectDirect(const Vec3d &point, cv::Mat *img)
      : x_world_(point), image_(img) {
    resizeParameters(1);
    installParameter(camera_, 0);
  }

  // explicit EdgeSE3ProjectDirect(const Vec2d &measurement) :
  // EdgeSE3ProjectDirect() {
  //   setMeasurement(measurement);
  // }

  void computeError() override {
    const VertexSE3Expmap *v =
        static_cast<const VertexSE3Expmap *>(_vertices[0]);
    Eigen::Vector3d x_local = v->estimate().map(x_world_);
    const auto *cam = static_cast<const CameraParametersXY *>(parameter(0));
    float x = x_local[0] * cam->fx / x_local[2] + cam->cx;
    float y = x_local[1] * cam->fy / x_local[2] + cam->cy;
    if (x - 4 < 0 || (x + 4) > image_->cols || (y - 4) < 0 ||
        (y + 4) > image_->rows) {
      _error(0, 0) = 0.0;
      this->setLevel(1);
    } else {
      _error(0, 0) = GetPixelValue(x, y) - _measurement;
    }
  }

  void linearizeOplus() override {
    if (level() == 1) {
      _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
      return;
    }

    VertexSE3Expmap *vj = static_cast<VertexSE3Expmap *>(_vertices[0]);
    g2o::SE3Quat Tcw(vj->estimate());
    Vec3d xyz_t = Tcw.map(x_world_);
    double x = xyz_t[0];
    double y = xyz_t[1];
    double z = xyz_t[2];

    const auto *cam = static_cast<const CameraParametersXY *>(parameter(0));
    const double inv_z = 1.0 / z;
    const double inv_z2 = inv_z * inv_z;

    // 这里就是你要单独取出的两个值
    const double fx_z = cam->fx * inv_z;
    const double fy_z = cam->fy * inv_z;

    float u = x * fx_z + cam->cx;
    float v = y * fy_z + cam->cy;

    Eigen::Matrix<double, 2, 3> tmp;

    tmp(0, 0) = fx_z;
    tmp(0, 1) = 0.0;
    tmp(0, 2) = -x / z * fx_z;

    tmp(1, 0) = 0.0;
    tmp(1, 1) = fy_z;
    tmp(1, 2) = -y / z * fy_z;
    Eigen::Matrix<double, 3, 6> J_point_pose;

    J_point_pose.rightCols<3>().setIdentity();
    J_point_pose.leftCols<3>() = -SO3d::hat(xyz_t);

    Eigen::Matrix<double, 1, 2> j_pixel_uv;
    j_pixel_uv(0, 0) = (GetPixelValue(u + 1, v) - GetPixelValue(u - 1, v)) / 2.;
    j_pixel_uv(0, 1) = (GetPixelValue(u, v + 1) - GetPixelValue(u, v - 1)) / 2;
    _jacobianOplusXi = j_pixel_uv * (tmp * J_point_pose);
  }

  bool read(std::istream &) override { return false; }
  bool write(std::ostream &) const override { return false; }

private:
  inline float GetPixelValue(float x, float y) {
    CV_Assert(image_->type() == CV_8UC1);

    if (x < 0.0f || y < 0.0f || x >= static_cast<float>(image_->cols - 1) ||
        y >= static_cast<float>(image_->rows - 1)) {
      return 0.0f;
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));

    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);

    const uchar *row0 = image_->ptr<uchar>(y0);
    const uchar *row1 = image_->ptr<uchar>(y0 + 1);

    const float i00 = static_cast<float>(row0[x0]);
    const float i10 = static_cast<float>(row0[x0 + 1]);
    const float i01 = static_cast<float>(row1[x0]);
    const float i11 = static_cast<float>(row1[x0 + 1]);

    return (1.0f - dx) * (1.0f - dy) * i00 + dx * (1.0f - dy) * i10 +
           (1.0f - dx) * dy * i01 + dx * dy * i11;
  }

private:
  Vec3d x_world_;
  cv::Mat *image_ = nullptr;
  CameraParametersXY *camera_ = nullptr;
};

void SolverSE3ProjectDirect(const std::vector<Vec3d> &points3d,
                            const std::vector<double> &reference_intensity,
                            cv::Mat &img, const cv::Mat &K, const cv::Mat &R,
                            const cv::Mat &t) {
  g2o::SparseOptimizer optimizer;
  using BlockSolverType = g2o::BlockSolverX;
  using LinearSolverType =
      g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

  auto linear_solver = std::make_unique<LinearSolverType>();
  auto block_solver =
      std::make_unique<BlockSolverType>(std::move(linear_solver));
  auto algorithm =
      new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

  optimizer.setAlgorithm(algorithm);
  optimizer.setVerbose(true);

  auto *camera = new CameraParametersXY(K.at<double>(0, 0), K.at<double>(1, 1),
                                        K.at<double>(0, 2), K.at<double>(1, 2));

  camera->setId(0);
  optimizer.addParameter(camera);

  Eigen::Matrix3d R_eigen = Mat2Eigen33(R);
  Eigen::Vector3d t_eigen(t.at<double>(0), t.at<double>(1), t.at<double>(2));

  auto *pose = new VertexSE3Expmap();
  pose->setId(0);
  pose->setEstimate(g2o::SE3Quat(R_eigen, t_eigen));
  optimizer.addVertex(pose);
  for (size_t i = 0; i < points3d.size(); ++i) {
    auto *edge = new EdgeSE3ProjectDirect(points3d[i], &img);
    edge->setVertex(0, pose);
    edge->setMeasurement(reference_intensity[i]);
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
    edge->setParameterId(0, 0);
    optimizer.addEdge(edge);
  }
  // auto *edge = new EdgeSE3ProjectDirect(points3d, &img);
  // edge->setVertex(0, pose);
  // edge->setMeasurement(reference_intensity);
  // edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
  // edge->setParameterId(0, 0);
  // optimizer.addEdge(edge);

  optimizer.initializeOptimization();
  optimizer.optimize(20);
}

} // namespace neves

#endif

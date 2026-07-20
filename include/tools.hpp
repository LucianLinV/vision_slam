#ifndef TOOLS_HPP
#define TOOLS_HPP

#include "vmath.hpp"

#include <ceres/autodiff_cost_function.h>
#include <ceres/problem.h>
#include <cstddef>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace neves {

using Vec3d = Vec3<double>;
using Mat3d = Mat3<double>;
using SO3d = SO3<double>;

struct So3PointResidual {
  So3PointResidual(const Vec3d &point, const Vec3d &observation)
      : point_(point), observation_(observation) {}

  template <typename T>
  bool operator()(const T *const quaternion_xyzw, T *residuals) const {
    Eigen::Map<const Quat<T>> quaternion(quaternion_xyzw);
    const SO3<T> R(quaternion);

    Eigen::Map<Vec3<T>> residual(residuals);
    residual = R * point_.template cast<T>() - observation_.template cast<T>();
    return true;
  }

  static ceres::CostFunction *Create(const Vec3d &point,
                                     const Vec3d &observation) {
    return new ceres::AutoDiffCostFunction<So3PointResidual, 3, 4>(
        new So3PointResidual(point, observation));
  }

  Vec3d point_;
  Vec3d observation_;
};

inline SO3d SolveSo3WithCeres(const SO3d &init_R,
                              const std::vector<Vec3d> &points,
                              const std::vector<Vec3d> &observations) {
  if (points.size() != observations.size()) {
    throw std::invalid_argument("points and observations size mismatch");
  }

  SO3d R = init_R;
  ceres::Problem problem;
  problem.AddParameterBlock(R.data(), SO3d::num_parameters);
  problem.SetManifold(R.data(), new ceres::EigenQuaternionManifold());

  for (std::size_t i = 0; i < points.size(); ++i) {
    problem.AddResidualBlock(
        So3PointResidual::Create(points[i], observations[i]), nullptr,
        R.data());
  }

  ceres::Solver::Options options;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.linear_solver_type = ceres::DENSE_QR;
  //options.dense_linear_algebra_library_type = ceres::CUDA;
  options.max_num_iterations = 20;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  return R;
}

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

  for (size_t i = 0; i < points.size(); ++i) {
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
} // namespace neves

#endif

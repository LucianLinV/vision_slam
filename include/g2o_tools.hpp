#ifndef G2O_TOOLS_HPP
#define G2O_TOOLS_HPP

#include "vmath.hpp"

#include <cstddef>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
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

} // namespace neves

#endif

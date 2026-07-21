#include "g2o_tools.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace {

constexpr double kEpsilon = 1e-9;

neves::Vec2d Project(const neves::CameraParametersXY &cam,
                     const g2o::SE3Quat &pose,
                     const neves::Vec3d &point) {
  const neves::Vec3d pc = pose.map(point);
  return neves::Vec2d(cam.fx * pc.x() / pc.z() + cam.cx,
                      cam.fy * pc.y() / pc.z() + cam.cy);
}

bool TestVertexSE3ExpmapOplus() {
  neves::VertexSE3Expmap vertex;
  vertex.setEstimate(g2o::SE3Quat());

  neves::Vec6d update;
  update << 0.10, -0.04, 0.03, 0.01, -0.02, 0.03;
  vertex.oplus(update.data());

  const g2o::SE3Quat expected = g2o::SE3Quat::exp(update);
  const neves::Vec3d probe(0.4, -0.2, 2.0);
  const double error = (vertex.estimate().map(probe) - expected.map(probe)).norm();

  std::cout << "SE3Expmap oplus error: " << error << '\n';
  return error < kEpsilon;
}

bool TestVertexSBAPointXYZOplus() {
  neves::VertexSBAPointXYZ vertex;
  const neves::Vec3d initial(0.5, -0.1, 3.0);
  const neves::Vec3d update(0.2, 0.3, -0.4);
  vertex.setEstimate(initial);
  vertex.oplus(update.data());

  const double error = (vertex.estimate() - (initial + update)).norm();

  std::cout << "SBAPointXYZ oplus error: " << error << '\n';
  return error < kEpsilon;
}

bool TestProjectionEdgeZeroError() {
  neves::CameraParametersXY cam(520.9, 521.0, 325.1, 249.7);
  const g2o::SE3Quat pose(Eigen::Quaterniond::Identity(),
                          neves::Vec3d(0.1, -0.05, 0.2));
  const neves::Vec3d point(0.4, -0.2, 3.0);
  const neves::Vec2d measurement = Project(cam, pose, point);

  g2o::SparseOptimizer optimizer;
  auto *camera = new neves::CameraParametersXY(cam);
  camera->setId(0);
  optimizer.addParameter(camera);

  auto *point_vertex = new neves::VertexSBAPointXYZ();
  point_vertex->setId(0);
  point_vertex->setEstimate(point);
  optimizer.addVertex(point_vertex);

  auto *pose_vertex = new neves::VertexSE3Expmap();
  pose_vertex->setId(1);
  pose_vertex->setEstimate(pose);
  pose_vertex->setFixed(true);
  optimizer.addVertex(pose_vertex);

  auto *edge = new neves::EdgeProjectXyz2UV(measurement);
  edge->setVertex(0, point_vertex);
  edge->setVertex(1, pose_vertex);
  edge->setInformation(Eigen::Matrix2d::Identity());
  edge->setParameterId(0, 0);
  optimizer.addEdge(edge);

  if (!optimizer.initializeOptimization()) {
    std::cerr << "failed to initialize projection edge test\n";
    return false;
  }

  edge->computeError();
  std::cout << "projection zero chi2: " << edge->chi2() << '\n';
  return edge->chi2() < kEpsilon;
}

bool TestPointOptimization() {
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

  const neves::CameraParametersXY cam(520.9, 521.0, 325.1, 249.7);
  auto *camera = new neves::CameraParametersXY(cam);
  camera->setId(0);
  optimizer.addParameter(camera);

  const neves::Vec3d true_point(0.35, -0.18, 3.2);
  auto *point_vertex = new neves::VertexSBAPointXYZ();
  point_vertex->setId(0);
  point_vertex->setEstimate(neves::Vec3d(0.6, -0.35, 2.6));
  optimizer.addVertex(point_vertex);

  const std::vector<neves::Vec3d> translations = {
      neves::Vec3d(0.0, 0.0, 0.0),    neves::Vec3d(0.12, 0.0, 0.02),
      neves::Vec3d(-0.10, 0.03, 0.0), neves::Vec3d(0.02, -0.10, 0.03),
      neves::Vec3d(0.08, 0.08, -0.02)};

  int next_id = 1;
  for (const auto &translation : translations) {
    g2o::SE3Quat pose(Eigen::Quaterniond::Identity(), translation);
    auto *pose_vertex = new neves::VertexSE3Expmap();
    pose_vertex->setId(next_id++);
    pose_vertex->setEstimate(pose);
    pose_vertex->setFixed(true);
    optimizer.addVertex(pose_vertex);

    auto *edge = new neves::EdgeProjectXyz2UV(Project(cam, pose, true_point));
    edge->setVertex(0, point_vertex);
    edge->setVertex(1, pose_vertex);
    edge->setInformation(Eigen::Matrix2d::Identity());
    edge->setParameterId(0, 0);
    optimizer.addEdge(edge);
  }

  if (!optimizer.initializeOptimization()) {
    std::cerr << "failed to initialize SBA point optimization\n";
    return false;
  }

  const auto start_time = std::chrono::steady_clock::now();
  const int iterations = optimizer.optimize(20);
  const auto end_time = std::chrono::steady_clock::now();
  const auto elapsed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                            start_time)
          .count();

  const double point_error = (point_vertex->estimate() - true_point).norm();

  std::cout << "optimizer iterations: " << iterations << '\n';
  std::cout << "sba solve time:       " << elapsed_us << " us ("
            << static_cast<double>(elapsed_us) / 1000.0 << " ms)\n";
  std::cout << "true point:           " << true_point.transpose() << '\n';
  std::cout << "estimated point:      " << point_vertex->estimate().transpose()
            << '\n';
  std::cout << "point error:          " << point_error << '\n';

  return iterations > 0 && point_error < 1e-6;
}

} // namespace

int main() {
  std::cout << std::setprecision(12);

  const bool ok = TestVertexSE3ExpmapOplus() && TestVertexSBAPointXYZOplus() &&
                  TestProjectionEdgeZeroError() && TestPointOptimization();

  return ok ? 0 : 1;
}

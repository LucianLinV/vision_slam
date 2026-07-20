#ifndef CERES_TOOLS_HPP
#define CERES_TOOLS_HPP

#include "vmath.hpp"

#include <ceres/autodiff_cost_function.h>
#include <ceres/problem.h>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace neves {

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
  // options.dense_linear_algebra_library_type = ceres::CUDA;
  options.max_num_iterations = 20;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  return R;
}

struct Se3PointResidual {
  Se3PointResidual(const Vec3d &point, const Vec3d &observation)
      : point_(point), observation_(observation) {}

  template <typename T>
  bool operator()(const T *const quaternion_xyzw, const T *const trans_xyz,
                  T *residuals) const {
    Eigen::Map<const Quat<T>> quaternion(quaternion_xyzw);
    const SO3<T> R(quaternion);

    Eigen::Map<const Vec3<T>> translation(trans_xyz);

    Eigen::Map<Vec3<T>> residual(residuals);
    residual = (R * point_.template cast<T>() + translation) -
               observation_.template cast<T>();
    return true;
  }

  static ceres::CostFunction *Create(const Vec3d &point,
                                     const Vec3d &observation) {
    return new ceres::AutoDiffCostFunction<Se3PointResidual, 3, 4, 3>(
        new Se3PointResidual(point, observation));
  }

  Vec3d point_;
  Vec3d observation_;
};

inline SE3d SolveSe3WithCeres(const SE3d &init_T,
                              const std::vector<Vec3d> &points,
                              const std::vector<Vec3d> &observations) {
  if (points.size() != observations.size()) {
    throw std::invalid_argument("points and observations size mismatch");
  }

  if (points.size() < 3) {
    throw std::invalid_argument(
        "At least three non-collinear point pairs are required");
  }

  SO3d R = init_T.so3();
  Vec3d translation = init_T.translation();

  ceres::Problem problem;
  problem.AddParameterBlock(R.data(), SO3d::num_parameters);
  problem.SetManifold(R.data(), new ceres::EigenQuaternionManifold());
  problem.AddParameterBlock(translation.data(), 3);

  for (std::size_t i = 0; i < points.size(); ++i) {
    problem.AddResidualBlock(
        Se3PointResidual::Create(points[i], observations[i]), nullptr, R.data(),
        translation.data());
  }

  ceres::Solver::Options options;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 20;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  if (!summary.IsSolutionUsable()) {
    throw std::runtime_error("Ceres failed: " + summary.BriefReport());
  }

  return SE3d(R, translation);
}

} // namespace neves

#endif

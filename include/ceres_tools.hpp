#ifndef CERES_TOOLS_HPP
#define CERES_TOOLS_HPP

#include "vmath.hpp"

#include <array>
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

struct BAPointResidual {
  BAPointResidual(const Vec3d &point, const Vec2d &observation,
                  const std::array<double, 9> &cam)
      : point_(point), observation_(observation), fx_(cam.at(0)),
        fy_(cam.at(4)), cx_(cam.at(2)), cy_(cam.at(5)) {}

  template <typename T>
  bool operator()(const T *const quaternion_xyzw, const T *const trans_xyz,
                  T *residuals) const {
    Eigen::Map<const Quat<T>> quaternion(quaternion_xyzw);
    const SO3<T> R(quaternion);

    Eigen::Map<const Vec3<T>> translation(trans_xyz);

    const Vec3<T> point_camera = R * point_.template cast<T>() + translation;

    const T X = point_camera.x();
    const T Y = point_camera.y();
    const T Z = point_camera.z();

    const T predicted_u = T(fx_) * X / Z + T(cx_);

    const T predicted_v = T(fy_) * Y / Z + T(cy_);

    // 预测像素 - 实际观测像素
    residuals[0] = predicted_u - T(observation_.x());

    residuals[1] = predicted_v - T(observation_.y());
    return true;
  }

  static ceres::CostFunction *Create(const Vec3d &point,
                                     const Vec2d &observation,
                                     const std::array<double, 9> &cam) {
    return new ceres::AutoDiffCostFunction<BAPointResidual, 2, 4, 3>(
        new BAPointResidual(point, observation, cam));
  }

  Vec3d point_;
  Vec2d observation_;
  double fx_;
  double fy_;
  double cx_;
  double cy_;
};

inline SE3d SolveBA32(const SE3d &init_T, const std::vector<Vec3d> &points,
                      const std::vector<Vec2d> &observations,
                      const std::array<double, 9> &cam) {
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
        BAPointResidual::Create(points[i], observations[i], cam), nullptr,
        R.data(), translation.data());
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

struct CeresSE3ProjectDirectResidual {
  CeresSE3ProjectDirectResidual(const Eigen::Vector3d &point,
                                double reference_intensity,
                                const cv::Mat &image,
                                const std::array<double, 9> &camera)
      : point_(point), reference_intensity_(reference_intensity), image_(image),
        fx_(camera.at(0)), fy_(camera.at(4)), cx_(camera.at(2)),
        cy_(camera.at(5)) {
    CV_Assert(!image_.empty());
    CV_Assert(image_.type() == CV_8UC1);
  }

  template <typename T>
  bool operator()(const T *const quaternion_xyzw, const T *const trans_xyz,
                  T *residuals) const {
    Eigen::Map<const Quat<T>> quaternion(quaternion_xyzw);
    const SO3<T> R(quaternion);

    Eigen::Map<const Vec3<T>> translation(trans_xyz);

    const Vec3<T> point_camera = R * point_.template cast<T>() + translation;

    const T X = point_camera.x();
    const T Y = point_camera.y();
    const T Z = point_camera.z();

    const double z_scalar = ScalarPart(Z);

    // 位于相机后方或者深度太小
    if (!std::isfinite(z_scalar) || z_scalar <= 1e-8) {
      return false;
    }

    const T predicted_u = T(fx_) * X / Z + T(cx_);

    const T predicted_v = T(fy_) * Y / Z + T(cy_);

    const double u_scalar = ScalarPart(predicted_u);

    const double v_scalar = ScalarPart(predicted_v);

    /*
     * 双线性插值会访问：
     * (x0, y0)
     * (x0+1, y0)
     * (x0, y0+1)
     * (x0+1, y0+1)
     */
    if (!std::isfinite(u_scalar) || !std::isfinite(v_scalar) ||
        u_scalar < 0.0 || v_scalar < 0.0 || u_scalar >= image_.cols - 1.0 ||
        v_scalar >= image_.rows - 1.0) {
      return false;
    }

    const T current_intensity = GetPixelValue(predicted_u, predicted_v);

    residuals[0] = current_intensity - T(reference_intensity_);

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d &point,
                                     double reference_intensity,
                                     const cv::Mat &image,
                                     const std::array<double, 9> &camera) {
    return new ceres::AutoDiffCostFunction<CeresSE3ProjectDirectResidual,
                                           1, // 一个光度残差
                                           4, // 四元数 [x,y,z,w]
                                           3  // 平移 [tx,ty,tz]
                                           >(
        new CeresSE3ProjectDirectResidual(point, reference_intensity, image,
                                          camera));
  }

  template <typename T> static double ScalarPart(const T &value) {
    return static_cast<double>(value);
  }

  /*
   * ceres::Jet 的标量部分保存在 a 中。
   * 这里只用标量部分确定四个相邻像素的位置；
   * 后续插值权重仍使用完整的 Jet，因此导数得以保留。
   */
  template <typename Scalar, int N>
  static double ScalarPart(const ceres::Jet<Scalar, N> &value) {
    return static_cast<double>(value.a);
  }

  template <typename T> T GetPixelValue(const T &x, const T &y) const {
    const double x_scalar = ScalarPart(x);
    const double y_scalar = ScalarPart(y);

    const int x0 = static_cast<int>(std::floor(x_scalar));

    const int y0 = static_cast<int>(std::floor(y_scalar));

    // dx、dy 保持为 T，保留 Jet 导数
    const T dx = x - T(x0);
    const T dy = y - T(y0);

    const uchar *row0 = image_.ptr<uchar>(y0);
    const uchar *row1 = image_.ptr<uchar>(y0 + 1);

    const T i00 = T(row0[x0]);
    const T i10 = T(row0[x0 + 1]);
    const T i01 = T(row1[x0]);
    const T i11 = T(row1[x0 + 1]);

    return (T(1.0) - dx) * (T(1.0) - dy) * i00 + dx * (T(1.0) - dy) * i10 +
           (T(1.0) - dx) * dy * i01 + dx * dy * i11;
  }

  cv::Mat image_;
  double reference_intensity_ = 0.0;
  Vec3d point_;
  double fx_;
  double fy_;
  double cx_;
  double cy_;
};

inline SE3d SolveBA32(const SE3d &init_T, const std::vector<Vec3d> &points3d,
                      const std::vector<double> &reference_intensities,
                      const cv::Mat &current_image,
                      const std::array<double, 9> &cam) {
  if (points3d.size() != reference_intensities.size()) {
    throw std::invalid_argument("points and observations size mismatch");
  }

  if (points3d.size() < 3) {
    throw std::invalid_argument(
        "At least three non-collinear point pairs are required");
  }

  SO3d R = init_T.so3();
  Vec3d translation = init_T.translation();

  ceres::Problem problem;
  problem.AddParameterBlock(R.data(), SO3d::num_parameters);
  problem.SetManifold(R.data(), new ceres::EigenQuaternionManifold());
  problem.AddParameterBlock(translation.data(), 3);

  for (std::size_t i = 0; i < points3d.size(); ++i) {
    problem.AddResidualBlock(
        CeresSE3ProjectDirectResidual::Create(points3d[i],
                                              reference_intensities[i],
                                              current_image, cam),
        nullptr, R.data(), translation.data());
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

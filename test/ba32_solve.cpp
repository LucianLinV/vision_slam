#include "ceres_tools.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

neves::Vec2d Project(const std::array<double, 9> &K, const neves::SE3d &T,
                     const neves::Vec3d &point) {
  const neves::Vec3d pc = T * point;
  return neves::Vec2d(K[0] * pc.x() / pc.z() + K[2],
                      K[4] * pc.y() / pc.z() + K[5]);
}

double PixelRmsError(const std::array<double, 9> &K, const neves::SE3d &T,
                     const std::vector<neves::Vec3d> &points,
                     const std::vector<neves::Vec2d> &observations) {
  double error = 0.0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    error += (Project(K, T, points[i]) - observations[i]).squaredNorm();
  }
  return std::sqrt(error / static_cast<double>(points.size()));
}

} // namespace

int main() {
  std::cout << std::setprecision(12);

  const std::array<double, 9> K = {
      520.0, 0.0, 320.0,
      0.0, 518.0, 240.0,
      0.0, 0.0, 1.0,
  };

  std::vector<neves::Vec3d> points;
  constexpr int point_count = 80;
  points.reserve(point_count);
  for (int i = 0; i < point_count; ++i) {
    const double k = static_cast<double>(i);
    points.emplace_back(1.2 * std::sin(0.31 * k),
                        0.9 * std::cos(0.23 * k),
                        4.0 + 0.04 * k + 0.2 * std::sin(0.17 * k));
  }

  const neves::Vec3d true_phi(0.08, -0.05, 0.12);
  const neves::Vec3d true_t(0.35, -0.18, 0.42);
  const neves::SE3d true_T(neves::SO3d::exp(true_phi), true_t);

  std::vector<neves::Vec2d> observations;
  observations.reserve(points.size());
  double observation_noise_rms = 0.0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const double k = static_cast<double>(i);
    const neves::Vec2d noise(1.5 * std::sin(0.43 * k),
                             1.2 * std::cos(0.37 * k));
    observation_noise_rms += noise.squaredNorm();
    observations.push_back(Project(K, true_T, points[i]) + noise);
  }
  observation_noise_rms =
      std::sqrt(observation_noise_rms / static_cast<double>(observations.size()));

  const neves::SE3d init_T(
      neves::SO3d::exp(neves::Vec3d(-0.05, 0.03, -0.02)),
      neves::Vec3d(0.10, 0.10, 0.20));

  const double initial_pixel_rms = PixelRmsError(K, init_T, points, observations);
  const neves::SE3d estimated_T =
      neves::SolveBA32(init_T, points, observations, K);
  const double final_pixel_rms =
      PixelRmsError(K, estimated_T, points, observations);

  const neves::SE3d init_error_T = init_T.inverse() * true_T;
  const neves::SE3d final_error_T = estimated_T.inverse() * true_T;
  const double initial_rot_error = init_error_T.so3().log().norm();
  const double final_rot_error = final_error_T.so3().log().norm();
  const double initial_trans_error = init_error_T.translation().norm();
  const double final_trans_error = final_error_T.translation().norm();

  std::cout << "point count:            " << points.size() << '\n';
  std::cout << "observation noise rms:  " << observation_noise_rms << " px\n";
  std::cout << "initial pixel rms:      " << initial_pixel_rms << " px\n";
  std::cout << "final pixel rms:        " << final_pixel_rms << " px\n";
  std::cout << "initial rot error:      " << initial_rot_error << " rad\n";
  std::cout << "final rot error:        " << final_rot_error << " rad\n";
  std::cout << "initial trans error:    " << initial_trans_error << '\n';
  std::cout << "final trans error:      " << final_trans_error << '\n';
  std::cout << "estimated so3:          "
            << estimated_T.so3().log().transpose() << '\n';
  std::cout << "estimated t:            "
            << estimated_T.translation().transpose() << '\n';

  const bool fit_improved = final_pixel_rms < initial_pixel_rms * 0.05;
  const bool pose_improved = final_rot_error < initial_rot_error * 0.2 &&
                             final_trans_error < initial_trans_error * 0.2;
  const bool fit_matches_noise = final_pixel_rms < observation_noise_rms * 1.3;

  return fit_improved && pose_improved && fit_matches_noise ? 0 : 1;
}

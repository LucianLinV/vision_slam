#include "tools/g2o_tools.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

cv::Mat EigenRotationToCvMat(const Eigen::Matrix3d &R) {
  cv::Mat cv_R(3, 3, CV_64F);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      cv_R.at<double>(r, c) = R(r, c);
    }
  }
  return cv_R;
}

cv::Mat EigenTranslationToCvMat(const neves::Vec3d &t) {
  cv::Mat cv_t(3, 1, CV_64F);
  cv_t.at<double>(0) = t.x();
  cv_t.at<double>(1) = t.y();
  cv_t.at<double>(2) = t.z();
  return cv_t;
}

neves::Vec2d Project(const cv::Mat &K, const neves::SE3d &pose,
                     const neves::Vec3d &point) {
  const neves::Vec3d pc = pose * point;
  return neves::Vec2d(K.at<double>(0, 0) * pc.x() / pc.z() +
                          K.at<double>(0, 2),
                      K.at<double>(1, 1) * pc.y() / pc.z() +
                          K.at<double>(1, 2));
}

double PixelRmsError(const cv::Mat &K, const neves::SE3d &pose,
                     const std::vector<neves::Vec3d> &points3d,
                     const std::vector<neves::Vec2d> &points2d) {
  double error = 0.0;
  for (std::size_t i = 0; i < points3d.size(); ++i) {
    error += (Project(K, pose, points3d[i]) - points2d[i]).squaredNorm();
  }
  return std::sqrt(error / static_cast<double>(points3d.size()));
}

double RotationErrorRad(const neves::SE3d &estimated,
                        const neves::SE3d &truth) {
  const Eigen::Matrix3d R_error =
      (estimated.inverse() * truth).rotationMatrix();
  return Eigen::AngleAxisd(R_error).angle();
}

} // namespace

int main() {
  std::cout << std::setprecision(12);

  const cv::Mat K = (cv::Mat_<double>(3, 3) << 520.0, 0.0, 320.0, 0.0, 518.0,
                     240.0, 0.0, 0.0, 1.0);

  std::vector<neves::Vec3d> points3d;
  constexpr int point_count = 80;
  points3d.reserve(point_count);
  for (int i = 0; i < point_count; ++i) {
    const double k = static_cast<double>(i);
    points3d.emplace_back(1.2 * std::sin(0.31 * k),
                          0.9 * std::cos(0.23 * k),
                          4.0 + 0.04 * k + 0.2 * std::sin(0.17 * k));
  }

  const neves::SE3d true_pose(
      neves::SO3d::exp(neves::Vec3d(0.08, -0.05, 0.12)),
      neves::Vec3d(0.35, -0.18, 0.42));

  std::vector<neves::Vec2d> points2d;
  points2d.reserve(points3d.size());
  double observation_noise_rms = 0.0;
  for (std::size_t i = 0; i < points3d.size(); ++i) {
    const double k = static_cast<double>(i);
    const neves::Vec2d noise(2.0 * std::sin(0.43 * k),
                             1.6 * std::cos(0.37 * k));
    observation_noise_rms += noise.squaredNorm();
    points2d.push_back(Project(K, true_pose, points3d[i]) + noise);
  }
  observation_noise_rms =
      std::sqrt(observation_noise_rms / static_cast<double>(points2d.size()));

  const neves::SE3d init_pose(
      neves::SO3d::exp(neves::Vec3d(-0.05, 0.03, -0.02)),
      neves::Vec3d(0.10, 0.10, 0.20));

  const cv::Mat init_R = EigenRotationToCvMat(init_pose.rotationMatrix());
  const cv::Mat init_t = EigenTranslationToCvMat(init_pose.translation());

  const double initial_pixel_rms =
      PixelRmsError(K, init_pose, points3d, points2d);

  const neves::SE3d estimated_T =
      neves::bundleAdjusterment(points3d, points2d, K, init_R, init_t);

  const double final_pixel_rms =
      PixelRmsError(K, estimated_T, points3d, points2d);
  const double init_rot_error = RotationErrorRad(init_pose, true_pose);
  const double final_rot_error = RotationErrorRad(estimated_T, true_pose);
  const double init_trans_error =
      (init_pose.translation() - true_pose.translation()).norm();
  const double final_trans_error =
      (estimated_T.translation() - true_pose.translation()).norm();

  std::cout << "point count:            " << points3d.size() << '\n';
  std::cout << "observation noise rms:  " << observation_noise_rms << " px\n";
  std::cout << "initial pixel rms:      " << initial_pixel_rms << " px\n";
  std::cout << "final pixel rms:        " << final_pixel_rms << " px\n";
  std::cout << "initial rot error:      " << init_rot_error << " rad\n";
  std::cout << "final rot error:        " << final_rot_error << " rad\n";
  std::cout << "initial trans error:    " << init_trans_error << '\n';
  std::cout << "final trans error:      " << final_trans_error << '\n';

  const bool pose_improved = final_rot_error < init_rot_error * 0.2 &&
                             final_trans_error < init_trans_error * 0.2;
  const bool fit_matches_noise = final_pixel_rms < observation_noise_rms * 1.3;

  return pose_improved && fit_matches_noise ? 0 : 1;
}

#include "tools.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Se3Stats {
  std::int64_t min_us = std::numeric_limits<std::int64_t>::max();
  std::int64_t max_us = 0;
  std::int64_t total_us = 0;
  double rotation_error = 0.0;
  double translation_error = 0.0;
  double rms_fit_error = 0.0;
  neves::SE3d estimated_T;
};

double ComputeSe3RmsError(const neves::SE3d &T,
                          const std::vector<neves::Vec3d> &points,
                          const std::vector<neves::Vec3d> &observations) {
  double rms_error = 0.0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    rms_error += std::pow((T * points[i] - observations[i]).norm(), 2);
  }
  return std::sqrt(rms_error / static_cast<double>(points.size()));
}

void FillSe3Errors(Se3Stats &stats, const neves::SE3d &true_T,
                   const std::vector<neves::Vec3d> &points,
                   const std::vector<neves::Vec3d> &observations) {
  const neves::SE3d error_T = stats.estimated_T.inverse() * true_T;
  stats.rotation_error = error_T.so3().log().norm();
  stats.translation_error = error_T.translation().norm();
  stats.rms_fit_error = ComputeSe3RmsError(stats.estimated_T, points, observations);
}

void PrintSe3Stats(const std::string &name, const Se3Stats &stats,
                   int repeat_count) {
  const double avg_us = static_cast<double>(stats.total_us) / repeat_count;

  std::cout << name << " estimated so3: "
            << stats.estimated_T.so3().log().transpose() << '\n';
  std::cout << name << " estimated t:   "
            << stats.estimated_T.translation().transpose() << '\n';
  std::cout << name << " solve min:     " << stats.min_us << " us ("
            << static_cast<double>(stats.min_us) / 1000.0 << " ms)\n";
  std::cout << name << " solve avg:     " << avg_us << " us ("
            << avg_us / 1000.0 << " ms)\n";
  std::cout << name << " solve max:     " << stats.max_us << " us ("
            << static_cast<double>(stats.max_us) / 1000.0 << " ms)\n";
  std::cout << name << " rot error:     " << stats.rotation_error << " rad\n";
  std::cout << name << " trans error:   " << stats.translation_error << '\n';
  std::cout << name << " rms fit error: " << stats.rms_fit_error << '\n';
}

int main() {
  std::cout << std::setprecision(12);

  std::vector<neves::Vec3d> points;
  constexpr int point_count = 1000;
  points.reserve(point_count);
  for (int i = 0; i < point_count; ++i) {
    const double x = std::sin(double(i) * 0.37) * 3.0;
    const double y = std::cos(double(i) * 0.19) * 2.0;
    const double z = 0.5 + std::sin(double(i) * 0.11);
    points.emplace_back(x, y, z);
  }

  const neves::Vec3d true_phi(0.2, -0.35, 0.15);
  const neves::Vec3d true_t(0.8, -0.4, 1.2);
  const neves::SE3d true_T(neves::SO3d::exp(true_phi), true_t);

  std::vector<neves::Vec3d> observations;
  observations.reserve(points.size());
  double noise_rms = 0.0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const double k = static_cast<double>(i);
    const neves::Vec3d noise(0.08 * std::sin(0.13 * k),
                             0.06 * std::cos(0.17 * k),
                             0.10 * std::sin(0.07 * k + 0.3));
    noise_rms += noise.squaredNorm();
    observations.push_back(true_T * points[i] + noise);
  }
  noise_rms = std::sqrt(noise_rms / static_cast<double>(points.size()));

  const neves::SE3d init_T(neves::SO3d::exp(neves::Vec3d::Zero()),
                           neves::Vec3d::Zero());

  constexpr int repeat_count = 200;
  Se3Stats ceres_stats;
  Se3Stats g2o_stats;

  for (int i = 0; i < repeat_count; ++i) {
    const auto start_time = std::chrono::steady_clock::now();
    ceres_stats.estimated_T =
        neves::SolveSe3WithCeres(init_T, points, observations);
    const auto end_time = std::chrono::steady_clock::now();
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                              start_time)
            .count();

    ceres_stats.min_us = std::min<std::int64_t>(ceres_stats.min_us, elapsed_us);
    ceres_stats.max_us = std::max<std::int64_t>(ceres_stats.max_us, elapsed_us);
    ceres_stats.total_us += elapsed_us;
  }

  for (int i = 0; i < repeat_count; ++i) {
    neves::SE3d estimated_T;
    const auto start_time = std::chrono::steady_clock::now();
    const bool success =
        neves::SolveSe3WithG2O(init_T, points, observations, estimated_T);
    const auto end_time = std::chrono::steady_clock::now();
    if (!success) {
      return 1;
    }
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                              start_time)
            .count();

    g2o_stats.estimated_T = estimated_T;
    g2o_stats.min_us = std::min<std::int64_t>(g2o_stats.min_us, elapsed_us);
    g2o_stats.max_us = std::max<std::int64_t>(g2o_stats.max_us, elapsed_us);
    g2o_stats.total_us += elapsed_us;
  }

  FillSe3Errors(ceres_stats, true_T, points, observations);
  FillSe3Errors(g2o_stats, true_T, points, observations);

  std::cout << "true so3:      " << true_phi.transpose() << '\n';
  std::cout << "true t:        " << true_t.transpose() << '\n';
  std::cout << "point count:   " << points.size() << '\n';
  std::cout << "repeat count:  " << repeat_count << '\n';
  std::cout << "noise rms:     " << noise_rms << '\n';
  PrintSe3Stats("ceres", ceres_stats, repeat_count);
  PrintSe3Stats("g2o", g2o_stats, repeat_count);

  return ceres_stats.rotation_error < 1e-2 &&
                 ceres_stats.translation_error < 1e-2 &&
                 g2o_stats.rotation_error < 1e-2 &&
                 g2o_stats.translation_error < 1e-2
             ? 0
             : 1;
}

#include "tools.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct SolveStats {
  std::int64_t min_us = std::numeric_limits<std::int64_t>::max();
  std::int64_t max_us = 0;
  std::int64_t total_us = 0;
  double angle_error = 0.0;
  double rms_error = 0.0;
  neves::SO3d estimated_R;
};

double ComputeRmsError(const neves::SO3d &R,
                       const std::vector<neves::Vec3d> &points,
                       const std::vector<neves::Vec3d> &observations) {
  double rms_error = 0.0;
  for (size_t i = 0; i < points.size(); ++i) {
    rms_error += std::pow((R * points[i] - observations[i]).norm(), 2);
  }
  return std::sqrt(rms_error / static_cast<double>(points.size()));
}

void PrintStats(const std::string &name, const SolveStats &stats,
                int repeat_count) {
  const double avg_us = static_cast<double>(stats.total_us) / repeat_count;

  std::cout << name << " estimated so3: " << stats.estimated_R.log().transpose()
            << '\n';
  std::cout << name << " solve min:     " << stats.min_us << " us ("
            << static_cast<double>(stats.min_us) / 1000.0 << " ms)\n";
  std::cout << name << " solve avg:     " << avg_us << " us ("
            << avg_us / 1000.0 << " ms)\n";
  std::cout << name << " solve max:     " << stats.max_us << " us ("
            << static_cast<double>(stats.max_us) / 1000.0 << " ms)\n";
  std::cout << name << " angle error:   " << stats.angle_error << " rad\n";
  std::cout << name << " rms error:     " << stats.rms_error << '\n';
}

int main() {
  std::vector<neves::Vec3d> points;
  constexpr int point_count = 1000000;
  points.reserve(point_count);
  for (int i = 0; i < point_count; ++i) {
    const double x = std::sin(double(i) * 0.37) * 3.0;
    const double y = std::cos(double(i) * 0.19) * 2.0;
    const double z = 0.5 + std::sin(double(i) * 0.11);
    points.emplace_back(x, y, z);
  }

  const neves::Vec3d true_phi(0.2, -0.35, 0.15);
  const neves::SO3d true_R = neves::SO3d::exp(true_phi);

  std::vector<neves::Vec3d> observations;
  observations.reserve(points.size());
  for (const auto &point : points) {
    observations.push_back(true_R * point);
  }

  const neves::SO3d init_R = neves::SO3d::exp(neves::Vec3d::Zero());

  constexpr int repeat_count = 200;
  SolveStats ceres_stats;
  SolveStats g2o_stats;

  for (int i = 0; i < repeat_count; ++i) {
    const auto start_time = std::chrono::steady_clock::now();
    ceres_stats.estimated_R =
        neves::SolveSo3WithCeres(init_R, points, observations);
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
    neves::SO3d estimated_R;
    const auto start_time = std::chrono::steady_clock::now();
    const bool success =
        neves::SolveSo3WithG2O(init_R, points, observations, estimated_R);
    const auto end_time = std::chrono::steady_clock::now();
    if (!success) {
      return 1;
    }
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                              start_time)
            .count();

    g2o_stats.estimated_R = estimated_R;
    g2o_stats.min_us = std::min<std::int64_t>(g2o_stats.min_us, elapsed_us);
    g2o_stats.max_us = std::max<std::int64_t>(g2o_stats.max_us, elapsed_us);
    g2o_stats.total_us += elapsed_us;
  }

  ceres_stats.angle_error =
      (ceres_stats.estimated_R.inverse() * true_R).log().norm();
  ceres_stats.rms_error =
      ComputeRmsError(ceres_stats.estimated_R, points, observations);
  g2o_stats.angle_error = (g2o_stats.estimated_R.inverse() * true_R).log().norm();
  g2o_stats.rms_error = ComputeRmsError(g2o_stats.estimated_R, points, observations);

  std::cout << "true so3:      " << true_phi.transpose() << '\n';
  std::cout << "point count:   " << points.size() << '\n';
  std::cout << "repeat count:  " << repeat_count << '\n';
  PrintStats("ceres", ceres_stats, repeat_count);
  PrintStats("g2o", g2o_stats, repeat_count);

  return ceres_stats.angle_error < 1e-8 && ceres_stats.rms_error < 1e-8 &&
                 g2o_stats.angle_error < 1e-8 && g2o_stats.rms_error < 1e-8
             ? 0
             : 1;
}

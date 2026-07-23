#include "ceres_tools.hpp"
#include "g2o_tools.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

struct TumFrame {
  std::filesystem::path rgb_path;
  std::filesystem::path depth_path;
};

struct SparsePoint {
  neves::Vec3d xyz_ref;
  cv::Point2f px_ref;
  double intensity = 0.0;
};

struct G2ODirectResult {
  g2o::SE3Quat pose;
  int iterations = 0;
  std::int64_t elapsed_us = 0;
};

struct CeresDirectResult {
  neves::SE3d pose;
  std::int64_t elapsed_us = 0;
};

std::filesystem::path ProjectRoot() {
  return std::filesystem::path(__FILE__).parent_path().parent_path();
}

std::filesystem::path TumRoot() { return ProjectRoot() / "sources" / "TUM"; }

cv::Mat MakeTumCameraMatrix() {
  return (cv::Mat_<double>(3, 3) << 520.9, 0.0, 325.1, 0.0, 521.0, 249.7,
          0.0, 0.0, 1.0);
}

std::array<double, 9> CameraArray(const cv::Mat &K) {
  return {K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
          K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
          K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2)};
}

void ConfigureOptimizer(g2o::SparseOptimizer &optimizer) {
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
}

bool LoadTumFrames(std::vector<TumFrame> &frames) {
  const auto root = TumRoot();
  std::ifstream fin(root / "associate.txt");
  if (!fin) {
    std::cerr << "failed to open TUM associate file\n";
    return false;
  }

  std::string line;
  while (frames.size() < 4 && std::getline(fin, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream iss(line);
    std::string rgb_time;
    std::string rgb_rel;
    std::string depth_time;
    std::string depth_rel;
    if (!(iss >> rgb_time >> rgb_rel >> depth_time >> depth_rel)) {
      continue;
    }

    frames.push_back({root / rgb_rel, root / depth_rel});
  }

  if (frames.size() != 4) {
    std::cerr << "expected 4 TUM frames, got " << frames.size() << '\n';
    return false;
  }

  return true;
}

float GetPixelValue(const cv::Mat &image, float x, float y) {
  if (x < 0.0f || y < 0.0f || x >= static_cast<float>(image.cols - 1) ||
      y >= static_cast<float>(image.rows - 1)) {
    return 0.0f;
  }

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const float dx = x - static_cast<float>(x0);
  const float dy = y - static_cast<float>(y0);

  const uchar *row0 = image.ptr<uchar>(y0);
  const uchar *row1 = image.ptr<uchar>(y0 + 1);

  const float i00 = static_cast<float>(row0[x0]);
  const float i10 = static_cast<float>(row0[x0 + 1]);
  const float i01 = static_cast<float>(row1[x0]);
  const float i11 = static_cast<float>(row1[x0 + 1]);

  return (1.0f - dx) * (1.0f - dy) * i00 + dx * (1.0f - dy) * i10 +
         (1.0f - dx) * dy * i01 + dx * dy * i11;
}

std::vector<SparsePoint> BuildSparsePoints(const cv::Mat &ref_gray,
                                           const cv::Mat &ref_depth,
                                           const cv::Mat &K) {
  std::vector<cv::KeyPoint> keypoints;
  cv::FAST(ref_gray, keypoints, 10, true);

  const double fx = K.at<double>(0, 0);
  const double fy = K.at<double>(1, 1);
  const double cx = K.at<double>(0, 2);
  const double cy = K.at<double>(1, 2);

  std::vector<SparsePoint> points;
  points.reserve(keypoints.size());

  for (const auto &kp : keypoints) {
    const int u = cvRound(kp.pt.x);
    const int v = cvRound(kp.pt.y);
    if (u < 4 || v < 4 || u + 4 >= ref_gray.cols || v + 4 >= ref_gray.rows) {
      continue;
    }

    const ushort depth_raw = ref_depth.ptr<ushort>(v)[u];
    if (depth_raw == 0) {
      continue;
    }

    const double z = static_cast<double>(depth_raw) / 5000.0;
    if (z <= 0.1 || z > 8.0) {
      continue;
    }

    SparsePoint point;
    point.xyz_ref =
        neves::Vec3d((kp.pt.x - cx) * z / fx, (kp.pt.y - cy) * z / fy, z);
    point.px_ref = kp.pt;
    point.intensity = GetPixelValue(ref_gray, kp.pt.x, kp.pt.y);
    points.push_back(point);
  }

  return points;
}

std::vector<neves::Vec3d> ExtractPoints3d(
    const std::vector<SparsePoint> &points) {
  std::vector<neves::Vec3d> points3d;
  points3d.reserve(points.size());
  for (const auto &point : points) {
    points3d.push_back(point.xyz_ref);
  }

  return points3d;
}

std::vector<double> ExtractIntensities(const std::vector<SparsePoint> &points) {
  std::vector<double> intensities;
  intensities.reserve(points.size());
  for (const auto &point : points) {
    intensities.push_back(point.intensity);
  }

  return intensities;
}

bool EstimateDirectPoseG2O(const std::vector<SparsePoint> &points,
                           cv::Mat &target_gray, const cv::Mat &K,
                           G2ODirectResult &result) {
  g2o::SparseOptimizer optimizer;
  ConfigureOptimizer(optimizer);

  auto *camera = new neves::CameraParametersXY(K.at<double>(0, 0),
                                               K.at<double>(1, 1),
                                               K.at<double>(0, 2),
                                               K.at<double>(1, 2));
  camera->setId(0);
  optimizer.addParameter(camera);

  auto *pose = new neves::VertexSE3Expmap();
  pose->setId(0);
  pose->setEstimate(g2o::SE3Quat());
  optimizer.addVertex(pose);

  int edge_count = 0;
  for (const auto &point : points) {
    auto *edge = new neves::EdgeSE3ProjectDirect(point.xyz_ref, &target_gray);
    edge->setVertex(0, pose);
    edge->setMeasurement(point.intensity);
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
    edge->setParameterId(0, 0);
    optimizer.addEdge(edge);
    ++edge_count;
  }

  if (edge_count < 20) {
    std::cerr << "not enough direct edges: " << edge_count << '\n';
    return false;
  }

  if (!optimizer.initializeOptimization()) {
    std::cerr << "failed to initialize direct pose optimization\n";
    return false;
  }

  const auto start_time = std::chrono::steady_clock::now();
  result.iterations = optimizer.optimize(20);
  const auto end_time = std::chrono::steady_clock::now();
  result.elapsed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                            start_time)
          .count();

  result.pose = pose->estimate();
  return result.iterations > 0;
}

bool EstimateDirectPoseCeres(const std::vector<SparsePoint> &points,
                             const cv::Mat &target_gray, const cv::Mat &K,
                             CeresDirectResult &result) {
  if (points.size() < 20) {
    std::cerr << "not enough direct residuals for Ceres: " << points.size()
              << '\n';
    return false;
  }

  const auto points3d = ExtractPoints3d(points);
  const auto intensities = ExtractIntensities(points);
  const auto camera = CameraArray(K);
  const neves::SE3d init_T;

  try {
    const auto start_time = std::chrono::steady_clock::now();
    result.pose = neves::SolveBA32(init_T, points3d, intensities, target_gray,
                                   camera);
    const auto end_time = std::chrono::steady_clock::now();
    result.elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                              start_time)
            .count();
  } catch (const std::exception &e) {
    std::cerr << "Ceres direct solve failed: " << e.what() << '\n';
    return false;
  }

  return true;
}

template <typename ProjectFn>
void DrawProjection(const cv::Mat &target_color,
                    const std::vector<SparsePoint> &points, const cv::Mat &K,
                    const std::filesystem::path &output_path,
                    ProjectFn project) {
  cv::Mat output = target_color.clone();
  const double fx = K.at<double>(0, 0);
  const double fy = K.at<double>(1, 1);
  const double cx = K.at<double>(0, 2);
  const double cy = K.at<double>(1, 2);

  int drawn = 0;
  for (const auto &point : points) {
    const neves::Vec3d pc = project(point.xyz_ref);
    if (pc.z() <= 0.0) {
      continue;
    }

    const double u = fx * pc.x() / pc.z() + cx;
    const double v = fy * pc.y() / pc.z() + cy;
    if (u < 0.0 || v < 0.0 || u >= output.cols || v >= output.rows) {
      continue;
    }

    cv::circle(output, cv::Point2f(static_cast<float>(u), static_cast<float>(v)),
               2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    ++drawn;
  }

  if (!cv::imwrite(output_path.string(), output)) {
    std::cerr << "failed to write projection image: " << output_path << '\n';
  }

  std::cout << "drawn projections: " << drawn << '\n';
  std::cout << "projection image:  " << output_path << '\n';
}

void DrawG2OProjection(const cv::Mat &target_color,
                       const std::vector<SparsePoint> &points,
                       const g2o::SE3Quat &pose, const cv::Mat &K,
                       const std::filesystem::path &output_path) {
  DrawProjection(target_color, points, K, output_path,
                 [&pose](const neves::Vec3d &point) { return pose.map(point); });
}

void DrawCeresProjection(const cv::Mat &target_color,
                         const std::vector<SparsePoint> &points,
                         const neves::SE3d &pose, const cv::Mat &K,
                         const std::filesystem::path &output_path) {
  DrawProjection(target_color, points, K, output_path,
                 [&pose](const neves::Vec3d &point) { return pose * point; });
}

bool RunTumSparseDirect() {
  std::vector<TumFrame> frames;
  if (!LoadTumFrames(frames)) {
    return false;
  }

  const cv::Mat K = MakeTumCameraMatrix();
  bool ok = true;

  for (std::size_t i = 0; i + 1 < frames.size(); ++i) {
    cv::Mat ref_color = cv::imread(frames[i].rgb_path.string(), cv::IMREAD_COLOR);
    cv::Mat target_color =
        cv::imread(frames[i + 1].rgb_path.string(), cv::IMREAD_COLOR);
    cv::Mat ref_depth =
        cv::imread(frames[i].depth_path.string(), cv::IMREAD_UNCHANGED);

    if (ref_color.empty() || target_color.empty() || ref_depth.empty()) {
      std::cerr << "failed to load frame pair " << i + 1 << " -> " << i + 2
                << '\n';
      ok = false;
      continue;
    }

    cv::Mat ref_gray;
    cv::Mat target_gray;
    cv::cvtColor(ref_color, ref_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(target_color, target_gray, cv::COLOR_BGR2GRAY);

    const auto points = BuildSparsePoints(ref_gray, ref_depth, K);
    std::cout << "pair " << i + 1 << " -> " << i + 2 << '\n';
    std::cout << "sparse points: " << points.size() << '\n';

    G2ODirectResult g2o_result;
    CeresDirectResult ceres_result;
    const bool g2o_success =
        EstimateDirectPoseG2O(points, target_gray, K, g2o_result);
    const bool ceres_success =
        EstimateDirectPoseCeres(points, target_gray, K, ceres_result);

    if (!g2o_success || !ceres_success) {
      ok = false;
      continue;
    }

    std::cout << "g2o iterations: " << g2o_result.iterations << '\n';
    std::cout << "g2o solve time: " << g2o_result.elapsed_us << " us ("
              << static_cast<double>(g2o_result.elapsed_us) / 1000.0
              << " ms)\n";
    std::cout << "g2o estimated pose:\n"
              << g2o_result.pose.to_homogeneous_matrix() << '\n';

    std::cout << "ceres solve time: " << ceres_result.elapsed_us << " us ("
              << static_cast<double>(ceres_result.elapsed_us) / 1000.0
              << " ms)\n";
    std::cout << "ceres estimated pose:\n" << ceres_result.pose.matrix() << '\n';

    const auto g2o_output_path =
        TumRoot() / ("direct_g2o_" + std::to_string(i + 1) + "_" +
                     std::to_string(i + 2) + ".png");
    const auto ceres_output_path =
        TumRoot() / ("direct_ceres_" + std::to_string(i + 1) + "_" +
                     std::to_string(i + 2) + ".png");
    DrawG2OProjection(target_color, points, g2o_result.pose, K, g2o_output_path);
    DrawCeresProjection(target_color, points, ceres_result.pose, K,
                        ceres_output_path);
  }

  return ok;
}

} // namespace

int main() {
  std::cout << std::setprecision(12);
  return RunTumSparseDirect() ? 0 : 1;
}

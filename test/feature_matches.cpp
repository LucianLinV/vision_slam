#include "cv_tools.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

namespace neves{

std::filesystem::path ProjectRoot() {
  return std::filesystem::path(__FILE__).parent_path().parent_path();
}

struct FeatureMatchStats {
  std::int64_t min_us = std::numeric_limits<std::int64_t>::max();
  std::int64_t max_us = 0;
  std::int64_t total_us = 0;
};

void PrintFeatureMatchStats(const FeatureMatchStats &stats, int repeat_count) {
  const double avg_us = static_cast<double>(stats.total_us) / repeat_count;

  std::cout << "match min:     " << stats.min_us << " us ("
            << static_cast<double>(stats.min_us) / 1000.0 << " ms)\n";
  std::cout << "match avg:     " << avg_us << " us (" << avg_us / 1000.0
            << " ms)\n";
  std::cout << "match max:     " << stats.max_us << " us ("
            << static_cast<double>(stats.max_us) / 1000.0 << " ms)\n";
}

} // namespace

int main() {
  using namespace neves;
  std::cout << std::setprecision(12);

  const auto source_dir = ProjectRoot() / "sources";
  const auto img1_path = source_dir / "1.png";
  const auto img2_path = source_dir / "2.png";
  const auto output_path = source_dir / "feature_matches.png";

  const cv::Mat img1 = cv::imread(img1_path.string(), cv::IMREAD_GRAYSCALE);
  const cv::Mat img2 = cv::imread(img2_path.string(), cv::IMREAD_GRAYSCALE);

  if (img1.empty() || img2.empty()) {
    std::cerr << "failed to load test images:\n"
              << "  " << img1_path << '\n'
              << "  " << img2_path << '\n';
    return 1;
  }

  std::vector<cv::KeyPoint> kp1;
  std::vector<cv::KeyPoint> kp2;
  std::vector<cv::DMatch> good_matches;

  constexpr int repeat_count = 200;
  FeatureMatchStats stats;

  for (int i = 0; i < repeat_count; ++i) {
    kp1.clear();
    kp2.clear();
    good_matches.clear();

    const auto start_time = std::chrono::steady_clock::now();
    FindFeatureMaches(img1, img2, kp1, kp2, good_matches);
    const auto end_time = std::chrono::steady_clock::now();
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                              start_time)
            .count();

    stats.min_us = std::min<std::int64_t>(stats.min_us, elapsed_us);
    stats.max_us = std::max<std::int64_t>(stats.max_us, elapsed_us);
    stats.total_us += elapsed_us;
  }

  std::cout << "image1 keypoints: " << kp1.size() << '\n';
  std::cout << "image2 keypoints: " << kp2.size() << '\n';
  std::cout << "good matches:     " << good_matches.size() << '\n';
  std::cout << "repeat count:     " << repeat_count << '\n';
  PrintFeatureMatchStats(stats, repeat_count);

  if (kp1.empty() || kp2.empty()) {
    std::cerr << "ORB did not find enough keypoints\n";
    return 1;
  }

  if (good_matches.empty()) {
    std::cerr << "FindFeatureMaches returned no matches\n";
    return 1;
  }

  cv::Mat match_image;
  cv::drawMatches(img1, kp1, img2, kp2, good_matches, match_image);

  if (!cv::imwrite(output_path.string(), match_image)) {
    std::cerr << "failed to write match image: " << output_path << '\n';
    return 1;
  }

  std::cout << "match image:      " << output_path << '\n';

  return 0;
}

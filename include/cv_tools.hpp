#pragma once

#include <algorithm>
#include <vector>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
namespace neves {



inline void FindOrb(const cv::Mat &img1, const cv::Mat &img2,
                    std::vector<cv::KeyPoint> &kp1,
                    std::vector<cv::KeyPoint> &kp2, cv::Mat &description1,
                    cv::Mat &description2) {
  cv::Ptr<cv::ORB> orb =
      cv::ORB::create(1000, 1.1f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
  orb->detect(img1, kp1);
  orb->detect(img2, kp2);
  orb->compute(img1, kp1, description1);
  orb->compute(img2, kp2, description2);
}

inline void FindFeatureMaches(const cv::Mat &img1, const cv::Mat &img2,
                              std::vector<cv::KeyPoint> &kp1,
                              std::vector<cv::KeyPoint> &kp2,
                              std::vector<cv::DMatch> &g_matches) {
  cv::Mat descriptors_1, descriptors_2;
  FindOrb(img1, img2, kp1, kp2, descriptors_1, descriptors_2);
  g_matches.clear();

  if (descriptors_1.empty() || descriptors_2.empty()) {
    return;
  }

  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector<cv::DMatch> matches;
  matcher.match(descriptors_1, descriptors_2, matches);
  if (matches.empty()) {
    return;
  }

  double min_dist = 10000, max_dist = 0;
  for (const auto &match : matches) {
    double dist = match.distance;
    if (dist < min_dist)
      min_dist = dist;
    if (dist > max_dist)
      max_dist = dist;
  }
  std::vector<cv::DMatch> d_matches;
  for (const auto &match : matches) {
    if (match.distance <= std::max(2* min_dist, 30.0)) {
      d_matches.push_back(match);
    }
  }
  if (d_matches.size() < 8) {
    return;
  }

  std::vector<cv::Point2f> pts1;
  std::vector<cv::Point2f> pts2;

  for (const auto &match : d_matches) {
    pts1.push_back(kp1[match.queryIdx].pt);
    pts2.push_back(kp2[match.trainIdx].pt);
  }

  std::vector<uchar> inlier_mask;

  cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC,
                                     3.0,  // ransacReprojThreshold，像素阈值
                                     0.99, // confidence
                                     inlier_mask);
  if (F.empty() || inlier_mask.size() != d_matches.size()) {
    return;
  }

  for (size_t i = 0; i < d_matches.size(); ++i) {
    if (inlier_mask[i]) {
      g_matches.push_back(d_matches[i]);
    }
  }
}

void PoseEstimation2d2d(std::vector<cv::KeyPoint> keypoints_1,
                        std::vector<cv::KeyPoint> keypoints_2,
                        std::vector<cv::DMatch> matches, cv::Mat &R,
                        cv::Mat &T) {
  cv::Mat K =
      (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  cv::Point2d principal_point(325.1, 249.7);
  int focal_length = 521;
  std::vector<cv::Point2f> points1;
  std::vector<cv::Point2f> points2;

  for (int i = 0; i < matches.size(); i++) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

  cv::Mat fundamental_matrix;
  fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
  std::cout << "fundamental_matrix is " << std::endl
            << fundamental_matrix << std::endl;

  cv::Mat essential_matrix;
  essential_matrix = cv::findEssentialMat(points1, points2, focal_length,
                                      principal_point, cv::RANSAC);
  std::cout << "essential_matrix is " << std::endl
            << essential_matrix << std::endl;

  //-- 计算单应矩阵
  cv::Mat homography_matrix;
  homography_matrix =
      cv::findHomography(points1, points2, cv::RANSAC, 3, cv::noArray(), 2000, 0.99);
  std::cout << "homography_matrix is " << std::endl
            << homography_matrix << std::endl;

  //-- 从本质矩阵中恢复旋转和平移信息.
  cv::recoverPose(essential_matrix, points1, points2, R, T, focal_length,
                  principal_point);
  std::cout << "R is " << std::endl << R << std::endl;
  std::cout << "t is " << std::endl << T << std::endl;
}

}
#ifndef VISUAL_ODOMETRY_HPP
#define VISUAL_ODOMETRY_HPP

#include "Frame.hpp"
#include "config.hpp"
#include "map.hpp"
#include "tools.hpp"

#include <algorithm>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace neves {

class VisualOdemetry {
public:
  typedef std::shared_ptr<VisualOdemetry> Ptr;

  enum class VState { INITIALIZING = -1, OK = 0, Lost = 1 };

  VState state_;
  Map::Ptr map_;
  Frame::Ptr ref_frame_;
  Frame::Ptr curr_frame_;

  cv::Ptr<cv::ORB> orb_;
  std::vector<cv::KeyPoint> keypoints_curr_;
  cv::Mat descriptors_curr_;

  std::vector<MapPoint::Ptr> match_3dpts_;
  std::vector<int> match_2dkp_index_;

  SE3d T_c_w_estimated_;
  int num_inliers_;
  int num_lost_;

  int num_of_features_;
  double scale_factor_;
  int level_pyramid_;
  float match_ratio_;
  int max_num_lost_;
  int min_inliers_;

  double key_frame_min_rot;
  double key_frame_min_trans;

public:
  VisualOdemetry()
      : state_(VState::INITIALIZING), map_(new Map), ref_frame_(nullptr),
        curr_frame_(nullptr), num_inliers_(0), num_lost_(0) {
    num_of_features_ = Config::get<int>("tracking.num_of_features");
    scale_factor_ = Config::get<double>("tracking.scale_factor");
    level_pyramid_ = Config::get<int>("tracking.level_pyramid");
    match_ratio_ = Config::get<float>("tracking.match_ratio");
    max_num_lost_ = Config::get<int>("tracking.max_num_lost");
    min_inliers_ = Config::get<int>("tracking.min_inliers");
    key_frame_min_rot = Config::get<double>("keyframe.min_rot");
    key_frame_min_trans = Config::get<double>("keyframe.min_trans");
    orb_ = cv::ORB::create(num_of_features_, scale_factor_, level_pyramid_);
  }

  ~VisualOdemetry() = default;

  bool addFrame(Frame::Ptr frame) {
    switch (state_) {
    case VState::INITIALIZING: {
      state_ = VState::OK;
      curr_frame_ = ref_frame_ = frame;
      extractKeyPoints();
      computeDescriptors();
      addKeyFrame();
      break;
    }
    case VState::OK: {
      curr_frame_ = frame;
      curr_frame_->T_c_w_ = ref_frame_->T_c_w_;

      extractKeyPoints();
      computeDescriptors();
      featureMatching();
      poseEstimationPnP();

      if (checkEstimatePose()) {
        curr_frame_->T_c_w_ = T_c_w_estimated_;
        num_lost_ = 0;

        if (checkKeyFrame()) {
          addKeyFrame();
        }
      } else {
        ++num_lost_;
        if (num_lost_ > max_num_lost_) {
          state_ = VState::Lost;
        }
        return false;
      }
      break;
    }
    case VState::Lost: {
      std::cout << "vo has lost." << std::endl;
      break;
    }
    }
    return true;
  }

protected:
  void extractKeyPoints() {
    orb_->detect(curr_frame_->color_, keypoints_curr_);
  }

  void computeDescriptors() {
    orb_->compute(curr_frame_->color_, keypoints_curr_, descriptors_curr_);
  }

  void featureMatching() {
    match_3dpts_.clear();
    match_2dkp_index_.clear();

    std::vector<MapPoint::Ptr> candidates;
    cv::Mat descriptors_map;
    for (auto &map_point_pair : map_->map_points_) {
      MapPoint::Ptr &map_point = map_point_pair.second;
      if (!map_point || map_point->descriptor_.empty()) {
        continue;
      }

      if (curr_frame_->isInFrame(map_point->pos_w_)) {
        ++map_point->visible_times_;
        candidates.push_back(map_point);
        descriptors_map.push_back(map_point->descriptor_);
      }
    }

    if (candidates.empty() || descriptors_map.empty() ||
        descriptors_curr_.empty()) {
      std::cout << "good matches: 0" << std::endl;
      return;
    }

    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptors_map, descriptors_curr_, matches);

    if (matches.empty()) {
      std::cout << "good matches: 0" << std::endl;
      return;
    }

    const auto min_match =
        std::min_element(matches.begin(), matches.end(),
                         [](const cv::DMatch &m1, const cv::DMatch &m2) {
                           return m1.distance < m2.distance;
                         });
    const float min_dis = min_match->distance;
    const float threshold = std::max<float>(min_dis * match_ratio_, 30.0f);

    for (const cv::DMatch &match : matches) {
      if (match.queryIdx < 0 ||
          match.queryIdx >= static_cast<int>(candidates.size()) ||
          match.trainIdx < 0 ||
          match.trainIdx >= static_cast<int>(keypoints_curr_.size())) {
        continue;
      }

      if (match.distance < threshold) {
        MapPoint::Ptr map_point = candidates[match.queryIdx];
        ++map_point->matched_times_;
        match_3dpts_.push_back(map_point);
        match_2dkp_index_.push_back(match.trainIdx);
      }
    }

    std::cout << "good matches: " << match_3dpts_.size() << std::endl;
  }

  void poseEstimationPnP() {
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;

    pts3d.reserve(match_3dpts_.size());
    pts2d.reserve(match_2dkp_index_.size());

    for (size_t i = 0; i < match_3dpts_.size(); ++i) {
      const int kp_index = match_2dkp_index_[i];
      if (kp_index < 0 || kp_index >= static_cast<int>(keypoints_curr_.size())) {
        continue;
      }

      pts3d.push_back(match_3dpts_[i]->getPositionCV());
      pts2d.push_back(keypoints_curr_[kp_index].pt);
    }

    if (pts3d.size() < 4) {
      num_inliers_ = 0;
      return;
    }

    cv::Mat K = (cv::Mat_<double>(3, 3) << curr_frame_->camera_->fx(), 0,
                 curr_frame_->camera_->cx(), 0, curr_frame_->camera_->fy(),
                 curr_frame_->camera_->cy(), 0, 0, 1);

    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat inliers;
    const bool pnp_success =
        cv::solvePnPRansac(pts3d, pts2d, K, cv::Mat(), rvec, tvec, false, 100,
                           4.0, 0.99, inliers);

    num_inliers_ = static_cast<int>(inliers.total());
    std::cout << "pnp inliers: " << num_inliers_ << std::endl;

    if (!pnp_success || inliers.empty()) {
      num_inliers_ = 0;
      return;
    }

    const Vec3d r(rvec.at<double>(0, 0), rvec.at<double>(1, 0),
                  rvec.at<double>(2, 0));
    const Vec3d t(tvec.at<double>(0, 0), tvec.at<double>(1, 0),
                  tvec.at<double>(2, 0));
    T_c_w_estimated_ = SE3d(SO3d::exp(r), t);

    std::vector<Vec3d> ba_pts3d;
    std::vector<Vec2d> ba_pts2d;
    ba_pts3d.reserve(inliers.total());
    ba_pts2d.reserve(inliers.total());

    for (int i = 0; i < inliers.rows; ++i) {
      const int idx = inliers.at<int>(i, 0);
      ba_pts3d.emplace_back(pts3d[idx].x, pts3d[idx].y, pts3d[idx].z);
      ba_pts2d.emplace_back(pts2d[idx].x, pts2d[idx].y);
    }

    T_c_w_estimated_ =
        bundleAdjusterment(ba_pts3d, ba_pts2d, K, T_c_w_estimated_);
  }

  void addKeyFrame() {
    std::cout << "adding a key-frame" << std::endl;

    for (size_t i = 0; i < keypoints_curr_.size(); ++i) {
      const double depth = curr_frame_->findDepth(keypoints_curr_[i]);
      if (depth <= 0.0) {
        continue;
      }

      const Vec3d point_world = curr_frame_->camera_->Pixel2World(
          Vec2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), depth,
          curr_frame_->T_c_w_);
      Vec3d norm = point_world - curr_frame_->getCamCenter();
      norm.normalize();

      MapPoint::Ptr map_point = MapPoint::createMapPoint(
          point_world, norm, descriptors_curr_.row(static_cast<int>(i)).clone(),
          curr_frame_.get());
      map_->insertMapPoint(map_point);
    }

    map_->insertKeyFrame(curr_frame_);
    ref_frame_ = curr_frame_;
  }

  bool checkEstimatePose() {
    if (num_inliers_ < min_inliers_) {
      std::cout << "reject because inlier is too small: " << num_inliers_
                << std::endl;
      return false;
    }

    const SE3d T_c_r = T_c_w_estimated_ * ref_frame_->T_c_w_.inverse();
    const Vec6d d = T_c_r.log();
    if (d.norm() > 5.0) {
      std::cout << "reject because motion is too large: " << d.norm()
                << std::endl;
      return false;
    }

    return true;
  }

  bool checkKeyFrame() {
    const SE3d T_c_r = T_c_w_estimated_ * ref_frame_->T_c_w_.inverse();
    const Vec6d d = T_c_r.log();
    const Vec3d trans = d.head<3>();
    const Vec3d rot = d.tail<3>();
    return rot.norm() > key_frame_min_rot ||
           trans.norm() > key_frame_min_trans;
  }
};

} // namespace neves

#endif

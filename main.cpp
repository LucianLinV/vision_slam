#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz.hpp>

#include "Camera.hpp"
#include "config.hpp"
#include "visual_odometry.hpp"

namespace {

cv::Affine3d SophusToAffine3d(const neves::SE3d &pose) {
  const Eigen::Matrix3d R = pose.rotationMatrix();
  const Eigen::Vector3d t = pose.translation();

  return cv::Affine3d(
      cv::Affine3d::Mat3(R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1),
                         R(1, 2), R(2, 0), R(2, 1), R(2, 2)),
      cv::Affine3d::Vec3(t.x(), t.y(), t.z()));
}

cv::Mat DrawTrackingImage(const cv::Mat &color,
                          const neves::VisualOdemetry &vo) {
  cv::Mat display;
  cv::drawKeypoints(color, vo.keypoints_curr_, display, cv::Scalar(0, 180, 255),
                    cv::DrawMatchesFlags::DEFAULT);

  for (const int keypoint_index : vo.match_2dkp_index_) {
    if (keypoint_index < 0 ||
        keypoint_index >= static_cast<int>(vo.keypoints_curr_.size())) {
      continue;
    }

    const cv::Point2f &pt = vo.keypoints_curr_[keypoint_index].pt;
    cv::circle(display, pt, 3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
  }

  cv::putText(display, "features: " + std::to_string(vo.keypoints_curr_.size()),
              cv::Point(15, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
  cv::putText(display,
              "matches: " + std::to_string(vo.match_2dkp_index_.size()) +
                  "  inliers: " + std::to_string(vo.num_inliers_),
              cv::Point(15, 55), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

  return display;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argc > 4) {
    std::cout << "usage: run_vo parameter_file [--no-viz] [--keep-window]"
              << std::endl;
    return 1;
  }

  bool show_viz = true;
  bool keep_window = false;
  for (int i = 2; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--no-viz") {
      show_viz = false;
    } else if (option == "--keep-window") {
      keep_window = true;
    } else {
      std::cout << "unknown option: " << option << std::endl;
      std::cout << "usage: run_vo parameter_file [--no-viz] [--keep-window]"
                << std::endl;
      return 1;
    }
  }

  Config::setParameterFile(argv[1]);

  const std::filesystem::path dataset_dir =
      Config::get<std::string>("dataset.root");
  const std::filesystem::path association_file =
      Config::get<std::string>("dataset.association_file");

  std::cout << "dataset: " << dataset_dir.string() << std::endl;

  std::ifstream fin(association_file);
  if (!fin) {
    std::cout << "please generate the associate file called associate.txt!"
              << std::endl;
    return 1;
  }

  std::vector<std::string> rgb_files;
  std::vector<std::string> depth_files;
  std::vector<double> rgb_times;
  std::vector<double> depth_times;

  std::string rgb_time;
  std::string rgb_file;
  std::string depth_time;
  std::string depth_file;
  while (fin >> rgb_time >> rgb_file >> depth_time >> depth_file) {
    rgb_times.push_back(std::atof(rgb_time.c_str()));
    depth_times.push_back(std::atof(depth_time.c_str()));
    rgb_files.push_back((dataset_dir / rgb_file).string());
    depth_files.push_back((dataset_dir / depth_file).string());
  }

  neves::Camera::Ptr camera(new neves::Camera(
      Config::get<double>("camera.fx"), Config::get<double>("camera.fy"),
      Config::get<double>("camera.cx"), Config::get<double>("camera.cy"),
      Config::get<double>("camera.depth_scale")));

  neves::VisualOdemetry::Ptr vo(new neves::VisualOdemetry);

  std::unique_ptr<cv::viz::Viz3d> vis;
  std::vector<cv::Point3d> trajectory;
  if (show_viz) {
    vis = std::make_unique<cv::viz::Viz3d>("Visual Odometry");
    cv::viz::WCoordinateSystem world_coor(1.0);
    cv::viz::WCoordinateSystem camera_coor(0.5);
    const cv::Point3d cam_pos(0, -1.0, -1.0);
    const cv::Point3d cam_focal_point(0, 0, 0);
    const cv::Point3d cam_y_dir(0, 1, 0);
    const cv::Affine3d viewer_pose =
        cv::viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
    vis->setViewerPose(viewer_pose);

    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    vis->showWidget("World", world_coor);
    vis->showWidget("Camera", camera_coor);
  }

  std::cout << "read total " << rgb_files.size() << " entries" << std::endl;
  for (size_t i = 0; i < rgb_files.size(); ++i) {
    cv::Mat color = cv::imread(rgb_files[i], cv::IMREAD_COLOR);
    cv::Mat depth = cv::imread(depth_files[i], cv::IMREAD_UNCHANGED);
    if (color.empty() || depth.empty()) {
      break;
    }

    neves::Frame::Ptr frame = neves::Frame::createFrame();
    frame->camera_ = camera;
    frame->color_ = color;
    frame->depth_ = depth;
    frame->time_stamp_ = rgb_times[i];

    const auto start = std::chrono::steady_clock::now();
    vo->addFrame(frame);
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    std::cout << "VO costs time: " << elapsed.count() << std::endl;

    if (vo->state_ == neves::VisualOdemetry::VState::Lost) {
      break;
    }

    if (show_viz) {
      const neves::SE3d T_w_c = frame->T_c_w_.inverse();
      const cv::Affine3d camera_pose = SophusToAffine3d(T_w_c);
      vis->setWidgetPose("Camera", camera_pose);

      const Eigen::Vector3d camera_center = T_w_c.translation();
      trajectory.emplace_back(camera_center.x(), camera_center.y(),
                              camera_center.z());
      if (trajectory.size() >= 2) {
        cv::viz::WPolyLine trajectory_widget(trajectory,
                                             cv::viz::Color::green());
        trajectory_widget.setRenderingProperty(cv::viz::LINE_WIDTH, 3.0);
        vis->showWidget("Trajectory", trajectory_widget);
      }

      cv::imshow("image", DrawTrackingImage(color, *vo));
      cv::waitKey(1);
      vis->spinOnce(10, true);
    }
  }

  if (show_viz && keep_window && vis) {
    if (!trajectory.empty()) {
      const cv::Point3d end = trajectory.back();
      cv::viz::WSphere end_point(end, 0.02, 16, cv::viz::Color::red());
      vis->showWidget("TrajectoryEnd", end_point);
    }
    std::cout << "VO finished. Close the viz window to exit." << std::endl;
    vis->spin();
  }

  return 0;
}

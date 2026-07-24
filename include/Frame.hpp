#ifndef FRAME_HPP
#define FRAME_HPP

#include "Camera.hpp"

#include <memory>
#include <utility>

namespace neves {

class Frame {

public:
  using Ptr = std::shared_ptr<Frame>;

  Frame() = default;

  explicit Frame(long id, double time_stamp, SE3d T_c_w, Camera::Ptr camera,
                 cv::Mat color, cv::Mat depth)
      : id_(id), time_stamp_(time_stamp), T_c_w_(T_c_w), camera_(camera),
        color_(std::move(color)), depth_(std::move(depth)) {}

  ~Frame() = default;

  static Ptr createFrame() {
    static long factory_id = 0;
    auto frame = std::make_shared<Frame>();
    frame->id_ = factory_id++;
    return frame;
  }

  double findDepth(const cv::KeyPoint &kp) {
    if (depth_.empty() || depth_.type() != CV_16UC1 || !camera_) {
      return -1.0;
    }

    int x = cvRound(kp.pt.x);
    int y = cvRound(kp.pt.y);

    if (x < 0 || y < 0 || x >= depth_.cols || y >= depth_.rows) {
      return -1.0;
    }

    ushort d = depth_.ptr<ushort>(y)[x];
    if (d != 0) {
      return camera_->depthRawToMeters(static_cast<double>(d));
    } else {
      int dx[4] = {1, 0, 1, 0};
      int dy[4] = {0, -1, 0, 1};
      for (int i = 0; i < 4; i++) {
        const int xx = x + dx[i];
        const int yy = y + dy[i];
        if (xx < 0 || yy < 0 || xx >= depth_.cols || yy >= depth_.rows) {
          continue;
        }
        d = depth_.ptr<ushort>(yy)[xx];
        if (d != 0) {
          return camera_->depthRawToMeters(static_cast<double>(d));
        }
      }
    }
    return -1.0;
  }

  Vec3d getCamCenter() const { return T_c_w_.inverse().translation(); }

  bool isInFrame(const Vec3d &pt_world) {
    if (color_.empty() || !camera_) {
      return false;
    }

    Vec3d p_cam = camera_->World2Camera(pt_world, T_c_w_);
    if (p_cam.z() <= 0.0) {
      return false;
    }

    Vec2d pixel = camera_->World2Pixel(pt_world, T_c_w_);
    return pixel.x() >= 0.0 && pixel.y() >= 0.0 &&
           pixel.x() < static_cast<double>(color_.cols) &&
           pixel.y() < static_cast<double>(color_.rows);
  }

public:
  unsigned long id_ = 0;
  double time_stamp_ = 0.0;
  SE3d T_c_w_; // transform from world to camera
  Camera::Ptr camera_;
  cv::Mat color_, depth_;
};



} // namespace neves

#endif

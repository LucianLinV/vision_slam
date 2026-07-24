#ifndef MAPPOINT_HPP
#define MAPPOINT_HPP

#include "Frame.hpp"
#include "tools.hpp"
#include <list>
namespace neves {

class MapPoint {
public:
  typedef std::shared_ptr<MapPoint> Ptr;
  unsigned long id_;
  inline static long factory_id_ = 0;
  bool good_;
  Vec3d pos_w_;
  Vec3d norm_;
  cv::Mat descriptor_;
  int observed_times_;
  int correct_times_;
  std::list<Frame *> observed_frames_;
  int matched_times_; // being an inliner in pose estimation
  int visible_times_; // being visible in current frame

  MapPoint()
      : id_(0), good_(true), pos_w_(Vec3d(0, 0, 0)),
        norm_(Vec3d(0, 0, 0)), observed_times_(0), correct_times_(0),
        matched_times_(0), visible_times_(0) {}
  MapPoint(unsigned long id, const Vec3d &position, const Vec3d &norm,
           Frame *frame = nullptr, const cv::Mat &descriptor = cv::Mat())
      : id_(id), good_(true), pos_w_(position), norm_(norm),
        descriptor_(descriptor.clone()), observed_times_(0), correct_times_(0),
        observed_frames_(), matched_times_(0), visible_times_(0) {
    if (frame != nullptr) {
      observed_frames_.push_back(frame);
    }
  }

  inline cv::Point3f getPositionCV() const {
    return cv::Point3f(pos_w_(0, 0), pos_w_(1, 0), pos_w_(2, 0));
  }

  static MapPoint::Ptr createMapPoint() {

    return MapPoint::Ptr(
        new MapPoint(factory_id_++, Vec3d(0, 0, 0), Vec3d(0, 0, 0)));
  }

  static MapPoint::Ptr createMapPoint(const Vec3d &pos_world, const Vec3d &norm,
                                      const cv::Mat &descriptor, Frame *frame) {
    return MapPoint::Ptr(
        new MapPoint(factory_id_++, pos_world, norm, frame, descriptor));
  }
};

} // namespace neves

#endif

#ifndef MAP_HPP
#define MAP_HPP
#include "Frame.hpp"
#include "mappoint.hpp"
#include <unordered_map>
namespace neves {
class Map {
public:
  typedef std::shared_ptr<Map> Ptr;
  std::unordered_map<unsigned long, MapPoint::Ptr> map_points_;
  std::unordered_map<unsigned long, Frame::Ptr> keyframes_;

  Map() {}

  void insertKeyFrame(Frame::Ptr frame) {
    keyframes_[frame->id_] = frame;
  }

  void insertMapPoint(MapPoint::Ptr map_point) {
    map_points_[map_point->id_] = map_point;
  }
};
} // namespace neves

#endif

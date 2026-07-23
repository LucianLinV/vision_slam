#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <array>
#include <memory>

#include <opencv2/core.hpp>

#include "vmath.hpp"

namespace neves {

class Camera {
public:
    using Ptr = std::shared_ptr<Camera>;

    Camera()
        : K(cv::Mat::eye(3, 3, CV_64F)), depth_scale_(0.0) {}

    explicit Camera(double fx, double fy, double cx, double cy,
                    double depth_scale = 0.0)
        : K((cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0,
             1.0)),
          depth_scale_(depth_scale) {}

    Camera(const std::array<double, 9> &camera_matrix,
           double depth_scale = 0.0)
        : K((cv::Mat_<double>(3, 3) << camera_matrix[0], camera_matrix[1],
             camera_matrix[2], camera_matrix[3], camera_matrix[4],
             camera_matrix[5], camera_matrix[6], camera_matrix[7],
             camera_matrix[8])),
          depth_scale_(depth_scale) {}

    double fx() const { return K.at<double>(0, 0); }
    double fy() const { return K.at<double>(1, 1); }
    double cx() const { return K.at<double>(0, 2); }
    double cy() const { return K.at<double>(1, 2); }

    double depthScale() const { return depth_scale_; }

    Vec3d pixel2camera(const Vec2d &pixel, double depth) const {
        return Vec3d((pixel.x() - cx()) * depth / fx(),
                     (pixel.y() - cy()) * depth / fy(),
                     depth);
    }

    Vec2d camera2pixel(const Vec3d &point_camera) const {
        return Vec2d(fx() * point_camera.x() / point_camera.z() + cx(),
                     fy() * point_camera.y() / point_camera.z() + cy());
    }

    Vec3d camera2world(const Vec3d &point_camera, const SE3d &Tcw) const {
        return Tcw.inverse() * point_camera;
    }

    Vec3d world2camera(const Vec3d &point_world, const SE3d &Tcw) const {
        return Tcw * point_world;
    }

    Vec3d pixel2world(const Vec2d &pixel, double depth, const SE3d &Tcw) const {
        return camera2world(pixel2camera(pixel, depth), Tcw);
    }

    Vec2d world2pixel(const Vec3d &point_world, const SE3d &Tcw) const {
        return camera2pixel(world2camera(point_world, Tcw));
    }

    double depthRawToMeters(double depth_raw) const {
        if (depth_scale_ == 0.0) {
            return depth_raw;
        }
        return depth_raw / depth_scale_;
    }

    double depthMetersToRaw(double depth_meters) const {
        if (depth_scale_ == 0.0) {
            return depth_meters;
        }
        return depth_meters * depth_scale_;
    }

public:
    cv::Mat K;
    double depth_scale_;
};

} // namespace neves

namespace neves {
using Camera = neves::Camera;
}

#endif

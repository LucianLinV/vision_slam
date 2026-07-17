#ifndef VMATH_HPP
#define VMATH_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

float rad2deg(float rad);
float deg2rad(float deg);

Eigen::Matrix3f Skew(const Eigen::Vector3f &v);

cv::Point3f pixel2camera(const cv::Point3f &pixel_p, const cv::Mat &K, float z);
cv::Point3f camera2pixel(cv::Point3f &camera_p, const cv::Mat &K,
                         float scale = 1.0f);
cv::Point3f pixel2world(const cv::Point3f &pixel_p, const cv::Mat &K, float z,
                        const cv::Mat &T, const cv::Mat &R);
cv::Point3f world2pixel(const cv::Point3f &world_p, const cv::Mat &K,
                        const cv::Mat &T, const cv::Mat &R);

Eigen::Vector3f pixel2camera(const Eigen::Vector3f &pixel_p, const cv::Mat &K,
                             float z);
Eigen::Vector3f pixel2camera(const Eigen::Vector3f &pixel_p,
                             const Eigen::Matrix3f &K, float z);
Eigen::Vector3f camera2pixel(const Eigen::Vector3f &camera_p, const cv::Mat &K,
                             float scale = 1.0f);
Eigen::Vector3f camera2pixel(const Eigen::Vector3f &camera_p,
                             const Eigen::Matrix3f &K, float scale = 1.0f);
Eigen::Vector3f pixel2world(const Eigen::Vector3f &pixel_p,
                            const Eigen::Matrix3f &K, float z,
                            const Eigen::Matrix4f &T);
Eigen::Vector3f pixel2world(const Eigen::Vector3f &pixel_p,
                            const Eigen::Matrix3f &K, float z,
                            const Eigen::Isometry3f &T);
Eigen::Vector3f world2pixel(const Eigen::Vector3f &world_p,
                            const Eigen::Matrix3f &K, const Eigen::Matrix4f &T);
Eigen::Vector3f world2pixel(const Eigen::Vector3f &world_p,
                            const Eigen::Matrix3f &K,
                            const Eigen::Isometry3f &T);

Eigen::Vector3d pixel2camera(const Eigen::Vector3d &pixel_p, const cv::Mat &K,
                             float z);
Eigen::Vector3d pixel2camera(const Eigen::Vector3d &pixel_p,
                             const Eigen::Matrix3f &K, float z);
Eigen::Vector3d camera2pixel(const Eigen::Vector3d &camera_p, const cv::Mat &K,
                             float scale = 1.0f);
Eigen::Vector3d camera2pixel(const Eigen::Vector3d &camera_p,
                             const Eigen::Matrix3f &K, float scale = 1.0f);
Eigen::Vector3d pixel2world(const Eigen::Vector3d &pixel_p,
                            const Eigen::Matrix3f &K, float z,
                            const Eigen::Matrix4d &T);
Eigen::Vector3d pixel2world(const Eigen::Vector3d &pixel_p,
                            const Eigen::Matrix3f &K, float z,
                            const Eigen::Isometry3d &T);
Eigen::Vector3d world2pixel(const Eigen::Vector3d &world_p,
                            const Eigen::Matrix3f &K, const Eigen::Matrix4d &T);
Eigen::Vector3d world2pixel(const Eigen::Vector3d &world_p,
                            const Eigen::Matrix3f &K,
                            const Eigen::Isometry3d &T);

cv::Mat RodriguesR(const cv::Point3f &n, float s);

Eigen::Matrix3f RodriguesR(const Eigen::Vector3f &n, float s);

float GetRotTheta(const cv::Mat &R);

float GetRotTheta(const cv::Point3f& r);

float GetRotTheta(const Eigen::Matrix3f& R);

#endif

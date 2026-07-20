#ifndef VMATH_HPP
#define VMATH_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/AngleAxis.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
namespace neves {
template <typename  Scalar>
using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
template <typename  Scalar>
using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
template <typename Scalar>
using Quat = Eigen::Quaternion<Scalar>;
template <typename  Scalar>
using SO3 = Sophus::SO3<Scalar>;
template <typename  Scalar>
using SE3 = Sophus::SE3<Scalar>;

using Vec3d = Vec3<double>;
using Vec6d = Eigen::Matrix<double, 6, 1>;
using Mat3d = Mat3<double>;
using Mat36d = Eigen::Matrix<double, 3, 6>;
using SO3d = SO3<double>;
using SE3d = SE3<double>;


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

float GetRotTheta(const cv::Point3f &r);

float GetRotTheta(const Eigen::Matrix3f &R);

float GetRotTheta(const Eigen::Vector3f &r);

template <typename Scalar>
Vec3<Scalar> R2so3(const Mat3<Scalar> &R) {
  SO3<Scalar> so3(R);
  return so3.log();
}

template <typename Scalar>
Mat3<Scalar> so3ToR(const Vec3<Scalar> &phi) {
  return SO3<Scalar>::exp(phi).matrix();
}

template <typename Scalar>
Mat3<Scalar> so2R(const Vec3<Scalar> &phi) {
  return so3ToR(phi);
}

template <typename Scalar>
Vec3<Scalar> q2so3(const Eigen::Quaternion<Scalar> &q) {
  return SO3<Scalar>(q).log();
}

template <typename Scalar>
Eigen::Quaternion<Scalar> so3ToQ(const Vec3<Scalar> &phi) {
  return SO3<Scalar>::exp(phi).unit_quaternion();
}

template <typename Scalar>
Eigen::Quaternion<Scalar> so2q(const Vec3<Scalar> &phi) {
  return so3ToQ(phi);
}

template <typename Scalar>
Vec3<Scalar> aa2so3(const Eigen::AngleAxis<Scalar> &aa) {
  return SO3<Scalar>(aa).log();
}

template <typename Scalar>
Mat3<Scalar> Upso3dataLR(const Mat3<Scalar> &R_old,
                         const Vec3<Scalar> &dphi) {
  SO3<Scalar> R(R_old);
  return (SO3<Scalar>::exp(dphi) * R).matrix();
}

template <typename Scalar>
Mat3<Scalar> Upso3dataRR(const Mat3<Scalar> &R_old,
                         const Vec3<Scalar> &dphi) {
  SO3<Scalar> R(R_old);
  return (R * SO3<Scalar>::exp(dphi)).matrix();
}

template <typename Scalar>
Eigen::Matrix<Scalar, 4, 4>
Upse3dataLR(const Eigen::Matrix<Scalar, 4, 4> &T_old,
            const Eigen::Matrix<Scalar, 6, 1> &dphi) {
  SE3<Scalar> T(T_old);
  return (SE3<Scalar>::exp(dphi) * T).matrix();
}

template <typename Scalar>
Eigen::Matrix<Scalar, 4, 4>
Upse3dataRR(const Eigen::Matrix<Scalar, 4, 4> &T_old,
            const Eigen::Matrix<Scalar, 6, 1> &dphi) {
  SE3<Scalar> T(T_old);
  return (T * SE3<Scalar>::exp(dphi)).matrix();
}


} // namespace neves

#endif

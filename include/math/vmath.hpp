#ifndef VSLAMP_MATH_VMATH_HPP
#define VSLAMP_MATH_VMATH_HPP

#include <algorithm>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

namespace neves {

template <typename Scalar> using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
template <typename Scalar> using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
template <typename Scalar> using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
template <typename Scalar> using Quat = Eigen::Quaternion<Scalar>;
template <typename Scalar> using SO3 = Sophus::SO3<Scalar>;
template <typename Scalar> using SE3 = Sophus::SE3<Scalar>;

using Vec3d = Vec3<double>;
using Vec2d = Vec2<double>;
using Vec6d = Eigen::Matrix<double, 6, 1>;
using Mat3d = Mat3<double>;
using Mat36d = Eigen::Matrix<double, 3, 6>;
using SO3d = SO3<double>;
using SE3d = SE3<double>;

namespace detail {

inline double CvMatAt(const cv::Mat &mat, int row, int col) {
  CV_Assert(!mat.empty());
  CV_Assert(row >= 0 && row < mat.rows && col >= 0 && col < mat.cols);

  switch (mat.depth()) {
  case CV_32F:
    return mat.at<float>(row, col);
  case CV_64F:
    return mat.at<double>(row, col);
  default:
    CV_Error(cv::Error::StsUnsupportedFormat,
             "Only CV_32F and CV_64F matrices are supported");
  }

  return 0.0;
}

inline double CvVecAt(const cv::Mat &mat, int index) {
  CV_Assert(!mat.empty());
  CV_Assert(mat.rows == 1 || mat.cols == 1);
  return mat.rows == 1 ? CvMatAt(mat, 0, index) : CvMatAt(mat, index, 0);
}

} // namespace detail

inline float Rad2Deg(float rad) { return rad * 180.0f / CV_PI; }

inline float Deg2Rad(float deg) { return deg * CV_PI / 180.0f; }

inline float rad2deg(float rad) { return Rad2Deg(rad); }

inline float deg2rad(float deg) { return Deg2Rad(deg); }

inline Eigen::Matrix3f Skew(const Eigen::Vector3f &v) {
  Eigen::Matrix3f K;
  K << 0.0f, -v.z(), v.y(), v.z(), 0.0f, -v.x(), -v.y(), v.x(), 0.0f;
  return K;
}

inline Eigen::Matrix3d Mat2Eigen33(const cv::Mat &R) {
  CV_Assert(!R.empty());
  CV_Assert(R.rows == 3 && R.cols == 3);

  Eigen::Matrix3d R_eigen;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      R_eigen(row, col) = detail::CvMatAt(R, row, col);
    }
  }

  return R_eigen;
}

inline cv::Point3f Pixel2Camera(const cv::Point3f &pixel_p, const cv::Mat &K,
                                float z) {
  const double fx = detail::CvMatAt(K, 0, 0);
  const double fy = detail::CvMatAt(K, 1, 1);
  const double cx = detail::CvMatAt(K, 0, 2);
  const double cy = detail::CvMatAt(K, 1, 2);

  return cv::Point3f(static_cast<float>((pixel_p.x - cx) * z / fx),
                     static_cast<float>((pixel_p.y - cy) * z / fy), z);
}

inline cv::Point3f Camera2Pixel(const cv::Point3f &camera_p, const cv::Mat &K,
                                float scale = 1.0f) {
  const double fx = detail::CvMatAt(K, 0, 0);
  const double fy = detail::CvMatAt(K, 1, 1);
  const double cx = detail::CvMatAt(K, 0, 2);
  const double cy = detail::CvMatAt(K, 1, 2);

  return cv::Point3f(
      static_cast<float>(camera_p.x * fx / camera_p.z + cx),
      static_cast<float>(camera_p.y * fy / camera_p.z + cy),
      camera_p.z * scale);
}

inline cv::Point3f Pixel2World(const cv::Point3f &pixel_p, const cv::Mat &K,
                               float z, const cv::Mat &T, const cv::Mat &R) {
  const cv::Point3f camera_p = Pixel2Camera(pixel_p, K, z);

  cv::Point3f world_p;
  world_p.x = static_cast<float>(detail::CvMatAt(R, 0, 0) * camera_p.x +
                                 detail::CvMatAt(R, 0, 1) * camera_p.y +
                                 detail::CvMatAt(R, 0, 2) * camera_p.z +
                                 detail::CvVecAt(T, 0));
  world_p.y = static_cast<float>(detail::CvMatAt(R, 1, 0) * camera_p.x +
                                 detail::CvMatAt(R, 1, 1) * camera_p.y +
                                 detail::CvMatAt(R, 1, 2) * camera_p.z +
                                 detail::CvVecAt(T, 1));
  world_p.z = static_cast<float>(detail::CvMatAt(R, 2, 0) * camera_p.x +
                                 detail::CvMatAt(R, 2, 1) * camera_p.y +
                                 detail::CvMatAt(R, 2, 2) * camera_p.z +
                                 detail::CvVecAt(T, 2));
  return world_p;
}

inline cv::Point3f World2Pixel(const cv::Point3f &world_p, const cv::Mat &K,
                               const cv::Mat &T, const cv::Mat &R) {
  cv::Point3f camera_p;
  camera_p.x = static_cast<float>(detail::CvMatAt(R, 0, 0) * world_p.x +
                                  detail::CvMatAt(R, 1, 0) * world_p.y +
                                  detail::CvMatAt(R, 2, 0) * world_p.z +
                                  detail::CvVecAt(T, 0));
  camera_p.y = static_cast<float>(detail::CvMatAt(R, 0, 1) * world_p.x +
                                  detail::CvMatAt(R, 1, 1) * world_p.y +
                                  detail::CvMatAt(R, 2, 1) * world_p.z +
                                  detail::CvVecAt(T, 1));
  camera_p.z = static_cast<float>(detail::CvMatAt(R, 0, 2) * world_p.x +
                                  detail::CvMatAt(R, 1, 2) * world_p.y +
                                  detail::CvMatAt(R, 2, 2) * world_p.z +
                                  detail::CvVecAt(T, 2));
  return Camera2Pixel(camera_p, K);
}

inline Eigen::Vector3f Pixel2Camera(const Eigen::Vector3f &pixel_p,
                                    const cv::Mat &K, float z) {
  const double fx = detail::CvMatAt(K, 0, 0);
  const double fy = detail::CvMatAt(K, 1, 1);
  const double cx = detail::CvMatAt(K, 0, 2);
  const double cy = detail::CvMatAt(K, 1, 2);

  return Eigen::Vector3f(static_cast<float>((pixel_p.x() - cx) * z / fx),
                         static_cast<float>((pixel_p.y() - cy) * z / fy), z);
}

inline Eigen::Vector3f Pixel2Camera(const Eigen::Vector3f &pixel_p,
                                    const Eigen::Matrix3f &K, float z) {
  return Eigen::Vector3f((pixel_p.x() - K(0, 2)) * z / K(0, 0),
                         (pixel_p.y() - K(1, 2)) * z / K(1, 1), z);
}

inline Eigen::Vector3f Camera2Pixel(const Eigen::Vector3f &camera_p,
                                    const cv::Mat &K, float scale = 1.0f) {
  const double fx = detail::CvMatAt(K, 0, 0);
  const double fy = detail::CvMatAt(K, 1, 1);
  const double cx = detail::CvMatAt(K, 0, 2);
  const double cy = detail::CvMatAt(K, 1, 2);

  return Eigen::Vector3f(
      static_cast<float>(camera_p.x() * fx / camera_p.z() + cx),
      static_cast<float>(camera_p.y() * fy / camera_p.z() + cy),
      camera_p.z() * scale);
}

inline Eigen::Vector3f Camera2Pixel(const Eigen::Vector3f &camera_p,
                                    const Eigen::Matrix3f &K,
                                    float scale = 1.0f) {
  return Eigen::Vector3f(camera_p.x() * K(0, 0) / camera_p.z() + K(0, 2),
                         camera_p.y() * K(1, 1) / camera_p.z() + K(1, 2),
                         camera_p.z() * scale);
}

inline Eigen::Vector3f Pixel2World(const Eigen::Vector3f &pixel_p,
                                   const Eigen::Matrix3f &K, float z,
                                   const Eigen::Matrix4f &T) {
  const Eigen::Vector3f camera_p = Pixel2Camera(pixel_p, K, z);
  const Eigen::Vector4f world_p =
      T * Eigen::Vector4f(camera_p.x(), camera_p.y(), camera_p.z(), 1.0f);
  return world_p.head<3>();
}

inline Eigen::Vector3f Pixel2World(const Eigen::Vector3f &pixel_p,
                                   const Eigen::Matrix3f &K, float z,
                                   const Eigen::Isometry3f &T) {
  return Pixel2World(pixel_p, K, z, T.matrix());
}

inline Eigen::Vector3f World2Pixel(const Eigen::Vector3f &world_p,
                                   const Eigen::Matrix3f &K,
                                   const Eigen::Matrix4f &T) {
  const Eigen::Vector4f camera_p =
      T.inverse() *
      Eigen::Vector4f(world_p.x(), world_p.y(), world_p.z(), 1.0f);
  const Eigen::Vector3f camera_p3 = camera_p.head<3>();
  return Camera2Pixel(camera_p3, K);
}

inline Eigen::Vector3f World2Pixel(const Eigen::Vector3f &world_p,
                                   const Eigen::Matrix3f &K,
                                   const Eigen::Isometry3f &T) {
  return World2Pixel(world_p, K, T.matrix());
}

inline Eigen::Vector3d Pixel2Camera(const Eigen::Vector3d &pixel_p,
                                    const cv::Mat &K, double z) {
  const double fx = detail::CvMatAt(K, 0, 0);
  const double fy = detail::CvMatAt(K, 1, 1);
  const double cx = detail::CvMatAt(K, 0, 2);
  const double cy = detail::CvMatAt(K, 1, 2);

  return Eigen::Vector3d((pixel_p.x() - cx) * z / fx,
                         (pixel_p.y() - cy) * z / fy, z);
}

inline Eigen::Vector3d Pixel2Camera(const Eigen::Vector3d &pixel_p,
                                    const Eigen::Matrix3f &K, double z) {
  return Eigen::Vector3d((pixel_p.x() - K(0, 2)) * z / K(0, 0),
                         (pixel_p.y() - K(1, 2)) * z / K(1, 1), z);
}

inline Eigen::Vector3d Camera2Pixel(const Eigen::Vector3d &camera_p,
                                    const cv::Mat &K, double scale = 1.0) {
  const double fx = detail::CvMatAt(K, 0, 0);
  const double fy = detail::CvMatAt(K, 1, 1);
  const double cx = detail::CvMatAt(K, 0, 2);
  const double cy = detail::CvMatAt(K, 1, 2);

  return Eigen::Vector3d(camera_p.x() * fx / camera_p.z() + cx,
                         camera_p.y() * fy / camera_p.z() + cy,
                         camera_p.z() * scale);
}

inline Eigen::Vector3d Camera2Pixel(const Eigen::Vector3d &camera_p,
                                    const Eigen::Matrix3f &K,
                                    double scale = 1.0) {
  return Eigen::Vector3d(camera_p.x() * K(0, 0) / camera_p.z() + K(0, 2),
                         camera_p.y() * K(1, 1) / camera_p.z() + K(1, 2),
                         camera_p.z() * scale);
}

inline Eigen::Vector3d Pixel2World(const Eigen::Vector3d &pixel_p,
                                   const Eigen::Matrix3f &K, double z,
                                   const Eigen::Matrix4d &T) {
  const Eigen::Vector3d camera_p = Pixel2Camera(pixel_p, K, z);
  const Eigen::Vector4d world_p =
      T * Eigen::Vector4d(camera_p.x(), camera_p.y(), camera_p.z(), 1.0);
  return world_p.head<3>();
}

inline Eigen::Vector3d Pixel2World(const Eigen::Vector3d &pixel_p,
                                   const Eigen::Matrix3f &K, double z,
                                   const Eigen::Isometry3d &T) {
  return Pixel2World(pixel_p, K, z, T.matrix());
}

inline Eigen::Vector3d World2Pixel(const Eigen::Vector3d &world_p,
                                   const Eigen::Matrix3f &K,
                                   const Eigen::Matrix4d &T) {
  const Eigen::Vector4d camera_p =
      T.inverse() * Eigen::Vector4d(world_p.x(), world_p.y(), world_p.z(), 1.0);
  const Eigen::Vector3d camera_p3 = camera_p.head<3>();
  return Camera2Pixel(camera_p3, K);
}

inline Eigen::Vector3d World2Pixel(const Eigen::Vector3d &world_p,
                                   const Eigen::Matrix3f &K,
                                   const Eigen::Isometry3d &T) {
  return World2Pixel(world_p, K, T.matrix());
}

inline cv::Mat RodriguesR(const cv::Point3f &n, float s) {
  cv::Point3f axis = n;
  const float norm =
      std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);

  if (norm < 1e-8f) {
    return cv::Mat::eye(3, 3, CV_32F);
  }

  axis.x /= norm;
  axis.y /= norm;
  axis.z /= norm;

  const cv::Mat rvec =
      (cv::Mat_<float>(3, 1) << axis.x * s, axis.y * s, axis.z * s);

  cv::Mat R;
  cv::Rodrigues(rvec, R);
  return R;
}

inline Eigen::Matrix3f RodriguesR(const Eigen::Vector3f &n, float s) {
  const float norm = n.norm();

  if (norm < 1e-5f) {
    return Eigen::Matrix3f::Identity();
  }

  const Eigen::Vector3f axis = n / norm;
  return Eigen::AngleAxisf(s, axis).toRotationMatrix();
}

inline float GetRotTheta(const cv::Mat &R) {
  CV_Assert(R.rows == 3 && R.cols == 3);
  float cos_s = static_cast<float>((cv::trace(R)[0] - 1.0) / 2.0);
  cos_s = std::clamp(cos_s, -1.0f, 1.0f);
  return std::acos(cos_s);
}

inline float GetRotTheta(const cv::Point3f &r) { return cv::norm(r); }

inline float GetRotTheta(const Eigen::Matrix3f &R) {
  float cos_s = static_cast<float>((R.trace() - 1.0f) / 2.0f);
  cos_s = std::clamp(cos_s, -1.0f, 1.0f);
  return std::acos(cos_s);
}

inline float GetRotTheta(const Eigen::Vector3f &r) { return r.norm(); }

template <typename Scalar> Vec3<Scalar> R2So3(const Mat3<Scalar> &R) {
  return SO3<Scalar>(R).log();
}

template <typename Scalar> Mat3<Scalar> So3ToR(const Vec3<Scalar> &phi) {
  return SO3<Scalar>::exp(phi).matrix();
}

template <typename Scalar> Mat3<Scalar> So2R(const Vec3<Scalar> &phi) {
  return So3ToR(phi);
}

template <typename Scalar>
Vec3<Scalar> Q2So3(const Eigen::Quaternion<Scalar> &q) {
  return SO3<Scalar>(q).log();
}

template <typename Scalar>
Eigen::Quaternion<Scalar> So3ToQ(const Vec3<Scalar> &phi) {
  return SO3<Scalar>::exp(phi).unit_quaternion();
}

template <typename Scalar>
Eigen::Quaternion<Scalar> So2Q(const Vec3<Scalar> &phi) {
  return So3ToQ(phi);
}

template <typename Scalar>
Vec3<Scalar> Aa2So3(const Eigen::AngleAxis<Scalar> &aa) {
  return SO3<Scalar>(aa).log();
}

template <typename Scalar>
Mat3<Scalar> UpdateSo3DataLeft(const Mat3<Scalar> &R_old,
                               const Vec3<Scalar> &dphi) {
  const SO3<Scalar> R(R_old);
  return (SO3<Scalar>::exp(dphi) * R).matrix();
}

template <typename Scalar>
Mat3<Scalar> UpdateSo3DataRight(const Mat3<Scalar> &R_old,
                                const Vec3<Scalar> &dphi) {
  const SO3<Scalar> R(R_old);
  return (R * SO3<Scalar>::exp(dphi)).matrix();
}

template <typename Scalar>
Eigen::Matrix<Scalar, 4, 4>
UpdateSe3DataLeft(const Eigen::Matrix<Scalar, 4, 4> &T_old,
                  const Eigen::Matrix<Scalar, 6, 1> &dphi) {
  const SE3<Scalar> T(T_old);
  return (SE3<Scalar>::exp(dphi) * T).matrix();
}

template <typename Scalar>
Eigen::Matrix<Scalar, 4, 4>
UpdateSe3DataRight(const Eigen::Matrix<Scalar, 4, 4> &T_old,
                   const Eigen::Matrix<Scalar, 6, 1> &dphi) {
  const SE3<Scalar> T(T_old);
  return (T * SE3<Scalar>::exp(dphi)).matrix();
}

inline cv::Point3f pixel2camera(const cv::Point3f &pixel_p, const cv::Mat &K,
                                float z) {
  return Pixel2Camera(pixel_p, K, z);
}

inline cv::Point3f camera2pixel(const cv::Point3f &camera_p, const cv::Mat &K,
                                float scale = 1.0f) {
  return Camera2Pixel(camera_p, K, scale);
}

inline cv::Point3f pixel2world(const cv::Point3f &pixel_p, const cv::Mat &K,
                               float z, const cv::Mat &T, const cv::Mat &R) {
  return Pixel2World(pixel_p, K, z, T, R);
}

inline cv::Point3f world2pixel(const cv::Point3f &world_p, const cv::Mat &K,
                               const cv::Mat &T, const cv::Mat &R) {
  return World2Pixel(world_p, K, T, R);
}

inline Eigen::Vector3f pixel2camera(const Eigen::Vector3f &pixel_p,
                                    const cv::Mat &K, float z) {
  return Pixel2Camera(pixel_p, K, z);
}

inline Eigen::Vector3f pixel2camera(const Eigen::Vector3f &pixel_p,
                                    const Eigen::Matrix3f &K, float z) {
  return Pixel2Camera(pixel_p, K, z);
}

inline Eigen::Vector3f camera2pixel(const Eigen::Vector3f &camera_p,
                                    const cv::Mat &K, float scale = 1.0f) {
  return Camera2Pixel(camera_p, K, scale);
}

inline Eigen::Vector3f camera2pixel(const Eigen::Vector3f &camera_p,
                                    const Eigen::Matrix3f &K,
                                    float scale = 1.0f) {
  return Camera2Pixel(camera_p, K, scale);
}

inline Eigen::Vector3f pixel2world(const Eigen::Vector3f &pixel_p,
                                   const Eigen::Matrix3f &K, float z,
                                   const Eigen::Matrix4f &T) {
  return Pixel2World(pixel_p, K, z, T);
}

inline Eigen::Vector3f pixel2world(const Eigen::Vector3f &pixel_p,
                                   const Eigen::Matrix3f &K, float z,
                                   const Eigen::Isometry3f &T) {
  return Pixel2World(pixel_p, K, z, T);
}

inline Eigen::Vector3f world2pixel(const Eigen::Vector3f &world_p,
                                   const Eigen::Matrix3f &K,
                                   const Eigen::Matrix4f &T) {
  return World2Pixel(world_p, K, T);
}

inline Eigen::Vector3f world2pixel(const Eigen::Vector3f &world_p,
                                   const Eigen::Matrix3f &K,
                                   const Eigen::Isometry3f &T) {
  return World2Pixel(world_p, K, T);
}

inline Eigen::Vector3d pixel2camera(const Eigen::Vector3d &pixel_p,
                                    const cv::Mat &K, double z) {
  return Pixel2Camera(pixel_p, K, z);
}

inline Eigen::Vector3d pixel2camera(const Eigen::Vector3d &pixel_p,
                                    const Eigen::Matrix3f &K, double z) {
  return Pixel2Camera(pixel_p, K, z);
}

inline Eigen::Vector3d camera2pixel(const Eigen::Vector3d &camera_p,
                                    const cv::Mat &K, double scale = 1.0) {
  return Camera2Pixel(camera_p, K, scale);
}

inline Eigen::Vector3d camera2pixel(const Eigen::Vector3d &camera_p,
                                    const Eigen::Matrix3f &K,
                                    double scale = 1.0) {
  return Camera2Pixel(camera_p, K, scale);
}

inline Eigen::Vector3d pixel2world(const Eigen::Vector3d &pixel_p,
                                   const Eigen::Matrix3f &K, double z,
                                   const Eigen::Matrix4d &T) {
  return Pixel2World(pixel_p, K, z, T);
}

inline Eigen::Vector3d pixel2world(const Eigen::Vector3d &pixel_p,
                                   const Eigen::Matrix3f &K, double z,
                                   const Eigen::Isometry3d &T) {
  return Pixel2World(pixel_p, K, z, T);
}

inline Eigen::Vector3d world2pixel(const Eigen::Vector3d &world_p,
                                   const Eigen::Matrix3f &K,
                                   const Eigen::Matrix4d &T) {
  return World2Pixel(world_p, K, T);
}

inline Eigen::Vector3d world2pixel(const Eigen::Vector3d &world_p,
                                   const Eigen::Matrix3f &K,
                                   const Eigen::Isometry3d &T) {
  return World2Pixel(world_p, K, T);
}

template <typename Scalar> Vec3<Scalar> R2so3(const Mat3<Scalar> &R) {
  return R2So3(R);
}

template <typename Scalar> Mat3<Scalar> so3ToR(const Vec3<Scalar> &phi) {
  return So3ToR(phi);
}

template <typename Scalar> Mat3<Scalar> so2R(const Vec3<Scalar> &phi) {
  return So2R(phi);
}

template <typename Scalar>
Vec3<Scalar> q2so3(const Eigen::Quaternion<Scalar> &q) {
  return Q2So3(q);
}

template <typename Scalar>
Eigen::Quaternion<Scalar> so3ToQ(const Vec3<Scalar> &phi) {
  return So3ToQ(phi);
}

template <typename Scalar>
Eigen::Quaternion<Scalar> so2q(const Vec3<Scalar> &phi) {
  return So2Q(phi);
}

template <typename Scalar>
Vec3<Scalar> aa2so3(const Eigen::AngleAxis<Scalar> &aa) {
  return Aa2So3(aa);
}

template <typename Scalar>
Mat3<Scalar> Upso3dataLR(const Mat3<Scalar> &R_old,
                         const Vec3<Scalar> &dphi) {
  return UpdateSo3DataLeft(R_old, dphi);
}

template <typename Scalar>
Mat3<Scalar> Upso3dataRR(const Mat3<Scalar> &R_old,
                         const Vec3<Scalar> &dphi) {
  return UpdateSo3DataRight(R_old, dphi);
}

template <typename Scalar>
Eigen::Matrix<Scalar, 4, 4>
Upse3dataLR(const Eigen::Matrix<Scalar, 4, 4> &T_old,
            const Eigen::Matrix<Scalar, 6, 1> &dphi) {
  return UpdateSe3DataLeft(T_old, dphi);
}

template <typename Scalar>
Eigen::Matrix<Scalar, 4, 4>
Upse3dataRR(const Eigen::Matrix<Scalar, 4, 4> &T_old,
            const Eigen::Matrix<Scalar, 6, 1> &dphi) {
  return UpdateSe3DataRight(T_old, dphi);
}

} // namespace neves

#endif

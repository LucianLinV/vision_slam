#include "vmath.hpp"

float rad2deg(float rad) { return rad * 180.0f / CV_PI; }

float deg2rad(float deg) { return deg * CV_PI / 180.0f; }

Eigen::Matrix3f Skew(const Eigen::Vector3f &v) {
  Eigen::Matrix3f K;
  K << 0.0f, -v.z(), v.y(), v.z(), 0.0f, -v.x(), -v.y(), v.x(), 0.0f;
  return K;
}

cv::Point3f pixel2camera(const cv::Point3f &pixel_p, const cv::Mat &K,
                         float z) {
  cv::Point3f camera_p;
  camera_p.x = (pixel_p.x - K.at<float>(0, 2)) * z / K.at<float>(0, 0);
  camera_p.y = (pixel_p.y - K.at<float>(1, 2)) * z / K.at<float>(1, 1);
  camera_p.z = z;
  return camera_p;
}

cv::Point3f camera2pixel(cv::Point3f &camera_p, const cv::Mat &K, float scale) {
  cv::Point3f pixel_p;
  pixel_p.z = camera_p.z * scale;
  pixel_p.x = camera_p.x * K.at<float>(0, 0) / camera_p.z + K.at<float>(0, 2);
  pixel_p.y = camera_p.y * K.at<float>(1, 1) / camera_p.z + K.at<float>(1, 2);
  return pixel_p;
}

cv::Point3f pixel2world(const cv::Point3f &pixel_p, const cv::Mat &K, float z,
                        const cv::Mat &T, const cv::Mat &R) {
  cv::Point3f camera_p = pixel2camera(pixel_p, K, z);
  cv::Point3f world_p;
  world_p.x = R.at<float>(0, 0) * camera_p.x + R.at<float>(0, 1) * camera_p.y +
              R.at<float>(0, 2) * camera_p.z + T.at<float>(0);
  world_p.y = R.at<float>(1, 0) * camera_p.x + R.at<float>(1, 1) * camera_p.y +
              R.at<float>(1, 2) * camera_p.z + T.at<float>(1);
  world_p.z = R.at<float>(2, 0) * camera_p.x + R.at<float>(2, 1) * camera_p.y +
              R.at<float>(2, 2) * camera_p.z + T.at<float>(2);
  return world_p;
}

cv::Point3f world2pixel(const cv::Point3f &world_p, const cv::Mat &K,
                        const cv::Mat &T, const cv::Mat &R) {
  cv::Point3f camera_p;
  camera_p.x = R.at<float>(0, 0) * world_p.x + R.at<float>(1, 0) * world_p.y +
               R.at<float>(2, 0) * world_p.z + T.at<float>(0);
  camera_p.y = R.at<float>(0, 1) * world_p.x + R.at<float>(1, 1) * world_p.y +
               R.at<float>(2, 1) * world_p.z + T.at<float>(1);
  camera_p.z = R.at<float>(0, 2) * world_p.x + R.at<float>(1, 2) * world_p.y +
               R.at<float>(2, 2) * world_p.z + T.at<float>(2);
  return camera2pixel(camera_p, K);
}

Eigen::Vector3f pixel2camera(const Eigen::Vector3f &pixel_p, const cv::Mat &K,
                             float z) {
  Eigen::Vector3f camera_p;
  camera_p.x() = (pixel_p.x() - K.at<float>(0, 2)) * z / K.at<float>(0, 0);
  camera_p.y() = (pixel_p.y() - K.at<float>(1, 2)) * z / K.at<float>(1, 1);
  camera_p.z() = z;
  return camera_p;
}

Eigen::Vector3f pixel2camera(const Eigen::Vector3f &pixel_p,
                             const Eigen::Matrix3f &K, float z) {
  Eigen::Vector3f camera_p;
  camera_p.x() = (pixel_p.x() - K(0, 2)) * z / K(0, 0);
  camera_p.y() = (pixel_p.y() - K(1, 2)) * z / K(1, 1);
  camera_p.z() = z;
  return camera_p;
}

Eigen::Vector3f camera2pixel(const Eigen::Vector3f &camera_p, const cv::Mat &K,
                             float scale) {
  Eigen::Vector3f pixel_p;
  pixel_p.z() = camera_p.z() * scale;
  pixel_p.x() =
      camera_p.x() * K.at<float>(0, 0) / camera_p.z() + K.at<float>(0, 2);
  pixel_p.y() =
      camera_p.y() * K.at<float>(1, 1) / camera_p.z() + K.at<float>(1, 2);
  return pixel_p;
}

Eigen::Vector3f camera2pixel(const Eigen::Vector3f &camera_p,
                             const Eigen::Matrix3f &K, float scale) {
  Eigen::Vector3f pixel_p;
  pixel_p.z() = camera_p.z() * scale;
  pixel_p.x() = camera_p.x() * K(0, 0) / camera_p.z() + K(0, 2);
  pixel_p.y() = camera_p.y() * K(1, 1) / camera_p.z() + K(1, 2);
  return pixel_p;
}

Eigen::Vector3f pixel2world(const Eigen::Vector3f &pixel_p,
                            const Eigen::Matrix3f &K, float z,
                            const Eigen::Matrix4f &T) {
  Eigen::Vector3f camera_p = pixel2camera(pixel_p, K, z);
  Eigen::Vector4f world_p =
      T * Eigen::Vector4f(camera_p.x(), camera_p.y(), camera_p.z(), 1.0f);
  return world_p.head<3>();
}

Eigen::Vector3f pixel2world(const Eigen::Vector3f &pixel_p,
                            const Eigen::Matrix3f &K, float z,
                            const Eigen::Isometry3f &T) {
  Eigen::Vector3f camera_p = pixel2camera(pixel_p, K, z);
  Eigen::Vector4f world_p =
      T.matrix() *
      Eigen::Vector4f(camera_p.x(), camera_p.y(), camera_p.z(), 1.0f);
  return world_p.head<3>();
}

Eigen::Vector3f world2pixel(const Eigen::Vector3f &world_p,
                            const Eigen::Matrix3f &K,
                            const Eigen::Matrix4f &T) {
  Eigen::Vector4f camera_p =
      T.inverse() *
      Eigen::Vector4f(world_p.x(), world_p.y(), world_p.z(), 1.0f);
  const Eigen::Vector3f camera_p3 = camera_p.head<3>();
  return camera2pixel(camera_p3, K);
}

Eigen::Vector3f world2pixel(const Eigen::Vector3f &world_p,
                            const Eigen::Matrix3f &K,
                            const Eigen::Isometry3f &T) {
  Eigen::Vector4f camera_p =
      T.inverse().matrix() *
      Eigen::Vector4f(world_p.x(), world_p.y(), world_p.z(), 1.0f);
  const Eigen::Vector3f camera_p3 = camera_p.head<3>();
  return camera2pixel(camera_p3, K);
}

Eigen::Vector3d pixel2camera(const Eigen::Vector3d &pixel_p, const cv::Mat &K,
                             float z) {
  Eigen::Vector3d camera_p;
  camera_p.x() = (pixel_p.x() - K.at<float>(0, 2)) * z / K.at<float>(0, 0);
  camera_p.y() = (pixel_p.y() - K.at<float>(1, 2)) * z / K.at<float>(1, 1);
  camera_p.z() = z;
  return camera_p;
}

Eigen::Vector3d pixel2camera(const Eigen::Vector3d &pixel_p,
                             const Eigen::Matrix3f &K, float z) {
  Eigen::Vector3d camera_p;
  camera_p.x() = (pixel_p.x() - K(0, 2)) * z / K(0, 0);
  camera_p.y() = (pixel_p.y() - K(1, 2)) * z / K(1, 1);
  camera_p.z() = z;
  return camera_p;
}

Eigen::Vector3d camera2pixel(const Eigen::Vector3d &camera_p, const cv::Mat &K,
                             float scale) {
  Eigen::Vector3d pixel_p;
  pixel_p.z() = camera_p.z() * scale;
  pixel_p.x() =
      camera_p.x() * K.at<float>(0, 0) / camera_p.z() + K.at<float>(0, 2);
  pixel_p.y() =
      camera_p.y() * K.at<float>(1, 1) / camera_p.z() + K.at<float>(1, 2);
  return pixel_p;
}

Eigen::Vector3d camera2pixel(const Eigen::Vector3d &camera_p,
                             const Eigen::Matrix3f &K, float scale) {
  Eigen::Vector3d pixel_p;
  pixel_p.z() = camera_p.z() * scale;
  pixel_p.x() = camera_p.x() * K(0, 0) / camera_p.z() + K(0, 2);
  pixel_p.y() = camera_p.y() * K(1, 1) / camera_p.z() + K(1, 2);
  return pixel_p;
}

Eigen::Vector3d pixel2world(const Eigen::Vector3d &pixel_p,
                            const Eigen::Matrix3f &K, float z,
                            const Eigen::Matrix4d &T) {
  Eigen::Vector3d camera_p = pixel2camera(pixel_p, K, z);
  Eigen::Vector4d world_p =
      T * Eigen::Vector4d(camera_p.x(), camera_p.y(), camera_p.z(), 1.0);
  return world_p.head<3>();
}

Eigen::Vector3d pixel2world(const Eigen::Vector3d &pixel_p,
                            const Eigen::Matrix3f &K, float z,
                            const Eigen::Isometry3d &T) {
  Eigen::Vector3d camera_p = pixel2camera(pixel_p, K, z);
  Eigen::Vector4d world_p =
      T.matrix() *
      Eigen::Vector4d(camera_p.x(), camera_p.y(), camera_p.z(), 1.0);
  return world_p.head<3>();
}

Eigen::Vector3d world2pixel(const Eigen::Vector3d &world_p,
                            const Eigen::Matrix3f &K,
                            const Eigen::Matrix4d &T) {
  Eigen::Vector4d camera_p =
      T.inverse() * Eigen::Vector4d(world_p.x(), world_p.y(), world_p.z(), 1.0);
  const Eigen::Vector3d camera_p3 = camera_p.head<3>();
  return camera2pixel(camera_p3, K);
}

Eigen::Vector3d world2pixel(const Eigen::Vector3d &world_p,
                            const Eigen::Matrix3f &K,
                            const Eigen::Isometry3d &T) {
  Eigen::Vector4d camera_p =
      T.inverse().matrix() *
      Eigen::Vector4d(world_p.x(), world_p.y(), world_p.z(), 1.0);
  const Eigen::Vector3d camera_p3 = camera_p.head<3>();
  return camera2pixel(camera_p3, K);
}

cv::Mat RodriguesR(const cv::Point3f &n, float s) {
  cv::Point3f axis = n;
  float norm = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);

  if (norm < 1e-8f) {
    return cv::Mat::eye(3, 3, CV_32F);
  }

  axis.x /= norm;
  axis.y /= norm;
  axis.z /= norm;

  cv::Mat rvec = (cv::Mat_<float>(3, 1) << axis.x * s, axis.y * s, axis.z * s);

  cv::Mat R;
  cv::Rodrigues(rvec, R);
  return R;
}

Eigen::Matrix3f RodriguesR(const Eigen::Vector3f &n, float s) {
  // Eigen::Vector3f axis = n;
  // float norm = axis.norm();
  // if(norm<1e-5f){
  //     return Eigen::Matrix3f::Identity();
  // }
  // axis.normalize();
  // float c = std::cos(s);
  // float sin_s = std::sin(s);

  // Eigen::Matrix3f K = Skew(axis);
  // Eigen::Matrix3f R =
  // c*Eigen::Matrix3f::Identity()+(1-c)*axis*axis.transpose()+sin_s*K;
  const float angle = n.norm();

  if (angle < 1e-5f) {
    return Eigen::Matrix3f::Identity();
  }

  const Eigen::Vector3f axis = n / angle;
  return Eigen::AngleAxisf(s, axis).toRotationMatrix();
}

float GetRotTheta(const cv::Mat &R) {
  CV_Assert(R.rows == 3 && R.cols == 3);
  float cos_s = static_cast<float>((cv::trace(R)[0] - 1.0) / 2.0f);
  cos_s = std::max(-1.0f, std::min(1.0f, cos_s));

  return std::acos(cos_s);
}

float GetRotTheta(const cv::Point3f &r) { return cv::norm(r); }

float GetRotTheta(const Eigen::Matrix3f &R) {
  float cos_s = static_cast<float>((R.trace() - 1.0) / 2.f);
  cos_s = std::max(-1.0f, std::min(1.0f, cos_s));

  return std::acos(cos_s);
}

float GetRotTheta(const Eigen::Vector3f &r) { return r.norm(); }
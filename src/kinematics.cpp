#include "kinematics.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr int kStateSize = 4;
constexpr int kMeasSize = 2;
constexpr int kControlSize = 0;

void set_transition(cv::Mat& F, double dt) {
  F = cv::Mat::eye(kStateSize, kStateSize, CV_32F);
  F.at<float>(0, 2) = static_cast<float>(dt);
  F.at<float>(1, 3) = static_cast<float>(dt);
}

}  // namespace

KinematicsFilter::KinematicsFilter() : kf_(kStateSize, kMeasSize, kControlSize) {
  kf_.transitionMatrix = cv::Mat::eye(kStateSize, kStateSize, CV_32F);
  kf_.measurementMatrix = cv::Mat::zeros(kMeasSize, kStateSize, CV_32F);
  kf_.measurementMatrix.at<float>(0, 0) = 1.0f;
  kf_.measurementMatrix.at<float>(1, 1) = 1.0f;

  cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-2));
  kf_.processNoiseCov.at<float>(2, 2) = 5.0f;
  kf_.processNoiseCov.at<float>(3, 3) = 5.0f;

  cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(8.0));
  cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1.0));
}

Sample KinematicsFilter::update(double t_s, cv::Point2f measured_px) {
  if (!initialized_) {
    kf_.statePost = (cv::Mat_<float>(kStateSize, 1) << measured_px.x, measured_px.y, 0.0f, 0.0f);
    kf_.statePre = kf_.statePost.clone();
    last_t_s_ = t_s;
    last_x_ = measured_px.x;
    last_y_ = measured_px.y;
    initialized_ = true;
    Sample out{0, t_s, measured_px.x, measured_px.y, 0.0, 0.0, 0.0, 0.0};
    out.simulated = false;
    return out;
  }

  const double dt = std::max(1e-4, t_s - last_t_s_);
  set_transition(kf_.transitionMatrix, dt);

  kf_.predict();

  cv::Mat measurement = (cv::Mat_<float>(kMeasSize, 1) << measured_px.x, measured_px.y);
  const cv::Mat estimated = kf_.correct(measurement);

  last_t_s_ = t_s;

  const double x = estimated.at<float>(0);
  const double y = estimated.at<float>(1);
  const double vx = estimated.at<float>(2);
  const double vy = estimated.at<float>(3);
  const double speed = std::hypot(vx, vy);
  distance_px_ += std::hypot(x - last_x_, y - last_y_);
  last_x_ = x;
  last_y_ = y;

  Sample out{0, t_s, x, y, vx, vy, speed, distance_px_};
  out.simulated = false;
  return out;
}

Sample KinematicsFilter::predict(double t_s) {
  if (!initialized_) {
    return {};
  }

  const double dt = std::max(1e-4, t_s - last_t_s_);
  set_transition(kf_.transitionMatrix, dt);

  const cv::Mat predicted = kf_.predict();
  last_t_s_ = t_s;

  const double x = predicted.at<float>(0);
  const double y = predicted.at<float>(1);
  const double vx = predicted.at<float>(2);
  const double vy = predicted.at<float>(3);
  const double speed = std::hypot(vx, vy);
  distance_px_ += std::hypot(x - last_x_, y - last_y_);
  last_x_ = x;
  last_y_ = y;

  Sample out{0, t_s, x, y, vx, vy, speed, distance_px_};
  out.simulated = true;
  return out;
}

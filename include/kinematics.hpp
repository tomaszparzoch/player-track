#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include "types.hpp"

class KinematicsFilter {
 public:
  KinematicsFilter();

  Sample update(double t_s, cv::Point2f measured_px);
  Sample predict(double t_s);
  bool is_initialized() const { return initialized_; }
  double distance_px() const { return distance_px_; }

 private:
  cv::KalmanFilter kf_;
  bool initialized_{false};
  double last_t_s_{0.0};
  double last_x_{0.0};
  double last_y_{0.0};
  double distance_px_{0.0};
};

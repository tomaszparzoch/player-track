#pragma once

#include <optional>

#include <opencv2/core.hpp>
#include <opencv2/tracking.hpp>

class PlayerTracker {
 public:
  bool init(const cv::Mat& frame, const cv::Rect2d& bbox);
  std::optional<cv::Rect2d> update(const cv::Mat& frame);

 private:
  cv::Ptr<cv::TrackerCSRT> tracker_;
};

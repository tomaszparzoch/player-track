#include "player_tracker.hpp"

bool PlayerTracker::init(const cv::Mat& frame, const cv::Rect2d& bbox) {
  if (frame.empty() || bbox.width <= 1.0 || bbox.height <= 1.0) {
    return false;
  }

  tracker_ = cv::TrackerCSRT::create();
  tracker_->init(frame, cv::Rect(bbox));
  return true;
}

std::optional<cv::Rect2d> PlayerTracker::update(const cv::Mat& frame) {
  if (!tracker_ || frame.empty()) {
    return std::nullopt;
  }

  cv::Rect bbox;
  if (!tracker_->update(frame, bbox)) {
    return std::nullopt;
  }
  if (bbox.width <= 1 || bbox.height <= 1) {
    return std::nullopt;
  }
  return cv::Rect2d(bbox);
}

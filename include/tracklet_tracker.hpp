#pragma once

#include <algorithm>
#include <vector>

#include <opencv2/core.hpp>

#include "detection_types.hpp"

struct Tracklet {
  int id{};
  cv::Rect2d bbox;
  int age{0};
  int time_since_update{0};
  bool active{true};
  float max_confidence{0.f};
  int detect_hits{0};
};

class TrackletTracker {
 public:
  void set_detect_frame_step(int step) { max_visible_miss_ = std::max(1, step); }

  std::vector<Tracklet> update(const std::vector<Detection>& detections);
  std::vector<Tracklet> predict();

 private:
  int next_id_{1};
  int max_visible_miss_{2};
  std::vector<Tracklet> tracks_;

  static double iou(const cv::Rect2d& a, const cv::Rect2d& b);
  std::vector<Tracklet> active_tracks() const;
};

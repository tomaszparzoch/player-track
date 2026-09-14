#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "detection_types.hpp"

struct Tracklet {
  int id{};
  cv::Rect2d bbox;
  int age{0};
  int time_since_update{0};
  bool active{true};
};

class TrackletTracker {
 public:
  std::vector<Tracklet> update(const std::vector<Detection>& detections);

 private:
  int next_id_{1};
  std::vector<Tracklet> tracks_;

  static double iou(const cv::Rect2d& a, const cv::Rect2d& b);
};

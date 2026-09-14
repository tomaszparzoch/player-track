#pragma once

#include <vector>

#include <opencv2/core.hpp>

struct Detection {
  cv::Rect2d box;
  float confidence{};
  int class_id{};
  int stable_id{0};
};

struct DetectionScanFrame {
  int frame_index{};
  cv::Mat image;
  std::vector<Detection> detections;
};

struct DetectionScanResult {
  std::vector<Detection> merged;
  std::vector<DetectionScanFrame> frames;
  int frames_scanned{};
};

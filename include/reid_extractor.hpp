#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "global_track.hpp"

struct ReIdFeatures {
  std::vector<float> histogram;
  cv::Scalar mean_color;
};

class ReIdExtractor {
 public:
  ReIdFeatures extract(const cv::Mat& frame, const cv::Rect2d& bbox) const;
  float similarity(const ReIdFeatures& a, const ReIdFeatures& b) const;
  TeamId classify_team(const ReIdFeatures& features);

 private:
  cv::Scalar team_a_color_{0, 0, 255};
  cv::Scalar team_b_color_{255, 255, 255};
  bool teams_initialized_{false};
  bool team_b_initialized_{false};
};

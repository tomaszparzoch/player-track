#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "global_track.hpp"

struct ReIdFeatures {
  std::vector<float> histogram;
  cv::Scalar mean_color;
  float green_bib_ratio{0.f};
};

class ReIdExtractor {
 public:
  void set_hall_mode(bool enabled) { hall_mode_ = enabled; }
  ReIdFeatures extract(const cv::Mat& frame, const cv::Rect2d& bbox) const;
  float similarity(const ReIdFeatures& a, const ReIdFeatures& b) const;
  TeamId classify_team(const ReIdFeatures& features);

 private:
  static float measure_green_bib_ratio(const cv::Mat& torso_bgr);

  cv::Scalar team_a_color_{0, 0, 255};
  cv::Scalar team_b_color_{255, 255, 255};
  bool teams_initialized_{false};
  bool team_b_initialized_{false};
  bool hall_mode_{false};
  float green_bib_threshold_{0.10f};
};

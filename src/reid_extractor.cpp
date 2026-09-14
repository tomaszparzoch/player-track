#include "reid_extractor.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>

ReIdFeatures ReIdExtractor::extract(const cv::Mat& frame, const cv::Rect2d& bbox) const {
  ReIdFeatures feat;
  const cv::Rect rect(bbox);
  if (rect.width <= 0 || rect.height <= 0) return feat;

  const cv::Rect torso(rect.x, rect.y, rect.width, std::max(1, rect.height / 2));
  const cv::Rect clipped = torso & cv::Rect(0, 0, frame.cols, frame.rows);
  if (clipped.area() <= 0) return feat;

  cv::Mat crop = frame(clipped);
  cv::Mat hsv;
  cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);

  const int h_bins = 16;
  const int s_bins = 8;
  const int hist_size[] = {h_bins, s_bins};
  const float h_ranges[] = {0, 180};
  const float s_ranges[] = {0, 256};
  const float* ranges[] = {h_ranges, s_ranges};
  const int channels[] = {0, 1};

  cv::Mat hist;
  cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, hist_size, ranges, true, false);
  cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);

  feat.histogram.resize(h_bins * s_bins);
  for (int i = 0; i < h_bins; ++i) {
    for (int j = 0; j < s_bins; ++j) {
      feat.histogram[i * s_bins + j] = hist.at<float>(i, j);
    }
  }

  feat.mean_color = cv::mean(crop);
  return feat;
}

float ReIdExtractor::similarity(const ReIdFeatures& a, const ReIdFeatures& b) const {
  if (a.histogram.empty() || b.histogram.empty()) return 0.0f;
  float sum = 0.0f;
  for (size_t i = 0; i < a.histogram.size(); ++i) {
    sum += std::sqrt(a.histogram[i] * b.histogram[i]);
  }
  return sum;
}

TeamId ReIdExtractor::classify_team(const ReIdFeatures& features) {
  if (!teams_initialized_) {
    team_a_color_ = features.mean_color;
    teams_initialized_ = true;
    return TeamId::TeamA;
  }
  const double dist_a = cv::norm(features.mean_color - team_a_color_);
  if (!team_b_initialized_) {
    if (dist_a > 40.0) {
      team_b_color_ = features.mean_color;
      team_b_initialized_ = true;
      return TeamId::TeamB;
    }
    return TeamId::TeamA;
  }
  const double dist_b = cv::norm(features.mean_color - team_b_color_);
  return dist_a <= dist_b ? TeamId::TeamA : TeamId::TeamB;
}

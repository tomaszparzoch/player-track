#include "reid_extractor.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace {

constexpr int kHBins = 16;
constexpr int kSBins = 8;

}  // namespace

float ReIdExtractor::measure_green_bib_ratio(const cv::Mat& torso_bgr) {
  if (torso_bgr.empty()) return 0.f;

  cv::Mat hsv;
  cv::cvtColor(torso_bgr, hsv, cv::COLOR_BGR2HSV);

  const int bib_y = std::max(0, torso_bgr.rows / 6);
  const int bib_h = std::max(1, torso_bgr.rows / 3);
  const cv::Rect bib_zone(0, bib_y, torso_bgr.cols, std::min(bib_h, torso_bgr.rows - bib_y));
  const cv::Mat bib_hsv = hsv(bib_zone);

  cv::Mat green_mask;
  cv::inRange(bib_hsv, cv::Scalar(35, 45, 45), cv::Scalar(92, 255, 255), green_mask);

  cv::Mat yellow_green_mask;
  cv::inRange(bib_hsv, cv::Scalar(25, 35, 45), cv::Scalar(34, 255, 255), yellow_green_mask);
  cv::bitwise_or(green_mask, yellow_green_mask, green_mask);

  const int green_px = cv::countNonZero(green_mask);
  const int total_px = bib_zone.area();
  if (total_px <= 0) return 0.f;
  return static_cast<float>(green_px) / static_cast<float>(total_px);
}

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

  const int hist_size[] = {kHBins, kSBins};
  const float h_ranges[] = {0, 180};
  const float s_ranges[] = {0, 256};
  const float* ranges[] = {h_ranges, s_ranges};
  const int channels[] = {0, 1};

  cv::Mat hist;
  cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, hist_size, ranges, true, false);
  cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);

  feat.histogram.resize(kHBins * kSBins);
  for (int i = 0; i < kHBins; ++i) {
    for (int j = 0; j < kSBins; ++j) {
      feat.histogram[i * kSBins + j] = hist.at<float>(i, j);
    }
  }

  feat.mean_color = cv::mean(crop);
  feat.green_bib_ratio = measure_green_bib_ratio(crop);
  return feat;
}

float ReIdExtractor::similarity(const ReIdFeatures& a, const ReIdFeatures& b) const {
  if (a.histogram.empty() || b.histogram.empty()) return 0.0f;
  float sum = 0.0f;
  for (size_t i = 0; i < a.histogram.size(); ++i) {
    sum += std::sqrt(a.histogram[i] * b.histogram[i]);
  }

  if (hall_mode_) {
    const float bib_diff = std::abs(a.green_bib_ratio - b.green_bib_ratio);
    if (bib_diff > 0.12f) {
      sum *= 0.15f;
    } else if (bib_diff > 0.06f) {
      sum *= 0.5f;
    }
  }
  return sum;
}

TeamId ReIdExtractor::classify_team(const ReIdFeatures& features) {
  if (hall_mode_) {
    return features.green_bib_ratio >= green_bib_threshold_ ? TeamId::TeamA : TeamId::TeamB;
  }

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

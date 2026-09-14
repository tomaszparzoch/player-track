#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "pitch_model.hpp"

class PitchCalibration {
 public:
  bool load(const std::filesystem::path& path);
  bool save(const std::filesystem::path& path) const;
  bool calibrate_interactive(const cv::Mat& frame, double pitch_length_m, double pitch_width_m);
  bool calibrate_lines_interactive(const cv::Mat& frame, const PitchModel& model);
  bool valid() const { return valid_; }

  cv::Point2d image_to_meters(double x_px, double y_px) const;
  cv::Point2f meters_to_image(double x_m, double y_m) const;
  double reprojection_error_px() const { return reproj_error_px_; }
  double pitch_length_m() const { return pitch_length_m_; }
  double pitch_width_m() const { return pitch_width_m_; }

 private:
  bool build_homography_rect(const std::vector<cv::Point2f>& image_pts, double pitch_length_m,
                             double pitch_width_m);
  bool build_homography_points(const std::vector<cv::Point2f>& image_pts,
                               const std::vector<cv::Point2f>& field_pts, double pitch_length_m,
                               double pitch_width_m);

  cv::Mat homography_;
  double pitch_length_m_{0.0};
  double pitch_width_m_{0.0};
  std::vector<cv::Point2f> image_points_;
  std::vector<cv::Point2f> field_points_;
  double reproj_error_px_{0.0};
  bool valid_{false};
};

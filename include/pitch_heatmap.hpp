#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

bool write_pitch_heatmap_png(const std::filesystem::path& path,
                             const std::vector<cv::Point2d>& positions_m, double pitch_length_m,
                             double pitch_width_m, int width_px = 1050, int height_px = 680);

bool write_image_heatmap_png(const std::filesystem::path& path,
                             const std::vector<cv::Point2d>& positions_px, int frame_width_px,
                             int frame_height_px);

struct PitchEventMark {
  cv::Point2d pos;
  bool has_pos{false};
  cv::Scalar color{0, 255, 255};
  std::string label;
};

bool write_pitch_event_diagram_png(const std::filesystem::path& path,
                                   const std::vector<PitchEventMark>& marks,
                                   double pitch_length_m, double pitch_width_m,
                                   bool pixel_mode, int width_px = 480, int height_px = 310);

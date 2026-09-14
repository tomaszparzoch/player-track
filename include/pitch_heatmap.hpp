#pragma once

#include <filesystem>
#include <vector>

#include <opencv2/core.hpp>

bool write_pitch_heatmap_png(const std::filesystem::path& path,
                             const std::vector<cv::Point2d>& positions_m, double pitch_length_m,
                             double pitch_width_m, int width_px = 1050, int height_px = 680);

bool write_image_heatmap_png(const std::filesystem::path& path,
                             const std::vector<cv::Point2d>& positions_px, int frame_width_px,
                             int frame_height_px);

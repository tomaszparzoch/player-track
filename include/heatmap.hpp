#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct HeatmapConfig {
  int cell_size = 16;
  double sigma_cells = 2.0;
  int svg_width = 960;
  int svg_height = 540;
};

bool write_player_heatmap_svg(const std::filesystem::path& path, int player_id,
                              const std::vector<cv::Point2f>& positions, int frame_width,
                              int frame_height, const HeatmapConfig& config = {});

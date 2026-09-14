#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "player_stats.hpp"
#include "project_config.hpp"

using ProgressCallback = std::function<void(double progress, const std::string& message)>;

struct MultiCameraResult {
  std::vector<PlayerStatsReport> player_reports;
  AnalysisQuality quality;
  std::filesystem::path global_csv;
  std::filesystem::path pdf_report;
  std::vector<std::filesystem::path> annotated_videos;
  double analysis_duration_s{0.0};
  bool success{false};
  std::string error;
};

MultiCameraResult run_multi_camera_session(const ProjectConfig& config,
                                           ProgressCallback progress = {});

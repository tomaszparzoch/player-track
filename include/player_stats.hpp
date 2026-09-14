#pragma once

#include <filesystem>
#include <utility>
#include <vector>

#include "global_track.hpp"
#include "project_config.hpp"

struct PlayerStatsReport {
  int player_number{};
  TeamId team{TeamId::Unknown};
  double total_distance_m{};
  double avg_speed_m_s{};
  double max_speed_m_s{};
  int sprint_count{};
  double sprint_distance_m{};
  double active_time_s{};
  double active_time_pct{};
  double defensive_third_pct{};
  double middle_third_pct{};
  double attacking_third_pct{};
  double hi_run_count{};
  float identity_confidence{};
  bool pixel_coords{false};
  std::filesystem::path heatmap_png;
  std::vector<std::pair<double, double>> speed_series;
};

struct AnalysisQuality {
  ProjectMode project_mode{ProjectMode::StaticOnly};
  double identity_confidence_pct{};
  double calib_confidence_pct{};
  std::string disclaimer;
};

class PlayerStatsEngine {
 public:
  static std::vector<PlayerStatsReport> compute(const ProjectConfig& config,
                                                const std::vector<ResolvedPlayer>& players,
                                                double total_duration_s,
                                                const std::filesystem::path& output_dir,
                                                cv::Size heatmap_frame_size = {});

  static AnalysisQuality compute_quality(const ProjectConfig& config,
                                         const std::vector<ResolvedPlayer>& players,
                                         double calib_good_pct);
};

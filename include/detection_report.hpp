#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "project_config.hpp"

struct TrackletReportRow {
  std::string camera_name;
  int camera_index{};
  int local_track_id{};
  double t_start_s{};
  double t_end_s{};
  double duration_s{};
  int observation_count{};
  int jersey_number{-1};
  float jersey_confidence{};
  float max_detect_confidence{};
  float avg_detect_confidence{};
  double movement_px{};
  double max_speed_px_s{};
  double bbox_x{};
  double bbox_y{};
  double bbox_w{};
  double bbox_h{};
  bool kept{false};
  std::string reject_reason;
};

struct DetectionReportSummary {
  ProjectConfig config;
  std::string detector_provider;
  int frames_processed{};
  int detect_frames{};
  int raw_detections_total{};
  int tracklets_total{};
  int tracklets_kept{};
  int tracklets_rejected{};
  int resolved_players{};
  std::vector<TrackletReportRow> tracklets;
};

bool write_detection_report(const std::filesystem::path& txt_path,
                            const std::filesystem::path& csv_path,
                            const DetectionReportSummary& summary);

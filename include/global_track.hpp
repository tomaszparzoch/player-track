#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

enum class TeamId { Unknown, TeamA, TeamB };

struct TrackObservation {
  double t_s{};
  int camera_index{};
  int local_track_id{};
  cv::Rect2d bbox;
  cv::Point2d image_pt;
  cv::Point2d world_m;
  double speed_m_s{};
  double speed_px_s{};
  float detect_confidence{};
  float calib_confidence{1.0f};
  float identity_confidence{0.0f};
  bool has_world{false};
  int jersey_number{-1};
  TeamId team{TeamId::Unknown};
};

struct GlobalTrackPoint {
  double t_s{};
  int player_number{-1};
  TeamId team{TeamId::Unknown};
  double x_m{};
  double y_m{};
  double speed_m_s{};
  float confidence{};
  int camera_index{-1};
  bool pixel_coords{false};
};

struct ResolvedPlayer {
  int player_number{-1};
  TeamId team{TeamId::Unknown};
  std::vector<GlobalTrackPoint> trajectory;
  float identity_confidence{0.0f};
};

std::string team_to_string(TeamId team);

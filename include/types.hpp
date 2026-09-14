#pragma once

struct Sample {
  int player_id{};
  double t_s{};
  double x_px{};
  double y_px{};
  double vx_px_s{};
  double vy_px_s{};
  double speed_px_s{};
  double distance_px{};
  double x_m{};
  double y_m{};
  double vx_m_s{};
  double vy_m_s{};
  double speed_m_s{};
  double distance_m{};
  bool use_meters{false};
  bool simulated{false};
};

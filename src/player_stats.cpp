#include "player_stats.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>

#include "pitch_heatmap.hpp"

namespace {

double zone_pct(const std::vector<GlobalTrackPoint>& traj, double span, int zone) {
  if (traj.empty() || span <= 0.0) return 0.0;
  int count = 0;
  for (const auto& pt : traj) {
    const double third = span / 3.0;
    const int z = pt.x_m < third ? 0 : (pt.x_m < 2.0 * third ? 1 : 2);
    if (z == zone) ++count;
  }
  return 100.0 * count / traj.size();
}

double sprint_threshold(const ProjectConfig& config, bool pixel_mode) {
  return pixel_mode ? config.sprint_threshold_m_s * 35.0 : config.sprint_threshold_m_s;
}

double hi_run_threshold(const ProjectConfig& config, bool pixel_mode) {
  return pixel_mode ? config.hi_run_threshold_m_s * 35.0 : config.hi_run_threshold_m_s;
}

}  // namespace

std::vector<PlayerStatsReport> PlayerStatsEngine::compute(
    const ProjectConfig& config, const std::vector<ResolvedPlayer>& players,
    double total_duration_s, const std::filesystem::path& output_dir,
    cv::Size heatmap_frame_size) {
  std::vector<PlayerStatsReport> reports;
  const std::filesystem::path heatmap_dir = output_dir / "heatmaps";
  std::error_code ec;
  std::filesystem::create_directories(heatmap_dir, ec);

  for (const auto& player : players) {
    if (player.trajectory.size() < 2) continue;

    const bool pixel_mode =
        !player.trajectory.empty() && player.trajectory.front().pixel_coords;
    const double zone_span =
        pixel_mode ? static_cast<double>(std::max(1, heatmap_frame_size.width))
                   : config.pitch_length_m;

    PlayerStatsReport report;
    report.player_number = player.player_number;
    report.team = player.team;
    report.identity_confidence = player.identity_confidence;
    report.pixel_coords = pixel_mode;

    const double sprint_thresh = sprint_threshold(config, pixel_mode);
    const double hi_run_thresh = hi_run_threshold(config, pixel_mode);

    double last_t = player.trajectory.front().t_s;
    double last_x = player.trajectory.front().x_m;
    double last_y = player.trajectory.front().y_m;
    report.active_time_s = 0.0;

    bool in_sprint = false;
    double sprint_start_t = 0.0;
    double sprint_dist = 0.0;

    bool in_hi = false;
    double hi_start_t = 0.0;

    for (size_t i = 1; i < player.trajectory.size(); ++i) {
      const auto& pt = player.trajectory[i];
      const double dt = std::max(1e-4, pt.t_s - last_t);
      const double dist = std::hypot(pt.x_m - last_x, pt.y_m - last_y);
      if (pt.confidence >= 0.5f) {
        report.total_distance_m += dist;
        report.active_time_s += dt;
        report.max_speed_m_s = std::max(report.max_speed_m_s, dist / dt);
        report.speed_series.emplace_back(pt.t_s, dist / dt);

        if (dist / dt >= sprint_thresh) {
          if (!in_sprint) {
            in_sprint = true;
            sprint_start_t = pt.t_s;
            sprint_dist = 0.0;
          }
          sprint_dist += dist;
        } else if (in_sprint) {
          if (pt.t_s - sprint_start_t >= 1.0) {
            ++report.sprint_count;
            report.sprint_distance_m += sprint_dist;
          }
          in_sprint = false;
        }

        if (dist / dt >= hi_run_thresh) {
          if (!in_hi) {
            in_hi = true;
            hi_start_t = pt.t_s;
          }
        } else if (in_hi) {
          if (pt.t_s - hi_start_t >= 3.0) {
            report.hi_run_count += 1.0;
          }
          in_hi = false;
        }
      }
      last_t = pt.t_s;
      last_x = pt.x_m;
      last_y = pt.y_m;
    }

    if (report.active_time_s > 0.0) {
      report.avg_speed_m_s = report.total_distance_m / report.active_time_s;
    }
    if (total_duration_s > 0.0) {
      report.active_time_pct = 100.0 * report.active_time_s / total_duration_s;
    }

    report.defensive_third_pct = zone_pct(player.trajectory, zone_span, 0);
    report.middle_third_pct = zone_pct(player.trajectory, zone_span, 1);
    report.attacking_third_pct = zone_pct(player.trajectory, zone_span, 2);

    const std::filesystem::path heatmap_path =
        heatmap_dir / ("player_" + std::to_string(report.player_number) + ".png");
    std::vector<cv::Point2d> pts;
    for (const auto& pt : player.trajectory) {
      if (pt.confidence >= 0.5f) pts.emplace_back(pt.x_m, pt.y_m);
    }
    bool wrote_heatmap = false;
    if (pixel_mode) {
      wrote_heatmap = write_image_heatmap_png(
          heatmap_path, pts, std::max(1, heatmap_frame_size.width),
          std::max(1, heatmap_frame_size.height));
    } else {
      wrote_heatmap =
          write_pitch_heatmap_png(heatmap_path, pts, config.pitch_length_m, config.pitch_width_m);
    }
    if (wrote_heatmap) {
      report.heatmap_png = heatmap_path;
    }

    reports.push_back(std::move(report));
  }

  return reports;
}

AnalysisQuality PlayerStatsEngine::compute_quality(const ProjectConfig& config,
                                                   const std::vector<ResolvedPlayer>& players,
                                                   double calib_good_pct) {
  AnalysisQuality q;
  q.project_mode = config.mode();
  q.calib_confidence_pct = calib_good_pct;

  double sum = 0.0;
  int count = 0;
  for (const auto& p : players) {
    sum += p.identity_confidence;
    ++count;
  }
  q.identity_confidence_pct = count > 0 ? 100.0 * sum / count : 0.0;

  const bool pixel_only = std::all_of(config.cameras.begin(), config.cameras.end(),
                                      [](const CameraConfig& c) { return !c.world_coords; });
  if (pixel_only) {
    q.disclaimer =
        "Pixel mode: positions, distances, speeds and heatmaps use image pixels (no pitch "
        "calibration). Zone stats split the frame width into left/middle/right thirds.";
  } else if (q.project_mode == ProjectMode::TacticalOnly) {
    q.disclaimer =
        "Metrics derived from panning tactical camera only. Expected position error 2-4 m, "
        "distance error up to 20%. Use static corner cameras for higher accuracy.";
  }
  return q;
}

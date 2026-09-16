#include "player_stats.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

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

cv::Mat crop_event_image(const cv::Mat& frame, const cv::Rect2d& bbox) {
  if (frame.empty()) return {};
  cv::Rect box(bbox);
  const int pad_x = std::max(2, static_cast<int>(std::lround(box.width * 0.12)));
  const int pad_y = std::max(2, static_cast<int>(std::lround(box.height * 0.08)));
  box.x -= pad_x;
  box.y -= pad_y;
  box.width += pad_x * 2;
  box.height += pad_y * 2;
  box &= cv::Rect(0, 0, frame.cols, frame.rows);
  if (box.width < 12 || box.height < 20) return {};
  return frame(box).clone();
}

cv::Mat load_event_crop(const ProjectConfig& config, const PlayerEventPhoto& photo) {
  if (!photo.image.empty()) return photo.image;
  if (photo.camera_index < 0 ||
      photo.camera_index >= static_cast<int>(config.cameras.size()) || photo.t_s < 0.0) {
    return {};
  }
  const auto& cam = config.cameras[static_cast<size_t>(photo.camera_index)];
  cv::VideoCapture cap(cam.video_path);
  if (!cap.isOpened()) return {};
  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps < 1.0) fps = 25.0;
  const int frame_idx =
      std::max(0, static_cast<int>(std::lround((photo.t_s - cam.offset_s) * fps)));
  cap.set(cv::CAP_PROP_POS_FRAMES, frame_idx);
  cv::Mat frame;
  if (!cap.read(frame) || frame.empty()) return {};
  return crop_event_image(frame, photo.bbox);
}

bool save_event_jpg(const cv::Mat& image, const std::filesystem::path& path) {
  if (image.empty()) return false;
  cv::Mat snap = image;
  if (snap.cols > 180 || snap.rows > 240) {
    const double scale = std::min(180.0 / snap.cols, 240.0 / snap.rows);
    cv::resize(snap, snap, {}, scale, scale, cv::INTER_AREA);
  }
  return cv::imwrite(path.string(), snap, {cv::IMWRITE_JPEG_QUALITY, 72});
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
    report.jersey_number = player.jersey_number;
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
      const double src_w = std::max(1, heatmap_frame_size.width);
      const double src_h = std::max(1, heatmap_frame_size.height);
      std::vector<cv::Point2d> scaled;
      scaled.reserve(pts.size());
      for (const auto& pt : pts) {
        scaled.emplace_back(pt.x / src_w * 480.0, pt.y / src_h * 270.0);
      }
      wrote_heatmap = write_image_heatmap_png(heatmap_path, scaled, 480, 270);
    } else {
      wrote_heatmap = write_pitch_heatmap_png(heatmap_path, pts, config.pitch_length_m,
                                              config.pitch_width_m, 420, 272);
    }
    if (wrote_heatmap) {
      report.heatmap_png = heatmap_path;
    }

    const std::filesystem::path snapshot_dir = output_dir / "snapshots";
    std::filesystem::create_directories(snapshot_dir, ec);
    auto event_path = [&](const std::string& suffix) {
      return snapshot_dir /
             ("player_" + std::to_string(report.player_number) + "_" + suffix + ".jpg");
    };

    std::map<int, int> cam_slot;
    std::vector<PitchEventMark> marks;
    for (const auto& ev : player.event_photos) {
      if (ev.camera_index < 0) continue;
      if (!cam_slot.count(ev.camera_index) && static_cast<int>(cam_slot.size()) < 2) {
        cam_slot[ev.camera_index] = static_cast<int>(cam_slot.size()) + 1;
      }
      const auto slot_it = cam_slot.find(ev.camera_index);
      if (slot_it == cam_slot.end()) continue;
      const int slot = slot_it->second;
      const cv::Mat crop = load_event_crop(config, ev);
      PitchEventMark mark;
      mark.pos = ev.world_m;
      mark.has_pos = ev.t_s >= 0.0;
      const std::string slot_s = std::to_string(slot);
      if (ev.kind == PlayerEventKind::Detect) {
        mark.color = slot == 1 ? cv::Scalar(80, 220, 80) : cv::Scalar(80, 220, 220);
        mark.label = "D" + slot_s;
        const auto path = event_path("detect_c" + slot_s);
        if (save_event_jpg(crop, path)) {
          (slot == 1 ? report.detect_cam1_jpg : report.detect_cam2_jpg) = path;
        }
      } else if (ev.kind == PlayerEventKind::Lost) {
        mark.color = slot == 1 ? cv::Scalar(40, 40, 220) : cv::Scalar(40, 140, 255);
        mark.label = "L" + slot_s;
        const auto path = event_path("lost_c" + slot_s);
        if (save_event_jpg(crop, path)) {
          (slot == 1 ? report.lost_cam1_jpg : report.lost_cam2_jpg) = path;
        }
      } else {
        mark.color = cv::Scalar(0, 220, 255);
        mark.label = "J" + slot_s;
        const auto path = event_path("join_c" + slot_s);
        if (save_event_jpg(crop, path)) {
          (slot == 1 ? report.join_cam1_jpg : report.join_cam2_jpg) = path;
        }
      }
      marks.push_back(std::move(mark));
    }

    if (!report.detect_cam1_jpg.empty()) {
      report.snapshot_jpg = report.detect_cam1_jpg;
    } else if (!player.snapshot.empty()) {
      const auto path = event_path("detect");
      if (save_event_jpg(player.snapshot, path)) {
        report.snapshot_jpg = path;
        report.detect_cam1_jpg = path;
      }
    }

    const std::filesystem::path event_map_path =
        heatmap_dir / ("player_" + std::to_string(report.player_number) + "_events.png");
    const double span_x = pixel_mode
                              ? static_cast<double>(std::max(1, heatmap_frame_size.width))
                              : config.pitch_length_m;
    const double span_y = pixel_mode
                              ? static_cast<double>(std::max(1, heatmap_frame_size.height))
                              : config.pitch_width_m;
    if (write_pitch_event_diagram_png(event_map_path, marks, span_x, span_y, pixel_mode)) {
      report.event_map_png = event_map_path;
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

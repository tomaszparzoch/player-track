#include "tracking_session.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <optional>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "csv_writer.hpp"
#include "heatmap.hpp"
#include "kinematics.hpp"
#include "person_detector.hpp"
#include "pitch_calibration.hpp"
#include "player_tracker.hpp"
#include "selection_ui.hpp"
#include "types.hpp"
#include "path_utils.hpp"

namespace {

struct WorldMetrics {
  bool initialized{false};
  double last_x_m{0.0};
  double last_y_m{0.0};
  double last_t_s{0.0};
  double distance_m{0.0};
};

struct PlayerState {
  int id{};
  PlayerTracker tracker;
  KinematicsFilter kinematics;
  WorldMetrics world;
  std::vector<cv::Point> trail;
  std::vector<cv::Point2f> heatmap_points;
  cv::Scalar color;
  cv::Point2f anchor_centroid;
  cv::Rect2d last_bbox;
  bool has_last_bbox{false};
  bool active{true};
  bool simulating{false};
  int simulated_frames{0};
};

struct InitPlayers {
  std::vector<cv::Rect2d> boxes;
  std::vector<int> ids;
};

cv::Point2f centroid(const cv::Rect2d& bbox) {
  return {static_cast<float>(bbox.x + bbox.width * 0.5),
          static_cast<float>(bbox.y + bbox.height * 0.5)};
}

double centroid_distance(const cv::Rect2d& box, const cv::Point2f& anchor) {
  const cv::Point2f c = centroid(box);
  return std::hypot(c.x - anchor.x, c.y - anchor.y);
}

void reconcile_tracker_identity(const cv::Mat& frame, std::vector<PlayerState>& players,
                              std::vector<std::optional<cv::Rect2d>>& boxes,
                              const std::vector<cv::Point2f>& prev_centroids) {
  for (size_t i = 0; i < players.size(); ++i) {
    if (!players[i].active || !boxes[i]) {
      continue;
    }
    for (size_t j = i + 1; j < players.size(); ++j) {
      if (!players[j].active || !boxes[j]) {
        continue;
      }

      const double d_ii = centroid_distance(*boxes[i], prev_centroids[i]);
      const double d_ij = centroid_distance(*boxes[i], prev_centroids[j]);
      const double d_jj = centroid_distance(*boxes[j], prev_centroids[j]);
      const double d_ji = centroid_distance(*boxes[j], prev_centroids[i]);

      const bool i_on_j = d_ij < d_ii * 0.75;
      const bool j_on_i = d_ji < d_jj * 0.75;
      if (!i_on_j || !j_on_i) {
        continue;
      }

      std::swap(boxes[i], boxes[j]);
      if (players[i].tracker.init(frame, *boxes[i])) {
        players[i].last_bbox = *boxes[i];
      } else {
        players[i].active = false;
        boxes[i] = std::nullopt;
      }
      if (players[j].tracker.init(frame, *boxes[j])) {
        players[j].last_bbox = *boxes[j];
      } else {
        players[j].active = false;
        boxes[j] = std::nullopt;
      }
    }
  }
}

void resolve_overlapping_trackers(const cv::Mat& frame, std::vector<PlayerState>& players,
                                  std::vector<std::optional<cv::Rect2d>>& boxes) {
  for (size_t i = 0; i < players.size(); ++i) {
    if (!players[i].active || !boxes[i]) {
      continue;
    }
    for (size_t j = i + 1; j < players.size(); ++j) {
      if (!players[j].active || !boxes[j]) {
        continue;
      }
      if (detection_iou(*boxes[i], *boxes[j]) <= 0.45) {
        continue;
      }

      const double d_i_own = centroid_distance(*boxes[i], players[i].anchor_centroid);
      const double d_i_on_j = centroid_distance(*boxes[i], players[j].anchor_centroid);
      const double d_j_own = centroid_distance(*boxes[j], players[j].anchor_centroid);
      const double d_j_on_i = centroid_distance(*boxes[j], players[i].anchor_centroid);

      const bool i_on_j = d_i_on_j < d_i_own * 0.85;
      const bool j_on_i = d_j_on_i < d_j_own * 0.85;

      if (i_on_j && !j_on_i) {
        cv::Rect2d corrected = *boxes[j];
        if (players[i].tracker.init(frame, corrected)) {
          boxes[i] = corrected;
          players[i].last_bbox = corrected;
        } else {
          players[i].active = false;
          boxes[i] = std::nullopt;
        }
      } else if (j_on_i && !i_on_j) {
        cv::Rect2d corrected = *boxes[i];
        if (players[j].tracker.init(frame, corrected)) {
          boxes[j] = corrected;
          players[j].last_bbox = corrected;
        } else {
          players[j].active = false;
          boxes[j] = std::nullopt;
        }
      } else if (i_on_j && j_on_i) {
        if (d_i_own <= d_j_own) {
          players[j].active = false;
          boxes[j] = std::nullopt;
        } else {
          players[i].active = false;
          boxes[i] = std::nullopt;
        }
      }
    }
  }
}

bool valid_bbox(const cv::Rect2d& bbox) {
  return bbox.width > 1.0 && bbox.height > 1.0;
}

Sample apply_meters(const Sample& sample, WorldMetrics& world, const PitchCalibration& cal) {
  Sample out = sample;
  const cv::Point2d meters = cal.image_to_meters(sample.x_px, sample.y_px);
  out.x_m = meters.x;
  out.y_m = meters.y;
  out.use_meters = true;

  if (!world.initialized) {
    world.initialized = true;
    world.last_x_m = out.x_m;
    world.last_y_m = out.y_m;
    world.last_t_s = out.t_s;
    out.vx_m_s = out.vy_m_s = out.speed_m_s = out.distance_m = 0.0;
    return out;
  }

  const double dt = std::max(1e-4, out.t_s - world.last_t_s);
  out.vx_m_s = (out.x_m - world.last_x_m) / dt;
  out.vy_m_s = (out.y_m - world.last_y_m) / dt;
  out.speed_m_s = std::hypot(out.vx_m_s, out.vy_m_s);
  world.distance_m += std::hypot(out.x_m - world.last_x_m, out.y_m - world.last_y_m);
  out.distance_m = world.distance_m;
  world.last_x_m = out.x_m;
  world.last_y_m = out.y_m;
  world.last_t_s = out.t_s;
  return out;
}

void draw_dashed_line(cv::Mat& frame, cv::Point a, cv::Point b, const cv::Scalar& color,
                      int thickness, int dash_len = 8) {
  const double dx = b.x - a.x;
  const double dy = b.y - a.y;
  const double len = std::hypot(dx, dy);
  if (len < 1.0) {
    return;
  }
  const int steps = static_cast<int>(len / dash_len);
  for (int i = 0; i < steps; i += 2) {
    const double t0 = static_cast<double>(i) / steps;
    const double t1 = static_cast<double>(std::min(i + 1, steps)) / steps;
    cv::line(frame, {cvRound(a.x + dx * t0), cvRound(a.y + dy * t0)},
             {cvRound(a.x + dx * t1), cvRound(a.y + dy * t1)}, color, thickness, cv::LINE_AA);
  }
}

void draw_dashed_rect(cv::Mat& frame, const cv::Rect& box, const cv::Scalar& color,
                      int thickness) {
  const cv::Point tl = box.tl();
  const cv::Point tr{box.x + box.width, box.y};
  const cv::Point bl{box.x, box.y + box.height};
  const cv::Point br = box.br();
  draw_dashed_line(frame, tl, tr, color, thickness);
  draw_dashed_line(frame, tr, br, color, thickness);
  draw_dashed_line(frame, br, bl, color, thickness);
  draw_dashed_line(frame, bl, tl, color, thickness);
}

cv::Rect2d bbox_at_centroid(const cv::Rect2d& reference, double x, double y) {
  cv::Rect2d box = reference;
  if (box.width < 2.0 || box.height < 2.0) {
    box = {x - 20.0, y - 40.0, 40.0, 80.0};
  }
  box.x = x - box.width * 0.5;
  box.y = y - box.height * 0.5;
  return box;
}

void draw_player_overlay(cv::Mat& frame, const PlayerState& player, const Sample& sample,
                         const cv::Rect2d& bbox, bool use_meters, bool simulated = false) {
  const cv::Rect box(bbox);
  const cv::Scalar draw_color =
      simulated ? cv::Scalar(player.color[0] * 0.55, player.color[1] * 0.55, player.color[2] * 0.55)
                : player.color;
  if (simulated) {
    draw_dashed_rect(frame, box, draw_color, 2);
  } else {
    cv::rectangle(frame, box, draw_color, 2);
  }
  const cv::Point center(cvRound(sample.x_px), cvRound(sample.y_px));
  cv::circle(frame, center, simulated ? 3 : 4, draw_color, simulated ? 1 : cv::FILLED);
  if (player.trail.size() >= 2) {
    const int thickness = simulated ? 1 : 2;
    cv::polylines(frame, player.trail, false, draw_color, thickness, cv::LINE_AA);
  }

  char label[96];
  if (use_meters) {
    std::snprintf(label, sizeof(label), "P%d%s v=%.2f d=%.1f m", player.id,
                  simulated ? "~" : "", sample.speed_m_s, sample.distance_m);
  } else {
    std::snprintf(label, sizeof(label), "P%d%s v=%.1f d=%.0f px", player.id, simulated ? "~" : "",
                  sample.speed_px_s, sample.distance_px);
  }
  cv::putText(frame, label, {box.x, std::max(20, box.y - 8)}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
              {255, 255, 255}, 2, cv::LINE_AA);
}

void begin_gap_simulation(PlayerState& player, const AppOptions& opts) {
  if (!opts.simulate_gap || !player.kinematics.is_initialized()) {
    return;
  }
  player.simulating = true;
  player.simulated_frames = 0;
}

void end_gap_simulation(PlayerState& player) {
  player.simulating = false;
  player.simulated_frames = 0;
}

std::vector<cv::Scalar> player_colors() {
  return {
      {0, 220, 0},
      {255, 140, 0},
      {255, 0, 255},
      {0, 220, 255},
      {255, 0, 0},
      {0, 128, 255},
      {255, 255, 0},
      {128, 0, 255},
  };
}

std::optional<InitPlayers> collect_initial_boxes(const cv::Mat& frame, cv::VideoCapture& cap,
                                                 const AppOptions& opts,
                                                 PersonDetector& detector) {
  DetectionScanResult scan;
    if (opts.detect_frames > 1) {
      const int min_hits = std::max(1, opts.detect_min_hits);
      scan = detect_persons_multi_frame(detector, cap, frame, opts.detect_frames,
                                        opts.detect_frame_step, opts.detect_confidence, min_hits);
    } else {
      DetectionScanFrame scan_frame;
      scan_frame.frame_index = 0;
      scan_frame.image = frame.clone();
      scan_frame.detections = detector.detect(frame, opts.detect_confidence);
      scan.frames.push_back(std::move(scan_frame));
      scan.merged = scan.frames.front().detections;
      scan.frames_scanned = 1;
      std::cerr << "Found " << scan.merged.size() << " person detection(s) on first frame\n";
    }

    assign_stable_detection_ids(scan.frames);

    if (opts.preview && !scan.frames.empty()) {
      show_detection_scan_preview(scan.frames);
    }

    const std::vector<Detection> candidates = build_tracking_init_candidates(scan);
    InitPlayers result;

    if (!opts.pick_detections.empty()) {
      const auto boxes = select_detections_by_index(candidates, opts.pick_detections);
      if (!boxes) {
        return std::nullopt;
      }
      result.boxes = *boxes;
    } else if (opts.interactive_select) {
      const auto boxes = select_detections_interactive(frame, candidates, opts.players);
      if (!boxes) {
        return std::nullopt;
      }
      result.boxes = *boxes;
    } else {
      const auto auto_picked = select_detections_spatial(candidates, opts.players);
      if (auto_picked.empty()) {
        std::cerr << "No person detections found after spatial filtering. Try --mode manual or "
                     "lower --detect-confidence\n";
        return std::nullopt;
      }
      std::cerr << "Auto-selected " << auto_picked.size() << " spatially separated player(s)\n";
      for (const Detection& det : auto_picked) {
        result.boxes.push_back(det.box);
        result.ids.push_back(det.stable_id > 0 ? det.stable_id
                                               : static_cast<int>(result.ids.size() + 1));
      }
      return result;
    }

    for (size_t i = 0; i < result.boxes.size(); ++i) {
      int id = static_cast<int>(i + 1);
      if (i < candidates.size()) {
        for (const Detection& det : candidates) {
          if (detection_iou(det.box, result.boxes[i]) > 0.9) {
            id = det.stable_id > 0 ? det.stable_id : id;
            break;
          }
        }
      }
      result.ids.push_back(id);
    }
    return result;
}

bool overlaps_any_box(const cv::Rect2d& box, const std::vector<std::optional<cv::Rect2d>>& boxes,
                      size_t skip_index) {
  for (size_t j = 0; j < boxes.size(); ++j) {
    if (j == skip_index || !boxes[j]) {
      continue;
    }
    if (detection_iou(box, *boxes[j]) > 0.35) {
      return true;
    }
  }
  return false;
}

bool overlaps_known_player(const cv::Rect2d& box, const std::vector<PlayerState>& players,
                           const std::vector<std::optional<cv::Rect2d>>& boxes,
                           size_t skip_index) {
  if (overlaps_any_box(box, boxes, skip_index)) {
    return true;
  }
  for (size_t j = 0; j < players.size(); ++j) {
    if (j == skip_index || !players[j].has_last_bbox) {
      continue;
    }
    if (detection_iou(box, players[j].last_bbox) > 0.30) {
      return true;
    }
  }
  return false;
}

size_t detection_index_near(const std::vector<Detection>& detections, const cv::Rect2d& box,
                            float min_iou) {
  size_t best_idx = detections.size();
  float best_iou = min_iou;
  for (size_t i = 0; i < detections.size(); ++i) {
    const float iou = static_cast<float>(detection_iou(detections[i].box, box));
    if (iou > best_iou) {
      best_iou = iou;
      best_idx = i;
    }
  }
  return best_idx;
}

bool reactivate_player_from_detections(PlayerState& player, cv::Rect2d& box, const cv::Mat& frame,
                                       const std::vector<Detection>& detections,
                                       const std::vector<std::optional<cv::Rect2d>>& current_boxes,
                                       size_t player_index, std::vector<bool>& det_used) {
  if (!player.has_last_bbox) {
    return false;
  }

  const auto match = find_detection_near_box(detections, player.last_bbox, 0.15f);
  if (!match) {
    cv::Rect2d search_box = player.last_bbox;
    search_box.x = player.anchor_centroid.x - search_box.width * 0.5;
    search_box.y = player.anchor_centroid.y - search_box.height * 0.5;
    const auto anchor_match = find_detection_near_box(detections, search_box, 0.12f);
    if (!anchor_match) {
      return false;
    }
    const size_t det_idx = detection_index_near(detections, anchor_match->box, 0.20f);
    if (det_idx < det_used.size() && det_used[det_idx]) {
      return false;
    }
    if (overlaps_any_box(anchor_match->box, current_boxes, player_index)) {
      return false;
    }
    if (!player.tracker.init(frame, anchor_match->box)) {
      return false;
    }
    box = anchor_match->box;
    player.last_bbox = anchor_match->box;
    player.has_last_bbox = true;
    player.active = true;
    end_gap_simulation(player);
    if (det_idx < det_used.size()) {
      det_used[det_idx] = true;
    }
    return true;
  }

  const size_t det_idx = detection_index_near(detections, match->box, 0.20f);
  if (det_idx < det_used.size() && det_used[det_idx]) {
    return false;
  }
  if (overlaps_any_box(match->box, current_boxes, player_index)) {
    return false;
  }
  if (!player.tracker.init(frame, match->box)) {
    return false;
  }
  box = match->box;
  player.last_bbox = match->box;
  player.has_last_bbox = true;
  player.active = true;
  end_gap_simulation(player);
  if (det_idx < det_used.size()) {
    det_used[det_idx] = true;
  }
  return true;
}

void discover_and_reactivate_players(const cv::Mat& frame, std::vector<PlayerState>& players,
                                     std::vector<std::optional<cv::Rect2d>>& boxes,
                                     PersonDetector& detector, const AppOptions& opts,
                                     int& next_player_id, const std::vector<cv::Scalar>& colors) {
  if (!opts.discover_players) {
    return;
  }

  const auto detections = detector.detect(frame, opts.detect_confidence);
  std::vector<bool> det_used(detections.size(), false);

  for (size_t i = 0; i < players.size(); ++i) {
    if (players[i].active || !players[i].has_last_bbox) {
      continue;
    }

    cv::Rect2d recovered_box;
    if (!reactivate_player_from_detections(players[i], recovered_box, frame, detections, boxes, i,
                                           det_used)) {
      continue;
    }
    boxes[i] = recovered_box;
    std::cerr << "Re-activated player " << players[i].id << "\n";
  }

  for (size_t d = 0; d < detections.size(); ++d) {
    if (det_used[d]) {
      continue;
    }
    if (static_cast<int>(players.size()) >= opts.players) {
      break;
    }

    const Detection& det = detections[d];
    if (overlaps_known_player(det.box, players, boxes, players.size())) {
      continue;
    }

    PlayerState state;
    const int new_id = next_player_id++;
    state.id = new_id;
    state.color = colors[(state.id - 1) % static_cast<int>(colors.size())];
    if (!state.tracker.init(frame, det.box)) {
      continue;
    }

    state.anchor_centroid = centroid(det.box);
    state.last_bbox = det.box;
    state.has_last_bbox = true;
    state.active = true;
    players.push_back(std::move(state));
    boxes.push_back(det.box);
    det_used[d] = true;
    std::cerr << "Discovered player " << new_id << " (conf=" << det.confidence << ")\n";
  }
}

bool try_redetect_player(PlayerState& player, cv::Rect2d& box, const cv::Mat& frame,
                        PersonDetector& detector, float confidence,
                        const std::vector<std::optional<cv::Rect2d>>& current_boxes,
                        size_t player_index,
                        const std::vector<cv::Rect2d>& reserved_boxes) {
  if (!player.has_last_bbox) {
    return false;
  }

  const auto detections = detector.detect(frame, confidence);
  cv::Rect2d search_box = player.last_bbox;
  const auto match = find_detection_near_box(detections, search_box, 0.15f);
  if (!match) {
    search_box.width = player.last_bbox.width;
    search_box.height = player.last_bbox.height;
    search_box.x = player.anchor_centroid.x - search_box.width * 0.5;
    search_box.y = player.anchor_centroid.y - search_box.height * 0.5;
    const auto anchor_match = find_detection_near_box(detections, search_box, 0.12f);
    if (!anchor_match) {
      return false;
    }
    if (overlaps_any_box(anchor_match->box, current_boxes, player_index)) {
      return false;
    }
    for (const cv::Rect2d& reserved : reserved_boxes) {
      if (detection_iou(anchor_match->box, reserved) > 0.35) {
        return false;
      }
    }
    if (!player.tracker.init(frame, anchor_match->box)) {
      return false;
    }
    box = anchor_match->box;
    player.last_bbox = anchor_match->box;
    player.has_last_bbox = true;
    player.active = true;
    end_gap_simulation(player);
    std::cerr << "Re-detected player " << player.id << " near anchor (conf="
              << anchor_match->confidence << ")\n";
    return true;
  }

  if (overlaps_any_box(match->box, current_boxes, player_index)) {
    return false;
  }
  for (const cv::Rect2d& reserved : reserved_boxes) {
    if (detection_iou(match->box, reserved) > 0.35) {
      return false;
    }
  }

  if (!player.tracker.init(frame, match->box)) {
    return false;
  }

  box = match->box;
  player.last_bbox = match->box;
  player.has_last_bbox = true;
  player.active = true;
  end_gap_simulation(player);
  std::cerr << "Re-detected player " << player.id << " (conf=" << match->confidence << ")\n";
  return true;
}

}  // namespace

int run_tracking_session(const AppOptions& opts) {
  AppOptions resolved_opts = opts;
  resolved_opts.model_path = resolve_resource_path_string(opts.model_path);

  const bool needs_detector = resolved_opts.mode == TrackingMode::Auto || resolved_opts.redetect_on_loss ||
                              resolved_opts.discover_players || resolved_opts.simulate_gap;
  if (needs_detector && !std::filesystem::exists(resolved_opts.model_path)) {
    std::cerr << "ONNX model required at: " << resolved_opts.model_path << "\n"
              << "Run: powershell -File scripts\\download_model.ps1 -Model yolov8s\n";
    return 1;
  }

  std::optional<PersonDetector> detector;
  if (needs_detector) {
    PersonDetector loaded;
    if (!loaded.load(resolved_opts.model_path, resolved_opts.dnn_backend, resolved_opts.detect_input_size)) {
      return 1;
    }
    detector = std::move(loaded);
  }

  cv::VideoCapture cap(resolved_opts.input);
  if (!cap.isOpened()) {
    std::cerr << "Failed to open video: " << opts.input << "\n";
    return 1;
  }

  cv::Mat frame;
  if (!cap.read(frame) || frame.empty()) {
    std::cerr << "Failed to read the first frame\n";
    return 1;
  }
  const int frame_width = frame.cols;
  const int frame_height = frame.rows;

  PitchCalibration calibration;
  const bool use_meters = opts.calib_path.has_value();
  if (use_meters) {
    if (opts.calibrate) {
      if (!calibration.calibrate_interactive(frame, opts.pitch_width_m, opts.pitch_height_m)) {
        std::cerr << "Pitch calibration cancelled or failed\n";
        return 1;
      }
      if (!calibration.save(*opts.calib_path)) {
        std::cerr << "Failed to save calibration: " << *opts.calib_path << "\n";
        return 1;
      }
      std::cerr << "Saved calibration to " << *opts.calib_path << "\n";
    } else if (!calibration.load(*opts.calib_path)) {
      std::cerr << "Failed to load calibration: " << *opts.calib_path << "\n";
      return 1;
    }
  }

  std::optional<InitPlayers> initial_opt;
  if (opts.mode == TrackingMode::Auto) {
    initial_opt = collect_initial_boxes(frame, cap, opts, *detector);
  } else {
    InitPlayers manual;
    manual.boxes = collect_manual_boxes(frame, opts);
    for (size_t i = 0; i < manual.boxes.size(); ++i) {
      manual.ids.push_back(static_cast<int>(i + 1));
    }
    initial_opt = std::move(manual);
  }
  if (!initial_opt) {
    return 1;
  }
  const InitPlayers& initial = *initial_opt;
  if (initial.boxes.empty()) {
    std::cerr << "No players selected\n";
    return 1;
  }
  if (static_cast<int>(initial.boxes.size()) > opts.players) {
    std::cerr << "Too many players selected (max " << opts.players << ")\n";
    return 1;
  }
  for (size_t i = 0; i < initial.boxes.size(); ++i) {
    if (!valid_bbox(initial.boxes[i])) {
      const int pid = i < initial.ids.size() ? initial.ids[i] : static_cast<int>(i + 1);
      std::cerr << "Invalid bounding box for player " << pid << "\n";
      return 1;
    }
  }

  const int player_count = static_cast<int>(initial.boxes.size());
  const auto colors = player_colors();
  std::vector<PlayerState> players;
  players.reserve(player_count);
  for (int i = 0; i < player_count; ++i) {
    PlayerState state;
    state.id = initial.ids[i];
    state.color = colors[(state.id - 1) % colors.size()];
    if (!state.tracker.init(frame, initial.boxes[i])) {
      std::cerr << "Failed to initialize CSRT tracker for player " << state.id << "\n";
      return 1;
    }
    state.anchor_centroid = centroid(initial.boxes[i]);
    state.last_bbox = initial.boxes[i];
    state.has_last_bbox = true;
    players.push_back(std::move(state));
  }

  int next_player_id = 1;
  for (const PlayerState& player : players) {
    next_player_id = std::max(next_player_id, player.id + 1);
  }
  std::cerr << "Tracking " << players.size() << " player(s) initially";
  if (opts.discover_players && static_cast<int>(players.size()) < opts.players) {
    std::cerr << ", will discover up to " << opts.players << " when visible";
  }
  if (opts.simulate_gap) {
    std::cerr << ", gap simulation up to " << opts.simulate_gap_max_frames << " frames";
  }
  std::cerr << "\n";

  CsvWriter csv;
  if (!csv.open(opts.output, use_meters && calibration.valid())) {
    std::cerr << "Failed to open output CSV: " << opts.output << "\n";
    return 1;
  }

  const double fps = cap.get(cv::CAP_PROP_FPS);
  const double fallback_fps = fps > 1.0 ? fps : 30.0;

  cv::VideoWriter writer;
  if (opts.video_out) {
    const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    writer.open(*opts.video_out, fourcc, fallback_fps, frame.size());
    if (!writer.isOpened()) {
      std::cerr << "Failed to open video output: " << *opts.video_out << "\n";
      return 1;
    }
  }

  int frame_index = 0;
  int written = 0;
  bool have_t0 = false;
  double t0_s = 0.0;
  double last_pos_s = -1.0;

  auto tracked_count = [&]() {
    return static_cast<int>(std::count_if(players.begin(), players.end(),
                                          [](const PlayerState& p) {
                                            return p.active || p.simulating;
                                          }));
  };

  auto compute_time = [&]() {
    double t_s = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
    if (t_s <= last_pos_s + 1e-6) {
      t_s = static_cast<double>(frame_index) / fallback_fps;
    }
    last_pos_s = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
    if (!have_t0) {
      t0_s = t_s;
      have_t0 = true;
    }
    return t_s - t0_s;
  };

  auto process_frame = [&](cv::Mat& current,
                           const std::vector<std::optional<cv::Rect2d>>& boxes) -> bool {
    const double t_s = compute_time();

    for (size_t i = 0; i < players.size(); ++i) {
      if (players[i].simulating) {
        Sample sample = players[i].kinematics.predict(t_s);
        sample.player_id = players[i].id;
        if (use_meters && calibration.valid()) {
          sample = apply_meters(sample, players[i].world, calibration);
        }
        csv.write(sample);
        ++written;

        const cv::Rect2d sim_box =
            bbox_at_centroid(players[i].last_bbox, sample.x_px, sample.y_px);
        players[i].trail.emplace_back(cvRound(sample.x_px), cvRound(sample.y_px));
        players[i].heatmap_points.emplace_back(static_cast<float>(sample.x_px),
                                               static_cast<float>(sample.y_px));
        draw_player_overlay(current, players[i], sample, sim_box,
                            use_meters && calibration.valid(), true);

        ++players[i].simulated_frames;
        if (players[i].simulated_frames >= opts.simulate_gap_max_frames) {
          players[i].simulating = false;
          std::cerr << "Stopped gap simulation for player " << players[i].id
                    << " (max " << opts.simulate_gap_max_frames << " frames)\n";
        }
        continue;
      }

      if (!players[i].active || !boxes[i]) {
        continue;
      }

      players[i].last_bbox = *boxes[i];
      players[i].has_last_bbox = true;

      Sample sample = players[i].kinematics.update(t_s, centroid(*boxes[i]));
      sample.player_id = players[i].id;
      if (use_meters && calibration.valid()) {
        sample = apply_meters(sample, players[i].world, calibration);
      }
      csv.write(sample);
      ++written;

      players[i].trail.emplace_back(cvRound(sample.x_px), cvRound(sample.y_px));
      players[i].heatmap_points.emplace_back(static_cast<float>(sample.x_px),
                                             static_cast<float>(sample.y_px));
      draw_player_overlay(current, players[i], sample, *boxes[i],
                          use_meters && calibration.valid());
    }

    if (writer.isOpened()) {
      writer.write(current);
    }
    if (opts.preview) {
      cv::imshow("player_track", current);
      const int key = cv::waitKey(1);
      if (key == 'q' || key == 'Q' || key == 27) {
        return false;
      }
    }
    return true;
  };

  std::vector<std::optional<cv::Rect2d>> current_boxes;
  current_boxes.reserve(players.size());
  for (size_t i = 0; i < players.size(); ++i) {
    current_boxes.push_back(initial.boxes[i]);
  }

  if (!process_frame(frame, current_boxes)) {
    std::cerr << "Wrote " << written << " samples to " << opts.output << "\n";
    return 0;
  }

  while (cap.read(frame) && !frame.empty()) {
    ++frame_index;

    std::vector<cv::Point2f> prev_centroids(players.size());
    for (size_t i = 0; i < players.size(); ++i) {
      prev_centroids[i] =
          players[i].has_last_bbox ? centroid(players[i].last_bbox) : players[i].anchor_centroid;
    }

    std::vector<cv::Rect2d> redetected_boxes;
    if (detector && frame_index % opts.discover_interval == 0) {
      const auto detections = detector->detect(frame, opts.detect_confidence);
      std::vector<bool> det_used(detections.size(), false);
      for (size_t i = 0; i < players.size(); ++i) {
        if (!players[i].simulating || !players[i].has_last_bbox) {
          continue;
        }
        cv::Rect2d recovered_box;
        if (!reactivate_player_from_detections(players[i], recovered_box, frame, detections,
                                               current_boxes, i, det_used)) {
          continue;
        }
        current_boxes[i] = recovered_box;
        redetected_boxes.push_back(recovered_box);
        std::cerr << "Re-detected player " << players[i].id << " after gap simulation\n";
      }
    }

    for (size_t i = 0; i < players.size(); ++i) {
      if (players[i].simulating) {
        current_boxes[i] = std::nullopt;
        continue;
      }

      if (!players[i].active) {
        current_boxes[i] = std::nullopt;
        continue;
      }
      current_boxes[i] = players[i].tracker.update(frame);
      if (current_boxes[i]) {
        players[i].last_bbox = *current_boxes[i];
        players[i].has_last_bbox = true;
      } else {
        cv::Rect2d recovered_box;
        if (opts.redetect_on_loss && detector &&
            try_redetect_player(players[i], recovered_box, frame, *detector,
                                opts.detect_confidence, current_boxes, i, redetected_boxes)) {
          current_boxes[i] = recovered_box;
          redetected_boxes.push_back(recovered_box);
          continue;
        }
        current_boxes[i] = std::nullopt;
        players[i].active = false;
        begin_gap_simulation(players[i], opts);
        std::cerr << "Tracker lost player " << players[i].id << " at t ≈ "
                  << (cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0) << " s.";
        if (players[i].simulating) {
          std::cerr << " Simulating trajectory until re-detect.";
        }
        std::cerr << "\n";
      }
    }

    reconcile_tracker_identity(frame, players, current_boxes, prev_centroids);
    resolve_overlapping_trackers(frame, players, current_boxes);

    if (opts.discover_players && detector && opts.discover_interval > 0 &&
        frame_index % opts.discover_interval == 0) {
      discover_and_reactivate_players(frame, players, current_boxes, *detector, opts,
                                      next_player_id, colors);
    }

    if (tracked_count() == 0 && !opts.discover_players) {
      break;
    }

    if (!process_frame(frame, current_boxes)) {
      break;
    }
  }

  std::cerr << "Wrote " << written << " samples to " << opts.output << "\n";
  for (const auto& player : players) {
    if (use_meters && calibration.valid()) {
      std::cerr << "Player " << player.id << " total distance: " << player.world.distance_m
                << " m\n";
    } else {
      std::cerr << "Player " << player.id << " total distance: "
                << player.kinematics.distance_px() << " px\n";
    }
  }

  std::filesystem::path heatmap_dir = opts.heatmap_dir.has_value()
                                           ? std::filesystem::path(*opts.heatmap_dir)
                                           : std::filesystem::path(opts.output).parent_path();
  if (heatmap_dir.empty()) {
    heatmap_dir = ".";
  }
  std::error_code ec;
  std::filesystem::create_directories(heatmap_dir, ec);

  HeatmapConfig heatmap_cfg;
  heatmap_cfg.svg_width = frame_width;
  heatmap_cfg.svg_height = frame_height;

  for (const auto& player : players) {
    if (player.heatmap_points.empty()) {
      std::cerr << "Skipped heatmap for player " << player.id << " (no samples)\n";
      continue;
    }
    const std::filesystem::path heatmap_path =
        heatmap_dir / ("player" + std::to_string(player.id) + "_heatmap.svg");
    if (write_player_heatmap_svg(heatmap_path, player.id, player.heatmap_points, frame_width,
                                 frame_height, heatmap_cfg)) {
      std::cerr << "Wrote heatmap: " << heatmap_path << "\n";
    } else {
      std::cerr << "Failed to write heatmap for player " << player.id << "\n";
    }
  }

  return 0;
}

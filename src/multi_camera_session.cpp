#include "multi_camera_session.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "detection_report.hpp"
#include "dynamic_pitch_calibrator.hpp"
#include "camera_sidecar.hpp"
#include "identity_graph.hpp"
#include "jersey_ocr.hpp"
#include "pdf_report.hpp"
#include "person_detector.hpp"
#include "pitch_calibration.hpp"
#include "reid_extractor.hpp"
#include "tracklet_tracker.hpp"
#include "path_utils.hpp"
#include "detection_types.hpp"
#include "video_compress.hpp"

namespace {

cv::Point2d foot_point(const cv::Rect2d& bbox) {
  return {bbox.x + bbox.width * 0.5, bbox.y + bbox.height * 0.95};
}

cv::Mat crop_detection_snapshot(const cv::Mat& frame, const cv::Rect2d& bbox) {
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

int analyze_frame_limit(double analyze_max_seconds, double fps) {
  if (analyze_max_seconds <= 0.0) return 0;
  const double use_fps = fps > 1.0 ? fps : 25.0;
  return static_cast<int>(std::ceil(analyze_max_seconds * use_fps));
}

double analyzed_duration(double video_duration_s, double analyze_max_seconds) {
  if (analyze_max_seconds <= 0.0) return video_duration_s;
  return std::min(video_duration_s, analyze_max_seconds);
}

void draw_label_block(cv::Mat& frame, cv::Point origin, const std::vector<std::string>& lines) {
  if (lines.empty()) return;

  constexpr double kFontScale = 0.45;
  constexpr int kThickness = 1;
  constexpr int kLineGap = 4;
  constexpr int kPadX = 4;
  constexpr int kPadY = 3;
  int max_width = 0;
  int total_height = 0;
  for (const auto& line : lines) {
    int baseline = 0;
    const cv::Size size =
        cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, kFontScale, kThickness, &baseline);
    max_width = std::max(max_width, size.width);
    total_height += size.height + baseline + kLineGap;
  }
  total_height -= kLineGap;

  const cv::Rect bg(origin.x, origin.y, max_width + 2 * kPadX, total_height + 2 * kPadY);
  const cv::Rect clipped = bg & cv::Rect(0, 0, frame.cols, frame.rows);
  if (clipped.width > 0 && clipped.height > 0) {
    cv::rectangle(frame, clipped, {0, 0, 0}, cv::FILLED);
  }

  int y = origin.y + kPadY;
  for (const auto& line : lines) {
    int baseline = 0;
    const cv::Size size =
        cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, kFontScale, kThickness, &baseline);
    y += size.height;
    cv::putText(frame, line, {origin.x + kPadX, y}, cv::FONT_HERSHEY_SIMPLEX, kFontScale,
                {255, 255, 255}, kThickness, cv::LINE_AA);
    y += baseline + kLineGap;
  }
}

void draw_overlay(cv::Mat& frame, const Tracklet& track, int jersey, float jersey_conf,
                  TeamId team, float marker_ratio, float detect_conf, bool confirmed,
                  bool hall_mode) {
  const cv::Rect box(track.bbox);
  const cv::Scalar color =
      confirmed ? (team == TeamId::TeamB ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0))
                : cv::Scalar(128, 128, 128);
  const int thickness = confirmed ? 2 : 1;
  cv::rectangle(frame, box, color, thickness);

  char line1[32];
  char line2[48];
  char line3[32];
  std::snprintf(line1, sizeof(line1), "Track T%d", track.id);
  if (hall_mode) {
    if (team == TeamId::TeamA) {
      std::snprintf(line2, sizeof(line2), "Znacznik: TAK  %.0f%%", marker_ratio * 100.0f);
    } else if (team == TeamId::TeamB) {
      std::snprintf(line2, sizeof(line2), "Znacznik: NIE");
    } else {
      std::snprintf(line2, sizeof(line2), "Znacznik: ?");
    }
  } else if (jersey > 0) {
    std::snprintf(line2, sizeof(line2), "Jersey #%d  ocr:%.0f%%", jersey, jersey_conf * 100.0f);
  } else {
    std::snprintf(line2, sizeof(line2), "Jersey #--  ocr:--");
  }
  std::snprintf(line3, sizeof(line3), "%s  det:%.0f%%", team_display_name(team, hall_mode).c_str(),
                detect_conf * 100.0f);

  const std::vector<std::string> lines = {line1, line2, line3};
  int block_height = 0;
  for (const auto& line : lines) {
    int baseline = 0;
    const cv::Size size = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
    block_height += size.height + baseline + 4;
  }
  block_height -= 4;

  int label_y = box.y + box.height + 4;
  if (label_y + block_height + 6 > frame.rows) {
    label_y = std::max(0, box.y - block_height - 6);
  }
  draw_label_block(frame, {box.x, label_y}, lines);
}

struct CachedOverlay {
  cv::Rect2d bbox;
  int local_track_id{};
  int camera_index{};
  float detect_conf{};
  TeamId team{TeamId::Unknown};
  float marker_ratio{};
};

void draw_global_overlay(cv::Mat& frame, const CachedOverlay& overlay,
                         const TrackletGlobalIdentity& identity, bool hall_mode) {
  const cv::Rect box(overlay.bbox);
  const TeamId team = identity.team != TeamId::Unknown ? identity.team : overlay.team;
  const cv::Scalar color =
      team == TeamId::TeamB ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0);
  cv::rectangle(frame, box, color, 2);

  char line1[32];
  char line2[48];
  char line3[32];
  std::snprintf(line1, sizeof(line1), "Player #%d", identity.player_number);
  if (hall_mode) {
    if (overlay.marker_ratio >= 0.10f) {
      std::snprintf(line2, sizeof(line2), "Znacznik: TAK  %.0f%%", overlay.marker_ratio * 100.0f);
    } else {
      std::snprintf(line2, sizeof(line2), "Znacznik: NIE  %.0f%%", overlay.marker_ratio * 100.0f);
    }
  } else if (identity.jersey_number > 0) {
    std::snprintf(line2, sizeof(line2), "Jersey #%d  id:%.0f%%", identity.jersey_number,
                  identity.identity_confidence * 100.0f);
  } else {
    std::snprintf(line2, sizeof(line2), "Jersey #--  id:--");
  }
  std::snprintf(line3, sizeof(line3), "%s  det:%.0f%%", team_display_name(team, hall_mode).c_str(),
                overlay.detect_conf * 100.0f);

  const std::vector<std::string> lines = {line1, line2, line3};
  int block_height = 0;
  for (const auto& line : lines) {
    int baseline = 0;
    const cv::Size size = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
    block_height += size.height + baseline + 4;
  }
  block_height -= 4;

  int label_y = box.y + box.height + 4;
  if (label_y + block_height + 6 > frame.rows) {
    label_y = std::max(0, box.y - block_height - 6);
  }
  draw_label_block(frame, {box.x, label_y}, lines);
}

struct CameraRenderCache {
  std::filesystem::path video_path;
  std::filesystem::path out_video;
  double fps{25.0};
  double offset_s{0.0};
  double output_duration_s{0.0};
  int max_frames{0};
  cv::Size export_size;
  std::vector<std::vector<CachedOverlay>> frame_overlays;
};

bool open_annotated_video_writer(cv::VideoWriter& writer, const std::filesystem::path& out_video,
                                 double fps, const cv::Size& frame_size,
                                 std::filesystem::path* resolved_path);

bool render_cached_annotated_video(const CameraRenderCache& cache, int camera_index,
                                   const IdentityResolution& identities, bool hall_mode,
                                   std::filesystem::path* actual_out_video) {
  cv::VideoCapture cap(cache.video_path.string());
  if (!cap.isOpened()) return false;

  cv::VideoWriter writer;
  std::filesystem::path out_path = cache.out_video;
  const cv::Size writer_size = cache.export_size.width > 0
                                   ? cache.export_size
                                   : cv::Size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                                              static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
  if (!open_annotated_video_writer(writer, cache.out_video, cache.fps, writer_size, &out_path)) {
    return false;
  }
  if (actual_out_video) {
    *actual_out_video = out_path;
  }

  cv::Mat frame;
  int frame_idx = 0;
  const int frames_to_write =
      cache.max_frames > 0
          ? cache.max_frames
          : static_cast<int>(cache.frame_overlays.size());
  while (cap.read(frame) && !frame.empty()) {
    if (frame_idx >= frames_to_write) break;
    for (const auto& overlay : cache.frame_overlays[static_cast<size_t>(frame_idx)]) {
      const auto key = std::make_pair(camera_index, overlay.local_track_id);
      const auto it = identities.tracklet_identity.find(key);
      if (it != identities.tracklet_identity.end()) {
        draw_global_overlay(frame, overlay, it->second, hall_mode);
      }
    }

    if (cache.export_size.width > 0 &&
        (frame.cols != cache.export_size.width || frame.rows != cache.export_size.height)) {
      cv::Mat export_frame;
      cv::resize(frame, export_frame, cache.export_size, 0, 0, cv::INTER_AREA);
      writer.write(export_frame);
    } else {
      writer.write(frame);
    }
    ++frame_idx;
  }

  writer.release();
  return true;
}

struct TrackletMetrics {
  float max_detect_conf{0.f};
  float avg_detect_conf{0.f};
  double movement_px{0.0};
  double max_speed_px_s{0.0};
  double duration_s{0.0};
  size_t observation_count{0};
};

TrackletMetrics compute_tracklet_metrics(const TrackletNode& node) {
  TrackletMetrics m;
  m.observation_count = node.observations.size();
  m.duration_s = node.t_end - node.t_start;
  if (node.observations.empty()) return m;

  double conf_sum = 0.0;
  cv::Point2d prev = node.observations.front().image_pt;
  for (const auto& obs : node.observations) {
    m.max_detect_conf = std::max(m.max_detect_conf, obs.detect_confidence);
    conf_sum += obs.detect_confidence;
    m.max_speed_px_s = std::max(m.max_speed_px_s, obs.speed_px_s);
    m.movement_px += std::hypot(obs.image_pt.x - prev.x, obs.image_pt.y - prev.y);
    prev = obs.image_pt;
  }
  m.avg_detect_conf =
      static_cast<float>(conf_sum / static_cast<double>(node.observations.size()));
  return m;
}

std::string reject_tracklet_reason(const TrackletNode& node, const TrackletMetrics& metrics,
                                   float detect_confidence, bool hall_mode) {
  if (metrics.observation_count < 5) return "too_few_observations";
  if (metrics.duration_s < 0.8) return "too_short";
  if (hall_mode) {
    if (node.team != TeamId::Unknown && metrics.max_detect_conf >= detect_confidence * 0.75f) {
      return {};
    }
    if (metrics.movement_px < 12.0) return "stationary";
    if (metrics.max_detect_conf < detect_confidence * 0.75f) return "low_confidence";
    return {};
  }
  if (node.jersey_number > 0 && node.jersey_confidence >= 0.35f) return {};
  if (metrics.max_detect_conf < detect_confidence * 0.75f) return "low_confidence";
  if (metrics.movement_px < 12.0) return "stationary_no_jersey";
  return {};
}

bool show_track_in_video(const Tracklet& track, const TrackletNode& node, float detect_confidence,
                         bool hall_mode) {
  if (hall_mode) {
    return node.team != TeamId::Unknown &&
           track.max_confidence >= detect_confidence * 0.75f &&
           node.observations.size() >= 3;
  }
  if (node.jersey_number > 0) return true;
  if (track.detect_hits >= 3 && track.max_confidence >= detect_confidence * 0.85f) return true;
  if (node.observations.size() >= 12 && track.max_confidence >= detect_confidence * 0.80f) {
    return true;
  }
  return false;
}

TrackletReportRow make_tracklet_row(const TrackletNode& node, const std::string& camera_name,
                                    int camera_index, const TrackletMetrics& metrics,
                                    float detect_confidence, bool hall_mode) {
  TrackletReportRow row;
  row.camera_name = camera_name;
  row.camera_index = camera_index;
  row.local_track_id = node.local_track_id;
  row.t_start_s = node.t_start;
  row.t_end_s = node.t_end;
  row.duration_s = metrics.duration_s;
  row.observation_count = static_cast<int>(metrics.observation_count);
  row.jersey_number = hall_mode ? -1 : node.jersey_number;
  row.jersey_confidence = hall_mode ? 0.f : node.jersey_confidence;
  row.max_detect_confidence = metrics.max_detect_conf;
  row.avg_detect_confidence = metrics.avg_detect_conf;
  row.movement_px = metrics.movement_px;
  row.max_speed_px_s = metrics.max_speed_px_s;
  if (!node.observations.empty()) {
    row.bbox_x = node.observations.front().bbox.x;
    row.bbox_y = node.observations.front().bbox.y;
    row.bbox_w = node.observations.front().bbox.width;
    row.bbox_h = node.observations.front().bbox.height;
  }
  row.reject_reason = reject_tracklet_reason(node, metrics, detect_confidence, hall_mode);
  row.kept = row.reject_reason.empty();
  return row;
}

void flush_tracklet(TrackletNode node, const std::string& camera_name, int camera_index,
                    float detect_confidence, bool hall_mode, IdentityGraph& graph,
                    DetectionReportSummary& report) {
  const TrackletMetrics metrics = compute_tracklet_metrics(node);
  TrackletReportRow row =
      make_tracklet_row(node, camera_name, camera_index, metrics, detect_confidence, hall_mode);
  report.tracklets.push_back(row);
  ++report.tracklets_total;
  if (row.kept) {
    ++report.tracklets_kept;
    graph.add_tracklet(std::move(node));
  } else {
    ++report.tracklets_rejected;
  }
}

struct VideoCodecCandidate {
  const char* ext;
  char f0;
  char f1;
  char f2;
  char f3;
};

bool try_open_video_writer(cv::VideoWriter& writer, const std::filesystem::path& path, int api,
                           int fourcc, double fps, const cv::Size& frame_size) {
  writer.release();
  if (api >= 0) {
    writer.open(path.string(), api, fourcc, fps, frame_size);
  } else {
    writer.open(path.string(), fourcc, fps, frame_size);
  }
  return writer.isOpened();
}

bool open_annotated_video_writer(cv::VideoWriter& writer, const std::filesystem::path& out_video,
                                 double fps, const cv::Size& frame_size,
                                 std::filesystem::path* resolved_path) {
  const double use_fps = fps > 1.0 ? fps : 25.0;
  const cv::Size size = frame_size;
  const VideoCodecCandidate candidates[] = {
      {".mp4", 'a', 'v', 'c', '1'},
      {".mp4", 'H', '2', '6', '4'},
      {".mp4", 'm', 'p', '4', 'v'},
      {".avi", 'M', 'J', 'P', 'G'},
  };

#ifdef _WIN32
  const int apis[] = {cv::CAP_MSMF, cv::CAP_FFMPEG, -1};
#else
  const int apis[] = {cv::CAP_FFMPEG, -1};
#endif

  for (const auto& candidate : candidates) {
    auto path = out_video;
    path.replace_extension(candidate.ext);
    const int fourcc =
        cv::VideoWriter::fourcc(candidate.f0, candidate.f1, candidate.f2, candidate.f3);
    for (const int api : apis) {
      if (!try_open_video_writer(writer, path, api, fourcc, use_fps, size)) {
        continue;
      }
      if (resolved_path) {
        *resolved_path = path;
      }
      std::cerr << "Annotated video: " << path.string() << " (codec "
                << candidate.f0 << candidate.f1 << candidate.f2 << candidate.f3 << ")\n";
      return true;
    }
  }

  std::cerr << "Failed to open annotated video writer: " << out_video.string()
            << " (tried mp4v/avc1/H264/MJPG)\n";
  return false;
}

}  // namespace

MultiCameraResult run_multi_camera_session(const ProjectConfig& config, ProgressCallback progress) {
  const auto analysis_start = std::chrono::steady_clock::now();
  MultiCameraResult result;
  std::string validation_error;
  if (!config.validate(&validation_error)) {
    result.error = validation_error;
    return result;
  }

  ProjectConfig resolved_config = config;
  resolved_config.model_path = resolve_resource_path_string(config.model_path);
  resolved_config.jersey_ocr_model_path = resolve_resource_path_string(config.jersey_ocr_model_path);
  resolved_config.output_dir =
      resolve_resource_path_string(config.output_dir.empty() ? "results" : config.output_dir);
  {
    std::error_code abs_ec;
    const auto abs_output = std::filesystem::absolute(resolved_config.output_dir, abs_ec);
    if (!abs_ec) {
      resolved_config.output_dir = abs_output.string();
    }
  }

  if (!std::filesystem::exists(resolved_config.model_path)) {
    result.error = "YOLO model missing: " + resolved_config.model_path +
                   "\nRun: powershell -File scripts\\download_model.ps1";
    return result;
  }

  PersonDetector detector;
  if (!detector.load(resolved_config.model_path, resolved_config.dnn_backend,
                     resolved_config.detect_input_size)) {
    result.error = "Failed to load detector";
    return result;
  }
  std::cerr << "Performance: detect every " << resolved_config.detect_frame_step
            << " frame(s), max width " << resolved_config.detect_max_width
            << ", provider " << detector.backend_name() << "\n";

  JerseyOcr jersey_ocr;
  jersey_ocr.load(resolved_config.jersey_ocr_model_path, resolved_config.dnn_backend);

  ReIdExtractor reid;
  reid.set_hall_mode(resolved_config.hall_mode);
  IdentityGraph graph;
  graph.set_hall_mode(resolved_config.hall_mode);
  PitchModel pitch_model{resolved_config.pitch_length_m, resolved_config.pitch_width_m};
  DetectionReportSummary detection_report;
  detection_report.config = resolved_config;
  detection_report.detector_provider = detector.backend_name();

  std::error_code ec;
  std::filesystem::create_directories(resolved_config.output_dir, ec);
  const std::filesystem::path compression_log_path =
      std::filesystem::path(resolved_config.output_dir) / "compression.log";
  {
    std::ofstream log_init(compression_log_path, std::ios::out | std::ios::trunc);
    if (log_init) {
      log_init << "Video compression log\n";
    }
  }

  double global_duration = 0.0;
  int total_frames_est = 0;
  for (const auto& cam : resolved_config.cameras) {
    cv::VideoCapture cap(cam.video_path);
    if (!cap.isOpened()) {
      result.error = "Failed to open: " + cam.video_path;
      return result;
    }
    const double fps = cap.get(cv::CAP_PROP_FPS);
    const double frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    const double dur = frames / std::max(1.0, fps);
    global_duration = std::max(
        global_duration,
        cam.offset_s + analyzed_duration(dur, resolved_config.analyze_max_seconds));
    const int frame_limit = analyze_frame_limit(resolved_config.analyze_max_seconds, fps);
    total_frames_est +=
        frame_limit > 0 ? std::min(static_cast<int>(frames), frame_limit)
                        : static_cast<int>(frames);
  }

  int processed_frames = 0;
  double calib_good_frames = 0.0;
  double calib_total_frames = 0.0;
  cv::Size heatmap_frame_size;
  std::vector<std::filesystem::path> per_camera_annotated;
  std::vector<double> per_camera_offsets;
  cv::Size combined_panel_size;
  std::vector<CameraRenderCache> render_caches;

  for (size_t cam_idx = 0; cam_idx < resolved_config.cameras.size(); ++cam_idx) {
    auto& cam = resolved_config.cameras[cam_idx];
    cv::VideoCapture cap(cam.video_path);
    if (!cap.isOpened()) continue;

    const double fps = cap.get(cv::CAP_PROP_FPS);
    const double use_fps = fps > 1.0 ? fps : 25.0;
    const int max_frames = analyze_frame_limit(resolved_config.analyze_max_seconds, use_fps);
    const double source_duration =
        cap.get(cv::CAP_PROP_FRAME_COUNT) / std::max(1.0, use_fps);

    PitchCalibration static_calib;
    DynamicPitchCalibrator dynamic_calib(pitch_model);

    if (!cam.is_panning() && cam.world_coords) {
      const auto sidecar_calib_path =
          camera_sidecar_path(cam.video_path).replace_extension(".calib");
      cam.calib_path = sidecar_calib_path.string();

      if (load_calibration_from_sidecar(cam.video_path, static_calib)) {
        save_camera_sidecar(cam.video_path, cam, &static_calib);
      } else if (std::filesystem::exists(sidecar_calib_path)) {
        static_calib.load(sidecar_calib_path);
      } else {
        cv::Mat first;
        cap.read(first);
        if (!first.empty()) {
          static_calib.calibrate_lines_interactive(first, pitch_model, cam.role);
          if (static_calib.valid()) {
            static_calib.save(sidecar_calib_path);
            save_camera_sidecar(cam.video_path, cam, &static_calib);
          }
        }
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
      }
    }

    TrackletTracker tracker;
    tracker.set_detect_frame_step(resolved_config.detect_frame_step);
    std::map<int, TrackletNode> active_nodes;
    std::map<int, cv::Point2d> last_world;
    std::map<int, cv::Point2d> last_image;
    std::map<int, double> last_t;
    std::vector<Tracklet> last_tracklets;

    const std::filesystem::path out_video =
        std::filesystem::path(resolved_config.output_dir) / (cam.name + "_annotated.mp4");
    const bool cache_video = resolved_config.write_annotated_video;
    cv::Size video_export_size;
    CameraRenderCache render_cache;
    render_cache.video_path = cam.video_path;
    render_cache.out_video = out_video;
    render_cache.fps = use_fps;
    render_cache.offset_s = cam.offset_s;
    render_cache.output_duration_s =
        analyzed_duration(source_duration, resolved_config.analyze_max_seconds);
    render_cache.max_frames =
        max_frames > 0 ? max_frames
                       : static_cast<int>(std::ceil(render_cache.output_duration_s * use_fps));

    DynamicCalibResult cached_dyn;
    int dyn_frame_counter = 0;

    cv::Mat frame;
    int frame_idx = 0;
    while (cap.read(frame) && !frame.empty()) {
      if (max_frames > 0 && frame_idx >= max_frames) break;
      const double t_s = cam.offset_s + frame_idx / use_fps;
      ++frame_idx;
      ++processed_frames;

      if (frame_idx == 1 && !frame.empty()) {
        heatmap_frame_size.width = std::max(heatmap_frame_size.width, frame.cols);
        heatmap_frame_size.height = std::max(heatmap_frame_size.height, frame.rows);
        video_export_size =
            annotated_export_size(frame.size(), resolved_config.annotated_video_max_width);
        render_cache.export_size = video_export_size;
      }

      if (progress && total_frames_est > 0) {
        progress(static_cast<double>(processed_frames) / total_frames_est * 0.85,
                 "Camera " + cam.name + " frame " + std::to_string(frame_idx));
      }

      const bool run_detect =
          frame_idx == 1 || frame_idx % std::max(1, resolved_config.detect_frame_step) == 0;

      float calib_conf = 1.0f;
      auto project_world = [&](const cv::Point2d& foot, cv::Point2d& world) -> bool {
        if (cam.is_panning()) {
          if (dyn_frame_counter % std::max(1, resolved_config.dynamic_calib_step) == 0 ||
              !cached_dyn.valid) {
            auto dyn = dynamic_calib.estimate(frame);
            dyn = dynamic_calib.smooth(dyn, cached_dyn);
            if (dyn.valid) cached_dyn = dyn;
          }
          ++dyn_frame_counter;
          if (!cached_dyn.valid || cached_dyn.confidence < 0.35f) return false;
          calib_conf = cached_dyn.confidence;
          const cv::Mat Hinv = cached_dyn.homography.inv();
          const cv::Mat src = (cv::Mat_<double>(3, 1) << foot.x, foot.y, 1.0);
          cv::Mat dst = Hinv * src;
          const double w = dst.at<double>(2, 0);
          if (std::abs(w) < 1e-9) return false;
          world = {dst.at<double>(0, 0) / w, dst.at<double>(1, 0) / w};
          return true;
        }
        if (!static_calib.valid() || !cam.world_coords) return false;
        world = static_calib.image_to_meters(foot.x, foot.y);
        calib_conf = 1.0f;
        return true;
      };

      ++calib_total_frames;
      cv::Point2d test_world;
      if (project_world({frame.cols * 0.5, static_cast<double>(frame.rows) * 0.9}, test_world)) {
        ++calib_good_frames;
      }

      std::vector<Tracklet> tracklets;
      if (run_detect) {
        ++detection_report.detect_frames;
        const auto detections = detector.detect(frame, resolved_config.detect_confidence,
                                                resolved_config.detect_max_width);
        detection_report.raw_detections_total += static_cast<int>(detections.size());
        tracklets = tracker.update(detections);
        last_tracklets = tracklets;
      } else if (!last_tracklets.empty()) {
        tracklets = tracker.predict();
        last_tracklets = tracklets;
      } else {
        tracklets = tracker.predict();
      }

      std::vector<CachedOverlay> frame_overlays;
      std::set<int> seen_ids;
      for (const auto& tr : tracklets) {
        seen_ids.insert(tr.id);
        const cv::Point2d foot = foot_point(tr.bbox);
        cv::Point2d world;
        const bool has_world = project_world(foot, world) && cam.world_coords;

        ReIdFeatures feat;
        TeamId team = TeamId::Unknown;
        if (run_detect) {
          feat = reid.extract(frame, tr.bbox);
          team = reid.classify_team(feat);
        } else if (active_nodes.count(tr.id)) {
          feat = active_nodes[tr.id].features;
          team = active_nodes[tr.id].team;
        } else {
          feat = reid.extract(frame, tr.bbox);
          team = reid.classify_team(feat);
        }

        if (active_nodes.find(tr.id) == active_nodes.end()) {
          TrackletNode node;
          node.camera_index = static_cast<int>(cam_idx);
          node.local_track_id = tr.id;
          node.t_start = t_s;
          node.features = feat;
          node.team = team;
          active_nodes[tr.id] = node;
        }
        auto& node = active_nodes[tr.id];
        node.t_end = t_s;
        node.features = feat;
        node.team = team;
        if (node.snapshot.empty()) {
          node.snapshot = crop_detection_snapshot(frame, tr.bbox);
        }
        node.last_snapshot = crop_detection_snapshot(frame, tr.bbox);

        float identity_conf = 0.5f;
        if (!resolved_config.hall_mode && run_detect &&
            frame_idx % std::max(1, resolved_config.ocr_frame_step) == 0) {
          if (auto ocr = jersey_ocr.recognize(frame, tr.bbox)) {
            graph.add_ocr_anchor(static_cast<int>(cam_idx), tr.id, ocr->number, ocr->confidence);
            node.jersey_number = ocr->number;
            node.jersey_confidence = ocr->confidence;
            identity_conf = std::max(identity_conf, ocr->confidence);
          }
        } else if (resolved_config.hall_mode) {
          node.jersey_number = -1;
          node.jersey_confidence = 0.f;
          identity_conf = std::max(identity_conf, feat.green_bib_ratio);
        }

        double speed = 0.0;
        double speed_px = 0.0;
        if (last_t.count(tr.id)) {
          const double dt = std::max(1e-4, t_s - last_t[tr.id]);
          if (has_world && last_world.count(tr.id)) {
            speed = std::hypot(world.x - last_world[tr.id].x, world.y - last_world[tr.id].y) / dt;
          }
          if (last_image.count(tr.id)) {
            speed_px =
                std::hypot(foot.x - last_image[tr.id].x, foot.y - last_image[tr.id].y) / dt;
          }
        }

        TrackObservation obs;
        obs.t_s = t_s;
        obs.camera_index = static_cast<int>(cam_idx);
        obs.local_track_id = tr.id;
        obs.bbox = tr.bbox;
        obs.image_pt = foot;
        obs.world_m = world;
        obs.speed_m_s = speed;
        obs.speed_px_s = speed_px;
        obs.detect_confidence = tr.max_confidence > 0.f ? tr.max_confidence : 0.5f;
        obs.calib_confidence = calib_conf;
        obs.identity_confidence = identity_conf;
        obs.has_world = has_world;
        obs.jersey_number = node.jersey_number;
        obs.team = team;
        node.observations.push_back(obs);

        if (has_world) {
          last_world[tr.id] = world;
        }
        last_image[tr.id] = foot;
        last_t[tr.id] = t_s;

        if (cache_video) {
          const bool confirmed = show_track_in_video(tr, node, resolved_config.detect_confidence,
                                                   resolved_config.hall_mode);
          if (confirmed) {
            const float detect_conf = tr.max_confidence > 0.f ? tr.max_confidence : 0.5f;
            frame_overlays.push_back(CachedOverlay{tr.bbox, tr.id, static_cast<int>(cam_idx),
                                                   detect_conf, team,
                                                   node.features.green_bib_ratio});
          }
        }
      }

      for (auto it = active_nodes.begin(); it != active_nodes.end();) {
        if (seen_ids.count(it->first) == 0 && t_s - it->second.t_end > 1.0) {
          flush_tracklet(std::move(it->second), cam.name, static_cast<int>(cam_idx),
                         resolved_config.detect_confidence, resolved_config.hall_mode, graph,
                         detection_report);
          it = active_nodes.erase(it);
        } else {
          ++it;
        }
      }

      if (cache_video) {
        render_cache.frame_overlays.push_back(std::move(frame_overlays));
      }
    }

    for (auto& [id, node] : active_nodes) {
      (void)id;
      flush_tracklet(std::move(node), cam.name, static_cast<int>(cam_idx),
                     resolved_config.detect_confidence, resolved_config.hall_mode, graph,
                     detection_report);
    }
    active_nodes.clear();
    if (cache_video) {
      render_caches.push_back(std::move(render_cache));
      if (combined_panel_size.width <= 0 && video_export_size.width > 0) {
        combined_panel_size = video_export_size;
      }
    }
  }

  IdentityResolution identity_resolution = graph.resolve_identities();
  const auto& resolved_players = identity_resolution.players;

  if (resolved_config.write_annotated_video) {
    if (progress) progress(0.86, "Render annotated videos...");
    const int render_frames_total = static_cast<int>(std::accumulate(
        render_caches.begin(), render_caches.end(), 0,
        [](int sum, const CameraRenderCache& cache) {
          return sum + static_cast<int>(cache.frame_overlays.size());
        }));
    int render_frames_done = 0;

    for (size_t cam_idx = 0; cam_idx < render_caches.size(); ++cam_idx) {
      auto& cache = render_caches[cam_idx];
      std::filesystem::path actual_out_video;
      if (!render_cached_annotated_video(cache, static_cast<int>(cam_idx), identity_resolution,
                                         resolved_config.hall_mode, &actual_out_video)) {
        continue;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
      if (resolved_config.annotated_video_compress) {
        std::string compress_msg;
        VideoCompressOptions compress_opts;
        compress_opts.crf = resolved_config.annotated_video_crf;
        compress_opts.max_width = resolved_config.annotated_video_max_width;
        compress_opts.match_source_bitrate = resolved_config.annotated_video_match_source_bitrate;
        if (cache.output_duration_s > 0.0) {
          compress_opts.max_duration_s = cache.output_duration_s;
        }
        const bool compressed = compress_video_h264(
            actual_out_video,
            resolve_resource_path(resolved_config.cameras[cam_idx].video_path), compress_opts,
            &compress_msg);
        write_compression_log(compression_log_path, compress_msg);
        std::cerr << compress_msg << '\n';
        if (!compressed) {
          std::cerr << "See " << compression_log_path.string() << " for details\n";
        }
      } else {
        write_compression_log(compression_log_path,
                              "Compression disabled in project settings for " +
                                  actual_out_video.filename().string());
      }
      per_camera_annotated.push_back(actual_out_video);
      per_camera_offsets.push_back(cache.offset_s);
      render_frames_done += static_cast<int>(cache.frame_overlays.size());
      if (progress && render_frames_total > 0) {
        progress(0.86 + 0.10 * static_cast<double>(render_frames_done) / render_frames_total,
                 "Render " + resolved_config.cameras[cam_idx].name);
      }
    }
  }

  if (resolved_config.write_annotated_video && per_camera_annotated.size() >= 2) {
    const std::filesystem::path combined_path =
        std::filesystem::path(resolved_config.output_dir) / "combined_annotated.mp4";
    std::string stack_msg;
    const double stack_fps =
        resolved_config.cameras.empty()
            ? 25.0
            : ([&]() {
                cv::VideoCapture probe(resolved_config.cameras.front().video_path);
                const double fps = probe.get(cv::CAP_PROP_FPS);
                return fps > 1.0 ? fps : 25.0;
              })();
    if (progress) progress(0.91, "Laczenie filmow z kamer...");
    double stack_max_duration = 0.0;
    if (resolved_config.analyze_max_seconds > 0.0) {
      for (size_t i = 0; i < render_caches.size(); ++i) {
        stack_max_duration = std::max(
            stack_max_duration, render_caches[i].offset_s + render_caches[i].output_duration_s);
      }
    }
    if (stack_videos_side_by_side(
            per_camera_annotated, per_camera_offsets, stack_fps, combined_panel_size,
            combined_path, &stack_msg,
            [&](double frac, const std::string& msg) {
              if (progress) progress(0.91 + frac * 0.06, msg);
            },
            stack_max_duration)) {
      write_compression_log(compression_log_path, stack_msg);
      std::cerr << stack_msg << '\n';
      if (resolved_config.annotated_video_compress) {
        std::string compress_msg;
        VideoCompressOptions compress_opts;
        compress_opts.crf = resolved_config.annotated_video_crf;
        compress_opts.max_width = 0;
        compress_opts.match_source_bitrate = resolved_config.annotated_video_match_source_bitrate;
        if (stack_max_duration > 0.0) {
          compress_opts.max_duration_s = stack_max_duration;
        }
        const bool compressed = compress_video_h264(
            combined_path, resolve_resource_path(resolved_config.cameras.front().video_path),
            compress_opts, &compress_msg);
        write_compression_log(compression_log_path, compress_msg);
        std::cerr << compress_msg << '\n';
        if (!compressed) {
          std::cerr << "Combined video compression failed\n";
        }
      }
      result.annotated_videos.clear();
      result.annotated_videos.push_back(combined_path);
    } else {
      write_compression_log(compression_log_path, stack_msg.empty() ? "Stack failed" : stack_msg);
    }
  } else if (resolved_config.write_annotated_video && per_camera_annotated.size() == 1) {
    result.annotated_videos.clear();
    result.annotated_videos.push_back(per_camera_annotated.front());
  }

  detection_report.frames_processed = processed_frames;

  detection_report.resolved_players = static_cast<int>(resolved_players.size());
  const double calib_pct =
      calib_total_frames > 0 ? 100.0 * calib_good_frames / calib_total_frames : 0.0;

  result.global_csv = std::filesystem::path(resolved_config.output_dir) / "tracks_global.csv";
  const bool pixel_export = std::all_of(
      resolved_config.cameras.begin(), resolved_config.cameras.end(),
      [](const CameraConfig& c) { return !c.world_coords; });
  std::ofstream csv(result.global_csv);
  if (pixel_export) {
    csv << "t_s,player_number,team,x_px,y_px,speed_px_s,confidence,camera_id\n";
  } else {
    csv << "t_s,player_number,team,x_m,y_m,speed_m_s,confidence,camera_id\n";
  }
  for (const auto& player : resolved_players) {
    for (const auto& pt : player.trajectory) {
      csv << pt.t_s << ',' << pt.player_number << ','
          << team_display_name(pt.team, resolved_config.hall_mode) << ',' << pt.x_m
          << ',' << pt.y_m << ',' << pt.speed_m_s << ',' << pt.confidence << ',' << pt.camera_index
          << '\n';
    }
  }

  result.player_reports =
      PlayerStatsEngine::compute(resolved_config, resolved_players, global_duration,
                                 resolved_config.output_dir, heatmap_frame_size);
  result.quality =
      PlayerStatsEngine::compute_quality(resolved_config, resolved_players, calib_pct);

  result.pdf_report = std::filesystem::path(resolved_config.output_dir) / "report.pdf";
  result.detection_report_txt =
      std::filesystem::path(resolved_config.output_dir) / "detection_report.txt";
  result.detection_report_csv =
      std::filesystem::path(resolved_config.output_dir) / "detection_report.csv";
  result.compression_log =
      std::filesystem::path(resolved_config.output_dir) / "compression.log";
  if (!write_detection_report(result.detection_report_txt, result.detection_report_csv,
                              detection_report)) {
    result.error = "Failed to write detection report";
    return result;
  }
  if (!write_pdf_report(result.pdf_report, resolved_config, result.player_reports,
                        result.quality)) {
    result.error = "Failed to write PDF report";
    return result;
  }

  result.success = true;
  result.analysis_duration_s = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - analysis_start)
                                   .count();
  if (progress) progress(1.0, "Done");
  return result;
}

#include "multi_camera_session.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "dynamic_pitch_calibrator.hpp"
#include "identity_graph.hpp"
#include "jersey_ocr.hpp"
#include "pdf_report.hpp"
#include "person_detector.hpp"
#include "pitch_calibration.hpp"
#include "reid_extractor.hpp"
#include "tracklet_tracker.hpp"
#include "path_utils.hpp"
#include "detection_types.hpp"

namespace {

cv::Point2d foot_point(const cv::Rect2d& bbox) {
  return {bbox.x + bbox.width * 0.5, bbox.y + bbox.height * 0.95};
}

void draw_overlay(cv::Mat& frame, const Tracklet& track, int jersey, TeamId team, float conf) {
  const cv::Rect box(track.bbox);
  const cv::Scalar color = team == TeamId::TeamB ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0);
  cv::rectangle(frame, box, color, 2);
  char label[64];
  if (jersey > 0) {
    std::snprintf(label, sizeof(label), "#%d %s", jersey, team_to_string(team).c_str());
  } else {
    std::snprintf(label, sizeof(label), "T%d ?", track.id);
  }
  if (conf < 0.5f) std::strcat(label, " ?");
  cv::putText(frame, label, {box.x, std::max(20, box.y - 8)}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
              {255, 255, 255}, 2, cv::LINE_AA);
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
      {".mp4", 'm', 'p', '4', 'v'},
      {".mp4", 'a', 'v', 'c', '1'},
      {".mp4", 'H', '2', '6', '4'},
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
  IdentityGraph graph;
  PitchModel pitch_model{resolved_config.pitch_length_m, resolved_config.pitch_width_m};

  std::error_code ec;
  std::filesystem::create_directories(resolved_config.output_dir, ec);

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
    global_duration = std::max(global_duration, cam.offset_s + dur);
    total_frames_est += static_cast<int>(frames);
  }

  int processed_frames = 0;
  double calib_good_frames = 0.0;
  double calib_total_frames = 0.0;
  cv::Size heatmap_frame_size;

  for (size_t cam_idx = 0; cam_idx < resolved_config.cameras.size(); ++cam_idx) {
    const auto& cam = resolved_config.cameras[cam_idx];
    cv::VideoCapture cap(cam.video_path);
    if (!cap.isOpened()) continue;

    const double fps = cap.get(cv::CAP_PROP_FPS);
    const double use_fps = fps > 1.0 ? fps : 25.0;

    PitchCalibration static_calib;
    DynamicPitchCalibrator dynamic_calib(pitch_model);

    if (!cam.is_panning() && cam.world_coords) {
      if (!cam.calib_path.empty() && std::filesystem::exists(cam.calib_path)) {
        static_calib.load(cam.calib_path);
      } else {
        cv::Mat first;
        cap.read(first);
        if (!first.empty()) {
          static_calib.calibrate_lines_interactive(first, pitch_model);
          if (static_calib.valid() && !cam.calib_path.empty()) {
            static_calib.save(cam.calib_path);
          }
        }
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
      }
    }

    TrackletTracker tracker;
    std::map<int, TrackletNode> active_nodes;
    std::map<int, cv::Point2d> last_world;
    std::map<int, cv::Point2d> last_image;
    std::map<int, double> last_t;
    std::vector<Tracklet> last_tracklets;

    cv::VideoWriter writer;
    const std::filesystem::path out_video =
        std::filesystem::path(resolved_config.output_dir) / (cam.name + "_annotated.mp4");
    std::filesystem::path actual_out_video = out_video;
    bool write_video = resolved_config.write_annotated_video;
    bool video_open_attempted = false;

    DynamicCalibResult cached_dyn;
    int dyn_frame_counter = 0;

    cv::Mat frame;
    int frame_idx = 0;
    while (cap.read(frame) && !frame.empty()) {
      const double t_s = cam.offset_s + frame_idx / use_fps;
      ++frame_idx;
      ++processed_frames;

      if (frame_idx == 1 && !frame.empty()) {
        heatmap_frame_size.width = std::max(heatmap_frame_size.width, frame.cols);
        heatmap_frame_size.height = std::max(heatmap_frame_size.height, frame.rows);
      }

      if (progress && total_frames_est > 0) {
        progress(static_cast<double>(processed_frames) / total_frames_est,
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
        const auto detections = detector.detect(frame, resolved_config.detect_confidence,
                                                resolved_config.detect_max_width);
        tracklets = tracker.update(detections);
        last_tracklets = tracklets;
      } else if (!last_tracklets.empty()) {
        std::vector<Detection> propagated;
        propagated.reserve(last_tracklets.size());
        for (const auto& tr : last_tracklets) {
          Detection det;
          det.box = tr.bbox;
          det.confidence = 0.5f;
          det.class_id = 0;
          propagated.push_back(det);
        }
        tracklets = tracker.update(propagated);
        last_tracklets = tracklets;
      } else {
        tracklets = tracker.update({});
      }

      if (write_video && !writer.isOpened() && !video_open_attempted) {
        video_open_attempted = true;
        if (!open_annotated_video_writer(writer, out_video, use_fps, frame.size(),
                                         &actual_out_video)) {
          write_video = false;
        }
      }

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

        float identity_conf = 0.5f;
        if (run_detect && frame_idx % std::max(1, resolved_config.ocr_frame_step) == 0) {
          if (auto ocr = jersey_ocr.recognize(frame, tr.bbox)) {
            graph.add_ocr_anchor(static_cast<int>(cam_idx), tr.id, ocr->number, ocr->confidence);
            node.jersey_number = ocr->number;
            node.jersey_confidence = ocr->confidence;
            identity_conf = std::max(identity_conf, ocr->confidence);
          }
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
        obs.detect_confidence = 0.8f;
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

        if (write_video && writer.isOpened()) {
          draw_overlay(frame, tr, node.jersey_number, team, identity_conf);
        }
      }

      for (auto it = active_nodes.begin(); it != active_nodes.end();) {
        if (seen_ids.count(it->first) == 0 && t_s - it->second.t_end > 1.0) {
          graph.add_tracklet(std::move(it->second));
          it = active_nodes.erase(it);
        } else {
          ++it;
        }
      }

      if (write_video && writer.isOpened()) {
        writer.write(frame);
      }
    }

    for (auto& [id, node] : active_nodes) {
      (void)id;
      graph.add_tracklet(std::move(node));
    }
    if (write_video && writer.isOpened()) {
      result.annotated_videos.push_back(actual_out_video);
    }
  }

  const auto resolved_players = graph.resolve();
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
      csv << pt.t_s << ',' << pt.player_number << ',' << team_to_string(pt.team) << ',' << pt.x_m
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

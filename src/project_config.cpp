#include "project_config.hpp"

#include <algorithm>
#include <fstream>

#include <nlohmann/json.hpp>

#include "app_options.hpp"
#include "path_utils.hpp"

namespace {

using json = nlohmann::json;

json camera_to_json(const CameraConfig& cam) {
  return json{
      {"name", cam.name},
      {"path", cam.video_path},
      {"role", camera_role_to_string(cam.role)},
      {"motion", camera_motion_to_string(cam.motion)},
      {"offset_s", cam.offset_s},
      {"world_coords", cam.world_coords},
      {"calib", cam.calib_path},
  };
}

CameraConfig camera_from_json(const json& j) {
  CameraConfig cam;
  cam.name = j.value("name", "");
  cam.video_path = j.value("path", "");
  cam.role = parse_camera_role(j.value("role", "unknown"));
  cam.motion = parse_camera_motion(j.value("motion", "static"));
  cam.offset_s = j.value("offset_s", 0.0);
  cam.world_coords = j.value("world_coords", true);
  cam.calib_path = j.value("calib", "");
  if (cam.name.empty()) {
    cam.name = cam.video_path;
  }
  if (cam.is_panning() && detect_project_mode({cam}) == ProjectMode::TacticalOnly) {
    cam.world_coords = true;
  }
  return cam;
}

}  // namespace

bool ProjectConfig::has_panning_camera() const {
  return std::any_of(cameras.begin(), cameras.end(),
                     [](const CameraConfig& c) { return c.is_panning(); });
}

bool ProjectConfig::validate(std::string* error) const {
  if (cameras.empty()) {
    if (error) *error = "Project needs at least one camera";
    return false;
  }
  if (pitch_length_m <= 0.0 || pitch_width_m <= 0.0) {
    if (error) *error = "Invalid pitch dimensions";
    return false;
  }
  for (const auto& cam : cameras) {
    if (cam.video_path.empty()) {
      if (error) *error = "Camera video path is empty";
      return false;
    }
  }
  return true;
}

bool save_project_config(const std::filesystem::path& path, const ProjectConfig& config) {
  json j;
  j["pitch"] = {{"length_m", config.pitch_length_m}, {"width_m", config.pitch_width_m}};
  j["hall_mode"] = config.hall_mode;
  j["sprint_threshold_m_s"] = config.sprint_threshold_m_s;
  j["hi_run_threshold_m_s"] = config.hi_run_threshold_m_s;
  j["output_dir"] = config.output_dir;
  j["model_path"] = config.model_path;
  j["jersey_ocr_model_path"] = config.jersey_ocr_model_path;
  j["detect_confidence"] = config.detect_confidence;
  j["detect_input_size"] = config.detect_input_size;
  j["detect_frame_step"] = config.detect_frame_step;
  j["detect_max_width"] = config.detect_max_width;
  j["ocr_frame_step"] = config.ocr_frame_step;
  j["dynamic_calib_step"] = config.dynamic_calib_step;
  j["dnn_backend"] = dnn_backend_to_string(config.dnn_backend);
  j["write_annotated_video"] = config.write_annotated_video;
  j["annotated_video_compress"] = config.annotated_video_compress;
  j["annotated_video_crf"] = config.annotated_video_crf;
  j["annotated_video_max_width"] = config.annotated_video_max_width;
  j["annotated_video_match_source_bitrate"] = config.annotated_video_match_source_bitrate;
  j["analyze_max_seconds"] = config.analyze_max_seconds;
  j["project_mode"] = project_mode_to_string(config.mode());
  j["cameras"] = json::array();
  for (const auto& cam : config.cameras) {
    j["cameras"].push_back(camera_to_json(cam));
  }

  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) return false;
  out << j.dump(2);
  return true;
}

std::optional<ProjectConfig> load_project_config(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) return std::nullopt;

  json j;
  try {
    in >> j;
  } catch (...) {
    return std::nullopt;
  }

  ProjectConfig config;
  if (j.contains("pitch")) {
    config.pitch_length_m = j["pitch"].value("length_m", 105.0);
    config.pitch_width_m = j["pitch"].value("width_m", 68.0);
  }
  config.hall_mode = j.value("hall_mode", false);
  config.sprint_threshold_m_s = j.value("sprint_threshold_m_s", 7.0);
  config.hi_run_threshold_m_s = j.value("hi_run_threshold_m_s", 5.5);
  config.output_dir = j.value("output_dir", "results");
  config.model_path = j.value("model_path", "models/yolov8s.onnx");
  config.jersey_ocr_model_path = j.value("jersey_ocr_model_path", "models/jersey_ocr.onnx");
  config.detect_confidence = j.value("detect_confidence", 0.40f);
  config.detect_input_size = j.value("detect_input_size", 640);
  config.detect_frame_step = std::max(1, j.value("detect_frame_step", 2));
  config.detect_max_width = j.value("detect_max_width", 1280);
  config.ocr_frame_step = std::max(1, j.value("ocr_frame_step", 15));
  config.dynamic_calib_step = std::max(1, j.value("dynamic_calib_step", 5));
  config.dnn_backend = parse_dnn_backend(j.value("dnn_backend", "auto"));
  config.write_annotated_video = j.value("write_annotated_video", true);
  config.annotated_video_compress = j.value("annotated_video_compress", true);
  config.annotated_video_crf = std::clamp(j.value("annotated_video_crf", 32), 23, 40);
  config.annotated_video_max_width = j.value("annotated_video_max_width", 1280);
  config.annotated_video_match_source_bitrate = j.value("annotated_video_match_source_bitrate", true);
  config.analyze_max_seconds = std::max(0.0, j.value("analyze_max_seconds", 0.0));

  if (j.contains("cameras") && j["cameras"].is_array()) {
    for (const auto& item : j["cameras"]) {
      config.cameras.push_back(camera_from_json(item));
    }
  }

  const ProjectMode mode = config.mode();
  if (mode == ProjectMode::TacticalOnly) {
    for (auto& cam : config.cameras) {
      cam.world_coords = true;
    }
  }

  return config;
}

std::filesystem::path last_session_config_path() {
  return application_directory() / "last_session.ptproj";
}

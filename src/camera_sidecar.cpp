#include "camera_sidecar.hpp"

#include <fstream>

#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::json;

}  // namespace

std::filesystem::path camera_sidecar_path(const std::filesystem::path& video_path) {
  return video_path.parent_path() / (video_path.stem().string() + ".ptcam.json");
}

bool save_camera_sidecar(const std::filesystem::path& video_path, const CameraConfig& cam,
                         const PitchCalibration* calib) {
  json j;
  j["video_path"] = video_path.string();
  j["name"] = cam.name;
  j["role"] = camera_role_to_string(cam.role);
  j["motion"] = camera_motion_to_string(cam.motion);
  j["world_coords"] = cam.world_coords;
  j["offset_s"] = cam.offset_s;

  if (calib != nullptr && calib->valid()) {
    const auto calib_path = camera_sidecar_path(video_path).replace_extension(".calib");
    if (calib->save(calib_path)) {
      j["calib_file"] = calib_path.string();
    }
  } else {
    const auto existing_calib = camera_sidecar_path(video_path).replace_extension(".calib");
    if (std::filesystem::exists(existing_calib)) {
      j["calib_file"] = existing_calib.string();
    }
  }

  const auto path = camera_sidecar_path(video_path);
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) return false;
  out << j.dump(2);
  return true;
}

bool load_camera_sidecar(const std::filesystem::path& video_path, CameraConfig& cam) {
  const auto path = camera_sidecar_path(video_path);
  if (!std::filesystem::exists(path)) return false;

  std::ifstream in(path);
  if (!in) return false;

  json j;
  try {
    in >> j;
  } catch (...) {
    return false;
  }

  cam.video_path = video_path.string();
  cam.name = j.value("name", video_path.stem().string());
  cam.role = parse_camera_role(j.value("role", "unknown"));
  cam.motion = parse_camera_motion(j.value("motion", "static"));
  cam.world_coords = j.value("world_coords", false);
  cam.offset_s = j.value("offset_s", 0.0);

  if (j.contains("calib_file")) {
    cam.calib_path = j.value("calib_file", "");
  } else {
    const auto default_calib = camera_sidecar_path(video_path).replace_extension(".calib");
    if (std::filesystem::exists(default_calib)) {
      cam.calib_path = default_calib.string();
    }
  }
  return true;
}

bool load_calibration_from_sidecar(const std::filesystem::path& video_path,
                                   PitchCalibration& calib) {
  CameraConfig dummy;
  dummy.video_path = video_path.string();
  load_camera_sidecar(video_path, dummy);

  std::filesystem::path calib_path = dummy.calib_path;
  if (calib_path.empty()) {
    calib_path = camera_sidecar_path(video_path).replace_extension(".calib");
  }
  if (!std::filesystem::exists(calib_path)) return false;
  return calib.load(calib_path);
}

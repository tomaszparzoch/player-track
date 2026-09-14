#include "camera_config.hpp"

#include <algorithm>

CameraRole parse_camera_role(const std::string& text) {
  if (text == "corner_tl") return CameraRole::CornerTL;
  if (text == "corner_tr") return CameraRole::CornerTR;
  if (text == "corner_bl") return CameraRole::CornerBL;
  if (text == "corner_br") return CameraRole::CornerBR;
  if (text == "side_left") return CameraRole::SideLeft;
  if (text == "side_right") return CameraRole::SideRight;
  if (text == "tactical_pan") return CameraRole::TacticalPan;
  return CameraRole::Unknown;
}

std::string camera_role_to_string(CameraRole role) {
  switch (role) {
    case CameraRole::CornerTL: return "corner_tl";
    case CameraRole::CornerTR: return "corner_tr";
    case CameraRole::CornerBL: return "corner_bl";
    case CameraRole::CornerBR: return "corner_br";
    case CameraRole::SideLeft: return "side_left";
    case CameraRole::SideRight: return "side_right";
    case CameraRole::TacticalPan: return "tactical_pan";
    default: return "unknown";
  }
}

CameraMotion parse_camera_motion(const std::string& text) {
  if (text == "panning") return CameraMotion::Panning;
  return CameraMotion::Static;
}

std::string camera_motion_to_string(CameraMotion motion) {
  return motion == CameraMotion::Panning ? "panning" : "static";
}

ProjectMode detect_project_mode(const std::vector<CameraConfig>& cameras) {
  bool has_static = false;
  bool has_panning = false;
  for (const auto& cam : cameras) {
    if (cam.is_panning()) {
      has_panning = true;
    } else {
      has_static = true;
    }
  }
  if (has_panning && !has_static) return ProjectMode::TacticalOnly;
  if (has_panning && has_static) return ProjectMode::StaticPlusTactical;
  return ProjectMode::StaticOnly;
}

std::string project_mode_to_string(ProjectMode mode) {
  switch (mode) {
    case ProjectMode::StaticPlusTactical: return "StaticPlusTactical";
    case ProjectMode::TacticalOnly: return "TacticalOnly";
    default: return "StaticOnly";
  }
}

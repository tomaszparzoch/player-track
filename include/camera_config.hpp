#pragma once

#include <string>
#include <vector>
#include <vector>

enum class CameraRole {
  CornerTL,
  CornerTR,
  CornerBL,
  CornerBR,
  SideLeft,
  SideRight,
  TacticalPan,
  Unknown
};

enum class CameraMotion { Static, Panning };

enum class ProjectMode { StaticOnly, StaticPlusTactical, TacticalOnly };

struct CameraConfig {
  std::string name;
  std::string video_path;
  CameraRole role{CameraRole::Unknown};
  CameraMotion motion{CameraMotion::Static};
  double offset_s{0.0};
  bool world_coords{false};
  std::string calib_path;

  bool is_panning() const { return motion == CameraMotion::Panning; }
};

CameraRole parse_camera_role(const std::string& text);
std::string camera_role_to_string(CameraRole role);
CameraMotion parse_camera_motion(const std::string& text);
std::string camera_motion_to_string(CameraMotion motion);
ProjectMode detect_project_mode(const std::vector<CameraConfig>& cameras);
std::string project_mode_to_string(ProjectMode mode);

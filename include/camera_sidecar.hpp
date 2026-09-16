#pragma once

#include <filesystem>
#include <optional>

#include "camera_config.hpp"
#include "pitch_calibration.hpp"

std::filesystem::path camera_sidecar_path(const std::filesystem::path& video_path);

bool save_camera_sidecar(const std::filesystem::path& video_path, const CameraConfig& cam,
                         const PitchCalibration* calib = nullptr);
bool load_camera_sidecar(const std::filesystem::path& video_path, CameraConfig& cam);
bool load_calibration_from_sidecar(const std::filesystem::path& video_path,
                                   PitchCalibration& calib);

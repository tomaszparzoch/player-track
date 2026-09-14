#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "camera_config.hpp"
#include "app_options.hpp"

struct ProjectConfig {
  double pitch_length_m{105.0};
  double pitch_width_m{68.0};
  double sprint_threshold_m_s{7.0};
  double hi_run_threshold_m_s{5.5};
  std::string output_dir{"results"};
  std::string model_path{"models/yolov8s.onnx"};
  std::string jersey_ocr_model_path{"models/jersey_ocr.onnx"};
  float detect_confidence{0.40f};
  int detect_input_size{640};
  int detect_frame_step{2};
  int detect_max_width{1280};
  int ocr_frame_step{15};
  int dynamic_calib_step{5};
  DnnBackend dnn_backend{DnnBackend::Auto};
  bool write_annotated_video{true};
  std::vector<CameraConfig> cameras;

  ProjectMode mode() const { return detect_project_mode(cameras); }
  bool has_panning_camera() const;
  bool validate(std::string* error = nullptr) const;
};

bool save_project_config(const std::filesystem::path& path, const ProjectConfig& config);
std::optional<ProjectConfig> load_project_config(const std::filesystem::path& path);

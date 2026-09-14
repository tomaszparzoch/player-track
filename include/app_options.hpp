#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

constexpr int kMaxPlayers = 24;

enum class TrackingMode { Manual, Auto };

enum class DnnBackend { Auto, Cpu, Cuda, DirectML, OpenVino };

DnnBackend parse_dnn_backend(const std::string& text);
std::string dnn_backend_to_string(DnnBackend backend);
void log_available_dnn_backends();

struct AppOptions {
  std::string input;
  std::string output = "track.csv";
  std::vector<std::optional<cv::Rect2d>> bboxes;
  int players = 14;
  bool discover_players = true;
  int discover_interval = 6;
  bool simulate_gap = true;
  int simulate_gap_max_frames = 90;
  bool preview = false;
  std::optional<std::string> video_out;
  std::optional<std::string> heatmap_dir;
  std::optional<std::string> calib_path;
  bool calibrate = false;
  double pitch_width_m = 40.32;
  double pitch_height_m = 16.5;
  TrackingMode mode = TrackingMode::Manual;
  std::string model_path = "models/yolov8s.onnx";
  float detect_confidence = 0.40f;
  int detect_input_size = 640;
  int detect_frames = 12;
  int detect_frame_step = 5;
  int detect_min_hits = 1;
  bool redetect_on_loss = true;
  bool interactive_select = false;
  DnnBackend dnn_backend = DnnBackend::Auto;
  std::vector<int> pick_detections;
};

void print_usage();
std::optional<AppOptions> parse_args(int argc, char** argv);

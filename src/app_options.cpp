#include "app_options.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string_view>

#include "onnx_infer.hpp"

namespace {

std::optional<cv::Rect2d> parse_bbox(std::string_view text) {
  std::stringstream ss{std::string(text)};
  std::string token;
  double values[4]{};
  int i = 0;
  while (std::getline(ss, token, ',')) {
    if (i >= 4) {
      return std::nullopt;
    }
    try {
      values[i++] = std::stod(token);
    } catch (...) {
      return std::nullopt;
    }
  }
  if (i != 4 || values[2] <= 1.0 || values[3] <= 1.0) {
    return std::nullopt;
  }
  return cv::Rect2d(values[0], values[1], values[2], values[3]);
}

bool set_bbox_option(AppOptions& opts, int index, std::string_view value) {
  if (index < 0 || index >= kMaxPlayers) {
    std::cerr << "Too many --bbox flags (max " << kMaxPlayers << ")\n";
    return false;
  }
  if (static_cast<int>(opts.bboxes.size()) <= index) {
    opts.bboxes.resize(index + 1);
  }
  opts.bboxes[index] = parse_bbox(value);
  if (!opts.bboxes[index]) {
    std::cerr << "Invalid bbox for player " << (index + 1) << ", expected x,y,w,h\n";
    return false;
  }
  return true;
}

std::optional<int> bbox_flag_index(std::string_view arg) {
  if (arg == "--bbox") {
    return 0;
  }
  if (arg.size() > 6 && arg.rfind("--bbox", 0) == 0) {
    try {
      const int index = std::stoi(std::string(arg.substr(6))) - 1;
      if (index >= 1 && index < kMaxPlayers) {
        return index;
      }
    } catch (...) {
    }
  }
  return std::nullopt;
}

std::optional<TrackingMode> parse_mode(std::string_view text) {
  if (text == "manual") {
    return TrackingMode::Manual;
  }
  if (text == "auto") {
    return TrackingMode::Auto;
  }
  return std::nullopt;
}

std::optional<DnnBackend> parse_dnn_backend_opt(std::string_view text) {
  if (text == "auto") return DnnBackend::Auto;
  if (text == "cpu") return DnnBackend::Cpu;
  if (text == "cuda") return DnnBackend::Cuda;
  if (text == "directml" || text == "dml") return DnnBackend::DirectML;
  if (text == "openvino") return DnnBackend::OpenVino;
  return std::nullopt;
}

}  // namespace

DnnBackend parse_dnn_backend(const std::string& text) {
  const auto parsed = parse_dnn_backend_opt(std::string_view(text));
  return parsed.value_or(DnnBackend::Auto);
}

std::string dnn_backend_to_string(DnnBackend backend) {
  switch (backend) {
    case DnnBackend::Cpu: return "cpu";
    case DnnBackend::Cuda: return "cuda";
    case DnnBackend::DirectML: return "directml";
    case DnnBackend::OpenVino: return "openvino";
    default: return "auto";
  }
}

void log_available_dnn_backends() {
  log_available_onnx_providers();
}

void print_usage() {
  std::cerr
      << "Usage:\n"
      << "  player_track --input <video> [--output track.csv]\n"
      << "               [--mode manual|auto] [--players N]\n"
      << "               [--bbox x,y,w,h] [--bbox2 ...]\n"
      << "               [--model models/yolov8s.onnx] [--detect-confidence 0.40]\n"
      << "               [--detect-input-size 640|1280] [--detect-frames N]\n"
      << "               [--detect-frame-step N] [--detect-min-hits N]\n"
      << "               [--redetect-on-loss] [--no-redetect-on-loss]\n"
      << "               [--discover-players] [--no-discover-players]\n"
      << "               [--discover-interval N]\n"
      << "               [--simulate-gap] [--no-simulate-gap]\n"
      << "               [--simulate-gap-max-frames N]\n"
      << "               [--interactive-select] [--dnn-backend auto|cpu|cuda|directml]\n"
      << "               [--pick-detections 1,2,3]\n"
      << "               [--calib pitch.calib | --calibrate pitch.calib]\n"
      << "               [--pitch-rect width_m,height_m] [--heatmap-dir dir]\n"
      << "               [--preview] [--video-out out.mp4]\n"
      << "               [--gui] [--project match.ptproj]\n";
}

std::optional<AppOptions> parse_args(int argc, char** argv) {
  AppOptions opts;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    auto need_value = [&](std::string_view name) -> std::optional<std::string> {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        return std::nullopt;
      }
      return std::string{argv[++i]};
    };

    if (arg == "--help" || arg == "-h") {
      print_usage();
      std::exit(0);
    }
    if (arg == "--preview") {
      opts.preview = true;
      continue;
    }
    if (arg == "--input") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      opts.input = *value;
      continue;
    }
    if (arg == "--output") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      opts.output = *value;
      continue;
    }
    if (arg == "--video-out") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      opts.video_out = *value;
      continue;
    }
    if (arg == "--calib") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      opts.calib_path = *value;
      continue;
    }
    if (arg == "--calibrate") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      opts.calib_path = *value;
      opts.calibrate = true;
      continue;
    }
    if (arg == "--pitch-rect") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      std::stringstream ss{*value};
      std::string token;
      double dims[2]{};
      int dim_i = 0;
      while (std::getline(ss, token, ',')) {
        if (dim_i >= 2) {
          std::cerr << "Invalid --pitch-rect, expected width_m,height_m\n";
          return std::nullopt;
        }
        try {
          dims[dim_i++] = std::stod(token);
        } catch (...) {
          std::cerr << "Invalid --pitch-rect, expected width_m,height_m\n";
          return std::nullopt;
        }
      }
      if (dim_i != 2 || dims[0] <= 0.0 || dims[1] <= 0.0) {
        std::cerr << "Invalid --pitch-rect, expected width_m,height_m\n";
        return std::nullopt;
      }
      opts.pitch_width_m = dims[0];
      opts.pitch_height_m = dims[1];
      continue;
    }
    if (arg == "--heatmap-dir") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      opts.heatmap_dir = *value;
      continue;
    }
    if (arg == "--mode") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      const auto mode = parse_mode(*value);
      if (!mode) {
        std::cerr << "Invalid --mode, expected manual or auto\n";
        return std::nullopt;
      }
      opts.mode = *mode;
      continue;
    }
    if (arg == "--model") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      opts.model_path = *value;
      continue;
    }
    if (arg == "--detect-confidence") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      try {
        opts.detect_confidence = std::stof(*value);
      } catch (...) {
        std::cerr << "Invalid --detect-confidence value\n";
        return std::nullopt;
      }
      if (opts.detect_confidence < 0.0f || opts.detect_confidence > 1.0f) {
        std::cerr << "--detect-confidence must be between 0 and 1\n";
        return std::nullopt;
      }
      continue;
    }
    if (arg == "--detect-input-size") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      try {
        opts.detect_input_size = std::stoi(*value);
      } catch (...) {
        std::cerr << "Invalid --detect-input-size value\n";
        return std::nullopt;
      }
      if (opts.detect_input_size < 320 || opts.detect_input_size > 1920) {
        std::cerr << "--detect-input-size must be between 320 and 1920\n";
        return std::nullopt;
      }
      continue;
    }
    if (arg == "--redetect-on-loss") {
      opts.redetect_on_loss = true;
      continue;
    }
    if (arg == "--no-redetect-on-loss") {
      opts.redetect_on_loss = false;
      continue;
    }
    if (arg == "--discover-players") {
      opts.discover_players = true;
      continue;
    }
    if (arg == "--no-discover-players") {
      opts.discover_players = false;
      continue;
    }
    if (arg == "--simulate-gap") {
      opts.simulate_gap = true;
      continue;
    }
    if (arg == "--no-simulate-gap") {
      opts.simulate_gap = false;
      continue;
    }
    if (arg == "--simulate-gap-max-frames") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      try {
        opts.simulate_gap_max_frames = std::stoi(*value);
      } catch (...) {
        std::cerr << "Invalid --simulate-gap-max-frames value\n";
        return std::nullopt;
      }
      if (opts.simulate_gap_max_frames < 1) {
        std::cerr << "--simulate-gap-max-frames must be at least 1\n";
        return std::nullopt;
      }
      continue;
    }
    if (arg == "--discover-interval") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      try {
        opts.discover_interval = std::stoi(*value);
      } catch (...) {
        std::cerr << "Invalid --discover-interval value\n";
        return std::nullopt;
      }
      if (opts.discover_interval < 1) {
        std::cerr << "--discover-interval must be at least 1\n";
        return std::nullopt;
      }
      continue;
    }
    if (arg == "--detect-frames") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      try {
        opts.detect_frames = std::stoi(*value);
      } catch (...) {
        std::cerr << "Invalid --detect-frames value\n";
        return std::nullopt;
      }
      if (opts.detect_frames < 1) {
        std::cerr << "--detect-frames must be at least 1\n";
        return std::nullopt;
      }
      continue;
    }
    if (arg == "--detect-min-hits") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      try {
        opts.detect_min_hits = std::stoi(*value);
      } catch (...) {
        std::cerr << "Invalid --detect-min-hits value\n";
        return std::nullopt;
      }
      if (opts.detect_min_hits < 1) {
        std::cerr << "--detect-min-hits must be at least 1\n";
        return std::nullopt;
      }
      continue;
    }
    if (arg == "--interactive-select") {
      opts.interactive_select = true;
      continue;
    }
    if (arg == "--detect-frame-step") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      try {
        opts.detect_frame_step = std::stoi(*value);
      } catch (...) {
        std::cerr << "Invalid --detect-frame-step value\n";
        return std::nullopt;
      }
      if (opts.detect_frame_step < 1) {
        std::cerr << "--detect-frame-step must be at least 1\n";
        return std::nullopt;
      }
      continue;
    }
    if (arg == "--pick-detections") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      std::stringstream ss{*value};
      std::string token;
      opts.pick_detections.clear();
      while (std::getline(ss, token, ',')) {
        if (token.empty()) {
          continue;
        }
        try {
          opts.pick_detections.push_back(std::stoi(token));
        } catch (...) {
          std::cerr << "Invalid --pick-detections value, expected comma-separated indices\n";
          return std::nullopt;
        }
      }
      if (opts.pick_detections.empty()) {
        std::cerr << "--pick-detections requires at least one index\n";
        return std::nullopt;
      }
      if (static_cast<int>(opts.pick_detections.size()) > opts.players) {
        std::cerr << "--pick-detections has more indices than --players\n";
        return std::nullopt;
      }
      continue;
    }
    if (arg == "--dnn-backend") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      const auto backend = parse_dnn_backend_opt(*value);
      if (!backend) {
        std::cerr << "Invalid --dnn-backend, expected auto, cpu, cuda, directml, or openvino\n";
        return std::nullopt;
      }
      opts.dnn_backend = *backend;
      continue;
    }
    if (arg == "--players") {
      const auto value = need_value(arg);
      if (!value) {
        return std::nullopt;
      }
      try {
        opts.players = std::stoi(*value);
      } catch (...) {
        std::cerr << "Invalid --players value\n";
        return std::nullopt;
      }
      if (opts.players < 1 || opts.players > kMaxPlayers) {
        std::cerr << "--players must be between 1 and " << kMaxPlayers << "\n";
        return std::nullopt;
      }
      continue;
    }

    const auto bbox_index = bbox_flag_index(arg);
    if (bbox_index) {
      const auto value = need_value(arg);
      if (!value || !set_bbox_option(opts, *bbox_index, *value)) {
        return std::nullopt;
      }
      continue;
    }

    std::cerr << "Unknown argument: " << arg << "\n";
    print_usage();
    return std::nullopt;
  }

  if (opts.input.empty()) {
    std::cerr << "--input is required\n";
    print_usage();
    return std::nullopt;
  }

  if (opts.mode == TrackingMode::Manual) {
    for (int i = 0; i < static_cast<int>(opts.bboxes.size()); ++i) {
      if (opts.bboxes[i] && i >= opts.players) {
        std::cerr << "--bbox" << (i == 0 ? "" : std::to_string(i + 1))
                  << " requires --players " << (i + 1) << " or more\n";
        return std::nullopt;
      }
    }
  }

  return opts;
}

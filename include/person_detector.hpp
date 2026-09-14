#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "app_options.hpp"
#include "onnx_infer.hpp"
#include "detection_types.hpp"

class PersonDetector;

double detection_iou(const cv::Rect2d& a, const cv::Rect2d& b);

std::vector<Detection> merge_detections(const std::vector<Detection>& detections,
                                        float iou_threshold = 0.55f, int min_hits = 1);

DetectionScanResult detect_persons_multi_frame(PersonDetector& detector, cv::VideoCapture& cap,
                                               const cv::Mat& first_frame, int frame_count,
                                               int frame_step, float confidence_threshold,
                                               int min_hits = 1);

std::optional<Detection> find_detection_near_box(const std::vector<Detection>& detections,
                                                 const cv::Rect2d& reference_box,
                                                 float min_iou = 0.15f);

class PersonDetector {
 public:
  PersonDetector() = default;
  PersonDetector(PersonDetector&&) noexcept = default;
  PersonDetector& operator=(PersonDetector&&) noexcept = default;
  PersonDetector(const PersonDetector&) = delete;
  PersonDetector& operator=(const PersonDetector&) = delete;

  bool load(const std::filesystem::path& model_path, DnnBackend backend, int input_size = 640);
  std::vector<Detection> detect(const cv::Mat& frame, float confidence_threshold,
                                int max_width = 0);
  std::string backend_name() const { return backend_name_; }
  int input_size() const { return input_size_; }

 private:
  OnnxInferSession session_;
  bool loaded_{false};
  int input_size_{640};
  std::string backend_name_ = "cpu";
  static constexpr int kPersonClassId = 0;
};

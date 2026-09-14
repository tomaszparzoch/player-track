#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include <opencv2/core.hpp>

#include "app_options.hpp"
#include "onnx_infer.hpp"

struct JerseyOcrResult {
  int number{-1};
  float confidence{0.0f};
};

class JerseyOcr {
 public:
  JerseyOcr() = default;
  JerseyOcr(JerseyOcr&&) noexcept = default;
  JerseyOcr& operator=(JerseyOcr&&) noexcept = default;
  JerseyOcr(const JerseyOcr&) = delete;
  JerseyOcr& operator=(const JerseyOcr&) = delete;

  bool load(const std::filesystem::path& model_path,
            DnnBackend backend = DnnBackend::Auto);
  std::optional<JerseyOcrResult> recognize(const cv::Mat& frame, const cv::Rect2d& bbox);

 private:
  OnnxInferSession session_;
  bool loaded_{false};

  std::optional<JerseyOcrResult> recognize_heuristic(const cv::Mat& frame,
                                                     const cv::Rect2d& bbox) const;
};

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

#include "app_options.hpp"

class OnnxInferSession {
 public:
  OnnxInferSession() = default;
  ~OnnxInferSession() = default;
  OnnxInferSession(OnnxInferSession&&) noexcept = default;
  OnnxInferSession& operator=(OnnxInferSession&&) noexcept = default;
  OnnxInferSession(const OnnxInferSession&) = delete;
  OnnxInferSession& operator=(const OnnxInferSession&) = delete;

  bool load(const std::filesystem::path& model_path, DnnBackend backend);
  cv::Mat run(const cv::Mat& nchw_blob, const std::vector<int64_t>& input_shape);
  const std::string& provider_name() const { return provider_name_; }
  bool loaded() const { return loaded_; }

 private:
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
  bool loaded_{false};
  std::string provider_name_ = "cpu";
};

void log_available_onnx_providers();

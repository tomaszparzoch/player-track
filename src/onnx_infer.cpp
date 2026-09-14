#include "onnx_infer.hpp"

#include <cstring>
#include <iostream>

#ifdef _WIN32
#include <dml_provider_factory.h>
#endif

namespace {

bool configure_provider(Ort::SessionOptions& options, DnnBackend backend, std::string& provider_name) {
  if (backend == DnnBackend::Cpu) {
    provider_name = "cpu";
    return true;
  }

  if (backend == DnnBackend::DirectML ||
      (backend == DnnBackend::Auto
#ifdef _WIN32
       && true
#else
       && false
#endif
       )) {
#ifdef _WIN32
    if (OrtSessionOptionsAppendExecutionProvider_DML(options, 0) == nullptr) {
      provider_name = "directml";
      return true;
    }
    std::cerr << "DirectML provider unavailable\n";
    if (backend == DnnBackend::DirectML) {
      return false;
    }
#endif
  }

  if (backend == DnnBackend::Cuda || backend == DnnBackend::Auto) {
    try {
      OrtCUDAProviderOptions cuda_options{};
      cuda_options.device_id = 0;
      options.AppendExecutionProvider_CUDA(cuda_options);
      provider_name = "cuda";
      return true;
    } catch (const Ort::Exception& ex) {
      if (backend == DnnBackend::Cuda) {
        std::cerr << "CUDA provider unavailable: " << ex.what() << "\n";
        return false;
      }
    }
  }

  if (backend == DnnBackend::OpenVino) {
    std::cerr << "OpenVINO is not supported with ONNX Runtime in this build\n";
    return false;
  }

  provider_name = "cpu";
  return true;
}

}  // namespace

Ort::Env& ort_env() {
  static Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "player_track"};
  return env;
}

bool OnnxInferSession::load(const std::filesystem::path& model_path, DnnBackend backend) {
  if (!std::filesystem::exists(model_path)) {
    std::cerr << "ONNX model not found: " << model_path << "\n";
    return false;
  }

  session_.reset();
  loaded_ = false;

  Ort::SessionOptions session_options;
  if (!configure_provider(session_options, backend, provider_name_)) {
    return false;
  }

  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options.SetIntraOpNumThreads(1);

#ifdef _WIN32
  const std::wstring wide_path = model_path.wstring();
  session_ = std::make_unique<Ort::Session>(ort_env(), wide_path.c_str(), session_options);
#else
  session_ = std::make_unique<Ort::Session>(ort_env(), model_path.string().c_str(), session_options);
#endif

  Ort::AllocatorWithDefaultOptions allocator;
  input_name_ = session_->GetInputNameAllocated(0, allocator).get();
  output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
  loaded_ = true;
  return true;
}

cv::Mat OnnxInferSession::run(const cv::Mat& nchw_blob, const std::vector<int64_t>& input_shape) {
  if (!loaded_ || !session_ || nchw_blob.empty()) {
    return {};
  }

  const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> shape = input_shape;
  if (shape.empty()) {
    shape = {1, nchw_blob.size[0], nchw_blob.size[1], nchw_blob.size[2]};
  }

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float*>(nchw_blob.ptr<float>()), nchw_blob.total(), shape.data(),
      shape.size());

  const char* input_names[] = {input_name_.c_str()};
  const char* output_names[] = {output_name_.c_str()};

  auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names,
                               1);
  if (outputs.empty() || !outputs.front().IsTensor()) {
    return {};
  }

  auto& output = outputs.front();
  const auto type_info = output.GetTensorTypeAndShapeInfo();
  const auto out_shape = type_info.GetShape();
  const float* data = output.GetTensorData<float>();

  const int dims = static_cast<int>(out_shape.size());
  if (dims <= 0) {
    return {};
  }

  std::vector<int> cv_sizes;
  cv_sizes.reserve(dims);
  for (int i = 0; i < dims; ++i) {
    cv_sizes.push_back(static_cast<int>(out_shape[i]));
  }

  cv::Mat result(dims, cv_sizes.data(), CV_32F);
  std::memcpy(result.ptr<float>(), data, type_info.GetElementCount() * sizeof(float));
  return result;
}

void log_available_onnx_providers() {
  std::cerr << "ONNX Runtime providers:";
  const auto providers = Ort::GetAvailableProviders();
  if (providers.empty()) {
    std::cerr << " (none)\n";
    return;
  }
  for (const auto& provider : providers) {
    std::cerr << ' ' << provider;
  }
  std::cerr << '\n';
}

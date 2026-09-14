#include "jersey_ocr.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

bool JerseyOcr::load(const std::filesystem::path& model_path, DnnBackend backend) {
  if (!std::filesystem::exists(model_path)) {
    loaded_ = false;
    return false;
  }
  loaded_ = session_.load(model_path, backend);
  return loaded_;
}

std::optional<JerseyOcrResult> JerseyOcr::recognize_heuristic(const cv::Mat& frame,
                                                             const cv::Rect2d& bbox) const {
  const cv::Rect rect(bbox);
  const cv::Rect jersey(rect.x, rect.y + rect.height / 4, rect.width,
                        std::max(1, rect.height / 3));
  const cv::Rect clipped = jersey & cv::Rect(0, 0, frame.cols, frame.rows);
  if (clipped.area() <= 100) return std::nullopt;

  cv::Mat crop = frame(clipped);
  cv::Mat gray;
  cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
  cv::Mat bw;
  cv::adaptiveThreshold(gray, bw, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11,
                        2);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(bw, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  int blob_count = 0;
  for (const auto& c : contours) {
    const cv::Rect r = cv::boundingRect(c);
    const double aspect = static_cast<double>(r.height) / std::max(1, r.width);
    if (r.height > clipped.height * 0.25 && aspect > 1.2 && aspect < 5.0) {
      ++blob_count;
    }
  }

  if (blob_count >= 1 && blob_count <= 2) {
    JerseyOcrResult result;
    result.number = blob_count == 1 ? 1 : 10;
    result.confidence = 0.35f;
    return result;
  }
  return std::nullopt;
}

std::optional<JerseyOcrResult> JerseyOcr::recognize(const cv::Mat& frame,
                                                    const cv::Rect2d& bbox) {
  if (bbox.width < 20 || bbox.height < 40) return std::nullopt;

  if (loaded_) {
    const cv::Rect rect(bbox);
    const cv::Rect jersey(rect.x, rect.y + rect.height / 4, rect.width,
                          std::max(1, rect.height / 3));
    const cv::Rect clipped = jersey & cv::Rect(0, 0, frame.cols, frame.rows);
    if (clipped.area() <= 0) return recognize_heuristic(frame, bbox);

    cv::Mat crop = frame(clipped);
    cv::Mat blob = cv::dnn::blobFromImage(crop, 1.0 / 255.0, cv::Size(64, 64));
    const std::vector<int64_t> input_shape = {1, blob.size[0], blob.size[1], blob.size[2]};
    cv::Mat out = session_.run(blob, input_shape);
    if (out.empty()) return recognize_heuristic(frame, bbox);

    cv::Point max_loc;
    double max_val = 0.0;
    cv::minMaxLoc(out.reshape(1, 1), nullptr, &max_val, nullptr, &max_loc);
    if (max_val < 0.3) return recognize_heuristic(frame, bbox);

    JerseyOcrResult result;
    result.number = max_loc.x;
    result.confidence = static_cast<float>(max_val);
    return result;
  }

  return recognize_heuristic(frame, bbox);
}

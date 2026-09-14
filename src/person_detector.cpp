#include "person_detector.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace {

struct Letterbox {
  float scale{};
  int pad_x{};
  int pad_y{};
};

Letterbox compute_letterbox(int width, int height, int input_size) {
  const float scale = std::min(static_cast<float>(input_size) / width,
                               static_cast<float>(input_size) / height);
  const int new_w = static_cast<int>(std::round(width * scale));
  const int new_h = static_cast<int>(std::round(height * scale));
  return {scale, (input_size - new_w) / 2, (input_size - new_h) / 2};
}

cv::Mat letterbox_image(const cv::Mat& frame, const Letterbox& lb, int input_size) {
  cv::Mat resized;
  cv::resize(frame, resized,
             cv::Size(static_cast<int>(std::round(frame.cols * lb.scale)),
                      static_cast<int>(std::round(frame.rows * lb.scale))));
  cv::Mat output(input_size, input_size, frame.type(), cv::Scalar(114, 114, 114));
  resized.copyTo(output(cv::Rect(lb.pad_x, lb.pad_y, resized.cols, resized.rows)));
  return output;
}

cv::Rect2d to_image_box(float cx, float cy, float w, float h, const Letterbox& lb) {
  const float x1 = (cx - w * 0.5f - static_cast<float>(lb.pad_x)) / lb.scale;
  const float y1 = (cy - h * 0.5f - static_cast<float>(lb.pad_y)) / lb.scale;
  const float x2 = (cx + w * 0.5f - static_cast<float>(lb.pad_x)) / lb.scale;
  const float y2 = (cy + h * 0.5f - static_cast<float>(lb.pad_y)) / lb.scale;
  return {x1, y1, x2 - x1, y2 - y1};
}

}  // namespace

bool PersonDetector::load(const std::filesystem::path& model_path, DnnBackend backend,
                          int input_size) {
  if (input_size < 320 || input_size > 1920) {
    std::cerr << "Invalid input size: " << input_size << " (use 320-1920)\n";
    return false;
  }

  if (!session_.load(model_path, backend)) {
    return false;
  }

  input_size_ = input_size;
  backend_name_ = session_.provider_name();
  loaded_ = true;
  log_available_onnx_providers();
  std::cerr << "Person detector: " << model_path.filename().string() << ", input " << input_size_
            << "px, provider " << backend_name_ << "\n";
  return true;
}

std::vector<Detection> PersonDetector::detect(const cv::Mat& frame, float confidence_threshold,
                                              int max_width) {
  if (!loaded_ || frame.empty()) {
    return {};
  }

  cv::Mat detect_frame = frame;
  double scale_back = 1.0;
  if (max_width > 0 && frame.cols > max_width) {
    scale_back = static_cast<double>(frame.cols) / max_width;
    cv::resize(frame, detect_frame, cv::Size(max_width, cvRound(frame.rows / scale_back)), 0, 0,
               cv::INTER_LINEAR);
  }

  const Letterbox lb = compute_letterbox(detect_frame.cols, detect_frame.rows, input_size_);
  const cv::Mat input = letterbox_image(detect_frame, lb, input_size_);
  cv::Mat blob =
      cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
  const std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_};
  cv::Mat output = session_.run(blob, input_shape);

  if (output.dims != 3) {
    std::cerr << "Unexpected YOLO output dimensions: " << output.dims << "\n";
    return {};
  }

  const int num_classes = output.size[1] - 4;
  const int num_preds = output.size[2];
  const bool channels_first = output.size[1] < output.size[2];

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (int i = 0; i < num_preds; ++i) {
    float cx{};
    float cy{};
    float w{};
    float h{};
    float best_score = 0.0f;
    int best_class = -1;

    if (channels_first) {
      cx = output.at<float>(0, 0, i);
      cy = output.at<float>(0, 1, i);
      w = output.at<float>(0, 2, i);
      h = output.at<float>(0, 3, i);
      for (int c = 0; c < num_classes; ++c) {
        const float score = output.at<float>(0, 4 + c, i);
        if (score > best_score) {
          best_score = score;
          best_class = c;
        }
      }
    } else {
      cx = output.at<float>(0, i, 0);
      cy = output.at<float>(0, i, 1);
      w = output.at<float>(0, i, 2);
      h = output.at<float>(0, i, 3);
      for (int c = 0; c < num_classes; ++c) {
        const float score = output.at<float>(0, i, 4 + c);
        if (score > best_score) {
          best_score = score;
          best_class = c;
        }
      }
    }

    if (best_class != kPersonClassId || best_score < confidence_threshold) {
      continue;
    }

    const cv::Rect2d image_box = to_image_box(cx, cy, w, h, lb);
    cv::Rect2d scaled_box = image_box;
    if (scale_back != 1.0) {
      scaled_box.x *= scale_back;
      scaled_box.y *= scale_back;
      scaled_box.width *= scale_back;
      scaled_box.height *= scale_back;
    }
    const cv::Rect clipped(cv::Rect(scaled_box) & cv::Rect(0, 0, frame.cols, frame.rows));
    if (clipped.width <= 1 || clipped.height <= 1) {
      continue;
    }

    class_ids.push_back(best_class);
    confidences.push_back(best_score);
    boxes.push_back(clipped);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, 0.55f, indices);

  std::vector<Detection> detections;
  detections.reserve(indices.size());
  for (const int idx : indices) {
    Detection det;
    det.box = boxes[idx];
    det.confidence = confidences[idx];
    det.class_id = class_ids[idx];
    detections.push_back(det);
  }

  std::sort(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
  return detections;
}

double detection_iou(const cv::Rect2d& a, const cv::Rect2d& b) {
  const double x1 = std::max(a.x, b.x);
  const double y1 = std::max(a.y, b.y);
  const double x2 = std::min(a.x + a.width, b.x + b.width);
  const double y2 = std::min(a.y + a.height, b.y + b.height);
  const double intersection = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
  const double union_area = a.area() + b.area() - intersection;
  if (union_area <= 0.0) {
    return 0.0;
  }
  return intersection / union_area;
}

std::vector<Detection> merge_detections(const std::vector<Detection>& detections,
                                        float iou_threshold, int min_hits) {
  std::vector<Detection> sorted = detections;
  std::sort(sorted.begin(), sorted.end(),
            [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

  struct Cluster {
    Detection best;
    int hits{1};
  };
  std::vector<Cluster> clusters;

  for (const Detection& det : sorted) {
    bool merged = false;
    for (Cluster& cluster : clusters) {
      if (detection_iou(det.box, cluster.best.box) >= iou_threshold) {
        cluster.hits++;
        if (det.confidence > cluster.best.confidence) {
          cluster.best = det;
        }
        merged = true;
        break;
      }
    }
    if (!merged) {
      clusters.push_back({det, 1});
    }
  }

  std::vector<Detection> merged;
  merged.reserve(clusters.size());
  for (const Cluster& cluster : clusters) {
    if (cluster.hits >= min_hits) {
      merged.push_back(cluster.best);
    }
  }
  std::sort(merged.begin(), merged.end(),
            [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
  return merged;
}

DetectionScanResult detect_persons_multi_frame(PersonDetector& detector, cv::VideoCapture& cap,
                                               const cv::Mat& first_frame, int frame_count,
                                               int frame_step, float confidence_threshold,
                                               int min_hits) {
  DetectionScanResult result;
  frame_count = std::max(1, frame_count);
  frame_step = std::max(1, frame_step);
  min_hits = std::max(1, min_hits);

  std::vector<Detection> all_detections;
  DetectionScanFrame scan_frame;
  scan_frame.frame_index = 0;
  scan_frame.image = first_frame.clone();
  scan_frame.detections = detector.detect(first_frame, confidence_threshold);
  all_detections.insert(all_detections.end(), scan_frame.detections.begin(),
                        scan_frame.detections.end());
  result.frames.push_back(std::move(scan_frame));
  result.frames_scanned = 1;

  for (int i = 1; i < frame_count; ++i) {
    const int target_frame = i * frame_step;
    cap.set(cv::CAP_PROP_POS_FRAMES, target_frame);
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
      break;
    }
    DetectionScanFrame next_frame;
    next_frame.frame_index = target_frame;
    next_frame.image = frame.clone();
    next_frame.detections = detector.detect(frame, confidence_threshold);
    all_detections.insert(all_detections.end(), next_frame.detections.begin(),
                          next_frame.detections.end());
    result.frames.push_back(std::move(next_frame));
    ++result.frames_scanned;
  }

  result.merged = merge_detections(all_detections, 0.55f, min_hits);
  std::cerr << "Multi-frame detect: scanned " << result.frames_scanned << " frame(s), raw "
            << all_detections.size() << " detection(s), merged " << result.merged.size()
            << " (min_hits=" << min_hits << ")\n";

  cap.set(cv::CAP_PROP_POS_FRAMES, 1);
  return result;
}

std::optional<Detection> find_detection_near_box(const std::vector<Detection>& detections,
                                                 const cv::Rect2d& reference_box,
                                                 float min_iou) {
  std::optional<Detection> best;
  float best_iou = min_iou;
  for (const Detection& det : detections) {
    const float iou = static_cast<float>(detection_iou(det.box, reference_box));
    if (iou > best_iou) {
      best_iou = iou;
      best = det;
    }
  }
  return best;
}

#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "person_detector.hpp"

int main(int argc, char** argv) {
  const std::string video = argc > 1 ? argv[1] : "test/20260912-2257-00.7889788.mp4";
  const std::string model = argc > 2 ? argv[2] : "models/yolov8s.onnx";

  cv::VideoCapture cap(video);
  if (!cap.isOpened()) {
    std::cerr << "Failed to open: " << video << "\n";
    return 1;
  }

  cv::Mat frame;
  if (!cap.read(frame) || frame.empty()) {
    std::cerr << "Failed to read first frame\n";
    return 1;
  }

  PersonDetector detector;
  const int input_size = argc > 4 ? std::stoi(argv[4]) : 640;
  if (!detector.load(model, DnnBackend::Auto, input_size)) {
    return 1;
  }

  const float confidence = argc > 3 ? std::stof(argv[3]) : 0.40f;
  const auto detections = detector.detect(frame, confidence);
  std::cout << "Backend: " << detector.backend_name() << "\n";
  std::cout << "Person detections: " << detections.size() << "\n";
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto& d = detections[i];
    std::cout << "  D" << (i + 1) << " conf=" << d.confidence << " box=(" << d.box.x << ","
              << d.box.y << "," << d.box.width << "," << d.box.height << ")\n";
  }

  return detections.empty() ? 2 : 0;
}

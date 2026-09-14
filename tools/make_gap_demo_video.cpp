#include <cmath>
#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

int main(int argc, char** argv) {
  const std::string output = argc > 1 ? argv[1] : "test/gap_demo.mp4";
  constexpr int width = 960;
  constexpr int height = 540;
  constexpr int fps = 30;
  constexpr int frames = 150;
  constexpr int gap_start = 42;
  constexpr int gap_end = 78;

  cv::VideoWriter writer(output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, {width, height});
  if (!writer.isOpened()) {
    std::cerr << "Failed to open output: " << output << "\n";
    return 1;
  }

  for (int i = 0; i < frames; ++i) {
    cv::Mat frame(height, width, CV_8UC3, cv::Scalar(35, 110, 45));
    cv::rectangle(frame, {0, height - 80, width, 80}, cv::Scalar(50, 130, 55), cv::FILLED);
    cv::line(frame, {0, height - 80}, {width, height - 80}, {255, 255, 255}, 2);

    const int x = 60 + i * 4;
    const int y = 260 + static_cast<int>(12 * std::sin(i * 0.12));
    const bool visible = i < gap_start || i > gap_end;
    if (visible) {
      cv::rectangle(frame, {x, y, 50, 110}, {0, 0, 230}, cv::FILLED);
      cv::circle(frame, {x + 25, y + 25}, 14, {30, 30, 200}, cv::FILLED);
    }

    if (i >= gap_start && i <= gap_end) {
      cv::rectangle(frame, {320, 120, 120, 300}, cv::Scalar(20, 20, 20), cv::FILLED);
      cv::putText(frame, "OCCLUSION", {335, 280}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {180, 180, 180},
                  2, cv::LINE_AA);
    }

    char label[48];
    std::snprintf(label, sizeof(label), "frame %d", i);
    cv::putText(frame, label, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 255, 255}, 2,
                cv::LINE_AA);
    writer.write(frame);
  }

  std::cout << "Wrote " << frames << " frames to " << output << " (gap " << gap_start << "-"
            << gap_end << ")\n";
  return 0;
}

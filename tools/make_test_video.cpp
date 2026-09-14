#include <cmath>
#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

int main(int argc, char** argv) {
  const std::string output = argc > 1 ? argv[1] : "test/run.mp4";
  constexpr int width = 640;
  constexpr int height = 480;
  constexpr int fps = 30;
  constexpr int frames = 90;

  cv::VideoWriter writer(output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, {width, height});
  if (!writer.isOpened()) {
    std::cerr << "Failed to open output: " << output << "\n";
    return 1;
  }

  for (int i = 0; i < frames; ++i) {
    cv::Mat frame(height, width, CV_8UC3, cv::Scalar(40, 120, 40));
    const int x = 40 + i * 4;
    const int y = 200 + static_cast<int>(10 * std::sin(i * 0.15));
    cv::rectangle(frame, {x, y, 60, 120}, {0, 0, 220}, cv::FILLED);
    writer.write(frame);
  }

  std::cout << "Wrote " << frames << " frames to " << output << "\n";
  return 0;
}

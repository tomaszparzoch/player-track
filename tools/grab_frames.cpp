#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: grab_frames <video> <output_dir> <frame_index> [...]\n";
    return 1;
  }

  cv::VideoCapture cap(argv[1]);
  if (!cap.isOpened()) {
    std::cerr << "Failed to open: " << argv[1] << "\n";
    return 1;
  }

  const std::string out_dir = argv[2];
  cv::Mat frame;
  for (int i = 3; i < argc; ++i) {
    const int index = std::stoi(argv[i]);
    cap.set(cv::CAP_PROP_POS_FRAMES, index);
    if (!cap.grab() || !cap.retrieve(frame) || frame.empty()) {
      std::cerr << "Failed to read frame " << index << "\n";
      continue;
    }
    const std::string path = out_dir + "/frame_" + std::to_string(index) + ".jpg";
    if (!cv::imwrite(path, frame)) {
      std::cerr << "Failed to write: " << path << "\n";
      return 1;
    }
    std::cout << path << "\n";
  }
  return 0;
}

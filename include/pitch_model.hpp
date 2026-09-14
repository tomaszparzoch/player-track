#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct PitchLandmark {
  std::string id;
  std::string label;
  cv::Point2d field_m;
};

struct PitchModel {
  double length_m{105.0};
  double width_m{68.0};

  std::vector<PitchLandmark> landmarks() const;
  std::optional<PitchLandmark> find_landmark(const std::string& id) const;
};

PitchModel default_pitch_model(double length_m = 105.0, double width_m = 68.0);

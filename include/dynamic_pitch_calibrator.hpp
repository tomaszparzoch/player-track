#pragma once

#include <optional>

#include <opencv2/core.hpp>

#include "pitch_calibration.hpp"
#include "pitch_model.hpp"

struct DynamicCalibResult {
  cv::Mat homography;
  float confidence{0.0f};
  double reproj_error_px{9999.0};
  bool valid{false};
};

class DynamicPitchCalibrator {
 public:
  explicit DynamicPitchCalibrator(PitchModel model);

  DynamicCalibResult estimate(const cv::Mat& frame) const;
  DynamicCalibResult smooth(const DynamicCalibResult& current,
                            const DynamicCalibResult& previous) const;

  void reset();

 private:
  PitchModel model_;
  cv::Mat last_H_;
  bool has_last_{false};
};

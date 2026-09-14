#include "dynamic_pitch_calibrator.hpp"

#include <algorithm>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace {

std::vector<cv::Point2f> detect_line_intersections(const cv::Mat& frame) {
  cv::Mat gray;
  if (frame.channels() == 3) {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = frame;
  }

  cv::Mat mask;
  cv::inRange(gray, 180, 255, mask);
  cv::Mat edges;
  cv::Canny(mask, edges, 50, 150);
  std::vector<cv::Vec4i> lines;
  cv::HoughLinesP(edges, lines, 1, CV_PI / 180.0, 80, 50, 10);

  std::vector<cv::Point2f> points;
  for (size_t i = 0; i < lines.size(); ++i) {
    for (size_t j = i + 1; j < lines.size(); ++j) {
      const cv::Point2f p1(lines[i][0], lines[i][1]);
      const cv::Point2f p2(lines[i][2], lines[i][3]);
      const cv::Point2f p3(lines[j][0], lines[j][1]);
      const cv::Point2f p4(lines[j][2], lines[j][3]);
      const cv::Point2f da = p2 - p1;
      const cv::Point2f db = p4 - p3;
      const float denom = da.x * db.y - da.y * db.x;
      if (std::abs(denom) < 1e-3f) continue;
      const cv::Point2f d = p3 - p1;
      const float t = (d.x * db.y - d.y * db.x) / denom;
      const cv::Point2f inter = p1 + t * da;
      if (inter.x >= 0 && inter.y >= 0 && inter.x < frame.cols && inter.y < frame.rows) {
        points.push_back(inter);
      }
    }
  }
  return points;
}

double reproj_error(const cv::Mat& H, const std::vector<cv::Point2f>& field_pts,
                    const std::vector<cv::Point2f>& image_pts) {
  std::vector<cv::Point2f> projected;
  cv::perspectiveTransform(field_pts, projected, H);
  double sum = 0.0;
  for (size_t i = 0; i < image_pts.size(); ++i) {
    sum += cv::norm(image_pts[i] - projected[i]);
  }
  return sum / static_cast<double>(image_pts.size());
}

}  // namespace

DynamicPitchCalibrator::DynamicPitchCalibrator(PitchModel model) : model_(std::move(model)) {}

DynamicCalibResult DynamicPitchCalibrator::estimate(const cv::Mat& frame) const {
  DynamicCalibResult result;
  if (frame.empty()) return result;

  const auto landmarks = model_.landmarks();
  const auto intersections = detect_line_intersections(frame);
  if (intersections.size() < 4) return result;

  std::vector<cv::Point2f> field_pts;
  std::vector<cv::Point2f> image_pts;
  const size_t count = std::min(intersections.size(), landmarks.size());
  for (size_t i = 0; i < count; ++i) {
    field_pts.push_back({static_cast<float>(landmarks[i].field_m.x),
                         static_cast<float>(landmarks[i].field_m.y)});
    image_pts.push_back(intersections[i]);
  }

  if (field_pts.size() < 4) return result;

  result.homography = cv::findHomography(field_pts, image_pts, cv::RANSAC, 5.0);
  if (result.homography.empty()) return result;

  result.reproj_error_px = reproj_error(result.homography, field_pts, image_pts);
  result.valid = true;
  result.confidence = static_cast<float>(
      std::clamp(1.0 - result.reproj_error_px / 40.0, 0.0, 1.0));
  return result;
}

DynamicCalibResult DynamicPitchCalibrator::smooth(const DynamicCalibResult& current,
                                                  const DynamicCalibResult& previous) const {
  if (!current.valid) return previous;
  if (!previous.valid) return current;

  DynamicCalibResult out = current;
  out.homography = 0.7 * current.homography + 0.3 * previous.homography;
  out.confidence = 0.7f * current.confidence + 0.3f * previous.confidence;
  return out;
}

void DynamicPitchCalibrator::reset() {
  has_last_ = false;
  last_H_.release();
}

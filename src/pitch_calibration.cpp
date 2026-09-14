#include "pitch_calibration.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {

struct RectCalibrateUi {
  std::vector<cv::Point2f> clicks;
  cv::Mat frame;
};

struct LineCalibrateUi {
  cv::Mat frame;
  PitchModel model;
  std::vector<std::string> landmark_ids;
  std::vector<cv::Point2f> image_points;
  std::vector<cv::Point2f> field_points;
  int current_index{0};
};

void on_rect_mouse(int event, int x, int y, int, void* userdata) {
  if (event != cv::EVENT_LBUTTONDOWN || userdata == nullptr) return;
  auto* ui = static_cast<RectCalibrateUi*>(userdata);
  if (ui->clicks.size() >= 4) return;
  ui->clicks.emplace_back(static_cast<float>(x), static_cast<float>(y));
}

void on_line_mouse(int event, int x, int y, int, void* userdata) {
  if (event != cv::EVENT_LBUTTONDOWN || userdata == nullptr) return;
  auto* ui = static_cast<LineCalibrateUi*>(userdata);
  if (ui->current_index >= static_cast<int>(ui->landmark_ids.size())) return;

  const auto lm = ui->model.find_landmark(ui->landmark_ids[ui->current_index]);
  if (!lm) return;

  ui->image_points.push_back({static_cast<float>(x), static_cast<float>(y)});
  ui->field_points.push_back(
      {static_cast<float>(lm->field_m.x), static_cast<float>(lm->field_m.y)});
  ++ui->current_index;
}

void draw_rect_ui(cv::Mat& canvas, const std::vector<cv::Point2f>& clicks) {
  const char* labels[] = {"TL", "TR", "BR", "BL"};
  for (size_t i = 0; i < clicks.size(); ++i) {
    const cv::Point p(cvRound(clicks[i].x), cvRound(clicks[i].y));
    cv::circle(canvas, p, 6, {0, 255, 255}, cv::FILLED);
    cv::putText(canvas, labels[i], {p.x + 8, p.y - 8}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 255},
                2, cv::LINE_AA);
  }
}

void draw_line_ui(cv::Mat& canvas, const LineCalibrateUi& ui) {
  for (size_t i = 0; i < ui.image_points.size(); ++i) {
    const cv::Point p(cvRound(ui.image_points[i].x), cvRound(ui.image_points[i].y));
    cv::circle(canvas, p, 6, {0, 255, 255}, cv::FILLED);
    const std::string label = i < ui.landmark_ids.size() ? ui.landmark_ids[i] : "?";
    cv::putText(canvas, label, {p.x + 8, p.y - 8}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255},
                1, cv::LINE_AA);
  }
}

std::optional<std::pair<double, double>> parse_pitch_rect(std::string_view text) {
  std::stringstream ss{std::string(text)};
  std::string token;
  double values[2]{};
  int i = 0;
  while (std::getline(ss, token, ',')) {
    if (i >= 2) return std::nullopt;
    try {
      values[i++] = std::stod(token);
    } catch (...) {
      return std::nullopt;
    }
  }
  if (i != 2 || values[0] <= 0.0 || values[1] <= 0.0) return std::nullopt;
  return std::make_pair(values[0], values[1]);
}

double compute_reproj_error(const cv::Mat& H, const std::vector<cv::Point2f>& image_pts,
                            const std::vector<cv::Point2f>& field_pts) {
  if (image_pts.empty() || image_pts.size() != field_pts.size()) return 9999.0;
  std::vector<cv::Point2f> projected;
  cv::perspectiveTransform(field_pts, projected, H);
  double sum = 0.0;
  for (size_t i = 0; i < image_pts.size(); ++i) {
    sum += cv::norm(image_pts[i] - projected[i]);
  }
  return sum / static_cast<double>(image_pts.size());
}

}  // namespace

bool PitchCalibration::build_homography_rect(const std::vector<cv::Point2f>& image_pts,
                                             double pitch_length_m, double pitch_width_m) {
  if (image_pts.size() != 4 || pitch_length_m <= 0.0 || pitch_width_m <= 0.0) return false;

  const std::vector<cv::Point2f> field_pts{
      {0.0f, 0.0f},
      {static_cast<float>(pitch_length_m), 0.0f},
      {static_cast<float>(pitch_length_m), static_cast<float>(pitch_width_m)},
      {0.0f, static_cast<float>(pitch_width_m)},
  };
  return build_homography_points(image_pts, field_pts, pitch_length_m, pitch_width_m);
}

bool PitchCalibration::build_homography_points(const std::vector<cv::Point2f>& image_pts,
                                               const std::vector<cv::Point2f>& field_pts,
                                               double pitch_length_m, double pitch_width_m) {
  if (image_pts.size() < 4 || image_pts.size() != field_pts.size()) return false;

  homography_ = cv::findHomography(field_pts, image_pts, cv::RANSAC, 3.0);
  if (homography_.empty()) {
    homography_ = cv::getPerspectiveTransform(field_pts, image_pts);
  }
  pitch_length_m_ = pitch_length_m;
  pitch_width_m_ = pitch_width_m;
  image_points_ = image_pts;
  field_points_ = field_pts;
  reproj_error_px_ = compute_reproj_error(homography_, image_pts, field_pts);
  valid_ = !homography_.empty();
  return valid_;
}

bool PitchCalibration::calibrate_interactive(const cv::Mat& frame, double pitch_length_m,
                                             double pitch_width_m) {
  if (frame.empty() || pitch_length_m <= 0.0 || pitch_width_m <= 0.0) return false;

  RectCalibrateUi ui;
  ui.frame = frame.clone();
  const std::string window = "Calibrate pitch: click TL, TR, BR, BL, then Enter";

  cv::namedWindow(window, cv::WINDOW_NORMAL);
  cv::setMouseCallback(window, on_rect_mouse, &ui);

  while (true) {
    cv::Mat canvas = ui.frame.clone();
    draw_rect_ui(canvas, ui.clicks);
    cv::putText(canvas, "Click 4 pitch corners: TL, TR, BR, BL", {20, 30},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 2, cv::LINE_AA);
    cv::imshow(window, canvas);

    const int key = cv::waitKey(30);
    if (key == 27) {
      cv::destroyWindow(window);
      return false;
    }
    if ((key == 13 || key == 32) && ui.clicks.size() == 4) break;
  }

  cv::destroyWindow(window);
  return build_homography_rect(ui.clicks, pitch_length_m, pitch_width_m);
}

bool PitchCalibration::calibrate_lines_interactive(const cv::Mat& frame, const PitchModel& model) {
  if (frame.empty()) return false;

  LineCalibrateUi ui;
  ui.frame = frame.clone();
  ui.model = model;
  for (const auto& lm : model.landmarks()) {
    ui.landmark_ids.push_back(lm.id);
  }

  const std::string window = "Line calibration: click landmarks, Enter when >=4, Esc cancel";
  cv::namedWindow(window, cv::WINDOW_NORMAL);
  cv::setMouseCallback(window, on_line_mouse, &ui);

  while (true) {
    cv::Mat canvas = ui.frame.clone();
    draw_line_ui(canvas, ui);
    std::string status = "Points: " + std::to_string(ui.image_points.size()) + " (min 4)";
    if (ui.current_index < static_cast<int>(ui.landmark_ids.size())) {
      const auto lm = model.find_landmark(ui.landmark_ids[ui.current_index]);
      if (lm) status += " | Next: " + lm->label;
    }
    cv::putText(canvas, status, {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 2,
                cv::LINE_AA);
    cv::putText(canvas, "Keys: Enter=finish, Backspace=undo, Esc=cancel", {20, 60},
                cv::FONT_HERSHEY_SIMPLEX, 0.55, {200, 200, 200}, 1, cv::LINE_AA);
    cv::imshow(window, canvas);

    const int key = cv::waitKey(30);
    if (key == 27) {
      cv::destroyWindow(window);
      return false;
    }
    if (key == 8 && !ui.image_points.empty()) {
      ui.image_points.pop_back();
      ui.field_points.pop_back();
      ui.current_index = std::max(0, ui.current_index - 1);
    }
    if ((key == 13 || key == 32) && ui.image_points.size() >= 4) break;
  }

  cv::destroyWindow(window);
  return build_homography_points(ui.image_points, ui.field_points, model.length_m, model.width_m);
}

bool PitchCalibration::save(const std::filesystem::path& path) const {
  if (!valid_ || image_points_.empty()) return false;

  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) return false;

  out << "pitch_length_m=" << pitch_length_m_ << '\n';
  out << "pitch_width_m=" << pitch_width_m_ << '\n';
  out << "reproj_error_px=" << reproj_error_px_ << '\n';
  for (size_t i = 0; i < image_points_.size(); ++i) {
    out << "point=" << image_points_[i].x << ',' << image_points_[i].y << ','
        << field_points_[i].x << ',' << field_points_[i].y << '\n';
  }
  return true;
}

bool PitchCalibration::load(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) return false;

  double pitch_length_m = 0.0;
  double pitch_width_m = 0.0;
  std::vector<cv::Point2f> image_pts;
  std::vector<cv::Point2f> field_pts;

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;

    if (line.rfind("pitch_length_m=", 0) == 0) {
      pitch_length_m = std::stod(line.substr(15));
      continue;
    }
    if (line.rfind("pitch_width_m=", 0) == 0) {
      pitch_width_m = std::stod(line.substr(14));
      continue;
    }
    if (line.rfind("pitch_rect_m=", 0) == 0) {
      const auto rect = parse_pitch_rect(line.substr(13));
      if (!rect) return false;
      pitch_length_m = rect->first;
      pitch_width_m = rect->second;
      continue;
    }
    if (line.rfind("reproj_error_px=", 0) == 0) continue;

    if (line.rfind("point=", 0) == 0) {
      std::stringstream ss(line.substr(6));
      std::string token;
      double values[4]{};
      int i = 0;
      while (std::getline(ss, token, ',')) {
        if (i >= 4) return false;
        values[i++] = std::stod(token);
      }
      if (i != 4) return false;
      image_pts.emplace_back(static_cast<float>(values[0]), static_cast<float>(values[1]));
      field_pts.emplace_back(static_cast<float>(values[2]), static_cast<float>(values[3]));
    }
  }

  if (image_pts.size() < 4) return false;
  if (pitch_length_m <= 0.0) pitch_length_m = field_pts.empty() ? 105.0 : field_pts[1].x;
  if (pitch_width_m <= 0.0) pitch_width_m = field_pts.empty() ? 68.0 : field_pts[2].y;

  return build_homography_points(image_pts, field_pts, pitch_length_m, pitch_width_m);
}

cv::Point2d PitchCalibration::image_to_meters(double x_px, double y_px) const {
  if (!valid_) return {x_px, y_px};

  const cv::Mat src = (cv::Mat_<double>(3, 1) << x_px, y_px, 1.0);
  cv::Mat dst = homography_.inv() * src;
  const double w = dst.at<double>(2, 0);
  if (std::abs(w) < 1e-9) return {0.0, 0.0};
  return {dst.at<double>(0, 0) / w, dst.at<double>(1, 0) / w};
}

cv::Point2f PitchCalibration::meters_to_image(double x_m, double y_m) const {
  if (!valid_) return {static_cast<float>(x_m), static_cast<float>(y_m)};

  const cv::Mat src = (cv::Mat_<double>(3, 1) << x_m, y_m, 1.0);
  cv::Mat dst = homography_ * src;
  const double w = dst.at<double>(2, 0);
  if (std::abs(w) < 1e-9) return {0.0f, 0.0f};
  return {static_cast<float>(dst.at<double>(0, 0) / w),
          static_cast<float>(dst.at<double>(1, 0) / w)};
}

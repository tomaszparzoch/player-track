#include "pitch_heatmap.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

bool write_pitch_heatmap_png(const std::filesystem::path& path,
                             const std::vector<cv::Point2d>& positions_m, double pitch_length_m,
                             double pitch_width_m, int width_px, int height_px) {
  if (positions_m.empty() || pitch_length_m <= 0.0 || pitch_width_m <= 0.0) return false;

  cv::Mat heat(height_px, width_px, CV_32FC1, cv::Scalar(0));
  const double sigma = 8.0;

  for (const auto& pt : positions_m) {
    const int cx = static_cast<int>(pt.x / pitch_length_m * width_px);
    const int cy = static_cast<int>(pt.y / pitch_width_m * height_px);
    const int radius = static_cast<int>(3.0 * sigma);
    for (int y = std::max(0, cy - radius); y < std::min(height_px, cy + radius + 1); ++y) {
      for (int x = std::max(0, cx - radius); x < std::min(width_px, cx + radius + 1); ++x) {
        const double dx = x - cx;
        const double dy = y - cy;
        heat.at<float>(y, x) += static_cast<float>(std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma)));
      }
    }
  }

  double max_val = 0.0;
  cv::minMaxLoc(heat, nullptr, &max_val);
  if (max_val <= 0.0) return false;

  cv::Mat normalized;
  heat.convertTo(normalized, CV_8UC1, 255.0 / max_val);
  cv::Mat color;
  cv::applyColorMap(normalized, color, cv::COLORMAP_JET);

  cv::Mat pitch(height_px, width_px, CV_8UC3, cv::Scalar(30, 120, 30));
  cv::addWeighted(pitch, 0.45, color, 0.55, 0.0, color);

  cv::line(color, {0, 0}, {width_px, 0}, {255, 255, 255}, 2);
  cv::line(color, {0, height_px - 1}, {width_px, height_px - 1}, {255, 255, 255}, 2);
  cv::line(color, {width_px / 2, 0}, {width_px / 2, height_px}, {255, 255, 255}, 2);

  return cv::imwrite(path.string(), color, {cv::IMWRITE_PNG_COMPRESSION, 9});
}

bool write_image_heatmap_png(const std::filesystem::path& path,
                             const std::vector<cv::Point2d>& positions_px, int frame_width_px,
                             int frame_height_px) {
  if (positions_px.empty() || frame_width_px <= 0 || frame_height_px <= 0) return false;

  const int width_px = std::max(320, frame_width_px);
  const int height_px = std::max(240, frame_height_px);
  cv::Mat heat(height_px, width_px, CV_32FC1, cv::Scalar(0));
  const double sigma = 8.0;

  for (const auto& pt : positions_px) {
    const int cx = std::clamp(static_cast<int>(std::round(pt.x)), 0, width_px - 1);
    const int cy = std::clamp(static_cast<int>(std::round(pt.y)), 0, height_px - 1);
    const int radius = static_cast<int>(3.0 * sigma);
    for (int y = std::max(0, cy - radius); y < std::min(height_px, cy + radius + 1); ++y) {
      for (int x = std::max(0, cx - radius); x < std::min(width_px, cx + radius + 1); ++x) {
        const double dx = x - cx;
        const double dy = y - cy;
        heat.at<float>(y, x) +=
            static_cast<float>(std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)));
      }
    }
  }

  double max_val = 0.0;
  cv::minMaxLoc(heat, nullptr, &max_val);
  if (max_val <= 0.0) return false;

  cv::Mat normalized;
  heat.convertTo(normalized, CV_8UC1, 255.0 / max_val);
  cv::Mat color;
  cv::applyColorMap(normalized, color, cv::COLORMAP_JET);

  cv::Mat backdrop(height_px, width_px, CV_8UC3, cv::Scalar(32, 32, 32));
  cv::addWeighted(backdrop, 0.35, color, 0.65, 0.0, color);
  cv::rectangle(color, {0, 0, width_px - 1, height_px - 1}, {180, 180, 180}, 2);

  return cv::imwrite(path.string(), color, {cv::IMWRITE_PNG_COMPRESSION, 9});
}

bool write_pitch_event_diagram_png(const std::filesystem::path& path,
                                   const std::vector<PitchEventMark>& marks,
                                   double pitch_length_m, double pitch_width_m, bool pixel_mode,
                                   int width_px, int height_px) {
  if (width_px < 80 || height_px < 80) return false;

  cv::Mat img(height_px, width_px, CV_8UC3, cv::Scalar(28, 110, 32));
  const int pad = 18;
  const cv::Rect field(pad, pad, width_px - 2 * pad, height_px - 2 * pad - 22);
  cv::rectangle(img, field, cv::Scalar(255, 255, 255), 2);
  cv::line(img, {field.x + field.width / 2, field.y},
           {field.x + field.width / 2, field.y + field.height}, cv::Scalar(255, 255, 255), 1);
  cv::circle(img, {field.x + field.width / 2, field.y + field.height / 2},
             std::max(8, field.height / 8), cv::Scalar(255, 255, 255), 1);

  auto to_px = [&](const cv::Point2d& pos) -> cv::Point {
    double nx = 0.5;
    double ny = 0.5;
    if (pixel_mode) {
      nx = pos.x / std::max(1.0, pitch_length_m);
      ny = pos.y / std::max(1.0, pitch_width_m);
    } else {
      nx = pos.x / std::max(1e-6, pitch_length_m);
      ny = pos.y / std::max(1e-6, pitch_width_m);
    }
    nx = std::clamp(nx, 0.0, 1.0);
    ny = std::clamp(ny, 0.0, 1.0);
    return {field.x + static_cast<int>(nx * field.width),
            field.y + static_cast<int>(ny * field.height)};
  };

  std::vector<cv::Point> join_pts;
  for (const auto& mark : marks) {
    if (!mark.has_pos) continue;
    const cv::Point px = to_px(mark.pos);
    if (mark.label.rfind("J", 0) == 0) {
      join_pts.push_back(px);
    }
    cv::circle(img, px, 7, mark.color, cv::FILLED);
    cv::circle(img, px, 7, cv::Scalar(0, 0, 0), 1);
    cv::putText(img, mark.label, {px.x + 8, px.y - 6}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  }
  if (join_pts.size() == 2) {
    cv::line(img, join_pts[0], join_pts[1], cv::Scalar(0, 220, 255), 2, cv::LINE_AA);
  }

  cv::putText(img, "D=detect  L=lost  J=join  (1/2 = camera)", {pad, height_px - 8},
              cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
  return cv::imwrite(path.string(), img, {cv::IMWRITE_PNG_COMPRESSION, 9});
}

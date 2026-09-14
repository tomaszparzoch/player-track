#include "selection_ui.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {

constexpr int kMaxDisplayWidth = 1600;
constexpr int kMaxDisplayHeight = 900;

cv::Size fit_window_size(const cv::Size& frame_size) {
  const double scale = std::min(
      1.0, std::min(static_cast<double>(kMaxDisplayWidth) / frame_size.width,
                    static_cast<double>(kMaxDisplayHeight) / frame_size.height));
  return {static_cast<int>(std::round(frame_size.width * scale)),
          static_cast<int>(std::round(frame_size.height * scale))};
}

void setup_display_window(const std::string& window, const cv::Size& frame_size) {
  cv::namedWindow(window, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
  cv::resizeWindow(window, fit_window_size(frame_size));
}

cv::Point window_to_image(const std::string& window, const cv::Size& frame_size, int x, int y) {
  const cv::Rect image_rect = cv::getWindowImageRect(window);
  if (image_rect.width <= 0 || image_rect.height <= 0) {
    return {x, y};
  }
  const int img_x =
      static_cast<int>((x - image_rect.x) * static_cast<double>(frame_size.width) / image_rect.width);
  const int img_y =
      static_cast<int>((y - image_rect.y) * static_cast<double>(frame_size.height) /
                       image_rect.height);
  return {img_x, img_y};
}

cv::Rect2d select_player_roi(const cv::Mat& frame, int player_id) {
  const std::string title = "Select player " + std::to_string(player_id) + ", then Enter/Space";
  setup_display_window(title, frame.size());
  const cv::Rect2d roi = cv::selectROI(title, frame, false, false);
  cv::destroyWindow(title);
  return roi;
}

bool valid_bbox(const cv::Rect2d& bbox) {
  return bbox.width > 1.0 && bbox.height > 1.0;
}

struct SelectDetectionsUi {
  std::string window;
  cv::Mat frame;
  std::vector<Detection> detections;
  std::vector<bool> selected;
  std::vector<int> selection_order;
  int max_players{};
};

int hit_detection(const std::vector<Detection>& detections, int x, int y) {
  for (int i = static_cast<int>(detections.size()) - 1; i >= 0; --i) {
    const cv::Rect box(detections[i].box);
    if (box.contains(cv::Point(x, y))) {
      return i;
    }
  }
  return -1;
}

void draw_detection_ui(cv::Mat& canvas, const SelectDetectionsUi& ui) {
  canvas = ui.frame.clone();
  const double font_scale = std::clamp(ui.frame.cols / 1280.0, 0.8, 2.0);
  const int box_thickness = std::max(2, static_cast<int>(std::round(ui.frame.cols / 500.0)));
  const int text_thickness = std::max(2, static_cast<int>(std::round(font_scale * 2.0)));

  for (size_t i = 0; i < ui.detections.size(); ++i) {
    const cv::Rect box(ui.detections[i].box);
    const bool is_selected = ui.selected[i];
    const cv::Scalar color = is_selected ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 180, 255);
    cv::rectangle(canvas, box, color, is_selected ? box_thickness + 1 : box_thickness);
    char label[32];
    const int sid = ui.detections[i].stable_id;
    if (sid > 0) {
      std::snprintf(label, sizeof(label), "P%d %.0f%%", sid, ui.detections[i].confidence * 100.0f);
    } else {
      std::snprintf(label, sizeof(label), "D%zu %.0f%%", i + 1, ui.detections[i].confidence * 100.0f);
    }
    const int label_y = std::max(static_cast<int>(30 * font_scale), box.y - 8);
    cv::putText(canvas, label, {box.x, label_y}, cv::FONT_HERSHEY_SIMPLEX, font_scale, color,
                text_thickness, cv::LINE_AA);
  }

  char status[128];
  std::snprintf(status, sizeof(status), "Selected %zu / %d - click to toggle, Enter to start, Esc cancel",
                ui.selection_order.size(), ui.max_players);
  const cv::Size text_size =
      cv::getTextSize(status, cv::FONT_HERSHEY_SIMPLEX, font_scale, text_thickness, nullptr);
  const cv::Rect banner(10, 10, text_size.width + 20, text_size.height + 20);
  cv::rectangle(canvas, banner, {0, 0, 0}, cv::FILLED);
  cv::putText(canvas, status, {20, 20 + text_size.height}, cv::FONT_HERSHEY_SIMPLEX, font_scale,
              {255, 255, 255}, text_thickness, cv::LINE_AA);
}

void on_mouse(int event, int x, int y, int, void* userdata) {
  if (event != cv::EVENT_LBUTTONDOWN || userdata == nullptr) {
    return;
  }
  auto* ui = static_cast<SelectDetectionsUi*>(userdata);
  const cv::Point image_pt = window_to_image(ui->window, ui->frame.size(), x, y);
  const int idx = hit_detection(ui->detections, image_pt.x, image_pt.y);
  if (idx < 0) {
    return;
  }

  if (ui->selected[idx]) {
    ui->selected[idx] = false;
    ui->selection_order.erase(
        std::remove(ui->selection_order.begin(), ui->selection_order.end(), idx),
        ui->selection_order.end());
    return;
  }

  if (static_cast<int>(ui->selection_order.size()) >= ui->max_players) {
    return;
  }

  ui->selected[idx] = true;
  ui->selection_order.push_back(idx);
}

}  // namespace

std::vector<cv::Rect2d> collect_manual_boxes(const cv::Mat& frame, const AppOptions& opts) {
  std::vector<cv::Rect2d> boxes;
  boxes.reserve(opts.players);

  for (int i = 0; i < opts.players; ++i) {
    if (i < static_cast<int>(opts.bboxes.size()) && opts.bboxes[i]) {
      boxes.push_back(*opts.bboxes[i]);
    } else {
      boxes.push_back(select_player_roi(frame, i + 1));
    }
  }

  return boxes;
}

std::optional<std::vector<cv::Rect2d>> select_detections_interactive(
    const cv::Mat& frame, const std::vector<Detection>& detections, int max_players) {
  if (detections.empty()) {
    std::cerr << "No person detections found. Try --mode manual or lower --detect-confidence\n";
    return std::nullopt;
  }

  SelectDetectionsUi ui;
  ui.frame = frame;
  ui.detections = detections;
  ui.selected.assign(detections.size(), false);
  ui.max_players = max_players;

  const std::string window = "Select detections - click to toggle, Enter to start";
  ui.window = window;
  setup_display_window(window, frame.size());
  cv::setMouseCallback(window, on_mouse, &ui);

  while (true) {
    cv::Mat canvas;
    draw_detection_ui(canvas, ui);
    cv::imshow(window, canvas);
    const int key = cv::waitKey(30);
    if (key == 27) {
      cv::destroyWindow(window);
      return std::nullopt;
    }
    if (key == 13 || key == 32) {
      break;
    }
  }

  cv::destroyWindow(window);

  if (ui.selection_order.empty()) {
    std::cerr << "No detections selected\n";
    return std::nullopt;
  }

  std::vector<cv::Rect2d> boxes;
  boxes.reserve(ui.selection_order.size());
  for (const int idx : ui.selection_order) {
    boxes.push_back(ui.detections[idx].box);
  }
  return boxes;
}

std::vector<cv::Scalar> detection_preview_colors() {
  return {
      {0, 220, 0}, {255, 140, 0}, {255, 0, 255}, {0, 220, 255}, {255, 0, 0},
      {0, 128, 255}, {255, 255, 0}, {128, 0, 255},
  };
}

void draw_detections_on_frame(cv::Mat& canvas, const std::vector<Detection>& detections) {
  const double font_scale = std::clamp(canvas.cols / 1280.0, 0.8, 2.0);
  const int box_thickness = std::max(2, static_cast<int>(std::round(canvas.cols / 500.0)));
  const int text_thickness = std::max(2, static_cast<int>(std::round(font_scale * 2.0)));
  const auto colors = detection_preview_colors();

  for (size_t i = 0; i < detections.size(); ++i) {
    const cv::Rect box(detections[i].box);
    const int sid = detections[i].stable_id;
    const cv::Scalar color =
        sid > 0 ? colors[(sid - 1) % colors.size()] : cv::Scalar(0, 180, 255);
    cv::rectangle(canvas, box, color, box_thickness);
    char label[32];
    if (sid > 0) {
      std::snprintf(label, sizeof(label), "P%d %.0f%%", sid, detections[i].confidence * 100.0f);
    } else {
      std::snprintf(label, sizeof(label), "? %.0f%%", detections[i].confidence * 100.0f);
    }
    cv::putText(canvas, label, {box.x, std::max(static_cast<int>(30 * font_scale), box.y - 8)},
                cv::FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv::LINE_AA);
  }
}

void assign_stable_detection_ids(std::vector<DetectionScanFrame>& frames) {
  if (frames.empty()) {
    return;
  }

  auto& frame0 = frames.front().detections;
  std::vector<size_t> order(frame0.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return frame0[a].box.x < frame0[b].box.x;
  });
  for (size_t i = 0; i < order.size(); ++i) {
    frame0[order[i]].stable_id = static_cast<int>(i + 1);
  }

  std::vector<cv::Rect2d> tracked_boxes;
  std::vector<int> tracked_ids;
  for (const Detection& det : frame0) {
    if (det.stable_id > 0) {
      tracked_boxes.push_back(det.box);
      tracked_ids.push_back(det.stable_id);
    }
  }

  for (size_t fi = 1; fi < frames.size(); ++fi) {
    std::vector<cv::Rect2d> next_boxes;
    std::vector<int> next_ids;

    for (Detection& det : frames[fi].detections) {
      int best_id = 0;
      float best_iou = 0.25f;
      for (size_t t = 0; t < tracked_ids.size(); ++t) {
        const float iou = static_cast<float>(detection_iou(det.box, tracked_boxes[t]));
        if (iou > best_iou) {
          best_iou = iou;
          best_id = tracked_ids[t];
        }
      }
      det.stable_id = best_id;
      if (best_id > 0) {
        next_boxes.push_back(det.box);
        next_ids.push_back(best_id);
      }
    }

    tracked_boxes = std::move(next_boxes);
    tracked_ids = std::move(next_ids);
  }
}

std::vector<Detection> build_tracking_init_candidates(const DetectionScanResult& scan,
                                                      float /*frame0_overlap_iou*/) {
  if (scan.frames.empty()) {
    return {};
  }

  // Trackers are initialized on video frame 0 — only use detections from that frame.
  // Multi-frame scan is for preview / confidence; later-frame boxes are different
  // positions of the same moving players and must not become separate trackers.
  const std::vector<Detection>& frame0 = scan.frames.front().detections;
  std::cerr << "Init candidates from frame 0: " << frame0.size()
            << " (multi-frame scan: " << scan.frames_scanned << " frames for preview only)\n";
  return frame0;
}

std::vector<Detection> select_detections_spatial(const std::vector<Detection>& detections,
                                                 int max_players, float max_overlap_iou) {
  std::vector<Detection> sorted = detections;
  std::sort(sorted.begin(), sorted.end(),
            [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

  std::vector<Detection> picked;
  for (const Detection& candidate : sorted) {
    if (static_cast<int>(picked.size()) >= max_players) {
      break;
    }
    bool overlaps = false;
    for (const Detection& existing : picked) {
      if (detection_iou(candidate.box, existing.box) > max_overlap_iou) {
        overlaps = true;
        break;
      }
    }
    if (!overlaps) {
      picked.push_back(candidate);
    }
  }
  return picked;
}

void show_detection_scan_preview(const std::vector<DetectionScanFrame>& frames, int delay_ms) {
  if (frames.empty()) {
    return;
  }

  const std::string window = "Detection scan";
  setup_display_window(window, frames.front().image.size());

  for (const DetectionScanFrame& scan : frames) {
    cv::Mat canvas = scan.image.clone();
    draw_detections_on_frame(canvas, scan.detections);

    char title[96];
    std::snprintf(title, sizeof(title), "Frame %d - %zu detection(s)", scan.frame_index,
                  scan.detections.size());
    const double font_scale = std::clamp(canvas.cols / 1280.0, 0.8, 2.0);
    cv::putText(canvas, title, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, font_scale, {255, 255, 255}, 2,
                cv::LINE_AA);

    cv::imshow(window, canvas);
    cv::waitKey(delay_ms);
  }

  cv::destroyWindow(window);
}

std::optional<std::vector<cv::Rect2d>> select_detections_by_index(
    const std::vector<Detection>& detections, const std::vector<int>& indices) {
  if (indices.empty()) {
    return std::nullopt;
  }

  std::vector<cv::Rect2d> boxes;
  boxes.reserve(indices.size());
  for (const int one_based : indices) {
    if (one_based < 1 || one_based > static_cast<int>(detections.size())) {
      std::cerr << "Invalid detection index: D" << one_based << " (found "
                << detections.size() << " detections)\n";
      return std::nullopt;
    }
    boxes.push_back(detections[one_based - 1].box);
  }
  return boxes;
}

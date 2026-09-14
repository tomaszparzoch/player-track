#pragma once

#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "app_options.hpp"
#include "detection_types.hpp"
#include "person_detector.hpp"

std::vector<cv::Rect2d> collect_manual_boxes(const cv::Mat& frame, const AppOptions& opts);

std::optional<std::vector<cv::Rect2d>> select_detections_interactive(
    const cv::Mat& frame, const std::vector<Detection>& detections, int max_players);

std::optional<std::vector<cv::Rect2d>> select_detections_by_index(
    const std::vector<Detection>& detections, const std::vector<int>& indices);

std::vector<Detection> build_tracking_init_candidates(const DetectionScanResult& scan,
                                                    float frame0_overlap_iou = 0.25f);

std::vector<Detection> select_detections_spatial(const std::vector<Detection>& detections,
                                                 int max_players, float max_overlap_iou = 0.35f);

void assign_stable_detection_ids(std::vector<DetectionScanFrame>& frames);

void show_detection_scan_preview(const std::vector<DetectionScanFrame>& frames, int delay_ms = 1200);

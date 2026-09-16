#include "tracklet_tracker.hpp"

#include <algorithm>

double TrackletTracker::iou(const cv::Rect2d& a, const cv::Rect2d& b) {
  const double x1 = std::max(a.x, b.x);
  const double y1 = std::max(a.y, b.y);
  const double x2 = std::min(a.x + a.width, b.x + b.width);
  const double y2 = std::min(a.y + a.height, b.y + b.height);
  const double inter = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
  const double uni = a.area() + b.area() - inter;
  return uni > 0.0 ? inter / uni : 0.0;
}

std::vector<Tracklet> TrackletTracker::active_tracks() const {
  std::vector<Tracklet> active;
  for (const auto& t : tracks_) {
    if (t.time_since_update <= max_visible_miss_) {
      active.push_back(t);
    }
  }
  return active;
}

std::vector<Tracklet> TrackletTracker::predict() {
  for (auto& track : tracks_) {
    ++track.time_since_update;
  }

  const int erase_after = std::max(30, max_visible_miss_ * 4);
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [erase_after](const Tracklet& t) {
                                 return t.time_since_update > erase_after;
                               }),
                tracks_.end());
  return active_tracks();
}

std::vector<Tracklet> TrackletTracker::update(const std::vector<Detection>& detections) {
  for (auto& track : tracks_) {
    ++track.time_since_update;
  }

  std::vector<bool> det_used(detections.size(), false);
  for (auto& track : tracks_) {
    if (!track.active) continue;

    double best_iou = 0.25;
    int best_idx = -1;
    for (size_t i = 0; i < detections.size(); ++i) {
      if (det_used[i]) continue;
      const double score = iou(track.bbox, detections[i].box);
      if (score > best_iou) {
        best_iou = score;
        best_idx = static_cast<int>(i);
      }
    }
    if (best_idx >= 0) {
      track.bbox = detections[best_idx].box;
      track.time_since_update = 0;
      ++track.age;
      ++track.detect_hits;
      track.max_confidence =
          std::max(track.max_confidence, detections[best_idx].confidence);
      det_used[best_idx] = true;
    }
  }

  for (size_t i = 0; i < detections.size(); ++i) {
    if (det_used[i]) continue;
    Tracklet t;
    t.id = next_id_++;
    t.bbox = detections[i].box;
    t.age = 1;
    t.detect_hits = 1;
    t.max_confidence = detections[i].confidence;
    tracks_.push_back(t);
  }

  const int erase_after = std::max(30, max_visible_miss_ * 4);
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [erase_after](const Tracklet& t) {
                                 return t.time_since_update > erase_after;
                               }),
                tracks_.end());

  return active_tracks();
}

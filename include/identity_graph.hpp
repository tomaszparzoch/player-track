#pragma once

#include <vector>

#include "global_track.hpp"
#include "reid_extractor.hpp"

struct TrackletNode {
  int camera_index{};
  int local_track_id{};
  double t_start{};
  double t_end{};
  TeamId team{TeamId::Unknown};
  int jersey_number{-1};
  float jersey_confidence{0.0f};
  ReIdFeatures features;
  std::vector<TrackObservation> observations;
};

class IdentityGraph {
 public:
  void add_tracklet(TrackletNode node);
  void add_ocr_anchor(int camera_index, int local_track_id, int number, float confidence);
  std::vector<ResolvedPlayer> resolve() const;

 private:
  std::vector<TrackletNode> nodes_;

  float link_score(const TrackletNode& a, const TrackletNode& b) const;
};

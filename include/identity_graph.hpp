#pragma once

#include <map>
#include <utility>
#include <vector>

#include "global_track.hpp"
#include "reid_extractor.hpp"

struct TrackletGlobalIdentity {
  int player_number{-1};
  int jersey_number{-1};
  TeamId team{TeamId::Unknown};
  float identity_confidence{0.0f};
};

struct IdentityResolution {
  std::vector<ResolvedPlayer> players;
  std::map<std::pair<int, int>, TrackletGlobalIdentity> tracklet_identity;
};

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
  cv::Mat snapshot;
  cv::Mat last_snapshot;
};

class IdentityGraph {
 public:
  void set_hall_mode(bool enabled) { hall_mode_ = enabled; }
  void add_tracklet(TrackletNode node);
  void add_ocr_anchor(int camera_index, int local_track_id, int number, float confidence);
  std::vector<ResolvedPlayer> resolve() const;
  IdentityResolution resolve_identities() const;

 private:
  std::vector<TrackletNode> nodes_;
  bool hall_mode_{false};

  float link_score(const TrackletNode& a, const TrackletNode& b) const;
};

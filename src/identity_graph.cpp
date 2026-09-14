#include "identity_graph.hpp"

#include <algorithm>
#include <map>
#include <set>

void IdentityGraph::add_tracklet(TrackletNode node) {
  nodes_.push_back(std::move(node));
}

void IdentityGraph::add_ocr_anchor(int camera_index, int local_track_id, int number,
                                   float confidence) {
  for (auto& node : nodes_) {
    if (node.camera_index == camera_index && node.local_track_id == local_track_id) {
      if (confidence > node.jersey_confidence) {
        node.jersey_number = number;
        node.jersey_confidence = confidence;
      }
    }
  }
}

float IdentityGraph::link_score(const TrackletNode& a, const TrackletNode& b) const {
  if (a.camera_index == b.camera_index && a.local_track_id == b.local_track_id) return 1.0f;

  ReIdExtractor reid;
  float score = reid.similarity(a.features, b.features);
  if (a.team != TeamId::Unknown && b.team != TeamId::Unknown && a.team != b.team) {
    score *= 0.1f;
  }
  if (a.jersey_number > 0 && b.jersey_number > 0) {
    if (a.jersey_number == b.jersey_number) {
      score += 0.5f;
    } else {
      score *= 0.05f;
    }
  }
  const double dt = std::abs((a.t_start + a.t_end) * 0.5 - (b.t_start + b.t_end) * 0.5);
  if (dt > 5.0) score *= 0.5f;
  return score;
}

std::vector<ResolvedPlayer> IdentityGraph::resolve() const {
  std::vector<ResolvedPlayer> players;
  if (nodes_.empty()) return players;

  std::vector<int> parent(nodes_.size());
  for (size_t i = 0; i < parent.size(); ++i) parent[i] = static_cast<int>(i);

  auto find = [&](int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };

  auto unite = [&](int a, int b) {
    a = find(a);
    b = find(b);
    if (a != b) parent[b] = a;
  };

  for (size_t i = 0; i < nodes_.size(); ++i) {
    for (size_t j = i + 1; j < nodes_.size(); ++j) {
      if (link_score(nodes_[i], nodes_[j]) >= 0.65f) {
        unite(static_cast<int>(i), static_cast<int>(j));
      }
    }
  }

  std::map<int, std::vector<int>> groups;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    groups[find(static_cast<int>(i))].push_back(static_cast<int>(i));
  }

  int next_fallback_id = 1;
  for (const auto& [root, indices] : groups) {
    (void)root;
    ResolvedPlayer player;
    std::map<int, float> number_votes;
    std::map<TeamId, int> team_votes;

    for (int idx : indices) {
      const auto& node = nodes_[idx];
      if (node.jersey_number > 0) {
        number_votes[node.jersey_number] += node.jersey_confidence;
      }
      if (node.team != TeamId::Unknown) {
        ++team_votes[node.team];
      }
      for (const auto& obs : node.observations) {
        if (obs.identity_confidence < 0.3f) continue;
        GlobalTrackPoint pt;
        pt.t_s = obs.t_s;
        pt.camera_index = obs.camera_index;
        pt.team = obs.team;
        if (obs.has_world) {
          pt.x_m = obs.world_m.x;
          pt.y_m = obs.world_m.y;
          pt.speed_m_s = obs.speed_m_s;
          pt.confidence = obs.identity_confidence * obs.calib_confidence;
          pt.pixel_coords = false;
        } else {
          pt.x_m = obs.image_pt.x;
          pt.y_m = obs.image_pt.y;
          pt.speed_m_s = obs.speed_px_s;
          pt.confidence = obs.identity_confidence;
          pt.pixel_coords = true;
        }
        player.trajectory.push_back(pt);
      }
    }

    int best_number = -1;
    float best_vote = 0.0f;
    for (const auto& [num, vote] : number_votes) {
      if (vote > best_vote) {
        best_vote = vote;
        best_number = num;
      }
    }
    player.player_number = best_number > 0 ? best_number : next_fallback_id++;

    TeamId best_team = TeamId::Unknown;
    int best_team_count = 0;
    for (const auto& [team, count] : team_votes) {
      if (count > best_team_count) {
        best_team_count = count;
        best_team = team;
      }
    }
    player.team = best_team;
    player.identity_confidence = best_vote > 0 ? std::min(1.0f, best_vote) : 0.4f;

    for (auto& pt : player.trajectory) {
      pt.player_number = player.player_number;
      pt.team = player.team;
    }

    std::sort(player.trajectory.begin(), player.trajectory.end(),
              [](const GlobalTrackPoint& a, const GlobalTrackPoint& b) { return a.t_s < b.t_s; });

    if (!player.trajectory.empty()) {
      players.push_back(std::move(player));
    }
  }

  return players;
}

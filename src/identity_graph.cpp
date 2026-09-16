#include "identity_graph.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <set>

namespace {

constexpr double kSpatialTimeWindowS = 0.8;
constexpr float kSpatialWeight = 0.65f;

bool temporally_overlap(const TrackletNode& a, const TrackletNode& b) {
  return a.t_start <= b.t_end && b.t_start <= a.t_end;
}

double spatial_max_dist_m(bool hall_mode) { return hall_mode ? 2.5 : 4.0; }

float spatial_link_score(const TrackletNode& a, const TrackletNode& b, bool hall_mode) {
  if (a.camera_index == b.camera_index) return 0.0f;

  const double max_dist = spatial_max_dist_m(hall_mode);
  float best = 0.0f;
  for (const auto& oa : a.observations) {
    if (!oa.has_world || oa.calib_confidence < 0.35f) continue;
    for (const auto& ob : b.observations) {
      if (!ob.has_world || ob.calib_confidence < 0.35f) continue;
      const double dt = std::abs(oa.t_s - ob.t_s);
      if (dt > kSpatialTimeWindowS) continue;

      const double dist_m =
          std::hypot(oa.world_m.x - ob.world_m.x, oa.world_m.y - ob.world_m.y);
      if (dist_m > max_dist) continue;

      const float time_factor =
          static_cast<float>(1.0 - dt / kSpatialTimeWindowS);
      const float dist_factor = static_cast<float>(1.0 - dist_m / max_dist);
      const float calib_factor = std::min(oa.calib_confidence, ob.calib_confidence);
      best = std::max(best, time_factor * dist_factor * calib_factor);
    }
  }
  return best;
}

IdentityResolution build_identity_resolution(
    const std::vector<TrackletNode>& nodes, bool hall_mode,
    const std::function<float(const TrackletNode&, const TrackletNode&)>& link_score_fn) {
  IdentityResolution result;
  if (nodes.empty()) return result;

  std::vector<int> parent(nodes.size());
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

  auto groups_conflict = [&](int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b) return false;
    bool a_has_marker = false;
    bool a_no_marker = false;
    bool b_has_marker = false;
    bool b_no_marker = false;
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (find(static_cast<int>(i)) != a) continue;
      if (hall_mode) {
        if (nodes[i].team == TeamId::TeamA) a_has_marker = true;
        if (nodes[i].team == TeamId::TeamB) a_no_marker = true;
      }
      for (size_t j = 0; j < nodes.size(); ++j) {
        if (find(static_cast<int>(j)) != b) continue;
        if (hall_mode) {
          if (nodes[j].team == TeamId::TeamA) b_has_marker = true;
          if (nodes[j].team == TeamId::TeamB) b_no_marker = true;
        }
        if (nodes[i].camera_index == nodes[j].camera_index &&
            temporally_overlap(nodes[i], nodes[j])) {
          return true;
        }
      }
    }
    if (hall_mode && ((a_has_marker && b_no_marker) || (a_no_marker && b_has_marker))) {
      return true;
    }
    return false;
  };

  struct Edge {
    int i{};
    int j{};
    float score{};
  };
  std::vector<Edge> edges;
  for (size_t i = 0; i < nodes.size(); ++i) {
    for (size_t j = i + 1; j < nodes.size(); ++j) {
      const float score = link_score_fn(nodes[i], nodes[j]);
      if (score >= 0.72f) {
        edges.push_back({static_cast<int>(i), static_cast<int>(j), score});
      }
    }
  }
  std::sort(edges.begin(), edges.end(),
            [](const Edge& a, const Edge& b) { return a.score > b.score; });
  for (const auto& edge : edges) {
    if (find(edge.i) == find(edge.j)) continue;
    if (groups_conflict(edge.i, edge.j)) continue;
    unite(edge.i, edge.j);
  }

  auto observation_at_time = [](const TrackletNode& node, double t_s) -> const TrackObservation* {
    if (node.observations.empty()) return nullptr;
    const TrackObservation* best = &node.observations.front();
    double best_dt = std::abs(best->t_s - t_s);
    for (const auto& obs : node.observations) {
      const double dt = std::abs(obs.t_s - t_s);
      if (dt < best_dt) {
        best_dt = dt;
        best = &obs;
      }
    }
    return best;
  };

  auto crop_for_time = [](const TrackletNode& node, double t_s) {
    if (node.last_snapshot.empty() ||
        std::abs(t_s - node.t_start) <= std::abs(t_s - node.t_end)) {
      return node.snapshot;
    }
    return node.last_snapshot;
  };

  auto add_event = [](ResolvedPlayer& player, PlayerEventKind kind, const TrackletNode& node,
                      const TrackObservation* obs, const cv::Mat& image) {
    PlayerEventPhoto photo;
    photo.kind = kind;
    photo.camera_index = node.camera_index;
    photo.image = image.empty() ? cv::Mat() : image.clone();
    if (obs) {
      photo.t_s = obs->t_s;
      photo.bbox = obs->bbox;
      photo.world_m = obs->has_world ? obs->world_m : obs->image_pt;
      photo.has_world = obs->has_world;
    } else {
      photo.t_s = kind == PlayerEventKind::Lost ? node.t_end : node.t_start;
    }
    player.event_photos.push_back(std::move(photo));
  };

  auto best_join_pair = [&](const TrackletNode& a, const TrackletNode& b) {
    struct Hit {
      float score{0.0f};
      const TrackObservation* oa{nullptr};
      const TrackObservation* ob{nullptr};
    } hit;
    const double max_dist = spatial_max_dist_m(hall_mode);
    for (const auto& oa : a.observations) {
      for (const auto& ob : b.observations) {
        const double dt = std::abs(oa.t_s - ob.t_s);
        float score = 0.0f;
        if (oa.has_world && ob.has_world && oa.calib_confidence >= 0.35f &&
            ob.calib_confidence >= 0.35f && dt <= kSpatialTimeWindowS) {
          const double dist_m =
              std::hypot(oa.world_m.x - ob.world_m.x, oa.world_m.y - ob.world_m.y);
          if (dist_m <= max_dist) {
            score = static_cast<float>((1.0 - dt / kSpatialTimeWindowS) * (1.0 - dist_m / max_dist) *
                                       std::min(oa.calib_confidence, ob.calib_confidence));
          }
        }
        if (score <= 0.0f && dt < 1.5) {
          score = static_cast<float>(0.05 * (1.0 - dt / 1.5));
        }
        if (score > hit.score) {
          hit.score = score;
          hit.oa = &oa;
          hit.ob = &ob;
        }
      }
    }
    return hit;
  };

  std::map<int, std::vector<int>> groups;
  for (size_t i = 0; i < nodes.size(); ++i) {
    groups[find(static_cast<int>(i))].push_back(static_cast<int>(i));
  }

  int next_fallback_id = 1;
  for (const auto& [root, indices] : groups) {
    (void)root;
    ResolvedPlayer player;
    std::map<int, float> number_votes;
    std::map<TeamId, int> team_votes;

    for (int idx : indices) {
      const auto& node = nodes[idx];
      if (!hall_mode && node.jersey_number > 0) {
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

    int snapshot_idx = -1;
    double snapshot_t = 1e300;
    for (int idx : indices) {
      if (nodes[idx].snapshot.empty()) continue;
      if (nodes[idx].t_start < snapshot_t) {
        snapshot_t = nodes[idx].t_start;
        snapshot_idx = idx;
      }
    }
    if (snapshot_idx >= 0) {
      player.snapshot = nodes[snapshot_idx].snapshot.clone();
    }

    std::map<int, std::vector<int>> by_camera;
    for (int idx : indices) {
      by_camera[nodes[idx].camera_index].push_back(idx);
    }
    std::vector<int> cam_ids;
    for (const auto& [cam, idxs] : by_camera) {
      (void)idxs;
      cam_ids.push_back(cam);
    }
    std::sort(cam_ids.begin(), cam_ids.end());
    const int shown_cams = std::min(2, static_cast<int>(cam_ids.size()));
    for (int c = 0; c < shown_cams; ++c) {
      const auto& cam_nodes = by_camera[cam_ids[c]];
      int detect_idx = cam_nodes.front();
      int lost_idx = cam_nodes.front();
      for (int idx : cam_nodes) {
        if (nodes[idx].t_start < nodes[detect_idx].t_start) detect_idx = idx;
        if (nodes[idx].t_end > nodes[lost_idx].t_end) lost_idx = idx;
      }
      const auto& detect_node = nodes[detect_idx];
      const auto& lost_node = nodes[lost_idx];
      const TrackObservation* detect_obs =
          detect_node.observations.empty() ? nullptr : &detect_node.observations.front();
      const TrackObservation* lost_obs =
          lost_node.observations.empty() ? nullptr : &lost_node.observations.back();
      add_event(player, PlayerEventKind::Detect, detect_node, detect_obs, detect_node.snapshot);
      add_event(player, PlayerEventKind::Lost, lost_node, lost_obs,
                lost_node.last_snapshot.empty() ? lost_node.snapshot : lost_node.last_snapshot);
    }

    if (shown_cams >= 2) {
      int best_a = -1;
      int best_b = -1;
      float best_link = -1.0f;
      for (int idx_a : by_camera[cam_ids[0]]) {
        for (int idx_b : by_camera[cam_ids[1]]) {
          const float score = link_score_fn(nodes[idx_a], nodes[idx_b]);
          if (score > best_link) {
            best_link = score;
            best_a = idx_a;
            best_b = idx_b;
          }
        }
      }
      if (best_a >= 0 && best_b >= 0) {
        const auto hit = best_join_pair(nodes[best_a], nodes[best_b]);
        const TrackObservation* oa =
            hit.oa ? hit.oa : observation_at_time(nodes[best_a], nodes[best_b].t_start);
        const TrackObservation* ob =
            hit.ob ? hit.ob : observation_at_time(nodes[best_b], nodes[best_a].t_start);
        const double ta = oa ? oa->t_s : nodes[best_a].t_start;
        const double tb = ob ? ob->t_s : nodes[best_b].t_start;
        add_event(player, PlayerEventKind::Join, nodes[best_a], oa, crop_for_time(nodes[best_a], ta));
        add_event(player, PlayerEventKind::Join, nodes[best_b], ob, crop_for_time(nodes[best_b], tb));
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
    player.player_number = next_fallback_id++;
    player.jersey_number = hall_mode ? -1 : best_number;

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

    if (player.trajectory.empty()) {
      continue;
    }

    TrackletGlobalIdentity identity;
    identity.player_number = player.player_number;
    identity.jersey_number = player.jersey_number;
    identity.team = player.team;
    identity.identity_confidence = player.identity_confidence;
    for (int idx : indices) {
      const auto& node = nodes[idx];
      result.tracklet_identity[{node.camera_index, node.local_track_id}] = identity;
    }

    result.players.push_back(std::move(player));
  }

  return result;
}

}  // namespace

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

  const bool same_camera = a.camera_index == b.camera_index;
  if (same_camera && temporally_overlap(a, b)) return 0.0f;

  if (hall_mode_ && a.team != TeamId::Unknown && b.team != TeamId::Unknown && a.team != b.team) {
    return 0.0f;
  }
  ReIdExtractor reid;
  reid.set_hall_mode(hall_mode_);
  float score = reid.similarity(a.features, b.features);
  if (a.team != TeamId::Unknown && b.team != TeamId::Unknown && a.team != b.team) {
    score *= 0.1f;
  }

  const float spatial = spatial_link_score(a, b, hall_mode_);
  const bool jersey_match =
      !hall_mode_ && a.jersey_number > 0 && b.jersey_number > 0 && a.jersey_number == b.jersey_number;
  const bool jersey_mismatch =
      !hall_mode_ && a.jersey_number > 0 && b.jersey_number > 0 && a.jersey_number != b.jersey_number;

  if (jersey_match && (same_camera || spatial > 0.2f)) {
    score += 0.5f;
  } else if (jersey_mismatch) {
    score *= 0.05f;
  }

  if (spatial > 0.0f) {
    score += spatial * kSpatialWeight;
    if (spatial >= 0.55f && a.team != TeamId::Unknown && a.team == b.team) {
      score += 0.15f;
    }
  }

  const double dt = std::abs((a.t_start + a.t_end) * 0.5 - (b.t_start + b.t_end) * 0.5);
  if (dt > 5.0) score *= 0.5f;

  if (same_camera) {
    const double gap = std::max(0.0, std::max(a.t_start, b.t_start) - std::min(a.t_end, b.t_end));
    if (!jersey_match && (score < 0.88f || gap > 3.0)) {
      return 0.0f;
    }
  } else if (spatial < 0.25f && !jersey_match) {
    return 0.0f;
  }
  return score;
}

std::vector<ResolvedPlayer> IdentityGraph::resolve() const {
  return resolve_identities().players;
}

IdentityResolution IdentityGraph::resolve_identities() const {
  return build_identity_resolution(nodes_, hall_mode_,
                                   [this](const TrackletNode& a, const TrackletNode& b) {
                                     return link_score(a, b);
                                   });
}

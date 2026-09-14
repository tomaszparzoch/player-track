#include "global_track.hpp"

std::string team_to_string(TeamId team) {
  switch (team) {
    case TeamId::TeamA: return "TeamA";
    case TeamId::TeamB: return "TeamB";
    default: return "Unknown";
  }
}

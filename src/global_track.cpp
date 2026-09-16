#include "global_track.hpp"

std::string team_to_string(TeamId team) {
  switch (team) {
    case TeamId::TeamA: return "TeamA";
    case TeamId::TeamB: return "TeamB";
    default: return "Unknown";
  }
}

std::string team_display_name(TeamId team, bool hall_mode) {
  if (!hall_mode) return team_to_string(team);
  switch (team) {
    case TeamId::TeamA: return "Zielone";
    case TeamId::TeamB: return "Bez znacznikow";
    default: return "Unknown";
  }
}

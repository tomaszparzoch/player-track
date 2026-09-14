#pragma once

#include <cmath>
#include <cstdio>
#include <string>

inline std::string format_duration(double seconds) {
  if (!std::isfinite(seconds) || seconds < 0.0) {
    return "--:--";
  }
  const int total = static_cast<int>(seconds + 0.5);
  const int h = total / 3600;
  const int m = (total % 3600) / 60;
  const int s = total % 60;
  char buf[32];
  if (h > 0) {
    std::snprintf(buf, sizeof(buf), "%d:%02d:%02d", h, m, s);
  } else {
    std::snprintf(buf, sizeof(buf), "%d:%02d", m, s);
  }
  return buf;
}

inline double estimate_remaining_seconds(double progress, double elapsed_seconds) {
  if (!std::isfinite(progress) || !std::isfinite(elapsed_seconds)) {
    return -1.0;
  }
  if (progress <= 0.005 || elapsed_seconds <= 0.0) {
    return -1.0;
  }
  if (progress >= 0.999) {
    return 0.0;
  }
  return elapsed_seconds * (1.0 - progress) / progress;
}

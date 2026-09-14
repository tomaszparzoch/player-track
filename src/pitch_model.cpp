#include "pitch_model.hpp"

PitchModel default_pitch_model(double length_m, double width_m) {
  return PitchModel{length_m, width_m};
}

std::vector<PitchLandmark> PitchModel::landmarks() const {
  const double hw = width_m * 0.5;
  const double hl = length_m * 0.5;
  const double pa_w = 40.32;
  const double pa_h = 16.5;
  const double pa_x = (length_m - pa_w) * 0.5;
  const double gc_r = 9.15;

  return {
      {"corner_tl", "Corner top-left", {0.0, 0.0}},
      {"corner_tr", "Corner top-right", {length_m, 0.0}},
      {"corner_bl", "Corner bottom-left", {0.0, width_m}},
      {"corner_br", "Corner bottom-right", {length_m, width_m}},
      {"center", "Center spot", {hl, hw}},
      {"pa_tl", "Penalty area TL", {pa_x, (width_m - pa_h) * 0.5}},
      {"pa_tr", "Penalty area TR", {pa_x + pa_w, (width_m - pa_h) * 0.5}},
      {"pa_bl", "Penalty area BL", {pa_x, (width_m + pa_h) * 0.5}},
      {"pa_br", "Penalty area BR", {pa_x + pa_w, (width_m + pa_h) * 0.5}},
      {"pa_tl_far", "Far penalty TL", {length_m - pa_x - pa_w, (width_m - pa_h) * 0.5}},
      {"pa_tr_far", "Far penalty TR", {length_m - pa_x, (width_m - pa_h) * 0.5}},
      {"pa_bl_far", "Far penalty BL", {length_m - pa_x - pa_w, (width_m + pa_h) * 0.5}},
      {"pa_br_far", "Far penalty BR", {length_m - pa_x, (width_m + pa_h) * 0.5}},
      {"gc_top", "Center circle top", {hl, hw - gc_r}},
      {"gc_bottom", "Center circle bottom", {hl, hw + gc_r}},
      {"halfway_top", "Halfway line top", {hl, 0.0}},
      {"halfway_bottom", "Halfway line bottom", {hl, width_m}},
  };
}

std::optional<PitchLandmark> PitchModel::find_landmark(const std::string& id) const {
  for (const auto& lm : landmarks()) {
    if (lm.id == id) return lm;
  }
  return std::nullopt;
}

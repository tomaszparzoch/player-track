#include "pitch_model.hpp"

#include <cmath>

namespace {

constexpr double kGoalWidthM = 7.32;

bool is_corner_role(CameraRole role) {
  return role == CameraRole::CornerTL || role == CameraRole::CornerTR ||
         role == CameraRole::CornerBL || role == CameraRole::CornerBR;
}

bool camera_on_upper_touchline(CameraRole role) {
  return role == CameraRole::CornerTL || role == CameraRole::CornerTR;
}

struct GoalPostCopy {
  std::string same_side_label;
  std::string same_side_instruction;
  std::string opposite_side_label;
  std::string opposite_side_instruction;
};

GoalPostCopy goal_post_copy(CameraRole role) {
  GoalPostCopy copy;
  copy.same_side_label = "Słupek dalszej bramki (linia boczna po stronie kamery)";
  copy.opposite_side_label = "Słupek dalszej bramki (przeciwna linia boczna)";
  if (camera_on_upper_touchline(role)) {
    copy.same_side_instruction =
        "Kliknij słupek dalszej bramki na tej samej długiej linii bocznej co narożnik przy "
        "kamerze (u Ciebie to linia boczna „u góry” boiska). To NIE jest lewo/prawo na "
        "ekranie — chodzi o linie boczne boiska.";
    copy.opposite_side_instruction =
        "Kliknij słupek dalszej bramki na przeciwnej długiej linii bocznej (po drugiej stronie "
        "szerokości boiska). Najlepiej wierzchołek słupka lub punkt na linii bramkowej.";
  } else {
    copy.same_side_instruction =
        "Kliknij słupek dalszej bramki na tej samej długiej linii bocznej co narożnik przy "
        "kamerze (u Ciebie to linia boczna „u dołu” boiska). To NIE jest lewo/prawo na "
        "ekranie — chodzi o linie boczne boiska.";
    copy.opposite_side_instruction =
        "Kliknij słupek dalszej bramki na przeciwnej długiej linii bocznej (po drugiej stronie "
        "szerokości boiska). Najlepiej wierzchołek słupka lub punkt na linii bramkowej.";
  }
  return copy;
}

}  // namespace

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
      {"corner_tl", "Corner top-left", "Narożnik boiska (góra-lewo).", {0.0, 0.0}},
      {"corner_tr", "Corner top-right", "Narożnik boiska (góra-prawo).", {length_m, 0.0}},
      {"corner_bl", "Corner bottom-left", "Narożnik boiska (dół-lewo).", {0.0, width_m}},
      {"corner_br", "Corner bottom-right", "Narożnik boiska (dół-prawo).", {length_m, width_m}},
      {"center", "Center spot", "Środek boiska (plama, punkt koła środkowego).", {hl, hw}},
      {"pa_tl", "Penalty area TL", "Róg pola karnego (góra-lewo).",
       {pa_x, (width_m - pa_h) * 0.5}},
      {"pa_tr", "Penalty area TR", "Róg pola karnego (góra-prawo).",
       {pa_x + pa_w, (width_m - pa_h) * 0.5}},
      {"pa_bl", "Penalty area BL", "Róg pola karnego (dół-lewo).",
       {pa_x, (width_m + pa_h) * 0.5}},
      {"pa_br", "Penalty area BR", "Róg pola karnego (dół-prawo).",
       {pa_x + pa_w, (width_m + pa_h) * 0.5}},
      {"pa_tl_far", "Far penalty TL", "Róg dalszego pola karnego (góra-lewo).",
       {length_m - pa_x - pa_w, (width_m - pa_h) * 0.5}},
      {"pa_tr_far", "Far penalty TR", "Róg dalszego pola karnego (góra-prawo).",
       {length_m - pa_x, (width_m - pa_h) * 0.5}},
      {"pa_bl_far", "Far penalty BL", "Róg dalszego pola karnego (dół-lewo).",
       {length_m - pa_x - pa_w, (width_m + pa_h) * 0.5}},
      {"pa_br_far", "Far penalty BR", "Róg dalszego pola karnego (dół-prawo).",
       {length_m - pa_x, (width_m + pa_h) * 0.5}},
      {"gc_top", "Center circle top", "Górna krawędź koła środkowego na linii środkowej.",
       {hl, hw - gc_r}},
      {"gc_bottom", "Center circle bottom", "Dolna krawędź koła środkowego na linii środkowej.",
       {hl, hw + gc_r}},
      {"halfway_top", "Halfway line top", "Przecięcie linii środkowej z górną linią boczną.",
       {hl, 0.0}},
      {"halfway_bottom", "Halfway line bottom",
       "Przecięcie linii środkowej z dolną linią boczną.", {hl, width_m}},
  };
}

std::vector<PitchLandmark> PitchModel::corner_calibration_landmarks(CameraRole role) const {
  if (!is_corner_role(role)) {
    return landmarks();
  }

  const double hw = width_m * 0.5;
  const double hl = length_m * 0.5;
  const double post_y0 = (width_m - kGoalWidthM) * 0.5;
  const double post_y1 = (width_m + kGoalWidthM) * 0.5;

  cv::Point2d near_corner;
  double far_goal_x = length_m;
  cv::Point2d halfway_near;
  cv::Point2d halfway_far;
  std::string corner_name;

  switch (role) {
    case CameraRole::CornerTL:
      near_corner = {0.0, 0.0};
      far_goal_x = length_m;
      halfway_near = {hl, 0.0};
      halfway_far = {hl, width_m};
      corner_name = "górny-lewy";
      break;
    case CameraRole::CornerTR:
      near_corner = {length_m, 0.0};
      far_goal_x = 0.0;
      halfway_near = {hl, 0.0};
      halfway_far = {hl, width_m};
      corner_name = "górny-prawy";
      break;
    case CameraRole::CornerBL:
      near_corner = {0.0, width_m};
      far_goal_x = length_m;
      halfway_near = {hl, width_m};
      halfway_far = {hl, 0.0};
      corner_name = "dolny-lewy";
      break;
    case CameraRole::CornerBR:
      near_corner = {length_m, width_m};
      far_goal_x = 0.0;
      halfway_near = {hl, width_m};
      halfway_far = {hl, 0.0};
      corner_name = "dolny-prawy";
      break;
    default:
      return landmarks();
  }

  const cv::Point2d post_upper_touchline{far_goal_x, post_y0};
  const cv::Point2d post_lower_touchline{far_goal_x, post_y1};
  const GoalPostCopy posts = goal_post_copy(role);
  const cv::Point2d post_same_side =
      camera_on_upper_touchline(role) ? post_upper_touchline : post_lower_touchline;
  const cv::Point2d post_opposite_side =
      camera_on_upper_touchline(role) ? post_lower_touchline : post_upper_touchline;

  return {
      {"corner_near",
       "Narożnik przy kamerze",
       "Kliknij narożnik boiska po stronie kamery (" + corner_name +
           "). To skrzyżowanie linii bocznej i bramkowej najbliżej statywu.",
       near_corner},
      {"goal_far_post_same_side",
       posts.same_side_label,
       posts.same_side_instruction,
       post_same_side},
      {"goal_far_post_opposite_side",
       posts.opposite_side_label,
       posts.opposite_side_instruction,
       post_opposite_side},
      {"center",
       "Środek boiska",
       "Kliknij środek boiska — plamę w centrum, punkt koła środkowego (przecięcie linii "
       "środkowej z osią boiska).",
       {hl, hw}},
      {"halfway_near_touchline",
       "Linia środkowa (bliższa linia boczna)",
       "Kliknij punkt, w którym linia środkowa boiska spotyka się z linią boczną bliżej kamery "
       "(widoczne T na boisku).",
       halfway_near},
      {"halfway_far_touchline",
       "Linia środkowa (dalsza linia boczna)",
       "Kliknij punkt, w którym linia środkowa spotyka się z przeciwną linią boczną boiska.",
       halfway_far},
  };
}

std::optional<PitchLandmark> PitchModel::find_landmark(const std::string& id) const {
  for (const auto& lm : landmarks()) {
    if (lm.id == id) return lm;
  }
  for (const auto& lm : corner_calibration_landmarks(CameraRole::CornerTL)) {
    if (lm.id == id) return lm;
  }
  return std::nullopt;
}

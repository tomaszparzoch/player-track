#include "pdf_report.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include <hpdf.h>

namespace {

HPDF_Font pdf_body_font(HPDF_Doc pdf) {
  return HPDF_GetFont(pdf, "Helvetica", nullptr);
}

HPDF_Font pdf_title_font(HPDF_Doc pdf) {
  HPDF_Font font = HPDF_GetFont(pdf, "Helvetica-Bold", nullptr);
  if (!font) {
    font = HPDF_GetFont(pdf, "Helvetica", "Bold");
  }
  return font;
}

void pdf_text_at(HPDF_Doc pdf, HPDF_Page page, float x, float y, float size,
                 const std::string& text) {
  HPDF_Page_BeginText(page);
  HPDF_Page_SetFontAndSize(page, pdf_body_font(pdf), size);
  HPDF_Page_MoveTextPos(page, x, y);
  HPDF_Page_ShowText(page, text.c_str());
  HPDF_Page_EndText(page);
}

void pdf_title_at(HPDF_Doc pdf, HPDF_Page page, float x, float y, float size,
                  const std::string& text) {
  HPDF_Page_BeginText(page);
  HPDF_Page_SetFontAndSize(page, pdf_title_font(pdf), size);
  HPDF_Page_MoveTextPos(page, x, y);
  HPDF_Page_ShowText(page, text.c_str());
  HPDF_Page_EndText(page);
}

void ensure_space(HPDF_Doc pdf, HPDF_Page* page, float* y, float needed) {
  if (*y - needed >= 36.0f) return;
  *page = HPDF_AddPage(pdf);
  HPDF_Page_SetSize(*page, HPDF_PAGE_SIZE_A4, HPDF_PAGE_PORTRAIT);
  *y = 800.0f;
}

void draw_image_fit(HPDF_Page page, HPDF_Image img, float x, float y, float max_w, float max_h) {
  if (!img) return;
  const float iw = static_cast<float>(HPDF_Image_GetWidth(img));
  const float ih = static_cast<float>(HPDF_Image_GetHeight(img));
  if (iw <= 0.0f || ih <= 0.0f) return;
  const float scale = std::min(max_w / iw, max_h / ih);
  const float w = iw * scale;
  const float h = ih * scale;
  HPDF_Page_DrawImage(page, img, x, y, w, h);
}

void draw_jpeg_thumb(HPDF_Doc pdf, HPDF_Page page, const std::filesystem::path& path, float x,
                     float y, float max_w, float max_h, const std::string& caption) {
  pdf_text_at(pdf, page, x, y + max_h + 8, 6, caption);
  if (path.empty() || !std::filesystem::exists(path)) return;
  HPDF_Image img = HPDF_LoadJpegImageFromFile(pdf, path.string().c_str());
  if (img) {
    draw_image_fit(page, img, x, y, max_w, max_h);
  }
}

void draw_png_fit(HPDF_Doc pdf, HPDF_Page page, const std::filesystem::path& path, float x,
                  float y, float max_w, float max_h) {
  if (path.empty() || !std::filesystem::exists(path)) return;
  HPDF_Image img = HPDF_LoadPngImageFromFile(pdf, path.string().c_str());
  if (img) {
    draw_image_fit(page, img, x, y, max_w, max_h);
  }
}

std::string fmt1(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.1f", v);
  return buf;
}

}  // namespace

bool write_pdf_report(const std::filesystem::path& path, const ProjectConfig& config,
                      const std::vector<PlayerStatsReport>& reports,
                      const AnalysisQuality& quality) {
  HPDF_Doc pdf = HPDF_New(nullptr, nullptr);
  if (!pdf) return false;
  HPDF_SetCompressionMode(pdf, HPDF_COMP_ALL);

  HPDF_Page page = HPDF_AddPage(pdf);
  HPDF_Page_SetSize(page, HPDF_PAGE_SIZE_A4, HPDF_PAGE_PORTRAIT);
  float y = 812;

  pdf_title_at(pdf, page, 40, y, 14, "Player Track Match Report");
  y -= 16;
  pdf_text_at(pdf, page, 40, y, 8,
              "Mode: " + project_mode_to_string(quality.project_mode) + "   Pitch: " +
                  std::to_string(static_cast<int>(config.pitch_length_m)) + "x" +
                  std::to_string(static_cast<int>(config.pitch_width_m)) + " m   Cameras: " +
                  std::to_string(config.cameras.size()) +
                  (config.hall_mode ? "   Hall: bibs" : ""));
  y -= 12;
  pdf_text_at(pdf, page, 40, y, 8,
              "Identity: " + fmt1(quality.identity_confidence_pct) +
                  "%   Calibration: " + fmt1(quality.calib_confidence_pct) + "%");
  y -= 12;
  if (!quality.disclaimer.empty()) {
    pdf_text_at(pdf, page, 40, y, 7, quality.disclaimer);
    y -= 12;
  }

  const bool pixel_mode = !reports.empty() && reports.front().pixel_coords;
  pdf_title_at(pdf, page, 40, y, 11, "Summary");
  y -= 13;
  if (pixel_mode) {
    pdf_text_at(pdf, page, 40, y, 7, "#  Team            Dist(px)  Spr  Avg   Max");
  } else {
    pdf_text_at(pdf, page, 40, y, 7, "#  Team            Dist(m)   Spr  Avg   Max");
  }
  y -= 11;

  for (const auto& r : reports) {
    ensure_space(pdf, &page, &y, 12);
    char line[192];
    if (r.pixel_coords) {
      std::snprintf(line, sizeof(line), "%-3d %-14s %8.0f %4d %5.1f %5.1f", r.player_number,
                    team_display_name(r.team, config.hall_mode).c_str(), r.total_distance_m,
                    r.sprint_count, r.avg_speed_m_s, r.max_speed_m_s);
    } else {
      std::snprintf(line, sizeof(line), "%-3d %-14s %8.0f %4d %5.2f %5.2f", r.player_number,
                    team_display_name(r.team, config.hall_mode).c_str(), r.total_distance_m,
                    r.sprint_count, r.avg_speed_m_s, r.max_speed_m_s);
    }
    pdf_text_at(pdf, page, 40, y, 8, line);
    y -= 11;
  }

  y -= 8;
  pdf_title_at(pdf, page, 40, y, 11, "Players");
  y -= 8;

  constexpr float kCardH = 268.0f;
  constexpr float kThumbW = 72.0f;
  constexpr float kThumbH = 88.0f;

  for (const auto& r : reports) {
    ensure_space(pdf, &page, &y, kCardH + 8);
    const float card_top = y;
    const float card_bottom = y - kCardH;

    HPDF_Page_SetRGBStroke(page, 0.82f, 0.82f, 0.82f);
    HPDF_Page_SetLineWidth(page, 0.6f);
    HPDF_Page_Rectangle(page, 40, card_bottom, 515, kCardH);
    HPDF_Page_Stroke(page);

    std::string title = "Player #" + std::to_string(r.player_number);
    if (r.jersey_number > 0) {
      title += "   jersey #" + std::to_string(r.jersey_number);
    }
    pdf_title_at(pdf, page, 48, card_top - 14, 11, title);
    pdf_text_at(pdf, page, 48, card_top - 26, 8,
                "Team: " + team_display_name(r.team, config.hall_mode) +
                    "   id: " + fmt1(r.identity_confidence * 100.0) + "%");

    const char* dist_u = r.pixel_coords ? "px" : "m";
    const char* spd_u = r.pixel_coords ? "px/s" : "m/s";
    char stats1[160];
    char stats2[160];
    std::snprintf(stats1, sizeof(stats1), "Dist %d%s   avg %s %s   max %s %s",
                  static_cast<int>(r.total_distance_m), dist_u, fmt1(r.avg_speed_m_s).c_str(),
                  spd_u, fmt1(r.max_speed_m_s).c_str(), spd_u);
    std::snprintf(stats2, sizeof(stats2), "Sprints %d (%d%s)   HI %d   Active %d%%   Zones %d/%d/%d",
                  r.sprint_count, static_cast<int>(r.sprint_distance_m), dist_u,
                  static_cast<int>(r.hi_run_count), static_cast<int>(r.active_time_pct),
                  static_cast<int>(r.defensive_third_pct), static_cast<int>(r.middle_third_pct),
                  static_cast<int>(r.attacking_third_pct));
    pdf_text_at(pdf, page, 48, card_top - 38, 8, stats1);
    pdf_text_at(pdf, page, 48, card_top - 50, 8, stats2);

    const float thumbs_y = card_top - 50 - 12 - kThumbH;
    draw_jpeg_thumb(pdf, page, r.detect_cam1_jpg, 48, thumbs_y, kThumbW, kThumbH, "Detect C1");
    draw_jpeg_thumb(pdf, page, r.detect_cam2_jpg, 128, thumbs_y, kThumbW, kThumbH, "Detect C2");
    draw_jpeg_thumb(pdf, page, r.lost_cam1_jpg, 208, thumbs_y, kThumbW, kThumbH, "Lost C1");
    draw_jpeg_thumb(pdf, page, r.lost_cam2_jpg, 288, thumbs_y, kThumbW, kThumbH, "Lost C2");
    draw_jpeg_thumb(pdf, page, r.join_cam1_jpg, 368, thumbs_y, kThumbW, kThumbH, "Join C1");
    draw_jpeg_thumb(pdf, page, r.join_cam2_jpg, 448, thumbs_y, kThumbW, kThumbH, "Join C2");

    const float maps_y = card_bottom + 8;
    pdf_text_at(pdf, page, 48, maps_y + 108, 6, "Event map");
    draw_png_fit(pdf, page, r.event_map_png, 48, maps_y, 250, 100);
    pdf_text_at(pdf, page, 310, maps_y + 108, 6, "Heatmap");
    draw_png_fit(pdf, page, r.heatmap_png, 310, maps_y, 230, 100);

    y = card_bottom - 8;
  }

  const HPDF_STATUS status = HPDF_SaveToFile(pdf, path.string().c_str());
  HPDF_Free(pdf);
  return status == HPDF_OK;
}

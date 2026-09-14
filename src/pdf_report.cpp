#include "pdf_report.hpp"

#include <cmath>
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

void pdf_text(HPDF_Doc pdf, HPDF_Page page, float x, float y, const std::string& text) {
  HPDF_Page_BeginText(page);
  HPDF_Page_SetFontAndSize(page, pdf_body_font(pdf), 11);
  HPDF_Page_MoveTextPos(page, x, y);
  HPDF_Page_ShowText(page, text.c_str());
  HPDF_Page_EndText(page);
}

void pdf_title(HPDF_Doc pdf, HPDF_Page page, float y, const std::string& text) {
  HPDF_Page_BeginText(page);
  HPDF_Page_SetFontAndSize(page, pdf_title_font(pdf), 16);
  HPDF_Page_MoveTextPos(page, 50, y);
  HPDF_Page_ShowText(page, text.c_str());
  HPDF_Page_EndText(page);
}

}  // namespace

bool write_pdf_report(const std::filesystem::path& path, const ProjectConfig& config,
                      const std::vector<PlayerStatsReport>& reports,
                      const AnalysisQuality& quality) {
  HPDF_Doc pdf = HPDF_New(nullptr, nullptr);
  if (!pdf) return false;

  HPDF_Page page = HPDF_AddPage(pdf);
  HPDF_Page_SetSize(page, HPDF_PAGE_SIZE_A4, HPDF_PAGE_PORTRAIT);
  float y = 800;

  pdf_title(pdf, page, y, "Player Track Match Report");
  y -= 30;
  pdf_text(pdf, page, 50, y, "Project mode: " + project_mode_to_string(quality.project_mode));
  y -= 18;
  pdf_text(pdf, page, 50, y, "Pitch: " + std::to_string(config.pitch_length_m) + " x " +
                               std::to_string(config.pitch_width_m) + " m");
  y -= 18;
  pdf_text(pdf, page, 50, y, "Cameras: " + std::to_string(config.cameras.size()));
  y -= 18;
  pdf_text(pdf, page, 50, y,
           "Identity confidence: " + std::to_string(quality.identity_confidence_pct) + "%");
  y -= 18;
  pdf_text(pdf, page, 50, y,
           "Calibration confidence: " + std::to_string(quality.calib_confidence_pct) + "%");
  y -= 24;
  if (!quality.disclaimer.empty()) {
    pdf_text(pdf, page, 50, y, quality.disclaimer);
    y -= 24;
  }

  pdf_title(pdf, page, y, "Summary");
  y -= 24;
  const bool pixel_mode = !reports.empty() && reports.front().pixel_coords;
  if (pixel_mode) {
    pdf_text(pdf, page, 50, y, "#  Team     Distance  Sprints  AvgSpeed  MaxSpeed (px, px/s)");
  } else {
    pdf_text(pdf, page, 50, y, "#  Team     Distance  Sprints  AvgSpeed  MaxSpeed");
  }
  y -= 16;

  for (const auto& r : reports) {
    if (y < 80) {
      page = HPDF_AddPage(pdf);
      HPDF_Page_SetSize(page, HPDF_PAGE_SIZE_A4, HPDF_PAGE_PORTRAIT);
      y = 800;
    }
    char line[256];
    if (r.pixel_coords) {
      std::snprintf(line, sizeof(line), "%-3d %-8s %7.0fpx %7d %7.1f %7.1f", r.player_number,
                    team_to_string(r.team).c_str(), r.total_distance_m, r.sprint_count,
                    r.avg_speed_m_s, r.max_speed_m_s);
    } else {
      std::snprintf(line, sizeof(line), "%-3d %-8s %7.0fm %7d %7.2f %7.2f", r.player_number,
                    team_to_string(r.team).c_str(), r.total_distance_m, r.sprint_count,
                    r.avg_speed_m_s, r.max_speed_m_s);
    }
    pdf_text(pdf, page, 50, y, line);
    y -= 14;
  }

  for (const auto& r : reports) {
    page = HPDF_AddPage(pdf);
    HPDF_Page_SetSize(page, HPDF_PAGE_SIZE_A4, HPDF_PAGE_PORTRAIT);
    y = 800;
    pdf_title(pdf, page, y, "Player #" + std::to_string(r.player_number));
    y -= 28;
    pdf_text(pdf, page, 50, y, "Team: " + team_to_string(r.team));
    y -= 16;
    if (r.pixel_coords) {
      pdf_text(pdf, page, 50, y,
               "Total distance: " + std::to_string(static_cast<int>(r.total_distance_m)) + " px");
      y -= 16;
      pdf_text(pdf, page, 50, y,
               "Average speed: " + std::to_string(r.avg_speed_m_s).substr(0, 5) + " px/s");
      y -= 16;
      pdf_text(pdf, page, 50, y,
               "Max speed: " + std::to_string(r.max_speed_m_s).substr(0, 5) + " px/s");
      y -= 16;
      pdf_text(pdf, page, 50, y, "Sprints: " + std::to_string(r.sprint_count) + " (" +
                                     std::to_string(static_cast<int>(r.sprint_distance_m)) +
                                     " px)");
      y -= 16;
      pdf_text(pdf, page, 50, y,
               "Active time: " + std::to_string(static_cast<int>(r.active_time_pct)) + "%");
      y -= 16;
      pdf_text(pdf, page, 50, y, "Frame zones L/M/R: " +
                                     std::to_string(static_cast<int>(r.defensive_third_pct)) +
                                     "/" +
                                     std::to_string(static_cast<int>(r.middle_third_pct)) + "/" +
                                     std::to_string(static_cast<int>(r.attacking_third_pct)) + "%");
    } else {
      pdf_text(pdf, page, 50, y,
               "Total distance: " + std::to_string(static_cast<int>(r.total_distance_m)) + " m");
      y -= 16;
      pdf_text(pdf, page, 50, y,
               "Average speed: " + std::to_string(r.avg_speed_m_s).substr(0, 5) + " m/s");
      y -= 16;
      pdf_text(pdf, page, 50, y,
               "Max speed: " + std::to_string(r.max_speed_m_s).substr(0, 5) + " m/s");
      y -= 16;
      pdf_text(pdf, page, 50, y, "Sprints: " + std::to_string(r.sprint_count) + " (" +
                                     std::to_string(static_cast<int>(r.sprint_distance_m)) + " m)");
      y -= 16;
      pdf_text(pdf, page, 50, y,
               "Active time: " + std::to_string(static_cast<int>(r.active_time_pct)) + "%");
      y -= 16;
      pdf_text(pdf, page, 50, y, "Zones D/M/A: " +
                                     std::to_string(static_cast<int>(r.defensive_third_pct)) + "/" +
                                     std::to_string(static_cast<int>(r.middle_third_pct)) + "/" +
                                     std::to_string(static_cast<int>(r.attacking_third_pct)) + "%");
    }
    y -= 16;
    pdf_text(pdf, page, 50, y,
             "High-intensity runs: " + std::to_string(static_cast<int>(r.hi_run_count)));
    y -= 24;

    if (!r.heatmap_png.empty() && std::filesystem::exists(r.heatmap_png)) {
      HPDF_Image img = HPDF_LoadPngImageFromFile(pdf, r.heatmap_png.string().c_str());
      if (img) {
        HPDF_Page_DrawImage(page, img, 50, y - 220, 320, 200);
      }
    }
  }

  const HPDF_STATUS status = HPDF_SaveToFile(pdf, path.string().c_str());
  HPDF_Free(pdf);
  return status == HPDF_OK;
}

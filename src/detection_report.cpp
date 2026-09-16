#include "detection_report.hpp"

#include <fstream>
#include <iomanip>
#include <map>

namespace {

void write_summary_block(std::ostream& out, const DetectionReportSummary& summary) {
  out << "Player Track Detection Report\n";
  out << "=============================\n\n";

  out << "Settings\n";
  out << "  model: " << summary.config.model_path << '\n';
  out << "  detect_confidence: " << summary.config.detect_confidence << '\n';
  out << "  detect_frame_step: " << summary.config.detect_frame_step << '\n';
  out << "  detect_input_size: " << summary.config.detect_input_size << '\n';
  out << "  detect_max_width: " << summary.config.detect_max_width << '\n';
  out << "  inference_provider: " << summary.detector_provider << '\n';
  out << "  cameras: " << summary.config.cameras.size() << '\n';
  out << '\n';

  out << "Summary\n";
  out << "  frames_processed: " << summary.frames_processed << '\n';
  out << "  detect_frames: " << summary.detect_frames << '\n';
  out << "  raw_yolo_detections: " << summary.raw_detections_total << '\n';
  out << "  tracklets_total: " << summary.tracklets_total << '\n';
  out << "  tracklets_kept: " << summary.tracklets_kept << '\n';
  out << "  tracklets_rejected: " << summary.tracklets_rejected << '\n';
  out << "  resolved_players: " << summary.resolved_players << '\n';
  out << '\n';

  std::map<std::string, int> reject_counts;
  for (const auto& row : summary.tracklets) {
    if (!row.kept && !row.reject_reason.empty()) {
      ++reject_counts[row.reject_reason];
    }
  }
  if (!reject_counts.empty()) {
    out << "Reject reasons\n";
    for (const auto& [reason, count] : reject_counts) {
      out << "  " << reason << ": " << count << '\n';
    }
    out << '\n';
  }

  out << "Notes\n";
  out << "  Empty boxes in annotated video are usually short-lived or stationary false\n";
  out << "  detections (field markings, shadows). Only kept tracklets go to CSV/stats.\n";
  out << "  Lower detect_confidence finds more players but also more false boxes.\n\n";
}

}  // namespace

bool write_detection_report(const std::filesystem::path& txt_path,
                            const std::filesystem::path& csv_path,
                            const DetectionReportSummary& summary) {
  {
    std::ofstream out(txt_path, std::ios::out | std::ios::trunc);
    if (!out) return false;
    write_summary_block(out, summary);

    out << "Tracklets\n";
    out << std::fixed << std::setprecision(2);
    out << "camera\ttrack_id\tt_start\tt_end\tduration_s\tobs\tjersey\tmax_conf\tavg_conf\t"
           "move_px\tmax_spd_px_s\tkept\treason\n";
    for (const auto& row : summary.tracklets) {
      out << row.camera_name << '\t' << row.local_track_id << '\t' << row.t_start_s << '\t'
          << row.t_end_s << '\t' << row.duration_s << '\t' << row.observation_count << '\t';
      if (row.jersey_number > 0) {
        out << row.jersey_number;
      } else {
        out << '-';
      }
      out << '\t' << row.max_detect_confidence << '\t' << row.avg_detect_confidence << '\t'
          << row.movement_px << '\t' << row.max_speed_px_s << '\t' << (row.kept ? "yes" : "no")
          << '\t' << row.reject_reason << '\n';
    }
  }

  {
    std::ofstream out(csv_path, std::ios::out | std::ios::trunc);
    if (!out) return false;
    out << "camera,camera_index,track_id,t_start_s,t_end_s,duration_s,observations,jersey,"
           "jersey_conf,max_detect_conf,avg_detect_conf,movement_px,max_speed_px_s,bbox_x,bbox_y,"
           "bbox_w,bbox_h,kept,reject_reason\n";
    out << std::fixed << std::setprecision(4);
    for (const auto& row : summary.tracklets) {
      out << row.camera_name << ',' << row.camera_index << ',' << row.local_track_id << ','
          << row.t_start_s << ',' << row.t_end_s << ',' << row.duration_s << ','
          << row.observation_count << ',';
      if (row.jersey_number > 0) {
        out << row.jersey_number;
      }
      out << ',' << row.jersey_confidence << ',' << row.max_detect_confidence << ','
          << row.avg_detect_confidence << ',' << row.movement_px << ',' << row.max_speed_px_s
          << ',' << row.bbox_x << ',' << row.bbox_y << ',' << row.bbox_w << ',' << row.bbox_h
          << ',' << (row.kept ? "yes" : "no") << ',' << row.reject_reason << '\n';
    }
  }

  return true;
}

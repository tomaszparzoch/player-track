#include "csv_writer.hpp"

#include <iomanip>

bool CsvWriter::open(const std::filesystem::path& path, bool use_meters) {
  out_.open(path, std::ios::out | std::ios::trunc);
  if (!out_) {
    return false;
  }
  use_meters_ = use_meters;
  if (use_meters_) {
    out_ << "t_s,player_id,x_m,y_m,vx_m_s,vy_m_s,speed_m_s,distance_m,simulated\n";
  } else {
    out_ << "t_s,player_id,x_px,y_px,vx_px_s,vy_px_s,speed_px_s,distance_px,simulated\n";
  }
  out_ << std::fixed << std::setprecision(3);
  return true;
}

void CsvWriter::write(const Sample& sample) {
  if (use_meters_) {
    out_ << sample.t_s << ',' << sample.player_id << ',' << sample.x_m << ',' << sample.y_m << ','
         << sample.vx_m_s << ',' << sample.vy_m_s << ',' << sample.speed_m_s << ','
         << sample.distance_m << ',' << (sample.simulated ? 1 : 0) << '\n';
    return;
  }
  out_ << sample.t_s << ',' << sample.player_id << ',' << sample.x_px << ',' << sample.y_px
       << ',' << sample.vx_px_s << ',' << sample.vy_px_s << ',' << sample.speed_px_s << ','
       << sample.distance_px << ',' << (sample.simulated ? 1 : 0) << '\n';
}

bool CsvWriter::is_open() const {
  return static_cast<bool>(out_);
}

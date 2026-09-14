#include "heatmap.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

struct Rgb {
  int r{};
  int g{};
  int b{};
};

Rgb heat_color(double t) {
  t = std::clamp(t, 0.0, 1.0);
  if (t < 0.25) {
    const double u = t / 0.25;
    return {0, static_cast<int>(80 + 175 * u), 255};
  }
  if (t < 0.5) {
    const double u = (t - 0.25) / 0.25;
    return {0, 255, static_cast<int>(255 * (1.0 - u))};
  }
  if (t < 0.75) {
    const double u = (t - 0.5) / 0.25;
    return {static_cast<int>(255 * u), 255, 0};
  }
  const double u = (t - 0.75) / 0.25;
  return {255, static_cast<int>(255 * (1.0 - u)), 0};
}

void splat_gaussian(std::vector<float>& grid, int cols, int rows, double gx, double gy,
                    double sigma_cells) {
  const int radius = static_cast<int>(std::ceil(3.0 * sigma_cells));
  const int cx = static_cast<int>(std::floor(gx));
  const int cy = static_cast<int>(std::floor(gy));
  const double inv_2_sigma2 = 1.0 / (2.0 * sigma_cells * sigma_cells);

  for (int y = std::max(0, cy - radius); y < std::min(rows, cy + radius + 1); ++y) {
    for (int x = std::max(0, cx - radius); x < std::min(cols, cx + radius + 1); ++x) {
      const double dx = x - gx;
      const double dy = y - gy;
      const double weight = std::exp(-(dx * dx + dy * dy) * inv_2_sigma2);
      grid[y * cols + x] += static_cast<float>(weight);
    }
  }
}

}  // namespace

bool write_player_heatmap_svg(const std::filesystem::path& path, int player_id,
                              const std::vector<cv::Point2f>& positions, int frame_width,
                              int frame_height, const HeatmapConfig& config) {
  if (positions.empty() || frame_width <= 0 || frame_height <= 0 || config.cell_size <= 0) {
    return false;
  }

  const int cols = (frame_width + config.cell_size - 1) / config.cell_size;
  const int rows = (frame_height + config.cell_size - 1) / config.cell_size;
  std::vector<float> grid(cols * rows, 0.0f);

  for (const auto& pt : positions) {
    const double gx = pt.x / config.cell_size;
    const double gy = pt.y / config.cell_size;
    if (gx < 0.0 || gy < 0.0 || gx >= cols || gy >= rows) {
      continue;
    }
    splat_gaussian(grid, cols, rows, gx, gy, config.sigma_cells);
  }

  const float max_value = *std::max_element(grid.begin(), grid.end());
  if (max_value <= 0.0f) {
    return false;
  }

  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    return false;
  }

  out << std::fixed << std::setprecision(2);
  out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << config.svg_width << "\" height=\""
      << config.svg_height << "\" viewBox=\"0 0 " << frame_width << ' ' << frame_height
      << "\">\n";
  out << "  <title>Player " << player_id << " heatmap</title>\n";
  out << "  <rect width=\"100%\" height=\"100%\" fill=\"#1a3d1a\"/>\n";

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const float value = grid[row * cols + col];
      if (value <= 0.0f) {
        continue;
      }
      const double t = value / max_value;
      if (t < 0.03) {
        continue;
      }
      const Rgb color = heat_color(t);
      const double opacity = 0.15 + 0.75 * t;
      const double x = col * config.cell_size;
      const double y = row * config.cell_size;
      out << "  <rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << config.cell_size
          << "\" height=\"" << config.cell_size << "\" fill=\"rgb(" << color.r << ',' << color.g
          << ',' << color.b << ")\" fill-opacity=\"" << opacity << "\"/>\n";
    }
  }

  for (const auto& pt : positions) {
    out << "  <circle cx=\"" << pt.x << "\" cy=\"" << pt.y << "\" r=\"2.5\" fill=\"white\" "
           "fill-opacity=\"0.25\"/>\n";
  }

  out << "</svg>\n";
  return true;
}

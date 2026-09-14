#include "app_options.hpp"
#include "gui_app.hpp"
#include "multi_camera_session.hpp"
#include "time_format.hpp"
#include "tracking_session.hpp"

#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
  if (argc == 1) {
    return run_gui_app();
  }

  bool gui = false;
  std::string project_path;
  std::vector<char*> filtered;
  filtered.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    if (arg == "--gui") {
      gui = true;
      continue;
    }
    if (arg == "--project") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --project\n";
        return 1;
      }
      project_path = argv[++i];
      continue;
    }
    filtered.push_back(argv[i]);
  }

  if (gui) {
    return run_gui_app();
  }

  if (!project_path.empty()) {
    const auto config = load_project_config(project_path);
    if (!config) {
      std::cerr << "Failed to load project: " << project_path << "\n";
      return 1;
    }
    const auto analysis_start = std::chrono::steady_clock::now();
    const auto result = run_multi_camera_session(*config, [&](double p, const std::string& msg) {
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - analysis_start).count();
      const double eta = estimate_remaining_seconds(p, elapsed);
      std::cerr << "[" << static_cast<int>(p * 100) << "%] " << msg << " | elapsed "
                << format_duration(elapsed);
      if (eta >= 0.0) {
        std::cerr << " | ETA ~" << format_duration(eta);
      }
      std::cerr << '\n';
    });
    if (!result.success) {
      std::cerr << result.error << "\n";
      return 1;
    }
    std::cerr << "Report: " << result.pdf_report << "\n";
    std::cerr << "Analysis time: " << format_duration(result.analysis_duration_s) << "\n";
    return 0;
  }

  const int new_argc = static_cast<int>(filtered.size());
  const auto opts = parse_args(new_argc, filtered.data());
  if (!opts) {
    return 1;
  }
  return run_tracking_session(*opts);
}

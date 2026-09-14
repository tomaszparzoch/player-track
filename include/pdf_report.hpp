#pragma once

#include <filesystem>

#include "player_stats.hpp"
#include "project_config.hpp"

bool write_pdf_report(const std::filesystem::path& path, const ProjectConfig& config,
                      const std::vector<PlayerStatsReport>& reports,
                      const AnalysisQuality& quality);

#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>

#include <opencv2/core.hpp>

struct VideoCompressOptions {
  int crf{32};
  int max_width{1280};
  bool match_source_bitrate{true};
  double max_duration_s{0.0};
};

struct SourceVideoInfo {
  int width{};
  int height{};
  double fps{};
  int64_t bitrate_bps{};
  double duration_s{};
};

bool compress_video_h264(const std::filesystem::path& annotated_path,
                         const std::filesystem::path& reference_source_path,
                         const VideoCompressOptions& options,
                         std::string* log_message = nullptr);

bool write_compression_log(const std::filesystem::path& path, const std::string& message);

std::optional<std::filesystem::path> find_ffmpeg_executable();
std::optional<std::filesystem::path> find_ffprobe_executable();
std::optional<SourceVideoInfo> probe_video_info(const std::filesystem::path& path);

cv::Size annotated_export_size(const cv::Size& frame_size, int max_width);

using VideoStackProgressCallback = std::function<void(double progress, const std::string& message)>;

bool stack_videos_side_by_side(const std::vector<std::filesystem::path>& video_paths,
                               const std::vector<double>& offsets_s, double fps,
                               const cv::Size& panel_size, const std::filesystem::path& out_path,
                               std::string* log_message = nullptr,
                               VideoStackProgressCallback progress = {},
                               double max_output_duration_s = 0.0);

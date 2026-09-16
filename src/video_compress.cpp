#include "video_compress.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

std::string format_bytes(std::uintmax_t bytes) {
  std::ostringstream out;
  if (bytes >= 1024 * 1024) {
    out.setf(std::ios::fixed);
    out.precision(1);
    out << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB";
  } else if (bytes >= 1024) {
    out << (bytes / 1024) << " KB";
  } else {
    out << bytes << " B";
  }
  return out.str();
}

std::string trim_line(std::string line) {
  while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) {
    line.pop_back();
  }
  return line;
}

struct CommandResult {
  int exit_code{-1};
  std::string output;
};

using ProcessChunkCallback = std::function<void(const std::string& chunk)>;

#ifdef _WIN32
std::wstring quote_arg_win(const std::wstring& arg) {
  if (arg.empty()) return L"\"\"";
  bool needs_quotes = false;
  for (const wchar_t c : arg) {
    if (c == L' ' || c == L'\t' || c == L'"') {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes) return arg;

  std::wstring out = L"\"";
  for (size_t i = 0; i < arg.size(); ++i) {
    int backslashes = 0;
    while (i < arg.size() && arg[i] == L'\\') {
      ++backslashes;
      ++i;
    }
    if (i == arg.size()) {
      out.append(static_cast<size_t>(backslashes * 2), L'\\');
      break;
    }
    if (arg[i] == L'"') {
      out.append(static_cast<size_t>(backslashes * 2 + 1), L'\\');
      out += L'"';
    } else {
      out.append(static_cast<size_t>(backslashes), L'\\');
      out += arg[i];
    }
  }
  out += L'"';
  return out;
}

std::wstring build_command_line(const std::filesystem::path& executable,
                                const std::vector<std::wstring>& args) {
  std::wstring cmdline = quote_arg_win(executable.wstring());
  for (const auto& arg : args) {
    cmdline += L' ';
    cmdline += quote_arg_win(arg);
  }
  return cmdline;
}

CommandResult run_process_capture(const std::filesystem::path& executable,
                                  const std::vector<std::wstring>& args,
                                  ProcessChunkCallback on_chunk) {
  CommandResult result;
  const std::wstring cmdline = build_command_line(executable, args);
  std::vector<wchar_t> cmdline_buf(cmdline.begin(), cmdline.end());
  cmdline_buf.push_back(L'\0');

  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;

  HANDLE stdout_read = nullptr;
  HANDLE stdout_write = nullptr;
  if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0)) {
    result.output = "Failed to create pipe for ffmpeg process\n";
    return result;
  }
  SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);

  STARTUPINFOW si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdOutput = stdout_write;
  si.hStdError = stdout_write;
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

  PROCESS_INFORMATION pi{};
  if (!CreateProcessW(executable.wstring().c_str(), cmdline_buf.data(), nullptr, nullptr, TRUE,
                      CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
    CloseHandle(stdout_read);
    CloseHandle(stdout_write);
    result.output = "Failed to start ffmpeg process\n";
    return result;
  }

  CloseHandle(stdout_write);
  CloseHandle(pi.hThread);

  char buffer[4096];
  DWORD bytes_read = 0;
  for (;;) {
    while (PeekNamedPipe(stdout_read, nullptr, 0, nullptr, &bytes_read, nullptr) && bytes_read > 0) {
      const DWORD chunk = std::min<DWORD>(bytes_read, static_cast<DWORD>(sizeof(buffer) - 1));
      if (!ReadFile(stdout_read, buffer, chunk, &bytes_read, nullptr) || bytes_read == 0) {
        break;
      }
      buffer[bytes_read] = '\0';
      result.output += buffer;
      if (on_chunk) {
        on_chunk(std::string(buffer, bytes_read));
      }
    }

    const DWORD wait = WaitForSingleObject(pi.hProcess, 50);
    if (wait == WAIT_OBJECT_0) {
      break;
    }
  }

  while (ReadFile(stdout_read, buffer, static_cast<DWORD>(sizeof(buffer) - 1), &bytes_read,
                  nullptr) &&
         bytes_read > 0) {
    buffer[bytes_read] = '\0';
    result.output += buffer;
    if (on_chunk) {
      on_chunk(std::string(buffer, bytes_read));
    }
  }

  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD exit_code = 1;
  GetExitCodeProcess(pi.hProcess, &exit_code);
  result.exit_code = static_cast<int>(exit_code);

  CloseHandle(stdout_read);
  CloseHandle(pi.hProcess);
  return result;
}
#endif

std::optional<std::string> run_capture_command(const std::filesystem::path& executable,
                                               const std::vector<std::wstring>& args) {
#ifdef _WIN32
  const auto result = run_process_capture(executable, args, {});
  if (result.exit_code != 0) return std::nullopt;
  const std::string line = trim_line(result.output);
  if (line.empty()) return std::nullopt;
  return line;
#else
  std::ostringstream cmd;
  cmd << quote_path(executable);
  for (const auto& arg : args) {
    cmd << ' ' << quote_path(std::filesystem::path(arg));
  }
  return run_capture_command_string(cmd.str());
#endif
}

std::optional<std::string> run_capture_command_string(const std::string& command) {
  FILE* pipe = _popen(command.c_str(), "r");
  if (!pipe) return std::nullopt;
  char buffer[512];
  if (!fgets(buffer, static_cast<int>(sizeof(buffer)), pipe)) {
    _pclose(pipe);
    return std::nullopt;
  }
  _pclose(pipe);
  const std::string line = trim_line(buffer);
  if (line.empty()) return std::nullopt;
  return line;
}

double parse_fps_token(const std::string& token) {
  const auto slash = token.find('/');
  if (slash == std::string::npos) {
    try {
      return std::stod(token);
    } catch (...) {
      return 0.0;
    }
  }
  try {
    const double num = std::stod(token.substr(0, slash));
    const double den = std::stod(token.substr(slash + 1));
    return den > 0.0 ? num / den : 0.0;
  } catch (...) {
    return 0.0;
  }
}

int64_t parse_int64_token(const std::string& token) {
  if (token.empty() || token == "N/A") return 0;
  try {
    return std::stoll(token);
  } catch (...) {
    return 0;
  }
}

#ifdef _WIN32
std::filesystem::path module_directory() {
  wchar_t buffer[MAX_PATH];
  const DWORD len = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
  if (len == 0 || len >= MAX_PATH) return {};
  return std::filesystem::path(buffer).parent_path();
}

void append_if_exists(const std::filesystem::path& path, std::vector<std::filesystem::path>& out) {
  if (!path.empty() && std::filesystem::exists(path)) {
    out.push_back(path);
  }
}

void append_ffmpeg_from_winget_packages(std::vector<std::filesystem::path>& out) {
  wchar_t* local_app_data = nullptr;
  size_t len = 0;
  if (_wdupenv_s(&local_app_data, &len, L"LOCALAPPDATA") != 0 || !local_app_data) {
    return;
  }
  const std::filesystem::path packages =
      std::filesystem::path(local_app_data) / "Microsoft" / "WinGet" / "Packages";
  free(local_app_data);

  std::error_code ec;
  if (!std::filesystem::exists(packages, ec)) return;

  for (const auto& package_entry : std::filesystem::directory_iterator(packages, ec)) {
    if (ec || !package_entry.is_directory()) continue;
    append_if_exists(package_entry.path() / "bin" / "ffmpeg.exe", out);
    append_if_exists(package_entry.path() / "bin" / "ffprobe.exe", out);
    for (const auto& sub_entry : std::filesystem::directory_iterator(package_entry.path(), ec)) {
      if (ec || !sub_entry.is_directory()) continue;
      append_if_exists(sub_entry.path() / "bin" / "ffmpeg.exe", out);
      append_if_exists(sub_entry.path() / "bin" / "ffprobe.exe", out);
    }
  }
}
#endif

std::optional<std::filesystem::path> find_executable_candidates(const char* name) {
  std::vector<std::filesystem::path> candidates;

#ifdef _WIN32
  const std::string where_cmd = std::string("where ") + name + " 2>nul";
  if (FILE* pipe = _popen(where_cmd.c_str(), "r")) {
    char buffer[512];
    while (fgets(buffer, static_cast<int>(sizeof(buffer)), pipe)) {
      const std::string line = trim_line(buffer);
      if (!line.empty()) {
        candidates.emplace_back(line);
      }
    }
    _pclose(pipe);
  }

  const auto exe_dir = module_directory();
  append_if_exists(exe_dir / (std::string(name) + ".exe"), candidates);
  append_ffmpeg_from_winget_packages(candidates);
#else
  const std::string which_cmd = std::string("command -v ") + name + " 2>/dev/null";
  if (const auto output = run_capture_command_string(which_cmd)) {
    candidates.emplace_back(*output);
  }
#endif

  for (const auto& candidate : candidates) {
    if (!std::filesystem::exists(candidate)) continue;
    const std::string text = candidate.string();
    if (text.find("Gyan.FFmpeg") != std::string::npos ||
        text.find("full_build") != std::string::npos) {
      return candidate;
    }
  }

  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  return std::nullopt;
}

std::optional<std::filesystem::path> sibling_tool(const std::filesystem::path& ffmpeg_path,
                                                  const char* tool_name) {
  const auto sibling = ffmpeg_path.parent_path() / (std::string(tool_name) + ".exe");
  if (std::filesystem::exists(sibling)) {
    return sibling;
  }
  return find_executable_candidates(tool_name);
}

std::filesystem::path absolute_path(const std::filesystem::path& path) {
  std::error_code ec;
  const auto abs_path = std::filesystem::absolute(path, ec);
  return ec ? path.lexically_normal() : abs_path.lexically_normal();
}

std::string quote_path(const std::filesystem::path& path) {
  return '"' + absolute_path(path).string() + '"';
}

CommandResult run_shell_command(const std::filesystem::path& executable,
                                const std::vector<std::wstring>& args,
                                ProcessChunkCallback on_chunk = {}) {
#ifdef _WIN32
  return run_process_capture(executable, args, on_chunk);
#else
  CommandResult result;
  std::ostringstream cmd;
  cmd << quote_path(executable);
  for (const auto& arg : args) {
    cmd << ' ' << quote_path(std::filesystem::path(arg));
  }
  const std::string wrapped = cmd.str() + " 2>&1";
  FILE* pipe = _popen(wrapped.c_str(), "r");
  if (!pipe) {
    result.output = "Failed to start ffmpeg process\n";
    return result;
  }

  char buffer[4096];
  while (fgets(buffer, static_cast<int>(sizeof(buffer)), pipe)) {
    result.output += buffer;
    if (on_chunk) {
      on_chunk(buffer);
    }
  }
  result.exit_code = _pclose(pipe);
  return result;
#endif
}

std::string format_command_for_log(const std::filesystem::path& executable,
                                   const std::vector<std::wstring>& args) {
  std::ostringstream out;
  out << quote_path(executable);
  for (const auto& arg : args) {
    out << ' ' << quote_path(std::filesystem::path(arg));
  }
  return out.str();
}

std::optional<double> probe_duration_seconds(const std::filesystem::path& ffprobe,
                                             const std::filesystem::path& path) {
#ifdef _WIN32
  const auto output = run_capture_command(
      ffprobe, {L"-v", L"error", L"-show_entries", L"format=duration", L"-of",
                L"default=noprint_wrappers=1:nokey=1", absolute_path(path).wstring()});
#else
  std::ostringstream cmd;
  cmd << quote_path(ffprobe)
      << " -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "
      << quote_path(path);
  const auto output = run_capture_command_string(cmd.str());
#endif
  if (!output) return std::nullopt;
  try {
    const double duration = std::stod(*output);
    return duration > 0.0 ? std::optional<double>{duration} : std::nullopt;
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<int64_t> bitrate_from_file_size(const std::filesystem::path& path, double duration_s) {
  if (duration_s <= 0.0) return std::nullopt;
  std::error_code ec;
  const auto size_bytes = std::filesystem::file_size(path, ec);
  if (ec || size_bytes == 0) return std::nullopt;
  return static_cast<int64_t>(static_cast<double>(size_bytes) * 8.0 / duration_s);
}

}  // namespace

std::optional<std::filesystem::path> find_ffmpeg_executable() {
  return find_executable_candidates("ffmpeg");
}

std::optional<std::filesystem::path> find_ffprobe_executable() {
  if (const auto ffmpeg = find_ffmpeg_executable()) {
    if (auto ffprobe = sibling_tool(*ffmpeg, "ffprobe")) {
      return ffprobe;
    }
  }
  return find_executable_candidates("ffprobe");
}

cv::Size annotated_export_size(const cv::Size& frame_size, int max_width) {
  if (max_width <= 0 || frame_size.width <= max_width) {
    return frame_size;
  }
  const double scale = static_cast<double>(max_width) / frame_size.width;
  return {max_width, std::max(1, static_cast<int>(std::round(frame_size.height * scale)))};
}

std::optional<SourceVideoInfo> probe_video_info(const std::filesystem::path& path) {
  const auto ffprobe = find_ffprobe_executable();
  const auto abs_path = absolute_path(path);
  if (!ffprobe || !std::filesystem::exists(abs_path)) return std::nullopt;

  SourceVideoInfo info;

  {
#ifdef _WIN32
    const auto output = run_capture_command(
        *ffprobe, {L"-v", L"error", L"-select_streams", L"v:0", L"-show_entries", L"stream=width",
                   L"-of", L"default=noprint_wrappers=1:nokey=1", abs_path.wstring()});
#else
    std::ostringstream cmd;
    cmd << quote_path(*ffprobe)
        << " -v error -select_streams v:0 -show_entries stream=width -of "
           "default=noprint_wrappers=1:nokey=1 "
        << quote_path(abs_path);
    const auto output = run_capture_command_string(cmd.str());
#endif
    if (output) {
      info.width = static_cast<int>(parse_int64_token(*output));
    }
  }
  {
#ifdef _WIN32
    const auto output = run_capture_command(
        *ffprobe, {L"-v", L"error", L"-select_streams", L"v:0", L"-show_entries", L"stream=height",
                   L"-of", L"default=noprint_wrappers=1:nokey=1", abs_path.wstring()});
#else
    std::ostringstream cmd;
    cmd << quote_path(*ffprobe)
        << " -v error -select_streams v:0 -show_entries stream=height -of "
           "default=noprint_wrappers=1:nokey=1 "
        << quote_path(abs_path);
    const auto output = run_capture_command_string(cmd.str());
#endif
    if (output) {
      info.height = static_cast<int>(parse_int64_token(*output));
    }
  }
  {
#ifdef _WIN32
    const auto output = run_capture_command(
        *ffprobe,
        {L"-v", L"error", L"-select_streams", L"v:0", L"-show_entries", L"stream=avg_frame_rate",
         L"-of", L"default=noprint_wrappers=1:nokey=1", abs_path.wstring()});
#else
    std::ostringstream cmd;
    cmd << quote_path(*ffprobe)
        << " -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of "
           "default=noprint_wrappers=1:nokey=1 "
        << quote_path(abs_path);
    const auto output = run_capture_command_string(cmd.str());
#endif
    if (output) {
      info.fps = parse_fps_token(*output);
    }
  }

  if (const auto duration = probe_duration_seconds(*ffprobe, abs_path)) {
    info.duration_s = *duration;
  }

  if (info.duration_s > 0.0) {
    if (const auto bps = bitrate_from_file_size(abs_path, info.duration_s)) {
      info.bitrate_bps = *bps;
    }
  }

  if (info.width <= 0 || info.height <= 0) return std::nullopt;
  return info;
}

namespace {

std::optional<int64_t> reference_bitrate_bps(const std::filesystem::path& reference_path,
                                             const std::filesystem::path& annotated_path) {
  if (const auto info = probe_video_info(reference_path)) {
    if (info->bitrate_bps > 0) {
      return info->bitrate_bps;
    }
  }

  const auto ffprobe = find_ffprobe_executable();
  if (!ffprobe) return std::nullopt;

  std::error_code ec;
  if (!std::filesystem::exists(reference_path, ec)) return std::nullopt;
  const std::uintmax_t reference_size = std::filesystem::file_size(reference_path, ec);
  if (ec || reference_size == 0) return std::nullopt;

  if (const auto annotated_duration = probe_duration_seconds(*ffprobe, annotated_path)) {
    return static_cast<int64_t>(reference_size * 8.0 / *annotated_duration);
  }
  if (const auto reference_duration = probe_duration_seconds(*ffprobe, reference_path)) {
    return static_cast<int64_t>(reference_size * 8.0 / *reference_duration);
  }
  return std::nullopt;
}

bool run_ffmpeg_once(const std::filesystem::path& ffmpeg, const std::vector<std::wstring>& args,
                     const std::filesystem::path& temp_path, CommandResult* last_result) {
  std::error_code ec;
  std::filesystem::remove(temp_path, ec);
  *last_result = run_shell_command(ffmpeg, args);
  if (last_result->exit_code != 0) {
    return false;
  }
  if (!std::filesystem::exists(temp_path)) {
    return false;
  }
  if (std::filesystem::file_size(temp_path) == 0) {
    return false;
  }
  return true;
}

std::vector<std::wstring> build_h264_encode_args(const std::filesystem::path& input_path,
                                                 const std::filesystem::path& temp_path,
                                                 const VideoCompressOptions& options, int use_crf,
                                                 const std::optional<int64_t>& reference_bitrate,
                                                 bool* used_bitrate, int64_t* target_bps) {
  std::vector<std::wstring> args = {L"-y",       L"-hide_banner", L"-loglevel", L"warning",
                                    L"-i",       input_path.wstring(),
                                    L"-an",      L"-c:v",         L"libx264",
                                    L"-pix_fmt", L"yuv420p",      L"-preset",   L"medium",
                                    L"-movflags", L"+faststart"};

  *used_bitrate = false;
  *target_bps = 0;
  if (options.max_duration_s > 0.0) {
    args.push_back(L"-t");
    args.push_back(std::to_wstring(options.max_duration_s));
  }
  if (options.match_source_bitrate && reference_bitrate) {
    *target_bps =
        std::clamp(static_cast<int64_t>(*reference_bitrate * 1.20), 350'000LL, 8'000'000LL);
    const int64_t maxrate_bps =
        std::clamp(static_cast<int64_t>(*target_bps * 1.30), *target_bps + 150'000LL, 10'000'000LL);
    const int64_t bufsize_bps = maxrate_bps * 2;
    args.push_back(L"-b:v");
    args.push_back(std::to_wstring(std::max<int64_t>(1, *target_bps / 1000)) + L"k");
    args.push_back(L"-maxrate");
    args.push_back(std::to_wstring(std::max<int64_t>(1, maxrate_bps / 1000)) + L"k");
    args.push_back(L"-bufsize");
    args.push_back(std::to_wstring(std::max<int64_t>(1, bufsize_bps / 1000)) + L"k");
    *used_bitrate = true;
  } else {
    args.push_back(L"-crf");
    args.push_back(std::to_wstring(use_crf));
  }

  args.push_back(temp_path.wstring());
  return args;
}

bool compress_video_h264_impl(const std::filesystem::path& annotated_path,
                              const std::filesystem::path& reference_source_path,
                              const VideoCompressOptions& options, std::string* log_message) {
  const auto input_path = absolute_path(annotated_path);
  const auto reference_path = absolute_path(reference_source_path);

  if (!std::filesystem::exists(input_path)) {
    if (log_message) *log_message = "Video not found for compression: " + input_path.string();
    return false;
  }

  const auto ffmpeg = find_ffmpeg_executable();
  if (!ffmpeg) {
    if (log_message) {
      *log_message =
          "ffmpeg not found (PATH/WinGet). Annotated video kept uncompressed. Install: winget install ffmpeg";
    }
    return false;
  }

  const int use_crf = std::clamp(options.crf, 23, 40);
  const std::uintmax_t before_size = std::filesystem::file_size(input_path);
  const auto temp_path = absolute_path(
      input_path.parent_path() / (input_path.stem().string() + ".compress.tmp.mp4"));

  std::error_code ec;
  std::filesystem::remove(temp_path, ec);

  const auto reference_bitrate = reference_bitrate_bps(reference_path, input_path);

  bool used_bitrate = false;
  int64_t target_bps = 0;
  const auto encode_args = build_h264_encode_args(input_path, temp_path, options, use_crf,
                                                  reference_bitrate, &used_bitrate, &target_bps);
  const std::string command_log = format_command_for_log(*ffmpeg, encode_args);

  CommandResult ffmpeg_result;
  bool encoded = false;
  for (int attempt = 0; attempt < 5; ++attempt) {
    if (attempt > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(250 * attempt));
    }
    if (run_ffmpeg_once(*ffmpeg, encode_args, temp_path, &ffmpeg_result)) {
      encoded = true;
      break;
    }
  }

  if (!encoded) {
    std::filesystem::remove(temp_path, ec);
    if (log_message) {
      *log_message = "ffmpeg compression failed for " + input_path.filename().string() +
                     " (exit " + std::to_string(ffmpeg_result.exit_code) + ")\nCommand: " +
                     command_log;
      if (!ffmpeg_result.output.empty()) {
        *log_message += "\nffmpeg output:\n" + ffmpeg_result.output;
      }
    }
    return false;
  }

  std::filesystem::remove(input_path, ec);
  std::filesystem::rename(temp_path, input_path, ec);
  if (ec) {
    if (log_message) *log_message = "Failed to replace annotated video after compression";
    return false;
  }

  const std::uintmax_t after_size = std::filesystem::file_size(input_path);
  std::error_code size_ec;
  const std::uintmax_t source_size = std::filesystem::exists(reference_path, size_ec)
                                         ? std::filesystem::file_size(reference_path, size_ec)
                                         : 0;

  if (source_size > 0 && after_size > source_size * 2) {
    const auto retry_args =
        build_h264_encode_args(input_path, temp_path,
                               VideoCompressOptions{.crf = std::max(use_crf, 34),
                                                    .max_width = 0,
                                                    .match_source_bitrate = false},
                               std::max(use_crf, 34), std::nullopt, &used_bitrate, &target_bps);
    std::filesystem::remove(temp_path, ec);
    if (run_ffmpeg_once(*ffmpeg, retry_args, temp_path, &ffmpeg_result) &&
        std::filesystem::file_size(temp_path) < after_size) {
      std::filesystem::remove(input_path, ec);
      std::filesystem::rename(temp_path, input_path, ec);
    } else {
      std::filesystem::remove(temp_path, ec);
    }
  }

  const std::uintmax_t final_size = std::filesystem::file_size(input_path);
  if (log_message) {
    *log_message = "Compressed " + input_path.filename().string() + ": " +
                   format_bytes(before_size) + " -> " + format_bytes(final_size);
    if (used_bitrate && reference_bitrate) {
      *log_message += " (target ~" + std::to_string(static_cast<int>(target_bps / 1000)) +
                       " kbps from source " + format_bytes(std::filesystem::file_size(reference_path)) +
                       ")";
    } else {
      *log_message += " (CRF " + std::to_string(use_crf) + ")";
    }
    if (options.max_width > 0) {
      *log_message += ", export width " + std::to_string(options.max_width);
    }
  }
  return true;
}

}  // namespace

bool compress_video_h264(const std::filesystem::path& annotated_path,
                         const std::filesystem::path& reference_source_path,
                         const VideoCompressOptions& options, std::string* log_message) {
  return compress_video_h264_impl(annotated_path, reference_source_path, options, log_message);
}

bool write_compression_log(const std::filesystem::path& path, const std::string& message) {
  std::ofstream out(path, std::ios::out | std::ios::app);
  if (!out) return false;
  out << message << '\n';
  return true;
}

namespace {

std::string format_filter_seconds(double seconds) {
  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(3);
  out << std::max(0.0, seconds);
  return out.str();
}

std::string build_hstack_filter(const std::vector<SourceVideoInfo>& infos,
                                const std::vector<double>& offsets_s, double max_duration,
                                const cv::Size& panel_size) {
  std::ostringstream fc;
  std::vector<std::string> labels;
  const bool use_scale = panel_size.width > 0 && panel_size.height > 0;

  for (size_t i = 0; i < infos.size(); ++i) {
    const double offset = i < offsets_s.size() ? offsets_s[i] : 0.0;
    double src_duration = infos[i].duration_s > 0.0 ? infos[i].duration_s : 0.0;
    const std::string out_label = "v" + std::to_string(i);
    std::vector<std::string> filters;

    if (offset > 0.001) {
      filters.push_back("tpad=start_duration=" + format_filter_seconds(offset) +
                        ":start_mode=add:color=black");
    } else if (offset < -0.001) {
      filters.push_back("trim=start=" + format_filter_seconds(std::abs(offset)) +
                        ",setpts=PTS-STARTPTS");
    }
    const double allowed_duration = std::max(0.0, max_duration - std::max(0.0, offset));
    if (allowed_duration > 0.05 && src_duration > allowed_duration + 0.05) {
      filters.push_back("trim=duration=" + format_filter_seconds(allowed_duration) +
                        ",setpts=PTS-STARTPTS");
      src_duration = allowed_duration;
    }
    if (use_scale) {
      filters.push_back("scale=" + std::to_string(panel_size.width) + ":" +
                        std::to_string(panel_size.height) + ":flags=bilinear");
    }
    const double visible_end = std::max(0.0, offset) + src_duration;
    const double stop_pad = max_duration - visible_end;
    if (stop_pad > 0.05) {
      filters.push_back("tpad=stop_mode=add:stop_duration=" + format_filter_seconds(stop_pad) +
                        ":color=black");
    }

    fc << "[" << i << ":v]";
    if (filters.empty()) {
      fc << "format=yuv420p";
    } else {
      for (size_t f = 0; f < filters.size(); ++f) {
        fc << filters[f];
        if (f + 1 < filters.size()) {
          fc << ',';
        }
      }
    }
    fc << '[' << out_label << "];";
    labels.push_back('[' + out_label + ']');
  }

  for (const auto& label : labels) {
    fc << label;
  }
  fc << "hstack=inputs=" << infos.size() << "[vout]";
  return fc.str();
}

void consume_ffmpeg_progress_lines(std::string* tail, const std::string& chunk, double max_duration_s,
                                   VideoStackProgressCallback progress) {
  if (!progress || max_duration_s <= 0.0) {
    return;
  }
  tail->append(chunk);
  size_t pos = 0;
  while ((pos = tail->find('\n')) != std::string::npos) {
    const std::string line = tail->substr(0, pos);
    tail->erase(0, pos + 1);
    if (line.rfind("out_time_ms=", 0) != 0) {
      continue;
    }
    try {
      const double out_ms = std::stod(line.substr(12));
      const double frac =
          std::clamp(out_ms / (max_duration_s * 1'000'000.0), 0.0, 0.999);
      progress(frac, "Laczenie filmow (ffmpeg)");
    } catch (...) {
    }
  }
}

bool stack_videos_with_ffmpeg(const std::vector<std::filesystem::path>& video_paths,
                              const std::vector<double>& offsets_s, double fps,
                              const cv::Size& panel_size, const std::filesystem::path& out_path,
                              std::string* log_message, VideoStackProgressCallback progress,
                              double max_output_duration_s) {
  const auto ffmpeg = find_ffmpeg_executable();
  if (!ffmpeg) {
    return false;
  }

  std::vector<SourceVideoInfo> infos;
  infos.reserve(video_paths.size());
  double max_duration = 0.0;
  for (size_t i = 0; i < video_paths.size(); ++i) {
    const auto info = probe_video_info(video_paths[i]);
    if (!info) {
      if (log_message) {
        *log_message = "Failed to probe video for stacking: " + video_paths[i].string();
      }
      return false;
    }
    infos.push_back(*info);
    const double offset = i < offsets_s.size() ? offsets_s[i] : 0.0;
    const double src_duration = info->duration_s > 0.0 ? info->duration_s : 0.0;
    max_duration = std::max(max_duration, std::max(0.0, offset) + src_duration);
  }
  if (max_output_duration_s > 0.0) {
    max_duration = max_output_duration_s;
  }
  if (max_duration <= 0.0) {
    max_duration = 1.0;
  }

  const std::string filter = build_hstack_filter(infos, offsets_s, max_duration, panel_size);
  if (filter.empty()) {
    return false;
  }

  std::vector<std::wstring> args = {L"-y",         L"-hide_banner", L"-loglevel", L"error",
                                    L"-progress",  L"pipe:1",       L"-nostats",  L"-hwaccel",
                                    L"auto",       L"-threads",     L"0"};
  for (const auto& path : video_paths) {
    args.push_back(L"-i");
    args.push_back(absolute_path(path).wstring());
  }
  args.push_back(L"-filter_complex");
  args.push_back(std::wstring(filter.begin(), filter.end()));
  args.push_back(L"-map");
  args.push_back(L"[vout]");
  args.push_back(L"-an");
  args.push_back(L"-c:v");
  args.push_back(L"libx264");
  args.push_back(L"-pix_fmt");
  args.push_back(L"yuv420p");
  args.push_back(L"-preset");
  args.push_back(L"veryfast");
  args.push_back(L"-movflags");
  args.push_back(L"+faststart");
  args.push_back(L"-crf");
  args.push_back(L"23");
  if (max_output_duration_s > 0.0) {
    args.push_back(L"-t");
    args.push_back(std::to_wstring(max_output_duration_s));
  }
  args.push_back(absolute_path(out_path).wstring());

  std::string progress_tail;
  const auto on_chunk = [&](const std::string& chunk) {
    consume_ffmpeg_progress_lines(&progress_tail, chunk, max_duration, progress);
  };

  const CommandResult result = run_shell_command(*ffmpeg, args, on_chunk);
  if (result.exit_code != 0 || !std::filesystem::exists(out_path)) {
    if (log_message) {
      *log_message = "ffmpeg stack failed (exit " + std::to_string(result.exit_code) + ")";
      if (!result.output.empty()) {
        *log_message += ": " + trim_line(result.output);
      }
    }
    return false;
  }

  if (progress) {
    progress(1.0, "Laczenie filmow zakonczone");
  }
  if (log_message) {
    *log_message = "Stacked " + std::to_string(video_paths.size()) +
                   " videos via ffmpeg (hwaccel decode) -> " + out_path.string();
  }
  return true;
}

bool stack_videos_opencv(const std::vector<std::filesystem::path>& video_paths,
                         const std::vector<double>& offsets_s, double fps,
                         const cv::Size& panel_size, const std::filesystem::path& out_path,
                         std::string* log_message, VideoStackProgressCallback progress,
                         double max_output_duration_s) {
  if (video_paths.empty()) return false;

  struct InputVideo {
    cv::VideoCapture cap;
    double fps{25.0};
    int frame_count{0};
    double offset_s{0.0};
    int current_src_frame{-1};
    cv::Mat current_panel;
  };

  std::vector<InputVideo> inputs;
  inputs.reserve(video_paths.size());
  double master_fps = fps > 1.0 ? fps : 25.0;

  for (size_t i = 0; i < video_paths.size(); ++i) {
    InputVideo input;
    input.cap.open(video_paths[i].string());
    if (!input.cap.isOpened()) {
      if (log_message) *log_message = "Failed to open video for stacking: " + video_paths[i].string();
      return false;
    }
    input.fps = input.cap.get(cv::CAP_PROP_FPS);
    if (input.fps <= 1.0) input.fps = master_fps;
    input.frame_count = static_cast<int>(input.cap.get(cv::CAP_PROP_FRAME_COUNT));
    input.offset_s = i < offsets_s.size() ? offsets_s[i] : 0.0;
    if (i == 0 && fps <= 1.0) master_fps = input.fps;
    inputs.push_back(std::move(input));
  }

  double max_duration = 0.0;
  for (const auto& input : inputs) {
    const double dur = input.offset_s + input.frame_count / std::max(1.0, input.fps);
    max_duration = std::max(max_duration, dur);
  }
  if (max_duration <= 0.0) max_duration = 1.0;
  if (max_output_duration_s > 0.0) {
    max_duration = max_output_duration_s;
  }

  const cv::Size use_panel =
      panel_size.width > 0 && panel_size.height > 0 ? panel_size : cv::Size(640, 360);
  cv::VideoWriter writer;
  const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  writer.open(out_path.string(), fourcc, master_fps,
              {use_panel.width * static_cast<int>(inputs.size()), use_panel.height});
  if (!writer.isOpened()) {
    if (log_message) *log_message = "Failed to open stacked video writer: " + out_path.string();
    return false;
  }

  cv::Mat black(use_panel, CV_8UC3, cv::Scalar(0, 0, 0));
  const int total_frames = static_cast<int>(std::ceil(max_duration * master_fps));

  for (int out_frame = 0; out_frame < total_frames; ++out_frame) {
    const double t_s = out_frame / master_fps;
    std::vector<cv::Mat> panels;
    panels.reserve(inputs.size());

    for (auto& input : inputs) {
      const double local_t = t_s - input.offset_s;
      if (local_t < 0.0) {
        panels.push_back(black.clone());
        continue;
      }
      const int src_frame = static_cast<int>(local_t * input.fps);
      if (src_frame >= input.frame_count) {
        panels.push_back(black.clone());
        continue;
      }
      if (src_frame == input.current_src_frame && !input.current_panel.empty()) {
        panels.push_back(input.current_panel);
        continue;
      }

      cv::Mat frame;
      if (src_frame == input.current_src_frame + 1) {
        if (!input.cap.read(frame) || frame.empty()) {
          panels.push_back(black.clone());
          continue;
        }
      } else {
        input.cap.set(cv::CAP_PROP_POS_FRAMES, src_frame);
        if (!input.cap.read(frame) || frame.empty()) {
          panels.push_back(black.clone());
          continue;
        }
      }

      cv::Mat panel;
      if (frame.size() != use_panel) {
        cv::resize(frame, panel, use_panel, 0, 0, cv::INTER_AREA);
      } else {
        panel = frame;
      }
      input.current_src_frame = src_frame;
      input.current_panel = panel;
      panels.push_back(input.current_panel);
    }

    cv::Mat combined;
    cv::hconcat(panels, combined);
    writer.write(combined);

    if (progress && total_frames > 0 &&
        (out_frame == 0 || out_frame == total_frames - 1 || out_frame % 5 == 0)) {
      progress(static_cast<double>(out_frame + 1) / total_frames,
               "Laczenie filmow " + std::to_string(out_frame + 1) + "/" +
                   std::to_string(total_frames));
    }
  }

  writer.release();
  if (log_message) {
    *log_message = "Stacked " + std::to_string(inputs.size()) + " videos (OpenCV fallback) -> " +
                   out_path.string();
  }
  return true;
}

}  // namespace

bool stack_videos_side_by_side(const std::vector<std::filesystem::path>& video_paths,
                               const std::vector<double>& offsets_s, double fps,
                               const cv::Size& panel_size, const std::filesystem::path& out_path,
                               std::string* log_message, VideoStackProgressCallback progress,
                               double max_output_duration_s) {
  if (stack_videos_with_ffmpeg(video_paths, offsets_s, fps, panel_size, out_path, log_message,
                               progress, max_output_duration_s)) {
    return true;
  }
  return stack_videos_opencv(video_paths, offsets_s, fps, panel_size, out_path, log_message,
                             progress, max_output_duration_s);
}

#include "path_utils.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

std::filesystem::path path_if_exists(const std::filesystem::path& path) {
  if (!path.empty() && std::filesystem::exists(path)) {
    return path;
  }
  return {};
}

}  // namespace

std::filesystem::path application_directory() {
#ifdef _WIN32
  char buffer[MAX_PATH]{};
  const DWORD len = GetModuleFileNameA(nullptr, buffer, MAX_PATH);
  if (len > 0 && len < MAX_PATH) {
    return std::filesystem::path(buffer).parent_path();
  }
#endif
  return std::filesystem::current_path();
}

std::filesystem::path resolve_resource_path(const std::filesystem::path& relative_path) {
  if (relative_path.empty()) {
    return relative_path;
  }

  if (relative_path.is_absolute()) {
    return relative_path;
  }

  if (const auto cwd_path = path_if_exists(relative_path); !cwd_path.empty()) {
    return cwd_path;
  }

  const std::filesystem::path exe_dir = application_directory();
  if (const auto exe_path = path_if_exists(exe_dir / relative_path); !exe_path.empty()) {
    return exe_path;
  }

  std::filesystem::path dir = exe_dir;
  for (int i = 0; i < 6; ++i) {
    const auto candidate = dir / relative_path;
    if (const auto found = path_if_exists(candidate); !found.empty()) {
      return found;
    }
    if (!dir.has_parent_path() || dir.parent_path() == dir) {
      break;
    }
    dir = dir.parent_path();
  }

  return relative_path;
}

std::string resolve_resource_path_string(const std::string& relative_path) {
  return resolve_resource_path(std::filesystem::path(relative_path)).string();
}

#pragma once

#include <filesystem>
#include <string>

std::filesystem::path application_directory();
std::filesystem::path resolve_resource_path(const std::filesystem::path& relative_path);
std::string resolve_resource_path_string(const std::string& relative_path);

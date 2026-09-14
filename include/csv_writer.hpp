#pragma once

#include <filesystem>
#include <fstream>

#include "types.hpp"

class CsvWriter {
 public:
  bool open(const std::filesystem::path& path, bool use_meters);
  void write(const Sample& sample);
  bool is_open() const;

 private:
  std::ofstream out_;
  bool use_meters_{false};
};

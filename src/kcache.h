// The kernel cache's folders and housekeeping, shared by the two engines (zero/clrt.h, src/clrt.h). Audit 2026-11-12 #3:
// the cache had no limit and no prune (1.4 GB, 2,786 files on the laptop), every driver update left all its files
// behind (the driver version is in each file's key), and two processes compiling the same kernel at once wrote one
// shared ".tmp" name.
//  - kernels\drv-<driver version>\: one folder per driver, so another driver's files go as a folder. The file names
//    (the hash of source, options, device and driver) are as before: both engines still share a driver's files.
//  - kcache_sweep, once per process at start: the old flat files of kernels\ removed; another driver's folder once
//    nothing in it was used for 7 days (two GPUs on two drivers do not wipe each other's); temporary files older
//    than an hour; and this driver's folder trimmed to ARC_KCACHE_GB (default 2 GB) by last use, oldest first, to
//    3/4 of it. A cache hit refreshes its file's time (kcache_touch).
//  - kcache_tmp: a temporary name of this process's own, renamed over the final name when complete.
#pragma once
#include "platform.h"  // (plat::process_id, PLAT_SEP)
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

inline std::string kcache_dir(const std::string& base, const std::string& driver) {
  std::string tag;
  for (char c : driver) tag.push_back((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '.' || c == '-' ? c : '_');
  if (tag.empty()) tag = "unknown";
  if (tag.size() > 64) tag.resize(64);
  return base + PLAT_SEP "drv-" + tag;
}

inline std::string kcache_tmp(const std::string& path) {
  static std::atomic<unsigned> seq{0};
  return path + ".tmp." + std::to_string(plat::process_id()) + "." + std::to_string(seq++);
}

inline void kcache_touch(const std::string& path) {
  std::error_code ec;
  std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), ec);
}

inline void kcache_sweep(const std::string& base, const std::string& cur) {
  static std::atomic<bool> done{false};
  if (done.exchange(true)) return;
  namespace fs = std::filesystem;
  std::error_code ec;
  const auto now = fs::file_time_type::clock::now();
  auto old = [&](const fs::file_time_type& t, std::chrono::hours h) { return now - t > h; };
  auto is_tmp = [](const fs::path& p) { return p.filename().string().find(".tmp") != std::string::npos; };
  for (fs::directory_iterator it(base, ec), end; !ec && it != end; it.increment(ec)) {
    const fs::path p = it->path();
    std::error_code e2;
    if (it->is_directory(e2)) {
      const std::string name = p.filename().string();
      if (name.rfind("drv-", 0) != 0 || p == fs::path(cur)) continue;
      fs::file_time_type newest{};
      for (fs::directory_iterator jt(p, e2), jend; !e2 && jt != jend; jt.increment(e2)) {
        std::error_code e3;
        const auto t = jt->last_write_time(e3);
        if (!e3 && t > newest) newest = t;
      }
      if (!e2 && old(newest, std::chrono::hours(24 * 7))) fs::remove_all(p, e2);
      continue;
    }
    const std::string ext = p.extension().string();
    // (the flat cache of before 2026-11-12, its binaries and the server's warm-up markers: no engine reads them now)
    if (ext == ".bin" || is_tmp(p) || p.filename().string().rfind("warm-", 0) == 0) fs::remove(p, e2);
  }
  struct F { fs::path p; fs::file_time_type t; uintmax_t n; };
  std::vector<F> files;
  uintmax_t total = 0;
  for (fs::directory_iterator it(cur, ec), end; !ec && it != end; it.increment(ec)) {
    std::error_code e2;
    if (!it->is_regular_file(e2)) continue;
    const auto t = it->last_write_time(e2);
    if (is_tmp(it->path())) {
      if (!e2 && old(t, std::chrono::hours(1))) fs::remove(it->path(), e2);
      continue;
    }
    if (it->path().filename().string().rfind("warm-", 0) == 0) continue;  // (warm-up markers: empty, kept)
    const uintmax_t n = it->file_size(e2);
    if (e2) continue;
    files.push_back({it->path(), t, n});
    total += n;
  }
  const char* g = getenv("ARC_KCACHE_GB");
  const double gb = g ? atof(g) : 2.0;
  const uintmax_t cap = (uintmax_t)((std::max)(0.25, gb) * 1073741824.0);
  if (total <= cap) return;
  std::sort(files.begin(), files.end(), [](const F& a, const F& b) { return a.t < b.t; });
  for (const F& f : files) {
    if (total <= cap / 4 * 3) break;
    std::error_code e2;
    if (fs::remove(f.p, e2)) total -= f.n;
  }
}

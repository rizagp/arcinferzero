// The data folder: %USERPROFILE%\arcinferzero (kernel cache, model packages and expert files, context / session /
// prefix caches, logs, model-dirs.txt, transcripts, recordings), or ARC_DATA_DIR when set. Until 2026-10-25 it was
// %LOCALAPPDATA%\arcinferzero (and until 2026-10-24 %LOCALAPPDATA%\arcinfer): when only an old one exists it is moved
// once (the same drive: a rename, instant, the packages are not copied) and the paths written in its small state files
// (last-model.txt, launcher.json, ctx-auto.txt) follow it. If the move fails (a file held open by an older process) the
// old folder is used as it is. scripts/app-dir.ps1 does the same for the scripts.
#pragma once
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

inline std::string app_dir() {
  static const std::string dir = [] {
    namespace fs = std::filesystem;
    if (const char* d = getenv("ARC_DATA_DIR"); d && *d) return std::string(d);
    const char* up = getenv("USERPROFILE");
    const char* la = getenv("LOCALAPPDATA");
    const fs::path now = fs::path(up ? up : ".") / "arcinferzero";
    std::error_code ec;
    if (fs::exists(now, ec) || !la) return now.string();
    for (const char* name : {"arcinferzero", "arcinfer"}) {
      const fs::path old = fs::path(la) / name;
      if (!fs::is_directory(old, ec)) continue;
      fs::rename(old, now, ec);
      if (ec) return old.string();
      // (the full paths some state files hold: a model package's path in last-model.txt, launcher.json, ctx-auto.txt)
      for (const char* f : {"last-model.txt", "launcher.json", "ctx-auto.txt"}) {
        const fs::path p = now / f;
        std::ifstream in(p, std::ios::binary);
        if (!in) continue;
        std::stringstream ss;
        ss << in.rdbuf();
        in.close();
        // (as written: backslashes, the folder joined with '/', and JSON's doubled backslashes)
        auto dbl = [](std::string x) {
          for (size_t i = 0; (i = x.find('\\', i)) != std::string::npos; i += 2) x.insert(i, 1, '\\');
          return x;
        };
        const std::string o = old.string(), n = now.string(), om = std::string(la) + "/" + name;
        const std::pair<std::string, std::string> subs[] = {{dbl(o), dbl(n)}, {o, n}, {om, n}};
        std::string s = ss.str();
        bool changed = false;
        for (const auto& [from, to] : subs)
          for (size_t i = 0; (i = s.find(from, i)) != std::string::npos; i += to.size()) {
            s.replace(i, from.size(), to);
            changed = true;
          }
        if (changed) std::ofstream(p, std::ios::binary) << s;
      }
      return now.string();
    }
    return now.string();
  }();
  return dir;
}

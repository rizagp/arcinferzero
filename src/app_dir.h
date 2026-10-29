// The data folder: %USERPROFILE%\.arcinferzero (kernel cache, model packages and expert files, context / session /
// prefix caches, logs, model-dirs.txt, transcripts, recordings), or ARC_DATA_DIR when set: a dot folder, as the other
// tools' folders in the user's profile (.cache, .pi, .vscode). Until 2026-10-29 it was %USERPROFILE%\arcinferzero,
// until 2026-10-25 %LOCALAPPDATA%\arcinferzero (and until 2026-10-24 %LOCALAPPDATA%\arcinfer): when only an old one
// exists it is moved once (the same drive: a rename, instant, the packages are not copied) and the paths written in its
// small state files (last-model.txt, launcher.json, server.json, zero\ctx-auto.txt, which holds them in lower case)
// follow it. If the move fails (a file held open by an older process) the old folder is used as it is.
// scripts/app-dir.ps1 does the same for the scripts.
#pragma once
#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

inline std::string app_dir() {
  static const std::string dir = [] {
    namespace fs = std::filesystem;
    if (const char* d = getenv("ARC_DATA_DIR"); d && *d) return std::string(d);
    const char* up = getenv("USERPROFILE");
    const char* la = getenv("LOCALAPPDATA");
    const fs::path home = fs::path(up ? up : ".");
    const fs::path now = home / ".arcinferzero";
    std::error_code ec;
    if (fs::exists(now, ec)) return now.string();
    // (the older places, newest first, with the text a state file may hold for them)
    std::vector<std::pair<fs::path, std::string>> olds = {{home / "arcinferzero", (home / "arcinferzero").string()}};
    if (la) {
      olds.push_back({fs::path(la) / "arcinferzero", std::string(la) + "/arcinferzero"});
      olds.push_back({fs::path(la) / "arcinfer", std::string(la) + "/arcinfer"});
    }
    for (const auto& [old, mixed] : olds) {
      if (!fs::is_directory(old, ec)) continue;
      fs::rename(old, now, ec);
      if (ec) return old.string();
      // (the full paths some state files hold: a model package's path in last-model.txt, launcher.json, server.json,
      //  the context cache's keys in zero\ctx-auto.txt; matched in any case, as Windows paths are)
      for (const char* f : {"last-model.txt", "launcher.json", "server.json", "ctx-auto.txt", "zero/ctx-auto.txt"}) {
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
        auto low = [](std::string x) {
          for (char& c : x) c = (char)std::tolower((unsigned char)c);
          return x;
        };
        const std::string o = old.string(), n = now.string();
        const bool lower_file = std::string(f).find("ctx-auto") != std::string::npos;
        const std::pair<std::string, std::string> subs[] = {{dbl(o), dbl(n)}, {o, n}, {mixed, n}};
        std::string s = ss.str();
        bool changed = false;
        for (const auto& [from, to0] : subs) {
          const std::string lf = low(from), to = lower_file ? low(to0) : to0;
          std::string ls = low(s);
          for (size_t i = 0; (i = ls.find(lf, i)) != std::string::npos;) {
            // (a whole name: "...\arcinfer" is not the start of "...\arcinferzero")
            const size_t e = i + lf.size();
            if (e < ls.size() && (std::isalnum((unsigned char)ls[e]) || ls[e] == '_' || ls[e] == '-')) { i = e; continue; }
            s.replace(i, from.size(), to);
            ls.replace(i, from.size(), low(to));
            i += to.size();
            changed = true;
          }
        }
        if (changed) std::ofstream(p, std::ios::binary) << s;
      }
      return now.string();
    }
    return now.string();
  }();
  return dir;
}

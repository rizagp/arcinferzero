// The web UI's memory (QUEUE F32.2 / F32.4): what the user asked the model to remember, as Markdown files in the data
// folder (src/app_dir.h), one dated fact per line, readable and editable by hand:
//   MEMORY.md                         the global memory: every chat and model (a plain chat reads only this one)
//   projects\<project id>\MEMORY.md   a project's own memory (server_projects.h), kept here and not in the project's
//                                     folder, so the model's notes are never committed or shared with that folder
// The web UI puts a chat's memory into its system text once, when the chat starts (a copy kept with the chat: a
// memory that changes later does not make the server process that chat's prompt again). The model adds and removes
// lines with the tools remember / forget, when the user asks; the user edits the whole text in Settings.
// Model-independent: served while a model loads or generates. Only for the server's own page or a client without an
// Origin (src/http.h foreign_ok), on this computer (not over the network).
#pragma once
#include "../src/app_dir.h"
#include "../src/json.h"
#include "server_files.h"
#include <ctime>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace memo {
namespace fs = std::filesystem;

// (~2,000 tokens: the memory is in every chat's prompt; a model on a small context keeps room for the conversation.
//  At the cap, remember says to merge old notes first. The whole text set by hand may be longer)
constexpr size_t kCap = 8000, kHandCap = 64000;

inline std::mutex& mu() {
  static std::mutex m;
  return m;
}
// the file of a scope: "" the global memory, else a project id (16 hex digits, checked by the caller)
inline fs::path path_of(const std::string& project) {
  const fs::path dir = fs::path(app_dir());  // (the exes run with the UTF-8 code page: a std::string path is UTF-8)
  return project.empty() ? dir / "MEMORY.md" : dir / "projects" / project / "MEMORY.md";
}
inline bool valid_project(const std::string& id) {
  if (id.size() != 16) return false;
  for (char c : id)
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  return true;
}
inline std::string read(const std::string& project) {
  std::string s;
  files::read_shared(path_of(project), s);
  return s;
}
inline bool write(const std::string& project, const std::string& text) {
  const fs::path p = path_of(project);
  std::error_code ec;
  fs::create_directories(p.parent_path(), ec);
  return files::write_shared(p, text);
}
inline std::string today() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  localtime_s(&tm, &t);
  char b[16];
  std::strftime(b, sizeof b, "%Y-%m-%d", &tm);
  return b;
}
// one line, as the model wrote it: no line breaks, no leading list marker (the file adds its own)
inline std::string one_line(std::string s) {
  for (char& c : s)
    if (c == '\r' || c == '\n' || c == '\t') c = ' ';
  size_t a = s.find_first_not_of(" -*"), b = s.find_last_not_of(' ');
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
inline std::string lower(std::string s) {
  for (char& c : s) c = (char)tolower((unsigned char)c);
  return s;
}

// remember: "- (date) fact" added at the end; refused when the memory is at its cap or the fact is already there
inline Json remember(const std::string& project, const std::string& fact_in) {
  std::lock_guard<std::mutex> g(mu());
  Json r = Json::object();
  const std::string fact = one_line(fact_in);
  if (fact.empty()) throw std::runtime_error("fact required: one short sentence to remember");
  if (fact.size() > 600) throw std::runtime_error("too long for one memory line (600 characters at most): say it shorter");
  std::string text = read(project);
  if (lower(text).find(lower(fact)) != std::string::npos) {
    r.set("ok", true);
    r.set("note", "already remembered");
    return r;
  }
  const std::string line = "- (" + today() + ") " + fact + "\n";
  if (text.size() + line.size() > kCap)
    throw std::runtime_error("the memory is full (" + std::to_string(text.size()) + " characters): forget or merge old "
                             "notes first (forget with a part of their text, then remember the merged note)");
  if (!text.empty() && text.back() != '\n') text += "\n";
  text += line;
  if (!write(project, text)) throw std::runtime_error("cannot write " + path_of(project).string());
  r.set("ok", true);
  r.set("remembered", line.substr(2, line.size() - 3));
  r.set("scope", project.empty() ? "global" : "project");
  return r;
}
// forget: every line that contains `match` (case-insensitive) removed; none found is an error the model can act on
inline Json forget(const std::string& project, const std::string& match_in) {
  std::lock_guard<std::mutex> g(mu());
  const std::string match = lower(one_line(match_in));
  if (match.size() < 3) throw std::runtime_error("match required: a part of the note's text (3 characters or more)");
  const std::string text = read(project);
  std::string kept;
  Json gone = Json::array();
  size_t at = 0;
  while (at < text.size()) {
    size_t e = text.find('\n', at);
    if (e == std::string::npos) e = text.size();
    const std::string line = text.substr(at, e - at);
    if (lower(line).find(match) != std::string::npos) gone.push(Json(line));
    else kept += line + "\n";
    at = e + 1;
  }
  if (!gone.size()) throw std::runtime_error("no note contains \"" + match_in + "\"");
  // (a short match such as "2026" would take every note: more than 3 at once only by hand, in Settings)
  if (gone.size() > 3)
    throw std::runtime_error(std::to_string(gone.size()) + " notes contain \"" + match_in + "\": nothing removed. Use a "
                             "longer part of the one note to forget (the user can remove many in Settings > Memory)");
  if (!write(project, kept)) throw std::runtime_error("cannot write " + path_of(project).string());
  Json r = Json::object();
  r.set("ok", true);
  r.set("forgotten", gone);
  return r;
}

// /v1/memory?project=<id>: GET {text, path, chars, cap}; PUT {text} sets the whole text (the Settings panel)
inline void api(const HttpRequest& rq, HttpResponse& rs) {
  auto fail = [&](int code, const std::string& m) {
    Json e = Json::object(), r = Json::object();
    e.set("message", m);
    r.set("error", e);
    rs.send(code, "application/json", r.dump());
  };
  std::string project;
  {
    const std::string q = "&" + rq.query + "&";
    const size_t p = q.find("&project=");
    if (p != std::string::npos) project = q.substr(p + 9, q.find('&', p + 9) - p - 9);
  }
  if (!project.empty() && !valid_project(project)) return fail(400, "project: 16 hex digits");
  // (this computer only: the memory goes into every chat's prompt, not to be read or set over the network)
  if (!rq.local_peer) return fail(403, "the memory is only for this computer (the web UI on 127.0.0.1)");
  try {
    if (rq.method == "GET") {
      Json r = Json::object();
      std::lock_guard<std::mutex> g(mu());
      const std::string text = read(project);
      r.set("text", text);
      r.set("path", path_of(project).string());
      r.set("chars", (double)text.size());
      r.set("cap", (double)kCap);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.method == "PUT") {
      const Json j = Json::parse(rq.body);
      const std::string text = j["text"].as_string();
      if (text.size() > kHandCap) return fail(400, "the memory is " + std::to_string(text.size()) + " characters: at most " + std::to_string(kHandCap));
      std::lock_guard<std::mutex> g(mu());
      if (!write(project, text)) return fail(500, "cannot write " + path_of(project).string());
      Json r = Json::object();
      r.set("ok", true);
      r.set("chars", (double)text.size());
      return rs.send(200, "application/json", r.dump());
    }
    return fail(405, "GET or PUT");
  } catch (const std::exception& e) {
    return fail(400, e.what());
  }
}
}  // namespace memo

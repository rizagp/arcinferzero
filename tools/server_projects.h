// The web UI's projects (QUEUE F32.3 / F32.4), as Claude's Projects: a folder of the user's own (a Google Drive
// folder through Drive for desktop works too) that a project's chats work in, with the project's instructions (its
// AGENTS.md, or CLAUDE.md) and its memory (tools/server_memory.h: projects\<id>\MEMORY.md in the data folder).
//   projects.json in the data folder   [{id, name, folder, created}]: the folders the user picked, and only those
// A project's id is 16 hex digits of the FNV-1a hash of its folder's path in lower case. Its files reach the model as
// "project:<id>/<path>" (read_file, search_file, digest_file; list_files with scope "project"); resolve() takes only
// a path inside the folder, after every link and junction is followed. The model writes into the chat's own folder
// (create_file) and into the project's only through the terminal, which asks the user. Only for the server's own page
// or a client without an Origin (src/http.h foreign_ok), and on this computer only (not over the network); a chat
// reads only its own project's files (server.cpp: the call's project_id).
#pragma once
#include "../src/app_dir.h"
#include "../src/json.h"
#include "server_files.h"
#include "server_memory.h"
#include <ctime>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace proj {
namespace fs = std::filesystem;

struct Project {
  std::string id, name, folder;
  int64_t created = 0;
};
inline std::mutex& mu() {
  static std::mutex m;
  return m;
}
inline fs::path list_path() { return fs::path(app_dir()) / "projects.json"; }
inline std::vector<Project> load() {
  std::vector<Project> out;
  std::string s;
  Json j;
  if (!files::read_shared(list_path(), s) || !Json::try_parse(s, j) || !j.is_array()) return out;
  for (const Json& x : j.items()) {
    Project p;
    p.id = x["id"].str();
    p.name = x["name"].str();
    p.folder = x["folder"].str();
    p.created = (int64_t)x["created"].as_number(0);
    if (memo::valid_project(p.id) && !p.folder.empty()) out.push_back(p);
  }
  return out;
}
inline bool save(const std::vector<Project>& ps) {
  Json j = Json::array();
  for (const Project& p : ps) {
    Json x = Json::object();
    x.set("id", p.id);
    x.set("name", p.name);
    x.set("folder", p.folder);
    x.set("created", (double)p.created);
    j.push(x);
  }
  std::error_code ec;
  fs::create_directories(list_path().parent_path(), ec);
  return files::write_shared(list_path(), j.dump());
}
inline std::string id_of(const fs::path& folder) {
#ifdef _WIN32
  std::wstring w = folder.wstring();
  for (wchar_t& c : w) c = (wchar_t)towlower(c);
  const std::string s = fs::path(w).string();
#else
  const std::string s = folder.lexically_normal().string();  // (case-sensitive names, their bytes as they are)
#endif
  uint64_t h = 1469598103934665603ull;
  for (unsigned char b : s) h = (h ^ b) * 1099511628211ull;
  char b[17];
  snprintf(b, sizeof b, "%016llx", (unsigned long long)h);
  return b;
}
inline bool find(const std::string& id, Project& out) {
  for (const Project& p : load())
    if (p.id == id) {
      out = p;
      return true;
    }
  return false;
}
// a folder the user may pick: an existing folder, not the data folder or inside it (the chats, caches and memory),
// not Windows' or the programs' folders
inline std::string refuse(const fs::path& f) {
  std::error_code ec;
  if (!f.is_absolute()) return "not a full path";
  if (!fs::is_directory(f, ec)) return "no such folder";
  auto under = [&](const fs::path& a, const fs::path& b) {  // (a inside b, or b itself; any case)
#ifndef _WIN32
    return plat::path_inside(a, b, true);  // (Linux: '/', case-sensitive)
#else
    std::wstring x = a.wstring(), y = b.wstring();
    for (wchar_t& c : x) c = (wchar_t)towlower(c);
    for (wchar_t& c : y) c = (wchar_t)towlower(c);
    while (!y.empty() && (y.back() == L'\\' || y.back() == L'/')) y.pop_back();
    return !y.empty() && x.compare(0, y.size(), y) == 0 && (x.size() == y.size() || x[y.size()] == L'\\');
#endif
  };
  const fs::path data = fs::weakly_canonical(fs::path(app_dir()), ec);
  if (under(f, data) || under(data, f)) return "the inferzero data folder (its chats, caches and memory) cannot be a project folder, nor a folder that holds it";
  for (const char* v : {"SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramData"})
    if (const char* d = getenv(v); d && *d && under(f, fs::path(d))) return std::string("a system folder (") + d + ")";
  if (under(f, f.root_path() / "Windows")) return "a Windows folder";
#ifndef _WIN32
  // (Linux: the system's own folders, and the home folder with its settings and keys, as the profile below on Windows)
  for (const char* d : {"/etc", "/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/boot", "/dev", "/proc", "/sys", "/run",
                        "/var/lib", "/var/log", "/snap"})
    if (under(f, fs::path(d))) return std::string("a system folder (") + d + ")";
  if (const char* up = plat::home_env(); up && *up) {
    const fs::path u = fs::weakly_canonical(fs::path(up), ec);
    if (under(u, f)) return "your whole home folder (or a folder that holds it): pick a folder in it, e.g. ~/Documents";
    for (const char* sub : {".ssh", ".gnupg", ".aws", ".azure", ".config", ".kube", ".docker", ".local/share/keyrings", ".pki"})
      if (under(f, u / sub)) return std::string("a settings or keys folder (") + sub + ")";
  }
#endif
  if (const char* up = getenv("USERPROFILE"); up && *up) {  // (the profile itself, and its settings and keys)
    const fs::path u = fs::weakly_canonical(fs::path(up), ec);
    if (under(u, f)) return "your whole user folder (or a folder that holds it): pick a folder in it, e.g. Documents";
    const char* tmp = getenv("TEMP");  // (AppData\Local\Temp: a work folder, not settings)
    const bool in_temp = tmp && *tmp && under(f, fs::weakly_canonical(fs::path(tmp), ec)) && !under(fs::weakly_canonical(fs::path(tmp), ec), f);
    for (const char* sub : {"AppData", ".ssh", ".gnupg", ".aws", ".azure", ".config"})
      if (under(f, u / sub) && !(in_temp && std::string(sub) == "AppData")) return std::string("a settings or keys folder (") + sub + ")";
  }
  if (f == f.root_path()) return "a whole drive: pick a folder on it";
  return "";
}
// "project:<id>/<path>" -> the file inside the project's folder (links and junctions followed); "" and err set when
// it is not a project file id, the project is unknown, or the path leaves the folder
// (the id's project given: p known, so a search over many files reads projects.json once)
inline bool split_id(const std::string& fid, std::string& id, std::string& rel) {
  if (fid.rfind("project:", 0) != 0) return false;
  const size_t sl = fid.find('/', 8);
  id = fid.substr(8, sl == std::string::npos ? std::string::npos : sl - 8);
  rel = sl == std::string::npos ? "" : fid.substr(sl + 1);
  return true;
}
inline bool resolve_in(const Project& p, std::string& rel, fs::path& file, std::string& err) {
  for (char& c : rel)
    if (c == '\\') c = '/';
  bool up = false;  // (a ".." part of the path; "v1..2.txt" is a name)
  for (size_t s = 0; s <= rel.size();) {
    const size_t e = std::min(rel.find('/', s), rel.size());
    up = up || rel.compare(s, e - s, "..") == 0;
    s = e + 1;
  }
  if (rel.empty() || rel.find(':') != std::string::npos || rel[0] == '/' || up) {
    err = "a path inside the project's folder, e.g. project:" + p.id + "/docs/plan.md";
    return false;
  }
  std::error_code ec;
  const fs::path root = fs::weakly_canonical(fs::path(p.folder), ec);
  if (ec || root.empty() || !root.is_absolute()) {
    err = "the project's folder is not there: " + p.folder;
    return false;
  }
  const fs::path f = fs::weakly_canonical(root / fs::path(rel), ec);
#ifdef _WIN32
  std::wstring r = root.wstring(), x = f.wstring();
  for (wchar_t& c : r) c = (wchar_t)towlower(c);
  for (wchar_t& c : x) c = (wchar_t)towlower(c);
  if (ec || x.compare(0, r.size(), r) != 0 || x.size() <= r.size() || x[r.size()] != L'\\') {
#else
  if (ec || !plat::path_inside(f, root, false)) {
#endif
    err = "outside the project's folder: " + rel;
    return false;
  }
  if (!fs::is_regular_file(f, ec)) {
    err = "no file " + rel + " in the project (list_files with scope \"project\" lists them)";
    return false;
  }
  file = f;
  return true;
}
inline bool resolve(const std::string& fid, Project& p, fs::path& file, std::string& rel, std::string& err) {
  std::string id;
  if (!split_id(fid, id, rel)) return false;
  if (!memo::valid_project(id) || !find(id, p)) {
    err = "no project " + id;
    return false;
  }
  return resolve_in(p, rel, file, err);
}
// the project's files, newest first: relative paths with '/', skipping hidden folders (.git), dependencies and caches;
// at most `limit` (more: true when there were more)
struct Entry {
  std::string rel;
  uint64_t bytes = 0;
  int64_t modified = 0;
};
inline std::vector<Entry> walk(const Project& p, size_t limit, bool& more) {
  std::vector<Entry> out;
  more = false;
  std::error_code ec;
  const fs::path root = fs::path(p.folder);
  fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
  static const char* const skip[] = {"node_modules", "__pycache__", "venv", ".venv", "bin", "obj", "dist", "build", "target"};
  size_t seen = 0;
  for (; it != end && !ec; it.increment(ec)) {
    const std::string name = it->path().filename().string();
    // (a junction or link: not followed, its files may be outside the folder; resolve() refuses them too)
    const bool reparse = plat::is_link(it->path());
    if (it->is_directory(ec)) {
      bool drop = reparse || (!name.empty() && name[0] == '.');
      for (const char* s : skip) drop = drop || files::lower(name) == s;
      if (drop || it.depth() >= 8) it.disable_recursion_pending();
      continue;
    }
    if (reparse || !it->is_regular_file(ec) || (!name.empty() && name[0] == '.')) continue;
    if (++seen > 20000) break;  // (a huge tree: the first part of it)
    Entry e;
    e.rel = fs::relative(it->path(), root, ec).generic_string();
    e.bytes = it->file_size(ec);
    const auto t = it->last_write_time(ec);
#ifdef _WIN32
    e.modified = (int64_t)std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count() - 11644473600LL;
#else
    // (unix seconds through clock_cast, as server_files.h: libstdc++'s file clock does not count from 1601 as MSVC's,
    //  and the subtraction above gave dates around the year 1453)
    e.modified = (int64_t)std::chrono::duration_cast<std::chrono::seconds>(
                     std::chrono::clock_cast<std::chrono::system_clock>(t).time_since_epoch()).count();
#endif
    out.push_back(e);
  }
  std::sort(out.begin(), out.end(), [](const Entry& a, const Entry& b) { return a.modified > b.modified; });
  if (out.size() > limit) {
    out.resize(limit);
    more = true;
  }
  return out;
}
// the project's instructions: AGENTS.md (else CLAUDE.md) in its folder, at most 16,000 characters
inline std::string instructions(const Project& p, std::string& from) {
  for (const char* n : {"AGENTS.md", "CLAUDE.md"}) {
    std::string s;
    if (files::read_shared(fs::path(p.folder) / n, s)) {
      from = n;
      if (s.size() > 16000) s = s.substr(0, 16000) + "\n[... cut at 16,000 characters]";
      return s;
    }
  }
  return "";
}
inline Json to_json(const Project& p, bool with_text) {
  Json j = Json::object();
  j.set("id", p.id);
  j.set("name", p.name);
  j.set("folder", p.folder);
  j.set("created", (double)p.created);
  std::error_code ec;
  j.set("exists", fs::is_directory(fs::path(p.folder), ec));
  if (with_text) {
    std::string from;
    j.set("instructions", instructions(p, from));
    j.set("instructions_file", from);
    j.set("memory", memo::read(p.id));
  }
  return j;
}

// /v1/projects: GET list; POST {folder, name} adds one (the same folder: that project back); /v1/projects/<id>: GET
// with its instructions and memory, DELETE drops it from the list (its folder and memory stay); /v1/projects/browse?
// path=<folder>: the folders in it (no path: the drives and the user's own folders), this computer only
inline void api(const HttpRequest& rq, HttpResponse& rs) {
  auto fail = [&](int code, const std::string& m) {
    Json e = Json::object(), r = Json::object();
    e.set("message", m);
    r.set("error", e);
    rs.send(code, "application/json", r.dump());
  };
  auto query = [&](const std::string& k) {  // (a %-encoded query value)
    const std::string q = "&" + rq.query + "&";
    const size_t p = q.find("&" + k + "=");
    if (p == std::string::npos) return std::string();
    const std::string v = q.substr(p + k.size() + 2, q.find('&', p + k.size() + 2) - p - k.size() - 2);
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
      if (v[i] == '%' && i + 2 < v.size() && isxdigit((unsigned char)v[i + 1]) && isxdigit((unsigned char)v[i + 2])) {
        out += (char)std::stoi(v.substr(i + 1, 2), nullptr, 16);
        i += 2;
      } else out += v[i] == '+' ? ' ' : v[i];
    }
    return out;
  };
  // (this computer only: a project's folder, files, instructions and memory are not for the network, --host 0.0.0.0)
  if (!rq.local_peer) return fail(403, "projects are only for this computer (the web UI on 127.0.0.1)");
  try {
    if (rq.path == "/v1/projects/browse") {
      const std::string path = query("path");
      Json d = Json::array();
      std::error_code ec;
      if (path.empty()) {  // (the drives, and the user's own folders)
#ifdef _WIN32
        const DWORD drives = GetLogicalDrives();
        for (int k = 0; k < 26; ++k)
          if (drives & (1u << k)) {
            const std::string root = std::string(1, (char)('A' + k)) + ":\\";
            if (fs::is_directory(fs::path(root), ec)) d.push(Json(root));
          }
#else
        d.push(Json(std::string("/")));  // (the root, then the home folder's usual folders)
#endif
        if (const char* up = plat::home_env())
          for (const char* sub : {"Documents", "Desktop", "Downloads", "OneDrive"})
            if (fs::is_directory(fs::path(up) / sub, ec)) d.push(Json((fs::path(up) / sub).string()));
        Json r = Json::object();
        r.set("folders", d);
        return rs.send(200, "application/json", r.dump());
      }
      const fs::path f = fs::path(path);
      if (!f.is_absolute() || !fs::is_directory(f, ec)) return fail(404, "no such folder");
      for (fs::directory_iterator it(f, fs::directory_options::skip_permission_denied, ec), end; it != end && !ec; it.increment(ec)) {
        if (!it->is_directory(ec)) continue;
        const std::string n = it->path().filename().string();
        if (n.empty() || n[0] == '.' || n[0] == '$' || plat::is_hidden_or_system(it->path())) continue;
        d.push(Json(n));
        if (d.size() >= 2000) break;
      }
      Json r = Json::object();
      r.set("path", f.string());
      r.set("parent", f == f.root_path() ? std::string() : f.parent_path().string());
      r.set("folders", d);
      std::string why = refuse(f);
      if (!why.empty()) r.set("refused", why);
      return rs.send(200, "application/json", r.dump());
    }
    std::lock_guard<std::mutex> g(mu());
    if (rq.path == "/v1/projects" && rq.method == "GET") {
      Json d = Json::array();
      for (const Project& p : load()) d.push(to_json(p, false));
      Json r = Json::object();
      r.set("data", d);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path == "/v1/projects" && rq.method == "POST") {
      const Json a = Json::parse(rq.body);
      std::error_code ec;
      if (!fs::path(a["folder"].str()).is_absolute()) return fail(400, "folder required: a full path");
      const fs::path f = fs::weakly_canonical(fs::path(a["folder"].str()), ec);
      if (ec) return fail(400, "folder required: a full path");
      const std::string why = refuse(f);
      if (!why.empty()) return fail(400, why);
      std::vector<Project> ps = load();
      const std::string id = id_of(f);
      for (const Project& p : ps)
        if (p.id == id) return rs.send(200, "application/json", to_json(p, true).dump());
      Project p;
      p.id = id;
      p.folder = f.string();
      p.name = a["name"].str().empty() ? f.filename().string() : a["name"].str();
      if (p.name.size() > 80) {  // (cut at 80 bytes, not inside a UTF-8 character)
        size_t n = 80;
        while (n > 0 && ((unsigned char)p.name[n] & 0xC0) == 0x80) --n;
        p.name.resize(n);
      }
      p.created = (int64_t)std::time(nullptr);
      ps.push_back(p);
      if (!save(ps)) return fail(500, "cannot write " + list_path().string());
      return rs.send(200, "application/json", to_json(p, true).dump());
    }
    if (rq.path.rfind("/v1/projects/", 0) == 0) {
      const std::string id = rq.path.substr(13);
      if (!memo::valid_project(id)) return fail(400, "project: 16 hex digits");
      std::vector<Project> ps = load();
      auto it = std::find_if(ps.begin(), ps.end(), [&](const Project& p) { return p.id == id; });
      if (it == ps.end()) return fail(404, "no project " + id);
      if (rq.method == "GET") return rs.send(200, "application/json", to_json(*it, true).dump());
      if (rq.method == "DELETE") {
        ps.erase(it);
        if (!save(ps)) return fail(500, "cannot write " + list_path().string());
        Json r = Json::object();
        r.set("ok", true);
        r.set("note", "removed from the list; its folder and its memory are kept");
        return rs.send(200, "application/json", r.dump());
      }
    }
    return fail(404, "no such projects call");
  } catch (const std::exception& e) {
    return fail(400, e.what());
  }
}
}  // namespace proj

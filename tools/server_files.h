// The web UI's files and chats on disk, in the data folder (src/app_dir.h):
//   uploads\<yyyy-mm>\<id>-<name>   every file attached in a chat (images, audio, video, documents); id = 16 hex digits
//                                   of the FNV-1a hash of its bytes, so the same file is kept once
//   chats\<id>.json                 one conversation (the web UI's JSON: title, times, model, messages); messages refer
//                                   to their files as "/files/<id>"
// Model-independent: served while a model loads or generates. Names are cut to safe ASCII and ids checked, so nothing
// is read or written outside these two folders.
#pragma once
#include "../src/app_dir.h"
#include "../src/json.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace files {
namespace fs = std::filesystem;

struct Info {
  std::string id, name, kind, path;
  uint64_t bytes = 0;
  int64_t created = 0;  // (unix seconds)
};

inline std::string lower(std::string s) {
  for (char& c : s) c = (char)tolower((unsigned char)c);
  return s;
}
inline std::string ext_of(const std::string& name) {
  const size_t d = name.rfind('.');
  return d == std::string::npos ? "" : lower(name.substr(d + 1));
}
inline std::string kind_of(const std::string& name) {
  static const std::set<std::string> img = {"png", "jpg", "jpeg", "gif", "webp", "bmp"},
                                     aud = {"wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "wma"},
                                     vid = {"mp4", "mov", "mkv", "webm", "wmv", "avi"};
  const std::string e = ext_of(name);
  return img.count(e) ? "image" : aud.count(e) ? "audio" : vid.count(e) ? "video" : "document";
}
inline std::string mime_of(const std::string& name) {
  static const std::map<std::string, std::string> m = {
      {"png", "image/png"}, {"jpg", "image/jpeg"}, {"jpeg", "image/jpeg"}, {"gif", "image/gif"}, {"webp", "image/webp"},
      {"bmp", "image/bmp"}, {"wav", "audio/wav"}, {"mp3", "audio/mpeg"}, {"m4a", "audio/mp4"}, {"aac", "audio/aac"},
      {"flac", "audio/flac"}, {"ogg", "audio/ogg"}, {"opus", "audio/ogg"}, {"wma", "audio/x-ms-wma"},
      {"mp4", "video/mp4"}, {"mov", "video/quicktime"}, {"mkv", "video/x-matroska"}, {"webm", "video/webm"},
      {"wmv", "video/x-ms-wmv"}, {"avi", "video/x-msvideo"}, {"txt", "text/plain; charset=utf-8"},
      {"md", "text/markdown; charset=utf-8"}, {"json", "application/json"}, {"csv", "text/csv; charset=utf-8"},
      {"pdf", "application/pdf"}, {"html", "text/plain; charset=utf-8"}, {"svg", "text/plain; charset=utf-8"}};
  auto it = m.find(ext_of(name));
  return it == m.end() ? "application/octet-stream" : it->second;  // (html / svg as text: never run as a page)
}
// a file name that is safe on disk and in a header: ASCII letters, digits and . _ - ( ) space; at most 80 characters
inline std::string safe_name(const std::string& in) {
  std::string base = in.substr(in.find_last_of("/\\") == std::string::npos ? 0 : in.find_last_of("/\\") + 1), out;
  for (unsigned char c : base) out += (isalnum(c) || strchr("._-() ", c)) && c < 128 ? (char)c : '_';
  while (!out.empty() && (out[0] == '.' || out[0] == ' ')) out.erase(0, 1);
  if (out.size() > 80) {
    const std::string e = ext_of(out);
    out = out.substr(0, 80 - (e.empty() ? 0 : e.size() + 1)) + (e.empty() ? "" : "." + e);
  }
  return out.empty() ? "file" : out;
}
inline bool valid_id(const std::string& id, size_t max = 64) {
  if (id.empty() || id.size() > max) return false;
  for (unsigned char c : id)
    if (!isalnum(c) && c != '-' && c != '_') return false;
  return true;
}
inline std::string hex64(uint64_t h) {
  char b[17];
  snprintf(b, sizeof b, "%016llx", (unsigned long long)h);
  return b;
}

class Store {
 public:
  Store() : up_(fs::path(app_dir()) / "uploads"), chats_(fs::path(app_dir()) / "chats") {}

  Info put(const std::string& name_in, const std::string& data) {
    std::lock_guard<std::mutex> g(mu_);
    scan();
    uint64_t h = 1469598103934665603ull;  // (FNV-1a of the bytes: the media cache's key too)
    for (unsigned char b : data) h = (h ^ b) * 1099511628211ull;
    const std::string id = hex64(h);
    if (auto it = by_id_.find(id); it != by_id_.end()) return it->second;
    char month[16];
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &now);
    strftime(month, sizeof month, "%Y-%m", &tm);
    const fs::path dir = up_ / month;
    std::error_code ec;
    fs::create_directories(dir, ec);
    const std::string name = safe_name(name_in);
    const fs::path p = dir / (id + "-" + name);
    {
      std::ofstream o(p, std::ios::binary);
      o.write(data.data(), (std::streamsize)data.size());
      if (!o) throw std::runtime_error("cannot write " + p.string());
    }
    Info i{id, name, kind_of(name), p.string(), (uint64_t)data.size(), (int64_t)now};
    by_id_[id] = i;
    return i;
  }
  bool get(const std::string& id, Info& out) {
    std::lock_guard<std::mutex> g(mu_);
    scan();
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return false;
    out = it->second;
    return true;
  }
  static bool read(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
  }
  std::vector<Info> list() {
    std::lock_guard<std::mutex> g(mu_);
    scan();
    std::vector<Info> v;
    for (auto& [id, i] : by_id_) v.push_back(i);
    std::sort(v.begin(), v.end(), [](const Info& a, const Info& b) { return a.created > b.created; });
    return v;
  }
  bool remove(const std::string& id) {
    std::lock_guard<std::mutex> g(mu_);
    scan();
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return false;
    std::error_code ec;
    fs::remove(it->second.path, ec);
    by_id_.erase(it);
    return !ec;
  }
  // the bytes of a message's file reference "/files/<id>" (false: not such a reference)
  bool resolve(const std::string& ref, std::string& out) {
    if (ref.rfind("/files/", 0) != 0) return false;
    Info i;
    if (!get(ref.substr(7), i) || !read(i.path, out)) throw std::runtime_error("file " + ref.substr(7) + " not found");
    return true;
  }

  // ---- chats
  std::string chat_path(const std::string& id) const { return (chats_ / (id + ".json")).string(); }
  bool chat_get(const std::string& id, std::string& out) { return valid_id(id) && read(chat_path(id), out); }
  void chat_put(const std::string& id, const std::string& body) {
    std::error_code ec;
    fs::create_directories(chats_, ec);
    const std::string p = chat_path(id), tmp = p + ".tmp";  // (written whole, then renamed: never half a chat)
    {
      std::ofstream o(tmp, std::ios::binary);
      o.write(body.data(), (std::streamsize)body.size());
      if (!o) throw std::runtime_error("cannot write " + tmp);
    }
    fs::rename(tmp, p, ec);
    if (ec) throw std::runtime_error("cannot write " + p + ": " + ec.message());
  }
  bool chat_remove(const std::string& id) {
    std::error_code ec;
    return valid_id(id) && fs::remove(chat_path(id), ec);
  }
  // every chat: its id and JSON text
  std::vector<std::pair<std::string, std::string>> chats_all() {
    std::vector<std::pair<std::string, std::string>> v;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(chats_, ec)) {
      if (!e.is_regular_file() || e.path().extension() != ".json") continue;
      std::string t;
      if (read(e.path().string(), t)) v.push_back({e.path().stem().string(), std::move(t)});
    }
    return v;
  }
  // the file ids a chat's JSON refers to: "/files/<id>" (urls) and "file_id":"<id>" (file parts)
  static std::set<std::string> refs(const std::string& text) {
    std::set<std::string> r;
    for (const char* pat : {"/files/", "\"file_id\":\""})
      for (size_t i = 0, n = strlen(pat); (i = text.find(pat, i)) != std::string::npos; i += n) {
        std::string fid = text.substr(i + n, 21);
        if (fid.rfind("file-", 0) == 0) fid = fid.substr(5);
        fid = fid.substr(0, 16);
        if (fid.size() == 16 && valid_id(fid)) r.insert(fid);
      }
    return r;
  }
  // file id -> the chats that refer to it
  std::map<std::string, std::vector<std::string>> uses() {
    std::map<std::string, std::vector<std::string>> u;
    for (const auto& [cid, text] : chats_all())
      for (const std::string& fid : refs(text)) u[fid].push_back(cid);
    return u;
  }

 private:
  // the uploads folder into by_id_, once (later puts and removes keep it current)
  void scan() {
    if (scanned_) return;
    scanned_ = true;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(up_, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
      if (!it->is_regular_file()) continue;
      const std::string fn = it->path().filename().string();
      if (fn.size() < 18 || fn[16] != '-' || !valid_id(fn.substr(0, 16))) continue;
      Info i;
      i.id = fn.substr(0, 16);
      i.name = fn.substr(17);
      i.kind = kind_of(i.name);
      i.path = it->path().string();
      i.bytes = (uint64_t)it->file_size(ec);
      const auto ft = it->last_write_time(ec);
      i.created = (int64_t)std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::clock_cast<std::chrono::system_clock>(ft).time_since_epoch()).count();
      by_id_[i.id] = i;
    }
  }
  fs::path up_, chats_;
  std::mutex mu_;
  bool scanned_ = false;
  std::map<std::string, Info> by_id_;
};

// one store for the whole server process (every loaded model's Server and the HTTP handler)
inline Store& store() {
  static Store s;
  return s;
}

// A message's saved files ({"type": "file", "file": {"file_id": "<id>" | "file-<id>" | "/files/<id>"}}, OpenAI's
// form) as what the model reads: an image or audio file as an image_url / input_audio part referring to it, any other
// file as a text part with its content. (The place where a document too long for the context will be cut down to its
// parts that matter for the question: the RAG index of docs/whisper-native.md's follow-up.)
inline constexpr size_t kMaxDocChars = 400000;
inline Json expand_files(const Json& messages) {
  Json out = Json::array();
  for (const Json& m : messages.items()) {
    if (!m["content"].is_array()) { out.push(m); continue; }
    Json mm = Json::object();
    for (const auto& [k, v] : m.entries())
      if (k != "content") mm.set(k, v);
    Json parts = Json::array();
    for (const Json& p : m["content"].items()) {
      const std::string ty = p["type"].str();
      if (ty != "file" && ty != "input_file") { parts.push(p); continue; }
      const Json& f = p[ty == "file" ? "file" : "input_file"];
      std::string id = f["file_id"].str(p["file_id"].str());
      if (id.rfind("/files/", 0) == 0) id = id.substr(7);
      if (id.rfind("file-", 0) == 0) id = id.substr(5);
      Info i;
      std::string data;
      Json t = Json::object();
      if (!valid_id(id, 16) || !store().get(id, i) || !Store::read(i.path, data)) {
        t.set("type", "text");
        t.set("text", "[a file that is no longer there: " + id + "]");
        parts.push(t);
        continue;
      }
      // what the model knows about the file besides its content: name, kind and format, size, when it was uploaded
      std::string meta;
      {
        char when[40] = "";
        const std::time_t ct = (std::time_t)i.created;
        std::tm tm{};
        if (localtime_s(&tm, &ct) == 0) strftime(when, sizeof when, "%a %d %b %Y %H:%M", &tm);
        const double kb = i.bytes / 1024.0;
        char sz[32];
        if (i.bytes < 1024) snprintf(sz, sizeof sz, "%llu bytes", (unsigned long long)i.bytes);
        else snprintf(sz, sizeof sz, kb >= 1024 ? "%.1f MB" : "%.0f KB", kb >= 1024 ? kb / 1024 : kb);
        meta = "[file: " + i.name + " | " + i.kind + ", " + (ext_of(i.name).empty() ? "no extension" : ext_of(i.name)) + ", " + sz +
               ", uploaded " + when + "]";
      }
      if (i.kind == "image" || i.kind == "audio") {  // (the metadata as a text part next to the image / audio)
        Json mt = Json::object();
        mt.set("type", "text");
        mt.set("text", meta);
        parts.push(mt);
      }
      if (i.kind == "image") {
        Json u = Json::object();
        u.set("url", "/files/" + id);
        t.set("type", "image_url");
        t.set("image_url", u);
      } else if (i.kind == "audio") {
        Json a = Json::object();
        a.set("data", "/files/" + id);
        a.set("format", ext_of(i.name));
        t.set("type", "input_audio");
        t.set("input_audio", a);
      } else {
        const bool binary = data.find('\0') < std::min<size_t>(data.size(), 8192);
        std::string text = binary ? "[" + i.kind + " file, not readable as text]" : data;
        if (!binary) meta.insert(meta.size() - 1, ", " + std::to_string(std::count(data.begin(), data.end(), '\n') + 1) + " lines");
        if (text.size() > kMaxDocChars) text = text.substr(0, kMaxDocChars) + "\n[... cut: the file goes on]";
        t.set("type", "text");
        t.set("text", "=== FILE: " + i.name + " ===\n" + meta + "\n" + text);
      }
      parts.push(t);
    }
    mm.set("content", parts);
    out.push(mm);
  }
  return out;
}

inline Json info_json(const Info& i) {
  Json j = Json::object();
  j.set("id", i.id);
  j.set("object", "file");
  j.set("filename", i.name);
  j.set("bytes", (double)i.bytes);
  j.set("created_at", (double)i.created);
  j.set("kind", i.kind);
  j.set("url", "/files/" + i.id);
  return j;
}

}  // namespace files

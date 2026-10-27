// The web UI's files and chats on disk, in the data folder (src/app_dir.h). One folder per chat:
//   chats\<yyyy-mm-dd> <title> (<chat id>)\     named at the chat's first save (its date and first words), then kept
//     chat.json                     the conversation (the web UI's JSON: title, times, model, messages); messages refer
//                                   to their files as "/files/<id>" (or "file_id": "<id>")
//     <name>                        every file of the chat under its own name: uploads, recordings, documents the
//                                   model wrote; a file from another chat (@) is a hard link, so its bytes are kept once
//     .files.json                   file id -> name and time (a file put into the folder by hand gets an id at the
//                                   next listing)
//     .src\<id>.json                a document the model wrote: its Markdown and options (read_file gives them back,
//                                   so a correction makes a new version from the same source)
//     .pages\<id>.json              a scanned PDF: the pages a vision model transcribed (read_file, search_file)
//   uploads\<yyyy-mm>\<id>-<name>   files of the first web UI version (still read; new files go into a chat folder)
// A file id is 16 hex digits of the FNV-1a hash of its bytes (the media cache's key too). Deleting a chat deletes its
// folder: its files go with it, unless another chat holds them too. Model-independent: served while a model loads or
// generates. Names are cut to safe ASCII and ids checked, so nothing is read or written outside these two folders.
#pragma once
#include "../src/app_dir.h"
#include "../src/json.h"
#include "doc_reader.h"
#include "pdf_images.h"
#include <cmath>
#include <functional>
#include <list>
#include <memory>
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
  std::string chat;     // (the chat whose folder holds this copy; "" : the old uploads folder)
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
      {"pdf", "application/pdf"}, {"docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
      {"doc", "application/msword"}, {"rtf", "application/rtf"}, {"tsv", "text/tab-separated-values; charset=utf-8"},
      {"xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}, {"xlsm", "application/vnd.ms-excel.sheet.macroEnabled.12"},
      {"pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"}, {"html", "text/plain; charset=utf-8"}, {"svg", "text/plain; charset=utf-8"}};
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

  // a file into chat's folder (chat "" : the old uploads folder); the same bytes already there: that file
  Info put(const std::string& name_in, const std::string& data, const std::string& chat = "") {
    std::lock_guard<std::recursive_mutex> g(mu_);
    scan();
    const std::string id = hash(data);
    const std::time_t now = std::time(nullptr);
    std::error_code ec;
    if (chat.empty() || !valid_id(chat)) {  // (the old layout: uploads\<yyyy-mm>\<id>-<name>)
      for (const Info& i : copies(id))
        if (i.chat.empty()) return i;
      char month[16];
      std::tm tm{};
      localtime_s(&tm, &now);
      strftime(month, sizeof month, "%Y-%m", &tm);
      fs::create_directories(up_ / month, ec);
      const std::string name = safe_name(name_in);
      const fs::path pth = up_ / month / (id + "-" + name);
      write(pth, data);
      Info i{id, name, kind_of(name), pth.string(), (uint64_t)data.size(), (int64_t)now, ""};
      by_id_[id].push_back(i);
      return i;
    }
    for (const Info& i : copies(id))
      if (i.chat == chat) return i;
    const fs::path dir = folder(chat, "");
    const std::string name = free_name(dir, safe_name(name_in));
    const fs::path pth = dir / name;
    bool linked = false;
    for (const Info& i : copies(id))  // (the bytes are in another chat's folder: a hard link to them)
      if (fs::create_hard_link(i.path, pth, ec), !ec) { linked = true; break; }
    if (!linked) write(pth, data);
    Info i{id, name, kind_of(name), pth.string(), (uint64_t)data.size(), (int64_t)now, chat};
    by_id_[id].push_back(i);
    save_index(dir, chat);
    return i;
  }
  // a saved file into chat's folder as well (a hard link; a copy if linking fails); false: no such file
  bool link(const std::string& id, const std::string& chat) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    scan();
    const auto v = copies(id);
    if (v.empty()) return false;
    for (const Info& i : v)
      if (i.chat == chat) return true;
    const fs::path dir = folder(chat, "");
    const std::string name = free_name(dir, v[0].name);
    std::error_code ec;
    fs::create_hard_link(v[0].path, dir / name, ec);
    if (ec) {
      ec.clear();
      fs::copy_file(v[0].path, dir / name, ec);
      if (ec) return false;
    }
    Info i = v[0];
    i.name = name;
    i.path = (dir / name).string();
    i.chat = chat;
    by_id_[id].push_back(i);
    std::string src;  // (a document the model wrote: its source goes along; a scan: its transcribed pages)
    if (source(id, src)) put_source(id, chat, src);
    if (pages(id, src)) put_pages(id, src);
    save_index(dir, chat);
    return true;
  }
  bool get(const std::string& id, Info& out) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    scan();
    std::error_code ec;
    for (int pass = 0; pass < 2; ++pass) {
      for (const Info& i : copies(id))
        if (fs::exists(i.path, ec)) {
          out = i;
          return true;
        }
      if (pass == 0) rescan();  // (moved or deleted by hand: the folders again)
    }
    return false;
  }
  static bool read(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
  }
  // every file once (its first copy), newest first; the folders read again, so changes made by hand show
  std::vector<Info> list() {
    std::lock_guard<std::recursive_mutex> g(mu_);
    rescan();
    std::vector<Info> v;
    for (auto& [id, c] : by_id_)
      if (!c.empty()) v.push_back(c[0]);
    std::sort(v.begin(), v.end(), [](const Info& a, const Info& b) { return a.created > b.created; });
    return v;
  }
  // the chats whose folder holds the file
  std::vector<std::string> holders(const std::string& id) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    scan();
    std::vector<std::string> r;
    for (const Info& i : copies(id))
      if (!i.chat.empty()) r.push_back(i.chat);
    return r;
  }
  // the file everywhere (every chat folder, its sources); false: there was none
  bool remove(const std::string& id) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    scan();
    const auto v = copies(id);
    if (v.empty()) return false;
    bool ok = true;
    for (const Info& i : v) ok = remove_copy(i) && ok;
    by_id_.erase(id);
    return ok;
  }
  // the file out of one chat's folder only (the other chats keep their copy)
  bool remove_from(const std::string& id, const std::string& chat) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    scan();
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return false;
    auto& v = it->second;
    for (size_t k = 0; k < v.size(); ++k)
      if (v[k].chat == chat) {
        const Info i = v[k];
        v.erase(v.begin() + (long long)k);
        if (v.empty()) by_id_.erase(it);
        return remove_copy(i);
      }
    return false;
  }
  // a document's source (the create_document arguments), in its chat's .src folder
  void put_source(const std::string& id, const std::string& chat, const std::string& json) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    const fs::path dir = (chat.empty() || !valid_id(chat) ? up_ : folder(chat, "")) / ".src";
    std::error_code ec;
    fs::create_directories(dir, ec);
    write(dir / (id + ".json"), json);
  }
  bool source(const std::string& id, std::string& out) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    if (!valid_id(id, 16)) return false;
    scan();
    for (const Info& i : copies(id))
      if (read((fs::path(i.path).parent_path() / ".src" / (id + ".json")).string(), out)) return true;
    return read((up_ / ".src" / (id + ".json")).string(), out);
  }
  // a scanned PDF's page transcriptions (JSON: {"pages": {"<n>": {"text", "model", "seconds"}}}), in the .pages folder
  // of every folder that holds the file: a chat deleted takes its copy, the other chats keep theirs
  void put_pages(const std::string& id, const std::string& json) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    if (!valid_id(id, 16)) return;
    scan();
    for (const Info& i : copies(id)) {
      const fs::path dir = fs::path(i.path).parent_path() / ".pages";
      std::error_code ec;
      fs::create_directories(dir, ec);
      try { write(dir / (id + ".json"), json); } catch (const std::exception&) {}
    }
  }
  bool pages(const std::string& id, std::string& out) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    if (!valid_id(id, 16)) return false;
    scan();
    for (const Info& i : copies(id))
      if (read((fs::path(i.path).parent_path() / ".pages" / (id + ".json")).string(), out)) return true;
    return false;
  }
  // the bytes of a message's file reference "/files/<id>" (false: not such a reference)
  bool resolve(const std::string& ref, std::string& out) {
    if (ref.rfind("/files/", 0) != 0) return false;
    Info i;
    if (!get(ref.substr(7), i) || !read(i.path, out)) throw std::runtime_error("file " + ref.substr(7) + " not found");
    return true;
  }

  // ---- chats
  // a chat being written: its folder without chat.json yet, made in the last day (its first answer runs, or its
  // files wait in the message box); "delete unused" leaves its files alone
  bool draft(const std::string& id) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    scan();
    auto it = folders_.find(id);
    std::error_code ec;
    if (it == folders_.end() || fs::exists(it->second / "chat.json", ec)) return false;
    const auto t = fs::last_write_time(it->second, ec);
    return !ec && fs::file_time_type::clock::now() - t < std::chrono::hours(24);
  }
  // the chat's folder ("" : none yet)
  std::string chat_dir(const std::string& id) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    scan();
    auto it = folders_.find(id);
    return it == folders_.end() ? "" : it->second.string();
  }
  // the chat's folder, made now if there is none yet (named from its date and the title, if one is given)
  std::string chat_dir_make(const std::string& id, const std::string& title = "") {
    std::lock_guard<std::recursive_mutex> g(mu_);
    if (!valid_id(id)) throw std::runtime_error("bad chat id");
    scan();
    return folder(id, title).string();
  }
  bool chat_get(const std::string& id, std::string& out) {
    if (!valid_id(id)) return false;
    std::lock_guard<std::recursive_mutex> g(mu_);
    scan();
    auto it = folders_.find(id);
    if (it != folders_.end() && read((it->second / "chat.json").string(), out)) return true;
    return read((chats_ / (id + ".json")).string(), out);  // (a chat of the first web UI version)
  }
  // the chat's JSON into its folder; the files its messages refer to are linked into the folder too
  void chat_put(const std::string& id, const std::string& body) {
    std::lock_guard<std::recursive_mutex> g(mu_);
    if (!valid_id(id)) throw std::runtime_error("bad chat id");
    scan();
    std::string title;
    try { title = Json::parse(body)["title"].str(); } catch (const std::exception&) {}
    fs::path dir = folder(id, title);
    std::error_code ec;
    // (the folder made at an upload, "<date> (<id>)", gets the chat's title: at the first save, or at a later one when
    //  that rename failed, e.g. a file in it was being read; Windows renames no folder with an open file. Its date
    //  stays the one it got.)
    const std::string dn = dir.filename().string();
    const bool untitled = dn.size() >= 12 && dn.compare(10, 2, " (") == 0;
    if ((untitled || !fs::exists(dir / "chat.json", ec)) && !title.empty() && title != "New chat") {
      std::string nn = folder_name(id, title);
      if (untitled) nn.replace(0, 10, dn.substr(0, 10));
      const fs::path named = chats_ / nn;
      if (named != dir && !fs::exists(named, ec)) {
        fs::rename(dir, named, ec);
        if (!ec) {
          rescan();
          retarget_terminal(named, dir, named);
          dir = named;
        }
        ec.clear();
      }
    }
    const fs::path p = dir / "chat.json", tmp = dir / "chat.json.tmp";  // (written whole, then renamed: never half a chat)
    try {
      write(tmp, body);
    } catch (...) {  // (a full disk: no half-written file left behind)
      fs::remove(tmp, ec);
      throw;
    }
    fs::rename(tmp, p, ec);
    if (ec) {
      const std::string why = ec.message();
      fs::remove(tmp, ec);
      throw std::runtime_error("cannot write " + p.string() + ": " + why);
    }
    fs::remove(chats_ / (id + ".json"), ec);  // (an old-layout copy of this chat: replaced by the folder)
    for (const std::string& fid : refs(body)) link(fid, id);
  }
  // the chat: its folder with its files (keep_files: only chat.json, the files stay in the folder)
  bool chat_remove(const std::string& id, bool keep_files = false) {
    if (!valid_id(id)) return false;
    std::lock_guard<std::recursive_mutex> g(mu_);
    scan();
    std::error_code ec;
    bool any = fs::remove(chats_ / (id + ".json"), ec);
    auto it = folders_.find(id);
    if (it != folders_.end()) {
      if (keep_files) any = fs::remove(it->second / "chat.json", ec) || any;
      else {
        any = fs::remove_all(it->second, ec) > 0 || any;
        rescan();
      }
    }
    return any;
  }
  // every chat: its id and JSON text
  std::vector<std::pair<std::string, std::string>> chats_all() {
    std::lock_guard<std::recursive_mutex> g(mu_);
    scan();  // (a folder deleted by hand has no chat.json to read; list() reads the folders again)
    std::vector<std::pair<std::string, std::string>> v;
    std::set<std::string> seen;
    std::string t;
    for (const auto& [id, dir] : folders_)
      if (read((dir / "chat.json").string(), t)) {
        v.push_back({id, t});
        seen.insert(id);
      }
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(chats_, ec))  // (the first web UI version's chats\<id>.json)
      if (e.is_regular_file() && e.path().extension() == ".json" && !seen.count(e.path().stem().string()) &&
          read(e.path().string(), t))
        v.push_back({e.path().stem().string(), t});
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
  // file id -> the chats whose messages refer to it
  std::map<std::string, std::vector<std::string>> uses() {
    std::map<std::string, std::vector<std::string>> u;
    for (const auto& [cid, text] : chats_all())
      for (const std::string& fid : refs(text)) u[fid].push_back(cid);
    return u;
  }

 private:
  // the chat's terminal session (.terminal.json: cwd, VIRTUAL_ENV, PATH as absolute paths) after its folder moved
  static void retarget_terminal(const fs::path& dir, const fs::path& from, const fs::path& to) {
    std::string t;
    if (!read((dir / ".terminal.json").string(), t)) return;
    auto json_esc = [](const std::string& x) {  // (as ConvertTo-Json writes a path: backslashes doubled)
      std::string o;
      for (char c : x) o += c == '\\' ? std::string("\\\\") : std::string(1, c);
      return o;
    };
    const std::string a = json_esc(from.string()), b = json_esc(to.string());
    bool any = false;
    for (size_t k = 0; (k = t.find(a, k)) != std::string::npos; k += b.size()) {
      t.replace(k, a.size(), b);
      any = true;
    }
    if (any) try { write(dir / ".terminal.json", t); } catch (const std::exception&) {}
  }
  // FNV-1a of a file, read in parts (a file put into the folder by hand or by a command may be gigabytes)
  static bool hash_file(const std::string& path, std::string& out) {
    std::ifstream f(fs::path(path), std::ios::binary);
    if (!f) return false;
    uint64_t h = 1469598103934665603ull;
    std::vector<char> b(1 << 20);
    while (f) {
      f.read(b.data(), (std::streamsize)b.size());
      for (std::streamsize k = 0; k < f.gcount(); ++k) h = (h ^ (unsigned char)b[(size_t)k]) * 1099511628211ull;
    }
    out = hex64(h);
    return true;
  }
  static std::string hash(const std::string& data) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char b : data) h = (h ^ b) * 1099511628211ull;
    return hex64(h);
  }
  static void write(const fs::path& p, const std::string& data) {
    std::ofstream o(p, std::ios::binary);
    o.write(data.data(), (std::streamsize)data.size());
    if (!o) throw std::runtime_error("cannot write " + p.string());
  }
  std::vector<Info> copies(const std::string& id) const {
    auto it = by_id_.find(id);
    return it == by_id_.end() ? std::vector<Info>{} : it->second;
  }
  // "<yyyy-mm-dd> <title> (<id>)": a name Windows takes (no trailing dot or space, short enough for the path)
  static std::string folder_name(const std::string& id, const std::string& title) {
    char day[16];
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &now);
    strftime(day, sizeof day, "%Y-%m-%d", &tm);
    std::string t;
    for (unsigned char c : title) t += c < 128 && (isalnum(c) || strchr(" -_,", c)) ? (char)c : ' ';
    std::string u;  // (spaces squeezed)
    for (char c : t)
      if (!(c == ' ' && (u.empty() || u.back() == ' '))) u += c;
    if (u.size() > 40) {  // (cut at a word's end)
      const size_t sp = u.rfind(' ', 40);
      u = u.substr(0, sp != std::string::npos && sp > 20 ? sp : 40);
    }
    while (!u.empty() && (u.back() == ' ' || u.back() == '.' || u.back() == ',')) u.pop_back();
    return std::string(day) + (u.empty() || u == "New chat" ? "" : " " + u) + " (" + id + ")";
  }
  fs::path folder(const std::string& chat, const std::string& title) {
    if (auto it = folders_.find(chat); it != folders_.end()) return it->second;
    const fs::path d = chats_ / folder_name(chat, title);
    std::error_code ec;
    fs::create_directories(d, ec);
    if (ec) throw std::runtime_error("cannot make " + d.string() + ": " + ec.message());
    folders_[chat] = d;
    return d;
  }
  // name, or "name (2).ext" ... when another file has it
  static std::string free_name(const fs::path& dir, const std::string& name) {
    std::error_code ec;
    if (name == "chat.json" || fs::exists(dir / name, ec)) {
      const size_t d = name.rfind('.');
      const std::string stem = d == std::string::npos || d == 0 ? name : name.substr(0, d),
                        ext = d == std::string::npos || d == 0 ? "" : name.substr(d);
      for (int k = 2;; ++k)
        if (!fs::exists(dir / (stem + " (" + std::to_string(k) + ")" + ext), ec)) return stem + " (" + std::to_string(k) + ")" + ext;
    }
    return name;
  }
  void save_index(const fs::path& dir, const std::string& chat) {
    Json j = Json::object();
    for (const auto& [id, v] : by_id_)
      for (const Info& i : v)
        if (i.chat == chat) {
          Json e = Json::object();
          e.set("name", i.name);
          e.set("created", (double)i.created);
          e.set("bytes", (double)i.bytes);
          j.set(id, e);
        }
    try { write(dir / ".files.json", j.dump()); } catch (const std::exception&) {}
  }
  bool remove_copy(const Info& i) {
    std::error_code ec;
    const fs::path dir = fs::path(i.path).parent_path();
    const bool ok = fs::remove(i.path, ec) || !fs::exists(i.path, ec);
    std::error_code ec2;
    fs::remove(dir / ".src" / (i.id + ".json"), ec2);
    fs::remove(dir / ".pages" / (i.id + ".json"), ec2);
    if (i.chat.empty()) fs::remove(up_ / ".src" / (i.id + ".json"), ec2);
    else {
      save_index(dir, i.chat);
      // (a folder left with nothing in it but its index, and no chat: gone too)
      bool empty = true;
      for (const auto& e : fs::directory_iterator(dir, ec2))
        if (e.path().filename() != ".files.json" && e.path().filename() != ".terminal.json" && !(e.is_directory() && fs::is_empty(e.path(), ec2))) empty = false;
      if (empty) {
        fs::remove_all(dir, ec2);
        folders_.erase(i.chat);
      }
    }
    return ok;
  }
  void rescan() {
    scanned_ = false;
    scan();
  }
  static int64_t mtime(const fs::directory_entry& e) {
    std::error_code ec;
    const auto ft = e.last_write_time(ec);
    return (int64_t)std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::clock_cast<std::chrono::system_clock>(ft).time_since_epoch()).count();
  }
  // the chat folders and the old uploads folder into by_id_ and folders_
  void scan() {
    if (scanned_) return;
    scanned_ = true;
    by_id_.clear();
    folders_.clear();
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(up_, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
      if (it->is_directory() && it->path().filename().string().rfind(".", 0) == 0) { it.disable_recursion_pending(); continue; }
      if (!it->is_regular_file()) continue;
      const std::string fn = it->path().filename().string();
      if (fn.size() < 18 || fn[16] != '-' || !valid_id(fn.substr(0, 16))) continue;
      Info i{fn.substr(0, 16), fn.substr(17), kind_of(fn.substr(17)), it->path().string(), (uint64_t)it->file_size(ec), mtime(*it), ""};
      by_id_[i.id].push_back(i);
    }
    ec.clear();
    for (const auto& d : fs::directory_iterator(chats_, ec)) {
      if (!d.is_directory()) continue;
      const std::string dn = d.path().filename().string();
      const size_t o = dn.rfind('(');
      if (dn.empty() || dn.back() != ')' || o == std::string::npos) continue;
      const std::string chat = dn.substr(o + 1, dn.size() - o - 2);
      if (!valid_id(chat) || folders_.count(chat)) continue;
      folders_[chat] = d.path();
      Json idx;
      std::string t;
      if (!read((d.path() / ".files.json").string(), t) || !Json::try_parse(t, idx) || !idx.is_object()) idx = Json::object();
      std::map<std::string, std::pair<std::string, Json>> by_name;  // (name -> id, entry)
      for (const auto& [id, e] : idx.entries()) by_name[e["name"].str()] = {id, e};
      bool changed = false;
      size_t found = 0;
      std::error_code ec2;
      for (const auto& f : fs::directory_iterator(d.path(), ec2)) {
        const std::string fn = f.path().filename().string();
        if (!f.is_regular_file() || fn.rfind(".", 0) == 0 || fn == "chat.json" || fn == "chat.json.tmp") continue;
        const uint64_t bytes = (uint64_t)f.file_size(ec2);
        Info i{"", fn, kind_of(fn), f.path().string(), bytes, mtime(f), chat};
        auto b = by_name.find(fn);
        if (b != by_name.end() && (uint64_t)b->second.second["bytes"].as_number(-1) == bytes && valid_id(b->second.first, 16)) {
          i.id = b->second.first;
          i.created = (int64_t)b->second.second["created"].as_number((double)i.created);
        } else {  // (new or changed by hand: hashed now, in parts)
          if (!hash_file(i.path, i.id)) continue;
          changed = true;
        }
        by_id_[i.id].push_back(i);
        ++found;
      }
      if (changed || found != by_name.size()) save_index(d.path(), chat);  // (files added or deleted by hand)
    }
  }
  fs::path up_, chats_;
  std::recursive_mutex mu_;
  bool scanned_ = false;
  std::map<std::string, std::vector<Info>> by_id_;  // (file id -> its copies, one per folder)
  std::map<std::string, fs::path> folders_;         // (chat id -> its folder)
};

// one store for the whole server process (every loaded model's Server and the HTTP handler)
inline Store& store() {
  static Store s;
  return s;
}

// a scanned PDF's transcribed pages (page -> text), from its .pages JSON
inline std::map<int, std::string> page_texts(const std::string& json) {
  std::map<int, std::string> m;
  Json j;
  if (Json::try_parse(json, j))
    for (const auto& [k, v] : j["pages"].entries())
      if (atoi(k.c_str()) > 0) m[atoi(k.c_str())] = v["text"].str();
  return m;
}

// A document's text, outline and search index (tools/doc_reader.h), kept for the last 8 files read (a 227-page PDF
// reads in 0.25 s; a question about it often comes in several tool calls). A PDF's scanned pages: the text a vision
// model transcribed from them so far (tools/pdf_images.h)
struct DocCache {
  std::mutex mu;
  std::list<std::pair<std::string, std::shared_ptr<const rdoc::Index>>> lru;
};
inline DocCache& doc_cache() {
  static DocCache c;
  return c;
}
inline void doc_index_forget(const std::string& id) {
  DocCache& c = doc_cache();
  std::lock_guard<std::mutex> g(c.mu);
  c.lru.remove_if([&](const auto& e) { return e.first == id; });
}
inline constexpr uint64_t kMaxReadBytes = 512ull << 20;
inline std::shared_ptr<const rdoc::Index> doc_index(const Info& i) {
  DocCache& c = doc_cache();
  {
    std::lock_guard<std::mutex> g(c.mu);
    for (auto it = c.lru.begin(); it != c.lru.end(); ++it)
      if (it->first == i.id) {
        c.lru.splice(c.lru.begin(), c.lru, it);
        return it->second;
      }
  }
  std::string data;
  if (i.bytes > kMaxReadBytes) {  // (a file this large is no document to read: a note instead of its bytes in RAM)
    rdoc::Doc d;
    d.note = "the file is too large to read here (" + std::to_string(i.bytes >> 20) + " MB; at most " +
             std::to_string(kMaxReadBytes >> 20) + " MB)";
    return std::make_shared<const rdoc::Index>(std::move(d));
  }
  if (!Store::read(i.path, data)) return nullptr;
  rdoc::Doc d = rdoc::read(i.name, data);
  if (d.pages > 0 && (ext_of(i.name) == "pdf" || data.compare(0, 5, "%PDF-") == 0)) {
    rdoc::pdfimg::scan_pages(data, d);
    std::string t;
    if (!d.scans.empty()) rdoc::pdfimg::add_pages(d, store().pages(i.id, t) ? page_texts(t) : std::map<int, std::string>());
  }
  auto ix = std::make_shared<const rdoc::Index>(std::move(d));
  std::lock_guard<std::mutex> g(c.mu);
  c.lru.push_front({i.id, ix});
  if (c.lru.size() > 8) c.lru.pop_back();
  return ix;
}

// Scanned pages read by the loaded vision model (tools/server.cpp gives `see`: an image's bytes and MIME type ->
// the text the model transcribed; it throws when it cannot). Each page's image comes out of the PDF
// (tools/pdf_images.h); its text goes into the file's .pages JSON, so the next read and search_file have it. Per page:
// {page, seconds, chars} or {page, note} (no image to read, or the model failed).
inline constexpr int kScanPagesPerCall = 4;  // (a page: ~1,000 image tokens and the text, 10-60 s)
inline Json transcribe_pages(const Info& i, const std::vector<int>& pages, const std::string& model,
                             const std::function<std::string(const std::string&, const std::string&)>& see) {
  static std::mutex mu;  // (one file's JSON is read, changed and written whole)
  Json out = Json::array();
  std::string data;
  if (!Store::read(i.path, data)) return out;
  for (int p : pages) {
    Json r = Json::object();
    r.set("page", (double)p);
    const rdoc::pdfimg::Image im = rdoc::pdfimg::page_image(data, p);
    if (im.bytes.empty()) {
      r.set("note", "not read: " + im.note);
      out.push(r);
      continue;
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::string text;
    try {
      text = see(im.bytes, im.mime);
    } catch (const std::exception& e) {
      r.set("note", std::string("the vision model failed: ") + e.what());
      out.push(r);
      continue;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    {
      std::lock_guard<std::mutex> g(mu);
      std::string t;
      Json j;
      if (!store().pages(i.id, t) || !Json::try_parse(t, j) || !j.is_object()) j = Json::object();
      Json pg = j["pages"].is_object() ? j["pages"] : Json::object(), e = Json::object();
      e.set("text", text);
      e.set("model", model);
      e.set("seconds", std::round(s));
      e.set("image", im.filter + " " + std::to_string(im.w) + "x" + std::to_string(im.h));
      pg.set(std::to_string(p), e);
      j.set("pages", pg);
      store().put_pages(i.id, j.dump());
    }
    r.set("seconds", std::round(s));
    r.set("chars", (double)text.size());
    out.push(r);
  }
  doc_index_forget(i.id);
  return out;
}
// a document longer than this goes to the model as its outline and beginning (~6K tokens; 100 pages are ~60K)
inline constexpr size_t kInlineChars = 24000;
inline int tokens_of(size_t chars) { return (int)(chars / 4 + 1); }  // (an estimate: ~4 characters a token)
// the outline as the model gets it: the top levels only, at most max entries (a section read whole includes its
// subsections, and its numbers stay those of the full outline, so reading section 7 of a chapter shows its parts)
inline constexpr size_t kOutlineMax = 60;
inline std::string pages_of(const rdoc::Section& s) {
  return s.page == s.page_to ? std::to_string(s.page) : std::to_string(s.page) + "-" + std::to_string(s.page_to);
}
inline Json outline_json(const rdoc::Index& ix, int depth = 0) {
  if (!depth) depth = ix.outline_depth(kOutlineMax);
  Json o = Json::array();
  size_t n = 0;
  for (size_t k = 0; k < ix.outline.size(); ++k) {
    const rdoc::Section& s = ix.outline[k];
    if (s.level > depth) continue;
    if (++n > kOutlineMax * 2) break;
    Json j = Json::object();
    j.set("section", (double)(k + 1));
    j.set("title", s.title);
    j.set("level", (double)s.level);
    if (ix.doc.pages > 1) j.set("pages", pages_of(s));
    j.set("tokens", (double)tokens_of(s.span_chars));
    o.push(j);
  }
  return o;
}
inline std::string outline_text(const rdoc::Index& ix) {
  const int depth = ix.outline_depth(kOutlineMax);
  std::string o;
  size_t n = 0, hidden = 0;
  for (size_t k = 0; k < ix.outline.size(); ++k) {
    const rdoc::Section& s = ix.outline[k];
    if (s.level > depth) { ++hidden; continue; }
    if (++n > kOutlineMax * 2) { o += "...\n"; break; }
    o += std::string((size_t)std::max(0, s.level - 1) * 2, ' ') + "section " + std::to_string(k + 1) + ": " + s.title +
         (ix.doc.pages > 1 ? " (p. " + pages_of(s) + ", ~" : " (~") + std::to_string(tokens_of(s.span_chars)) + " tokens)\n";
  }
  if (hidden) o += "(" + std::to_string(hidden) + " subsections not listed: reading a section shows its subsections)\n";
  return o;
}

// A message's saved files ({"type": "file", "file": {"file_id": "<id>" | "file-<id>" | "/files/<id>"}}, OpenAI's
// form) as what the model reads: an image or audio file as an image_url / input_audio part referring to it, any other
// file as a text part with its text (DOCX and PDF too, tools/doc_reader.h). With the file tools (tools: true) a long
// document goes in as its outline and beginning: the model reads the sections it needs (read_file) or searches it
// (search_file) instead of a prompt of the whole document; "long": over kInlineChars, or over a quarter of the
// conversation's context (ctx tokens, 0: not known) at 3 characters a token (tables and Indonesian text run short).
inline constexpr size_t kMaxDocChars = 400000;
// the characters one file read may give the model: a fifth of the context (3 characters a token), 1,500 at least
inline size_t read_budget(int ctx) { return ctx > 0 ? std::max<size_t>(1500, (size_t)ctx * 3 / 5) : (size_t)-1; }
inline Json expand_files(const Json& messages, bool tools = false, int ctx = 0) {
  const size_t inline_max = std::min(kInlineChars, ctx > 0 ? (size_t)ctx * 3 / 4 : kInlineChars);
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
        const auto ix = doc_index(i);
        std::string text;
        if (!ix || ix->doc.paras.empty()) text = "[" + (ix && !ix->doc.note.empty() ? ix->doc.note : i.kind + " file, not readable as text") + "]";
        else {
          const std::string full = ix->text(0, ix->doc.paras.size());
          meta.insert(meta.size() - 1, (ix->doc.pages > 0 ? ", " + std::to_string(ix->doc.pages) + " pages" : std::string()) + ", ~" +
                                           std::to_string(tokens_of(full.size())) + " tokens");
          if (!ix->doc.note.empty()) meta += "\n[" + ix->doc.note + "]";
          if (!tools || full.size() <= inline_max) {
            text = full;
            if (text.size() > kMaxDocChars) text = text.substr(0, kMaxDocChars) + "\n[... cut: the file goes on]";
          } else {
            size_t cut = std::min(full.size(), (size_t)3000);
            while (cut < full.size() && cut > 0 && ((unsigned char)full[cut] & 0xC0) == 0x80) --cut;
            text = "[A long document: only its outline and beginning are here. Read the parts the question needs with "
                   "read_file {\"file_id\": \"" + i.id + "\", \"section\": N} (or \"pages\": \"A-B\"), or find passages "
                   "with search_file {\"file_id\": \"" + i.id + "\", \"query\": \"...\"}. Do not guess its content.]\n"
                   "Outline:\n" + outline_text(*ix) + "--- beginning ---\n" + full.substr(0, cut) + "\n[...]";
          }
        }
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

// MCP servers for the web UI's model (the server as an MCP client): programs such as blender-mcp that give the model
// tools of another application. Configured in %USERPROFILE%\.inferzero\mcp.json in the usual form
//   { "mcpServers": { "blender": { "command": "uvx", "args": ["blender-mcp"], "env": { "X": "1" } } } }
// (stdio servers: one process per server, newline-delimited JSON-RPC 2.0 on its stdin / stdout, protocol 2025-06-18).
// A server is used only where it is switched on: in plain chats ("global") or per project (mcp-enabled.json). It is
// started at its first use and stays running (with its children, in a job object) until it is stopped, its config
// changes or the server ends. Only for connections from this machine (server.cpp): these are programs on this computer.
// Endpoints: GET /v1/mcp[?project=] (servers, switches, tools of the running ones), GET / PUT /v1/mcp/config (the
// file's text), PUT /v1/mcp/enabled, POST /v1/mcp/start | stop | tools | call.
#pragma once
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace mcp {
namespace fs = std::filesystem;

inline constexpr const char* kProtocol = "2025-06-18";
inline constexpr size_t kMaxLine = 64u << 20;       // (one message: a screenshot in base64 fits)
inline constexpr size_t kMaxResultChars = 100000;   // (text given to the model from one call)
inline constexpr size_t kStderrTail = 4096;         // (kept for the error shown when a server fails to start)

inline fs::path config_path() { return fs::path(app_dir()) / "mcp.json"; }
inline fs::path enabled_path() { return fs::path(app_dir()) / "mcp-enabled.json"; }

inline std::wstring w8(const std::string& s) {
  if (s.empty()) return L"";
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
  std::wstring w(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
  return w;
}
inline std::string u8(const std::wstring& w) {
  if (w.empty()) return "";
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
  std::string s(n, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
  return s;
}
inline bool unbase64(const std::string& in, std::string& out) {
  auto v = [](char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
  };
  out.clear();
  out.reserve(in.size() / 4 * 3);
  unsigned acc = 0;
  int bits = 0;
  for (char c : in) {
    if (c == '=' ) break;
    if (c == '\r' || c == '\n' || c == ' ') continue;
    const int x = v(c);
    if (x < 0) return false;
    acc = acc << 6 | (unsigned)x;
    if ((bits += 6) >= 8) out += (char)(acc >> (bits -= 8) & 0xff);
  }
  return true;
}
// a name the model can call: letters, digits, _ and -, at most 64 characters
inline std::string fn_part(const std::string& s) {
  std::string o;
  for (unsigned char c : s) o += (std::isalnum(c) || c == '_' || c == '-') ? (char)c : '_';
  return o.empty() ? "_" : o;
}
inline std::string fn_name(const std::string& server, const std::string& tool) {
  std::string n = fn_part(server) + "__" + fn_part(tool);
  if (n.size() > 64) {  // (cut, with a short hash of the whole so two long names stay apart)
    uint32_t h = 2166136261u;
    for (unsigned char c : server + "/" + tool) h = (h ^ c) * 16777619u;
    char b[9];
    snprintf(b, sizeof b, "%08x", h);
    n = n.substr(0, 55) + "_" + b;
  }
  return n;
}

struct Config {
  std::string name, command, cwd;
  std::vector<std::string> args;
  std::vector<std::pair<std::string, std::string>> env;
  int timeout_s = 300;  // (one tool call; "timeout" in seconds in mcp.json)
  std::string key() const {
    std::string k = command + "\x1f" + cwd + "\x1f" + std::to_string(timeout_s);
    for (auto& a : args) k += "\x1f" + a;
    for (auto& [n, v] : env) k += "\x1e" + n + "=" + v;
    return k;
  }
};

// mcp.json's servers ("mcpServers", or "servers" as VS Code writes it); err: why the file cannot be read (none: empty)
inline std::vector<Config> configs(std::string& err) {
  std::vector<Config> out;
  err.clear();
  std::error_code ec;
  if (!fs::exists(config_path(), ec)) return out;
  std::ifstream f(config_path(), std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  Json j;
  try {
    j = Json::parse(ss.str());
  } catch (const std::exception& e) {
    err = std::string("mcp.json is not valid JSON: ") + e.what();
    return out;
  }
  const Json& s = j.has("mcpServers") ? j["mcpServers"] : j["servers"];
  for (auto& [name, v] : s.entries()) {
    if (!v.is_object()) continue;
    Config c;
    c.name = name;
    c.command = v["command"].str();
    c.cwd = v["cwd"].str();
    for (const Json& a : v["args"].items()) c.args.push_back(a.is_string() ? a.as_string() : a.dump());
    for (auto& [k, x] : v["env"].entries()) c.env.push_back({k, x.is_string() ? x.as_string() : x.dump()});
    if (v["timeout"].is_number()) c.timeout_s = (int)v["timeout"].as_clamped(300, 5, 24 * 3600);
    if (v["disabled"].as_bool()) continue;
    out.push_back(std::move(c));
  }
  return out;
}

// CommandLineToArgvW's quoting
inline void append_arg(std::wstring& cl, const std::wstring& a) {
  if (!cl.empty()) cl += L' ';
  if (!a.empty() && a.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
    cl += a;
    return;
  }
  cl += L'"';
  for (size_t i = 0;; ++i) {
    size_t bs = 0;
    while (i < a.size() && a[i] == L'\\') ++i, ++bs;
    if (i == a.size()) {
      cl.append(bs * 2, L'\\');
      break;
    }
    if (a[i] == L'"') {
      cl.append(bs * 2 + 1, L'\\');
      cl += L'"';
    } else {
      cl.append(bs, L'\\');
      cl += a[i];
    }
  }
  cl += L'"';
}
// the program's full path: as given, or found in PATH with .exe / .cmd / .bat / .com (npx is npx.cmd)
inline std::wstring find_program(const std::string& command) {
  const std::wstring c = w8(command);
  std::vector<wchar_t> b(32768);
  std::wstring found;
  auto look = [&](const wchar_t* ext) {
    const DWORD n = SearchPathW(nullptr, c.c_str(), ext, (DWORD)b.size(), b.data(), nullptr);
    const DWORD at = n && n < b.size() ? GetFileAttributesW(b.data()) : INVALID_FILE_ATTRIBUTES;
    if (at != INVALID_FILE_ATTRIBUTES && !(at & FILE_ATTRIBUTE_DIRECTORY)) found.assign(b.data(), n);
    return !found.empty();
  };
  if (fs::path(c).has_extension()) look(nullptr);
  else
    for (const wchar_t* ext : {L".exe", L".cmd", L".bat", L".com"})
      if (look(ext)) break;
  return found;
}

struct Tool {
  std::string name, title, description;
  Json schema;
  bool read_only = false;
};

class Client {
 public:
  explicit Client(Config c) : cfg_(std::move(c)) {}
  ~Client() { stop(); }
  const Config& cfg() const { return cfg_; }

  // started once (callers at the same time wait for the first); throws with the reason and the server's last output
  void ensure_started() {
    std::lock_guard<std::mutex> lk(start_mu_);
    if (started_ && !dead_) return;
    if (started_) cleanup();
    start_error_.clear();
    try {
      spawn();
      Json p = Json::object(), caps = Json::object(), info = Json::object();
      p.set("protocolVersion", kProtocol);
      caps.set("roots", Json::object());
      p.set("capabilities", caps);
      info.set("name", "inferzero");
      info.set("version", "1.0");
      p.set("clientInfo", info);
      const Json r = request("initialize", p, 120000);  // (uvx / npx may download the server at its first start)
      server_name_ = r["serverInfo"]["name"].str(cfg_.name);
      instructions_ = r["instructions"].str();
      notify("notifications/initialized", Json::object());
      load_tools();
      started_ = true;
    } catch (const std::exception& e) {
      start_error_ = e.what();
      const std::string tail = stderr_tail();
      cleanup();
      throw std::runtime_error(start_error_ + (tail.empty() ? "" : "; its last output: " + tail));
    }
  }
  bool running() const { return started_ && !dead_; }
  std::string start_error() const { return start_error_; }
  std::string instructions() const { return instructions_; }
  std::vector<Tool> tools() {
    if (tools_changed_.exchange(false)) {
      try { load_tools(); } catch (...) {}
    }
    std::lock_guard<std::mutex> lk(mu_);
    return tools_;
  }
  // tools/call: the result object (content, isError, structuredContent); cancelled: asked every 200 ms
  Json call(const std::string& tool, const Json& args, const std::function<bool()>& cancelled) {
    ensure_started();
    Json p = Json::object();
    p.set("name", tool);
    p.set("arguments", args.is_object() ? args : Json::object());
    return request("tools/call", p, (int64_t)cfg_.timeout_s * 1000, cancelled);
  }
  void stop() {
    std::lock_guard<std::mutex> lk(start_mu_);
    cleanup();
    started_ = false;
  }

 private:
  Config cfg_;
  std::mutex start_mu_, mu_, write_mu_;
  std::condition_variable cv_;
  std::map<int64_t, Json> replies_;
  int64_t next_id_ = 1;
  std::atomic<bool> dead_{false}, tools_changed_{false};
  bool started_ = false;
  std::string start_error_, server_name_, instructions_, err_tail_;
  std::vector<Tool> tools_;
  HANDLE proc_ = nullptr, job_ = nullptr, in_ = nullptr, out_ = nullptr, err_ = nullptr;
  std::thread reader_, err_reader_;

  std::string stderr_tail() {
    std::lock_guard<std::mutex> lk(mu_);
    std::string t = err_tail_;
    while (!t.empty() && (t.back() == '\n' || t.back() == '\r' || t.back() == ' ')) t.pop_back();
    if (t.size() > 600) t = "..." + t.substr(t.size() - 600);
    return t;
  }

  void spawn() {
    if (cfg_.command.empty()) throw std::runtime_error("no \"command\" in mcp.json");
    const std::wstring prog = find_program(cfg_.command);
    if (prog.empty()) throw std::runtime_error("cannot find the program \"" + cfg_.command + "\" (not in PATH)");
    std::wstring cl;
    std::wstring ext = fs::path(prog).extension().wstring();
    for (auto& ch : ext) ch = (wchar_t)towlower(ch);
    const bool script = ext == L".cmd" || ext == L".bat";
    std::wstring app;
    if (script) {  // (a batch file runs through cmd.exe: /s takes the outer quotes off the rest)
      wchar_t sys[MAX_PATH];
      GetSystemDirectoryW(sys, MAX_PATH);
      app = std::wstring(sys) + L"\\cmd.exe";
      std::wstring inner;
      append_arg(inner, prog);
      for (auto& a : cfg_.args) append_arg(inner, w8(a));
      cl = L"cmd.exe /d /s /c \"" + inner + L"\"";
    } else {
      app = prog;
      append_arg(cl, prog);
      for (auto& a : cfg_.args) append_arg(cl, w8(a));
    }
    // the environment: this server's, with the config's variables set over it
    std::vector<std::pair<std::wstring, std::wstring>> envv;
    if (wchar_t* e = GetEnvironmentStringsW()) {
      for (wchar_t* p = e; *p; p += wcslen(p) + 1) {
        std::wstring s = p;
        const size_t eq = s.find(L'=', 1);
        if (eq != std::wstring::npos) envv.push_back({s.substr(0, eq), s.substr(eq + 1)});
      }
      FreeEnvironmentStringsW(e);
    }
    // (Python servers read and write their pipes in UTF-8, as the protocol wants, not in the ANSI code page; the
    //  config can still set these)
    std::vector<std::pair<std::string, std::string>> envs = {{"PYTHONIOENCODING", "utf-8"}, {"PYTHONUTF8", "1"}};
    for (auto& kv : cfg_.env) envs.push_back(kv);
    for (auto& [k, v] : envs) {
      const std::wstring wk = w8(k);
      bool set = false;
      for (auto& [ek, ev] : envv)
        if (_wcsicmp(ek.c_str(), wk.c_str()) == 0) ev = w8(v), set = true;
      if (!set) envv.push_back({wk, w8(v)});
    }
    std::sort(envv.begin(), envv.end(), [](auto& a, auto& b) { return _wcsicmp(a.first.c_str(), b.first.c_str()) < 0; });
    std::wstring block;
    for (auto& [k, v] : envv) block += k + L"=" + v + L'\0';
    block += L'\0';

    SIZE_T alen = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &alen);
    std::vector<char> abuf(alen);
    std::wstring wcwd = w8(cfg_.cwd);
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE in_rd = nullptr, out_wr = nullptr, err_wr = nullptr;
    auto close = [](HANDLE& h) { if (h) CloseHandle(h), h = nullptr; };
    if (!CreatePipe(&in_rd, &in_, &sa, 0) || !CreatePipe(&out_, &out_wr, &sa, 0) || !CreatePipe(&err_, &err_wr, &sa, 0)) {
      close(in_rd), close(out_wr), close(err_wr), close(in_), close(out_), close(err_);
      throw std::runtime_error("cannot make a pipe");
    }
    SetHandleInformation(in_, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(out_, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(err_, HANDLE_FLAG_INHERIT, 0);
    job_ = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
    li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (job_) SetInformationJobObject(job_, JobObjectExtendedLimitInformation, &li, sizeof li);
    STARTUPINFOEXW si{};
    si.StartupInfo.cb = sizeof si;
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = in_rd;
    si.StartupInfo.hStdOutput = out_wr;
    si.StartupInfo.hStdError = err_wr;
    HANDLE inherit[3] = {in_rd, out_wr, err_wr};  // (only these: another request's pipes stay out of the child)
    si.lpAttributeList = (LPPROC_THREAD_ATTRIBUTE_LIST)abuf.data();
    const bool init = alen && InitializeProcThreadAttributeList(si.lpAttributeList, 1, 0, &alen);
    const bool attrs = init && UpdateProcThreadAttribute(si.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit,
                                                         sizeof inherit, nullptr, nullptr);
    PROCESS_INFORMATION pi{};
    const BOOL ok = attrs && CreateProcessW(app.c_str(), cl.data(), nullptr, nullptr, TRUE,
                                            CREATE_NO_WINDOW | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
                                            block.data(), wcwd.empty() ? nullptr : wcwd.c_str(), &si.StartupInfo, &pi);
    const DWORD err = GetLastError();
    if (init) DeleteProcThreadAttributeList(si.lpAttributeList);
    close(in_rd), close(out_wr), close(err_wr);
    if (!ok) {
      close(in_), close(out_), close(err_), close(job_);
      throw std::runtime_error("cannot start " + u8(prog) + " (error " + std::to_string(err) + ")");
    }
    if (job_) AssignProcessToJobObject(job_, pi.hProcess);
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    proc_ = pi.hProcess;
    dead_ = false;
    {
      std::lock_guard<std::mutex> lk(mu_);
      replies_.clear();
      err_tail_.clear();
    }
    reader_ = std::thread([this] { read_loop(); });
    err_reader_ = std::thread([this] {
      char b[4096];
      DWORD n;
      while (ReadFile(err_, b, sizeof b, &n, nullptr) && n) {
        std::lock_guard<std::mutex> lk(mu_);
        err_tail_.append(b, n);
        if (err_tail_.size() > kStderrTail * 2) err_tail_.erase(0, err_tail_.size() - kStderrTail);
      }
    });
  }

  void cleanup() {
    {
      std::lock_guard<std::mutex> lk(write_mu_);  // (the reader thread may be answering a ping)
      if (in_) CloseHandle(in_), in_ = nullptr;  // (stdin closed: a well-behaved server ends by itself)
    }
    if (proc_) {
      WaitForSingleObject(proc_, 1500);
      if (job_) TerminateJobObject(job_, 1);
      TerminateProcess(proc_, 1);
      WaitForSingleObject(proc_, 3000);
    } else if (job_) {
      TerminateJobObject(job_, 1);
    }
    // (a grandchild outside the job may still hold a pipe: the reads are cancelled rather than waited for)
    for (int k = 0; k < 20 && reader_.joinable() && !dead_; ++k) Sleep(100);
    if (out_) CancelIoEx(out_, nullptr);
    if (err_) CancelIoEx(err_, nullptr);
    if (reader_.joinable()) reader_.join();
    if (err_reader_.joinable()) err_reader_.join();
    if (out_) CloseHandle(out_), out_ = nullptr;
    if (err_) CloseHandle(err_), err_ = nullptr;
    if (proc_) CloseHandle(proc_), proc_ = nullptr;
    if (job_) CloseHandle(job_), job_ = nullptr;
    dead_ = true;
    cv_.notify_all();
    std::lock_guard<std::mutex> lk(mu_);
    tools_.clear();
  }

  void read_loop() {
    std::string buf;
    char b[65536];
    DWORD n;
    bool overlong = false;
    while (ReadFile(out_, b, sizeof b, &n, nullptr) && n) {
      size_t from = 0;
      for (size_t i = 0; i < n; ++i) {
        if (b[i] != '\n') continue;
        if (!overlong) buf.append(b + from, i - from);
        if (!overlong) try { handle_line(buf); } catch (...) {}  // (out of memory: that message lost, not the server)
        buf.clear();
        overlong = false;
        from = i + 1;
      }
      if (!overlong) buf.append(b + from, n - from);
      if (buf.size() > kMaxLine) buf.clear(), overlong = true;  // (dropped: its caller times out with a reason)
    }
    dead_ = true;
    cv_.notify_all();
  }

  void handle_line(std::string& line) {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
    if (line.empty() || line[0] != '{') return;  // (a server that logs to stdout: not a message)
    Json m;
    try {
      m = Json::parse(line);
    } catch (...) {
      return;
    }
    const bool has_id = m.has("id") && !m["id"].is_null();
    if (m.has("method")) {
      const std::string method = m["method"].str();
      if (!has_id) {
        if (method == "notifications/tools/list_changed") tools_changed_ = true;
        return;
      }
      Json r = Json::object();
      r.set("jsonrpc", "2.0");
      r.set("id", m["id"]);
      if (method == "ping") {
        r.set("result", Json::object());
      } else if (method == "roots/list") {
        Json res = Json::object();
        res.set("roots", Json::array());
        r.set("result", res);
      } else {
        Json e = Json::object();
        e.set("code", -32601);
        e.set("message", "not supported by this client: " + method);
        r.set("error", e);
      }
      try { send(r); } catch (...) {}  // (on the reader thread: an exception here would end the whole server)
      return;
    }
    if (!has_id || !m["id"].is_number()) return;
    std::lock_guard<std::mutex> lk(mu_);
    replies_[m["id"].as_int()] = std::move(m);
    cv_.notify_all();
  }

  void send(const Json& m) {
    std::lock_guard<std::mutex> lk(write_mu_);
    if (!in_) throw std::runtime_error("the server is not running");
    const std::string s = m.dump() + "\n";
    DWORD w = 0;
    for (size_t off = 0; off < s.size(); off += w)
      if (!WriteFile(in_, s.data() + off, (DWORD)std::min<size_t>(s.size() - off, 1u << 20), &w, nullptr) || !w)
        throw std::runtime_error("the server closed its input");
  }
  void notify(const std::string& method, const Json& params) {
    Json m = Json::object();
    m.set("jsonrpc", "2.0");
    m.set("method", method);
    m.set("params", params);
    send(m);
  }
  Json request(const std::string& method, const Json& params, int64_t timeout_ms,
               const std::function<bool()>& cancelled = nullptr) {
    int64_t id;
    {
      std::lock_guard<std::mutex> lk(mu_);
      id = next_id_++;
    }
    Json m = Json::object();
    m.set("jsonrpc", "2.0");
    m.set("id", (double)id);
    m.set("method", method);
    m.set("params", params);
    send(m);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
      auto it = replies_.find(id);
      if (it != replies_.end()) {
        Json r = std::move(it->second);
        replies_.erase(it);
        lk.unlock();
        if (r.has("error")) {
          const Json& e = r["error"];
          throw std::runtime_error(method + ": " + e["message"].str("error " + e["code"].dump()));
        }
        return r["result"];
      }
      if (dead_) throw std::runtime_error("the MCP server " + cfg_.name + " ended");
      const bool over = std::chrono::steady_clock::now() >= deadline;
      lk.unlock();
      const bool gone = !over && cancelled && cancelled();
      if (over || gone) {
        Json p = Json::object();
        p.set("requestId", (double)id);
        p.set("reason", over ? "timed out" : "the user stopped it");
        try { notify("notifications/cancelled", p); } catch (...) {}
        throw std::runtime_error(method + (over ? ": no answer within " + std::to_string(timeout_ms / 1000) + " s" : ": stopped"));
      }
      lk.lock();
      cv_.wait_for(lk, std::chrono::milliseconds(200));
    }
  }
  void load_tools() {
    std::vector<Tool> ts;
    std::string cursor;
    for (int page = 0; page < 50; ++page) {
      Json p = Json::object();
      if (!cursor.empty()) p.set("cursor", cursor);
      const Json r = request("tools/list", p, 60000);
      for (const Json& t : r["tools"].items()) {
        Tool x;
        x.name = t["name"].str();
        if (x.name.empty()) continue;
        x.title = t["title"].str(t["annotations"]["title"].str());
        x.description = t["description"].str();
        x.schema = t["inputSchema"].is_object() ? t["inputSchema"] : Json::object();
        if (!x.schema.has("type")) x.schema.set("type", "object");
        if (!x.schema.has("properties")) x.schema.set("properties", Json::object());
        x.read_only = t["annotations"]["readOnlyHint"].as_bool();
        ts.push_back(std::move(x));
      }
      cursor = r["nextCursor"].str();
      if (cursor.empty()) break;
    }
    std::lock_guard<std::mutex> lk(mu_);
    tools_ = std::move(ts);
  }
};

// the running servers, by name
struct Manager {
  std::mutex mu;
  std::map<std::string, std::shared_ptr<Client>> clients;
};
inline Manager& manager() {
  static Manager* m = new Manager;  // (never destroyed: the job objects end the processes with this one)
  return *m;
}
// the server's client (made, not started; a changed config replaces a running one); null: not in mcp.json
inline std::shared_ptr<Client> client(const std::string& name, std::string& err) {
  const auto cs = configs(err);
  for (const Config& c : cs) {
    if (c.name != name) continue;
    std::shared_ptr<Client> old;
    std::shared_ptr<Client> cl;
    {
      std::lock_guard<std::mutex> lk(manager().mu);
      auto& slot = manager().clients[name];
      if (slot && slot->cfg().key() != c.key()) old = std::move(slot);
      if (!slot) slot = std::make_shared<Client>(c);
      cl = slot;
    }
    if (old) old->stop();
    return cl;
  }
  if (err.empty()) err = "no MCP server \"" + name + "\" in " + config_path().string();
  return nullptr;
}
inline void stop_all() {
  std::map<std::string, std::shared_ptr<Client>> cs;
  {
    std::lock_guard<std::mutex> lk(manager().mu);
    cs.swap(manager().clients);
  }
  for (auto& [n, c] : cs) c->stop();
}

// switches: {"global": [names], "projects": {"<id>": [names]}}
inline std::mutex& enabled_mu() {
  static std::mutex m;
  return m;
}
inline Json enabled_all() {
  std::ifstream f(enabled_path(), std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  try {
    Json j = Json::parse(ss.str());
    if (j.is_object()) return j;
  } catch (...) {}
  return Json::object();
}
inline std::vector<std::string> enabled(const std::string& project) {
  std::lock_guard<std::mutex> lk(enabled_mu());
  const Json j = enabled_all();
  const Json& l = project.empty() ? j["global"] : j["projects"][project];
  std::vector<std::string> out;
  for (const Json& x : l.items())
    if (x.is_string()) out.push_back(x.as_string());
  return out;
}
inline void set_enabled(const std::string& project, const std::string& name, bool on) {
  std::lock_guard<std::mutex> lk(enabled_mu());
  Json j = enabled_all();
  auto edit = [&](const Json& l) {
    Json o = Json::array();
    for (const Json& x : l.items())
      if (x.is_string() && x.as_string() != name) o.push(x);
    if (on) o.push(name);
    return o;
  };
  if (project.empty()) {
    j.set("global", edit(j["global"]));
  } else {
    Json ps = j["projects"].is_object() ? j["projects"] : Json::object();
    ps.set(project, edit(ps[project]));
    j.set("projects", ps);
  }
  const fs::path tmp = enabled_path().string() + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    f << j.dump();
    if (!f) throw std::runtime_error("cannot write " + tmp.string());
  }
  std::error_code ec;
  fs::rename(tmp, enabled_path(), ec);
  if (ec) throw std::runtime_error("cannot write " + enabled_path().string() + ": " + ec.message());
}

inline Json tool_json(const std::string& server, const Tool& t) {
  Json f = Json::object(), fn = Json::object();
  fn.set("name", fn_name(server, t.name));
  std::string d = "[" + server + " (MCP)" + (t.title.empty() || t.title == t.name ? "" : ": " + t.title) + "] " + t.description;
  if (d.size() > 2000) d = d.substr(0, 2000) + "...";
  fn.set("description", d);
  fn.set("parameters", t.schema);
  f.set("type", "function");
  f.set("function", fn);
  f.set("server", server);
  f.set("tool", t.name);
  f.set("read_only", t.read_only);
  return f;
}

// a tools/call result as the model's text; an image saved by save (its note returned) when save is set
inline std::string result_text(const Json& r, const std::function<std::string(const std::string& data, const std::string& mime)>& save) {
  std::string out;
  auto add = [&](const std::string& s) {
    if (!out.empty()) out += "\n";
    out += s;
  };
  for (const Json& c : r["content"].items()) {
    const std::string t = c["type"].str();
    if (t == "text") {
      add(c["text"].str());
    } else if (t == "image") {
      std::string bytes;
      if (!save || !unbase64(c["data"].str(), bytes) || bytes.empty()) {
        add("[an image (" + c["mimeType"].str("image") + "), not kept]");
        continue;
      }
      add(save(bytes, c["mimeType"].str("image/png")));
    } else if (t == "audio") {
      add("[audio (" + c["mimeType"].str("audio") + "), not kept]");
    } else if (t == "resource") {
      const Json& res = c["resource"];
      if (res["text"].is_string()) add("[" + res["uri"].str() + "]\n" + res["text"].str());
      else add("[a binary resource: " + res["uri"].str() + "]");
    } else if (t == "resource_link") {
      add("[resource: " + c["uri"].str() + (c["name"].is_string() ? " (" + c["name"].str() + ")" : "") + "]");
    }
  }
  if (out.empty() && r.has("structuredContent")) out = r["structuredContent"].dump();
  if (out.empty()) out = "(no output)";
  if (out.size() > kMaxResultChars)
    out = out.substr(0, kMaxResultChars) + "\n[cut: " + std::to_string(out.size()) + " characters in all]";
  return out;
}

inline Json server_json(const Config& c, const std::vector<std::string>& on) {
  Json s = Json::object(), a = Json::array(), ts = Json::array();
  s.set("name", c.name);
  s.set("command", c.command);
  for (auto& x : c.args) a.push(x);
  s.set("args", a);
  s.set("enabled", std::find(on.begin(), on.end(), c.name) != on.end());
  std::shared_ptr<Client> cl;
  {
    std::lock_guard<std::mutex> lk(manager().mu);
    auto it = manager().clients.find(c.name);
    if (it != manager().clients.end() && it->second->cfg().key() == c.key()) cl = it->second;
  }
  s.set("running", cl && cl->running());
  if (cl && !cl->start_error().empty() && !cl->running()) s.set("error", cl->start_error());
  if (cl && cl->running())
    for (const Tool& t : cl->tools()) ts.push(tool_json(c.name, t));
  s.set("tools", ts);
  return s;
}
inline Json list(const std::string& project) {
  std::string err;
  const auto cs = configs(err);
  const auto on = enabled(project);
  Json r = Json::object(), ss = Json::array();
  r.set("config_path", config_path().string());
  if (!err.empty()) r.set("error", err);
  for (const Config& c : cs) ss.push(server_json(c, on));
  r.set("servers", ss);
  return r;
}

// project: "" for plain chats, else a known project's id (checked by the caller); cancelled: the client went away
inline void api(const HttpRequest& rq, HttpResponse& rs, const std::string& project, const std::function<bool()>& cancelled) {
  auto fail = [&](int code, const std::string& m) {
    Json e = Json::object(), r = Json::object();
    e.set("message", m);
    r.set("error", e);
    return rs.send(code, "application/json", r.dump());
  };
  try {
    if (rq.path == "/v1/mcp" && rq.method == "GET") return rs.send(200, "application/json", list(project).dump());
    if (rq.path == "/v1/mcp/config" && rq.method == "GET") {
      std::ifstream f(config_path(), std::ios::binary);
      std::stringstream ss;
      ss << f.rdbuf();
      Json r = Json::object();
      r.set("path", config_path().string());
      r.set("text", ss.str());
      return rs.send(200, "application/json", r.dump());
    }
    const Json in = rq.body.empty() ? Json::object() : Json::parse(rq.body);
    if (rq.path == "/v1/mcp/config" && rq.method == "PUT") {
      const std::string text = in["text"].str();
      Json parsed;
      try {
        parsed = Json::parse(text);
      } catch (const std::exception& e) {
        return fail(400, std::string("not valid JSON: ") + e.what());
      }
      if (!parsed.is_object() || (!parsed["mcpServers"].is_object() && !parsed["servers"].is_object()))
        return fail(400, "mcp.json needs an object \"mcpServers\": { \"name\": { \"command\": ..., \"args\": [...] } }");
      const fs::path tmp = config_path().string() + ".tmp";
      {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << text;
        if (!f) return fail(500, "cannot write " + tmp.string());
      }
      std::error_code ec;
      fs::rename(tmp, config_path(), ec);
      if (ec) return fail(500, "cannot write " + config_path().string() + ": " + ec.message());
      // (servers no longer in the file, or changed: stopped now rather than at their next use)
      std::string err;
      const auto cs = configs(err);
      std::vector<std::shared_ptr<Client>> gone;
      {
        std::lock_guard<std::mutex> lk(manager().mu);
        for (auto it = manager().clients.begin(); it != manager().clients.end();) {
          bool keep = false;
          for (const Config& c : cs) keep |= c.name == it->first && c.key() == it->second->cfg().key();
          if (keep) ++it;
          else gone.push_back(it->second), it = manager().clients.erase(it);
        }
      }
      for (auto& c : gone) c->stop();
      return rs.send(200, "application/json", list(project).dump());
    }
    if (rq.path == "/v1/mcp/enabled" && rq.method == "PUT") {
      const std::string name = in["server"].str();
      if (name.empty()) return fail(400, "server: a name from mcp.json");
      set_enabled(project, name, in["on"].as_bool());
      return rs.send(200, "application/json", list(project).dump());
    }
    if (rq.method != "POST") return fail(405, "method not allowed");
    if (rq.path == "/v1/mcp/start" || rq.path == "/v1/mcp/stop") {
      std::string err;
      auto cl = client(in["server"].str(), err);
      if (!cl) return fail(404, err);
      if (rq.path == "/v1/mcp/stop") cl->stop();
      else
        try {
          cl->ensure_started();
        } catch (const std::exception& e) {
          return fail(502, e.what());
        }
      return rs.send(200, "application/json", list(project).dump());
    }
    if (rq.path == "/v1/mcp/tools") {  // (the switched-on servers' tools, each started if it is not running)
      std::string err;
      const auto cs = configs(err);
      const auto on = enabled(project);
      std::vector<std::shared_ptr<Client>> cls;
      for (const Config& c : cs)
        if (std::find(on.begin(), on.end(), c.name) != on.end())
          if (auto cl = client(c.name, err)) cls.push_back(cl);
      std::vector<std::string> errs(cls.size());
      std::vector<std::thread> th;
      for (size_t i = 0; i < cls.size(); ++i)
        th.emplace_back([&, i] {
          try { cls[i]->ensure_started(); } catch (const std::exception& e) { errs[i] = e.what(); }
        });
      for (auto& t : th) t.join();
      Json r = Json::object(), ts = Json::array(), es = Json::array();
      std::string instr;
      for (size_t i = 0; i < cls.size(); ++i) {
        const std::string& n = cls[i]->cfg().name;
        if (!errs[i].empty()) {
          Json e = Json::object();
          e.set("server", n);
          e.set("message", errs[i]);
          es.push(e);
          continue;
        }
        for (const Tool& t : cls[i]->tools()) ts.push(tool_json(n, t));
        std::string s = cls[i]->instructions();
        if (s.size() > 2000) s = s.substr(0, 2000) + "...";
        if (!s.empty()) instr += "\n" + n + " (MCP server): " + s;
      }
      r.set("tools", ts);
      r.set("errors", es);
      r.set("instructions", instr);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path == "/v1/mcp/call") {
      std::string err;
      auto cl = client(in["server"].str(), err);
      if (!cl) return fail(404, err);
      const std::string chat = files::valid_id(in["chat_id"].str()) ? in["chat_id"].str() : "";
      const std::string tool = in["tool"].str();
      Json out = Json::object(), fl = Json::array();
      Json res;
      try {
        res = cl->call(tool, in["arguments"], cancelled);
      } catch (const std::exception& e) {
        return fail(502, e.what());
      }
      int nimg = 0;
      auto save = [&](const std::string& data, const std::string& mime) -> std::string {
        const std::string ext = mime.find("jpeg") != std::string::npos || mime.find("jpg") != std::string::npos ? ".jpg"
                                : mime.find("webp") != std::string::npos ? ".webp" : mime.find("gif") != std::string::npos ? ".gif" : ".png";
        files::Info fi = files::store().put(files::safe_name(cl->cfg().name + " " + tool + (nimg ? " " + std::to_string(nimg + 1) : "") + ext), data, chat);
        ++nimg;
        fl.push(files::info_json(fi));
        return "[an image from " + tool + ": saved in this chat's files as file_id \"" + fi.id + "\"]";
      };
      out.set("text", result_text(res, save));
      out.set("is_error", res["isError"].as_bool());
      out.set("files", fl);
      return rs.send(200, "application/json", out.dump());
    }
    return fail(404, "no such MCP endpoint");
  } catch (const std::exception& e) {
    return fail(400, e.what());
  }
}

}  // namespace mcp

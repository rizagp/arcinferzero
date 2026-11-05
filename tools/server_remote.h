// Remote access (docs/remote-access.md): the web UI over the internet through a tunnel (Cloudflare Tunnel: cloudflared
// on this computer forwards https://<name> to http://127.0.0.1:<port>), with a password.
//
// A request is remote when its Host header is one of the names in remote.json (data folder; a "*." name matches its
// subdomains, as *.trycloudflare.com for a quick tunnel) and remote access is on. Without that the server answers only
// requests for localhost / 127.0.0.1 (or its address with --host), as before.
//   - Every remote request needs a session: a cookie set by POST /v1/remote/login with the password (HttpOnly, Secure,
//     SameSite=Strict; 30 days, or until logout, a password change, access turned off or its host names changed).
//     Without one, a page gets the login page and an API call 401; anything else (a body) is refused before it is
//     read (head_gate). Sessions live in this process only: a restarted server asks for the password again.
//   - Origin: a remote page may only send its own (https://<name>); with SameSite=Strict that rules out other sites.
//   - Without "allow_tools" (Settings: Full access remotely) a remote session is not "this computer": rq.local_peer
//     is taken away, so every local-only check of the handlers refuses it (projects and their files, memory, MCP,
//     slots, loading models, the OCR save into a folder), and blocked() fences the same paths and run_command again.
//     The remote settings themselves are always local only.
//   - Wrong passwords: one second's wait each; 8 within 15 minutes lock that visitor (CF-Connecting-IP) out for 15
//     minutes, 40 from all visitors lock every sign-in; attempts being checked count until they are done.
//   - A request through a tunnel or proxy (its forwarding headers) for any other host is refused: a tunnel that
//     rewrote the Host header to 127.0.0.1 would otherwise pass as local, without a password.
// The password is kept as PBKDF2-HMAC-SHA256 (200,000 rounds, 16-byte random salt; Windows CNG, bcrypt.lib).
#pragma once
#include "../src/app_dir.h"
#include "../src/http.h"
#include "../src/json.h"
#include <bcrypt.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#pragma comment(lib, "bcrypt.lib")

namespace remote {
namespace fs = std::filesystem;

inline std::string hex(const unsigned char* p, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s(n * 2, '0');
  for (size_t i = 0; i < n; ++i) s[2 * i] = d[p[i] >> 4], s[2 * i + 1] = d[p[i] & 15];
  return s;
}
inline std::vector<unsigned char> unhex(const std::string& s) {
  std::vector<unsigned char> v;
  auto nib = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; };
  if (s.size() % 2) return v;
  for (size_t i = 0; i < s.size(); i += 2) {
    const int a = nib(s[i]), b = nib(s[i + 1]);
    if (a < 0 || b < 0) return {};
    v.push_back((unsigned char)(a * 16 + b));
  }
  return v;
}
inline std::string random_hex(size_t bytes) {
  std::vector<unsigned char> b(bytes);
  if (BCryptGenRandom(nullptr, b.data(), (ULONG)b.size(), BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0)
    throw std::runtime_error("no random numbers from Windows (BCryptGenRandom)");
  return hex(b.data(), b.size());
}
inline std::string pbkdf2(const std::string& password, const std::vector<unsigned char>& salt, unsigned iters) {
  BCRYPT_ALG_HANDLE alg = nullptr;
  if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, BCRYPT_ALG_HANDLE_HMAC_FLAG) != 0)
    throw std::runtime_error("no SHA-256 from Windows (BCryptOpenAlgorithmProvider)");
  unsigned char out[32];
  const long r = BCryptDeriveKeyPBKDF2(alg, (PUCHAR)password.data(), (ULONG)password.size(), (PUCHAR)salt.data(),
                                       (ULONG)salt.size(), iters, out, sizeof out, 0);
  BCryptCloseAlgorithmProvider(alg, 0);
  if (r != 0) throw std::runtime_error("PBKDF2 failed (BCryptDeriveKeyPBKDF2)");
  return hex(out, sizeof out);
}
inline bool equal_ct(const std::string& a, const std::string& b) {  // (constant time for equal lengths)
  if (a.size() != b.size()) return false;
  unsigned char d = 0;
  for (size_t i = 0; i < a.size(); ++i) d |= (unsigned char)(a[i] ^ b[i]);
  return d == 0;
}
inline std::string lower(std::string s) {
  for (char& c : s) c = (char)tolower((unsigned char)c);
  return s;
}

struct Access {
  std::mutex mu;
  bool enabled = false, allow_tools = false;
  std::vector<std::string> hosts;  // lower case, no port
  std::string salt, hash;          // hex
  unsigned iters = 200000;
  struct Session { std::chrono::system_clock::time_point until; };
  std::map<std::string, Session> sessions;  // token -> expiry
  static constexpr int kDays = 30;

  static fs::path path() { return fs::path(app_dir()) / "remote.json"; }
  void load() {
    std::lock_guard<std::mutex> g(mu);
    std::ifstream f(path(), std::ios::binary);
    if (!f) return;
    std::stringstream ss;
    ss << f.rdbuf();
    Json j;
    if (!Json::try_parse(ss.str(), j) || !j.is_object()) return;
    enabled = j["enabled"].as_bool(false);
    allow_tools = j["allow_tools"].as_bool(false);
    hosts.clear();
    for (const Json& h : j["hosts"].items()) {
      const std::string s = clean_host(h.str());
      if (!s.empty()) hosts.push_back(s);
    }
    salt = j["salt"].str();
    hash = j["hash"].str();
    iters = (unsigned)j["iterations"].as_clamped(200000, 10000, 10000000);
  }
  bool save_locked() {
    Json j = Json::object();
    j.set("enabled", enabled);
    j.set("allow_tools", allow_tools);
    Json h = Json::array();
    for (auto& s : hosts) h.push(s);
    j.set("hosts", h);
    j.set("salt", salt);
    j.set("hash", hash);
    j.set("iterations", (int)iters);
    std::error_code ec;
    fs::create_directories(path().parent_path(), ec);
    const fs::path tmp = path().string() + ".tmp";
    {
      std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
      if (!f) return false;
      f << j.dump(true);
      if (!f) return false;
    }
    fs::rename(tmp, path(), ec);
    return !ec;
  }
  // a host name as configured: lower case, without scheme, path or port; "" when it is not a plausible DNS name
  static std::string clean_host(std::string s) {
    s = lower(s);
    for (const char* pre : {"https://", "http://"})
      if (s.rfind(pre, 0) == 0) s = s.substr(strlen(pre));
    s = s.substr(0, s.find('/'));
    s = s.substr(0, s.find(':'));
    if (s.empty() || s.size() > 253) return "";
    const bool wild = s.rfind("*.", 0) == 0;
    const std::string body = wild ? s.substr(2) : s;
    if (body.find('.') == std::string::npos) return "";  // (a bare name or "*.com": too broad)
    for (char c : body)
      if (!(isalnum((unsigned char)c) || c == '-' || c == '.')) return "";
    return s;
  }
  // the Host header of a request names one of the remote hosts
  bool is_remote_host(const std::string& host_header) {
    std::lock_guard<std::mutex> g(mu);
    if (!enabled || hash.empty()) return false;
    const std::string h = lower(host_header.substr(0, host_header.find(':')));
    for (const std::string& c : hosts) {
      if (c.rfind("*.", 0) == 0) {
        const std::string suf = c.substr(1);  // ".trycloudflare.com"
        if (h.size() > suf.size() && h.compare(h.size() - suf.size(), suf.size(), suf) == 0) return true;
      } else if (h == c) {
        return true;
      }
    }
    return false;
  }
  static std::string cookie_token(const HttpRequest& rq) {
    const std::string c = rq.header("cookie");
    for (size_t i = 0; i < c.size();) {
      size_t e = c.find(';', i);
      if (e == std::string::npos) e = c.size();
      std::string kv = c.substr(i, e - i);
      const size_t a = kv.find_first_not_of(' ');
      kv = a == std::string::npos ? "" : kv.substr(a);
      if (kv.rfind("azr_session=", 0) == 0) return kv.substr(12);
      i = e + 1;
    }
    return "";
  }
  bool session_ok(const HttpRequest& rq) {
    const std::string t = cookie_token(rq);
    if (t.size() != 64) return false;
    std::lock_guard<std::mutex> g(mu);
    auto it = sessions.find(t);
    if (it == sessions.end()) return false;
    if (std::chrono::system_clock::now() > it->second.until) {
      sessions.erase(it);
      return false;
    }
    return true;
  }
  bool tools_allowed() {
    std::lock_guard<std::mutex> g(mu);
    return allow_tools;
  }
  // what a remote session may not reach (unless allow_tools)
  bool blocked(const HttpRequest& rq) {
    const std::string& p = rq.path;
    if (p == "/v1/remote/config" || p.rfind("/v1/remote/config", 0) == 0) return true;  // (always local only)
    {
      std::lock_guard<std::mutex> g(mu);
      if (allow_tools) return false;
    }
    // (a second fence: the gate also takes local_peer away, which the handlers of these check already)
    auto under = [&](const char* pre) {
      const size_t n = strlen(pre);
      return p.compare(0, n, pre) == 0 && (p.size() == n || p[n] == '/');
    };
    if (p == "/v1/tools/run_command") return true;
    if (under("/v1/projects")) return true;  // (folder paths, the folder browser, adding folders)
    if (under("/v1/mcp")) return true;       // (mcp.json holds the servers' tokens; /v1/mcp/tools starts programs)
    if (p == "/v1/memory" && rq.method != "GET") return true;
    if (p == "/v1/tools/remember" || p == "/v1/tools/forget") return true;
    if (p == "/v1/transcripts/import") return true;  // (reads a folder of this computer)
    if (p == "/v1/server/load" || p == "/v1/server/unload" || p == "/v1/server/slots") return true;
    return false;
  }

  static std::string login_page() {
    return R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>arcinferzero: sign in</title>
<style>
:root { --bg: #f0f2f5; --card: #fff; --text: #111b21; --muted: #667781; --accent: #00a884; --line: #e9edef; --bad: #d0453f; }
@media (prefers-color-scheme: dark) { :root { --bg: #0b141a; --card: #202c33; --text: #e9edef; --muted: #8696a0; --line: #2a3942; --bad: #f15c6d; } }
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: var(--bg); color: var(--text);
       font: 15px/1.45 "Segoe UI", system-ui, sans-serif; padding: 16px; }
form { width: 100%; max-width: 340px; background: var(--card); border-radius: 14px; padding: 28px 24px;
       box-shadow: 0 8px 30px rgba(0,0,0,.12); animation: rise .35s cubic-bezier(.2,.8,.2,1); }
@keyframes rise { from { opacity: 0; transform: translateY(12px) scale(.98); } }
@keyframes shake { 25% { transform: translateX(-6px); } 75% { transform: translateX(6px); } }
@media (prefers-reduced-motion: reduce) { form { animation: none; } .shake { animation: none !important; } }
h1 { font-size: 20px; margin: 0 0 4px; } p { margin: 0 0 18px; color: var(--muted); font-size: 13px; }
input { width: 100%; font: inherit; padding: 11px 12px; border: 1px solid var(--line); border-radius: 9px; background: transparent; color: inherit; }
input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent); }
button { margin-top: 14px; width: 100%; font: inherit; font-weight: 600; padding: 11px; border: 0; border-radius: 9px;
         background: var(--accent); color: #fff; cursor: pointer; transition: filter .15s, transform .1s; }
button:hover { filter: brightness(1.06); } button:active { transform: scale(.98); } button:disabled { opacity: .6; cursor: default; }
.err { color: var(--bad); font-size: 13px; min-height: 18px; margin-top: 10px; }
.shake { animation: shake .3s; }
</style></head><body>
<form id="f"><h1>arcinferzero</h1><p>This server is private. Enter its password to continue.</p>
<input id="pw" type="password" autocomplete="current-password" placeholder="Password" aria-label="Password" autofocus required>
<button id="go" type="submit">Sign in</button><div class="err" id="err" role="alert"></div></form>
<script>
const f = document.getElementById("f"), pw = document.getElementById("pw"), go = document.getElementById("go"), err = document.getElementById("err");
f.addEventListener("submit", async (e) => {
  e.preventDefault(); go.disabled = true; err.textContent = "";
  try {
    const r = await fetch("/v1/remote/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ password: pw.value }) });
    if (r.ok) { location.replace("/"); return; }
    const j = await r.json().catch(() => ({}));
    err.textContent = (j.error && j.error.message) || "Sign-in failed";
  } catch { err.textContent = "The server did not answer"; }
  go.disabled = false; pw.select(); f.classList.remove("shake"); void f.offsetWidth; f.classList.add("shake");
});
</script></body></html>)HTML";
  }
  static void deny(HttpResponse& rs, int status, const std::string& msg) {
    Json e = Json::object();
    e.set("message", msg);
    e.set("type", "remote_access");
    Json j = Json::object();
    j.set("error", e);
    rs.send(status, "application/json", j.dump());
  }
  // POST /v1/remote/login {password}
  // Throttle: per visitor (Cloudflare's CF-Connecting-IP; "" without it) 8 wrong passwords in 15 minutes lock that
  // visitor out for 15 minutes; all visitors together 40, which locks every sign-in (many addresses at once). Attempts
  // still being checked count as wrong ones until they are not, so parallel guesses cannot pass the limit.
  struct Tries {
    std::vector<std::chrono::steady_clock::time_point> fails;
    int inflight = 0;
    std::chrono::steady_clock::time_point locked_until{};
  };
  std::map<std::string, Tries> tries;  // visitor -> its tries
  Tries all_tries;
  static constexpr int kPerVisitor = 8, kAll = 40;
  static void prune(Tries& t, std::chrono::steady_clock::time_point now) {
    t.fails.erase(std::remove_if(t.fails.begin(), t.fails.end(), [&](auto x) { return now - x > std::chrono::minutes(15); }),
                  t.fails.end());
  }
  void login(const HttpRequest& rq, HttpResponse& rs) {
    const auto now = std::chrono::steady_clock::now();
    const std::string who = rq.header("cf-connecting-ip").substr(0, 64);
    {
      std::lock_guard<std::mutex> g(mu);
      if (tries.size() > 10000) tries.clear();  // (memory: a flood of addresses; the global limit still holds)
      Tries& t = tries[who];
      prune(t, now);
      prune(all_tries, now);
      if (now < t.locked_until || now < all_tries.locked_until ||
          (int)t.fails.size() + t.inflight >= kPerVisitor || (int)all_tries.fails.size() + all_tries.inflight >= kAll)
        return deny(rs, 429, "too many wrong passwords: try again in 15 minutes");
      ++t.inflight, ++all_tries.inflight;
    }
    bool ok = false;
    try {
      Json j;
      if (Json::try_parse(rq.body, j)) {
        const std::string pw = j["password"].str();
        std::string salt_c, hash_c;
        unsigned it;
        {
          std::lock_guard<std::mutex> g(mu);
          salt_c = salt, hash_c = hash, it = iters;
        }
        ok = !hash_c.empty() && !pw.empty() && pw.size() <= 1024 && equal_ct(pbkdf2(pw, unhex(salt_c), it), hash_c);
      }
      if (!ok) std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (...) {
      std::lock_guard<std::mutex> g(mu);
      --tries[who].inflight, --all_tries.inflight;
      throw;
    }
    {
      std::lock_guard<std::mutex> g(mu);
      Tries& t = tries[who];
      --t.inflight, --all_tries.inflight;
      if (!ok) {
        t.fails.push_back(now);
        all_tries.fails.push_back(now);
        if ((int)t.fails.size() >= kPerVisitor) t.locked_until = now + std::chrono::minutes(15), t.fails.clear();
        if ((int)all_tries.fails.size() >= kAll) all_tries.locked_until = now + std::chrono::minutes(15), all_tries.fails.clear();
      } else {
        t.fails.clear();
      }
    }
    if (!ok) return deny(rs, 401, "wrong password");
    const std::string tok = random_hex(32);
    {
      std::lock_guard<std::mutex> g(mu);
      const auto t = std::chrono::system_clock::now();
      for (auto i = sessions.begin(); i != sessions.end();) i = t > i->second.until ? sessions.erase(i) : std::next(i);
      sessions[tok] = {t + std::chrono::hours(24 * kDays)};
    }
    rs.extra += "Set-Cookie: azr_session=" + tok + "; Path=/; Max-Age=" + std::to_string(kDays * 86400) +
                "; HttpOnly; Secure; SameSite=Strict\r\n";
    rs.send(200, "application/json", "{\"ok\":true}");
  }
  void logout(const HttpRequest& rq, HttpResponse& rs) {
    {
      std::lock_guard<std::mutex> g(mu);
      sessions.erase(cookie_token(rq));
    }
    rs.extra += "Set-Cookie: azr_session=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Strict\r\n";
    rs.send(200, "application/json", "{\"ok\":true}");
  }
  Json config_json() {
    std::lock_guard<std::mutex> g(mu);
    Json j = Json::object();
    j.set("enabled", enabled);
    j.set("allow_tools", allow_tools);
    Json h = Json::array();
    for (auto& s : hosts) h.push(s);
    j.set("hosts", h);
    j.set("password_set", !hash.empty());
    j.set("sessions", (int)sessions.size());
    return j;
  }
  // GET / PUT /v1/remote/config (this computer only): {enabled, allow_tools, hosts, password (to set a new one),
  // sign_out_all}
  void config(const HttpRequest& rq, HttpResponse& rs) {
    if (rq.method == "GET") return rs.send(200, "application/json", config_json().dump());
    if (rq.method != "PUT") return deny(rs, 405, "GET or PUT");
    Json j;
    if (!Json::try_parse(rq.body, j) || !j.is_object()) return deny(rs, 400, "a JSON object is required");
    std::string nsalt, nhash;
    if (j.has("password")) {
      const std::string pw = j["password"].str();
      if (pw.size() < 10) return deny(rs, 400, "the password needs at least 10 characters");
      if (pw.size() > 1024) return deny(rs, 400, "the password is too long");
      nsalt = random_hex(16);
      nhash = pbkdf2(pw, unhex(nsalt), 200000);
    }
    std::vector<std::string> nh;
    bool set_hosts = false;
    if (j.has("hosts")) {
      set_hosts = true;
      for (const Json& h : j["hosts"].items()) {
        const std::string s = clean_host(h.str());
        if (s.empty()) return deny(rs, 400, "not a host name: " + h.str());
        nh.push_back(s);
      }
    }
    {
      std::lock_guard<std::mutex> g(mu);
      // (sessions end when access goes off, the host names change or a session would get more than it signed in for)
      const bool was_on = enabled, had_tools = allow_tools;
      const std::vector<std::string> old_hosts = hosts;
      if (j.has("enabled")) enabled = j["enabled"].as_bool(false);
      if (j.has("allow_tools")) allow_tools = j["allow_tools"].as_bool(false);
      if (set_hosts) hosts = nh;
      if ((was_on && !enabled) || hosts != old_hosts || (!had_tools && allow_tools)) sessions.clear();
      if (!nhash.empty()) salt = nsalt, hash = nhash, iters = 200000, sessions.clear();  // (a new password signs everyone out)
      if (j["sign_out_all"].as_bool(false)) sessions.clear();
      if (enabled && hash.empty()) enabled = false;  // (never on without a password)
      if (!save_locked()) return deny(rs, 500, "could not write remote.json");
    }
    rs.send(200, "application/json", config_json().dump());
  }

  // The gate (HttpServer::gate): 0 = not a remote request (the usual checks), 1 = answered here, 2 = a remote
  // request that may go on to the handler (rq.remote set).
  int gate(HttpRequest& rq, HttpResponse& rs) {
    if (!is_remote_host(rq.header("host"))) {
      // (a request that came through a tunnel or proxy but names another host, e.g. one set to rewrite the Host header
      //  to 127.0.0.1: it would pass as a local request without a password, so it is refused. Cloudflare's headers,
      //  and the forwarding headers other proxies add)
      if (proxied(rq)) {
        deny(rs, 403, "this request came through a tunnel or proxy for a host name that is not in the remote access "
                      "settings (the tunnel must not rewrite the Host header; only Cloudflare Tunnel is supported)");
        return 1;
      }
      return 0;
    }
    rq.remote = true;
    const std::string origin = rq.header("origin");
    if (!origin.empty() && lower(origin) != "https://" + host_only(rq.header("host"))) {
      deny(rs, 403, "cross-origin requests are not allowed");
      return 1;
    }
    if (rq.path == "/v1/remote/login" && rq.method == "POST") { login(rq, rs); return 1; }
    if (rq.path == "/v1/remote/logout" && rq.method == "POST") { logout(rq, rs); return 1; }
    // (the app manifest and its icon: a browser fetches the manifest without the cookie)
    if (rq.method == "GET" && (rq.path == "/manifest.webmanifest" || rq.path == "/icon.svg")) {
      rq.local_peer = false;
      return 2;
    }
    if (!session_ok(rq)) {
      const bool page = rq.method == "GET" && rq.path.rfind("/v1/", 0) != 0 && rq.path != "/health";
      if (page) rs.send(200, "text/html; charset=utf-8", login_page());
      else deny(rs, 401, "sign in first");
      return 1;
    }
    if (blocked(rq)) {
      deny(rs, 403, "not available over remote access (Settings > Remote access > Full access remotely, on this "
                    "computer)");
      return 1;
    }
    // (without full access a remote session is not "this computer": every handler's own local-only check refuses it
    //  too, the OCR save into a folder, project files, memory, slots, loading models)
    if (!tools_allowed()) rq.local_peer = false;
    return 2;
  }
  static std::string host_only(const std::string& h) { return lower(h.substr(0, h.find(':'))); }
  static bool proxied(const HttpRequest& rq) {
    for (const char* k : {"cf-connecting-ip", "cf-ray", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto",
                          "forwarded", "cdn-loop", "x-real-ip"})
      if (!rq.header(k).empty()) return true;
    return false;
  }
  // Before the body is read (HttpServer::head_gate): a remote request without a session may only fetch a page (the
  // sign-in page) or send a small sign-in; anything else is refused at once, so the internet cannot fill the
  // connections or the memory with bodies before the password. "" = go on.
  std::string head_gate(const HttpRequest& rq) {
    if (!is_remote_host(rq.header("host"))) return "";
    if (session_ok(rq)) return "";
    if (rq.method == "GET" || rq.method == "HEAD") return rq.header("content-length").empty() && rq.header("transfer-encoding").empty() ? "" : "no body";
    if (rq.method == "POST" && (rq.path == "/v1/remote/login" || rq.path == "/v1/remote/logout")) {
      size_t n = 0;
      const std::string cl = rq.header("content-length");
      if (!rq.header("transfer-encoding").empty()) return "sign-in with a Content-Length";
      if (!cl.empty() && (cl.size() > 6 || (n = (size_t)strtoul(cl.c_str(), nullptr, 10)) > 4096)) return "sign-in too large";
      return "";
    }
    return "sign in first";
  }
};

inline Access& access() {
  static Access a;
  return a;
}
}  // namespace remote

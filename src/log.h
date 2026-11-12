// The log (docs/logging.md): every program's messages on stderr go through here (zlog::eprintf in place of
// fprintf(stderr, ...), and LOG_ERR / LOG_WRN / LOG_INF / LOG_DBG). Force-included in every file (build.ps1 /FI).
//
// Off (the default: runz and the tools) a message is written as fprintf(stderr) wrote it, byte for byte (the bench
// scripts parse runz's output); LOG_DBG is dropped. On (the server, zlog::start) each line is written whole, with a
// prefix, under one lock:
//   2026-11-12 14:13:05.123 I listening on http://127.0.0.1:8090
// (local time; I info, W warning, E error, D debug). A message printed in parts by one thread (fprintf without "\n",
// then the rest) is kept until its "\n", so another thread's line no longer lands inside it. A line starting with
// "warning" / "error" (any case) from an old call is a W / E. zlog::raw writes a line without the prefix: lines that
// programs read (progress: P % ..., zero-server.ps1 and the launcher).
// Switches (the server's options): --log-verbosity N / -lv N (0 errors, 1 + warnings, 2 + info (default), 3 + debug;
// also ARC_LOG_LEVEL), --log-timestamps on|off, --log-colors auto|on|off, --log-file F (also to F), --log-disable.
#pragma once
#include <cctype>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <io.h>
#include <mutex>
#include <string>

namespace zlog {

enum Level { ERR = 0, WRN = 1, INF = 2, DBG = 3 };

struct Config {
  bool on = false;       // (prefixes, whole lines; the server)
  int level = INF;       // (lines above it are dropped)
  bool stamps = true;    // (the date and time in front)
  bool colors = false;   // (ANSI colors for W / E / D on a console)
  bool disabled = false; // (--log-disable: nothing at all; a crash report still is)
  FILE* file = nullptr;  // (--log-file: every line there too)
};
inline Config& cfg() {
  static Config c;
  return c;
}
inline std::mutex& mu() {  // (never destroyed: a thread that logs while the process ends still finds it)
  static std::mutex* m = new std::mutex;
  return *m;
}

inline void stamp(char* b, size_t n) {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  const int ms = (int)(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);
  std::tm tm{};
  localtime_s(&tm, &t);
  snprintf(b, n, "%04d-%02d-%02d %02d:%02d:%02d.%03d ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour,
           tm.tm_min, tm.tm_sec, ms);
}

// one finished line (without its "\n"), with the prefix. Never throws: a lock that cannot be taken writes the line
// without it (a torn line rather than a lost one)
inline void emit(int lvl, const char* s, size_t n) noexcept {
  Config& c = cfg();
  if (c.disabled || lvl > c.level) return;
  char pre[48] = "";
  if (c.stamps) stamp(pre, sizeof pre);
  static const char kL[4] = {'E', 'W', 'I', 'D'};
  const size_t pl = strlen(pre);
  snprintf(pre + pl, sizeof pre - pl, "%c ", kL[lvl < 0 ? 0 : lvl > 3 ? 3 : lvl]);
  const char* col = !c.colors ? "" : lvl == ERR ? "\x1b[31m" : lvl == WRN ? "\x1b[33m" : lvl == DBG ? "\x1b[90m" : "";
  // (control bytes but the tab shown as '?': a CR or an escape in text from a model file or a request rewrote the
  //  line on a console, or wrote console codes; audit 2026-11-13 #6. Copied only when there is one)
  std::string clean;
  for (size_t i = 0; i < n; ++i) {
    const unsigned char ch = (unsigned char)s[i];
    if ((ch < 0x20 && ch != '\t') || ch == 0x7f) {
      if (clean.empty()) try { clean.assign(s, n); } catch (...) { break; }
      clean[i] = '?';
    }
  }
  if (!clean.empty()) s = clean.c_str();
  const int len = n > (size_t)INT32_MAX ? INT32_MAX : (int)n;
  bool locked = false;
  try {
    mu().lock();
    locked = true;
  } catch (...) {}
  fprintf(stderr, "%s%s%.*s%s\n", col, pre, len, s, *col ? "\x1b[0m" : "");
  fflush(stderr);
  if (c.file) {
    fprintf(c.file, "%s%.*s\n", pre, len, s);
    fflush(c.file);
  }
  if (locked) mu().unlock();
}
// an old call's line: E / W when it says so at its start, after a tag such as "inferzero: " or "whisper: "
inline int guess(const char* s, size_t n) noexcept {
  size_t k = 0;  // (past a tag: up to 24 letters, digits, '-', '_', ' ' or '.', then ": ")
  for (size_t i = 0; i < n && i < 26; ++i) {
    const char ch = s[i];
    if (ch == ':' && i + 1 < n && s[i + 1] == ' ') { k = i + 2; break; }
    if (!(isalnum((unsigned char)ch) || ch == '-' || ch == '_' || ch == ' ' || ch == '.')) break;
  }
  auto starts = [&](size_t at, const char* w) {
    const size_t m = strlen(w);
    if (n < at + m) return false;
    for (size_t i = 0; i < m; ++i)
      if (tolower((unsigned char)s[at + i]) != w[i]) return false;
    return true;
  };
  auto has = [&](const char* w, size_t within) {  // (w in the first `within` characters, any case)
    const size_t m = strlen(w), lim = n < within ? n : within;
    for (size_t i = 0; i + m <= lim; ++i)
      if (starts(i, w)) return true;
    return false;
  };
  // ("Level Zero error 0x...", "... the GPU ran out of memory ...", "device lost": errors wherever they stand)
  if (has(" error 0x", 48) || has("out of memory", n) || has("device lost", n) || has("device_lost", n)) return ERR;
  for (size_t at : {(size_t)0, k}) {
    if (starts(at, "error") || starts(at, "fatal") || starts(at, "crashed")) return ERR;
    if (starts(at, "warning") || starts(at, "failed") || starts(at, "cannot") || starts(at, "could not") ||
        starts(at, "not found"))
      return WRN;
  }
  return INF;
}
// a thread's unfinished line when the thread ends (else it would be lost)
struct Pending {
  std::string s;
  ~Pending() {
    if (!s.empty() && cfg().on) emit(guess(s.data(), s.size()), s.data(), s.size());
  }
};
inline std::string& pending_line() {
  thread_local Pending p;
  return p.s;
}
inline void vlog(int lvl, const char* fmt, va_list ap) noexcept {
  Config& c = cfg();
  if (!c.on) {  // (as fprintf(stderr) did)
    if (lvl == DBG) return;
    vfprintf(stderr, fmt, ap);
    return;
  }
  va_list a2, a3;
  va_copy(a2, ap);
  va_copy(a3, ap);
  try {
    char small[1024];
    const int n = vsnprintf(small, sizeof small, fmt, ap);
    std::string& p = pending_line();
    if (n >= 0) {
      if ((size_t)n < sizeof small) {
        p.append(small, (size_t)n);
      } else {
        const size_t at = p.size();
        p.resize(at + (size_t)n + 1);
        vsnprintf(&p[at], (size_t)n + 1, fmt, a2);
        p.resize(at + (size_t)n);
      }
      size_t from = 0;
      for (size_t nl; (nl = p.find('\n', from)) != std::string::npos; from = nl + 1) {
        size_t e = nl;
        if (e > from && p[e - 1] == '\r') --e;
        emit(lvl < 0 ? guess(p.data() + from, e - from) : lvl, p.data() + from, e - from);
      }
      p.erase(0, from);
      if (p.size() > 65536) {  // (a line that never ends: written in pieces rather than held)
        emit(lvl < 0 ? guess(p.data(), p.size()) : lvl, p.data(), p.size());
        p.clear();
      }
    }
  } catch (...) {  // (out of memory: the message as fprintf(stderr) wrote it)
    vfprintf(stderr, fmt, a3);
  }
  va_end(a3);
  va_end(a2);
}
// fprintf(stderr, ...)'s replacement (the old calls): info, or W / E by its start
inline void eprintf(const char* fmt, ...) noexcept {
  va_list ap;
  va_start(ap, fmt);
  vlog(-1, fmt, ap);
  va_end(ap);
}
inline void log(int lvl, const char* fmt, ...) noexcept {
  va_list ap;
  va_start(ap, fmt);
  vlog(lvl, fmt, ap);
  va_end(ap);
}
// a line for programs to read: no prefix, never filtered (unless --log-disable)
inline void raw(const char* fmt, ...) noexcept {
  Config& c = cfg();
  if (c.disabled && c.on) return;
  va_list ap, a2;
  va_start(ap, fmt);
  va_copy(a2, ap);
  bool locked = false;
  if (c.on) try {
      mu().lock();
      locked = true;
    } catch (...) {}
  vfprintf(stderr, fmt, ap);
  fflush(stderr);
  if (c.on && c.file) {
    vfprintf(c.file, fmt, a2);
    fflush(c.file);
  }
  if (locked) mu().unlock();
  va_end(a2);
  va_end(ap);
}
inline bool level_name(const std::string& s, int& out) {
  if (s.size() == 1 && s[0] >= '0' && s[0] <= '3') { out = s[0] - '0'; return true; }
  static const char* names[4] = {"error", "warn", "info", "debug"};
  for (int i = 0; i < 4; ++i)
    if (s == names[i]) { out = i; return true; }
  return false;
}
// the server's start: prefixes on; ARC_LOG_LEVEL (0-3 or error / warn / info / debug) read here, the options after
inline void start() {
  Config& c = cfg();
  c.on = true;
  if (const char* e = getenv("ARC_LOG_LEVEL")) {
    int l = INF;
    if (level_name(e, l)) c.level = l;
  }
  // (colors only on a terminal that shows them: Windows Terminal sets WT_SESSION; a redirected log never)
  c.colors = getenv("WT_SESSION") && _isatty(_fileno(stderr));
}

}  // namespace zlog

// (the fprintf that never runs: MSVC checks a printf format only for the CRT's functions (C4477), so each call is also
//  written as one for the check; zlog::eprintf, the old calls, kept their arguments as they were)
#define ZLOG_CHECKED_(lvl, ...) ((void)(false && fprintf(stderr, __VA_ARGS__)), zlog::log((lvl), __VA_ARGS__))
#define LOG_ERR(...) ZLOG_CHECKED_(zlog::ERR, __VA_ARGS__)
#define LOG_WRN(...) ZLOG_CHECKED_(zlog::WRN, __VA_ARGS__)
#define LOG_INF(...) ZLOG_CHECKED_(zlog::INF, __VA_ARGS__)
#define LOG_DBG(...) ZLOG_CHECKED_(zlog::DBG, __VA_ARGS__)

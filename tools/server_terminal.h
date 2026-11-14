// The web UI's terminal tool: one PowerShell command for the model, run in its chat's folder after the user allowed it
// in the page (Settings > Terminal: ask each time, or run without asking). Only for connections from this machine
// (see server.cpp): a server opened to the network with --host never runs a command for another computer.
// The command runs in a job object, so a timeout ends it and every process it started; its output (stdout and stderr
// together) is kept up to kMaxOut: the start and the end of a longer one.
// Linux: bash instead of PowerShell, a process group instead of the job object, the same session state file. (The
// group is ended at a timeout or a cancel, but not when the server itself ends, a crash or a kill: no job object closes
// with it. A command still running then loses its output pipe, so it ends at its next write (SIGPIPE) or runs on to
// its own end; a service manager ends the rest, docs/linux-port.md)
#pragma once
#ifdef _WIN32
#include <windows.h>
#else
#include "../src/json.h"
#include "../src/platform_proc.h"  // (plat::ignore_sigpipe)
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace term {

struct Result {
  int exit_code = -1;
  std::string output;
  bool timed_out = false, truncated = false, cancelled = false;
  double seconds = 0;
};

inline constexpr size_t kMaxOut = 64 * 1024;

// the head (3/4 of kMaxOut) and the tail of a long output, as both systems keep it
struct OutKeep {
  std::string head, tail;
  size_t total = 0;
  void add(const char* b, size_t n) {
    total += n;
    try {
      const size_t room = kMaxOut * 3 / 4 > head.size() ? kMaxOut * 3 / 4 - head.size() : 0;
      head.append(b, std::min<size_t>(room, n));
      if (n > room) {
        tail.append(b + room, n - room);
        if (tail.size() > kMaxOut / 4 * 2) tail.erase(0, tail.size() - kMaxOut / 4);
      }
    } catch (...) {}
  }
  void into(Result& r) {
    if (tail.size() > kMaxOut / 4) tail.erase(0, tail.size() - kMaxOut / 4);
    r.truncated = total > head.size() + tail.size();
    r.output = head + (r.truncated ? "\n[... " + std::to_string(total - head.size() - tail.size()) + " bytes left out ...]\n" : "") + tail;
  }
};

#ifndef _WIN32
// ---------------------------------------------------------------- Linux: bash
// this program's folder (runz, serverz and the other tools next to it)
inline std::string exe_dir() {
  std::error_code ec;
  const std::filesystem::path p = std::filesystem::read_symlink("/proc/self/exe", ec);
  return ec ? std::string(".") : p.parent_path().string();
}
// the environment as name -> value
inline std::map<std::string, std::string> env_map(char** env) {
  std::map<std::string, std::string> m;
  for (char** e = env; e && *e; ++e) {
    const char* eq = strchr(*e, '=');
    if (eq) m[std::string(*e, eq - *e)] = eq + 1;
  }
  return m;
}
// a word in single quotes for bash ('it'\''s')
inline std::string sh_quote(const std::string& s) {
  std::string o = "'";
  for (char c : s) o += c == '\'' ? std::string("'\\''") : std::string(1, c);
  return o + "'";
}
// (variables bash sets itself on every start: not carried in the session's state)
inline bool shell_own(const std::string& k) { return k == "PWD" || k == "OLDPWD" || k == "SHLVL" || k == "_"; }

// command (bash) in cwd; extra_path goes first in PATH (':'-separated); state: the session's file ("": a fresh
// shell), the same JSON as on Windows ({cwd, env: the variables that differ from a fresh shell's, null for one
// removed}): applied to the child's environment and folder here, written back here from what the shell left
// (its folder and `env -0` at its exit, an EXIT trap: also after `exit`). A venv's deactivate is defined again while
// one is active. cancelled: asked every 200 ms, as on Windows.
inline Result run(const std::string& command, const std::string& cwd, const std::string& extra_path, int timeout_s,
                  const std::string& state = "", const std::function<bool()>& cancelled = nullptr) {
  Result r;
  plat::ignore_sigpipe();
  // the fresh shell's environment: the server's, PATH with extra_path first
  std::map<std::string, std::string> base = env_map(environ);
  // (no PATH at all, as under `env -i`: the usual folders, so the shell still finds ls, grep, python3)
  base["PATH"] = extra_path + ":" + (base.count("PATH") ? base["PATH"] : std::string("/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin"));
  // (no locale named, as for a systemd service: UTF-8, as the Windows half sets its output encoding; else tools print
  //  non-ASCII as '?' or octal escapes)
  if (!base.count("LANG") && !base.count("LC_ALL") && !base.count("LC_CTYPE")) base["LC_ALL"] = "C.UTF-8";
  std::map<std::string, std::string> env = base;
  std::string dir = cwd;
  if (!state.empty()) {
    std::ifstream f(state, std::ios::binary);
    const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Json j;
    if (!s.empty() && Json::try_parse(s, j) && j.is_object()) {
      for (const auto& [k, v] : j["env"].entries()) {
        if (v.is_null()) env.erase(k);
        else if (v.is_string()) env[k] = v.str();
      }
      std::error_code ec;
      if (!j["cwd"].str().empty() && std::filesystem::is_directory(j["cwd"].str(), ec)) dir = j["cwd"].str();
    }
  }
  // the script in a file of its own (no length limit, and the command's own stdin stays /dev/null), and the files the
  // EXIT trap writes the folder and the environment to; each one in `gone` as soon as it is made (a later mkstemp that
  // throws left the earlier files behind)
  struct Gone {
    std::vector<std::string> p;
    ~Gone() {
      for (const std::string& s : p) if (!s.empty()) unlink(s.c_str());
    }
  } gone;
  gone.p.reserve(3);  // (no allocation between a file's making and its entry)
  auto tmp = [&gone](const char* name) {
    gone.p.push_back(plat::temp_dir() + "/inferzero-" + name + "-XXXXXX");
    std::string& p = gone.p.back();
    const int fd = mkstemp(p.data());
    if (fd < 0) {
      p.clear();  // (no file made: the name, which mkstemp may have changed, is not removed)
      throw std::runtime_error("cannot make a temporary file");
    }
    ::close(fd);
    return p;
  };
  const std::string script = tmp("cmd"), out_pwd = state.empty() ? "" : tmp("pwd"), out_env = state.empty() ? "" : tmp("env");
  {
    std::string body;
    if (!state.empty())
      // (the two paths in variables: the command may change its own arguments, `set --`, which the trap read before)
      body = "__az_pwd=" + sh_quote(out_pwd) + "; __az_env=" + sh_quote(out_env) + "\n"
             "trap '__az_rc=$?; pwd -P > \"$__az_pwd\" 2>/dev/null; env -0 > \"$__az_env\" 2>/dev/null; exit $__az_rc' EXIT\n"
             "if [ -n \"$VIRTUAL_ENV\" ] && [ -n \"$_OLD_VIRTUAL_PATH\" ]; then deactivate() { export PATH=\"$_OLD_VIRTUAL_PATH\"; "
             "unset VIRTUAL_ENV _OLD_VIRTUAL_PATH VIRTUAL_ENV_PROMPT; unset -f deactivate; }; fi\n";
    body += command + "\n";
    std::ofstream o(script, std::ios::binary);
    o << body;
    if (!o) throw std::runtime_error("cannot write the command's script");
  }
  std::vector<std::string> envs;
  for (const auto& [k, v] : env) envs.push_back(k + "=" + v);
  std::vector<char*> envp;
  for (std::string& e : envs) envp.push_back(e.data());
  envp.push_back(nullptr);
  std::vector<std::string> args = {"bash", "--noprofile", "--norc", script};
  std::vector<char*> argv;
  for (std::string& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);
  {
    std::error_code ec;
    if (!dir.empty() && !std::filesystem::is_directory(dir, ec)) throw std::runtime_error("the terminal's folder is not there: " + dir);
  }
  int po[2];
  if (pipe2(po, O_CLOEXEC) != 0) throw std::runtime_error("cannot make a pipe");
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
  posix_spawn_file_actions_adddup2(&fa, po[1], 1);
  posix_spawn_file_actions_adddup2(&fa, po[1], 2);
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34))
  posix_spawn_file_actions_addclosefrom_np(&fa, 3);  // (only its three: not the server's other files, a --log-file)
#endif
  if (!dir.empty()) posix_spawn_file_actions_addchdir_np(&fa, dir.c_str());
  posix_spawnattr_t at;
  posix_spawnattr_init(&at);
  posix_spawnattr_setflags(&at, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK);
  posix_spawnattr_setpgroup(&at, 0);  // (a group of its own: a timeout ends everything it started, as the job object)
  sigset_t def, none;
  sigemptyset(&def);
  sigaddset(&def, SIGPIPE);
  sigemptyset(&none);
  posix_spawnattr_setsigdefault(&at, &def);
  posix_spawnattr_setsigmask(&at, &none);
  const auto t0 = std::chrono::steady_clock::now();
  pid_t pid = -1;
  const int sp = posix_spawnp(&pid, "bash", &fa, &at, argv.data(), envp.data());
  posix_spawn_file_actions_destroy(&fa);
  posix_spawnattr_destroy(&at);
  ::close(po[1]);
  if (sp != 0) {
    ::close(po[0]);
    throw std::runtime_error(std::string("cannot start bash (") + strerror(sp) + ")");
  }
  fcntl(po[0], F_SETFL, fcntl(po[0], F_GETFL) | O_NONBLOCK);
  OutKeep keep;
  char b[8192];
  // (false: the pipe ended. At most 1 MB a call: a command writing as fast as it is read, `yes`, kept the loop going
  //  for good, the timeout and the cancel never asked; after kill(-pid) too, a process outside the group writing)
  auto drain = [&] {
    for (size_t got = 0; got < ((size_t)1 << 20);) {
      const ssize_t n = ::read(po[0], b, sizeof b);
      if (n > 0) { keep.add(b, (size_t)n); got += (size_t)n; continue; }
      return !(n == 0 || (errno != EAGAIN && errno != EINTR));
    }
    return true;
  };
  const auto deadline = t0 + std::chrono::seconds(timeout_s);
  bool open = true, exited = false, killed = false;
  int st = 0;
  // (the shell's end seen without reaping it, WNOWAIT: its group is killed while it is a zombie, whose id no other
  //  process can take meanwhile; reaped after)
  auto ended = [&] {
    siginfo_t si{};
    return waitid(P_PID, pid, &si, WEXITED | WNOHANG | WNOWAIT) == 0 && si.si_pid == pid;
  };
  while (!exited) {
    pollfd p{po[0], POLLIN, 0};
    if (open && poll(&p, 1, 200) > 0) open = drain();
    if (!open) usleep(20000);
    if ((exited = ended())) break;
    const bool over = std::chrono::steady_clock::now() >= deadline, gone_ = !over && cancelled && cancelled();
    if (!over && !gone_) continue;
    (over ? r.timed_out : r.cancelled) = true;
    killed = true;
    break;
  }
  // what it left running in the background ends with it (its pipe closes), as the job object's end does; then the
  // shell reaped; what is still in the pipe read for up to 2 s
  kill(-pid, SIGKILL);
  while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
  for (int k = 0; k < 20 && open; ++k) {
    pollfd p{po[0], POLLIN, 0};
    if (poll(&p, 1, 100) > 0) open = drain();
  }
  ::close(po[0]);
  r.exit_code = killed ? 1 : WIFEXITED(st) ? WEXITSTATUS(st) : WIFSIGNALED(st) ? 128 + WTERMSIG(st) : -1;
  // the session's state from what the shell left (a killed command left nothing: the state stays as it was)
  if (!state.empty() && !killed) {
    std::ifstream fp(out_pwd), fe(out_env, std::ios::binary);
    std::string pwd;
    std::getline(fp, pwd);
    const std::string raw((std::istreambuf_iterator<char>(fe)), std::istreambuf_iterator<char>());
    if (!raw.empty()) {
      std::map<std::string, std::string> now;
      for (size_t i = 0; i < raw.size();) {
        const size_t e = raw.find('\0', i), end = e == std::string::npos ? raw.size() : e;
        const std::string kv = raw.substr(i, end - i);
        const size_t eq = kv.find('=');
        if (eq != std::string::npos) now[kv.substr(0, eq)] = kv.substr(eq + 1);
        i = end + 1;
      }
      Json ej = Json::object();
      for (const auto& [k, v] : now)
        if (!shell_own(k) && (!base.count(k) || base[k] != v)) ej.set(k, Json(v));
      for (const auto& [k, v] : base)
        if (!shell_own(k) && !now.count(k)) ej.set(k, Json());
      Json sj = Json::object();
      sj.set("cwd", Json(pwd.empty() ? dir : pwd));
      sj.set("env", ej);
      static std::atomic<uint64_t> seq{0};  // (two commands of one chat at once: two temporary files, never one)
      const std::string tmp_state = state + "." + std::to_string(plat::process_id()) + "-" + std::to_string(++seq) + ".tmp";
      {
        std::ofstream o(tmp_state, std::ios::binary | std::ios::trunc);
        o << sj.dump();
        o.close();
        if (!o) {
          unlink(tmp_state.c_str());
          throw std::runtime_error("cannot write the terminal's state");
        }
      }
      if (::rename(tmp_state.c_str(), state.c_str()) != 0) unlink(tmp_state.c_str());
    }
  }
  keep.into(r);
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

#else
inline std::wstring wide(const std::string& s) {
  if (s.empty()) return L"";
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
  std::wstring w(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
  return w;
}
inline std::string base64(const unsigned char* p, size_t n) {
  static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string o;
  for (size_t i = 0; i < n; i += 3) {
    const unsigned v = p[i] << 16 | (i + 1 < n ? p[i + 1] << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
    o += t[v >> 18 & 63];
    o += t[v >> 12 & 63];
    o += i + 1 < n ? t[v >> 6 & 63] : '=';
    o += i + 2 < n ? t[v & 63] : '=';
  }
  return o;
}
// this program's folder (runz, serverz and the other tools next to it)
inline std::string exe_dir() {
  wchar_t b[MAX_PATH * 2];
  const DWORD n = GetModuleFileNameW(nullptr, b, (DWORD)std::size(b));
  std::wstring w(b, n);
  w = w.substr(0, w.find_last_of(L"\\/"));
  const int m = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
  std::string s(m, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), m, nullptr, nullptr);
  return s;
}

// The chat's terminal session: each command is its own PowerShell, so what it changes is carried to the next one
// through the state file (JSON: cwd, and env, the environment variables that differ from a fresh PowerShell's, null
// for one removed). An activated .venv (VIRTUAL_ENV, PATH), a cd, $env:X = ... stay; PowerShell variables, functions
// and programs left running do not. The venv's deactivate is defined again while one is active.
inline std::string session_script(const std::string& state, const std::string& command) {
  auto q = [](std::string s) {
    for (size_t i = 0; (i = s.find('\'', i)) != std::string::npos; i += 2) s.insert(i, "'");
    return "'" + s + "'";
  };
  return "$__azS = " + q(state) + "; $__azB = @{}; Get-ChildItem Env: | ForEach-Object { $__azB[$_.Name] = $_.Value }\n"
         "if (Test-Path -LiteralPath $__azS) { try {\n"
         "  $__azJ = [IO.File]::ReadAllText($__azS) | ConvertFrom-Json\n"
         "  foreach ($p in $__azJ.env.PSObject.Properties) { if ($null -eq $p.Value) { Remove-Item -LiteralPath ('Env:' + $p.Name) -ErrorAction SilentlyContinue } else { Set-Item -LiteralPath ('Env:' + $p.Name) -Value $p.Value } }\n"
         "  if ($__azJ.cwd -and (Test-Path -LiteralPath $__azJ.cwd)) { Set-Location -LiteralPath $__azJ.cwd }\n"
         "} catch {} }\n"
         "if ($env:VIRTUAL_ENV -and $env:_OLD_VIRTUAL_PATH) { function global:deactivate { $env:PATH = $env:_OLD_VIRTUAL_PATH; "
         "Remove-Item Env:VIRTUAL_ENV, Env:_OLD_VIRTUAL_PATH, Env:VIRTUAL_ENV_PROMPT -ErrorAction SilentlyContinue } }\n"
         "$__azOk = $true\n"
         "try {\n" + command + "\n$__azOk = $?\n} finally {\n"
         "  $__azE = @{}; Get-ChildItem Env: | ForEach-Object { if ($__azB[$_.Name] -cne $_.Value) { $__azE[$_.Name] = $_.Value } }\n"
         "  foreach ($k in $__azB.Keys) { if (-not (Test-Path -LiteralPath ('Env:' + $k))) { $__azE[$k] = $null } }\n"
         "  try { [IO.File]::WriteAllText($__azS, (@{ cwd = (Get-Location).ProviderPath; env = $__azE } | ConvertTo-Json -Compress)) } catch {}\n"
         "}\n"
         "if (-not $__azOk) { exit 1 }\n";
}

// command (PowerShell) in cwd; extra_path goes first in PATH; state: the session's file ("" : none, a fresh shell);
// cancelled: asked every 200 ms, true ends the command like a timeout (the client closed its request: Stop, Esc)
inline Result run(const std::string& command, const std::string& cwd, const std::string& extra_path, int timeout_s,
                  const std::string& state = "", const std::function<bool()>& cancelled = nullptr) {
  Result r;
  auto q = [](std::string s) {  // (a PowerShell single-quoted string)
    for (size_t i = 0; (i = s.find('\'', i)) != std::string::npos; i += 2) s.insert(i, "'");
    return "'" + s + "'";
  };
  const std::string script = "$ProgressPreference = 'SilentlyContinue'; [Console]::OutputEncoding = [Text.Encoding]::UTF8; "
                             "$OutputEncoding = [Text.Encoding]::UTF8; $env:PATH = " + q(extra_path) + " + ';' + $env:PATH\n" +
                             (state.empty() ? command : session_script(state, command));
  const std::wstring ws = wide(script);
  const std::wstring cl = L"powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand " +
                          wide(base64((const unsigned char*)ws.data(), ws.size() * sizeof(wchar_t)));
  // (what can throw, allocated before the first handle: from the pipe to CreateProcessW nothing can leak them)
  SIZE_T alen = 0;
  InitializeProcThreadAttributeList(nullptr, 1, 0, &alen);
  std::vector<char> abuf(alen);
  std::wstring cmd = cl, wcwd = wide(cwd);
  SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
  HANDLE rd = nullptr, wr = nullptr;
  if (!CreatePipe(&rd, &wr, &sa, 0)) throw std::runtime_error("cannot make a pipe");
  SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
  HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
  HANDLE job = CreateJobObjectW(nullptr, nullptr);
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
  li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (job) SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof li);
  // (only these two handles go to the child: a command started at the same time by another request must not get
  //  this one's pipe, or this pipe stays open, and this call waits, until that command ends)
  STARTUPINFOEXW si{};
  si.StartupInfo.cb = sizeof si;
  si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  si.StartupInfo.hStdInput = nul != INVALID_HANDLE_VALUE ? nul : nullptr;  // (no NUL: no input, not handle -1, which is
  si.StartupInfo.hStdOutput = wr;                                          //  this process itself)
  si.StartupInfo.hStdError = wr;
  HANDLE inherit[2];
  size_t nh = 0;
  if (nul != INVALID_HANDLE_VALUE) inherit[nh++] = nul;
  inherit[nh++] = wr;
  si.lpAttributeList = (LPPROC_THREAD_ATTRIBUTE_LIST)abuf.data();
  const bool init = alen && InitializeProcThreadAttributeList(si.lpAttributeList, 1, 0, &alen);
  const bool attrs = init && UpdateProcThreadAttribute(si.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit,
                                                       nh * sizeof(HANDLE), nullptr, nullptr);
  PROCESS_INFORMATION pi{};
  const auto t0 = std::chrono::steady_clock::now();
  // (no handle list: no process at all. Inheriting without one hands the child every inheritable handle of the server,
  //  another request's pipe among them)
  const BOOL ok = attrs && CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE,
                                          CREATE_NO_WINDOW | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
                                          nullptr, wcwd.empty() ? nullptr : wcwd.c_str(), &si.StartupInfo, &pi);
  const DWORD err = attrs ? GetLastError() : 0;
  if (init) DeleteProcThreadAttributeList(si.lpAttributeList);
  CloseHandle(wr);
  if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
  if (!ok) {
    CloseHandle(rd);
    if (job) CloseHandle(job);
    throw std::runtime_error(attrs ? "cannot start PowerShell (error " + std::to_string(err) + ")"
                                   : std::string("cannot start PowerShell (no handle list for its output)"));
  }
  // (before it runs: its children are in the job too. Not in one: a timeout ends PowerShell itself only)
  const bool in_job = job && AssignProcessToJobObject(job, pi.hProcess);
  ResumeThread(pi.hThread);
  CloseHandle(pi.hThread);
  // the output, read on its own thread (a full pipe would stop the command): the first part and the last part.
  // (Its state is shared with the thread, which owns the read handle too: a read that never ends, a process outside
  //  the job holding the pipe, leaves the thread behind instead of this call waiting for it in join(); audit
  //  2026-11-12 #15. CancelIoEx is only best effort on a synchronous pipe)
  struct Out {
    HANDLE rd = nullptr;
    std::mutex mu;
    std::string head, tail;
    size_t total = 0;
    std::atomic<bool> done{false};
    ~Out() { if (rd) CloseHandle(rd); }
  };
  std::shared_ptr<Out> out;
  std::thread reader;
  try {  // (no reader thread: the command is stopped and its handles closed, not left running with a full pipe)
    out = std::make_shared<Out>();
    out->rd = rd;
    reader = std::thread([o = out] {
      char b[8192];
      DWORD n;
      while (ReadFile(o->rd, b, sizeof b, &n, nullptr) && n) {
        std::lock_guard<std::mutex> lk(o->mu);
        o->total += n;
        try {  // (out of memory: this part is left out, counted in total, and the pipe still drained; an exception
               //  leaving the thread would end the whole server)
          const size_t room = kMaxOut * 3 / 4 > o->head.size() ? kMaxOut * 3 / 4 - o->head.size() : 0;
          o->head.append(b, std::min<size_t>(room, n));
          if (n > room) {
            o->tail.append(b + room, n - room);
            if (o->tail.size() > kMaxOut / 4 * 2) o->tail.erase(0, o->tail.size() - kMaxOut / 4);
          }
        } catch (...) {}
      }
      o->done = true;
    });
  } catch (...) {
    if (in_job) TerminateJobObject(job, 1);
    TerminateProcess(pi.hProcess, 1);
    CloseHandle(pi.hProcess);
    if (!out) CloseHandle(rd);  // (else out's last owner closes it)
    if (job) CloseHandle(job);
    throw;
  }
  // the command's end, its timeout, or the client gone (polled: a request whose page stopped it)
  const auto deadline = t0 + std::chrono::seconds(timeout_s);
  while (WaitForSingleObject(pi.hProcess, 200) == WAIT_TIMEOUT) {
    const bool over = std::chrono::steady_clock::now() >= deadline, gone = !over && cancelled && cancelled();
    if (!over && !gone) continue;
    (over ? r.timed_out : r.cancelled) = true;
    if (in_job) TerminateJobObject(job, 1);
    TerminateProcess(pi.hProcess, 1);
    WaitForSingleObject(pi.hProcess, 5000);
    break;
  }
  DWORD code = 0;
  GetExitCodeProcess(pi.hProcess, &code);
  r.exit_code = (int)code;
  if (in_job) TerminateJobObject(job, 1);  // (what it left running in the background ends with it: its pipe closes)
  // (a process that left the job, e.g. one started through COM or WMI, may still hold the pipe: the read ends after
  //  2 s at most instead of waiting for it)
  for (int k = 0; k < 20 && !out->done; ++k) Sleep(100);
  if (!out->done) CancelIoEx(out->rd, nullptr);
  {  // (a synchronous read: CancelSynchronousIo on the reading thread, tried again while it lives, see server_mcp.h)
    HANDLE h = (HANDLE)reader.native_handle();
    for (int k = 0; k < 20 && !out->done && WaitForSingleObject(h, 50) == WAIT_TIMEOUT; ++k) CancelSynchronousIo(h);
  }
  if (out->done) reader.join();
  else reader.detach();  // (it ends, and closes the pipe, when the process holding it does)
  CloseHandle(pi.hProcess);
  if (job) CloseHandle(job);
  std::string head, tail;
  size_t total = 0;
  {
    std::lock_guard<std::mutex> lk(out->mu);
    head = out->head;
    tail = out->tail;
    total = out->total;
  }
  if (tail.size() > kMaxOut / 4) tail.erase(0, tail.size() - kMaxOut / 4);
  r.truncated = total > head.size() + tail.size();
  r.output = head + (r.truncated ? "\n[... " + std::to_string(total - head.size() - tail.size()) + " bytes left out ...]\n" : "") + tail;
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

#endif

}  // namespace term

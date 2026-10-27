// The web UI's terminal tool: one PowerShell command for the model, run in its chat's folder after the user allowed it
// in the page (Settings > Terminal: ask each time, or run without asking). Only for connections from this machine
// (see server.cpp): a server opened to the network with --host never runs a command for another computer.
// The command runs in a job object, so a timeout ends it and every process it started; its output (stdout and stderr
// together) is kept up to kMaxOut: the start and the end of a longer one.
#pragma once
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace term {

struct Result {
  int exit_code = -1;
  std::string output;
  bool timed_out = false, truncated = false;
  double seconds = 0;
};

inline constexpr size_t kMaxOut = 64 * 1024;

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

// command (PowerShell) in cwd; extra_path goes first in PATH; state: the session's file ("" : none, a fresh shell)
inline Result run(const std::string& command, const std::string& cwd, const std::string& extra_path, int timeout_s,
                  const std::string& state = "") {
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
  si.StartupInfo.hStdInput = nul;
  si.StartupInfo.hStdOutput = wr;
  si.StartupInfo.hStdError = wr;
  HANDLE inherit[2];
  size_t nh = 0;
  if (nul != INVALID_HANDLE_VALUE) inherit[nh++] = nul;
  inherit[nh++] = wr;
  SIZE_T alen = 0;
  InitializeProcThreadAttributeList(nullptr, 1, 0, &alen);
  std::vector<char> abuf(alen);
  si.lpAttributeList = (LPPROC_THREAD_ATTRIBUTE_LIST)abuf.data();
  const bool attrs = InitializeProcThreadAttributeList(si.lpAttributeList, 1, 0, &alen) &&
                     UpdateProcThreadAttribute(si.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit,
                                               nh * sizeof(HANDLE), nullptr, nullptr);
  PROCESS_INFORMATION pi{};
  std::wstring cmd = cl, wcwd = wide(cwd);
  const auto t0 = std::chrono::steady_clock::now();
  const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE,
                                 CREATE_NO_WINDOW | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | (attrs ? EXTENDED_STARTUPINFO_PRESENT : 0),
                                 nullptr, wcwd.empty() ? nullptr : wcwd.c_str(), &si.StartupInfo, &pi);
  if (attrs) DeleteProcThreadAttributeList(si.lpAttributeList);
  CloseHandle(wr);
  if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
  if (!ok) {
    CloseHandle(rd);
    if (job) CloseHandle(job);
    throw std::runtime_error("cannot start PowerShell (error " + std::to_string(GetLastError()) + ")");
  }
  if (job) AssignProcessToJobObject(job, pi.hProcess);  // (before it runs: its children are in the job too)
  ResumeThread(pi.hThread);
  CloseHandle(pi.hThread);
  // the output, read on its own thread (a full pipe would stop the command): the first part and the last part
  std::string head, tail;
  size_t total = 0;
  std::thread reader([&] {
    char b[8192];
    DWORD n;
    while (ReadFile(rd, b, sizeof b, &n, nullptr) && n) {
      total += n;
      const size_t room = kMaxOut * 3 / 4 > head.size() ? kMaxOut * 3 / 4 - head.size() : 0;
      head.append(b, std::min<size_t>(room, n));
      if (n > room) {
        tail.append(b + room, n - room);
        if (tail.size() > kMaxOut / 4 * 2) tail.erase(0, tail.size() - kMaxOut / 4);
      }
    }
  });
  if (WaitForSingleObject(pi.hProcess, (DWORD)timeout_s * 1000) == WAIT_TIMEOUT) {
    r.timed_out = true;
    if (job) TerminateJobObject(job, 1);
    else TerminateProcess(pi.hProcess, 1);
    WaitForSingleObject(pi.hProcess, 5000);
  }
  DWORD code = 0;
  GetExitCodeProcess(pi.hProcess, &code);
  r.exit_code = (int)code;
  if (job) TerminateJobObject(job, 1);  // (what it left running in the background ends with it: its pipe closes)
  reader.join();
  CloseHandle(rd);
  CloseHandle(pi.hProcess);
  if (job) CloseHandle(job);
  if (tail.size() > kMaxOut / 4) tail.erase(0, tail.size() - kMaxOut / 4);
  r.truncated = total > head.size() + tail.size();
  r.output = head + (r.truncated ? "\n[... " + std::to_string(total - head.size() - tail.size()) + " bytes left out ...]\n" : "") + tail;
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

}  // namespace term

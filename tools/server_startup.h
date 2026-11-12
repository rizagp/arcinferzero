// The log's first lines (docs/logging.md): which build, on what system and GPU, with which switches; what a report of
// a problem needs before anything else. The model's own lines follow at its load (server_load.h: model, memory).
#pragma once
#include <intrin.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace srvstart {

// bin\PREBUILT.txt's first line ("inferzero 2026.11.12.12 (built ..., commit ...)") in a package; else a build here
inline std::string version(const std::filesystem::path& exe_dir) {
  std::ifstream f(exe_dir / "PREBUILT.txt");
  std::string l;
  if (f && std::getline(f, l) && !l.empty()) {
    if (l.size() >= 3 && (unsigned char)l[0] == 0xEF && (unsigned char)l[1] == 0xBB && (unsigned char)l[2] == 0xBF) l.erase(0, 3);
    while (!l.empty() && (l.back() == '\r' || l.back() == ' ')) l.pop_back();
    return l;
  }
  return std::string("inferzero built from source (") + __DATE__ + ")";
}
inline std::string windows() {
  typedef LONG(WINAPI * Rtl)(OSVERSIONINFOW*);
  OSVERSIONINFOW v{};
  v.dwOSVersionInfoSize = sizeof v;
  if (HMODULE nt = GetModuleHandleW(L"ntdll.dll"))
    if (auto f = (Rtl)(void*)GetProcAddress(nt, "RtlGetVersion"); f && f(&v) == 0)
      return std::string(v.dwBuildNumber >= 22000 ? "Windows 11" : "Windows 10") + " build " + std::to_string(v.dwBuildNumber);
  return "Windows";
}
inline std::string cpu() {
  int r[4] = {};
  __cpuid(r, 0x80000000);
  if ((unsigned)r[0] < 0x80000004u) return "CPU";
  char b[49] = {};
  for (int i = 0; i < 3; ++i) {
    __cpuid(r, 0x80000002 + i);
    memcpy(b + 16 * i, r, 16);
  }
  std::string s = b;
  while (!s.empty() && s.back() == ' ') s.pop_back();
  while (!s.empty() && s.front() == ' ') s.erase(0, 1);
  return s;
}
// the ARC_* switches set in the environment (any case: Windows' names are), a value that looks secret not shown, and
// the user's folder in a value as %USERPROFILE% (the log is pasted into issues)
inline std::string switches() {
  std::string o;
  const char* up_env = getenv("USERPROFILE");
  const std::string home = up_env ? up_env : "";
  auto lower = [](std::string s) {
    for (char& c : s) c = (char)tolower((unsigned char)c);
    return s;
  };
  for (char** e = _environ; e && *e; ++e) {
    const std::string kv = *e;
    if (kv.size() < 4 || lower(kv.substr(0, 4)) != "arc_") continue;
    const size_t eq = kv.find('=');
    const std::string k = kv.substr(0, eq);
    std::string v = eq == std::string::npos ? "" : kv.substr(eq + 1);
    const std::string up = lower(k);
    bool secret = false;
    for (const char* w : {"token", "secret", "pass", "key", "auth", "cred", "cookie"})
      secret = secret || up.find(w) != std::string::npos;
    if (!home.empty())
      for (size_t at; (at = lower(v).find(lower(home))) != std::string::npos;) v.replace(at, home.size(), "%USERPROFILE%");
    for (char& c : v)
      if ((unsigned char)c < 0x20) c = '?';
    o += (o.empty() ? "" : " ") + k + "=" + (secret ? std::string("(set)") : v);
  }
  return o.empty() ? "none" : o;
}
inline void print_head(const std::filesystem::path& exe_dir) {
  LOG_INF("%s\n", version(exe_dir).c_str());
  MEMORYSTATUSEX ms{};
  ms.dwLength = sizeof ms;
  GlobalMemoryStatusEx(&ms);
  LOG_INF("system: %s, %s, %u logical processors, RAM %.1f GiB (%.1f free)\n", windows().c_str(), cpu().c_str(),
          std::thread::hardware_concurrency(), ms.ullTotalPhys / 1073741824.0, ms.ullAvailPhys / 1073741824.0);
  LOG_INF("ARC_* switches set: %s\n", switches().c_str());
}
// (the GPU runtime started here: after the NPU's zeInit(0), see server.cpp)
inline void print_gpu() {
  ClRt rt;
  if (rt.init(false)) {
    char name[256] = {}, drv[128] = {};
    cl_uint eus = 0, mhz = 0;
    cl_ulong mem = 0;
    // (the OpenCL runtime's own answers, as gpu_bench asks them: the Level Zero shim knows only the memory size)
    zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x102B, sizeof(name) - 1, name, nullptr);  // CL_DEVICE_NAME
    zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x102D, sizeof(drv) - 1, drv, nullptr);   // CL_DRIVER_VERSION
    zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x1002, sizeof(eus), &eus, nullptr);       // CL_DEVICE_MAX_COMPUTE_UNITS
    zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x100C, sizeof(mhz), &mhz, nullptr);       // CL_DEVICE_MAX_CLOCK_FREQUENCY
    clGetDeviceInfo(rt.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
    LOG_INF("GPU: %s, driver %s, %u EUs, %u MHz, %.1f GiB %s, XMX %s, %s sub-groups\n", name, drv, eus, mhz,
            mem / 1073741824.0, zrt::integrated() ? "(shared with the CPU)" : "of its own", rt.xmx ? "yes" : "no",
            rt.sg8 ? "8- and 16-lane" : "16-lane");
    if (rt.queue) clReleaseCommandQueue(rt.queue);
    if (rt.ctx) clReleaseContext(rt.ctx);
  } else {
    LOG_ERR("GPU: none found (Level Zero did not start: is the Intel graphics driver installed?)\n");
  }
}
// (never stops the server: a part that fails says so and the start goes on)
inline void print(const std::filesystem::path& exe_dir, bool gpu) noexcept {
  try {
    if (gpu) print_gpu();
    else print_head(exe_dir);
  } catch (const std::exception& e) {
    LOG_WRN("start-up information incomplete: %s\n", e.what());
  } catch (...) {
    LOG_WRN("start-up information incomplete\n");
  }
}

}  // namespace srvstart

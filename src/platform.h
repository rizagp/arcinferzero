// The platform layer (docs/linux-port.md, step 1): what the engine and the server need from the operating system,
// behind one small interface, so a Linux build swaps this file's other half instead of each caller. On Windows every
// call is the Win32 call the code made before, unchanged (the Windows build stays bit-identical). Started 2026-11-13
// with the GPU runtimes' loading and the user's home folder; files, mapping, sockets and processes follow.
#pragma once
#include <cstdlib>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#ifndef __stdcall
#define __stdcall  // (the GPU runtimes' calling convention: the only one on x86-64 Linux)
#endif
#endif

namespace plat {

// A shared library by its Windows and its Linux name ("ze_loader.dll" / "libze_loader.so.1"): nullptr when absent.
inline void* dl_open(const char* win_name, const char* linux_name) {
#ifdef _WIN32
  (void)linux_name;
  return (void*)LoadLibraryA(win_name);
#else
  (void)win_name;
  return dlopen(linux_name, RTLD_NOW | RTLD_LOCAL);
#endif
}
// A library this process has loaded already: nullptr when it is not loaded (on Windows no new reference; on Linux
// RTLD_NOLOAD adds one, never closed: the library stays loaded for the process anyway)
inline void* dl_loaded(const char* win_name, const char* linux_name) {
#ifdef _WIN32
  (void)linux_name;
  return (void*)GetModuleHandleA(win_name);
#else
  (void)win_name;
  return dlopen(linux_name, RTLD_NOW | RTLD_NOLOAD);
#endif
}
inline void* dl_sym(void* lib, const char* name) {
#ifdef _WIN32
  return (void*)GetProcAddress((HMODULE)lib, name);
#else
  return dlsym(lib, name);
#endif
}
inline void dl_close(void* lib) {
#ifdef _WIN32
  FreeLibrary((HMODULE)lib);
#else
  dlclose(lib);
#endif
}

// The user's home folder: %USERPROFILE% on Windows, $HOME on Linux; nullptr when the variable is not set.
inline const char* home_env() {
#ifdef _WIN32
  return getenv("USERPROFILE");
#else
  return getenv("HOME");
#endif
}

}  // namespace plat

// The platform layer (docs/linux-port.md, step 1): what the engine and the server need from the operating system,
// behind one small interface, so a Linux build swaps this file's other half instead of each caller. On Windows every
// call is the Win32 call the code made before, unchanged (the Windows build stays bit-identical). Started 2026-11-13
// with the GPU runtimes' loading and the user's home folder; files, mapping, sockets and processes follow.
#pragma once
#include <cstdint>
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
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
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

// A file mapped read-only into memory (model files, drafters): open, then its size, then the mapping, in three steps
// so each caller keeps its own checks and messages between them. Windows: CreateFileA (shared reading),
// GetFileSizeEx, CreateFileMappingA + MapViewOfFile, the calls the callers made before. Linux: open, fstat, mmap.
class FileMap {
 public:
  FileMap() = default;
  FileMap(const FileMap&) = delete;
  FileMap& operator=(const FileMap&) = delete;
  ~FileMap() { close(); }
  bool open(const char* path) {
    close();  // (a second open: the first file let go, not leaked)
#ifdef _WIN32
    file_ = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    return file_ != INVALID_HANDLE_VALUE;
#else
    // (no FILE_SHARE_READ lock on Linux: a file rewritten while mapped faults (SIGBUS) where Windows refused to open)
    fd_ = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) return false;
    struct stat st{};
    if (fstat(fd_, &st) != 0 || S_ISDIR(st.st_mode)) {  // (a folder: refused here, as CreateFileA refuses it)
      close();
      return false;
    }
    return true;
#endif
  }
  bool size(uint64_t& out) const {
#ifdef _WIN32
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(file_, &sz)) return false;
    out = (uint64_t)sz.QuadPart;
#else
    struct stat st{};
    if (fstat(fd_, &st) != 0) return false;
    out = (uint64_t)st.st_size;
#endif
    return true;
  }
  // 0: mapped (data()); 1: the mapping could not be made; 2: its view could not be made
  int map() {
#ifdef _WIN32
    map_ = CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!map_) return 1;
    base_ = (const uint8_t*)MapViewOfFile(map_, FILE_MAP_READ, 0, 0, 0);
    return base_ ? 0 : 2;
#else
    uint64_t n = 0;
    if (!size(n) || n == 0) return 1;  // (as Windows: an empty file has no mapping)
    void* p = mmap(nullptr, (size_t)n, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) return 2;
    base_ = (const uint8_t*)p;
    len_ = (size_t)n;
    return 0;
#endif
  }
  const uint8_t* data() const { return base_; }
  void close() {
#ifdef _WIN32
    if (base_) UnmapViewOfFile(base_);
    if (map_) CloseHandle(map_);
    if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
    map_ = nullptr;
    file_ = INVALID_HANDLE_VALUE;
#else
    if (base_) munmap((void*)base_, len_);
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    len_ = 0;
#endif
    base_ = nullptr;
  }

 private:
  const uint8_t* base_ = nullptr;
#ifdef _WIN32
  HANDLE file_ = INVALID_HANDLE_VALUE, map_ = nullptr;
#else
  int fd_ = -1;
  size_t len_ = 0;
#endif
};

// The user's home folder: %USERPROFILE% on Windows, $HOME on Linux; nullptr when the variable is not set.
inline const char* home_env() {
#ifdef _WIN32
  return getenv("USERPROFILE");
#else
  return getenv("HOME");
#endif
}

}  // namespace plat

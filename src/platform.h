// The platform layer (docs/linux-port.md, step 1): what the engine and the server need from the operating system,
// behind one small interface, so a Linux build swaps this file's other half instead of each caller. On Windows every
// call is the Win32 call the code made before, unchanged (the Windows build stays bit-identical). Started 2026-11-13
// with the GPU runtimes' loading and the user's home folder; files, mapping, sockets and processes follow.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>  // (_get_osfhandle)
#else
#include <cerrno>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifndef __stdcall
#define __stdcall  // (the GPU runtimes' calling convention: the only one on x86-64 Linux)
#endif
#endif

// The folder separator as a string literal, to join with others: app_dir() + PLAT_SEP "kernels" (Windows: the same
// bytes as the "\\kernels" written before)
#ifdef _WIN32
#define PLAT_SEP "\\"
#else
#define PLAT_SEP "/"
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
  // the same by a std::filesystem::path: on Windows its wide form (CreateFileW, as Whisper's model.bin was opened before;
  // its callers still pass the path through a narrow string, src/whisper.h, so names outside the code page do not reach
  // here intact yet), on Linux its bytes
  bool open_fs(const std::filesystem::path& p) {
#ifdef _WIN32
    close();
    file_ = CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    return file_ != INVALID_HANDLE_VALUE;
#else
    return open(p.c_str());
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

// A file read without the OS file cache, several reads in flight on numbered slots (the expert file, zero/expert_store.h:
// a whole expert per read, straight into memory the GPU copies from). Buffers, offsets and lengths 4 KB aligned.
// Windows: FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, one manual-reset event per slot, ReadFile /
// WaitForMultipleObjects / GetOverlappedResult / CancelIoEx, the calls expert_store.h made before. Linux (first form):
// O_DIRECT and a pread at start(), so a read is done when it starts (io_uring later, for reads in flight).
class AsyncFile {
 public:
  static constexpr int kMaxSlots = 64;
  AsyncFile() = default;
  AsyncFile(const AsyncFile&) = delete;
  AsyncFile& operator=(const AsyncFile&) = delete;
  ~AsyncFile() { close(); }
  // 0: open with `slots` slots; 1: the file cannot be opened; 2: the slots' events cannot be made
  int open(const char* path, int slots) {
    close();
    if (slots < 1 || slots > kMaxSlots) return 2;
    n_ = slots;
#ifdef _WIN32
    h_ = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
                     nullptr);
    if (h_ == INVALID_HANDLE_VALUE) return 1;
    for (int s = 0; s < n_; ++s)
      if (!(ev_[s] = CreateEventW(nullptr, TRUE, FALSE, nullptr))) return 2;
#else
    fd_ = ::open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd_ < 0 && errno == EINVAL) fd_ = ::open(path, O_RDONLY | O_CLOEXEC);  // (a filesystem without O_DIRECT: cached)
    if (fd_ < 0) return 1;
#endif
    return 0;
  }
  // a read of n bytes at `off` into dst on slot s; false: it failed to start (error() says why)
  bool start(int s, void* dst, uint32_t n, uint64_t off) {
#ifdef _WIN32
    OVERLAPPED& ov = ov_[s];
    ZeroMemory(&ov, sizeof ov);
    ov.Offset = (DWORD)off;
    ov.OffsetHigh = (DWORD)(off >> 32);
    ov.hEvent = ev_[s];
    ResetEvent(ev_[s]);
    return ReadFile(h_, dst, (DWORD)n, nullptr, &ov) || GetLastError() == ERROR_IO_PENDING;
#else
    size_t got = 0;
    ok_[s] = false;
    while (got < n) {
      const ssize_t r = pread(fd_, (char*)dst + got, n - got, (off_t)(off + got));
      if (r < 0) {
        if (errno == EINTR) continue;
        err_[s] = (unsigned long)errno;
        return false;
      }
      if (r == 0) return true;  // (the file ends first: started, and finish() says it failed, as a short read)
      got += (size_t)r;
    }
    ok_[s] = true;
    return true;
#endif
  }
  // waits for one of the n slots in ss to finish: its index in ss, or -1 (the wait failed)
  int wait_any(const int* ss, int n) {
#ifdef _WIN32
    HANDLE evs[kMaxSlots];
    for (int i = 0; i < n; ++i) evs[i] = ev_[ss[i]];
    const DWORD w = WaitForMultipleObjects((DWORD)n, evs, FALSE, INFINITE);
    if (w >= WAIT_OBJECT_0 + (DWORD)n) return -1;
    return (int)(w - WAIT_OBJECT_0);
#else
    (void)ss;
    return n > 0 ? 0 : -1;  // (every read finished in start)
#endif
  }
  // the finished read on slot s: true when it read everything it asked for
  bool finish(int s) {
#ifdef _WIN32
    DWORD got = 0;
    return GetOverlappedResult(h_, &ov_[s], &got, FALSE) != 0;
#else
    return ok_[s];
#endif
  }
  // a read in flight on slot s cancelled and waited for (no late write into its buffer)
  void cancel(int s) {
#ifdef _WIN32
    CancelIoEx(h_, &ov_[s]);
    DWORD got = 0;
    GetOverlappedResult(h_, &ov_[s], &got, TRUE);
#else
    (void)s;
#endif
  }
  // why the last start() on slot s, or the last wait (s = -1), failed: GetLastError() on Windows (per thread), errno
  unsigned long error(int s = -1) const {
#ifdef _WIN32
    (void)s;
    return GetLastError();
#else
    return s >= 0 ? err_[s] : 0;
#endif
  }
  void close() {
#ifdef _WIN32
    for (int s = 0; s < kMaxSlots; ++s)
      if (ev_[s]) CloseHandle(ev_[s]), ev_[s] = nullptr;
    if (h_ != INVALID_HANDLE_VALUE) CloseHandle(h_);
    h_ = INVALID_HANDLE_VALUE;
#else
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
#endif
    n_ = 0;
  }

 private:
  int n_ = 0;
#ifdef _WIN32
  HANDLE h_ = INVALID_HANDLE_VALUE;
  HANDLE ev_[kMaxSlots] = {};
  OVERLAPPED ov_[kMaxSlots] = {};
#else
  int fd_ = -1;
  unsigned long err_[kMaxSlots] = {};  // (per slot: the two reading threads use slots of their own)
  bool ok_[kMaxSlots] = {};
#endif
};

// A file written without the OS file cache (the expert file: 15-52 GB through the Windows cache filled the RAM of a 32 GB
// laptop): created anew, written in 4 KB multiples from page-aligned buffers. Windows: CreateFileA (no sharing,
// CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH) and WriteFile in pieces of up to 1 GiB, as
// expert_store.h did. Linux: O_DIRECT (plain writes where a filesystem has none) and write().
class DirectWriter {
 public:
  DirectWriter() = default;
  DirectWriter(const DirectWriter&) = delete;
  DirectWriter& operator=(const DirectWriter&) = delete;
  ~DirectWriter() { close(); }
  bool open(const char* path) {
    close();
#ifdef _WIN32
    h_ = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, nullptr);
    return h_ != INVALID_HANDLE_VALUE;
#else
    fd_ = ::open(path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT | O_CLOEXEC, 0644);
    if (fd_ < 0 && errno == EINVAL) fd_ = ::open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    return fd_ >= 0;
#endif
  }
  bool is_open() const {
#ifdef _WIN32
    return h_ != INVALID_HANDLE_VALUE;
#else
    return fd_ >= 0;
#endif
  }
  // all n bytes, or false
  bool write(const void* p, uint64_t n) {
    while (n) {
      const uint64_t c = n < (1ull << 30) ? n : (1ull << 30);
#ifdef _WIN32
      DWORD got = 0;
      if (!WriteFile(h_, p, (DWORD)c, &got, nullptr) || got != (DWORD)c) return false;
#else
      const ssize_t got = ::write(fd_, p, (size_t)c);
      if (got < 0 && errno == EINTR) continue;
      if (got != (ssize_t)c) return false;
#endif
      p = (const uint8_t*)p + c;
      n -= c;
    }
    return true;
  }
  void close() {
#ifdef _WIN32
    if (h_ != INVALID_HANDLE_VALUE) CloseHandle(h_);
    h_ = INVALID_HANDLE_VALUE;
#else
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
#endif
  }

 private:
#ifdef _WIN32
  HANDLE h_ = INVALID_HANDLE_VALUE;
#else
  int fd_ = -1;
#endif
};

// A file read in order, opened shared for writing and deleting (the server's media, src/http.h send_file: a file being
// played can still be replaced or deleted). Windows: CreateFileW (FILE_SHARE_READ | WRITE | DELETE,
// FILE_FLAG_SEQUENTIAL_SCAN), GetFileSizeEx, SetFilePointerEx, ReadFile, the calls send_file made before. Linux: open,
// fstat, lseek, read (no locks to share: a file replaced while read keeps its old contents for this reader).
class SeqFile {
 public:
  SeqFile() = default;
  SeqFile(const SeqFile&) = delete;
  SeqFile& operator=(const SeqFile&) = delete;
  ~SeqFile() { close(); }
  bool open(const std::filesystem::path& p) {
    close();
#ifdef _WIN32
    h_ = CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                     OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    return h_ != INVALID_HANDLE_VALUE;
#else
    // (O_NONBLOCK: a FIFO would block the open until a writer came; only a regular file is read, then blocking reads)
    fd_ = ::open(p.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd_ < 0) return false;
    struct stat st{};
    if (fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode)) {  // (a folder, a FIFO, a device: refused)
      close();
      return false;
    }
    fcntl(fd_, F_SETFL, fcntl(fd_, F_GETFL) & ~O_NONBLOCK);
    return true;
#endif
  }
  bool size(uint64_t& out) const {
#ifdef _WIN32
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(h_, &sz)) return false;
    out = (uint64_t)sz.QuadPart;
#else
    struct stat st{};
    if (fstat(fd_, &st) != 0) return false;
    out = (uint64_t)st.st_size;
#endif
    return true;
  }
  void seek(uint64_t off) {
#ifdef _WIN32
    LARGE_INTEGER pos{};
    pos.QuadPart = (LONGLONG)off;
    SetFilePointerEx(h_, pos, nullptr, FILE_BEGIN);
#else
    lseek(fd_, (off_t)off, SEEK_SET);
#endif
  }
  // up to n bytes: the count read, 0 at the end or on an error
  size_t read(void* dst, size_t n) {
#ifdef _WIN32
    DWORD rd = 0;
    return ReadFile(h_, dst, (DWORD)n, &rd, nullptr) ? (size_t)rd : 0;
#else
    while (true) {
      const ssize_t r = ::read(fd_, dst, n);
      if (r < 0 && errno == EINTR) continue;
      return r > 0 ? (size_t)r : 0;
    }
#endif
  }
  void close() {
#ifdef _WIN32
    if (h_ != INVALID_HANDLE_VALUE) CloseHandle(h_);
    h_ = INVALID_HANDLE_VALUE;
#else
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
#endif
  }

 private:
#ifdef _WIN32
  HANDLE h_ = INVALID_HANDLE_VALUE;
#else
  int fd_ = -1;
#endif
};

// Page-aligned memory of n bytes, zeroed (VirtualAlloc; mmap on Linux): nullptr when there is none. Freed by page_free(p, n).
inline void* page_alloc(size_t n) {
#ifdef _WIN32
  return VirtualAlloc(nullptr, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
  void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  return p == MAP_FAILED ? nullptr : p;
#endif
}
inline void page_free(void* p, size_t n) {
#ifdef _WIN32
  (void)n;
  if (p) VirtualFree(p, 0, MEM_RELEASE);
#else
  if (p) munmap(p, n);
#endif
}
// The pages this process touched leave its working set (they stay cached for reuse): after a pass over a mapped file.
// Linux: nothing (the kernel reclaims mapped file pages by itself).
inline void trim_working_set() {
#ifdef _WIN32
  SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
#endif
}
// `from` renamed to `to`, an existing `to` replaced: false on failure (last_error() says why)
inline bool replace_file(const char* from, const char* to) {
#ifdef _WIN32
  return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING) != 0;
#else
  return ::rename(from, to) == 0;
#endif
}
inline void remove_file(const char* path) {
#ifdef _WIN32
  DeleteFileA(path);
#else
  ::unlink(path);
#endif
}
inline unsigned long last_error() {
#ifdef _WIN32
  return GetLastError();
#else
  return (unsigned long)errno;
#endif
}

// The physical memory free now, in bytes (GlobalMemoryStatusEx's ullAvailPhys; Linux: /proc/meminfo's MemAvailable,
// which counts the file cache that can be dropped, as Windows' standby list is counted). 0 when unknown.
inline uint64_t ram_available() {
#ifdef _WIN32
  MEMORYSTATUSEX ms{sizeof ms};
  GlobalMemoryStatusEx(&ms);
  return ms.ullAvailPhys;
#else
  FILE* f = fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256];
  unsigned long long kb = 0;
  while (fgets(line, sizeof line, f))
    if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
  fclose(f);
  return (uint64_t)kb * 1024;
#endif
}
// The physical memory in all, in bytes (ullTotalPhys; Linux: sysconf's pages)
inline uint64_t ram_total() {
#ifdef _WIN32
  MEMORYSTATUSEX ms{sizeof ms};
  GlobalMemoryStatusEx(&ms);
  return ms.ullTotalPhys;
#else
  return (uint64_t)sysconf(_SC_PHYS_PAGES) * (uint64_t)sysconf(_SC_PAGESIZE);
#endif
}
// A FILE*'s position past 2 GB (_fseeki64 / _ftelli64; fseeko / ftello on Linux, whose off_t is 64-bit there)
inline int fseek64(FILE* f, int64_t off, int whence) {
#ifdef _WIN32
  return _fseeki64(f, off, whence);
#else
  return fseeko(f, (off_t)off, whence);
#endif
}
inline int64_t ftell64(FILE* f) {
#ifdef _WIN32
  return _ftelli64(f);
#else
  return (int64_t)ftello(f);
#endif
}
// An environment variable set for this process (and the children it starts): _putenv_s / setenv.
inline void set_env(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}
inline unsigned long process_id() {
#ifdef _WIN32
  return GetCurrentProcessId();
#else
  return (unsigned long)getpid();
#endif
}
inline unsigned long thread_id() {
#ifdef _WIN32
  return GetCurrentThreadId();
#else
  return (unsigned long)gettid();
#endif
}

// A time as local / UTC calendar time (localtime_s / gmtime_s; localtime_r / gmtime_r): true when it converted.
inline bool local_time(std::time_t t, std::tm& out) {
#ifdef _WIN32
  return localtime_s(&out, &t) == 0;
#else
  return localtime_r(&t, &out) != nullptr;
#endif
}
inline bool utc_time(std::time_t t, std::tm& out) {
#ifdef _WIN32
  return gmtime_s(&out, &t) == 0;
#else
  return gmtime_r(&t, &out) != nullptr;
#endif
}
// fopen by a std::filesystem::path: _wfopen with its wide form on Windows (any characters), its bytes on Linux.
// mode: "rb", "wb", "ab"...
inline FILE* fopen_path(const std::filesystem::path& p, const char* mode) {
#ifdef _WIN32
  wchar_t wm[8] = {};
  for (int i = 0; i < 7 && mode[i]; ++i) wm[i] = (wchar_t)mode[i];
  return _wfopen(p.c_str(), wm);
#else
  return fopen(p.c_str(), mode);
#endif
}
// A link to elsewhere, not followed by folder walks: a reparse point on Windows (junction, symbolic link), a symbolic
// link on Linux. false when it cannot be told.
inline bool is_link(const std::filesystem::path& p) {
#ifdef _WIN32
  const DWORD at = GetFileAttributesW(p.c_str());
  return at != INVALID_FILE_ATTRIBUTES && (at & FILE_ATTRIBUTE_REPARSE_POINT);
#else
  std::error_code ec;
  return std::filesystem::is_symlink(std::filesystem::symlink_status(p, ec));
#endif
}
// A file or folder the system hides (Windows' hidden / system attributes; Linux: none beyond the leading dot, which
// the callers check by name).
inline bool is_hidden_or_system(const std::filesystem::path& p) {
#ifdef _WIN32
  const DWORD at = GetFileAttributesW(p.c_str());
  return at != INVALID_FILE_ATTRIBUTES && (at & (FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM));
#else
  (void)p;
  return false;
#endif
}
#ifndef _WIN32
// (Linux) a inside b, at a '/' boundary, case-sensitive as the filesystem is (b itself counts when or_same); the
// callers pass canonical paths. Windows' callers keep their own case-blind '\' comparisons.
inline bool path_inside(const std::filesystem::path& a, const std::filesystem::path& b, bool or_same) {
  std::string x = a.lexically_normal().string(), y = b.lexically_normal().string();
  while (x.size() > 1 && x.back() == '/') x.pop_back();
  while (y.size() > 1 && y.back() == '/') y.pop_back();
  if (y.empty() || x.compare(0, y.size(), y) != 0) return false;
  if (x.size() == y.size()) return or_same;
  return y == "/" || x[y.size()] == '/';
}
#endif
// Two open files are one file on disk (Windows: volume and file index; Linux: device and inode).
inline bool same_file(FILE* a, FILE* b) {
#ifdef _WIN32
  BY_HANDLE_FILE_INFORMATION a1{}, a2{};
  const HANDLE ha = (HANDLE)_get_osfhandle(_fileno(a)), hb = (HANDLE)_get_osfhandle(_fileno(b));
  return ha != INVALID_HANDLE_VALUE && GetFileInformationByHandle(ha, &a1) && GetFileInformationByHandle(hb, &a2) &&
         a1.dwVolumeSerialNumber == a2.dwVolumeSerialNumber && a1.nFileIndexHigh == a2.nFileIndexHigh &&
         a1.nFileIndexLow == a2.nFileIndexLow;
#else
  struct stat sa{}, sb{};
  return fstat(fileno(a), &sa) == 0 && fstat(fileno(b), &sb) == 0 && sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino;
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

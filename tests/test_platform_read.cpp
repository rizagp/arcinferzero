// Linux read fallback without a GPU: emulate a valid, CPU-writable USM range which O_DIRECT cannot pin.
#ifndef _WIN32
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
static char* usm = nullptr;
static size_t usm_bytes = 0;
static int faults = 0;
static ssize_t test_pread(int fd, void* dst, size_t n, off_t off) {
  const uintptr_t p = reinterpret_cast<uintptr_t>(dst), base = reinterpret_cast<uintptr_t>(usm);
  if (usm && p >= base && p - base < usm_bytes) { ++faults; errno = EFAULT; return -1; }
  return ::pread(fd, dst, n, off);
}
#define pread test_pread
#include "../src/platform.h"
#undef pread
int main() {
  char path[] = "/tmp/inferzero-read-XXXXXX";
  int fd = mkstemp(path);
  if (fd < 0) return 1;
  char expected[16384];
  for (size_t i = 0; i < sizeof expected; ++i) expected[i] = (char)(i * 7 + i / 4096);
  if (write(fd, expected, sizeof expected) != sizeof expected) return 2;
  ::close(fd);
  if (posix_memalign(reinterpret_cast<void**>(&usm), 4096, sizeof expected)) return 3;
  usm_bytes = sizeof expected;
  plat::AsyncFile file;
  if (file.open(path, 2)) return 4;
  unlink(path);
  if (!file.start(0, usm, 4096, 4096) || !file.finish(0) || memcmp(usm, expected + 4096, 4096)) return 5;
  if (faults != 1) return 6;
  if (!file.start(0, usm, 4096, 0) || !file.finish(0) || memcmp(usm, expected, 4096)) return 7;
  if (faults != 1) return 8;
  if (!file.start(0, usm, 8192, 4096) || !file.finish(0) || memcmp(usm, expected + 4096, 8192)) return 9;
  if (!file.start(1, usm, 4096, 8192) || !file.finish(1) || memcmp(usm, expected + 8192, 4096)) return 10;
  if (!file.start(0, usm, 4096, sizeof expected) || file.finish(0) || file.error(0) != EIO) return 11;
  file.close();
  free(usm);
  puts("test_platform_read: all passed");
}
#else
#include <cstdio>
int main() { puts("test_platform_read: Linux-only check skipped"); }
#endif

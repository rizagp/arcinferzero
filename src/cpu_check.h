// Force-included into every program by build.ps1 (/FI): the programs are built /arch:AVX2, so on a CPU without AVX2
// / FMA (an older PC with an Arc A380 or A310) they died at their first vector instruction with 0xC000001D and no
// message. This checks first, in a static object of the program's only translation unit, before main and before the
// other static objects (this header comes first), and stops with a line saying why. (Audit 2026-11-12 #4. The check
// is integer code and library calls only: nothing in it needs AVX. No windows.h here: its min / max macros would
// reach every source.)
#pragma once
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <intrin.h>
#endif
#include <immintrin.h>

namespace inferzero_cpu {
inline bool has_avx2() {
#ifndef _WIN32
  // (GCC / Clang: libgcc's CPU model, which counts AVX only when the OS saves the registers, as the xgetbv test below)
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  int r[4] = {};
  __cpuid(r, 0);
  if (r[0] < 7) return false;
  __cpuid(r, 1);
  const bool fma = (r[2] >> 12) & 1, osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
  if (!fma || !osxsave || !avx) return false;
  if ((_xgetbv(0) & 6) != 6) return false;  // (the OS saves the vector registers)
  __cpuidex(r, 7, 0);
  return (r[1] >> 5) & 1;
#endif
}
struct Check {
  Check() {
    if (has_avx2()) return;
    fputs("inferzero needs a CPU with AVX2 and FMA (Intel Core 4th generation / Haswell or newer, AMD Zen or newer); "
          "this one has none, so the programs cannot run here.\n", stderr);
    fflush(stderr);
    std::_Exit(3);
  }
};
static Check check_;
}  // namespace inferzero_cpu

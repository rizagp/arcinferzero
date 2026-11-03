// Memory read bandwidth: the CPU (1..all threads), the GPU, and both at once. Generation reads every weight once per
// token, so it runs at the speed memory is read. xenolith (a Gemma engine for Xe-LP laptops) decodes on the CPU
// because there the CPU reads RAM faster than the iGPU; this says whether that holds here, and whether the CPU and
// the GPU reading together get more than either alone (then a matrix's rows could be split between them).
//   bin\membw_probe.exe [--gib 1] [--seconds 3] [--threads 1,2,4,8,all]
// Each test reads its buffers over and over for --seconds and counts the bytes; the buffers (--gib each, CPU and GPU
// apart) are larger than every cache. On an iGPU both are the same RAM; on a discrete card the GPU's is its VRAM.
#include "../zero/clrt.h"
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

static const char* kSrc = R"CLC(
// UNROLL independent 16-byte loads in flight per work-item, each step's loads contiguous over the whole grid
__kernel void rd(__global const uint4* x, int n4, __global uint* out) {
  uint4 a[UNROLL];
  for (int u = 0; u < UNROLL; ++u) a[u] = (uint4)(0);
  const int g = get_global_size(0);
  for (int i = get_global_id(0); i + (UNROLL - 1) * g < n4; i += UNROLL * g)
    for (int u = 0; u < UNROLL; ++u) a[u] += x[i + u * g];
  for (int u = 1; u < UNROLL; ++u) a[0] += a[u];
  if ((a[0].x ^ a[0].y ^ a[0].z ^ a[0].w) == 0x9e3779b9u) out[0] = 1;  // (keeps the reads; almost never true)
}
)CLC";

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// one CPU thread: 32-byte loads over [p, p + n), summed so the compiler keeps them; passes until stop
static void cpu_reader(const uint8_t* p, size_t n, const std::atomic<bool>& stop, std::atomic<uint64_t>& bytes,
                       std::atomic<uint64_t>& sink) {
  __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
  uint64_t done = 0;
  while (!stop.load(std::memory_order_relaxed)) {
    for (size_t i = 0; i + 128 <= n; i += 128) {
      a0 = _mm256_add_epi64(a0, _mm256_load_si256((const __m256i*)(p + i)));
      a1 = _mm256_add_epi64(a1, _mm256_load_si256((const __m256i*)(p + i + 32)));
      a2 = _mm256_add_epi64(a2, _mm256_load_si256((const __m256i*)(p + i + 64)));
      a3 = _mm256_add_epi64(a3, _mm256_load_si256((const __m256i*)(p + i + 96)));
    }
    done += n;
  }
  a0 = _mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3));
  sink += (uint64_t)_mm256_extract_epi64(a0, 0) ^ (uint64_t)_mm256_extract_epi64(a0, 3);
  bytes += done;
}

struct Result { double cpu_gbs = 0, gpu_gbs = 0; };

int main(int argc, char** argv) {
  double gib = 1.0, seconds = 3.0;
  int unroll = 8, groups_arg = 0;  // (125H: unroll 8 72-73 GB/s at 512-8192 groups; 1-4 swung 36-71)
  std::string tlist = "1,2,4,8,all";
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--gib") gib = atof(argv[i + 1]);
    else if (a == "--seconds") seconds = atof(argv[i + 1]);
    else if (a == "--threads") tlist = argv[i + 1];
    else if (a == "--unroll") unroll = std::max(1, atoi(argv[i + 1]));      // (GPU loads in flight per work-item)
    else if (a == "--groups") groups_arg = std::max(1, atoi(argv[i + 1]));  // (GPU work-groups of 256)
  }
  const int hw = (int)std::max(1u, std::thread::hardware_concurrency());
  std::vector<int> threads;
  for (size_t s = 0; s <= tlist.size();) {
    size_t e = tlist.find(',', s);
    if (e == std::string::npos) e = tlist.size();
    const std::string t = tlist.substr(s, e - s);
    const int n = t == "all" ? hw : atoi(t.c_str());
    if (n >= 1 && n <= hw && std::find(threads.begin(), threads.end(), n) == threads.end()) threads.push_back(n);
    s = e + 1;
  }
  const size_t bytes = ((size_t)(gib * (1ull << 30)) + 4095) / 4096 * 4096;

  ClRt cl;
  if (!cl.init()) return 1;
  // the CPU's buffer: plain host memory, every page touched before timing
  uint8_t* hbuf = (uint8_t*)_aligned_malloc(bytes, 4096);
  if (!hbuf) { fprintf(stderr, "could not allocate %.1f GiB of host memory\n", gib); return 1; }
  memset(hbuf, 1, bytes);
  // the GPU's: a device buffer (on an iGPU the same RAM)
  cl_program prog = cl.build(kSrc, "-cl-std=CL3.0 -DUNROLL=" + std::to_string(unroll));
  cl_kernel k = cl.kernel(prog, "rd");
  cl_mem dbuf = cl.buffer(CL_MEM_READ_WRITE, bytes), dout = cl.buffer(CL_MEM_READ_WRITE, 64);
  const uint32_t one = 0x01010101u;
  CL_CHECK(clEnqueueFillBuffer(cl.queue, dbuf, &one, 4, 0, bytes, 0, nullptr, nullptr));
  clFinish(cl.queue);
  const int n4 = (int)(bytes / 16);
  set_args(k, 0, dbuf, n4, dout);
  // (default 2048 work-groups of 256: every thread slot of a 160-EU card filled; each work-item 16 bytes per load,
  //  the grid's loads contiguous)
  const uint32_t lws = 256, groups = groups_arg ? (uint32_t)groups_arg : 2048u;
  ZE_CHECK(zeKernelSetGroupSize(k->k, lws, 1, 1));
  ze_group_count_t gc{groups, 1, 1};
  auto gpu_pass = [&] {
    ZE_CHECK(zeCommandListAppendLaunchKernel(zrt::D().cl, k->k, &gc, nullptr, 0, nullptr));
    zrt::sync();
  };
  gpu_pass();  // (build, first touch)

  // one test: `nt` CPU threads (0: none) over the host buffer, the GPU (or not) over its own, for `seconds`
  auto run = [&](int nt, bool gpu) {
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> cb{0}, sink{0};
    uint64_t gb = 0;
    double gpu_t = 0;
    std::vector<std::thread> th;
    const size_t slice = bytes / (nt ? nt : 1) / 128 * 128;
    const auto t0 = clk::now();
    for (int t = 0; t < nt; ++t) th.emplace_back(cpu_reader, hbuf + slice * t, slice, std::cref(stop), std::ref(cb), std::ref(sink));
    std::thread gt;
    if (gpu)
      gt = std::thread([&] {
        const auto g0 = clk::now();
        while (!stop.load(std::memory_order_relaxed)) { gpu_pass(); gb += bytes; }
        gpu_t = secs(g0, clk::now());
      });
    while (secs(t0, clk::now()) < seconds) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    stop = true;
    for (auto& t : th) t.join();
    if (gt.joinable()) gt.join();
    const double cpu_t = secs(t0, clk::now());
    Result r;
    r.cpu_gbs = nt ? cb / cpu_t / 1e9 : 0;
    r.gpu_gbs = gpu ? gb / gpu_t / 1e9 : 0;
    if (sink == 42) printf(" ");  // (uses the sums)
    return r;
  };

  printf("buffers: %.2f GiB for the CPU, %.2f GiB for the GPU; %.1f s per test; %d hardware threads\n",
         bytes / 1073741824.0, bytes / 1073741824.0, seconds, hw);
  printf("%-34s %10s %10s %10s\n", "", "CPU GB/s", "GPU GB/s", "together");
  const Result g = run(0, true);
  printf("%-34s %10s %10.1f %10.1f\n", "GPU alone", "-", g.gpu_gbs, g.gpu_gbs);
  double best_cpu = 0;
  int best_n = 0;
  for (int n : threads) {
    const Result c = run(n, false);
    char what[64];
    snprintf(what, sizeof what, "CPU alone, %d thread%s", n, n > 1 ? "s" : "");
    printf("%-34s %10.1f %10s %10.1f\n", what, c.cpu_gbs, "-", c.cpu_gbs);
    if (c.cpu_gbs > best_cpu) { best_cpu = c.cpu_gbs; best_n = n; }
  }
  for (int n : threads) {
    const Result b = run(n, true);
    char what[64];
    snprintf(what, sizeof what, "GPU + CPU %d thread%s", n, n > 1 ? "s" : "");
    printf("%-34s %10.1f %10.1f %10.1f\n", what, b.cpu_gbs, b.gpu_gbs, b.cpu_gbs + b.gpu_gbs);
  }
  printf("\nbest CPU alone: %.1f GB/s (%d threads) against the GPU's %.1f GB/s\n", best_cpu, best_n, g.gpu_gbs);
  _aligned_free(hbuf);
  return 0;
}

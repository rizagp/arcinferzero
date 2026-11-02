// Multi-GPU sync probe (QUEUE F31.2): can a split model's sums run without the CPU in the middle? Today each sum
// of a generated token (zero/mgpu/engine_tp.h tp_reduce, ~128 per 27B token) is: each GPU copies its part to host
// memory, its CPU thread waits for it, the threads meet at a spin barrier, then each GPU reads all parts (4 B60s:
// ~57 us per sum). Level Zero lets a GPU wait on an event another GPU signals, from an event pool made for several
// devices (the driver keeps such a pool in host memory, so no peer access is needed). This measures, for 2..N GPUs:
//   1. whether such a pool can be made, and whether a 20 KB part sent GPU a -> host -> GPU b through an event
//      arrives intact for every pair;
//   2. microseconds per sum: today's path (copy, host wait, CPU barrier, read) against events (copy signals, the
//      other GPUs wait on the GPU, read), M sums in a row;
//   3. the failsafe: a GPU waiting on an event no GPU will signal is released by the host (zeEventHostSignal);
//   4. whether the driver has the experimental GPU memory waits (zexCommandListAppendWaitOnMemory64).
//   bin\event_probe.exe              all GPUs
//   bin\event_probe.exe --sums 2000  more sums per timing (default 1000)
//   bin\event_probe.exe --sim 2      2 command lists on GPU 0 (a test of the tool itself on one GPU)
#include "ze_min.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <string>
#include <thread>
#include <vector>

// (not in ze_min.h: only this tool uses them; looked up so a driver without one still runs the rest)
typedef ze_result_t(__stdcall* PFN_sig)(ze_command_list_handle_t, ze_event_handle_t);
typedef ze_result_t(__stdcall* PFN_wait)(ze_command_list_handle_t, uint32_t, ze_event_handle_t*);
typedef ze_result_t(__stdcall* PFN_reset)(ze_command_list_handle_t, ze_event_handle_t);
typedef ze_result_t(__stdcall* PFN_ext)(ze_driver_handle_t, const char*, void**);
static PFN_sig AppendSignal = nullptr;
static PFN_wait AppendWait = nullptr;
static PFN_reset AppendReset = nullptr;
static PFN_ext GetExt = nullptr;

#define ZE_EVENT_POOL_FLAG_HOST_VISIBLE 0x1u
#define ZE_EVENT_SCOPE_FLAG_HOST 0x4u

struct Gpu {
  ze_device_handle_t dev = nullptr;
  ze_device_properties_t p{};
  ze_command_list_handle_t cl = nullptr;
  void* part = nullptr;   // (device: this GPU's part)
  void* parts = nullptr;  // (device: all parts read back, the stand-in for tp_add)
};

static double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
static const uint64_t kWait = 5000000000ull;  // (5 s: a wait that does not finish is reported, not hung on)

int main(int argc, char** argv) {
  int sums = 1000, sim = 0;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--sums") && i + 1 < argc) sums = std::max(50, atoi(argv[++i]));
    else if (!strcmp(argv[i], "--sim") && i + 1 < argc) sim = std::max(2, atoi(argv[++i]));
  }
  if (!ze_load()) { fprintf(stderr, "cannot load ze_loader.dll (Level Zero, part of the Intel graphics driver)\n"); return 1; }
  HMODULE zl = GetModuleHandleA("ze_loader.dll");
  AppendSignal = (PFN_sig)GetProcAddress(zl, "zeCommandListAppendSignalEvent");
  AppendWait = (PFN_wait)GetProcAddress(zl, "zeCommandListAppendWaitOnEvents");
  AppendReset = (PFN_reset)GetProcAddress(zl, "zeCommandListAppendEventReset");
  GetExt = (PFN_ext)GetProcAddress(zl, "zeDriverGetExtensionFunctionAddress");
  if (!AppendSignal || !AppendWait || !AppendReset) { fprintf(stderr, "ze_loader.dll lacks the event functions\n"); return 1; }
  ZE_CHECK(zeInit(1));
  uint32_t nd = 0;
  ZE_CHECK(zeDriverGet(&nd, nullptr));
  std::vector<ze_driver_handle_t> drvs(nd);
  ZE_CHECK(zeDriverGet(&nd, drvs.data()));
  ze_driver_handle_t drv = nullptr;
  std::vector<ze_device_handle_t> devs;
  for (ze_driver_handle_t d : drvs) {  // (the driver with the most GPUs, as mgpu_probe)
    uint32_t n = 0;
    if (zeDeviceGet(d, &n, nullptr) != 0 || !n) continue;
    std::vector<ze_device_handle_t> all(n), here;
    zeDeviceGet(d, &n, all.data());
    for (ze_device_handle_t x : all) {
      ze_device_properties_t p{};
      p.stype = ZE_STYPE_DEVICE_PROPERTIES;
      if (zeDeviceGetProperties(x, &p) == 0 && p.type == 1) here.push_back(x);
    }
    if (here.size() > devs.size()) { devs = here; drv = d; }
  }
  if (devs.empty()) { fprintf(stderr, "no Level Zero GPU found\n"); return 1; }
  std::vector<Gpu> g(sim ? sim : devs.size());
  for (size_t i = 0; i < g.size(); ++i) {
    g[i].dev = devs[sim ? 0 : i];
    g[i].p.stype = ZE_STYPE_DEVICE_PROPERTIES;
    zeDeviceGetProperties(g[i].dev, &g[i].p);
    printf("GPU %zu: %s (device 0x%04x)%s\n", i, g[i].p.name, g[i].p.deviceId, sim ? " (--sim: GPU 0)" : "");
  }
  const int N = (int)g.size();
  if (N < 2) {
    printf("one GPU: nothing to sync (run with --sim 2 to test the tool on it)\n");
    return 0;
  }
  ze_context_desc_t cd{ZE_STYPE_CONTEXT_DESC, nullptr, 0};
  ze_context_handle_t ctx = nullptr;
  ZE_CHECK(zeContextCreate(drv, &cd, &ctx));
  const size_t part = 20 * 1024;  // (a 27B token's sum: 5120 floats)
  for (int i = 0; i < N; ++i) {
    uint32_t ng = 0, ord = 0, nq = 1;
    zeDeviceGetCommandQueueGroupProperties(g[i].dev, &ng, nullptr);
    std::vector<ze_command_queue_group_properties_t> gp(ng);
    for (auto& q : gp) q.stype = ZE_STYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    zeDeviceGetCommandQueueGroupProperties(g[i].dev, &ng, gp.data());
    for (uint32_t k = 0; k < ng; ++k)
      if (gp[k].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) { ord = k; nq = std::max(1u, gp[k].numQueues); break; }
    // (--sim: each list on its own queue of the group where there are several, so one list's wait cannot hold the
    //  other's signal behind it)
    ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, ord, sim ? (uint32_t)i % nq : 0, 2, 2, 0};
    ZE_CHECK(zeCommandListCreateImmediate(ctx, g[i].dev, &qd, &g[i].cl));
    ze_device_mem_alloc_desc_t dd{ZE_STYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    ZE_CHECK(zeMemAllocDevice(ctx, &dd, part, 64, g[i].dev, &g[i].part));
    ZE_CHECK(zeMemAllocDevice(ctx, &dd, part * N, 64, g[i].dev, &g[i].parts));
  }
  // host memory: 2 halves x N parts (as engine_tp.h), and a page to check what arrived
  char* host = nullptr;
  ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
  ZE_CHECK(zeMemAllocHost(ctx, &hd, part * N * 2, 4096, (void**)&host));
  char* check = nullptr;
  ZE_CHECK(zeMemAllocHost(ctx, &hd, part, 4096, (void**)&check));

  // ---- 1. an event pool for all the GPUs
  std::vector<ze_device_handle_t> pdev;
  for (const Gpu& x : g)
    if (std::find(pdev.begin(), pdev.end(), x.dev) == pdev.end()) pdev.push_back(x.dev);
  ze_event_pool_desc_t pd{ZE_STYPE_EVENT_POOL_DESC, nullptr, ZE_EVENT_POOL_FLAG_HOST_VISIBLE, (uint32_t)(2 * N + 1)};
  ze_event_pool_handle_t pool = nullptr;
  const ze_result_t pr = zeEventPoolCreate(ctx, &pd, (uint32_t)pdev.size(), pdev.data(), &pool);
  printf("\nevent pool for %zu device(s): %s", pdev.size(), pr == 0 ? "made\n" : "");
  if (pr != 0) {
    printf("error 0x%x: GPU-side sums are not possible with this driver\n", pr);
    return 0;
  }
  std::vector<ze_event_handle_t> ev(2 * N + 1);
  for (int i = 0; i < 2 * N + 1; ++i) {
    ze_event_desc_t ed{ZE_STYPE_EVENT_DESC, nullptr, (uint32_t)i, ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};
    ZE_CHECK(zeEventCreate(pool, &ed, &ev[i]));
  }
  auto E = [&](int half, int r) { return ev[half * N + r]; };

  // ---- 1b. every pair: GPU a's part (a pattern) to host memory, signalling; GPU b waits on the GPU, reads it
  bool all_ok = true;
  for (int a = 0; a < N; ++a)
    for (int b = 0; b < N; ++b) {
      if (a == b) continue;
      std::vector<unsigned char> pat(part);
      for (size_t k = 0; k < part; ++k) pat[k] = (unsigned char)(k * 7 + a * 31 + b * 13 + 1);
      memcpy(check, pat.data(), part);
      ZE_CHECK(zeCommandListAppendMemoryCopy(g[a].cl, g[a].part, check, part, nullptr, 0, nullptr));
      ZE_CHECK(zeCommandListHostSynchronize(g[a].cl, UINT64_MAX));
      memset(check, 0, part);
      zeEventHostReset(E(0, a));
      // (b's GPU waits first, the signal comes later; --sim: the signal first, as both lists may share one queue,
      //  where a wait ahead of the signal would hold it forever)
      ze_event_handle_t ea = E(0, a);
      if (sim) ZE_CHECK(zeCommandListAppendMemoryCopy(g[a].cl, host, g[a].part, part, ea, 0, nullptr));
      ZE_CHECK(AppendWait(g[b].cl, 1, &ea));
      ZE_CHECK(zeCommandListAppendMemoryCopy(g[b].cl, g[b].parts, host, part, nullptr, 0, nullptr));
      ZE_CHECK(zeCommandListAppendMemoryCopy(g[b].cl, check, g[b].parts, part, nullptr, 0, nullptr));
      if (!sim) ZE_CHECK(zeCommandListAppendMemoryCopy(g[a].cl, host, g[a].part, part, ea, 0, nullptr));
      const ze_result_t ra = zeCommandListHostSynchronize(g[a].cl, kWait), rb = zeCommandListHostSynchronize(g[b].cl, kWait);
      const bool ok = ra == 0 && rb == 0 && !memcmp(check, pat.data(), part);
      all_ok = all_ok && ok;
      printf("  GPU %d -> host -> GPU %d through an event: %s\n", a, b,
             ok ? "intact" : ra || rb ? "did not finish within 5 s" : "WRONG DATA");
      if (ra || rb) {
        printf("GPU-side waits do not work here: stopping\n");
        return 0;
      }
    }

  // ---- 2. microseconds per sum, M sums in a row
  auto reset_all = [&] {
    for (auto e : ev) zeEventHostReset(e);
  };
  // today: each GPU's thread copies its part, waits for its GPU, meets the others at a spin barrier, then its GPU
  // reads all parts (the read waited for before the next sum, as tp_add finishes before the next layer's work)
  auto host_path = [&](int n) {
    std::atomic<int> arrived{0}, gen{0};
    auto barrier = [&] {
      const int my = gen.load();
      if (arrived.fetch_add(1) + 1 == n) {
        arrived = 0;
        gen.fetch_add(1);
      } else {
        while (gen.load() == my) _mm_pause();
      }
    };
    std::vector<std::thread> th;
    const double t0 = now_us();
    for (int r = 0; r < n; ++r)
      th.emplace_back([&, r] {
        for (int s = 0; s < sums; ++s) {
          char* half = host + (size_t)(s & 1) * N * part;
          zeCommandListAppendMemoryCopy(g[r].cl, half + r * part, g[r].part, part, nullptr, 0, nullptr);
          zeCommandListHostSynchronize(g[r].cl, UINT64_MAX);
          barrier();
          zeCommandListAppendMemoryCopy(g[r].cl, g[r].parts, half, part * n, nullptr, 0, nullptr);
        }
        zeCommandListHostSynchronize(g[r].cl, UINT64_MAX);
      });
    for (auto& t : th) t.join();
    return (now_us() - t0) / sums;
  };
  // events: per sum and GPU, appended ahead by one CPU thread: reset its event of this half, copy its part signalling
  // it, wait on the GPU for the others' events, read all parts. The CPU waits once at the end.
  auto event_path = [&](int n) {
    reset_all();
    const double t0 = now_us();
    for (int s = 0; s < sums; ++s) {
      const int h = s & 1;
      char* half = host + (size_t)h * N * part;
      for (int r = 0; r < n; ++r) {  // (every GPU's copy appended before any wait: lists sharing a queue, --sim)
        if (s >= 2) AppendReset(g[r].cl, E(h, r));
        zeCommandListAppendMemoryCopy(g[r].cl, half + r * part, g[r].part, part, E(h, r), 0, nullptr);
      }
      for (int r = 0; r < n; ++r) {
        std::vector<ze_event_handle_t> others;
        for (int k = 0; k < n; ++k)
          if (k != r) others.push_back(E(h, k));
        AppendWait(g[r].cl, (uint32_t)others.size(), others.data());
        zeCommandListAppendMemoryCopy(g[r].cl, g[r].parts, half, part * n, nullptr, 0, nullptr);
      }
    }
    bool ok = true;
    for (int r = 0; r < n; ++r) ok = zeCommandListHostSynchronize(g[r].cl, (uint64_t)sums * 1000000ull) == 0 && ok;
    return ok ? (now_us() - t0) / sums : -1.0;
  };
  printf("\nmicroseconds per sum of a 20 KB part (%d sums in a row, median of 5 runs):\n", sums);
  double best_host = 1e9, best_ev = 1e9;
  for (int n = 2; n <= N; n = n == N ? N + 1 : std::min(N, n * 2)) {
    std::vector<double> th, te;
    host_path(n);  // (warm-up)
    for (int k = 0; k < 5; ++k) th.push_back(host_path(n));
    bool ev_ok = event_path(n) >= 0;
    for (int k = 0; k < 5 && ev_ok; ++k) {
      const double t = event_path(n);
      if (t < 0) ev_ok = false;
      else te.push_back(t);
    }
    const double mh = median(th), me = ev_ok ? median(te) : -1;
    char evs[32] = "did not finish";
    if (ev_ok) snprintf(evs, sizeof evs, "%.1f us", me);
    printf("  %d GPUs: today's path %.1f us, events %s\n", n, mh, evs);
    if (n == N) best_host = mh, best_ev = ev_ok ? me : 1e9;
  }

  // ---- 3. the failsafe: a GPU waiting on an event nothing signals, released by the host
  {
    ze_event_handle_t dead = ev[2 * N];
    zeEventHostReset(dead);
    AppendWait(g[0].cl, 1, &dead);
    const ze_result_t before = zeCommandListHostSynchronize(g[0].cl, 100000000ull);  // (100 ms: must not finish)
    const double t0 = now_us();
    zeEventHostSignal(dead);
    const ze_result_t after = zeCommandListHostSynchronize(g[0].cl, kWait);
    printf("\nfailsafe: GPU 0 waiting on an event no GPU signals: %s; released by the host: %s (%.2f ms)\n",
           before == ZE_RESULT_NOT_READY ? "still waiting after 100 ms (as it should)" : "did NOT wait",
           after == 0 ? "yes" : "NO", (now_us() - t0) / 1000);
  }

  // ---- 4. the experimental GPU memory waits (a counter in host memory instead of events)
  if (GetExt) {
    const char* names[] = {"zexCommandListAppendWaitOnMemory64", "zexCommandListAppendWaitOnMemory",
                           "zexCommandListAppendWriteToMemory"};
    printf("\ndriver extensions:");
    for (const char* nm : names) {
      void* f = nullptr;
      printf(" %s %s;", nm, GetExt(drv, nm, &f) == 0 && f ? "yes" : "no");
    }
    printf("\n");
  }
  printf("\npairs intact: %s\n", all_ok ? "yes" : "NO");
  if (best_ev < 1e9)
    printf("27B over %d GPUs, 128 sums per token: today ~%.1f ms per token in sums, with events ~%.1f ms\n", N,
           128 * best_host / 1000, 128 * best_ev / 1000);
  printf("Paste this whole output into the GitHub issue.\n");
  return 0;
}

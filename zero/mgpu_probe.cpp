// Multi-GPU probe: what moving data between GPUs costs on this machine, the numbers a multi-GPU arcinferzero
// depends on. Tensor parallel token generation syncs twice per layer (27B: ~128 small transfers per token), so the
// latency of a ~20 KB transfer decides whether splitting the weights over GPUs speeds generation up; prompt chunks
// move ~40 MB, where bandwidth counts. Measures, for every pair of GPUs: whether the driver allows peer access, and
// a 20 KB / 40 MB copy GPU A -> GPU B directly (peer) and through host memory, each on the compute queue and on a
// copy-engine queue where the GPU has one. With one GPU it runs GPU 0 to itself (a check of the tool, not of a link).
//   bin\mgpu_probe.exe            all GPUs
//   bin\mgpu_probe.exe --reps 500 more repetitions for the latencies (default 200)
#include "ze_min.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

typedef ze_result_t(__stdcall* PFN_canAccessPeer)(ze_device_handle_t, ze_device_handle_t, uint8_t*);
static PFN_canAccessPeer zeDeviceCanAccessPeer_ = nullptr;

struct Gpu {
  ze_device_handle_t dev = nullptr;
  ze_device_properties_t p{};
  uint64_t mem = 0;
  uint32_t compute_ord = 0, copy_ord = UINT32_MAX;
  ze_command_list_handle_t cl = nullptr, cl_copy = nullptr;
};

static double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static ze_command_list_handle_t make_list(ze_context_handle_t ctx, ze_device_handle_t dev, uint32_t ordinal) {
  ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, ordinal, 0, 2 /* in order */, 2 /* async */, 0};
  ze_command_list_handle_t cl = nullptr;
  return zeCommandListCreateImmediate(ctx, dev, &qd, &cl) == 0 ? cl : nullptr;
}

// one copy on list `cl` and its completion seen by the host, timed; median of reps (after 10 warm-up copies)
static double copy_us(ze_command_list_handle_t cl, void* dst, const void* src, size_t bytes, int reps) {
  std::vector<double> t;
  for (int r = 0; r < reps + 10; ++r) {
    const double t0 = now_us();
    if (zeCommandListAppendMemoryCopy(cl, dst, src, bytes, nullptr, 0, nullptr) != 0) return -1;
    if (zeCommandListHostSynchronize(cl, UINT64_MAX) != 0) return -1;
    if (r >= 10) t.push_back(now_us() - t0);
  }
  return median(t);
}
// A -> host on A's list, then host -> B on B's list (the path without peer access)
static double via_host_us(ze_command_list_handle_t a, ze_command_list_handle_t b, void* dst, void* host,
                          const void* src, size_t bytes, int reps) {
  std::vector<double> t;
  for (int r = 0; r < reps + 10; ++r) {
    const double t0 = now_us();
    if (zeCommandListAppendMemoryCopy(a, host, src, bytes, nullptr, 0, nullptr) != 0) return -1;
    if (zeCommandListHostSynchronize(a, UINT64_MAX) != 0) return -1;
    if (zeCommandListAppendMemoryCopy(b, dst, host, bytes, nullptr, 0, nullptr) != 0) return -1;
    if (zeCommandListHostSynchronize(b, UINT64_MAX) != 0) return -1;
    if (r >= 10) t.push_back(now_us() - t0);
  }
  return median(t);
}

int main(int argc, char** argv) {
  int reps = 200;
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::max(10, atoi(argv[++i]));
  if (!ze_load()) { fprintf(stderr, "cannot load ze_loader.dll (Level Zero, part of the Intel graphics driver)\n"); return 1; }
  HMODULE zl = GetModuleHandleA("ze_loader.dll");
  zeDeviceCanAccessPeer_ = (PFN_canAccessPeer)GetProcAddress(zl, "zeDeviceCanAccessPeer");
  ZE_CHECK(zeInit(1));
  uint32_t nd = 0;
  ZE_CHECK(zeDriverGet(&nd, nullptr));
  std::vector<ze_driver_handle_t> drvs(nd);
  ZE_CHECK(zeDriverGet(&nd, drvs.data()));
  // the driver with the most GPUs (one context holds them all; Intel's GPUs share one driver)
  ze_driver_handle_t drv = nullptr;
  std::vector<Gpu> gpus;
  for (ze_driver_handle_t d : drvs) {
    uint32_t n = 0;
    if (zeDeviceGet(d, &n, nullptr) != 0 || !n) continue;
    std::vector<ze_device_handle_t> devs(n);
    zeDeviceGet(d, &n, devs.data());
    std::vector<Gpu> here;
    for (ze_device_handle_t x : devs) {
      Gpu g;
      g.dev = x;
      g.p.stype = ZE_STYPE_DEVICE_PROPERTIES;
      if (zeDeviceGetProperties(x, &g.p) != 0 || g.p.type != 1) continue;  // (1: GPU)
      uint32_t nm = 1;
      ze_device_memory_properties_t mp{};
      mp.stype = ZE_STYPE_DEVICE_MEMORY_PROPERTIES;
      if (zeDeviceGetMemoryProperties(x, &nm, &mp) == 0) g.mem = mp.totalSize;
      uint32_t ng = 0;
      zeDeviceGetCommandQueueGroupProperties(x, &ng, nullptr);
      std::vector<ze_command_queue_group_properties_t> gp(ng);
      for (auto& q : gp) q.stype = ZE_STYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
      zeDeviceGetCommandQueueGroupProperties(x, &ng, gp.data());
      for (uint32_t i = 0; i < ng; ++i) {
        if (gp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) { g.compute_ord = i; break; }
      }
      for (uint32_t i = 0; i < ng; ++i)
        if ((gp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) && !(gp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)) {
          g.copy_ord = i;
          break;
        }
      here.push_back(g);
    }
    if (here.size() > gpus.size()) { gpus = here; drv = d; }
  }
  if (gpus.empty()) { fprintf(stderr, "no Level Zero GPU found\n"); return 1; }
  printf("GPUs: %zu\n", gpus.size());
  for (size_t i = 0; i < gpus.size(); ++i) {
    const Gpu& g = gpus[i];
    printf("  GPU %zu: %s (device 0x%04x), %.1f GiB, %u EUs, copy engine %s\n", i, g.p.name, g.p.deviceId,
           g.mem / 1073741824.0, g.p.numSlices * g.p.numSubslicesPerSlice * g.p.numEUsPerSubslice,
           g.copy_ord != UINT32_MAX ? "yes" : "no");
  }
  ze_context_desc_t cd{ZE_STYPE_CONTEXT_DESC, nullptr, 0};
  ze_context_handle_t ctx = nullptr;
  ZE_CHECK(zeContextCreate(drv, &cd, &ctx));
  for (Gpu& g : gpus) {
    g.cl = make_list(ctx, g.dev, g.compute_ord);
    if (g.copy_ord != UINT32_MAX) g.cl_copy = make_list(ctx, g.dev, g.copy_ord);
    if (!g.cl) { fprintf(stderr, "cannot create a command list on %s\n", g.p.name); return 1; }
  }

  const size_t small = 20 * 1024, big = (size_t)40 << 20;  // a token's sync (5120 floats) / a prompt chunk's rows
  std::vector<void*> dbuf(gpus.size(), nullptr);
  for (size_t i = 0; i < gpus.size(); ++i) {
    ze_device_mem_alloc_desc_t dd{ZE_STYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    ZE_CHECK(zeMemAllocDevice(ctx, &dd, big * 2, 64, gpus[i].dev, &dbuf[i]));
    const int zero = 0;
    zeCommandListAppendMemoryFill(gpus[i].cl, dbuf[i], &zero, 4, big * 2, nullptr, 0, nullptr);
    zeCommandListHostSynchronize(gpus[i].cl, UINT64_MAX);
  }
  void* host = nullptr;
  ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
  ZE_CHECK(zeMemAllocHost(ctx, &hd, big, 64, &host));

  // baseline: a 4-byte copy and the host seeing it complete (the floor of every sync below)
  printf("\nGPU 0 4-byte copy + host wait: %.1f us (the floor of every sync)\n",
         copy_us(gpus[0].cl, (char*)dbuf[0] + big, dbuf[0], 4, reps));
  // within one GPU (memory to memory): the reference for the links below
  {
    const double s = copy_us(gpus[0].cl, (char*)dbuf[0] + big, dbuf[0], small, reps);
    const double b = copy_us(gpus[0].cl, (char*)dbuf[0] + big, dbuf[0], big, std::max(10, reps / 20));
    printf("GPU 0 -> GPU 0 memory: 20 KB %.1f us, 40 MB %.2f ms (%.1f GB/s)\n", s, b / 1000, big / b / 1000);
  }

  const bool one = gpus.size() == 1;
  if (one) printf("\nOne GPU: the pair below is GPU 0 to itself (checks the tool, not a link between GPUs)\n");
  printf("\n%-8s %-6s %-22s %12s %12s %10s\n", "pair", "peer", "path", "20 KB (us)", "40 MB (ms)", "GB/s");
  double best_small = 1e30;
  for (size_t a = 0; a < gpus.size(); ++a)
    for (size_t b = 0; b < gpus.size(); ++b) {
      if (a == b && !one) continue;
      uint8_t peer = 0;
      if (zeDeviceCanAccessPeer_) zeDeviceCanAccessPeer_(gpus[a].dev, gpus[b].dev, &peer);
      char pair[16];
      snprintf(pair, sizeof(pair), "%zu -> %zu", a, b);
      void* src = dbuf[a];
      void* dst = (char*)dbuf[b] + big;
      struct Path { const char* name; ze_command_list_handle_t cl; };
      std::vector<Path> direct = {{"direct, compute queue", gpus[a].cl}};
      if (gpus[a].cl_copy) direct.push_back({"direct, copy engine", gpus[a].cl_copy});
      for (const Path& p : direct) {
        const double s = copy_us(p.cl, dst, src, small, reps);
        const double bb = copy_us(p.cl, dst, src, big, std::max(10, reps / 20));
        if (s < 0 || bb < 0) {
          printf("%-8s %-6s %-22s %12s %12s %10s\n", pair, peer ? "yes" : "no", p.name, "failed", "failed", "-");
          continue;
        }
        best_small = std::min(best_small, s);
        printf("%-8s %-6s %-22s %12.1f %12.2f %10.1f\n", pair, peer ? "yes" : "no", p.name, s, bb / 1000, big / bb / 1000);
      }
      std::vector<Path> hosted = {{"via host, compute", nullptr}};
      if (gpus[a].cl_copy && gpus[b].cl_copy) hosted.push_back({"via host, copy engine", nullptr});
      for (size_t k = 0; k < hosted.size(); ++k) {
        ze_command_list_handle_t la = k ? gpus[a].cl_copy : gpus[a].cl, lb = k ? gpus[b].cl_copy : gpus[b].cl;
        const double s = via_host_us(la, lb, dst, host, src, small, reps);
        const double bb = via_host_us(la, lb, dst, host, src, big, std::max(10, reps / 20));
        if (s < 0 || bb < 0) {
          printf("%-8s %-6s %-22s %12s %12s %10s\n", pair, peer ? "yes" : "no", hosted[k].name, "failed", "failed", "-");
          continue;
        }
        best_small = std::min(best_small, s);
        printf("%-8s %-6s %-22s %12.1f %12.2f %10.1f\n", pair, peer ? "yes" : "no", hosted[k].name, s, bb / 1000,
               big / bb / 1000);
      }
    }

  // what it means for a 27B (64 layers) split over the GPUs: two syncs per layer and token
  if (!one && best_small < 1e29) {
    const double per_tok_ms = 128 * best_small / 1000;
    printf("\n27B tensor parallel: ~128 syncs per token x %.1f us = %.1f ms per token added", best_small, per_tok_ms);
    printf(" (one B60 generates a token of the 27B IQ3_S in ~46 ms)\n");
  }
  printf("\nPlease paste this whole output into the GitHub issue.\n");
  for (void* p : dbuf) zeMemFree(ctx, p);
  zeMemFree(ctx, host);
  return 0;
}

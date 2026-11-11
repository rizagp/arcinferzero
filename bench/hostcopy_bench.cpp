// inferzero benchmark: zero/mgpu/hostcopy.h against the whole-block copy it replaces, size by size -- and the three
// lines a caller needs.
//
//   Build:  .\build.ps1 bench\hostcopy_bench.cpp
//   Run:    bin\hostcopy_bench.exe [MB ...]     (default: 0.25 0.375 0.5 1 2 4 8 16 40; fractions are MB too)
//
// The three lines, with every other argument defaulted (2 MB pieces, a ring of 4, one host sync per piece):
//   hostcopy::HostCopy copy;
//   copy.init(ctx, list_a, list_b);   // list_a copies GPU A -> host, list_b host -> GPU B: one thread each
//   copy.copy(dst, src, bytes);       // for any size: it takes the faster of the two ways itself
//   copy.term();                      // (the destructor does it too)
//
// The two ways copy() chooses between, both measured below:
//   copy_whole        the old path: the whole block GPU A -> host, the host waits, then host -> GPU B and waits
//                     again. Two ~13.6 GB/s links one after the other (6.4 ms for 40 MB on two B60s, 6.5 GB/s).
//   copy_pipelined    the block in pieces: one thread copies GPU A -> host into a ring, the other host -> GPU B out
//                     of it, so both links work at once (3.45 ms for 40 MB, 12.1 GB/s, 1.85x).
// copy() picks the pieces for a copy of 384 KB and up, splitting it into pieces of about an eighth of its size (a
// 1 MB copy in 4 x 256 KB, 8 MB in 8 x 1 MB, 40 MB in 20 x 2 MB); below that the whole-block copy is faster. The
// two ways meet at about 256 KB, where two 128 KB pieces barely overlap and the hand-overs cost as much as they
// save.
#include "../zero/mgpu/hostcopy.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

// The median of `reps` timed calls of f (microseconds; -1 if a copy failed).
template <class F>
static float median_us(F f, int reps, int warm = 5) {
  std::vector<float> t;
  t.reserve((size_t)reps);
  for (int r = 0; r < reps + warm; ++r) {
    const double t0 = hostcopy::now_us();
    const bool ok = f();
    const double dt = hostcopy::now_us() - t0;
    if (!ok) return -1;
    if (r >= warm) t.push_back((float)dt);
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main(int argc, char** argv) {
  // ---- Level Zero, up to two GPUs, a context, and one immediate command list per direction (a thread each) --------
  if (!ze_load()) {
    fprintf(stderr, "cannot load ze_loader.dll (Level Zero, part of the Intel graphics driver)\n");
    return 1;
  }
  ZE_CHECK(zeInit(1));
  uint32_t nd = 0;
  ZE_CHECK(zeDriverGet(&nd, nullptr));
  std::vector<ze_driver_handle_t> drivers(nd);
  ZE_CHECK(zeDriverGet(&nd, drivers.data()));
  ze_driver_handle_t driver = nullptr;
  std::vector<ze_device_handle_t> gpus;
  for (ze_driver_handle_t d : drivers) {  // (the driver that has the most GPUs: Intel's cards share one)
    uint32_t n = 0;
    if (zeDeviceGet(d, &n, nullptr) != 0 || n == 0) continue;
    std::vector<ze_device_handle_t> here(n);
    zeDeviceGet(d, &n, here.data());
    std::vector<ze_device_handle_t> use;
    for (ze_device_handle_t h : here) {
      ze_device_properties_t p{};
      p.stype = ZE_STYPE_DEVICE_PROPERTIES;
      if (zeDeviceGetProperties(h, &p) == 0 && p.type == 1) use.push_back(h);
    }
    if (use.size() > gpus.size()) {
      gpus = use;
      driver = d;
    }
  }
  if (gpus.empty()) {
    fprintf(stderr, "no Level Zero GPU found\n");
    return 1;
  }
  // (with one GPU the copy runs GPU 0 -> GPU 0: a check of the code, not of a link between two cards)
  ze_device_handle_t dev_a = gpus[0], dev_b = gpus.size() > 1 ? gpus[1] : gpus[0];
  ze_context_desc_t cd{ZE_STYPE_CONTEXT_DESC, nullptr, 0};
  ze_context_handle_t ctx = nullptr;
  ZE_CHECK(zeContextCreate(driver, &cd, &ctx));
  const auto compute_list = [&](ze_device_handle_t dev) {
    uint32_t ng = 0;
    zeDeviceGetCommandQueueGroupProperties(dev, &ng, nullptr);
    std::vector<ze_command_queue_group_properties_t> groups(ng);
    for (auto& g : groups) g.stype = ZE_STYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    zeDeviceGetCommandQueueGroupProperties(dev, &ng, groups.data());
    uint32_t ordinal = 0;
    for (uint32_t i = 0; i < ng; ++i)
      if (groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
        ordinal = i;
        break;
      }
    ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, ordinal, 0, 2 /* in order */, 2 /* async */, 0};
    ze_command_list_handle_t cl = nullptr;
    ZE_CHECK(zeCommandListCreateImmediate(ctx, dev, &qd, &cl));
    return cl;
  };
  ze_command_list_handle_t list_a = compute_list(dev_a), list_b = compute_list(dev_b);

  // ---- the sizes to measure (MB on the command line, fractions allowed: 0.25 1 40) --------------------------------
  std::vector<size_t> sizes;
  for (int i = 1; i < argc; ++i) {
    const double mb = atof(argv[i]);
    if (mb > 0) sizes.push_back((size_t)(mb * 1048576.0));
  }
  if (sizes.empty())
    sizes = {size_t(256) << 10, size_t(384) << 10, size_t(512) << 10, size_t(1) << 20, size_t(2) << 20,
             size_t(4) << 20,   size_t(8) << 20,   size_t(16) << 20,  size_t(40) << 20};
  const size_t max_bytes = *std::max_element(sizes.begin(), sizes.end());

  // ---- the memory: a source buffer on GPU A and a destination on GPU B --------------------------------------------
  ze_device_mem_alloc_desc_t dd{ZE_STYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  void* src = nullptr;  // GPU A
  void* dst = nullptr;  // GPU B
  ZE_CHECK(zeMemAllocDevice(ctx, &dd, max_bytes, 64, dev_a, &src));
  ZE_CHECK(zeMemAllocDevice(ctx, &dd, max_bytes, 64, dev_b, &dst));

  // ---- the whole subject in three lines (defaults: 2 MB pieces, a ring of 4, one host sync per piece) -------------
  hostcopy::HostCopy copy;
  if (!copy.init(ctx, list_a, list_b)) {  // (the two lists must be different handles: one thread each)
    fprintf(stderr, "init failed (pieces must be at least %zu KB)\n", hostcopy::HostCopy::kMinChunk >> 10);
    return 1;
  }
  const size_t pieces = (max_bytes + copy.chunk - 1) / copy.chunk;
  printf("%zu GPU(s); pieces %zu MB, ring %d, %d per host sync; copying GPU 0 -> GPU %d (%zu pieces at most)\n",
         gpus.size(), copy.chunk >> 20, copy.slots, copy.sync_every, gpus.size() > 1 ? 1 : 0, pieces);

  // ---- the comparison: the old path, the pieces, and what copy() picks -------------------------------------------
  printf("\n%8s %8s | %10s %7s | %15s %7s | %9s %8s\n", "bytes", "piece", "copy_whole", "GB/s", "copy_pipelined",
         "GB/s", "copy()", "speedup");
  for (size_t n : sizes) {
    const int reps = n >= (size_t)8 << 20 ? 10 : 20;
    // (each measurement twice, the better kept: the first block after a new size is a few us slower, which at 512 KB
    //  is a tenth of the copy)
    const auto best = [&](auto f) {
      const float a = median_us(f, reps), b = median_us(f, reps);
      return a < 0 || b < 0 ? -1.0f : std::min(a, b);
    };
    // (copy_whole puts the whole block in the ring, so it can only take what fits in it)
    const float whole =
        n <= (size_t)copy.slots * copy.chunk ? best([&] { return copy.copy_whole(dst, src, n); }) : -1;
    const size_t piece = copy.piece_for(n);  // (0 = copy() takes the whole-block path for this size)
    float pipelined = -1;
    if (piece != 0) {
      const float a = copy.benchmark_us(dst, src, n, reps, 10, piece);
      const float b = copy.benchmark_us(dst, src, n, reps, 10, piece);
      pipelined = a < 0 || b < 0 ? -1.0f : std::min(a, b);
    }
    const float chosen = best([&] { return copy.copy(dst, src, n); });
    if (chosen < 0) {
      printf("   failed\n");
      continue;
    }
    char label[16];
    if (n >= (size_t)1 << 20)
      snprintf(label, sizeof label, "%zu MB", n >> 20);
    else
      snprintf(label, sizeof label, "%zu KB", n >> 10);
    char w[24] = "-", p[24] = "-", wg[24] = "-", pg[24] = "-";
    if (whole > 0) {
      snprintf(w, sizeof w, "%.3f", whole / 1000);
      snprintf(wg, sizeof wg, "%.1f", n / whole / 1000);
    }
    if (pipelined > 0) {
      snprintf(p, sizeof p, "%.3f", pipelined / 1000);
      snprintf(pg, sizeof pg, "%.1f", n / pipelined / 1000);
    }
    char sp[16] = "-";
    if (whole > 0) snprintf(sp, sizeof sp, "%.2fx", whole / chosen);
    printf("%8s %7zuK | %10s %7s | %15s %7s | %6.3f ms %7s  (%s)\n", label, piece >> 10, w, wg, p, pg,
           chosen / 1000, sp, piece != 0 ? "pieces" : "whole");
  }
  printf("\n(speedup: copy() against copy_whole, where copy_whole applies; copy() splits a copy from %zu KB up into\n"
         " pieces of about an eighth of it, and copies smaller blocks whole)\n",
         hostcopy::HostCopy::kMinPipelined >> 10);

  copy.term();
  zeMemFree(ctx, src);
  zeMemFree(ctx, dst);
  return 0;
}

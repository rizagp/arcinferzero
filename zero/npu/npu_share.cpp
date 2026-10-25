// arcinferzero NPU experiment, step 5: one block of system memory seen by both the GPU's and the NPU's Level Zero
// contexts (ZE_extension_external_memmap_sysmem: zeMemAllocHost with a descriptor that names existing memory), so the
// GPU could write the int8 activations a graph reads and read back its output without copies. The descriptor's
// structure type is found here (the published header is not on this machine): a candidate counts only when the
// pointer zeMemAllocHost returns writes the same bytes as the original memory.
// usage: npu_share
#include "../ze_min.h"
#include <cstring>
#include <vector>

struct ze_external_memmap_sysmem_ext_desc_t {
  uint32_t stype;
  const void* pNext;
  const void* pSystemMemory;
  uint64_t size;
};

static bool find_ctx(int type, ze_driver_handle_t& drv, ze_device_handle_t& dev, ze_context_handle_t& ctx) {
  uint32_t nd = 0;
  zeDriverGet(&nd, nullptr);
  std::vector<ze_driver_handle_t> ds(nd);
  zeDriverGet(&nd, ds.data());
  for (auto d : ds) {
    uint32_t n = 1;
    ze_device_handle_t x = nullptr;
    if (zeDeviceGet(d, &n, &x) != 0 || !n) continue;
    ze_device_properties_t p{};
    p.stype = ZE_STYPE_DEVICE_PROPERTIES;
    zeDeviceGetProperties(x, &p);
    if ((int)p.type != type) continue;
    drv = d;
    dev = x;
    ze_context_desc_t cd{ZE_STYPE_CONTEXT_DESC, nullptr, 0};
    return zeContextCreate(d, &cd, &ctx) == 0;
  }
  return false;
}

// the stype under which zeMemAllocHost maps `mem` itself, or 0
static uint32_t probe(ze_context_handle_t ctx, const char* who, uint8_t* mem, size_t bytes, void** mapped) {
  for (uint32_t st = 0x00020000u; st <= 0x00020060u; ++st) {
    ze_external_memmap_sysmem_ext_desc_t ed{st, nullptr, mem, bytes};
    ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, &ed, 0};
    void* p = nullptr;
    if (zeMemAllocHost(ctx, &hd, bytes, 4096, &p) != 0 || !p) continue;
    memset(mem, 0, 64);
    ((volatile uint8_t*)p)[7] = 0x5a;  // through the mapping, seen in the original memory?
    const bool same = mem[7] == 0x5a;
    if (same) {
      printf("%s: stype 0x%08x maps the memory (pointer %p, memory %p)\n", who, st, p, (void*)mem);
      *mapped = p;
      return st;
    }
    zeMemFree(ctx, p);
  }
  printf("%s: no stype in 0x00020000..0x00020060 maps the memory\n", who);
  return 0;
}

int main() {
  if (!ze_load() || zeInit(0) != ZE_RESULT_SUCCESS) { fprintf(stderr, "Level Zero init failed\n"); return 1; }
  ze_driver_handle_t gdrv = nullptr, ndrv = nullptr;
  ze_device_handle_t gdev = nullptr, ndev = nullptr;
  ze_context_handle_t gctx = nullptr, nctx = nullptr;
  if (!find_ctx(1, gdrv, gdev, gctx) || !find_ctx(5, ndrv, ndev, nctx)) { fprintf(stderr, "GPU or NPU missing\n"); return 1; }
  const size_t bytes = 64u << 20;
  uint8_t* mem = (uint8_t*)VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (!mem) return 1;
  void *gp = nullptr, *np = nullptr;
  const uint32_t gst = probe(gctx, "GPU", mem, bytes, &gp), nst = probe(nctx, "NPU", mem, bytes, &np);
  if (!gst || !nst) return 1;
  // GPU writes through its mapping (a fill on the GPU), the NPU's mapping and the CPU see it
  ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0, 0, 0};
  ze_command_list_handle_t cl = nullptr;
  if (zeCommandListCreateImmediate(gctx, gdev, &qd, &cl) != 0) return 1;
  const uint8_t pat = 0xa7;
  if (zeCommandListAppendMemoryFill(cl, gp, &pat, 1, bytes, nullptr, 0, nullptr) != 0 || zeCommandListHostSynchronize(cl, UINT64_MAX) != 0) {
    fprintf(stderr, "GPU fill failed\n");
    return 1;
  }
  size_t bad = 0;
  for (size_t i = 0; i < bytes; i += 4096) bad += mem[i] != pat || ((uint8_t*)np)[i + 1] != pat;
  printf("GPU fill of %zu MB through its mapping: %s (pages wrong: %zu)\n", bytes >> 20, bad ? "NOT seen" : "seen by the CPU and the NPU mapping", bad);
  return bad ? 1 : 0;
}

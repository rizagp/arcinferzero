// arcinferzero NPU feasibility probe (QUEUE F30): every Level Zero driver with its extensions and every device
// (type, name, clock, memory), so the NPU (Meteor Lake "AI Boost", device type VPU) and the graph extension it runs
// compiled models through can be found. The engine itself initializes Level Zero for GPUs only (zeInit(1)).
// usage: npu_probe
#include "../ze_min.h"
#include <cstring>
#include <vector>

struct ze_driver_extension_properties_t { char name[256]; uint32_t version; };
typedef ze_result_t (*PFN_ext)(ze_driver_handle_t, uint32_t*, ze_driver_extension_properties_t*);

int main() {
  HMODULE h = LoadLibraryA("ze_loader.dll");
  if (!h) { fprintf(stderr, "cannot load ze_loader.dll\n"); return 1; }
  auto zeInit_ = (ze_result_t(*)(uint32_t))(void*)GetProcAddress(h, "zeInit");
  auto zeDriverGet_ = (ze_result_t(*)(uint32_t*, ze_driver_handle_t*))(void*)GetProcAddress(h, "zeDriverGet");
  auto zeDeviceGet_ = (ze_result_t(*)(ze_driver_handle_t, uint32_t*, ze_device_handle_t*))(void*)GetProcAddress(h, "zeDeviceGet");
  auto zeDeviceGetProperties_ =
      (ze_result_t(*)(ze_device_handle_t, ze_device_properties_t*))(void*)GetProcAddress(h, "zeDeviceGetProperties");
  auto zeDeviceGetMemoryProperties_ = (ze_result_t(*)(ze_device_handle_t, uint32_t*, ze_device_memory_properties_t*))(
      void*)GetProcAddress(h, "zeDeviceGetMemoryProperties");
  auto zeDriverGetExtensionProperties_ = (PFN_ext)(void*)GetProcAddress(h, "zeDriverGetExtensionProperties");
  if (!zeInit_ || !zeDriverGet_ || !zeDeviceGet_ || !zeDeviceGetProperties_) { fprintf(stderr, "missing entry points\n"); return 1; }
  const ze_result_t ri = zeInit_(0);  // (0: every device type; 1 would be GPUs only, 2 NPUs (VPUs) only)
  printf("zeInit(0): 0x%x\n", ri);
  if (ri != ZE_RESULT_SUCCESS) return 1;
  uint32_t nd = 0;
  zeDriverGet_(&nd, nullptr);
  std::vector<ze_driver_handle_t> drv(nd);
  zeDriverGet_(&nd, drv.data());
  printf("%u driver(s)\n", nd);
  static const char* kType[] = {"?", "GPU", "CPU", "FPGA", "MCA", "VPU (NPU)"};
  for (uint32_t d = 0; d < nd; ++d) {
    uint32_t ne = 0;
    std::vector<ze_driver_extension_properties_t> ext;
    if (zeDriverGetExtensionProperties_ && zeDriverGetExtensionProperties_(drv[d], &ne, nullptr) == 0 && ne) {
      ext.resize(ne);
      zeDriverGetExtensionProperties_(drv[d], &ne, ext.data());
    }
    uint32_t ndev = 0;
    zeDeviceGet_(drv[d], &ndev, nullptr);
    std::vector<ze_device_handle_t> dev(ndev);
    zeDeviceGet_(drv[d], &ndev, dev.data());
    printf("\ndriver %u: %u device(s), %u extension(s)\n", d, ndev, ne);
    for (uint32_t i = 0; i < ndev; ++i) {
      ze_device_properties_t p{};
      p.stype = ZE_STYPE_DEVICE_PROPERTIES;
      zeDeviceGetProperties_(dev[i], &p);
      printf("  device %u: %s, type %s, vendor 0x%x id 0x%x, clock %u MHz, max alloc %.1f GiB, %u slices x %u sub x %u EU\n", i,
             p.name, p.type < 6 ? kType[p.type] : "?", p.vendorId, p.deviceId, p.coreClockRate,
             p.maxMemAllocSize / 1073741824.0, p.numSlices, p.numSubslicesPerSlice, p.numEUsPerSubslice);
      if (zeDeviceGetMemoryProperties_) {
        uint32_t nm = 0;
        zeDeviceGetMemoryProperties_(dev[i], &nm, nullptr);
        std::vector<ze_device_memory_properties_t> mp(nm);
        for (auto& m : mp) m.stype = ZE_STYPE_DEVICE_MEMORY_PROPERTIES;
        if (nm) zeDeviceGetMemoryProperties_(dev[i], &nm, mp.data());
        for (const auto& m : mp)
          printf("    memory \"%s\": %.1f GiB, clock %u MHz, bus %u bits\n", m.name, m.totalSize / 1073741824.0, m.maxClockRate,
                 m.maxBusWidth);
      }
    }
    for (const auto& e : ext)
      if (strstr(e.name, "graph") || strstr(e.name, "Graph") || strstr(e.name, "npu") || strstr(e.name, "NPU") ||
          strstr(e.name, "mutable") || strstr(e.name, "command_queue_npu"))
        printf("  extension %s (version 0x%x)\n", e.name, e.version);
    printf("  (all extensions:");
    for (const auto& e : ext) printf(" %s", e.name);
    printf(")\n");
  }
  return 0;
}

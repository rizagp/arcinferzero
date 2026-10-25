// arcinferzero NPU experiment, step 2: the NPU driver's graph extension table and its device graph properties
// (graph extension version, compiler version, graph formats, the newest OpenVINO opset the compiler takes): sensible
// values here mean the declarations in ze_npu.h line up with the driver. usage: npu_graph_props
#include "ze_npu.h"
#include <vector>

int main() {
  HMODULE h = LoadLibraryA("ze_loader.dll");
  if (!h) { fprintf(stderr, "cannot load ze_loader.dll\n"); return 1; }
#define P(name, type) auto name = (type)(void*)GetProcAddress(h, #name)
  P(zeInit, ze_result_t (*)(uint32_t));
  P(zeDriverGet, ze_result_t (*)(uint32_t*, ze_driver_handle_t*));
  P(zeDeviceGet, ze_result_t (*)(ze_driver_handle_t, uint32_t*, ze_device_handle_t*));
  P(zeDeviceGetProperties, ze_result_t (*)(ze_device_handle_t, ze_device_properties_t*));
  P(zeDriverGetExtensionFunctionAddress, ze_result_t (*)(ze_driver_handle_t, const char*, void**));
#undef P
  if (!zeInit || !zeDriverGetExtensionFunctionAddress) { fprintf(stderr, "missing entry points\n"); return 1; }
  if (zeInit(0) != ZE_RESULT_SUCCESS) { fprintf(stderr, "zeInit failed\n"); return 1; }
  uint32_t nd = 0;
  zeDriverGet(&nd, nullptr);
  std::vector<ze_driver_handle_t> drv(nd);
  zeDriverGet(&nd, drv.data());
  for (uint32_t d = 0; d < nd; ++d) {
    uint32_t n = 1;
    ze_device_handle_t dev = nullptr;
    if (zeDeviceGet(drv[d], &n, &dev) != 0 || !n) continue;
    ze_device_properties_t p{};
    p.stype = ZE_STYPE_DEVICE_PROPERTIES;
    zeDeviceGetProperties(dev, &p);
    if (p.type != 5) continue;  // (5: VPU, the NPU)
    printf("NPU: %s (driver %u)\n", p.name, d);
    for (const char* name : {"ZE_extension_graph_1_8", "ZE_extension_graph_1_4", "ZE_extension_graph"}) {
      ze_graph_dditable_t* t = nullptr;
      const ze_result_t r = zeDriverGetExtensionFunctionAddress(drv[d], name, (void**)&t);
      printf("  %s: 0x%x, table %p\n", name, r, (void*)t);
      if (r != 0 || !t) continue;
      ze_device_graph_properties_t gp{};
      gp.stype = ZE_STYPE_DEVICE_GRAPH_PROPERTIES;
      const ze_result_t rg = t->pfnDeviceGetGraphProperties(dev, &gp);
      printf("    device graph properties 0x%x: extension version 0x%x, compiler %u.%u, formats 0x%x, max opset %u\n", rg,
             gp.graphExtensionVersion, gp.compilerVersion.major, gp.compilerVersion.minor, gp.graphFormatsSupported,
             gp.maxOVOpsetVersionSupported);
      break;
    }
  }
  return 0;
}

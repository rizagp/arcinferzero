// arcinferzero NPU experiment: the NPU through Level Zero's graph extension (no OpenVINO): device and context,
// graphs compiled from npu::Ir with a blob cache (the driver's compiler takes 10-60 s for a large graph: its native
// binary is saved under %LOCALAPPDATA%\arcinferzero\npu and loaded as ZE_GRAPH_FORMAT_NATIVE next time), memory
// shared with another Level Zero context (external memmap), arguments by name, runs on an immediate command list.
#pragma once
#include "npu_ir.h"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace npu {

// ze_external_memmap_sysmem_ext_desc_t (ZE_extension_external_memmap_sysmem): zeMemAllocHost on existing memory;
// the structure type found by zero/npu/npu_share on this driver
struct ExtMemDesc { uint32_t stype; const void* pNext; const void* pSystemMemory; uint64_t size; };
constexpr uint32_t kExtMemStype = 0x00020037u;

// existing page-aligned system memory as host USM of context ctx (the same address), or nullptr
inline void* import_memory(ze_context_handle_t ctx, void* mem, size_t bytes) {
  ExtMemDesc ed{kExtMemStype, nullptr, mem, bytes};
  ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, &ed, 0};
  void* p = nullptr;
  if (zeMemAllocHost(ctx, &hd, bytes, 4096, &p) != 0 || p != mem) return nullptr;
  return p;
}

inline uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  for (size_t i = 0; i < n; ++i) h = (h ^ ((const uint8_t*)p)[i]) * 1099511628211ull;
  return h;
}

struct Rt {
  ze_driver_handle_t drv = nullptr;
  ze_device_handle_t dev = nullptr;
  ze_context_handle_t ctx = nullptr;
  ze_command_list_handle_t cl = nullptr;
  ze_graph_dditable_t* g = nullptr;
  ze_device_graph_properties_t gp{};
  std::string cache_dir;
  std::vector<void*> blobs_;  // (native binaries of the graphs loaded from the cache)
  // (needs ze_load() and zeInit(0) done: zeInit(1), GPUs only, hides the NPU driver)
  bool init() {
    auto getExt = (ze_result_t(*)(ze_driver_handle_t, const char*, void**))(void*)GetProcAddress(
        GetModuleHandleA("ze_loader.dll"), "zeDriverGetExtensionFunctionAddress");
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
      if (p.type == 5) { drv = d; dev = x; }
    }
    if (!dev || !getExt || getExt(drv, "ZE_extension_graph_1_8", (void**)&g) != 0 || !g) return false;
    gp.stype = ZE_STYPE_DEVICE_GRAPH_PROPERTIES;
    g->pfnDeviceGetGraphProperties(dev, &gp);
    ze_context_desc_t cd{ZE_STYPE_CONTEXT_DESC, nullptr, 0};
    if (zeContextCreate(drv, &cd, &ctx) != 0) return false;
    ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0, 0, 0};
    if (zeCommandListCreateImmediate(ctx, dev, &qd, &cl) != 0) return false;
    if (const char* la = getenv("LOCALAPPDATA")) cache_dir = std::string(la) + "\\arcinferzero\\npu";
    return true;
  }
  // compiled graph of ir: from the blob cache when it holds one for this model text and compiler, else compiled (and
  // saved); `cached` tells which
  ze_graph_handle_t compile(const Ir& ir, double& ms, bool& cached) {
    const std::vector<uint8_t> in = ir.serialize(gp.compilerVersion);
    char hex[17];
    snprintf(hex, sizeof hex, "%016llx", (unsigned long long)fnv1a(in.data(), in.size()));
    const std::string path = cache_dir.empty() ? std::string() : cache_dir + "\\" + hex + ".blob";
    std::vector<uint8_t> blob;
    if (!path.empty()) {
      std::ifstream f(path, std::ios::binary);
      if (f) blob.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    const auto t0 = std::chrono::steady_clock::now();
    ze_graph_handle_t gr = nullptr;
    cached = false;
    if (!blob.empty()) {
      // (the driver reads the native binary in place for the graph's lifetime: page-aligned memory, kept)
      uint8_t* ab = (uint8_t*)VirtualAlloc(nullptr, blob.size(), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
      if (ab) {
        memcpy(ab, blob.data(), blob.size());
        ze_graph_desc_t gd{ZE_STYPE_GRAPH_DESC, nullptr, ZE_GRAPH_FORMAT_NATIVE, blob.size(), ab, ""};
        cached = g->pfnCreate(ctx, dev, &gd, &gr) == 0;
        if (!cached) { gr = nullptr; VirtualFree(ab, 0, MEM_RELEASE); }  // (another driver's blob: compiled below)
        else blobs_.push_back(ab);
      }
    }
    if (!gr) {
      ze_graph_desc_t gd{ZE_STYPE_GRAPH_DESC, nullptr, ZE_GRAPH_FORMAT_NGRAPH_LITE, in.size(), in.data(), ""};
      if (g->pfnCreate(ctx, dev, &gd, &gr) != 0) {
        uint32_t ln = 0;
        if (g->pfnBuildLogGetString && g->pfnBuildLogGetString(gr, &ln, nullptr) == 0 && ln) {
          std::string log(ln, '\0');
          g->pfnBuildLogGetString(gr, &ln, log.data());
          fprintf(stderr, "NPU build log: %s\n", log.c_str());
        }
        return nullptr;
      }
      size_t bn = 0;
      if (!path.empty() && g->pfnGetNativeBinary(gr, &bn, nullptr) == 0 && bn) {
        std::vector<uint8_t> nb(bn);
        if (g->pfnGetNativeBinary(gr, &bn, nb.data()) == 0) {
          std::error_code ec;
          std::filesystem::create_directories(cache_dir, ec);
          const std::string tmp = path + ".tmp";
          bool ok = false;
          {
            std::ofstream o(tmp, std::ios::binary);
            o.write((const char*)nb.data(), (std::streamsize)bn);
            o.close();
            ok = !o.fail();
          }
          if (ok) std::filesystem::rename(tmp, path, ec);
          else std::filesystem::remove(tmp, ec);
        }
      }
    }
    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return gr;
  }
  void* host(size_t bytes) {
    ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
    void* p = nullptr;
    return zeMemAllocHost(ctx, &hd, bytes, 4096, &p) == 0 ? p : nullptr;
  }
  int arg(ze_graph_handle_t gr, const char* name, bool output = false) {
    ze_graph_properties_t pr{ZE_STYPE_GRAPH_PROPERTIES, nullptr, 0};
    g->pfnGetProperties(gr, &pr);
    for (uint32_t i = 0; i < pr.numGraphArgs; ++i) {
      ze_graph_argument_properties_t ap{};
      ap.stype = ZE_STYPE_GRAPH_ARGUMENT_PROPERTIES;
      g->pfnGetArgumentProperties(gr, i, &ap);
      if ((ap.type != 0) == output && !strcmp(ap.name, name)) return (int)i;
    }
    return -1;
  }
  bool set(ze_graph_handle_t gr, const char* name, void* p, bool output = false) {
    const int i = arg(gr, name, output);
    if (i < 0) fprintf(stderr, "NPU graph has no %s %s\n", output ? "output" : "input", name);
    return i >= 0 && g->pfnSetArgumentValue(gr, (uint32_t)i, p) == 0;
  }
  // output arguments in order of the graph (their names are the producing layers' output names)
  std::vector<int> outputs(ze_graph_handle_t gr) {
    std::vector<int> r;
    ze_graph_properties_t pr{ZE_STYPE_GRAPH_PROPERTIES, nullptr, 0};
    g->pfnGetProperties(gr, &pr);
    for (uint32_t i = 0; i < pr.numGraphArgs; ++i) {
      ze_graph_argument_properties_t ap{};
      ap.stype = ZE_STYPE_GRAPH_ARGUMENT_PROPERTIES;
      g->pfnGetArgumentProperties(gr, i, &ap);
      if (ap.type != 0) r.push_back((int)i);
    }
    return r;
  }
  bool initialize(ze_graph_handle_t gr) {
    const ze_result_t a = g->pfnAppendGraphInitialize(cl, gr, nullptr, 0, nullptr);
    const ze_result_t s = a == 0 ? zeCommandListHostSynchronize(cl, UINT64_MAX) : 0;
    if (a != 0 || s != 0) fprintf(stderr, "NPU graph initialize: append 0x%x, synchronize 0x%x\n", a, s);
    return a == 0 && s == 0;
  }
  bool start(ze_graph_handle_t gr) { return g->pfnAppendGraphExecute(cl, gr, nullptr, nullptr, 0, nullptr) == 0; }
  bool wait() { return zeCommandListHostSynchronize(cl, UINT64_MAX) == 0; }
};

}  // namespace npu

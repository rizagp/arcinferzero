// inferzero: the subset of the OpenCL API the engine uses (it began as an OpenCL engine), implemented on Level Zero, so that
// zero/engine.h (begun as a copy of the OpenCL engine's src/engine.h) runs unchanged on top of it. Buffers are device USM allocations (a sub-buffer is
// a pointer into its parent), commands go to one in-order immediate command list (the same order as the former OpenCL engine's in-order
// OpenCL queue, hence the same results), and kernels are the native binaries of the OpenCL compiler: built through
// OpenCL.dll (loaded under other names) and the kernel cache (%USERPROFILE%\.inferzero\kernels) is the one the OpenCL engine used.
#pragma once
#include "../src/app_dir.h"
#include "ze_min.h"
#include <dxgi1_4.h>
#pragma comment(lib, "dxgi.lib")
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_uint cl_bool;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_map_flags;
typedef cl_ulong cl_queue_properties;
typedef cl_uint cl_device_info;
typedef cl_uint cl_profiling_info;
typedef cl_uint cl_buffer_create_type;

namespace zrt { struct Dev; }
struct ZMem {
  char* ptr;
  size_t size;
  bool owner;
  zrt::Dev* d = nullptr;  // (the GPU it was made on: freed there)
};
struct ZKernel {
  ze_kernel_handle_t k;
  std::string name;
  uint32_t gx = 0, gy = 0, gz = 0;  // group size last set
  // the value each argument was last given (as passed to Level Zero: a buffer as its device pointer then, which the
  // engine's slots repoint): the same value again is not set again (a launch keeps the arguments it was appended with)
  // (gen: zrt::mem_gen when it was set. Level Zero takes a buffer argument's allocation when it is set, not only its
  //  address: after a free, a new buffer at the same address must be set again, see zrt::mem_free)
  struct Arg { uint8_t sz = 0; bool set = false; uint32_t gen = 0; unsigned char v[16]; };
  std::vector<Arg> args;
};
struct ZProg { ze_module_handle_t m; bool indirect = false; std::string opts; };  // indirect: built with -DXCACHE (see kernel())
typedef ZMem* cl_mem;
typedef ZKernel* cl_kernel;
typedef ZProg* cl_program;
typedef struct _z_queue* cl_command_queue;
typedef struct _z_context* cl_context;
typedef struct _z_device* cl_device_id;
typedef struct _z_platform* cl_platform_id;
typedef struct _z_event* cl_event;

#define CL_SUCCESS 0
#define CL_TRUE 1
#define CL_FALSE 0
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_USE_HOST_PTR (1 << 3)
#define CL_MEM_ALLOC_HOST_PTR (1 << 4)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_MAP_READ (1 << 0)
#define CL_MAP_WRITE (1 << 1)
#define CL_BUFFER_CREATE_TYPE_REGION 0x1220
#define CL_PROFILING_COMMAND_START 0x1282
#define CL_PROFILING_COMMAND_END 0x1283
struct cl_buffer_region { size_t origin, size; };

#define CL_CHECK(expr)                                                                       \
  do {                                                                                       \
    cl_int _e = (expr);                                                                      \
    if (_e != CL_SUCCESS) {                                                                  \
      fprintf(stderr, "inferzero error %d at %s:%d: %s\n", _e, __FILE__, __LINE__, #expr); \
      fflush(stderr);                                                                        \
      _Exit(1); /* (as ZE_CHECK: exit() runs destructors under running threads) */            \
    }                                                                                        \
  } while (0)

// ---- the runtime state: one context for every GPU of the driver; per GPU (Dev) its device, one in-order immediate
// command list, its buffers. A thread works on zrt::D(): the device its zrt::cur names, device 0 when it set none (the
// server's threads, Whisper / Qwen3-ASR, the expert prefetch); a multi-GPU shard's thread sets its own (engine_tp.h).
namespace zrt {
struct Mapping;
struct Dev {
  ze_device_handle_t dev = nullptr;
  ze_command_list_handle_t cl = nullptr;
  std::unordered_set<ZMem*> mems;
  uint64_t global_mem = 0;
  uint64_t device_bytes = 0;  // device USM allocated by the engine
  char* stage = nullptr;      // host USM bounce buffer for reads / writes
  size_t stage_n = 0;
  ze_command_list_handle_t cl_copy = nullptr;  // (see copy_list)
  const char* cl_copy_kind = "";
  ze_event_handle_t copy_ev[4] = {};  // (see copy_event)
  std::vector<ze_event_handle_t> ev_free;  // (see ev_get)
  double ev_ns_per_tick = 0;
  uint64_t ev_mask = ~0ull;
  std::vector<Mapping>* maps = nullptr;  // (see clEnqueueMapBuffer)
  int index = 0;
  // (a split model's GPUs, zero/mgpu/engine_tp.h: sync() gives up after this long, or once *abort is set because
  //  another GPU failed, instead of waiting forever for a GPU that hangs; one GPU: no limit, as always)
  // (atomic: TpEngine sets and restores them while a thread it gave up on may still be waiting in sync())
  std::atomic<uint64_t> sync_timeout_ns{UINT64_MAX};
  std::atomic<const std::atomic<bool>*> abort{nullptr};
  std::atomic<std::atomic<bool>*> fault{nullptr};  // (set when a wait gave up on its GPU: the split model is not trusted again)
  // (ARC_BATCH, see zrt::batch_max: the regular lists kernels are recorded into, each appended to `cl` when full and
  //  signalling `done` when it has run; `rec` the one being recorded, rec_n kernels in it, rec_cap its size)
  struct RecList { ze_command_list_handle_t l = nullptr; ze_event_handle_t done = nullptr; bool used = false; };
  std::vector<RecList> rec_pool;
  int rec_next = 0, rec_cur = 0, rec_n = 0, rec_cap = 0;
  ze_command_list_handle_t rec = nullptr;
};
// a wait that gave up (Dev::sync_timeout_ns / abort): the split model's shards are out of step afterwards
struct GpuWaitError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
inline ze_context_handle_t ctx = nullptr;
inline ze_driver_handle_t driver = nullptr;
inline Dev dev0;
inline std::vector<Dev*> devs{&dev0};  // devs[i]: GPU i (open_devices)
inline thread_local Dev* cur = nullptr;
// (a thread doing a split model's work: its shard thread, or the server's thread inside a mirrored call. Only its
//  waits take the GPU's limit and abort: Whisper / Qwen3-ASR on GPU 0 wait as always, and their long waits never
//  mark the split model faulty)
inline thread_local bool tp_guarded = false;
// (a split model's shard being made: an allocation that fails throws instead of ending the process, so the server
//  can load the model on one GPU instead. ARC_TP_OOM=r (tests): the first allocation of 256 MB or more on shard r fails)
inline thread_local bool oom_throws = false;
inline thread_local bool oom_test = false;
inline Dev& D() { return cur ? *cur : dev0; }
// Every free of Level Zero memory the engine makes goes through mem_free, which counts it: clSetKernelArg's cache
// takes an argument as unchanged only within the same count. Level Zero keeps the allocation a buffer argument was
// set with, so a buffer freed and another made at the same address (the allocator reuses them) left a kernel whose
// cached argument still named the freed allocation: the next launch read it inside zeCommandListAppendLaunchKernel
// and the process ended, 0xC0000005 in ze_intel_gpu64.dll (2026-11-04, and three times on 2026-11-10 at the start of a
// meeting's transcription: Qwen3-ASR's encoder grows its buffers after the server's 1-s warm-up clip)
inline std::atomic<uint32_t> mem_gen{1};
inline ze_result_t mem_free(ze_context_handle_t c, void* p) {
  const ze_result_t r = zeMemFree(c, p);
  mem_gen.fetch_add(1, std::memory_order_relaxed);
  return r;
}
// Called when the GPU runs out of memory (the share is shared with every other program: a browser or the desktop
// taking more makes a launch or allocation fail): frees some memory (the engine shrinks its expert cache) and returns
// true, or false when it cannot; the failed call is then retried.
inline bool (*oom_hook)() = nullptr;

// ---- ARC_BATCH=n (off by default): kernels recorded into regular in-order command lists of up to n kernels, each
// appended to the immediate list when full (zeCommandListImmediateAppendCommandListsExp, Level Zero 1.9), instead of
// each kernel appended there on its own. Same commands in the same order on the same list: the same results. What
// changes is how the GPU gets them: on a Pro B60 (bench/replay_probe.cpp, 2026-11-06) 700 small kernels took 2.56 us
// each appended one by one against 0.66 replayed from a recorded list, ~3 ms of a 27B token. The lists start at 4
// kernels after every wait (an idle GPU gets work soon) and double up to n. Anything else that goes to the
// immediate list (copies, fills, barriers, waits) first appends the list being recorded (imm(), sync()). Kernels
// that signal an event (profiling, ARC_ZE_WATCH) and ARC_ZE_SYNC_EACH go to the immediate list as before.
inline int batch_max() {
  static const int n = getenv("ARC_BATCH") ? std::max(0, std::min(4096, atoi(getenv("ARC_BATCH")))) : 0;
  return n;
}
struct BatchFns {
  struct ListDesc { ze_structure_type_t stype; const void* pNext; uint32_t ordinal, flags; };
  ze_result_t(__stdcall* create)(ze_context_handle_t, ze_device_handle_t, const ListDesc*, ze_command_list_handle_t*) = nullptr;
  ze_result_t(__stdcall* close)(ze_command_list_handle_t) = nullptr;
  ze_result_t(__stdcall* reset)(ze_command_list_handle_t) = nullptr;
  ze_result_t(__stdcall* append)(ze_command_list_handle_t, uint32_t, ze_command_list_handle_t*, ze_event_handle_t, uint32_t,
                                 ze_event_handle_t*) = nullptr;
  bool ok = false;
};
inline BatchFns& batch_fns() {
  static BatchFns f = [] {
    BatchFns b;
    HMODULE zl = GetModuleHandleA("ze_loader.dll");
    if (!zl || !driver) return b;
    b.create = (decltype(b.create))GetProcAddress(zl, "zeCommandListCreate");
    b.close = (decltype(b.close))GetProcAddress(zl, "zeCommandListClose");
    b.reset = (decltype(b.reset))GetProcAddress(zl, "zeCommandListReset");
    b.append = (decltype(b.append))GetProcAddress(zl, "zeCommandListImmediateAppendCommandListsExp");
    if (!b.append) {
      typedef ze_result_t(__stdcall * PFN_ext)(ze_driver_handle_t, const char*, void**);
      auto ext = (PFN_ext)GetProcAddress(zl, "zeDriverGetExtensionFunctionAddress");
      void* p = nullptr;
      if (ext && ext(driver, "zeCommandListImmediateAppendCommandListsExp", &p) == 0) b.append = (decltype(b.append))p;
    }
    b.ok = b.create && b.close && b.reset && b.append;
    fprintf(stderr, b.ok ? "inferzero: ARC_BATCH=%d: kernels recorded in lists of up to %d\n"
                         : "inferzero: ARC_BATCH=%d: this driver cannot append lists to an immediate list; off\n",
            batch_max(), batch_max());
    return b;
  }();
  return f;
}
inline bool batching() { return batch_max() > 0 && batch_fns().ok; }
// the list being recorded appended to d's immediate list (no-op when none)
inline void flush(Dev& d) {
  if (!d.rec) return;
  ze_command_list_handle_t l = d.rec;
  d.rec = nullptr;  // (before anything that could wait: an oom_hook that synchronizes comes back here)
  d.rec_n = 0;
  d.rec_cap = std::min(d.rec_cap * 2, batch_max());
  ZE_CHECK(batch_fns().close(l));
  Dev::RecList& r = d.rec_pool[d.rec_cur];
  ze_result_t e = batch_fns().append(d.cl, 1, &l, r.done, 0, nullptr);
  for (int i = 0; e == (ze_result_t)0x70000003 && i < 8 && oom_hook && oom_hook(); ++i)  // (out of device memory)
    e = batch_fns().append(d.cl, 1, &l, r.done, 0, nullptr);
  ZE_CHECK(e);
  r.used = true;
}
// d's immediate list, for a command that is not a kernel: what was recorded goes first
inline ze_command_list_handle_t imm(Dev& d) { flush(d); return d.cl; }
inline ze_command_list_handle_t imm() { return imm(D()); }
// the list to record d's next kernel into: a free one of the pool (64 lists; one whose earlier recording has not run
// yet is waited for, which only happens when the GPU is 64 lists behind)
inline void sync();
inline ze_command_list_handle_t rec_list(Dev& d) {
  if (d.rec) return d.rec;
  const int kPool = 64;
  if (d.rec_pool.empty()) {
    uint32_t ng = 0, ord = 0;
    ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(d.dev, &ng, nullptr));
    std::vector<ze_command_queue_group_properties_t> gp(ng);
    for (auto& g : gp) g.stype = ZE_STYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(d.dev, &ng, gp.data()));
    for (uint32_t i = 0; i < ng; ++i)
      if (gp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) { ord = i; break; }  // (the immediate list's group)
    ze_event_pool_desc_t pd{ZE_STYPE_EVENT_POOL_DESC, nullptr, 0x1 /* host visible */, (uint32_t)kPool};
    ze_event_pool_handle_t pool = nullptr;
    ZE_CHECK(zeEventPoolCreate(ctx, &pd, 1, &d.dev, &pool));
    d.rec_pool.resize(kPool);
    for (int i = 0; i < kPool; ++i) {
      BatchFns::ListDesc ld{0xfu /* ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC */, nullptr, ord, 8 /* in order */};
      ZE_CHECK(batch_fns().create(ctx, d.dev, &ld, &d.rec_pool[i].l));
      ze_event_desc_t ed{ZE_STYPE_EVENT_DESC, nullptr, (uint32_t)i, 0x4 /* signal: host scope */, 0};
      ZE_CHECK(zeEventCreate(pool, &ed, &d.rec_pool[i].done));
    }
  }
  d.rec_cur = d.rec_next;
  d.rec_next = (d.rec_next + 1) % kPool;
  Dev::RecList& r = d.rec_pool[d.rec_cur];
  if (r.used) {
    // (a split model's shard: through sync(), whose wait has the time limit and the abort of the other GPUs; it waits
    //  for the whole list, which holds this one's earlier recording. d is D() here: the only caller passes it)
    if (tp_guarded) sync();
    else ZE_CHECK(zeEventHostSynchronize(r.done, UINT64_MAX));
    ZE_CHECK(zeEventHostReset(r.done));
    ZE_CHECK(batch_fns().reset(r.l));
    r.used = false;
  }
  if (d.rec_cap < 4) d.rec_cap = std::min(4, batch_max());
  d.rec = r.l;
  return r.l;
}

// (the driver's UINT64_MAX wait does not spin a core: measured 2026-11-04, QUEUE 31.11 (1); polling with sleeps
//  instead gained nothing)
inline std::atomic<uint64_t> sync_n{0};  // (host waits, counted with ARC_LAUNCH_STATS)
inline bool launch_stats_on();
inline void sync() {
  Dev& d = D();
  if (launch_stats_on()) ++sync_n;
  if (d.rec) {
    flush(d);
    d.rec_cap = 0;  // (the GPU will be idle: the next lists start small again)
  }
  if (!tp_guarded || (d.sync_timeout_ns.load() == UINT64_MAX && !d.abort.load())) {
    ZE_CHECK(zeCommandListHostSynchronize(d.cl, UINT64_MAX));
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();  // (in pieces of 100 ms: see Dev::sync_timeout_ns)
  for (;;) {  // (the limit and abort read again each piece: TpEngine::disarm can call a wait off)
    const ze_result_t r = zeCommandListHostSynchronize(d.cl, 100000000ull);
    if (r == ZE_RESULT_SUCCESS) return;
    if (r != ZE_RESULT_NOT_READY) ZE_CHECK(r);
    const std::atomic<bool>* ab = d.abort.load();
    if (ab && ab->load(std::memory_order_relaxed))
      throw GpuWaitError("GPU " + std::to_string(d.index) + ": wait left, another GPU failed");
    const uint64_t lim = d.sync_timeout_ns.load();
    if ((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count() > lim) {
      if (std::atomic<bool>* f = d.fault.load()) f->store(true);
      throw GpuWaitError("GPU " + std::to_string(d.index) + ": its work did not finish within " +
                         std::to_string(lim / 1000000000ull) + " s (the GPU hangs?)");
    }
  }
}
// A second immediate command list for copies that run while `cl` executes kernels (the expert cache's prompt
// prefetch, xcache.h): on a copy engine when the device has one, else on another queue of the compute group.
inline ze_command_list_handle_t copy_list() {
  if (D().cl_copy) return D().cl_copy;
  uint32_t ng = 0;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(D().dev, &ng, nullptr));
  std::vector<ze_command_queue_group_properties_t> gp(ng);
  for (auto& g : gp) g.stype = ZE_STYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(D().dev, &ng, gp.data()));
  int ord = -1, idx = 0;
  for (uint32_t i = 0; i < ng && ord < 0; ++i)  // a copy-only group
    if ((gp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) && !(gp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE))
      ord = (int)i, D().cl_copy_kind = "copy engine";
  for (uint32_t i = 0; i < ng && ord < 0; ++i)  // else the compute group, its second queue when it has one
    if (gp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)
      ord = (int)i, idx = gp[i].numQueues > 1 ? 1 : 0, D().cl_copy_kind = idx ? "second compute queue" : "compute queue";
  if (ord < 0) ord = 0;
  ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, (uint32_t)ord, (uint32_t)idx, 2 /* in order */,
                             2 /* asynchronous */, 0};
  ZE_CHECK(zeCommandListCreateImmediate(ctx, D().dev, &qd, &D().cl_copy));
  fprintf(stderr, "inferzero: expert copies on a %s (queue group %d, queue %d)\n", D().cl_copy_kind, ord, idx);
  return D().cl_copy;
}
// Events the copy list signals after a batch of copies: 0 and 1 (used alternately) the main list waits on before
// kernels read the copied memory (the wait, device scope, has the compute side invalidate its caches, which a host
// synchronization alone does not promise); 2 and 3 the copy thread waits on before it reuses a part of its staging.
inline ze_event_handle_t copy_event(int i) {
  if (!D().copy_ev[0]) {
    ze_event_pool_desc_t pd{ZE_STYPE_EVENT_POOL_DESC, nullptr, 0x1 /* host visible */, 4};
    ze_event_pool_handle_t pool = nullptr;
    ZE_CHECK(zeEventPoolCreate(ctx, &pd, 1, &D().dev, &pool));
    for (uint32_t k = 0; k < 4; ++k) {
      ze_event_desc_t ed{ZE_STYPE_EVENT_DESC, nullptr, k, 0x4 /* signal: host scope */, 0x2 /* wait: device scope */};
      ZE_CHECK(zeEventCreate(pool, &ed, &D().copy_ev[k]));
    }
  }
  return D().copy_ev[i & 3];
}
// this thread's GPU shares the system RAM (an iGPU: ZE_DEVICE_PROPERTY_FLAG_INTEGRATED)
inline bool integrated() {
  ze_device_properties_t dp{};
  dp.stype = ZE_STYPE_DEVICE_PROPERTIES;
  return zeDeviceGetProperties(D().dev, &dp) == 0 && (dp.flags & 1u);
}
// kernel timestamps for profiling (engine.profile: clEnqueueNDRangeKernel asked for an event): events come from pools
// of 4096 host-visible timestamp events and go back to a free list on clReleaseEvent
inline ze_event_handle_t ev_get() {
  if (D().ev_free.empty()) {
    if (D().ev_ns_per_tick == 0) {
      ze_device_properties_t dp{};
      dp.stype = ZE_STYPE_DEVICE_PROPERTIES;  // (this stype: timerResolution in ns per tick)
      ZE_CHECK(zeDeviceGetProperties(D().dev, &dp));
      D().ev_ns_per_tick = dp.timerResolution ? (double)dp.timerResolution : 1.0;
      if (dp.kernelTimestampValidBits && dp.kernelTimestampValidBits < 64) D().ev_mask = (1ull << dp.kernelTimestampValidBits) - 1;
    }
    const uint32_t n = 4096;
    ze_event_pool_desc_t pd{ZE_STYPE_EVENT_POOL_DESC, nullptr, 0x1 | 0x4 /* host visible, kernel timestamps */, n};
    ze_event_pool_handle_t pool;
    ZE_CHECK(zeEventPoolCreate(ctx, &pd, 1, &D().dev, &pool));
    for (uint32_t i = 0; i < n; ++i) {
      ze_event_desc_t ed{ZE_STYPE_EVENT_DESC, nullptr, i, 0x4 /* host scope */, 0};
      ze_event_handle_t e;
      ZE_CHECK(zeEventCreate(pool, &ed, &e));
      D().ev_free.push_back(e);
    }
  }
  ze_event_handle_t e = D().ev_free.back();
  D().ev_free.pop_back();
  return e;
}
inline void* host_alloc(size_t n) {
  ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
  void* p = nullptr;
  if (oom_throws) {
    if (zeMemAllocHost(ctx, &hd, n, 4096, &p) != ZE_RESULT_SUCCESS)
      throw std::runtime_error("host allocation of " + std::to_string(n >> 20) + " MB failed");
    return p;
  }
  ZE_CHECK(zeMemAllocHost(ctx, &hd, n, 4096, &p));
  return p;
}
inline char* staging(size_t n) {
  if (n > D().stage_n) {
    // (the new one first, through host_alloc: a split model's shard short of memory throws (oom_throws) and leaves
    //  the old buffer in place)
    const size_t sz = std::max(n, (size_t)1 << 22);
    char* p = (char*)host_alloc(sz);
    if (D().stage) { sync(); mem_free(ctx, D().stage); }
    D().stage = p;
    D().stage_n = sz;
  }
  return D().stage;
}
inline void* device_alloc(size_t n) {
  ze_device_mem_alloc_desc_t dd{ZE_STYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  void* p = nullptr;
  ze_result_t r = ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
  if (oom_test && n >= (256u << 20)) oom_test = false;
  else r = zeMemAllocDevice(ctx, &dd, std::max<size_t>(n, 64), 4096, D().dev, &p);
  for (int i = 0; r != ZE_RESULT_SUCCESS && i < 8 && oom_hook && oom_hook(); ++i)
    r = zeMemAllocDevice(ctx, &dd, std::max<size_t>(n, 64), 4096, D().dev, &p);
  if (r != ZE_RESULT_SUCCESS) {
    char msg[160];
    snprintf(msg, sizeof msg, "device allocation of %.1f MB failed on GPU %d (0x%x; %.2f GiB allocated)", n / 1e6,
             D().index, r, D().device_bytes / 1073741824.0);
    if (oom_throws) throw std::runtime_error(msg);
    fprintf(stderr, "inferzero: %s\n", msg);
    fflush(stderr);
    _Exit(1);
  }
  D().device_bytes += n;
  return p;
}
// What Windows lets this process keep resident on the GPU now (WDDM budget, DXGI) and what it uses: the GPU share is
// an upper bound, the budget shrinks when the system is short of memory. Returns false without the numbers.
inline bool video_budget(uint64_t& budget, uint64_t& usage) {
  IDXGIFactory4* f = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&f))) return false;
  bool ok = false;
  IDXGIAdapter1* a1 = nullptr;
  for (UINT i = 0; !ok && f->EnumAdapters1(i, &a1) != DXGI_ERROR_NOT_FOUND; ++i) {
    DXGI_ADAPTER_DESC1 d;
    a1->GetDesc1(&d);
    IDXGIAdapter3* a3 = nullptr;
    if (d.VendorId == 0x8086 && SUCCEEDED(a1->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&a3))) {
      DXGI_QUERY_VIDEO_MEMORY_INFO m{};
      if (SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &m))) {
        budget = m.Budget;
        usage = m.CurrentUsage;
        ok = true;
      }
      a3->Release();
    }
    a1->Release();
  }
  f->Release();
  return ok;
}
inline int note_layer = -1, note_T = -1;  // (diagnostics: the layer and rows the engine is working on)
inline void print_memory() {
  if (note_layer >= 0) fprintf(stderr, "  (layer %d, %d rows)\n", note_layer, note_T);
  uint64_t b = 0, u = 0;
  MEMORYSTATUSEX ms{sizeof ms};
  GlobalMemoryStatusEx(&ms);
  if (video_budget(b, u))
    fprintf(stderr, "  GPU memory: Windows budget %.2f GiB, in use %.2f GiB (allocated by the engine %.2f GiB); RAM free %.2f GiB\n",
            b / 1073741824.0, u / 1073741824.0, D().device_bytes / 1073741824.0, ms.ullAvailPhys / 1073741824.0);
}
// memory the engine allocated itself (host USM, device pools) as a buffer kernels can take as an argument
inline ZMem* wrap(void* p, size_t n) {
  ZMem* m = new ZMem{(char*)p, n, false, &D()};
  D().mems.insert(m);
  return m;
}
struct Mapping { ZMem* m; size_t off, n; void* host; };
inline std::vector<Mapping>& maps() {
  if (!D().maps) D().maps = new std::vector<Mapping>;
  return *D().maps;
}

// OpenCL.dll, only to compile kernels (its native binaries load into Level Zero)
namespace rcl {
typedef cl_int(__stdcall* GetPlatformIDs)(cl_uint, void**, cl_uint*);
typedef cl_int(__stdcall* GetDeviceIDs)(void*, cl_ulong, cl_uint, void**, cl_uint*);
typedef cl_int(__stdcall* GetDeviceInfo)(void*, cl_uint, size_t, void*, size_t*);
typedef void*(__stdcall* CreateContext)(const intptr_t*, cl_uint, void* const*, void*, void*, cl_int*);
typedef void*(__stdcall* CreateProgramWithSource)(void*, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int(__stdcall* BuildProgram)(void*, cl_uint, void* const*, const char*, void*, void*);
typedef cl_int(__stdcall* GetProgramBuildInfo)(void*, void*, cl_uint, size_t, void*, size_t*);
typedef cl_int(__stdcall* GetProgramInfo)(void*, cl_uint, size_t, void*, size_t*);
typedef cl_int(__stdcall* ReleaseProgram)(void*);
inline GetPlatformIDs getPlatformIDs;
inline GetDeviceIDs getDeviceIDs;
inline GetDeviceInfo getDeviceInfo;
inline CreateContext createContext;
inline CreateProgramWithSource createProgramWithSource;
inline BuildProgram buildProgram;
inline GetProgramBuildInfo getProgramBuildInfo;
inline GetProgramInfo getProgramInfo;
inline ReleaseProgram releaseProgram;
inline void *device = nullptr, *context = nullptr;
inline bool load() {
  if (context) return true;  // (a second engine in the process: the same compiler context, no second DLL reference)
  HMODULE h = LoadLibraryA("OpenCL.dll");
  if (!h) return false;
#define RCL(var, name) var = (decltype(var))(void*)GetProcAddress(h, name); if (!var) { FreeLibrary(h); return false; }
  RCL(getPlatformIDs, "clGetPlatformIDs") RCL(getDeviceIDs, "clGetDeviceIDs") RCL(getDeviceInfo, "clGetDeviceInfo")
  RCL(createContext, "clCreateContext") RCL(createProgramWithSource, "clCreateProgramWithSource")
  RCL(buildProgram, "clBuildProgram") RCL(getProgramBuildInfo, "clGetProgramBuildInfo")
  RCL(getProgramInfo, "clGetProgramInfo") RCL(releaseProgram, "clReleaseProgram")
#undef RCL
  auto fail = [&] {  // (no compiler context: the DLL let go, a later call starts over)
    device = context = nullptr;
    FreeLibrary(h);
    return false;
  };
  cl_uint np = 0;
  if (getPlatformIDs(0, nullptr, &np) != 0 || !np) return fail();
  std::vector<void*> plats(np);
  getPlatformIDs(np, plats.data(), nullptr);
  for (void* p : plats) {
    cl_uint nd = 0;
    if (getDeviceIDs(p, 1u << 2, 1, &device, &nd) == 0 && nd) break;
  }
  if (!device) return fail();
  cl_int err = 0;
  context = createContext(nullptr, 1, &device, nullptr, nullptr, &err);
  if (err != 0 || !context) return fail();
  return true;
}
}  // namespace rcl
}  // namespace zrt

// ---- the OpenCL calls of the engine
inline cl_int clFinish(cl_command_queue) { zrt::sync(); return CL_SUCCESS; }
inline cl_int clFlush(cl_command_queue) { return CL_SUCCESS; }
inline cl_int clEnqueueCopyBuffer(cl_command_queue, cl_mem src, cl_mem dst, size_t so, size_t doff, size_t n, cl_uint,
                                  const cl_event*, cl_event* ev) {
  if (ev) *ev = nullptr;
  if (!n) return CL_SUCCESS;
  if (so + n > src->size || doff + n > dst->size) {  // (refused as OpenCL does, not copied over what follows)
    fprintf(stderr, "inferzero: copy of %zu bytes past a buffer's end (source %zu + %zu of %zu, destination %zu + %zu of %zu; refused)\n",
            n, so, n, src->size, doff, n, dst->size);
    return -30;
  }
  ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), dst->ptr + doff, src->ptr + so, n, nullptr, 0, nullptr));
  return CL_SUCCESS;
}
inline bool out_of_bounds(const char* what, cl_mem b, size_t off, size_t n) {
  if (b && off + n <= b->size) return false;
  fprintf(stderr, "inferzero: %s of %zu bytes at offset %zu past the end of a %zu-byte buffer (refused)\n", what, n,
          off, b ? b->size : (size_t)0);
  return true;
}
inline cl_int clEnqueueReadBuffer(cl_command_queue, cl_mem b, cl_bool, size_t off, size_t n, void* host, cl_uint,
                                  const cl_event*, cl_event* ev) {
  if (ev) *ev = nullptr;
  if (out_of_bounds("read", b, off, n)) return -30;
  char* s = zrt::staging(n);
  ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), s, b->ptr + off, n, nullptr, 0, nullptr));
  zrt::sync();
  memcpy(host, s, n);
  return CL_SUCCESS;
}
inline cl_int clEnqueueWriteBuffer(cl_command_queue, cl_mem b, cl_bool, size_t off, size_t n, const void* host, cl_uint,
                                   const cl_event*, cl_event* ev) {
  if (ev) *ev = nullptr;
  if (out_of_bounds("write", b, off, n)) return -30;
  char* s = zrt::staging(n);
  zrt::sync();  // (the bounce buffer may still be read by an earlier copy)
  memcpy(s, host, n);
  ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), b->ptr + off, s, n, nullptr, 0, nullptr));
  zrt::sync();
  return CL_SUCCESS;
}
inline cl_int clEnqueueFillBuffer(cl_command_queue, cl_mem b, const void* pattern, size_t ps, size_t off, size_t n,
                                  cl_uint, const cl_event*, cl_event* ev) {
  if (ev) *ev = nullptr;
  if (out_of_bounds("fill", b, off, n)) return -30;
  ZE_CHECK(zeCommandListAppendMemoryFill(zrt::imm(), b->ptr + off, pattern, ps, n, nullptr, 0, nullptr));
  zrt::sync();  // (the pattern is the caller's)
  return CL_SUCCESS;
}
inline void* clEnqueueMapBuffer(cl_command_queue, cl_mem b, cl_bool, cl_map_flags flags, size_t off, size_t n, cl_uint,
                                const cl_event*, cl_event* ev, cl_int* err) {
  if (ev) *ev = nullptr;
  void* h = zrt::host_alloc(n);
  if (flags & CL_MAP_READ) {
    ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), h, b->ptr + off, n, nullptr, 0, nullptr));
    zrt::sync();
  }
  zrt::maps().push_back({b, off, n, h});
  if (err) *err = CL_SUCCESS;
  return h;
}
inline cl_int clEnqueueUnmapMemObject(cl_command_queue, cl_mem b, void* p, cl_uint, const cl_event*, cl_event* ev) {
  if (ev) *ev = nullptr;
  for (size_t i = 0; i < zrt::maps().size(); ++i)
    if (zrt::maps()[i].host == p && zrt::maps()[i].m == b) {
      ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), b->ptr + zrt::maps()[i].off, p, zrt::maps()[i].n, nullptr, 0, nullptr));
      zrt::sync();
      zrt::mem_free(zrt::ctx, p);
      zrt::maps().erase(zrt::maps().begin() + i);
      return CL_SUCCESS;
    }
  return -30;
}
inline cl_mem clCreateBuffer(cl_context, cl_mem_flags flags, size_t n, void* host, cl_int* err) {
  ZMem* m = new ZMem{(char*)zrt::device_alloc(n), n, true, &zrt::D()};
  zrt::D().mems.insert(m);
  if ((flags & CL_MEM_COPY_HOST_PTR) && host) clEnqueueWriteBuffer(nullptr, m, CL_TRUE, 0, n, host, 0, nullptr, nullptr);
  if (err) *err = CL_SUCCESS;
  return m;
}
inline cl_mem clCreateSubBuffer(cl_mem parent, cl_mem_flags, cl_buffer_create_type, const void* region, cl_int* err) {
  const cl_buffer_region* r = (const cl_buffer_region*)region;
  if (r->origin + r->size > parent->size) {
    if (err) *err = -13;
    return nullptr;
  }
  ZMem* m = new ZMem{parent->ptr + r->origin, r->size, false, &zrt::D()};
  zrt::D().mems.insert(m);
  if (err) *err = CL_SUCCESS;
  return m;
}
inline cl_int clReleaseMemObject(cl_mem m) {
  if (!m) return CL_SUCCESS;
  zrt::Dev& d = m->d ? *m->d : zrt::D();
  if (m->owner) { zeCommandListHostSynchronize(zrt::imm(d), UINT64_MAX); zrt::mem_free(zrt::ctx, m->ptr); d.device_bytes -= m->size; }
  d.mems.erase(m);
  delete m;
  return CL_SUCCESS;
}
// a buffer argument (a registered cl_mem) becomes its USM pointer; anything else is passed as given
// ARC_LAUNCH_STATS=1 (diagnostics, runz prints them per generated token): kernel launches, and the host time spent
// setting their arguments and appending them (the CPU's share of a step, against the GPU's)
namespace zrt {
inline const bool launch_stats = (getenv("ARC_LAUNCH_STATS") && atoi(getenv("ARC_LAUNCH_STATS")) != 0);
inline bool launch_stats_on() { return launch_stats; }
inline std::atomic<uint64_t> launch_n{0}, launch_ns{0}, setarg_ns{0};
struct LaunchTimer {
  std::atomic<uint64_t>& acc;
  std::chrono::steady_clock::time_point t0 = launch_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  ~LaunchTimer() {
    if (launch_stats) acc += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
  }
};

// ARC_ZE_WATCH=ms (diagnostics: which kernel runs long or loses the device; slow, every launch waited for alone).
// Each launch signals an event of its own and is waited for on that event, not on the shared command list, so the
// time and an error belong to it and not to another engine's work. A launch over `ms` is printed with its global /
// local sizes and arguments ("slow kernel"); one still running after 1.5 s is printed while it runs (a TDR is coming:
// Windows resets the GPU at 2 s); an error prints the last 64 launches (name, sizes, arguments, time) and ends the
// process as before. The arguments are the values given to Level Zero (a buffer as its device address).
inline const long watch_ms = getenv("ARC_ZE_WATCH") ? atol(getenv("ARC_ZE_WATCH")) : 0;
struct WatchEntry {
  uint64_t seq = 0;
  std::string name;
  uint32_t gs[3]{}, ls[3]{};
  std::vector<ZKernel::Arg> args;
  double ms = -1;  // (-1: not finished)
};
inline std::mutex watch_mu;
inline WatchEntry watch_ring[64];
inline uint64_t watch_seq = 0;
inline std::string watch_args(const std::vector<ZKernel::Arg>& args) {
  std::string s;
  char b[48];
  for (size_t i = 0; i < args.size(); ++i) {
    const ZKernel::Arg& a = args[i];
    if (!a.set) snprintf(b, sizeof b, "%s?", i ? " " : "");
    else if (a.sz == 4) {
      int32_t v;
      memcpy(&v, a.v, 4);
      snprintf(b, sizeof b, "%s%d", i ? " " : "", v);
    } else if (a.sz == 8) {
      uint64_t v;
      memcpy(&v, a.v, 8);
      snprintf(b, sizeof b, "%s0x%llx", i ? " " : "", (unsigned long long)v);
    } else {
      snprintf(b, sizeof b, "%s(%u bytes)", i ? " " : "", (unsigned)a.sz);
    }
    s += b;
  }
  return s;
}
inline void watch_print(const WatchEntry& e, const char* what) {
  fprintf(stderr, "%s #%llu %s: %s ms, global %u x %u x %u, local %u x %u x %u, args %s\n", what, (unsigned long long)e.seq,
          e.name.c_str(), e.ms < 0 ? "not finished" : std::to_string((long long)std::llround(e.ms)).c_str(), e.gs[0], e.gs[1],
          e.gs[2], e.ls[0], e.ls[1], e.ls[2], watch_args(e.args).c_str());
}
// (called by clEnqueueNDRangeKernel once the launch is appended; keep: the caller asked for the event, else it goes
//  back to the pool)
inline void watch_wait(ZKernel* k, ze_event_handle_t ev, bool keep, const uint32_t gs[3], const uint32_t ls[3]) {
  // (this launch's own copy for its prints: the ring's slot is shared, another thread's launches can reuse it while
  //  this one waits; the time goes back into the slot only while it still holds this launch)
  WatchEntry e;
  e.name = k->name;
  for (int d = 0; d < 3; ++d) e.gs[d] = gs[d], e.ls[d] = ls[d];
  e.args = k->args;
  {
    std::lock_guard<std::mutex> g(watch_mu);
    e.seq = ++watch_seq;
    watch_ring[e.seq % 64] = e;
  }
  const auto t0 = std::chrono::steady_clock::now();
  bool warned = false;
  for (;;) {
    const ze_result_t r = zeEventHostSynchronize(ev, 50000000ull);  // (50 ms a piece)
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (r == ZE_RESULT_SUCCESS) {
      e.ms = ms;
      {
        std::lock_guard<std::mutex> g(watch_mu);
        if (watch_ring[e.seq % 64].seq == e.seq) watch_ring[e.seq % 64].ms = ms;
      }
      if (ms > (double)watch_ms) watch_print(e, "slow kernel");
      break;
    }
    if (r != ZE_RESULT_NOT_READY) {
      fprintf(stderr, "Level Zero error 0x%x waiting for kernel #%llu %s after %.0f ms; the last launches, oldest first:\n", r,
              (unsigned long long)e.seq, k->name.c_str(), ms);
      std::lock_guard<std::mutex> g(watch_mu);
      for (uint64_t s = watch_seq > 63 ? watch_seq - 63 : 1; s <= watch_seq; ++s) watch_print(watch_ring[s % 64], "  launch");
      if (ze_fail_hook) ze_fail_hook();
      fflush(stderr);
      _Exit(1);
    }
    if (!warned && ms > 1500) {
      warned = true;
      watch_print(e, "kernel still running after 1.5 s (a GPU timeout is near)");
      fflush(stderr);
    }
  }
  if (!keep) {
    zeEventHostReset(ev);
    D().ev_free.push_back(ev);
  }
}
}  // namespace zrt
inline cl_int clSetKernelArg(cl_kernel k, cl_uint i, size_t sz, const void* v) {
  zrt::LaunchTimer lt{zrt::setarg_ns};
  void* p = nullptr;
  if (sz == sizeof(void*) && v) {
    ZMem* m = *(ZMem* const*)v;
    if (m && zrt::D().mems.count(m)) {
      p = m->ptr;
      v = &p;
    }
  }
  // (ARC_ARG_CACHE=0: every argument set at every launch, as before. Not the cause of 2026-11-04's lost devices: the
  //  server lost it with the cache off too, at a GPU timeout (TDR, LiveKernelEvent 141))
  static const bool cache = !(getenv("ARC_ARG_CACHE") && atoi(getenv("ARC_ARG_CACHE")) == 0);
  ZKernel::Arg* a = nullptr;
  const uint32_t gen = zrt::mem_gen.load(std::memory_order_relaxed);  // (read before the set: a free meanwhile counts)
  // (the values are kept with the cache on, and with ARC_ZE_WATCH for its launch ring, where they are only recorded)
  if ((cache || zrt::watch_ms > 0) && v && sz <= sizeof(ZKernel::Arg::v) && i < 64) {  // (local memory, v null: set each time)
    if (k->args.size() <= i) k->args.resize(i + 1);
    a = &k->args[i];
    if (cache && a->set && a->sz == sz && a->gen == gen && memcmp(a->v, v, sz) == 0) return CL_SUCCESS;
  }
  if (zeKernelSetArgumentValue(k->k, i, sz, v) != ZE_RESULT_SUCCESS) {
    if (a) a->set = false;
    return -51;
  }
  if (a) {
    a->sz = (uint8_t)sz;
    a->gen = gen;
    memcpy(a->v, v, sz);
    a->set = true;
  }
  return CL_SUCCESS;
}
namespace zrt {
// the launch this thread is appending (src/crash_log.h prints it when the process crashes inside the driver's
// append: its kernel, sizes and arguments, and which 8-byte arguments lie in no live buffer of this GPU)
inline thread_local const ZKernel* last_k = nullptr;
inline thread_local uint32_t last_gs[3]{}, last_ls[3]{};
inline void print_last_launch() {
  if (!last_k) return;
  fprintf(stderr, "  the launch this thread was appending: %s, global %u x %u x %u, local %u x %u x %u\n   arguments: %s\n",
          last_k->name.c_str(), last_gs[0], last_gs[1], last_gs[2], last_ls[0], last_ls[1], last_ls[2],
          watch_args(last_k->args).c_str());
  for (size_t i = 0; i < last_k->args.size(); ++i) {
    const ZKernel::Arg& a = last_k->args[i];
    if (!a.set || a.sz != 8) continue;
    uint64_t v;
    memcpy(&v, a.v, 8);
    if (!v) continue;
    bool in = false;
    for (const ZMem* m : D().mems)
      if (v >= (uint64_t)m->ptr && v < (uint64_t)m->ptr + m->size) { in = true; break; }
    if (!in) fprintf(stderr, "   argument %zu (0x%llx) lies in no live buffer of this GPU\n", i, (unsigned long long)v);
  }
  fflush(stderr);
}
}  // namespace zrt
inline cl_int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel k, cl_uint dim, const size_t*, const size_t* g,
                                     const size_t* l, cl_uint, const cl_event*, cl_event* ev) {
  zrt::LaunchTimer lt{zrt::launch_ns};
  if (zrt::launch_stats) ++zrt::launch_n;
  if (ev) *ev = nullptr;
  uint32_t ls[3] = {1, 1, 1}, gs[3] = {1, 1, 1};
  for (cl_uint d = 0; d < dim; ++d) gs[d] = (uint32_t)g[d];
  if (l) {
    for (cl_uint d = 0; d < dim; ++d) ls[d] = (uint32_t)l[d];
  } else {
    ZE_CHECK(zeKernelSuggestGroupSize(k->k, gs[0], gs[1], gs[2], &ls[0], &ls[1], &ls[2]));
    // (Level Zero runs whole work-groups only: a group size that does not divide the global size would start
    //  work-items past it, and these kernels have no bounds checks; take the largest divisor not above the
    //  suggestion)
    for (int d = 0; d < 3; ++d)
      while (ls[d] > 1 && gs[d] % ls[d]) --ls[d];
  }
  if (ls[0] != k->gx || ls[1] != k->gy || ls[2] != k->gz) {
    ZE_CHECK(zeKernelSetGroupSize(k->k, ls[0], ls[1], ls[2]));
    k->gx = ls[0], k->gy = ls[1], k->gz = ls[2];
  }
  ze_group_count_t gc{(gs[0] + ls[0] - 1) / ls[0], (gs[1] + ls[1] - 1) / ls[1], (gs[2] + ls[2] - 1) / ls[2]};
  if (gs[0] % ls[0] || gs[1] % ls[1] || gs[2] % ls[2]) {
    // OpenCL would run a smaller last work-group; here the last group runs whole and its extra work-items would
    // read and write past the data. A launch like that is a bug in the caller (round the global size and give the
    // kernel a bounds check), so it fails loudly instead of corrupting memory.
    fprintf(stderr, "inferzero: %s: global size %u x %u x %u is not a multiple of the work-group %u x %u x %u\n",
            k->name.c_str(), gs[0], gs[1], gs[2], ls[0], ls[1], ls[2]);
    return -54;  // (CL_INVALID_WORK_GROUP_SIZE)
  }
  zrt::last_k = k;
  for (int d = 0; d < 3; ++d) zrt::last_gs[d] = gs[d], zrt::last_ls[d] = ls[d];
  ze_event_handle_t sig = ev || zrt::watch_ms > 0 ? zrt::ev_get() : nullptr;
  if (ev) *ev = (cl_event)sig;
  // (ARC_BATCH: recorded, see zrt::batch_max; a kernel with an event, ARC_ZE_SYNC_EACH or ARC_FAKE_OOM: as before)
  static const bool batch_ok = zrt::batching() && !(getenv("ARC_ZE_SYNC_EACH") && atoi(getenv("ARC_ZE_SYNC_EACH")) != 0) &&
                               !getenv("ARC_FAKE_OOM");
  if (batch_ok && !sig) {
    zrt::Dev& d = zrt::D();
    if (zeCommandListAppendLaunchKernel(zrt::rec_list(d), k->k, &gc, nullptr, 0, nullptr) == ZE_RESULT_SUCCESS) {
      if (++d.rec_n >= d.rec_cap) zrt::flush(d);
      return CL_SUCCESS;
    }
    // (a recording that failed: what the list holds goes now, and this kernel through the immediate list below)
  }
  ze_result_t r = zeCommandListAppendLaunchKernel(zrt::imm(), k->k, &gc, sig, 0, nullptr);
  static const long fake_oom = getenv("ARC_FAKE_OOM") ? atol(getenv("ARC_FAKE_OOM")) : 0;  // (test: launch N "fails")
  static long launches = 0;
  if (fake_oom && ++launches % fake_oom == 0 && r == ZE_RESULT_SUCCESS) {
    zrt::sync();  // (the kernel was appended: let it finish, then report the failure as if it had not been)
    fprintf(stderr, "(test: launch %ld of %s reported as out of device memory)\n", launches, k->name.c_str());
    if (zrt::oom_hook && zrt::oom_hook()) return CL_SUCCESS;
  }
  for (int i = 0; r == (ze_result_t)0x70000003 && i < 8 && zrt::oom_hook && zrt::oom_hook(); ++i)  // (out of device memory)
    r = zeCommandListAppendLaunchKernel(zrt::imm(), k->k, &gc, sig, 0, nullptr);
  if (r != ZE_RESULT_SUCCESS) {
    fprintf(stderr, "Level Zero error 0x%x launching %s (global %u x %u x %u, local %u x %u x %u)\n", r, k->name.c_str(),
            gs[0], gs[1], gs[2], ls[0], ls[1], ls[2]);
    if (ze_fail_hook) ze_fail_hook();
    fflush(stderr);
    _Exit(1);
  }
  if (zrt::watch_ms > 0) {  // (ARC_ZE_WATCH: this launch waited for alone, timed, kept in the ring; see zrt::watch_wait)
    zrt::watch_wait(k, sig, ev != nullptr, gs, ls);
    return CL_SUCCESS;
  }
  static const bool sync_each = (getenv("ARC_ZE_SYNC_EACH") && atoi(getenv("ARC_ZE_SYNC_EACH")) != 0);  // (debug: which kernel loses the device)
  if (sync_each) {
    const ze_result_t s = zeCommandListHostSynchronize(zrt::D().cl, UINT64_MAX);
    if (s != ZE_RESULT_SUCCESS) {
      fprintf(stderr, "Level Zero error 0x%x running %s (global %u x %u x %u, local %u x %u x %u)\n", s, k->name.c_str(),
              gs[0], gs[1], gs[2], ls[0], ls[1], ls[2]);
      if (ze_fail_hook) ze_fail_hook();
      fflush(stderr);
      _Exit(1);
    }
  }
  return CL_SUCCESS;
}
inline cl_int clGetDeviceInfo(cl_device_id, cl_device_info p, size_t sz, void* v, size_t* ret) {
  if (p == CL_DEVICE_GLOBAL_MEM_SIZE && sz >= 8) { *(cl_ulong*)v = zrt::D().global_mem; if (ret) *ret = 8; return CL_SUCCESS; }
  return -30;
}
inline cl_int clWaitForEvents(cl_uint, const cl_event*) { zrt::sync(); return CL_SUCCESS; }
inline cl_int clGetEventProfilingInfo(cl_event, cl_profiling_info, size_t sz, void* v, size_t*) {
  if (v && sz >= 8) *(cl_ulong*)v = 0;
  return CL_SUCCESS;
}
inline cl_int clReleaseEvent(cl_event e) {  // (only kernel events are real: back to the free list)
  if (e) { ZE_CHECK(zeEventHostReset((ze_event_handle_t)e)); zrt::D().ev_free.push_back((ze_event_handle_t)e); }
  return CL_SUCCESS;
}
inline cl_int clReleaseKernel(cl_kernel k) { if (k) { zeKernelDestroy(k->k); delete k; } return CL_SUCCESS; }
inline cl_int clReleaseProgram(cl_program p) { if (p) { zeModuleDestroy(p->m); delete p; } return CL_SUCCESS; }
inline cl_int clReleaseCommandQueue(cl_command_queue) { return CL_SUCCESS; }
inline cl_int clReleaseContext(cl_context) { return CL_SUCCESS; }

struct ClRt {
  std::string device_id, cache_dir;
  bool sg8 = true;  // (sub-groups of 8 lanes: Xe-LPG yes; Xe2 / Xe3 (Lunar Lake, Battlemage, Panther Lake) only 16 / 32)
  bool soft_fail = false;  // (a failed compile, module or kernel creation throws instead of ending the process: XMX check)
  bool xmx = false;  // (the driver offers DPAS: Xe2 / Xe3, Arrow Lake H; the engine still checks it, Engine::xmx_check)
  bool block2d = false;  // (cl_intel_subgroup_2d_block_io: 2D block loads, Xe2 / Xe3; zero/xe2/engine_xe2.h ARC_GEMM_2D)
  static uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
  }
  // (non-null placeholders: the engine passes them back to the calls above)
  cl_platform_id platform = (cl_platform_id)1;
  cl_device_id device = (cl_device_id)1;
  cl_context ctx = (cl_context)1;
  cl_command_queue queue = (cl_command_queue)1;

  bool init(bool = true) {
    if (!zrt::rcl::load()) { fprintf(stderr, "cannot load OpenCL.dll (needed to compile kernels)\n"); return false; }
    char name[256] = {}, drv[256] = {};
    zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x102B, sizeof(name) - 1, name, nullptr);  // CL_DEVICE_NAME
    zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x102D, sizeof(drv) - 1, drv, nullptr);   // CL_DRIVER_VERSION
    device_id = std::string(name) + "|" + drv;  // (the same key as the OpenCL engine's kernel cache)
    {  // CL_DEVICE_SUB_GROUP_SIZES_INTEL; ARC_NO_SG8=1 takes the 16-lane kernels anyway (tests)
      size_t sz[8] = {}, ret = 0;
      if (zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x4108, sizeof sz, sz, &ret) == 0 && ret >= sizeof(size_t)) {
        sg8 = false;
        for (size_t i = 0; i < ret / sizeof(size_t) && i < 8; ++i) sg8 = sg8 || sz[i] == 8;
      }
      if (getenv("ARC_NO_SG8") && atoi(getenv("ARC_NO_SG8")) != 0) sg8 = false;
      if (!sg8) fprintf(stderr, "inferzero: no 8-lane sub-groups on this GPU: 16-lane kernels\n");
    }
    {  // CL_DEVICE_EXTENSIONS: DPAS from OpenCL C (Engine::xmx_check decides whether it is used); ARC_NO_XMX=1: off
      std::string ext(16384, '\0');
      size_t ret = 0;
      if (zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x1030, ext.size() - 1, ext.data(), &ret) == 0) {
        xmx = ext.find("cl_intel_subgroup_matrix_multiply_accumulate") != std::string::npos;
        block2d = ext.find("cl_intel_subgroup_2d_block_io") != std::string::npos;
      }
      if (getenv("ARC_NO_XMX") && atoi(getenv("ARC_NO_XMX")) != 0) xmx = false;
    }
    if (!getenv("ARC_NO_KCACHE")) cache_dir = app_dir() + "\\kernels";
    // a second engine in the process (the server switching models, after the first one freed its memory) keeps the
    // runtime: the context, its command lists and events, the staging buffer (all of zrt:: belongs to that context)
    if (zrt::ctx) return true;
    if (!ze_load()) { fprintf(stderr, "cannot load ze_loader.dll (Level Zero)\n"); return false; }
    ZE_CHECK(zeInit(1));
    // the first driver whose device is a GPU (zeInit(1) lists GPU drivers only; after a zeInit(0) of the NPU
    // experiment, ARC_NPU=1, the NPU's driver is listed too and could come first)
    uint32_t nd = 0;
    ZE_CHECK(zeDriverGet(&nd, nullptr));
    std::vector<ze_driver_handle_t> drvs(nd);
    ZE_CHECK(zeDriverGet(&nd, drvs.data()));
    ze_driver_handle_t drvh = nullptr;
    for (ze_driver_handle_t d : drvs) {
      uint32_t n = 1;
      ze_device_handle_t x = nullptr;
      if (zeDeviceGet(d, &n, &x) != 0 || !n) continue;
      ze_device_properties_t p{};
      p.stype = ZE_STYPE_DEVICE_PROPERTIES;
      if (zeDeviceGetProperties(x, &p) == 0 && p.type == 1) { drvh = d; zrt::dev0.dev = x; break; }
    }
    if (!drvh) { fprintf(stderr, "no Level Zero GPU driver\n"); return false; }
    uint32_t nm = 1;
    ze_device_memory_properties_t mp{};
    mp.stype = ZE_STYPE_DEVICE_MEMORY_PROPERTIES;
    ZE_CHECK(zeDeviceGetMemoryProperties(zrt::dev0.dev, &nm, &mp));
    zrt::dev0.global_mem = mp.totalSize;
    ze_context_desc_t cd{ZE_STYPE_CONTEXT_DESC, nullptr, 0};
    ZE_CHECK(zeContextCreate(drvh, &cd, &zrt::ctx));  // (every GPU of the driver: host memory they all reach)
    zrt::driver = drvh;
    ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 2 /* in order */, 2 /* asynchronous */, 0};
    ZE_CHECK(zeCommandListCreateImmediate(zrt::ctx, zrt::dev0.dev, &qd, &zrt::dev0.cl));
    ze_fail_hook = zrt::print_memory;
    return true;
  }
  // GPUs 1..n-1 of the driver next to device 0 (after init), each with its own command list: zrt::devs. sim: n
  // "GPUs" that are all device 0, each with its own list (multi-GPU tests on one GPU, ARC_TP_SIM). Returns the
  // number open, or 0 and why in err (fewer GPUs than asked, others of another kind).
  static int open_devices(int n, bool sim, std::string& err) {
    if (!zrt::ctx || !zrt::driver) { err = "the runtime is not started"; return 0; }
    if ((int)zrt::devs.size() >= n) return n;
    std::vector<ze_device_handle_t> gpus;
    if (sim) {
      gpus.assign(n, zrt::dev0.dev);
    } else {
      uint32_t c = 0;
      ZE_CHECK(zeDeviceGet(zrt::driver, &c, nullptr));
      std::vector<ze_device_handle_t> all(c);
      ZE_CHECK(zeDeviceGet(zrt::driver, &c, all.data()));
      gpus.push_back(zrt::dev0.dev);
      ze_device_properties_t p0{};
      p0.stype = ZE_STYPE_DEVICE_PROPERTIES;
      ZE_CHECK(zeDeviceGetProperties(zrt::dev0.dev, &p0));
      int other = 0;  // (GPUs of another kind: the kernels are compiled for GPU 0's, a module of them would not load)
      for (ze_device_handle_t x : all) {
        ze_device_properties_t p{};
        p.stype = ZE_STYPE_DEVICE_PROPERTIES;
        if (x == zrt::dev0.dev || zeDeviceGetProperties(x, &p) != 0 || p.type != 1) continue;
        if (p.deviceId == p0.deviceId) gpus.push_back(x);
        else ++other;
      }
      if ((int)gpus.size() < n) {
        err = std::to_string(n) + " GPUs asked for, the driver has " + std::to_string(gpus.size()) + " of GPU 0's kind" +
              (other ? " (and " + std::to_string(other) + " of another kind, which cannot share a model with it)" : "");
        return 0;
      }
    }
    for (int i = (int)zrt::devs.size(); i < n; ++i) {
      zrt::Dev* d = new zrt::Dev{};  // (kept until the process ends, as device 0's state: a later engine reuses it)
      d->dev = gpus[i];
      d->index = i;
      uint32_t nm = 1;
      ze_device_memory_properties_t mp{};
      mp.stype = ZE_STYPE_DEVICE_MEMORY_PROPERTIES;
      ZE_CHECK(zeDeviceGetMemoryProperties(d->dev, &nm, &mp));
      d->global_mem = mp.totalSize;
      ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 2 /* in order */, 2 /* asynchronous */, 0};
      ZE_CHECK(zeCommandListCreateImmediate(zrt::ctx, d->dev, &qd, &d->cl));
      zrt::devs.push_back(d);
    }
    return n;
  }

  // The native binary for (source, options): from the shared kernel cache, else compiled by OpenCL and saved there.
  cl_program build(const std::string& src, const std::string& opts) {
    std::string path;
    std::vector<uint8_t> bin;
    if (!cache_dir.empty()) {
      char hex[17];
      snprintf(hex, sizeof(hex), "%016llx", (unsigned long long)fnv1a(src + "\x01" + opts + "\x01" + device_id));
      path = cache_dir + "\\" + hex + ".bin";
      // (ARC_KLIST=1: which cached binary holds which options, for reading its assembly: ocloc disasm)
      static const bool klist = getenv("ARC_KLIST") && atoi(getenv("ARC_KLIST")) != 0;
      if (klist) fprintf(stderr, "kernel binary %s.bin: %s\n", hex, opts.c_str());
      std::ifstream f(path, std::ios::binary);
      if (f) bin.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    bool cached = !bin.empty();
  compile:
    if (bin.empty()) {
      cl_int err = 0;
      const char* s = src.c_str();
      const size_t sn = src.size();
      void* prog = zrt::rcl::createProgramWithSource(zrt::rcl::context, 1, &s, &sn, &err);
      err = zrt::rcl::buildProgram(prog, 1, &zrt::rcl::device, opts.c_str(), nullptr, nullptr);
      if (err != 0) {
        size_t ln = 0;
        zrt::rcl::getProgramBuildInfo(prog, zrt::rcl::device, 0x1183, 0, nullptr, &ln);
        std::string log(ln, 0);
        zrt::rcl::getProgramBuildInfo(prog, zrt::rcl::device, 0x1183, ln, log.data(), nullptr);
        zrt::rcl::releaseProgram(prog);
        if (soft_fail) throw std::runtime_error("kernel build failed: " + log.substr(0, 200));
        fprintf(stderr, "build log:\n%s\n", log.c_str());
        CL_CHECK(err);
      }
      size_t bn = 0;
      zrt::rcl::getProgramInfo(prog, 0x1165, sizeof(bn), &bn, nullptr);
      try {
        bin.resize(bn);
      } catch (...) {
        zrt::rcl::releaseProgram(prog);
        throw;
      }
      uint8_t* bp = bin.data();
      zrt::rcl::getProgramInfo(prog, 0x1166, sizeof(bp), &bp, nullptr);
      zrt::rcl::releaseProgram(prog);
      if (!path.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cache_dir, ec);
        const std::string tmp = path + ".tmp";
        bool ok = false;
        {
          std::ofstream o(tmp, std::ios::binary);
          o.write((const char*)bin.data(), (std::streamsize)bin.size());
          o.close();
          ok = !o.fail();
        }
        if (ok) std::filesystem::rename(tmp, path, ec);  // (a short write (disk full) is never installed)
        else std::filesystem::remove(tmp, ec);
      }
    }
    // The OpenCL build option -cl-intel-256-GRF-per-thread does not survive into a Level Zero module made from the
    // native binary: the kernels ran with the default register file and spilled (attn_decode at HD 256: 4 KB per
    // thread, 1.9 -> 3.6 ms per layer at 64K context). The module asks for the large register file itself.
    const bool big_grf = opts.find("-cl-intel-256-GRF-per-thread") != std::string::npos;
    ze_module_desc_t md{ZE_STYPE_MODULE_DESC, nullptr, 1 /* native */, bin.size(), bin.data(),
                        big_grf ? "-ze-opt-large-register-file" : "", nullptr};
    ZProg* p = new ZProg{};
    p->indirect = opts.find("-DXCACHE") != std::string::npos;
    p->opts = opts;
    ze_module_build_log_handle_t log = nullptr;
    const ze_result_t r = zeModuleCreate(zrt::ctx, zrt::D().dev, &md, &p->m, &log);
    if (r != ZE_RESULT_SUCCESS && cached) {  // a damaged cache file: compiled again from the source
      if (log) zeModuleBuildLogDestroy(log);
      delete p;
      std::error_code ec;
      std::filesystem::remove(path, ec);
      bin.clear();
      cached = false;
      goto compile;
    }
    if (r != ZE_RESULT_SUCCESS) {
      size_t ln = 0;
      zeModuleBuildLogGetString(log, &ln, nullptr);
      std::string s(ln, 0);
      zeModuleBuildLogGetString(log, &ln, s.data());
      if (log) zeModuleBuildLogDestroy(log);  // (before the throw: it was left behind on a soft failure)
      delete p;
      if (soft_fail) throw std::runtime_error("zeModuleCreate failed: " + s.substr(0, 200));
      fprintf(stderr, "zeModuleCreate failed 0x%x: %s\n", r, s.c_str());
      fflush(stderr);
      _Exit(1);
    }
    if (log) zeModuleBuildLogDestroy(log);
    return p;
  }
  cl_kernel kernel(cl_program prog, const char* name) {
    ZKernel* k = new ZKernel{};
    k->name = name;
    ze_kernel_desc_t kd{ZE_STYPE_KERNEL_DESC, nullptr, 0, name};
    const ze_result_t r = zeKernelCreate(prog->m, &kd, &k->k);
    if (r != ZE_RESULT_SUCCESS) {
      delete k;
      if (soft_fail) throw std::runtime_error(std::string("zeKernelCreate failed: ") + name);
      fprintf(stderr, "zeKernelCreate(%s) failed 0x%x\n", name, r);
      exit(1);
    }
    // The expert kernels (-DXCACHE) reach memory that is not one of their arguments (each expert's address comes from
    // a table): with indirect access the driver keeps every allocation resident for them; without it an expert in a
    // piece of the cache that Windows had paged out was a page fault ("device lost" now and then, in the server).
    // Only for them: on every kernel it made the driver touch every allocation per launch (27B: slower).
    if (prog->indirect) zeKernelSetIndirectAccess(k->k, 7 /* device | host | shared */);
    static const bool kprops = (getenv("ARC_ZE_KPROPS") && atoi(getenv("ARC_ZE_KPROPS")) != 0);  // (debug: private / spill memory per kernel)
    if (kprops) {
      ze_kernel_properties_t kp{};
      kp.stype = (ze_structure_type_t)0x1e;
      if (zeKernelGetProperties(k->k, &kp) == ZE_RESULT_SUCCESS && (kp.privateMemSize || kp.spillMemSize))
        fprintf(stderr, "kernel %s: private %u bytes, spill %u bytes per work-item/thread [%s]\n", name, kp.privateMemSize,
                kp.spillMemSize, prog->opts.c_str());
    }
    return k;
  }
  size_t spill_bytes(cl_kernel) { return 0; }
  cl_mem buffer(cl_mem_flags flags, size_t bytes, const void* host = nullptr) {
    cl_int err;
    cl_mem m = clCreateBuffer(ctx, flags | (host ? CL_MEM_COPY_HOST_PTR : 0), bytes, (void*)host, &err);
    CL_CHECK(err);
    return m;
  }
  static double event_ms(cl_event e) {  // (kernel time from the Level Zero timestamps; call after clFinish)
    if (!e) return 0.0;
    ze_kernel_timestamp_result_t ts{};
    if (zeEventQueryKernelTimestamp((ze_event_handle_t)e, &ts) != ZE_RESULT_SUCCESS) return 0.0;
    const uint64_t t = (ts.context.kernelEnd - ts.context.kernelStart) & zrt::D().ev_mask;
    return t * zrt::D().ev_ns_per_tick * 1e-6;
  }
};

inline std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// A kernel program kept in parts, files in a fixed order, that were one file once (ops.cl 3,516 lines, decode.cl
// 2,047). A kernel of part k builds from parts 0..k: the text before it is what it was in the one file, so its code is
// too, and a norm's or a GEMV's build no longer compiles the attention, encoder, Flash Next or MoE kernels after it.
struct KernelParts {
  static constexpr std::initializer_list<const char*> kOps = {"ops.cl", "ops_attn.cl", "ops_attn_g.cl", "ops_vit.cl",
                                                              "ops_flashnext.cl", "ops_dflash.cl", "ops_gdn.cl"};
  static constexpr std::initializer_list<const char*> kDecode = {"decode.cl", "decode_moe.cl"};
  std::vector<std::string> prefix;              // prefix[k]: parts 0..k
  std::unordered_map<std::string, int> part_of;  // kernel name -> its part
  void load(const std::string& dir, std::initializer_list<const char*> parts) {
    std::string s;
    prefix.clear();
    part_of.clear();
    for (const char* f : parts) {
      const std::string t = read_file(dir + "/" + f);
      for (size_t i = t.find("__kernel void "); i != std::string::npos; i = t.find("__kernel void ", i + 1)) {
        size_t a = i + 14, b = a;
        while (b < t.size() && (isalnum((unsigned char)t[b]) || t[b] == '_')) ++b;
        part_of.emplace(t.substr(a, b - a), (int)prefix.size());
      }
      prefix.push_back(s += t);
    }
  }
  const std::string& all() const { return prefix.back(); }
  const std::string& for_kernel(const char* name) const {
    const auto it = part_of.find(name);
    return it == part_of.end() ? all() : prefix[it->second];  // (an unknown name: the whole set, its build error)
  }
};

template <typename T> inline void set_args(cl_kernel k, cl_uint i, const T& v) { CL_CHECK(clSetKernelArg(k, i, sizeof(T), &v)); }
template <typename T, typename... R> inline void set_args(cl_kernel k, cl_uint i, const T& v, const R&... rest) {
  set_args(k, i, v);
  set_args(k, i + 1, rest...);
}

// What recording a generation step and replaying it could save (the idea behind vLLM's CUDA graphs): the same
// sequence of small dependent kernels, as the engine launches them now (each appended to the in-order immediate
// command list) against a regular command list recorded once and executed again (in order: a barrier between
// kernels, or the in-order list flag). The difference is the GPU's idle time between kernels that a replay removes.
//   bin\replay_probe.exe [--kernels 700] [--floats 65536] [--reps 30]
// 700 kernels ~ one token of the 35B-A3B, 930 of the 27B (ARC_LAUNCH_STATS=1 with runz counts them).
#include "../zero/clrt.h"
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

static const char* kSrc = R"CLC(
__kernel void step(__global float* y, __global const float* x, int n) {
  const int i = get_global_id(0);
  if (i < n) y[i] = y[i] * 0.999f + x[i];
}
)CLC";

struct ze_command_list_desc_t {
  ze_structure_type_t stype;
  const void* pNext;
  uint32_t commandQueueGroupOrdinal, flags;  // flags 8: in order (ZE_COMMAND_LIST_FLAG_IN_ORDER)
};
typedef struct _ze_command_queue_handle_t* ze_command_queue_handle_t;
typedef struct _ze_fence_handle_t* ze_fence_handle_t;
typedef ze_result_t(__stdcall* PFN_listCreate)(ze_context_handle_t, ze_device_handle_t, const ze_command_list_desc_t*,
                                              ze_command_list_handle_t*);
typedef ze_result_t(__stdcall* PFN_listClose)(ze_command_list_handle_t);
typedef ze_result_t(__stdcall* PFN_queueCreate)(ze_context_handle_t, ze_device_handle_t, const ze_command_queue_desc_t*,
                                               ze_command_queue_handle_t*);
typedef ze_result_t(__stdcall* PFN_queueExec)(ze_command_queue_handle_t, uint32_t, ze_command_list_handle_t*, ze_fence_handle_t);
typedef ze_result_t(__stdcall* PFN_queueSync)(ze_command_queue_handle_t, uint64_t);

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  int nk = 700, nf = 65536, reps = 30;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--kernels") nk = atoi(argv[i + 1]);
    else if (a == "--floats") nf = atoi(argv[i + 1]);
    else if (a == "--reps") reps = atoi(argv[i + 1]);
  }
  ClRt cl;
  if (!cl.init()) return 1;
  HMODULE zl = GetModuleHandleA("ze_loader.dll");
  auto listCreate = (PFN_listCreate)GetProcAddress(zl, "zeCommandListCreate");
  auto listClose = (PFN_listClose)GetProcAddress(zl, "zeCommandListClose");
  auto queueCreate = (PFN_queueCreate)GetProcAddress(zl, "zeCommandQueueCreate");
  auto queueExec = (PFN_queueExec)GetProcAddress(zl, "zeCommandQueueExecuteCommandLists");
  auto queueSync = (PFN_queueSync)GetProcAddress(zl, "zeCommandQueueSynchronize");
  if (!listCreate || !listClose || !queueCreate || !queueExec || !queueSync) { fprintf(stderr, "Level Zero: no regular command lists\n"); return 1; }
  uint32_t ng = 0, ord = 0;
  zeDeviceGetCommandQueueGroupProperties(zrt::D().dev, &ng, nullptr);
  std::vector<ze_command_queue_group_properties_t> gp(ng, ze_command_queue_group_properties_t{ZE_STYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr});
  zeDeviceGetCommandQueueGroupProperties(zrt::D().dev, &ng, gp.data());
  for (uint32_t g = 0; g < ng; ++g)
    if (gp[g].flags & 1) { ord = g; break; }

  cl_program prog = cl.build(kSrc, "-cl-std=CL3.0");
  cl_kernel k = cl.kernel(prog, "step");
  cl_mem y = cl.buffer(CL_MEM_READ_WRITE, (size_t)nf * 4), x = cl.buffer(CL_MEM_READ_WRITE, (size_t)nf * 4);
  const float zero = 0.0f;
  CL_CHECK(clEnqueueFillBuffer(cl.queue, y, &zero, 4, 0, (size_t)nf * 4, 0, nullptr, nullptr));
  CL_CHECK(clEnqueueFillBuffer(cl.queue, x, &zero, 4, 0, (size_t)nf * 4, 0, nullptr, nullptr));
  set_args(k, 0, y, x, nf);
  const size_t gws = ((size_t)nf + 255) / 256 * 256, lws = 256;
  clFinish(cl.queue);
  ze_group_count_t gc{(uint32_t)(gws / lws), 1, 1};
  ZE_CHECK(zeKernelSetGroupSize(k->k, (uint32_t)lws, 1, 1));
  auto ms_since = [](std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
  };

  // 1. now: every kernel appended to the immediate in-order list, then a host wait
  std::vector<double> imm;
  for (int r = 0; r < reps; ++r) {
    const auto t = std::chrono::steady_clock::now();
    for (int i = 0; i < nk; ++i) ZE_CHECK(zeCommandListAppendLaunchKernel(zrt::D().cl, k->k, &gc, nullptr, 0, nullptr));
    zrt::sync();
    imm.push_back(ms_since(t));
  }
  // 2. regular lists, recorded once and replayed: with a barrier after each kernel, and with the in-order flag
  ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, ord, 0, 0, 2 /* async */, 0};
  ze_command_queue_handle_t q = nullptr;
  ZE_CHECK(queueCreate(zrt::ctx, zrt::D().dev, &qd, &q));
  auto replay = [&](uint32_t flags, bool barriers, const char* what) {
    ze_command_list_desc_t ld{0xfu /* ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC */, nullptr, ord, flags};
    ze_command_list_handle_t rl = nullptr;
    if (listCreate(zrt::ctx, zrt::D().dev, &ld, &rl) != 0) { printf("%-40s not supported\n", what); return; }
    const auto tr = std::chrono::steady_clock::now();
    for (int i = 0; i < nk; ++i) {
      ZE_CHECK(zeCommandListAppendLaunchKernel(rl, k->k, &gc, nullptr, 0, nullptr));
      if (barriers) ZE_CHECK(zeCommandListAppendBarrier(rl, nullptr, 0, nullptr));
    }
    ZE_CHECK(listClose(rl));
    const double rec = ms_since(tr);
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      ZE_CHECK(queueExec(q, 1, &rl, nullptr));
      ZE_CHECK(queueSync(q, UINT64_MAX));
      t.push_back(ms_since(t0));
    }
    printf("%-40s %8.3f ms per step (%.2f us per kernel; recorded once in %.2f ms)\n", what, median(t), median(t) * 1000 / nk, rec);
    zeCommandListDestroy(rl);
  };
  printf("%d dependent kernels of %d floats each, median of %d steps\n", nk, nf, reps);
  printf("%-40s %8.3f ms per step (%.2f us per kernel)\n", "immediate list (the engine now)", median(imm), median(imm) * 1000 / nk);
  replay(0, true, "replayed list, barrier after each kernel");
  replay(8, false, "replayed list, in-order flag");
  // (laptop, Core Ultra 5 125H iGPU: 700 kernels of 4,096 floats 1.36 ms immediate, 1.07 replayed; of 65,536 floats
  //  1.95 against 2.01: a replay saves ~0.4 us a kernel at most there)
  return 0;
}

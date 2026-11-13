// The 8-lane XMX prompt GEMM (zero/kernels/gemm_xw8.cl: the Arc 140T's H8R GEMM) alone, at a model's shapes, in TOPS
// and as a share of the matrix engines' peak measured in the same run; each kernel variant's results compared bit for
// bit with the engine's default. One run takes about a minute instead of a model sweep (bench\xw_sweep.ps1).
// Also: what the f16 DPAS keeps inside (products, the 16-term sum, the accumulator), for the chunked DeltaNet on XMX,
// which missed its start-up check on the 140T by f16's precision (QUEUE row 4).
//
// usage (from the package folder): bin\xw_peak.exe [--m 4096] [--iters 10] [--shape NxK[:a]]... [--only NAME]
//   default shapes: the Qwen3.8 27B's FFN, gate 17408x5120, up 17408x5120 with the fused SwiGLU (:2), down 5120x17408
//   (split-K in 4 pieces, as the engine runs it). --emu: DPAS emulated (any GPU; correctness only, no speed meaning).
//   Without XMX (the 125H) the DPAS is emulated by itself.
// Variants (gemm_xw8.cl's switches): the default 32 x 16 tiles; XW_PACKA (activations in the DPAS order: 2 reads per 8
// tokens instead of 8); XW_PACKW (weights in the DPAS order: no register shuffles); both; both at 32 x 32 and 16 x 32;
// since 2026-11-13 the wider ones too, 16 x 64, 64 x 16, 32 x 64 and 64 x 32 (the spill column says which fit).
// The packed copies are made here on the CPU from the same data, so every variant must give the default's bits.
#include "../zero/clrt.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const char* kProbe = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#if defined(K_I8_8)
// the matrix engines' int8 peak, as gpu_bench: 8 independent DPAS chains per sub-group
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void probe(__global const int* x, int iters, __global int* out) {
  const int l = get_sub_group_local_id();
  const int8 a = vload8(0, x + 8 * l), b = vload8(0, x + 64 + 8 * l);
  int8 c[8];
  for (int i = 0; i < 8; ++i) c[i] = (int8)(i);
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = intel_sub_group_i8_i8_matrix_mad_k32(a, b, c[i]);
  }
  int s = 0;
  for (int i = 0; i < 8; ++i) s += c[i].s0 ^ c[i].s1 ^ c[i].s2 ^ c[i].s3 ^ c[i].s4 ^ c[i].s5 ^ c[i].s6 ^ c[i].s7;
  if (s == 0x7f3e9a15) out[get_global_id(0)] = s;
}
#elif defined(P_F16_8) || defined(P_F16_16)
// what f16 DPAS keeps: every row and column the same values, so the operand layouts do not matter beyond "value 0 of
// each row" (8 lanes: lane 0's low half; 16 lanes: lane 0)
#ifdef P_F16_8
#define SG 8
typedef int8 AT;
#define AV(v0, v1) ((int8)(as_int((half2)((v0), (v1)))))
#else
#define SG 16
typedef short8 AT;
#define AV(v0, v1) ((short8)(as_short((half)(v0))))
#endif
#define MAD(a, b, c) intel_sub_group_f16_f16_matrix_mad_k16((a), (b), (c))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void prec(__global float* out, int reps) {
  const int l = get_sub_group_local_id();
  const half one = (half)1.0f, e12 = (half)(1.0f / 4096), z = (half)0.0f, p = (half)(1.0f + 1.0f / 1024);
  const int8 B1 = (int8)(as_int((half2)(one, one))), BP = (int8)(as_int((half2)(p, p)));
#ifdef P_F16_8
  const AT A1 = AV(l == 0 ? one : e12, e12), AP = AV(p, p), A3 = AV(l == 0 ? e12 : z, z);
#else
  const AT A1 = AV(l == 0 ? one : e12, 0), AP = AV(p, 0), A3 = AV(l == 0 ? e12 : z, 0);
#endif
  // 1: the sum inside one DPAS (1 + 15 x 2^-12); 2: the products ((1 + 2^-10)^2 x 16); 3: the accumulator (1 + 2^-12);
  // 4: 3 chained reps times; 5: as 4, each DPAS from 0 and added in f32 (the fix if 4 loses and 5 does not)
  const float8 r1 = MAD(A1, B1, (float8)(0.0f)), r2 = MAD(AP, BP, (float8)(0.0f)), r3 = MAD(A3, B1, (float8)(1.0f));
  float8 r4 = (float8)(1.0f), r5 = (float8)(1.0f);
  for (int i = 0; i < reps; ++i) r4 = MAD(A3, B1, r4);
  for (int i = 0; i < reps; ++i) r5 += MAD(A3, B1, (float8)(0.0f));
  if (l == 0) { out[0] = r1.s0; out[1] = r2.s0; out[2] = r3.s0; out[3] = r4.s0; out[4] = r5.s0; }
}
#endif
)CLC";

using clk = std::chrono::steady_clock;

struct Shape { int N, K, a; const char* name; };
struct Variant { const char* name; const char* defs; int tm, nc; bool pa, pw; };

static uint32_t g_seed = 12345;
static uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed; }

int main(int argc, char** argv) {
  int M = 4096, iters = 10;
  bool force_emu = false;
  std::string only, kdir = "zero/kernels";
  std::vector<Shape> shapes;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--m" && i + 1 < argc) M = atoi(argv[++i]);
    else if (a == "--iters" && i + 1 < argc) iters = std::max(1, atoi(argv[++i]));
    else if (a == "--only" && i + 1 < argc) only = argv[++i];
    else if (a == "--kernels" && i + 1 < argc) kdir = argv[++i];
    else if (a == "--emu") force_emu = true;
    else if (a == "--shape" && i + 1 < argc) {
      int n = 0, k = 0, ac = 0;
      const std::string s = argv[++i];
      if (sscanf(s.c_str(), "%dx%d:%d", &n, &k, &ac) < 2) { fprintf(stderr, "--shape NxK[:a]\n"); return 2; }
      shapes.push_back({n, k, ac, "custom"});
    } else {
      fprintf(stderr, "usage: xw_peak [--m 4096] [--iters 10] [--shape NxK[:a]]... [--only NAME] [--emu] [--kernels DIR]\n");
      return 2;
    }
  }
  if (shapes.empty()) shapes = {{17408, 5120, 0, "gate"}, {17408, 5120, 2, "up+SwiGLU"}, {5120, 17408, 0, "down"}};
  if (M <= 0 || M % 32) { fprintf(stderr, "--m must be a positive multiple of 32\n"); return 2; }
  for (const Shape& s : shapes)
    if (s.N % 32 || s.K % 256 || s.N <= 0 || s.K <= 0 || (s.a != 0 && s.a != 2)) {
      fprintf(stderr, "shape %dx%d: N must be a multiple of 32, K of 256, a 0 or 2\n", s.N, s.K);
      return 2;
    }

  ClRt cl;
  if (!cl.init()) return 1;
  char name[256] = {}, drv[128] = {};
  cl_uint eus = 0, mhz = 0;
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x102B, sizeof(name) - 1, name, nullptr);
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x102D, sizeof(drv) - 1, drv, nullptr);
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x1002, sizeof(eus), &eus, nullptr);
  zrt::rcl::getDeviceInfo(zrt::rcl::device, 0x100C, sizeof(mhz), &mhz, nullptr);
  printf("xw_peak: %s, %u EUs, %u MHz, driver %s\n", name, eus, mhz, drv);
  if (!cl.sg8) {
    printf("This GPU has no 8-lane sub-groups: gemm_xw8 is the 8-lane kernel (Arc 140T, Xe-LPG); nothing to measure.\n");
    return 0;
  }
  const bool emu = force_emu || !cl.xmx;
  cl.soft_fail = true;
  std::vector<int> host(256);
  for (int i = 0; i < 256; ++i) host[i] = (int)((0x01020304u * (uint32_t)(i + 1)) ^ 0x5a5a5a5au) & 0x3f3f3f3f;
  cl_mem bx = cl.buffer(CL_MEM_READ_WRITE, host.size() * 4, host.data());

  // ---- the peak: int8 DPAS chains (gpu_bench's probe), best of ~1 s
  double peak = 0;
  if (!emu) {
    try {
      const int groups = std::max(256u, eus * 16), piters = 2048;
      cl_mem bo = cl.buffer(CL_MEM_READ_WRITE, (size_t)groups * 64 * 4);
      cl_kernel k = cl.kernel(cl.build(kProbe, "-cl-std=CL3.0 -DK_I8_8"), "probe");
      int it = piters;
      set_args(k, 0, bx, it, bo);
      ZE_CHECK(zeKernelSetGroupSize(k->k, 32, 1, 1));
      ze_group_count_t gc{(uint32_t)groups, 1, 1};
      double best = 1e30, spent = 0;
      for (int r = 0; r < 200 && (r < 3 || spent < 1.0); ++r) {
        const auto t0 = clk::now();
        ZE_CHECK(zeCommandListAppendLaunchKernel(zrt::D().cl, k->k, &gc, nullptr, 0, nullptr));
        zrt::sync();
        const double t = std::chrono::duration<double>(clk::now() - t0).count();
        if (r) { best = std::min(best, t); spent += t; }
      }
      peak = (double)groups * 4 * (2.0 * 8 * 8 * 32) * 8 * piters / best / 1e12;
      printf("int8 XMX peak (measured): %.2f TOPS\n", peak);
    } catch (const std::exception& e) {
      printf("int8 XMX peak: not measured (%s)\n", e.what());
    }
    // ---- f16 DPAS precision
    try {
      cl_mem bp = cl.buffer(CL_MEM_READ_WRITE, 64);
      cl_kernel k = cl.kernel(cl.build(kProbe, "-cl-std=CL3.0 -DP_F16_8"), "prec");
      int reps = 64;
      set_args(k, 0, bp, reps);
      size_t g[1] = {8}, l[1] = {8};
      CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 1, nullptr, g, l, 0, nullptr, nullptr));
      float r[5] = {};
      CL_CHECK(clEnqueueReadBuffer(cl.queue, bp, CL_TRUE, 0, sizeof r, r, 0, nullptr, nullptr));
      const double ex[5] = {1.0 + 15.0 / 4096, 16.0 * (1.0 + 1.0 / 512 + 1.0 / 1048576), 1.0 + 1.0 / 4096,
                            1.0 + 64.0 / 4096, 1.0 + 64.0 / 4096};
      const char* what[5] = {"sum inside one DPAS", "products", "accumulator", "64 chained DPAS",
                             "64 DPAS from 0, f32 adds"};
      printf("f16 DPAS precision (exact in f32; \"rounded\" = fewer bits kept):\n");
      for (int i = 0; i < 5; ++i)
        printf("  %-26s %.9f  exact %.9f  %s\n", what[i], r[i], ex[i], (double)r[i] == ex[i] ? "exact" : "ROUNDED");
    } catch (const std::exception& e) {
      printf("f16 DPAS precision: not measured (%s)\n", e.what());
    }
  } else {
    printf("DPAS emulated (%s): the results are checked, the times mean nothing\n", cl.xmx ? "--emu" : "no XMX here");
  }

  const std::string src = read_file(kdir + "/gemm_xw8.cl");
  const std::vector<Variant> all = {
      {"default 32x16", "", 32, 1, false, false},  {"packA 32x16", "-DXW_PACKA", 32, 1, true, false},
      {"packW 32x16", "-DXW_PACKW", 32, 1, false, true}, {"packAW 32x16", "-DXW_PACKA -DXW_PACKW", 32, 1, true, true},
      {"packAW 32x32", "-DXW_PACKA -DXW_PACKW", 32, 2, true, true},
      {"packAW 16x32", "-DXW_PACKA -DXW_PACKW", 16, 2, true, true}, {"plain 32x32", "", 32, 2, false, false},
      // (2026-11-13, beyond 32 x 32; ocloc arl-h at 256 registers: 16 x 64 plain and 64 x 16 packed no spill, the
      //  others spill)
      {"plain 16x64", "", 16, 4, false, false},  {"packAW 16x64", "-DXW_PACKA -DXW_PACKW", 16, 4, true, true},
      {"plain 64x16", "", 64, 1, false, false},  {"packAW 64x16", "-DXW_PACKA -DXW_PACKW", 64, 1, true, true},
      {"plain 32x64", "", 32, 4, false, false},  {"packAW 64x32", "-DXW_PACKA -DXW_PACKW", 64, 2, true, true},
  };
  std::vector<Variant> vs;
  for (const Variant& v : all)
    if (vs.empty() || only.empty() || only == v.name) vs.push_back(v);  // (the default always: the reference)

  struct Row { double ms = 0, best = 0; bool ok = true, ran = false; size_t diff = 0; size_t spill = 0; };
  std::vector<std::vector<Row>> res(vs.size(), std::vector<Row>(shapes.size()));
  std::vector<cl_kernel> ks(vs.size(), nullptr);
  for (size_t v = 0; v < vs.size(); ++v) {
    const std::string opts = std::string("-cl-std=CL3.0 -cl-mad-enable -cl-intel-256-GRF-per-thread -DXW_TM=") +
                             std::to_string(vs[v].tm) + " -DXW_NC=" + std::to_string(vs[v].nc) + " " + vs[v].defs +
                             (emu ? " -DXW_EMU" : "");
    try {
      ks[v] = cl.kernel(cl.build(src, opts), "gemm_xw8");
      ze_kernel_properties_t kp{};
      kp.stype = (ze_structure_type_t)0x1e;
      if (zeKernelGetProperties(ks[v]->k, &kp) == ZE_RESULT_SUCCESS)
        for (Row& r : res[v]) r.spill = kp.spillMemSize;
    } catch (const std::exception& e) {
      printf("%s: does not build (%s)\n", vs[v].name, e.what());
    }
  }

  for (size_t si = 0; si < shapes.size(); ++si) {
    const Shape& sh = shapes[si];
    const int N = sh.N, K = sh.K, nsb = K / 256, nch = K / 64, ng = N / 16;
    // H8R data: int8 weights in gemm.cl's layout [group][chunk][16 words][16 columns], one scale per column; int8
    // activations [M][K], one scale per token
    std::vector<uint32_t> wq((size_t)ng * nch * 256), wh((size_t)N);
    for (auto& x : wq) x = rnd();
    for (auto& x : wh) { const float f = 0.001f + (rnd() % 10000) * 1e-6f; memcpy(&x, &f, 4); }
    std::vector<uint32_t> aq((size_t)M * K / 4);
    for (auto& x : aq) x = rnd();
    std::vector<float> asc((size_t)M), c0((size_t)M * N);
    for (auto& x : asc) x = 0.002f + (rnd() % 10000) * 2e-6f;
    for (auto& x : c0) x = -2.0f + (rnd() % 10000) * 4e-4f;
    // the packed copies (gemm_xw8.cl's header)
    std::vector<uint32_t> ap(aq.size()), wp(wq.size());
    for (int m = 0; m < M; ++m)
      for (int c = 0; c < nch; ++c)
        for (int h = 0; h < 2; ++h)
          for (int l = 0; l < 8; ++l)
            ap[((size_t)(m / 8) * nch + c) * 128 + h * 64 + (m % 8) * 8 + l] = aq[((size_t)m * K + c * 64 + h * 32 + 4 * l) / 4];
    for (size_t gc = 0; gc < (size_t)ng * nch; ++gc)
      for (int b = 0; b < 4; ++b)
        for (int i = 0; i < 8; ++i)
          for (int l = 0; l < 8; ++l)
            wp[gc * 256 + b * 64 + i * 8 + l] = wq[gc * 256 + ((b >> 1) * 8 + i) * 16 + (b & 1) * 8 + l];
    cl_mem bwq = cl.buffer(CL_MEM_READ_ONLY, wq.size() * 4, wq.data()), bwp = cl.buffer(CL_MEM_READ_ONLY, wp.size() * 4, wp.data());
    cl_mem bwh = cl.buffer(CL_MEM_READ_ONLY, wh.size() * 4, wh.data());
    cl_mem baq = cl.buffer(CL_MEM_READ_ONLY, aq.size() * 4, aq.data()), bap = cl.buffer(CL_MEM_READ_ONLY, ap.size() * 4, ap.data());
    cl_mem bas = cl.buffer(CL_MEM_READ_ONLY, asc.size() * 4, asc.data());
    cl_mem bc = cl.buffer(CL_MEM_READ_WRITE, c0.size() * 4);
    std::vector<float> ref, out(c0.size());
    const int splits = sh.a != 2 && K > 8192 ? (nsb + 16) / 17 : 1;  // (h8_splits: pieces of 17 superblocks)
    for (size_t v = 0; v < vs.size(); ++v) {
      if (!ks[v] || M % vs[v].tm || N % (16 * vs[v].nc)) continue;  // (a shape the tile does not divide: skipped)
      const Variant& va = vs[v];
      Row& row = res[v][si];
      const int tm = va.tm, nb = N / (16 * va.nc);
      int wgy = 1;
      while (wgy < 8 && (M / tm) % (wgy * 2) == 0) wgy *= 2;
      size_t gws[2] = {(size_t)8 * nb * (M / tm / wgy), (size_t)wgy}, lws[2] = {8, (size_t)wgy};
      cl_kernel k = ks[v];
      auto launch = [&] {
        for (int sp = 0; sp < splits; ++sp) {
          int s0 = nsb * sp / splits, s1 = nsb * (sp + 1) / splits, ac = sp == 0 ? sh.a : 1, MM = M, NN = N, KK = K,
              ldc = N, coff = 0, rg = 4;
          set_args(k, 0, va.pw ? bwp : bwq, bwh, va.pa ? bap : baq, bas, bc, MM, NN, KK, ldc, coff, ac, rg, s0, s1);
          CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 2, nullptr, gws, lws, 0, nullptr, nullptr));
        }
      };
      try {
        CL_CHECK(clEnqueueWriteBuffer(cl.queue, bc, CL_TRUE, 0, c0.size() * 4, c0.data(), 0, nullptr, nullptr));
        launch();
        CL_CHECK(clEnqueueReadBuffer(cl.queue, bc, CL_TRUE, 0, out.size() * 4, out.data(), 0, nullptr, nullptr));
        if (ref.empty()) {
          ref = out;
          size_t changed = 0, bad = 0;
          for (size_t i = 0; i < out.size(); ++i) { changed += out[i] != c0[i]; bad += !std::isfinite(out[i]); }
          if (bad || changed < out.size() / 2) printf("  (%s: the reference looks wrong: %zu not finite, %zu changed)\n", sh.name, bad, changed);
        } else {
          for (size_t i = 0; i < out.size(); ++i) row.diff += memcmp(&out[i], &ref[i], 4) != 0;
          row.ok = row.diff == 0;
        }
        std::vector<double> ms;
        for (int it = 0; it < iters; ++it) {
          const auto t0 = clk::now();
          launch();
          zrt::sync();
          ms.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        row.ms = ms[ms.size() / 2], row.best = ms[0], row.ran = true;
      } catch (const std::exception& e) {
        printf("%s on %s: failed (%s)\n", va.name, sh.name, e.what());
      }
    }
    for (cl_mem m : {bwq, bwp, bwh, baq, bap, bas, bc}) clReleaseMemObject(m);
  }

  // ---- the table: per shape and the shapes together
  printf("\n%d tokens; TOPS from the median of %d runs%s\n", M, iters, peak > 0 ? ", % of the measured peak" : "");
  printf("%-14s %8s", "variant", "spill");
  for (const Shape& s : shapes) printf(" | %-10s %5dx%-5d", s.name, s.N, s.K);
  printf(" | %-16s | %s\n", "all shapes", "bits vs default");
  for (size_t v = 0; v < vs.size(); ++v) {
    printf("%-14s %6zu B", vs[v].name, res[v][0].spill);
    double ops = 0, ms = 0;
    bool all_ran = true;
    size_t diff = 0;
    for (size_t si = 0; si < shapes.size(); ++si) {
      const Row& r = res[v][si];
      const double o = 2.0 * M * shapes[si].N * shapes[si].K;
      if (!r.ran) { printf(" | %-22s", "-"); all_ran = false; continue; }
      const double t = o / r.ms / 1e9;
      if (peak > 0) printf(" | %7.2f ms %5.1f T %3.0f%%", r.ms, t, 100 * t / peak);
      else printf(" | %7.2f ms %5.2f T     ", r.ms, t);
      ops += o, ms += r.ms, diff += r.diff;
    }
    if (all_ran && ms > 0) {
      const double t = ops / ms / 1e9;
      if (peak > 0) printf(" | %5.1f T  %3.0f%%    ", t, 100 * t / peak);
      else printf(" | %5.2f T          ", t);
    } else printf(" | %-16s", "-");
    printf(" | %s\n", v == 0 ? "(reference)" : !all_ran ? "-" : diff ? (std::to_string(diff) + " values differ").c_str() : "identical");
  }
  return 0;
}

// arcinferzero NPU experiment, step 2: one Qwen FFN input projection pair (ffn_gate + ffn_up of a real layer) split
// between the GPU and the NPU, end to end, outside the engine:
//   GPU  quantizes the activation rows per row (int8 + scale) straight into memory the NPU's context maps too
//        (external memmap), then runs the engine's prompt GEMM (zero/kernels/gemm.cl, W8) on its share of the columns;
//   NPU  at the same time, the other share in the "fqi" graph form (int8 inputs, FakeQuantize with constant bounds, the
//        row and column scales after the product: exact per-row int8 on its int8 units), its outputs in the same
//        shared memory; the compiled graph comes from the blob cache after the first run;
//   GPU  then copies the NPU's f16 outputs into the f32 result columns.
// Checked: the NPU's columns against the same int8 scheme on the CPU (from the int8 rows the GPU wrote) and against
// f32; the merge against the NPU's outputs bit for bit. Timed: GPU alone over all columns against the split, runs
// alternated (the laptop's temperature moves both). The GPU GEMM's own weights are random W8 (its timing only: its
// accuracy is the engine's, tested elsewhere).
// usage: npu_ffn model.gguf [layer = 11] [tokens = 4096] [npu share = 0.333] [reps = 6]
#include "../clrt.h"
#include "npu_rt.h"
#include "../../src/gguf.h"
#include "../../src/quants.h"
#include <algorithm>
#include <cmath>
#include <random>

static const char* kGpuSrc = R"CLC(
// one work-group of 256 per row: absmax, then q = rint(x * 127 / absmax); the scale written as f16 absmax (the fqi
// graph's row scale: its FakeQuantize maps -127..127 to -1..1)
__kernel void rowq(__global const float* x, int K, __global char* q, __global half* s) {
  const int r = get_group_id(0), l = get_local_id(0);
  __local float red[256];
  float m = 0.0f;
  for (int k = l; k < K; k += 256) m = fmax(m, fabs(x[(size_t)r * K + k]));
  red[l] = m;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = 128; o > 0; o >>= 1) {
    if (l < o) red[l] = fmax(red[l], red[l + o]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float amax = red[0], inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  for (int k = l; k < K; k += 256) q[(size_t)r * K + k] = (char)clamp((int)rint(x[(size_t)r * K + k] * inv), -127, 127);
  if (l == 0) vstore_half(amax, r, s);
}
// C[m][coff + j] = c16[m][j]: the NPU's columns into the f32 result
__kernel void merge(__global const half* c16, int nn, __global float* C, int ldc, int coff) {
  const int j = get_global_id(0), m = get_global_id(1);
  C[(size_t)m * ldc + coff + j] = vload_half((size_t)m * nn + j, c16);
}
)CLC";

static std::vector<float> rows_f32(const GgufFile& g, const std::string& name, int r0, int n, int& K) {
  const GgufTensor& t = g.get(name);
  K = (int)t.dims[0];
  const size_t rb = t.nbytes / t.dims[1];
  std::vector<float> w((size_t)n * K);
#pragma omp parallel for
  for (int r = 0; r < n; ++r) dequant_row(t.type, t.data + (size_t)(r0 + r) * rb, K, &w[(size_t)r * K]);
  return w;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: npu_ffn model.gguf [layer] [tokens] [npu share] [reps]\n"); return 1; }
  setvbuf(stdout, nullptr, _IONBF, 0);
  const int L = argc > 2 ? atoi(argv[2]) : 11, M = argc > 3 ? atoi(argv[3]) : 4096, reps = argc > 5 ? atoi(argv[5]) : 6;
  const double share = argc > 4 ? atof(argv[4]) : 0.333;
  GgufFile gg(argv[1]);
  const std::string p = "blk." + std::to_string(L) + ".";
  const GgufTensor& tg = gg.get(p + "ffn_gate.weight");
  const int K = (int)tg.dims[0], N = (int)tg.dims[1];
  const int Nn = std::max(64, (int)std::lround(N * share / 64.0) * 64), Ng = N - Nn;  // (both multiples of 16)
  if (K % 256 || Ng % 16 || M % 128) { fprintf(stderr, "unsupported shape\n"); return 1; }
  printf("layer %d ffn_gate / ffn_up: K %d, N %d -> GPU %d + NPU %d columns, %d tokens\n", L, K, N, Ng, Nn, M);

  // Level Zero for every device type first (the GPU runtime below asks for GPUs only, which would hide the NPU)
  if (!ze_load() || zeInit(0) != ZE_RESULT_SUCCESS) { fprintf(stderr, "Level Zero init failed\n"); return 1; }
  ClRt gpu;
  if (!gpu.init()) return 1;
  npu::Rt npu;
  if (!npu.init()) { fprintf(stderr, "no NPU\n"); return 1; }

  // ---- shared memory: the int8 rows and their scales (GPU writes, NPU reads), the NPU's two f16 outputs
  const size_t a_off = 0, sa_off = a_off + (size_t)M * K, cg_off = (sa_off + (size_t)M * 2 + 4095) / 4096 * 4096,
               cu_off = cg_off + (size_t)M * Nn * 2, total = (cu_off + (size_t)M * Nn * 2 + 65535) / 65536 * 65536;
  uint8_t* shm = (uint8_t*)VirtualAlloc(nullptr, total, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (!shm || npu::import_memory(zrt::ctx, shm, total) != shm || npu::import_memory(npu.ctx, shm, total) != shm) {
    fprintf(stderr, "cannot map the shared memory into both contexts\n");
    return 1;
  }
  auto view = [&](size_t off, size_t bytes) {  // a cl_mem of the GPU runtime on part of the shared block
    ZMem* m = new ZMem{(char*)shm + off, bytes, false};
    zrt::D().mems.insert(m);
    return (cl_mem)m;
  };
  cl_mem g_aq = view(a_off, (size_t)M * K), g_sa = view(sa_off, (size_t)M * 2), g_cg = view(cg_off, (size_t)M * Nn * 2),
         g_cu = view(cu_off, (size_t)M * Nn * 2);
  (void)g_cu;  // (up's NPU columns are checked in the shared memory directly; the merge is timed with gate's)

  // ---- activations: unit-RMS rows (a few large values) times the layer's FFN norm weights, as the GPU holds them
  std::vector<float> nw(K, 1.0f);
  if (const GgufTensor* t = gg.find(p + "post_attention_norm.weight")) dequant_row(t->type, t->data, K, nw.data());
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x((size_t)M * K);
  for (int m = 0; m < M; ++m) {
    double ss = 0;
    for (int k = 0; k < K; ++k) {
      const float a = nd(rng), b = nd(rng);
      x[(size_t)m * K + k] = a * (1.0f + 0.5f * b * b);
      ss += (double)x[(size_t)m * K + k] * x[(size_t)m * K + k];
    }
    const float inv = (float)(1.0 / std::sqrt(ss / K + 1e-6));
    for (int k = 0; k < K; ++k) x[(size_t)m * K + k] *= inv * nw[k];
  }
  cl_mem g_x = gpu.buffer(CL_MEM_READ_ONLY, x.size() * 4, x.data());
  cl_mem g_c = gpu.buffer(CL_MEM_READ_WRITE, (size_t)M * N * 4);  // (gate's f32 result; up's alike, not kept here)

  // ---- the NPU's weight rows (the last Nn of gate and up): int8 with one scale per row
  printf("reading the NPU's weight rows...\n");
  int Kw = 0;
  const std::vector<float> wg = rows_f32(gg, p + "ffn_gate.weight", Ng, Nn, Kw), wu = rows_f32(gg, p + "ffn_up.weight", Ng, Nn, Kw);
  int8_t* bq[2] = {(int8_t*)npu.host((size_t)Nn * K), (int8_t*)npu.host((size_t)Nn * K)};
  uint16_t* bs[2] = {(uint16_t*)npu.host((size_t)Nn * 2), (uint16_t*)npu.host((size_t)Nn * 2)};
  uint16_t* sa16 = (uint16_t*)(shm + sa_off);
  std::vector<float> bsf[2];
  for (int t = 0; t < 2; ++t) {
    const std::vector<float>& w = t == 0 ? wg : wu;
    bsf[t].resize(Nn);
    for (int r = 0; r < Nn; ++r) {
      float amax = 0;
      for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(w[(size_t)r * K + k]));
      const float inv = amax > 0 ? 127.0f / amax : 0.0f;
      for (int k = 0; k < K; ++k) bq[t][(size_t)r * K + k] = (int8_t)std::clamp((int)std::lrint(w[(size_t)r * K + k] * inv), -127, 127);
      bs[t][r] = npu::f2h(amax);
      bsf[t][r] = npu::h2f(bs[t][r]) / 127.0f;
    }
  }
  // ---- the NPU graph: both products on one quantized A
  npu::Ir ir;
  const auto ilo = ir.const_f16("ilo", {1, 1}, {-127.0f}), ihi = ir.const_f16("ihi", {1, 1}, {127.0f});
  const auto olo = ir.const_f16("olo", {1, 1}, {-1.0f}), ohi = ir.const_f16("ohi", {1, 1}, {1.0f});
  auto fq = [&](const char* nm, int rows) {
    const auto pin = ir.param(nm, {"i8", {rows, K}});
    return ir.fake_quantize(std::string(nm) + "_fq", ir.convert(std::string(nm) + "_f", pin, "f16"), ilo, ihi, olo, ohi, 255);
  };
  const auto A = fq("A", M);
  const auto SA = ir.param("SA", {"f16", {M, 1}});
  for (int t = 0; t < 2; ++t) {
    const std::string nm = t == 0 ? "G" : "U";
    const auto B = fq(t == 0 ? "BG" : "BU", Nn);
    const auto SB = ir.param(t == 0 ? "SBG" : "SBU", {"f16", {1, Nn}});
    ir.result(ir.multiply(nm + "_sb", ir.multiply(nm + "_sa", ir.matmul_tb(nm, A, B), SA), SB), nm + "_out");
  }
  double cms = 0;
  bool cached = false;
  ze_graph_handle_t gr = npu.compile(ir, cms, cached);
  if (!gr) return 1;
  printf("NPU graph %s in %.0f ms\n", cached ? "loaded from the blob cache" : "compiled (saved to the blob cache)", cms);
  const std::vector<int> outs = npu.outputs(gr);
  if (outs.size() != 2 || !npu.set(gr, "A", shm + a_off) || !npu.set(gr, "SA", shm + sa_off) || !npu.set(gr, "BG", bq[0]) ||
      !npu.set(gr, "BU", bq[1]) || !npu.set(gr, "SBG", bs[0]) || !npu.set(gr, "SBU", bs[1]) ||
      npu.g->pfnSetArgumentValue(gr, outs[0], shm + cg_off) != 0 || npu.g->pfnSetArgumentValue(gr, outs[1], shm + cu_off) != 0) {
    fprintf(stderr, "cannot set the NPU graph's arguments\n");
    return 1;
  }
  if (!npu.initialize(gr)) { fprintf(stderr, "NPU graph initialize failed\n"); return 1; }

  // ---- the GPU's prompt GEMM (zero/kernels/gemm.cl as the engine builds it for W8 weights) on random W8
  const int SG = 16, nsb = K / 256, G = N / SG, TM = 16;
  std::vector<uint32_t> wq((size_t)G * nsb * 4 * 16 * SG), wh((size_t)G * nsb * 10 * SG);
  for (auto& v : wq) v = (rng() & 0x7F7F7F7Fu) ^ (rng() & 0x80808080u);
  for (size_t gs = 0; gs < (size_t)G * nsb; ++gs)
    for (int l = 0; l < SG; ++l) {
      uint32_t* h = &wh[gs * 10 * SG];
      const float d = 0.001f;
      memcpy(&h[l], &d, 4);
      h[SG + l] = 0;
      for (int f = 2; f < 6; ++f) h[f * SG + l] = (rng() & 0x3F3F3F3Fu) | 0x01010101u;
      for (int f = 6; f < 10; ++f) h[f * SG + l] = 0;
    }
  std::vector<int8_t> aq((size_t)M * K);
  for (auto& v : aq) v = (int8_t)((int)(rng() % 255) - 127);
  std::vector<float> asc((size_t)(K / 32) * M, 0.01f);
  std::vector<uint16_t> absum((size_t)(K / 16) * M, 0);
  cl_mem bwq = gpu.buffer(CL_MEM_READ_ONLY, wq.size() * 4, wq.data()), bwh = gpu.buffer(CL_MEM_READ_ONLY, wh.size() * 4, wh.data());
  cl_mem baq = gpu.buffer(CL_MEM_READ_ONLY, aq.size(), aq.data()), basc = gpu.buffer(CL_MEM_READ_ONLY, asc.size() * 4, asc.data());
  cl_mem babs = gpu.buffer(CL_MEM_READ_ONLY, absum.size() * 2, absum.data());
  cl_kernel kg = gpu.kernel(gpu.build(read_file("zero/kernels/gemm.cl"), "-cl-std=CL3.0 -cl-mad-enable -cl-intel-256-GRF-per-thread "
                                                                        "-DACT32 -DQTYPE=100 -DTM=16"), "gemm");
  cl_program aux = gpu.build(kGpuSrc, "-cl-std=CL3.0");
  cl_kernel krq = gpu.kernel(aux, "rowq"), kmg = gpu.kernel(aux, "merge");
  int wgy = 1;
  while (wgy < 8 && (M / TM) % (wgy * 2) == 0) wgy *= 2;
  auto gemm = [&](int ncols) {  // (one projection: the first ncols columns)
    size_t gws[2] = {(size_t)ncols * (M / TM / wgy), (size_t)wgy}, lws[2] = {16, (size_t)wgy};
    int MM = M, NN = ncols, KK = K, ldc = N, coff = 0, acc = 0, rg = 4, s0 = 0, s1 = nsb;
    set_args(kg, 0, bwq, bwh, baq, basc, babs, g_c, MM, NN, KK, ldc, coff, acc, rg, s0, s1);
    CL_CHECK(clEnqueueNDRangeKernel(gpu.queue, kg, 2, nullptr, gws, lws, 0, nullptr, nullptr));
  };
  auto rowq = [&]() {
    int KK = K;
    set_args(krq, 0, g_x, KK, g_aq, g_sa);
    size_t gws[1] = {(size_t)M * 256}, lws[1] = {256};
    CL_CHECK(clEnqueueNDRangeKernel(gpu.queue, krq, 1, nullptr, gws, lws, 0, nullptr, nullptr));
  };
  auto merge = [&](cl_mem c16) {
    int nn = Nn, ldc = N, coff = Ng;
    set_args(kmg, 0, c16, nn, g_c, ldc, coff);
    size_t gws[2] = {(size_t)Nn, (size_t)M};
    CL_CHECK(clEnqueueNDRangeKernel(gpu.queue, kmg, 2, nullptr, gws, nullptr, 0, nullptr, nullptr));
  };
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto ms_since = [&](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double, std::milli>(now() - t).count(); };
  auto run_gpu_alone = [&]() {
    const auto t0 = now();
    gemm(N);  // gate
    gemm(N);  // up
    clFinish(gpu.queue);
    return ms_since(t0);
  };
  double npu_ms = 0;
  auto run_split = [&]() {
    const auto t0 = now();
    rowq();
    clFinish(gpu.queue);
    const auto tn = now();
    if (!npu.start(gr)) { fprintf(stderr, "NPU start failed\n"); exit(1); }
    gemm(Ng);
    gemm(Ng);
    clFinish(gpu.queue);
    if (!npu.wait()) { fprintf(stderr, "NPU run failed\n"); exit(1); }
    npu_ms = ms_since(tn);
    merge(g_cg);
    clFinish(gpu.queue);
    return ms_since(t0);
  };

  // ---- one checked run
  run_split();
  // the NPU's outputs against the same int8 scheme (from the GPU's int8 rows) and against f32; the merge bit for bit
  std::vector<float> cgpu((size_t)M * N);
  CL_CHECK(clEnqueueReadBuffer(gpu.queue, g_c, CL_TRUE, 0, cgpu.size() * 4, cgpu.data(), 0, nullptr, nullptr));
  const int8_t* aq_shm = (const int8_t*)(shm + a_off);
  double es[2] = {0, 0}, ef[2] = {0, 0}, r2[2] = {0, 0};
  size_t merge_bad = 0;
  for (int t = 0; t < 2; ++t) {
    const uint16_t* c16 = (const uint16_t*)(shm + (t == 0 ? cg_off : cu_off));
    const std::vector<float>& w = t == 0 ? wg : wu;
    for (int m = 0; m < M; m += M / 32)
      for (int j = 0; j < Nn; j += 7) {
        double s = 0, f = 0;
        for (int k = 0; k < K; ++k) {
          s += (double)aq_shm[(size_t)m * K + k] * bq[t][(size_t)j * K + k];
          f += (double)x[(size_t)m * K + k] * w[(size_t)j * K + k];
        }
        s *= (npu::h2f(sa16[m]) / 127.0) * bsf[t][j];
        const double got = npu::h2f(c16[(size_t)m * Nn + j]);
        es[t] += (got - s) * (got - s);
        ef[t] += (got - f) * (got - f);
        r2[t] += f * f;
        if (t == 0 && cgpu[(size_t)m * N + Ng + j] != (float)got) ++merge_bad;
      }
  }
  printf("NPU gate / up vs the same int8 scheme: %.3f %% / %.3f %%; vs f32: %.3f %% / %.3f %%; merge: %s\n",
         100 * std::sqrt(es[0] / r2[0]), 100 * std::sqrt(es[1] / r2[1]), 100 * std::sqrt(ef[0] / r2[0]), 100 * std::sqrt(ef[1] / r2[1]),
         merge_bad ? "WRONG" : "bit-identical");

  // ---- timing, alternated
  run_gpu_alone();
  std::vector<double> ta, ts, tn;
  for (int r = 0; r < reps; ++r) {
    ta.push_back(run_gpu_alone());
    ts.push_back(run_split());
    tn.push_back(npu_ms);
  }
  auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  const double a = med(ta), s = med(ts);
  printf("gate + up, %d tokens: GPU alone %.1f ms; split %.1f ms (NPU part from its start %.1f ms) -> %.2fx\n", M, a, s, med(tn), a / s);
  return 0;
}

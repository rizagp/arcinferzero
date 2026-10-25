// arcinferzero NPU experiment, step 4: the matrix product C[M][N] = A[M][K] . B[N][K]^T on the NPU in the forms the
// engine could hand it, each timed and checked against an exact CPU computation of the same scheme:
//   f16  A, B in f16
//   i8   A int8 + one scale per row, B int8 + one scale per row (dequantized in the graph: Convert, Multiply)
//   w8   A in f16, B int8 + one scale per row (weight-only int8)
//   fq   A, B in f16 through FakeQuantize (255 levels) with per-row bounds given at run time (graph inputs): the int8
//        product of i8, in the pattern the NPU compiler runs on its int8 units
//   fqn / fqi  the same with constant bounds on rows normalized to [-1, 1] (fqn, f16) or int8 inputs (fqi), the scales
//        multiplied in after the product
//   w8g  A in f16, B as the engine's W8 weights (int8 with one scale per 32 values, [N][K / 32]) dequantized in the
//        graph (Reshape, Convert, Multiply): the f16 product on the GPU's own decoded weights
// usage: npu_gemm_modes [M K N] [reps] [modes, e.g. f16,i8,w8]
#include "npu_ir.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

struct Npu {
  ze_driver_handle_t drv = nullptr;
  ze_device_handle_t dev = nullptr;
  ze_context_handle_t ctx = nullptr;
  ze_command_list_handle_t cl = nullptr;
  ze_graph_dditable_t* g = nullptr;
  ze_device_graph_properties_t gp{};
  bool init() {
    if (!ze_load() || zeInit(0) != ZE_RESULT_SUCCESS) return false;
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
    return zeCommandListCreateImmediate(ctx, dev, &qd, &cl) == 0;
  }
  ze_graph_handle_t compile(const npu::Ir& ir, double& ms) {
    const std::vector<uint8_t> in = ir.serialize(gp.compilerVersion);
    ze_graph_desc_t gd{ZE_STYPE_GRAPH_DESC, nullptr, ZE_GRAPH_FORMAT_NGRAPH_LITE, in.size(), in.data(), ""};
    ze_graph_handle_t gr = nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    const ze_result_t r = g->pfnCreate(ctx, dev, &gd, &gr);
    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (r != 0) {
      fprintf(stderr, "graph create failed: 0x%x\n", r);
      uint32_t ln = 0;
      if (g->pfnBuildLogGetString && g->pfnBuildLogGetString(gr, &ln, nullptr) == 0 && ln) {
        std::string log(ln, '\0');
        g->pfnBuildLogGetString(gr, &ln, log.data());
        fprintf(stderr, "build log: %s\n", log.c_str());
      }
      if (getenv("NPU_DUMP")) fprintf(stderr, "%s\n", ir.xml().c_str());
      return nullptr;
    }
    return gr;
  }
  void* host(size_t bytes) {
    ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
    void* p = nullptr;
    return zeMemAllocHost(ctx, &hd, bytes, 4096, &p) == 0 ? p : nullptr;
  }
  // argument index by its name (inputs: the parameters' names; the output: the last argument)
  int arg(ze_graph_handle_t gr, const char* name) {
    ze_graph_properties_t pr{ZE_STYPE_GRAPH_PROPERTIES, nullptr, 0};
    g->pfnGetProperties(gr, &pr);
    for (uint32_t i = 0; i < pr.numGraphArgs; ++i) {
      ze_graph_argument_properties_t ap{};
      ap.stype = ZE_STYPE_GRAPH_ARGUMENT_PROPERTIES;
      g->pfnGetArgumentProperties(gr, i, &ap);
      if (!strcmp(ap.name, name) || (!strcmp(name, "out") && ap.type != 0)) return (int)i;
    }
    fprintf(stderr, "no graph argument %s; the graph has:", name);
    for (uint32_t i = 0; i < pr.numGraphArgs; ++i) {
      ze_graph_argument_properties_t ap{};
      ap.stype = ZE_STYPE_GRAPH_ARGUMENT_PROPERTIES;
      g->pfnGetArgumentProperties(gr, i, &ap);
      fprintf(stderr, " [%u %s type %u]", i, ap.name, ap.type);
    }
    fprintf(stderr, "\n");
    return -1;
  }
  bool set(ze_graph_handle_t gr, const char* name, void* p) {
    const int i = arg(gr, name);
    return i >= 0 && g->pfnSetArgumentValue(gr, (uint32_t)i, p) == 0;
  }
  // median of reps runs, ms
  double run(ze_graph_handle_t gr, int reps) {
    if (g->pfnAppendGraphInitialize(cl, gr, nullptr, 0, nullptr) != 0 || zeCommandListHostSynchronize(cl, UINT64_MAX) != 0) return -1;
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
      const auto s0 = std::chrono::steady_clock::now();
      if (g->pfnAppendGraphExecute(cl, gr, nullptr, nullptr, 0, nullptr) != 0 || zeCommandListHostSynchronize(cl, UINT64_MAX) != 0)
        return -1;
      t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
  }
};

int main(int argc, char** argv) {
  const int M = argc > 3 ? atoi(argv[1]) : 2048, K = argc > 3 ? atoi(argv[2]) : 5120, N = argc > 3 ? atoi(argv[3]) : 17408;
  const int reps = argc > 4 ? atoi(argv[4]) : 10;
  const std::string modes = argc > 5 ? argv[5] : "f16,i8,w8";
  setvbuf(stdout, nullptr, _IONBF, 0);
  Npu n;
  if (!n.init()) { fprintf(stderr, "NPU init failed\n"); return 1; }
  printf("NPU, compiler %u.%u; C[%d][%d] = A[%d][%d] . B[%d][%d]^T\n", n.gp.compilerVersion.major, n.gp.compilerVersion.minor, M,
         N, M, K, N, K);
  // data: activations ~ N(0, 1) with a few large channels, weights ~ N(0, 0.02)
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> a((size_t)M * K), b((size_t)N * K);
  for (int k = 0, big = 0; k < K; ++k) {
    const float ch = (k * 2654435761u) % 97 == 0 ? 8.0f : 1.0f;  // (~1 % of the channels 8x larger)
    big += ch > 1;
    for (int m = 0; m < M; ++m) a[(size_t)m * K + k] = nd(rng) * ch;
  }
  for (auto& x : b) x = nd(rng) * 0.02f;
  auto rowq = [](const std::vector<float>& x, int rows, int K, std::vector<int8_t>& q, std::vector<float>& s) {
    q.resize((size_t)rows * K);
    s.resize(rows);
    for (int r = 0; r < rows; ++r) {
      float amax = 0;
      for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(x[(size_t)r * K + k]));
      s[r] = npu::h2f(npu::f2h(amax / 127.0f));  // (the scale as the graph holds it, f16)
      const float inv = s[r] > 0 ? 1.0f / s[r] : 0.0f;
      for (int k = 0; k < K; ++k) q[(size_t)r * K + k] = (int8_t)std::clamp((int)std::lrint(x[(size_t)r * K + k] * inv), -127, 127);
    }
  };
  std::vector<int8_t> qa, qb;
  std::vector<float> sa, sb;
  rowq(a, M, K, qa, sa);
  rowq(b, N, K, qb, sb);
  // (w8g: the weights as the engine's W8, int8 with one f16 scale per 32 values)
  std::vector<int8_t> qb32((size_t)N * K);
  std::vector<float> sb32((size_t)N * (K / 32));
  for (size_t blk = 0; blk < sb32.size(); ++blk) {
    float amax = 0;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(b[blk * 32 + i]));
    sb32[blk] = npu::h2f(npu::f2h(amax / 127.0f));
    const float inv = sb32[blk] > 0 ? 1.0f / sb32[blk] : 0.0f;
    for (int i = 0; i < 32; ++i) qb32[blk * 32 + i] = (int8_t)std::clamp((int)std::lrint(b[blk * 32 + i] * inv), -127, 127);
  }
  const double flops = 2.0 * M * K * N;
  size_t pos = 0;
  while (pos < modes.size()) {
    const size_t e = modes.find(',', pos);
    const std::string mode = modes.substr(pos, e == std::string::npos ? std::string::npos : e - pos);
    pos = e == std::string::npos ? modes.size() : e + 1;
    npu::Ir ir;
    npu::Ir::Out A, B;
    // (cv / cvk: the product as a 1x1 convolution, NCHW: A as [1][K][M][1] (A^T), the result [1][N][M][1] (C^T);
    //  f16k / cvk: B a constant of the graph, so the compiler lays it out for its tiles)
    const bool conv = mode == "cv" || mode == "cvk", bconst = mode == "f16k" || mode == "cvk";
    if (mode == "f16" || mode == "f16k") {
      A = ir.param("A", {"f16", {M, K}});
      B = bconst ? ir.const_f16("B", {N, K}, b) : ir.param("B", {"f16", {N, K}});
    } else if (conv) {
      A = ir.param("A", {"f16", {1, K, M, 1}});
      B = bconst ? ir.const_f16("B", {N, K, 1, 1}, b) : ir.param("B", {"f16", {N, K, 1, 1}});
    } else if (mode == "i8" || mode == "w8") {
      if (mode == "i8") {
        const auto qai = ir.param("A", {"i8", {M, K}});
        const auto sai = ir.param("SA", {"f16", {M, 1}});
        A = ir.multiply("A_dq", ir.convert("A_f", qai, "f16"), sai);
      } else {
        A = ir.param("A", {"f16", {M, K}});
      }
      const auto qbi = ir.param("B", {"i8", {N, K}});
      const auto sbi = ir.param("SB", {"f16", {N, 1}});
      B = ir.multiply("B_dq", ir.convert("B_f", qbi, "f16"), sbi);
    } else if (mode == "fq") {
      const auto af = ir.param("A", {"f16", {M, K}});
      const auto la = ir.param("LA", {"f16", {M, 1}});
      const auto ha = ir.param("HA", {"f16", {M, 1}});
      const auto bf = ir.param("B", {"f16", {N, K}});
      const auto lb = ir.param("LB", {"f16", {N, 1}});
      const auto hb = ir.param("HB", {"f16", {N, 1}});
      A = ir.fake_quantize("A_fq", af, la, ha, 255);
      B = ir.fake_quantize("B_fq", bf, lb, hb, 255);
    } else if (mode == "w8g") {
      A = ir.param("A", {"f16", {M, K}});
      const auto qbi = ir.param("B", {"i8", {N, K}});
      const auto sbi = ir.param("SB", {"f16", {N, K / 32, 1}});
      const auto b3 = ir.convert("B_f", ir.reshape("B_3", qbi, {N, K / 32, 32}), "f16");
      B = ir.reshape("B_dq", ir.multiply("B_s", b3, sbi), {N, K});
    } else if (mode == "fqn" || mode == "fqi") {
      // constant bounds: rows normalized to [-1, 1] (fqn, f16) or the int8 values themselves (fqi, i8 -> f16), written
      // out on [-1, 1]; the per-row scales (127 x amax / 127 = amax) multiplied in after the product
      const bool i8in = mode == "fqi";
      const float r = i8in ? 127.0f : 1.0f;
      const auto ilo = ir.const_f16("ilo", {1, 1}, {-r}), ihi = ir.const_f16("ihi", {1, 1}, {r});
      const auto olo = ir.const_f16("olo", {1, 1}, {-1.0f}), ohi = ir.const_f16("ohi", {1, 1}, {1.0f});
      auto in = [&](const char* nm, int rows) {
        const auto p = ir.param(nm, {i8in ? "i8" : "f16", {rows, K}});
        return ir.fake_quantize(std::string(nm) + "_fq", i8in ? ir.convert(std::string(nm) + "_f", p, "f16") : p, ilo, ihi, olo,
                                ohi, 255);
      };
      A = in("A", M);
      B = in("B", N);
    } else {
      fprintf(stderr, "unknown mode %s\n", mode.c_str());
      continue;
    }
    if (mode == "fqn" || mode == "fqi") {
      const auto sa_in = ir.param("SA", {"f16", {M, 1}}), sb_in = ir.param("SB", {"f16", {1, N}});
      ir.result(ir.multiply("C_sb", ir.multiply("C_sa", ir.matmul_tb("C", A, B), sa_in), sb_in));
    } else if (conv) {
      ir.result(ir.conv1x1("C", A, B));
    } else {
      ir.result(ir.matmul_tb("C", A, B));
    }
    double cms = 0;
    if (getenv("NPU_TRACE")) fprintf(stderr, "[%s] compiling\n", mode.c_str());
    ze_graph_handle_t gr = n.compile(ir, cms);
    if (getenv("NPU_TRACE")) fprintf(stderr, "[%s] compiled %p\n", mode.c_str(), (void*)gr);
    if (!gr) continue;
    // inputs in host memory the NPU reads, the output it writes
    const bool a8 = mode == "i8" || mode == "fqi", b16 = mode == "f16" || mode == "fq" || mode == "fqn" || conv || bconst;
    const bool qrows = mode == "i8" || mode == "fq" || mode == "fqn" || mode == "fqi";  // (A as qa x sa)
    void* pa = n.host(a8 ? (size_t)M * K : (size_t)M * K * 2);
    void* pb = n.host(b16 ? (size_t)N * K * 2 : (size_t)N * K);
    void* psa = n.host((size_t)M * 2);
    const bool w8g = mode == "w8g";
    void* psb = n.host(w8g ? sb32.size() * 2 : (size_t)N * 2);
    void* pc = n.host((size_t)M * N * 2);
    void* pla = n.host((size_t)M * 2);
    void* pha = n.host((size_t)M * 2);
    void* plb = n.host((size_t)N * 2);
    void* phb = n.host((size_t)N * 2);
    if (!pa || !pb || !psa || !psb || !pc || !pla || !pha || !plb || !phb) { fprintf(stderr, "host allocation failed\n"); return 1; }
    // (fq: the bounds +- 127 * scale, so the 255 levels are the int8 grid of i8)
    for (int i = 0; i < M; ++i) { ((uint16_t*)pha)[i] = npu::f2h(127.0f * sa[i]); ((uint16_t*)pla)[i] = npu::f2h(-127.0f * sa[i]); }
    for (int i = 0; i < N; ++i) { ((uint16_t*)phb)[i] = npu::f2h(127.0f * sb[i]); ((uint16_t*)plb)[i] = npu::f2h(-127.0f * sb[i]); }
    const bool fqc = mode == "fqn" || mode == "fqi";  // (the scales after the product: 127 x amax / 127)
    for (size_t i = 0; i < (size_t)M * K; ++i) {
      if (a8) ((int8_t*)pa)[i] = qa[i];
      else if (mode == "fqn") ((uint16_t*)pa)[i] = npu::f2h(qa[i] / 127.0f);
      else if (conv) ((uint16_t*)pa)[(i % K) * M + i / K] = npu::f2h(a[i]);  // (A^T: [K][M])
      else ((uint16_t*)pa)[i] = npu::f2h(a[i]);
    }
    for (size_t i = 0; i < (size_t)N * K; ++i) {
      if (mode == "fqn") ((uint16_t*)pb)[i] = npu::f2h(qb[i] / 127.0f);
      else if (b16) ((uint16_t*)pb)[i] = npu::f2h(b[i]);
      else if (w8g) ((int8_t*)pb)[i] = qb32[i];
      else ((int8_t*)pb)[i] = qb[i];
    }
    for (int i = 0; i < M; ++i) ((uint16_t*)psa)[i] = npu::f2h(fqc ? 127.0f * sa[i] : sa[i]);
    if (w8g) for (size_t i = 0; i < sb32.size(); ++i) ((uint16_t*)psb)[i] = npu::f2h(sb32[i]);
    else for (int i = 0; i < N; ++i) ((uint16_t*)psb)[i] = npu::f2h(fqc ? 127.0f * sb[i] : sb[i]);
    if (!n.set(gr, "A", pa)) return 1;
    if (!bconst && !n.set(gr, "B", pb)) return 1;
    if (mode == "i8" || fqc) if (!n.set(gr, "SA", psa)) return 1;
    if (mode == "i8" || mode == "w8" || fqc || w8g) if (!n.set(gr, "SB", psb)) return 1;
    if (mode == "fq") {
      if (!n.set(gr, "LA", pla)) return 1;
      if (!n.set(gr, "HA", pha)) return 1;
      if (!n.set(gr, "LB", plb)) return 1;
      if (!n.set(gr, "HB", phb)) return 1;
    }
    if (!n.set(gr, "out", pc)) return 1;
    if (getenv("NPU_TRACE")) fprintf(stderr, "[%s] running\n", mode.c_str());
    const double ms = n.run(gr, reps);
    if (ms < 0) { fprintf(stderr, "%s: execute failed\n", mode.c_str()); continue; }
    // the same scheme on the CPU (exact integer sums times the scales; f16 inputs as stored), and plain f32
    double e_scheme = 0, e_f32 = 0, r2 = 0;
    const int rows = std::min(M, 32);
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < N; j += std::max(1, N / 512)) {
        double s = 0, f = 0;
        for (int k = 0; k < K; ++k) {
          const double av = qrows ? (double)qa[(size_t)i * K + k] * sa[i] : npu::h2f(npu::f2h(a[(size_t)i * K + k]));
          const double bv = mode == "f16" || conv || bconst ? npu::h2f(npu::f2h(b[(size_t)j * K + k]))
                            : w8g ? (double)qb32[(size_t)j * K + k] * sb32[((size_t)j * K + k) / 32]
                                  : (double)qb[(size_t)j * K + k] * sb[j];
          s += av * bv;
          f += (double)a[(size_t)i * K + k] * b[(size_t)j * K + k];
        }
        const double got = npu::h2f(((uint16_t*)pc)[conv ? (size_t)j * M + i : (size_t)i * N + j]);  // (conv: C^T)
        e_scheme += (got - s) * (got - s);
        e_f32 += (got - f) * (got - f);
        r2 += f * f;
      }
    printf("%-4s compile %5.0f ms, run %8.3f ms = %5.2f T(FL)OPS; error vs the same scheme %.3f %%, vs f32 %.3f %%\n", mode.c_str(),
           cms, ms, flops / (ms * 1e-3) / 1e12, 100 * std::sqrt(e_scheme / r2), 100 * std::sqrt(e_f32 / r2));
    for (void* p : {pa, pb, psa, psb, pc, pla, pha, plb, phb}) zeMemFree(n.ctx, p);
    n.g->pfnDestroy(gr);
  }
  return 0;
}

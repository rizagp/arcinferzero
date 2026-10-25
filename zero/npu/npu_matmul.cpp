// arcinferzero NPU experiment, step 3: one MatMul graph on the NPU through Level Zero's graph extension alone (no
// OpenVINO runtime): an OpenVINO IR v11 model written here (C[M][N] = A[M][K] . B[N][K]^T, f16), serialized for the
// driver's compiler (ZE_GRAPH_FORMAT_NGRAPH_LITE), compiled, run on host memory, checked against the CPU and timed.
// usage: npu_matmul [M K N] [reps]      (default: a small check, then 2048 x 4096 x 4096)
#include "ze_npu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static uint16_t f2h(float f) {  // f32 -> f16, round to nearest even (normal range; the test values are small)
  uint32_t x;
  memcpy(&x, &f, 4);
  const uint32_t s = (x >> 16) & 0x8000u;
  int e = (int)((x >> 23) & 0xff) - 127 + 15;
  uint32_t m = x & 0x7fffffu;
  if (e <= 0) return (uint16_t)s;
  if (e >= 31) return (uint16_t)(s | 0x7c00u);
  uint32_t h = s | ((uint32_t)e << 10) | (m >> 13);
  const uint32_t rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
  return (uint16_t)h;
}
static float h2f(uint16_t h) {
  const uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ffu;
  uint32_t x;
  if (e == 0) {
    if (m == 0) x = s;
    else {  // subnormal
      int k = 0;
      uint32_t mm = m;
      while (!(mm & 0x400u)) { mm <<= 1; ++k; }
      x = s | ((uint32_t)(113 - k) << 23) | ((mm & 0x3ffu) << 13);
    }
  } else if (e == 31) x = s | 0x7f800000u | (m << 13);
  else x = s | ((e + 112) << 23) | (m << 13);
  float f;
  memcpy(&f, &x, 4);
  return f;
}

static std::string port(int id, const char* names, int d0, int d1) {
  std::string s = "<port id=\"" + std::to_string(id) + "\" precision=\"FP16\"";
  if (names) s += std::string(" names=\"") + names + "\"";
  return s + "><dim>" + std::to_string(d0) + "</dim><dim>" + std::to_string(d1) + "</dim></port>";
}
// A scalar f16 constant at `offset` bytes of the weights blob.
static std::string const_layer(int id, const std::string& name, size_t offset) {
  return "<layer id=\"" + std::to_string(id) + "\" name=\"" + name + "\" type=\"Const\" version=\"opset1\"><data element_type=\"f16\" "
         "shape=\"1,1\" offset=\"" + std::to_string(offset) + "\" size=\"2\"/><output><port id=\"0\" precision=\"FP16\"><dim>1</dim>"
         "<dim>1</dim></port></output></layer>\n";
}
static std::string edge(int fl, int fp, int tl, int tp) {
  return "<edge from-layer=\"" + std::to_string(fl) + "\" from-port=\"" + std::to_string(fp) + "\" to-layer=\"" + std::to_string(tl) +
         "\" to-port=\"" + std::to_string(tp) + "\"/>\n";
}
// C = A . B^T; with fq, each input through a 256-level FakeQuantize first (A in [-ra, ra], B in [-rb, rb]): the
// pattern the NPU compiler runs as an int8 matrix product. The FakeQuantize bounds go to `w` (f16 scalars).
static std::string matmul_ir(int M, int K, int N, bool fq, float ra, float rb, std::vector<uint8_t>& w) {
  w.clear();
  auto scalar = [&](float v) {
    const size_t off = w.size();
    const uint16_t h = f2h(v);
    w.push_back((uint8_t)(h & 0xff));
    w.push_back((uint8_t)(h >> 8));
    return off;
  };
  const std::string m = std::to_string(M), k = std::to_string(K), n = std::to_string(N);
  std::string L = "<?xml version=\"1.0\"?>\n<net name=\"mm\" version=\"11\">\n<layers>\n";
  L += "<layer id=\"0\" name=\"A\" type=\"Parameter\" version=\"opset1\"><data shape=\"" + m + "," + k +
       "\" element_type=\"f16\"/><output>" + port(0, "A", M, K) + "</output></layer>\n";
  L += "<layer id=\"1\" name=\"B\" type=\"Parameter\" version=\"opset1\"><data shape=\"" + n + "," + k +
       "\" element_type=\"f16\"/><output>" + port(0, "B", N, K) + "</output></layer>\n";
  std::string E;
  int a_src = 0, b_src = 0;
  if (fq) {
    // layers 10..13: A's bounds, 14: FakeQuantize(A); 20..23: B's bounds, 24: FakeQuantize(B)
    for (int t = 0; t < 2; ++t) {
      const float r = t == 0 ? ra : rb;
      const int base = t == 0 ? 10 : 20, rows = t == 0 ? M : N;
      const char* nm = t == 0 ? "A" : "B";
      L += const_layer(base + 0, std::string(nm) + "_il", scalar(-r));
      L += const_layer(base + 1, std::string(nm) + "_ih", scalar(r));
      L += const_layer(base + 2, std::string(nm) + "_ol", scalar(-r));
      L += const_layer(base + 3, std::string(nm) + "_oh", scalar(r));
      L += "<layer id=\"" + std::to_string(base + 4) + "\" name=\"" + nm + "_fq\" type=\"FakeQuantize\" version=\"opset1\">"
           "<data levels=\"256\" auto_broadcast=\"numpy\"/><input>" + port(0, nullptr, rows, K) +
           "<port id=\"1\" precision=\"FP16\"><dim>1</dim><dim>1</dim></port><port id=\"2\" precision=\"FP16\"><dim>1</dim><dim>1</dim></port>"
           "<port id=\"3\" precision=\"FP16\"><dim>1</dim><dim>1</dim></port><port id=\"4\" precision=\"FP16\"><dim>1</dim><dim>1</dim></port>"
           "</input><output>" + port(5, nullptr, rows, K) + "</output></layer>\n";
      E += edge(t, 0, base + 4, 0);
      for (int j = 0; j < 4; ++j) E += edge(base + j, 0, base + 4, j + 1);
    }
    a_src = 14;
    b_src = 24;
  }
  const int ap = fq ? 5 : 0;
  L += "<layer id=\"2\" name=\"mm\" type=\"MatMul\" version=\"opset1\"><data transpose_a=\"false\" transpose_b=\"true\"/>"
       "<input>" + port(0, nullptr, M, K) + port(1, nullptr, N, K) + "</input><output>" + port(2, "C", M, N) + "</output></layer>\n";
  L += "<layer id=\"3\" name=\"C/sink_port_0\" type=\"Result\" version=\"opset1\"><input>" + port(0, nullptr, M, N) +
       "</input></layer>\n</layers>\n<edges>\n";
  E += edge(a_src, ap, 2, 0) + edge(b_src, ap, 2, 1) + edge(2, 2, 3, 0);
  return L + E + "</edges>\n<rt_info/>\n</net>\n";
}
// the driver compiler's input: compiler version (2 x uint16), the number of parts (2), xml size + xml, weights size +
// weights (as the NPU plugin's driver adapter serializes a model)
static std::vector<uint8_t> serialize(ze_graph_compiler_version_info_t cv, const std::string& xml, const std::vector<uint8_t>& w) {
  std::vector<uint8_t> b;
  auto put = [&](const void* p, size_t n) { b.insert(b.end(), (const uint8_t*)p, (const uint8_t*)p + n); };
  const uint32_t parts = 2;
  const uint64_t xs = xml.size(), ws = w.size();
  put(&cv, sizeof cv);
  put(&parts, 4);
  put(&xs, 8);
  put(xml.data(), xml.size());
  put(&ws, 8);
  if (!w.empty()) put(w.data(), w.size());
  return b;
}

int main(int argc, char** argv) {
  if (!ze_load() || zeInit(0) != ZE_RESULT_SUCCESS) { fprintf(stderr, "Level Zero init failed\n"); return 1; }
  auto getExt = (ze_result_t(*)(ze_driver_handle_t, const char*, void**))(void*)GetProcAddress(
      GetModuleHandleA("ze_loader.dll"), "zeDriverGetExtensionFunctionAddress");
  uint32_t nd = 0;
  zeDriverGet(&nd, nullptr);
  std::vector<ze_driver_handle_t> drv(nd);
  zeDriverGet(&nd, drv.data());
  ze_driver_handle_t ndrv = nullptr;
  ze_device_handle_t dev = nullptr;
  for (auto d : drv) {
    uint32_t n = 1;
    ze_device_handle_t x = nullptr;
    if (zeDeviceGet(d, &n, &x) != 0 || !n) continue;
    ze_device_properties_t p{};
    p.stype = ZE_STYPE_DEVICE_PROPERTIES;
    zeDeviceGetProperties(x, &p);
    if (p.type == 5) { ndrv = d; dev = x; }
  }
  if (!dev || !getExt) { fprintf(stderr, "no NPU\n"); return 1; }
  ze_graph_dditable_t* g = nullptr;
  if (getExt(ndrv, "ZE_extension_graph_1_8", (void**)&g) != 0 || !g) { fprintf(stderr, "no graph extension\n"); return 1; }
  ze_device_graph_properties_t gp{};
  gp.stype = ZE_STYPE_DEVICE_GRAPH_PROPERTIES;
  g->pfnDeviceGetGraphProperties(dev, &gp);
  ze_context_desc_t cd{ZE_STYPE_CONTEXT_DESC, nullptr, 0};
  ze_context_handle_t ctx = nullptr;
  if (zeContextCreate(ndrv, &cd, &ctx) != 0) { fprintf(stderr, "zeContextCreate failed\n"); return 1; }
  ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0, 0, 0};
  ze_command_list_handle_t cl = nullptr;
  const ze_result_t rcl = zeCommandListCreateImmediate(ctx, dev, &qd, &cl);
  if (rcl != 0) { fprintf(stderr, "zeCommandListCreateImmediate failed: 0x%x\n", rcl); return 1; }

  struct Case { int M, K, N, reps; };
  std::vector<Case> cases;
  if (argc >= 4) cases.push_back({atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), argc > 4 ? atoi(argv[4]) : 10});
  else cases = {{64, 256, 128, 3}, {2048, 4096, 4096, 10}};
  const char* flags = getenv("NPU_FLAGS") ? getenv("NPU_FLAGS") : "";
  for (const Case& c : cases) {
    static const bool fq = getenv("NPU_Q") && atoi(getenv("NPU_Q")) != 0;  // (int8 through FakeQuantize)
    std::vector<uint8_t> wts;
    const std::string xml = matmul_ir(c.M, c.K, c.N, fq, 1.0f, 0.1f, wts);
    if (getenv("NPU_DUMP")) printf("%s\n", xml.c_str());
    const std::vector<uint8_t> in = serialize(gp.compilerVersion, xml, wts);
    ze_graph_desc_t gd{ZE_STYPE_GRAPH_DESC, nullptr, ZE_GRAPH_FORMAT_NGRAPH_LITE, in.size(), in.data(), flags};
    ze_graph_handle_t gr = nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    const ze_result_t rc = g->pfnCreate(ctx, dev, &gd, &gr);
    const double tc = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("M %d K %d N %d (%s): graph create 0x%x in %.0f ms\n", c.M, c.K, c.N, fq ? "int8 FakeQuantize" : "f16", rc, tc);
    if (rc != 0) {
      uint32_t ln = 0;
      if (g->pfnBuildLogGetString && g->pfnBuildLogGetString(gr, &ln, nullptr) == 0 && ln) {
        std::string log(ln, '\0');
        g->pfnBuildLogGetString(gr, &ln, log.data());
        printf("build log: %s\n", log.c_str());
      }
      return 1;
    }
    ze_graph_properties_t pr{ZE_STYPE_GRAPH_PROPERTIES, nullptr, 0};
    g->pfnGetProperties(gr, &pr);
    printf("  %u arguments:", pr.numGraphArgs);
    for (uint32_t i = 0; i < pr.numGraphArgs; ++i) {
      ze_graph_argument_properties_t ap{};
      ap.stype = ZE_STYPE_GRAPH_ARGUMENT_PROPERTIES;
      g->pfnGetArgumentProperties(gr, i, &ap);
      printf(" [%s %s %ux%u prec %u/%u]", ap.type == 0 ? "in" : "out", ap.name, ap.dims[0], ap.dims[1], ap.networkPrecision,
             ap.devicePrecision);
    }
    printf("\n");
    // host memory the NPU reads and writes (the laptop's RAM is shared)
    ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
    void *A = nullptr, *B = nullptr, *C = nullptr;
    const size_t sa = (size_t)c.M * c.K * 2, sb = (size_t)c.N * c.K * 2, sc = (size_t)c.M * c.N * 2;
    if (zeMemAllocHost(ctx, &hd, sa, 4096, &A) || zeMemAllocHost(ctx, &hd, sb, 4096, &B) || zeMemAllocHost(ctx, &hd, sc, 4096, &C)) {
      fprintf(stderr, "zeMemAllocHost failed\n");
      return 1;
    }
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> af((size_t)c.M * c.K), bf((size_t)c.N * c.K);
    for (size_t i = 0; i < af.size(); ++i) { ((uint16_t*)A)[i] = f2h(u(rng)); af[i] = h2f(((uint16_t*)A)[i]); }
    for (size_t i = 0; i < bf.size(); ++i) { ((uint16_t*)B)[i] = f2h(u(rng) * 0.1f); bf[i] = h2f(((uint16_t*)B)[i]); }
    g->pfnSetArgumentValue(gr, 0, A);
    g->pfnSetArgumentValue(gr, 1, B);
    g->pfnSetArgumentValue(gr, 2, C);
    ze_result_t r = g->pfnAppendGraphInitialize(cl, gr, nullptr, 0, nullptr);
    if (r == 0) r = zeCommandListHostSynchronize(cl, UINT64_MAX);
    if (r != 0) { fprintf(stderr, "graph initialize failed: 0x%x\n", r); return 1; }
    double best = 1e30;
    std::vector<double> times;
    for (int rep = 0; rep < c.reps; ++rep) {
      const auto s0 = std::chrono::steady_clock::now();
      r = g->pfnAppendGraphExecute(cl, gr, nullptr, nullptr, 0, nullptr);
      if (r == 0) r = zeCommandListHostSynchronize(cl, UINT64_MAX);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
      if (r != 0) { fprintf(stderr, "graph execute failed: 0x%x\n", r); return 1; }
      best = std::min(best, ms);
      times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    const double med = times[times.size() / 2];
    // check a sample of outputs against the CPU (f32 sums of the f16 inputs)
    double maxrel = 0;
    const int rows = std::min(c.M, 16);
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < c.N; j += std::max(1, c.N / 64)) {
        double ref = 0;
        for (int k = 0; k < c.K; ++k) ref += (double)af[(size_t)i * c.K + k] * bf[(size_t)j * c.K + k];
        const double got = h2f(((uint16_t*)C)[(size_t)i * c.N + j]);
        maxrel = std::max(maxrel, std::fabs(got - ref) / (std::fabs(ref) + 0.05));
      }
    const double tops = 2.0 * c.M * c.K * c.N / (best * 1e-3) / 1e12;
    printf("  median %.3f ms, best of %d: %.3f ms = %.2f T(FL)OPS; max rel. error vs CPU %.2e\n", med, c.reps, best, tops, maxrel);
    zeMemFree(ctx, A);
    zeMemFree(ctx, B);
    zeMemFree(ctx, C);
    g->pfnDestroy(gr);
  }
  return 0;
}

// arcinferzero NPU experiment, accuracy study (CPU only): how far an NPU-friendly int8 matrix product (one scale per
// activation row, one per weight row, as a FakeQuantize / dequantize graph can express) lands from f32, against the
// engine's int8 scheme (one scale per 32 values for both activations and the W8 weights), on real weights of a GGUF
// layer and realistic inputs: normalized rows times the layer's own norm weights (their large channels are the
// outliers int8 meets in prompts) for the up / gate projections, and silu(gate) * up of those for the down projection.
// usage: npu_quant_study model.gguf [layer = 11] [tokens = 128]
#include "../../src/gguf.h"
#include "../../src/quants.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

// rows x K matrix of a GGUF tensor [K, rows] in f32
static std::vector<float> load_matrix(const GgufFile& g, const std::string& name, int& rows, int& K) {
  const GgufTensor& t = g.get(name);
  K = (int)t.dims[0];
  rows = (int)t.dims[1];
  std::vector<float> w((size_t)rows * K);
  const size_t rb = t.nbytes / rows;
#pragma omp parallel for
  for (int r = 0; r < rows; ++r) dequant_row(t.type, t.data + (size_t)r * rb, K, &w[(size_t)r * K]);
  return w;
}
static std::vector<float> load_vec(const GgufFile& g, const std::string& name) {
  const GgufTensor& t = g.get(name);
  std::vector<float> v(t.nelements());
  dequant_row(t.type, t.data, (int)v.size(), v.data());
  return v;
}
// y[M][N] = x[M][K] . w[N][K]^T in double
static std::vector<double> gemm_ref(const std::vector<float>& x, const std::vector<float>& w, int M, int K, int N) {
  std::vector<double> y((size_t)M * N);
#pragma omp parallel for collapse(2)
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double s = 0;
      const float* a = &x[(size_t)m * K];
      const float* b = &w[(size_t)n * K];
      for (int k = 0; k < K; ++k) s += (double)a[k] * b[k];
      y[(size_t)m * N + n] = s;
    }
  return y;
}
// symmetric int8 with one scale per `group` values along K (group = K: one per row)
struct Q8 { std::vector<int8_t> q; std::vector<float> s; int group; };
static Q8 quant(const std::vector<float>& x, int rows, int K, int group) {
  Q8 r{std::vector<int8_t>((size_t)rows * K), std::vector<float>((size_t)rows * (K / group)), group};
#pragma omp parallel for
  for (int i = 0; i < rows; ++i)
    for (int g0 = 0; g0 < K; g0 += group) {
      float amax = 0;
      for (int k = g0; k < g0 + group; ++k) amax = std::max(amax, std::fabs(x[(size_t)i * K + k]));
      const float sc = amax / 127.0f, inv = amax > 0 ? 127.0f / amax : 0.0f;
      r.s[(size_t)i * (K / group) + g0 / group] = sc;
      for (int k = g0; k < g0 + group; ++k) r.q[(size_t)i * K + k] = (int8_t)std::lrint(x[(size_t)i * K + k] * inv);
    }
  return r;
}
// int8 product with exact integer sums per group (the group sizes of a and b must be equal, or b's a multiple: then
// a's group size must divide it; here the groups are either both 32 or both the whole row)
static std::vector<double> gemm_q(const Q8& a, const Q8& b, int M, int K, int N) {
  std::vector<double> y((size_t)M * N);
  const int G = std::min(a.group, b.group);
#pragma omp parallel for collapse(2)
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double s = 0;
      for (int g0 = 0; g0 < K; g0 += G) {
        int acc = 0;
        for (int k = g0; k < g0 + G; ++k) acc += (int)a.q[(size_t)m * K + k] * b.q[(size_t)n * K + k];
        s += (double)acc * a.s[(size_t)m * (K / a.group) + g0 / a.group] * b.s[(size_t)n * (K / b.group) + g0 / b.group];
      }
      y[(size_t)m * N + n] = s;
    }
  return y;
}
static void report(const char* what, const std::vector<double>& ref, const std::vector<double>& y, int M, int N) {
  double e2 = 0, r2 = 0, worst_row = 0;
  for (int m = 0; m < M; ++m) {
    double re = 0, rr = 0;
    for (int n = 0; n < N; ++n) {
      const double d = y[(size_t)m * N + n] - ref[(size_t)m * N + n];
      re += d * d;
      rr += ref[(size_t)m * N + n] * ref[(size_t)m * N + n];
    }
    e2 += re;
    r2 += rr;
    worst_row = std::max(worst_row, std::sqrt(re / std::max(rr, 1e-30)));
  }
  printf("  %-44s relative error %.4f %%, worst token row %.4f %%\n", what, 100 * std::sqrt(e2 / r2), 100 * worst_row);
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: npu_quant_study model.gguf [layer] [tokens]\n"); return 1; }
  const int L = argc > 2 ? atoi(argv[2]) : 11, M = argc > 3 ? atoi(argv[3]) : 128;
  GgufFile g(argv[1]);
  const std::string p = "blk." + std::to_string(L) + ".";
  int N = 0, K = 0, Nd = 0, Kd = 0, Ng = 0, Kg = 0;
  const std::vector<float> wu = load_matrix(g, p + "ffn_up.weight", N, K);
  const std::vector<float> wg = load_matrix(g, p + "ffn_gate.weight", Ng, Kg);
  const std::vector<float> wd = load_matrix(g, p + "ffn_down.weight", Nd, Kd);
  std::vector<float> nw;
  for (const char* nm : {"post_attention_norm.weight", "ffn_norm.weight"})
    if (g.find(p + nm)) { nw = load_vec(g, p + nm); break; }
  if (nw.size() != (size_t)K) { fprintf(stderr, "no FFN norm weights of size %d\n", K); return 1; }
  float nmax = 0, nmean = 0;
  for (float v : nw) { nmax = std::max(nmax, std::fabs(v)); nmean += std::fabs(v) / K; }
  printf("layer %d: up / gate %d x %d, down %d x %d, %d tokens; norm weights mean |w| %.3f, max %.3f\n", L, N, K, Nd, Kd, M, nmean, nmax);
  // the FFN input: unit-RMS rows (Student-t-ish: a few large values per row) times the norm weights
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
  const Q8 x32 = quant(x, M, K, 32), xrow = quant(x, M, K, K);
  printf("ffn_up (input: normalized rows x norm weights)\n");
  const std::vector<double> yu = gemm_ref(x, wu, M, K, N);
  report("engine: activations and W8 per 32", yu, gemm_q(x32, quant(wu, N, K, 32), M, K, N), M, N);
  report("NPU: activations per row, weights per row", yu, gemm_q(xrow, quant(wu, N, K, K), M, K, N), M, N);
  report("activations per row, W8 per 32", yu, gemm_q(xrow, quant(wu, N, K, 32), M, K, N), M, N);
  report("activations per 32, weights per row", yu, gemm_q(x32, quant(wu, N, K, K), M, K, N), M, N);
  for (int G : {1280, 640, 256}) {
    const std::string w = "NPU: K in groups of " + std::to_string(G) + " (one product each)";
    report(w.c_str(), yu, gemm_q(quant(x, M, K, G), quant(wu, N, K, G), M, K, N), M, N);
  }
  // the down projection's input: silu(gate) * up, from the f32 products
  const std::vector<double> yg = gemm_ref(x, wg, M, K, Ng);
  std::vector<float> h((size_t)M * Kd);
  for (size_t i = 0; i < h.size(); ++i) {
    const double gv = yg[i];
    h[i] = (float)(gv / (1.0 + std::exp(-gv)) * yu[i]);
  }
  float hmax = 0;
  double hrms = 0;
  for (float v : h) { hmax = std::max(hmax, std::fabs(v)); hrms += (double)v * v; }
  printf("ffn_down (input: silu(gate) * up; rms %.4f, max |h| %.3f)\n", std::sqrt(hrms / h.size()), hmax);
  const std::vector<double> yd = gemm_ref(h, wd, M, Kd, Nd);
  report("engine: activations and W8 per 32", yd, gemm_q(quant(h, M, Kd, 32), quant(wd, Nd, Kd, 32), M, Kd, Nd), M, Nd);
  report("NPU: activations per row, weights per row", yd, gemm_q(quant(h, M, Kd, Kd), quant(wd, Nd, Kd, Kd), M, Kd, Nd), M, Nd);
  report("activations per row, W8 per 32", yd, gemm_q(quant(h, M, Kd, Kd), quant(wd, Nd, Kd, 32), M, Kd, Nd), M, Nd);
  report("activations per 32, weights per row", yd, gemm_q(quant(h, M, Kd, 32), quant(wd, Nd, Kd, Kd), M, Kd, Nd), M, Nd);
  for (int G : {4352, 2176, 1088, 256}) {
    const std::string w = "NPU: K in groups of " + std::to_string(G) + " (one product each)";
    report(w.c_str(), yd, gemm_q(quant(h, M, Kd, G), quant(wd, Nd, Kd, G), M, Kd, Nd), M, Nd);
  }
  return 0;
}

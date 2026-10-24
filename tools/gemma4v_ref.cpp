// f32 CPU reference of the Gemma 4 vision encoder (mmproj gemma4v: E2B / E4B, 26B-A4B), the C++ (AVX2 + OpenMP)
// version of tools/gemma4v_ref.py, written from the mmproj and llama.cpp's tools/mtmd/models/gemma4v.cpp: patches in
// [-1, 1], 2D positions (table per axis), per layer RMS norm, clamped q / k / v projections, q / k RMS norms, V RMS
// normed without weight, NEOX RoPE (theta 100: columns on the first half of a head, rows on the second), unscaled
// softmax attention, clamped output projection, post norm; a GELU-tanh gated FFN the same way; then 3x3 average
// pooling, times sqrt(E), std bias / scale (26B), RMS norm and the projection. Compares with the GPU encoder's output
// that bin/vision_testz.exe writes (<image>.g4v.bin, and the resized pixels <image>.g4v.rgb).
// Lower-precision simulations (environment): REF_VW=q8 weights as Q8_0 (int8 per 32 values); REF_VA=1 / 2 activations
// int8 per 32 values on one / two passes (the value, then its rounding residual); REF_VATT=f16 / i8 attention q / K / V
// in f16, or int8 per row with P in 8 bits, or (qk8) q / K int8 per row with V f16 and P float; REF_VOUT=file writes the embeddings as vision_testz does (header n_tok,
// n_out, 0; f32), for vision_testz ARC_VIS_EMB (the model's KL on them).
// usage: gemma4v_ref mmproj.gguf image.png   (after: vision_testz model.gguf mmproj.gguf image.png)
#include "../src/gguf.h"
#include "../src/quants.h"
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Vec = std::vector<float>;
static const GgufFile* G = nullptr;
using Clk = std::chrono::steady_clock;
static double t_gemm = 0, t_att = 0;  // (seconds in the matrix products and in the attention)
struct Timer {
  double& t;
  Clk::time_point a = Clk::now();
  ~Timer() { t += std::chrono::duration<double>(Clk::now() - a).count(); }
};

static Vec tensor(const std::string& name) {  // (f32, converted in parallel pieces of 4096 values)
  const GgufTensor& t = G->get(name);
  Vec v(t.nelements());
  const size_t piece = 4096, n = v.size();
  if (n % piece || (t.type != GGML_F16 && t.type != GGML_BF16 && t.type != GGML_F32)) {
    dequant_row(t.type, t.data, (int)n, v.data());
    return v;
  }
  const size_t eb = t.type == GGML_F32 ? 4 : 2;
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < (long long)(n / piece); ++i) dequant_row(t.type, t.data + i * piece * eb, (int)piece, v.data() + i * piece);
  return v;
}
static bool has(const std::string& name) { return G->find(name) != nullptr; }
static float scalar(const std::string& name) { return tensor(name)[0]; }

// ---- simulations of the GPU's number formats
static int env_va() { return getenv("REF_VA") ? atoi(getenv("REF_VA")) : 0; }
static std::string env_s(const char* k) { return getenv(k) ? getenv(k) : ""; }
// int8 per 32 values of each row of K (absmax / 127; a last partial block as if padded with zeros, as the GPU pads
// K to whole superblocks), `passes` times on the remaining residual (the sum kept)
static void q8_rows(float* a, size_t n, int K, int passes) {
  const int nb = (K + 31) / 32;
#pragma omp parallel for schedule(static)
  for (long long b = 0; b < (long long)(n / K) * nb; ++b) {
    const size_t row = b / nb;
    const int c0 = (int)(b % nb) * 32, len = std::min(32, K - c0);
    float* x = a + row * K + c0;
    float r[32] = {}, out[32] = {};
    memcpy(r, x, len * 4);
    for (int p = 0; p < passes; ++p) {
      float mx = 0;
      for (float v : r) mx = std::max(mx, std::fabs(v));
      const float d = mx / 127.0f;
      for (int i = 0; i < 32; ++i) {
        const float q = d > 0 ? std::nearbyint(r[i] / d) * d : 0.0f;
        out[i] += q;
        r[i] -= q;
      }
    }
    memcpy(x, out, len * 4);
  }
}
static float f16_round(float x) {  // (nearest even at 10 mantissa bits)
  uint32_t u;
  memcpy(&u, &x, 4);
  u = (u + 0xFFF + ((u >> 13) & 1)) & ~0x1FFFu;
  memcpy(&x, &u, 4);
  return x;
}

// ---- C[M][N] = A[M][K] W[N][K]^T: tiles of 64 rows x 64 weight rows (both in L2), 4 x 2 microkernel on AVX2
static inline float hsum(__m256 v) {
  const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  const __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
  return _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 1)));
}
static void gemm(const float* A, int M, int K, const float* W, int N, float* C) {
  Timer tm{t_gemm};
  const int TM = 64, TN = 64, K8 = K / 8 * 8;
  const int nt = (M + TM - 1) / TM * ((N + TN - 1) / TN);
#pragma omp parallel for schedule(dynamic, 1)
  for (int t = 0; t < nt; ++t) {
    const int m0 = t / ((N + TN - 1) / TN) * TM, n0 = t % ((N + TN - 1) / TN) * TN;
    const int m1 = std::min(M, m0 + TM), n1 = std::min(N, n0 + TN);
    for (int m = m0; m < m1; m += 4) {
      const int mr = std::min(4, m1 - m);
      for (int n = n0; n < n1; n += 2) {
        const int nr = std::min(2, n1 - n);
        __m256 acc[4][2];
        for (auto& r : acc) r[0] = r[1] = _mm256_setzero_ps();
        const float* a[4];
        const float* w[2];
        for (int i = 0; i < 4; ++i) a[i] = A + (size_t)(m + std::min(i, mr - 1)) * K;
        for (int j = 0; j < 2; ++j) w[j] = W + (size_t)(n + std::min(j, nr - 1)) * K;
        for (int k = 0; k < K8; k += 8) {
          const __m256 w0 = _mm256_loadu_ps(w[0] + k), w1 = _mm256_loadu_ps(w[1] + k);
          for (int i = 0; i < 4; ++i) {
            const __m256 av = _mm256_loadu_ps(a[i] + k);
            acc[i][0] = _mm256_fmadd_ps(av, w0, acc[i][0]);
            acc[i][1] = _mm256_fmadd_ps(av, w1, acc[i][1]);
          }
        }
        for (int i = 0; i < mr; ++i)
          for (int j = 0; j < nr; ++j) {
            float s = hsum(acc[i][j]);
            for (int k = K8; k < K; ++k) s += a[i][k] * w[j][k];
            C[(size_t)(m + i) * N + n + j] = s;
          }
      }
    }
  }
}

struct Lin {
  Vec w;
  int in = 0, out = 0;
  bool clamp_in = false, clamp_out = false;
  float in_lo = 0, in_hi = 0, out_lo = 0, out_hi = 0;
};
static Lin load_lin(const std::string& name) {
  Lin l;
  const GgufTensor& t = G->get(name + ".weight");
  l.in = (int)t.dims[0];
  l.out = (int)(t.nelements() / l.in);
  l.w = tensor(name + ".weight");
  if (env_s("REF_VW") == "q8") q8_rows(l.w.data(), l.w.size(), l.in, 1);
  if (has(name + ".input_min")) { l.clamp_in = true; l.in_lo = scalar(name + ".input_min"); l.in_hi = scalar(name + ".input_max"); }
  if (has(name + ".output_min")) { l.clamp_out = true; l.out_lo = scalar(name + ".output_min"); l.out_hi = scalar(name + ".output_max"); }
  return l;
}
// y [M][out] = clamp(clamp(a) W^T), a [M][in] (copied: clamping and the activation simulation do not touch the input)
static Vec apply(const Lin& l, const Vec& a, int M) {
  Vec x = a;
  if (l.clamp_in)
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)x.size(); ++i) x[i] = std::clamp(x[i], l.in_lo, l.in_hi);
  if (const int va = env_va(); va > 0) q8_rows(x.data(), x.size(), l.in, va);
  Vec y((size_t)M * l.out);
  gemm(x.data(), M, l.in, l.w.data(), l.out, y.data());
  if (l.clamp_out)
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)y.size(); ++i) y[i] = std::clamp(y[i], l.out_lo, l.out_hi);
  return y;
}
// RMS norm of rows of n values (weight w, or none)
static void rms(float* x, size_t rows, int n, const float* w, float eps) {
#pragma omp parallel for schedule(static)
  for (long long r = 0; r < (long long)rows; ++r) {
    float* v = x + r * n;
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double)v[i] * v[i];
    const float s = (float)(1.0 / std::sqrt(ss / n + eps));
    for (int i = 0; i < n; ++i) v[i] = v[i] * s * (w ? w[i] : 1.0f);
  }
}
static float gelu_tanh(float x) { return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x))); }

// unscaled softmax attention of one head: q / k / v [N][hd] -> o [N][hd], QB query rows at a time, as two small
// matrix products with 4 x 3 register tiles (12 accumulators: per step 3 loads and 4 broadcasts for 12 FMAs). Scores
// S = Q K^T with K transposed to [hd][N] (tiles of 4 rows x 24 keys); softmax per row (SVML exp, a sum in double);
// P.V by blocks of KB keys whose V rows stay in L1 (tiles of 4 rows x 24 dims).
static void attention(const float* q, const float* k, const float* v, int N, int hd, float* o, bool p8) {
  Timer tm{t_att};
  const int NP = (N + 23) / 24 * 24;  // (keys padded to whole tiles: zero keys, their scores masked)
  Vec kt((size_t)hd * NP, 0.0f);
  for (int j = 0; j < N; ++j)
    for (int d = 0; d < hd; ++d) kt[(size_t)d * NP + j] = k[(size_t)j * hd + d];
  const int QB = 8, KB = 128, ND = hd / 24;  // (hd 72: 3 tiles of 24 dims; 64 is not used by the 26B: see below)
  const bool tiles = hd % 24 == 0;
#pragma omp parallel
  {
    Vec s((size_t)QB * NP), acc((size_t)QB * hd);
    std::vector<double> z(QB);
#pragma omp for schedule(dynamic, 2)
    for (int i0 = 0; i0 < N; i0 += QB) {
      const int qn = std::min(QB, N - i0);
      const float* qr[QB];
      for (int r = 0; r < QB; ++r) qr[r] = q + (size_t)(i0 + std::min(r, qn - 1)) * hd;
      for (int g = 0; g < QB; g += 4)
        for (int j = 0; j < NP; j += 24) {
          __m256 c[4][3];
          for (auto& row : c) row[0] = row[1] = row[2] = _mm256_setzero_ps();
          for (int d = 0; d < hd; ++d) {
            const float* kd = &kt[(size_t)d * NP + j];
            const __m256 k0 = _mm256_loadu_ps(kd), k1 = _mm256_loadu_ps(kd + 8), k2 = _mm256_loadu_ps(kd + 16);
            for (int r = 0; r < 4; ++r) {
              const __m256 qv = _mm256_broadcast_ss(qr[g + r] + d);
              c[r][0] = _mm256_fmadd_ps(qv, k0, c[r][0]);
              c[r][1] = _mm256_fmadd_ps(qv, k1, c[r][1]);
              c[r][2] = _mm256_fmadd_ps(qv, k2, c[r][2]);
            }
          }
          for (int r = 0; r < 4; ++r)
            for (int t = 0; t < 3; ++t) _mm256_storeu_ps(&s[(size_t)(g + r) * NP + j + 8 * t], c[r][t]);
        }
      for (int r = 0; r < QB; ++r) {
        float* sr = &s[(size_t)r * NP];
        float mx = sr[0];
        for (int j = 1; j < N; ++j) mx = std::max(mx, sr[j]);
        const __m256 mv = _mm256_set1_ps(mx);
        __m256d zd = _mm256_setzero_pd();
        int j = 0;
        for (; j + 8 <= N; j += 8) {
          __m256 p = _mm256_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(sr + j), mv));
          if (p8) p = _mm256_div_ps(_mm256_round_ps(_mm256_mul_ps(p, _mm256_set1_ps(255.0f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm256_set1_ps(255.0f));
          _mm256_storeu_ps(sr + j, p);
          zd = _mm256_add_pd(zd, _mm256_add_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(p)), _mm256_cvtps_pd(_mm256_extractf128_ps(p, 1))));
        }
        double zz = 0;
        double zl[4];
        _mm256_storeu_pd(zl, zd);
        zz = zl[0] + zl[1] + zl[2] + zl[3];
        for (; j < N; ++j) {
          float p = std::exp(sr[j] - mx);
          if (p8) p = std::nearbyint(p * 255.0f) / 255.0f;
          sr[j] = p;
          zz += p;
        }
        for (j = N; j < NP; ++j) sr[j] = 0.0f;  // (the padding keys)
        z[r] = zz;
      }
      std::fill(acc.begin(), acc.end(), 0.0f);
      for (int j0 = 0; j0 < N; j0 += KB) {
        const int j1 = std::min(N, j0 + KB);
        for (int g = 0; g < QB; g += 4) {
          if (!tiles) {  // (other head sizes: a row at a time)
            for (int r = g; r < g + 4; ++r)
              for (int jj = j0; jj < j1; ++jj)
                for (int d = 0; d < hd; ++d) acc[(size_t)r * hd + d] += s[(size_t)r * NP + jj] * v[(size_t)jj * hd + d];
            continue;
          }
          for (int dt = 0; dt < ND; ++dt) {
            __m256 c[4][3];
            for (int r = 0; r < 4; ++r)
              for (int t = 0; t < 3; ++t) c[r][t] = _mm256_loadu_ps(&acc[(size_t)(g + r) * hd + 24 * dt + 8 * t]);
            for (int jj = j0; jj < j1; ++jj) {
              const float* vr = v + (size_t)jj * hd + 24 * dt;
              const __m256 v0 = _mm256_loadu_ps(vr), v1 = _mm256_loadu_ps(vr + 8), v2 = _mm256_loadu_ps(vr + 16);
              for (int r = 0; r < 4; ++r) {
                const __m256 pv = _mm256_broadcast_ss(&s[(size_t)(g + r) * NP + jj]);
                c[r][0] = _mm256_fmadd_ps(pv, v0, c[r][0]);
                c[r][1] = _mm256_fmadd_ps(pv, v1, c[r][1]);
                c[r][2] = _mm256_fmadd_ps(pv, v2, c[r][2]);
              }
            }
            for (int r = 0; r < 4; ++r)
              for (int t = 0; t < 3; ++t) _mm256_storeu_ps(&acc[(size_t)(g + r) * hd + 24 * dt + 8 * t], c[r][t]);
          }
        }
      }
      for (int r = 0; r < qn; ++r) {
        const float inv = (float)(1.0 / z[r]);
        for (int d = 0; d < hd; ++d) o[(size_t)(i0 + r) * hd + d] = acc[(size_t)r * hd + d] * inv;
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: gemma4v_ref mmproj.gguf image.png\n"); return 1; }
  const auto t0 = std::chrono::steady_clock::now();
  GgufFile f(argv[1]);
  G = &f;
  const std::string img = argv[2];
  std::vector<uint8_t> raw;
  {
    FILE* fr = fopen((img + ".g4v.rgb").c_str(), "rb");
    if (!fr) { fprintf(stderr, "no %s.g4v.rgb: run vision_testz first\n", img.c_str()); return 1; }
    fseek(fr, 0, SEEK_END);
    raw.resize(ftell(fr));
    fseek(fr, 0, SEEK_SET);
    if (fread(raw.data(), 1, raw.size(), fr) != raw.size()) return 1;
    fclose(fr);
  }
  int W, H;
  memcpy(&W, raw.data(), 4);
  memcpy(&H, raw.data() + 4, 4);
  const uint8_t* px = raw.data() + 8;
  const int E = (int)f.get_int("clip.vision.embedding_length", 0), nh = (int)f.get_int("clip.vision.attention.head_count", 0);
  const int P = (int)f.get_int("clip.vision.patch_size", 0), L = (int)f.get_int("clip.vision.block_count", 0);
  const float eps = (float)f.get_float("clip.vision.attention.layer_norm_epsilon", 1e-6);
  const int pool = 3, nx = W / P, ny = H / P, N = nx * ny, hd = E / nh, K0 = 3 * P * P;
  const std::string vatt = env_s("REF_VATT");
  // patches (row-major), each (channel, row, column), pixels mapped to [-1, 1]; patch embedding, 2D positions
  Vec pt((size_t)N * K0);
  for (int n = 0; n < N; ++n) {
    const int py = n / nx, pxx = n % nx;
    for (int c = 0; c < 3; ++c)
      for (int r = 0; r < P; ++r)
        for (int cc = 0; cc < P; ++cc)
          pt[(size_t)n * K0 + (c * P + r) * P + cc] = 2.0f * px[((size_t)(py * P + r) * W + pxx * P + cc) * 3 + c] / 255.0f - 1.0f;
  }
  Vec x((size_t)N * E);
  {
    const Vec wpe = tensor("v.patch_embd.weight");
    gemm(pt.data(), N, K0, wpe.data(), E, x.data());
    const GgufTensor& tt = f.get("v.position_embd.weight");
    const int psz = (int)tt.dims[1];
    const Vec tab = tensor("v.position_embd.weight");  // [2][pos_size][E]
    for (int n = 0; n < N; ++n)
      for (int e = 0; e < E; ++e)
        x[(size_t)n * E + e] += tab[((size_t)0 * psz + n % nx) * E + e] + tab[((size_t)1 * psz + n / nx) * E + e];
  }
  // RoPE tables: dims i < hd/2 rotate by the column, the rest by the row; NEOX pairs (i, i + hd/4) in each half
  const int h2 = hd / 2, h4 = hd / 4;
  std::vector<double> inv(h4);
  for (int i = 0; i < h4; ++i) inv[i] = std::pow(100.0, -2.0 * i / h2);
  auto rope = [&](float* r, int col, int row) {
    for (int half = 0; half < 2; ++half) {
      float* v = r + half * h2;
      const int pos = half ? row : col;
      for (int i = 0; i < h4; ++i) {
        const double a = pos * inv[i];
        const float c = (float)std::cos(a), s = (float)std::sin(a), x0 = v[i], x1 = v[i + h4];
        v[i] = x0 * c - x1 * s;
        v[i + h4] = x0 * s + x1 * c;
      }
    }
  };
  for (int il = 0; il < L; ++il) {
    const std::string b = "v.blk." + std::to_string(il) + ".";
    Vec h = x;
    rms(h.data(), N, E, tensor(b + "ln1.weight").data(), eps);
    Vec q = apply(load_lin(b + "attn_q"), h, N), k = apply(load_lin(b + "attn_k"), h, N), v = apply(load_lin(b + "attn_v"), h, N);
    rms(q.data(), (size_t)N * nh, hd, tensor(b + "attn_q_norm.weight").data(), eps);
    rms(k.data(), (size_t)N * nh, hd, tensor(b + "attn_k_norm.weight").data(), eps);
    rms(v.data(), (size_t)N * nh, hd, nullptr, eps);
#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n)
      for (int hh = 0; hh < nh; ++hh) {
        rope(&q[((size_t)n * nh + hh) * hd], n % nx, n / nx);
        rope(&k[((size_t)n * nh + hh) * hd], n % nx, n / nx);
      }
    // per head [N][hd] (the simulations of the attention's storage applied here)
    Vec o((size_t)N * E);
    for (int hh = 0; hh < nh; ++hh) {
      Vec qh((size_t)N * hd), kh((size_t)N * hd), vh((size_t)N * hd), oh((size_t)N * hd);
#pragma omp parallel for schedule(static)
      for (int n = 0; n < N; ++n)
        for (int d = 0; d < hd; ++d) {
          const size_t src = ((size_t)n * nh + hh) * hd + d, dst = (size_t)n * hd + d;
          qh[dst] = q[src];
          kh[dst] = k[src];
          vh[dst] = v[src];
        }
      if (vatt == "f16") {
        for (Vec* t : {&qh, &kh, &vh})
          for (float& e : *t) e = f16_round(e);
      } else if (vatt == "i8" || vatt == "qk8") {  // (qk8: q / K int8 per row, V f16 as the GPU keeps it, P float)
        for (Vec* t : {&qh, &kh, &vh}) {
          if (t == &vh && vatt == "qk8") {
            for (float& e : vh) e = f16_round(e);
            continue;
          }
          for (int n = 0; n < N; ++n) {
            float* r = t->data() + (size_t)n * hd, mx = 0;
            for (int d = 0; d < hd; ++d) mx = std::max(mx, std::fabs(r[d]));
            const float dd = mx / 127.0f;
            for (int d = 0; d < hd; ++d) r[d] = dd > 0 ? std::nearbyint(r[d] / dd) * dd : 0.0f;
          }
        }
      }
      attention(qh.data(), kh.data(), vh.data(), N, hd, oh.data(), vatt == "i8");
#pragma omp parallel for schedule(static)
      for (int n = 0; n < N; ++n) memcpy(&o[(size_t)n * E + hh * hd], &oh[(size_t)n * hd], hd * 4);
    }
    Vec t = apply(load_lin(b + "attn_out"), o, N);
    rms(t.data(), N, E, tensor(b + "attn_post_norm.weight").data(), eps);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)x.size(); ++i) x[i] += t[i];
    h = x;
    rms(h.data(), N, E, tensor(b + "ln2.weight").data(), eps);
    Vec g = apply(load_lin(b + "ffn_gate"), h, N);
    {
      const Vec u = apply(load_lin(b + "ffn_up"), h, N);
#pragma omp parallel for schedule(static)
      for (long long i = 0; i < (long long)g.size(); ++i) g[i] = gelu_tanh(g[i]) * u[i];
    }
    Vec d = apply(load_lin(b + "ffn_down"), g, N);
    rms(d.data(), N, E, tensor(b + "ffn_post_norm.weight").data(), eps);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)x.size(); ++i) x[i] += d[i];
    fprintf(stderr, "\rlayer %d / %d", il + 1, L);
  }
  // 3x3 average pooling, times sqrt(E), std bias / scale, RMS norm, projection
  const int px3 = nx / pool, py3 = ny / pool, M = px3 * py3;
  Vec gp((size_t)M * E, 0.0f);
  for (int ty = 0; ty < py3; ++ty)
    for (int tx = 0; tx < px3; ++tx)
      for (int dy = 0; dy < pool; ++dy)
        for (int dx = 0; dx < pool; ++dx) {
          const float* src = &x[((size_t)(ty * pool + dy) * nx + tx * pool + dx) * E];
          float* dst = &gp[((size_t)ty * px3 + tx) * E];
          for (int e = 0; e < E; ++e) dst[e] += src[e] / (pool * pool);
        }
  const float se = std::sqrt((float)E);
  for (float& e : gp) e *= se;
  if (has("v.std_bias")) {
    const Vec sb = tensor("v.std_bias"), ss = tensor("v.std_scale");
    for (int m = 0; m < M; ++m)
      for (int e = 0; e < E; ++e) gp[(size_t)m * E + e] = (gp[(size_t)m * E + e] - sb[e]) * ss[e];
  }
  rms(gp.data(), M, E, nullptr, eps);
  const Vec proj = tensor("mm.input_projection.weight");
  const int n_out = (int)(proj.size() / E);
  Vec ref((size_t)M * n_out);
  gemm(gp.data(), M, E, proj.data(), n_out, ref.data());
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  fprintf(stderr, "\rf32 reference in %.1f s (%d threads; matrix products %.1f s, attention %.1f s)\n", secs,
          omp_get_max_threads(), t_gemm, t_att);
  if (const char* vo = getenv("REF_VOUT")) {
    FILE* fo = fopen(vo, "wb");
    const int hdr[3] = {M, n_out, 0};
    fwrite(hdr, 4, 3, fo);
    fwrite(ref.data(), 4, ref.size(), fo);
    fclose(fo);
  }
  // against the GPU encoder's output
  FILE* fg = fopen((img + ".g4v.bin").c_str(), "rb");
  int hg[3] = {0, 0, 0};
  if (!fg || fread(hg, 4, 3, fg) != 3 || hg[0] != M || hg[1] != n_out) {
    fprintf(stderr, "no matching %s.g4v.bin to compare with\n", img.c_str());
    return 0;
  }
  Vec gpu(ref.size());
  if (fread(gpu.data(), 4, gpu.size(), fg) != gpu.size()) return 1;
  fclose(fg);
  double cmin = 1, csum = 0, rsum = 0, rmax = 0;
  for (int m = 0; m < M; ++m) {
    double dot = 0, na = 0, nb = 0, dd = 0;
    for (int e = 0; e < n_out; ++e) {
      const double a = ref[(size_t)m * n_out + e], g = gpu[(size_t)m * n_out + e];
      dot += a * g;
      na += a * a;
      nb += g * g;
      dd += (a - g) * (a - g);
    }
    const double c = dot / std::sqrt(na * nb), r = std::sqrt(dd / na);
    cmin = std::min(cmin, c);
    csum += c;
    rsum += r;
    rmax = std::max(rmax, r);
  }
  printf("%d tokens: cosine min %.6f mean %.6f; relative error mean %.3f %% max %.3f %%\n", M, cmin, csum / M,
         rsum / M * 100, rmax * 100);
  return 0;
}

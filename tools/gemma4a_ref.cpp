// f32 CPU reference of the Gemma 4 audio encoder of the small models (mmproj gemma4a: E2B / E4B), written from the
// mmproj and llama.cpp's tools/mtmd/models/gemma4a.cpp + mtmd-audio.cpp (mtmd_audio_preprocessor_gemma4a): 30-second
// clips; log-mel (semicausal padding, 320-sample periodic Hann window in a 512-point DFT, hop 160, magnitude, 128 HTK
// mel bins, ln(max(mel, 0.001)); in double); two 3x3 stride-2 convolutions with a LayerNorm over channels and ReLU;
// the input projection; 12 conformer blocks (half-step SiLU FFN, chunked local attention: 12 keys back including
// itself, sinusoidal relative positions through attn_k_rel, q * 1/sqrt(D)/ln 2 * per_dim_scale, k * ln(1+e)/ln 2,
// soft cap 50; a convolution module: GLU, causal depthwise conv of width 5, RMS norm, SiLU; a second half-step FFN;
// the block's RMS norm), every matrix with the mmproj's input / output clamps; the output projection with its bias,
// an RMS norm without weight and the projection into the language model. Weights are the mmproj's (BF16 / F32) as
// f32, activations f32, sums of products in f32 FMA (8 lanes), norms in double.
// Compares with the GPU encoder's rows that bin/vision_testz.exe writes (<clip>.g4a.bin) and writes its own rows with
// the same header to <clip>.g4a.ref.bin (for vision_testz ARC_AUDIO_EMB and tools/ref_forward REF_EMB).
// Lower-precision simulations (environment): REF_AW=f16 the weights rounded to f16 as the GPU stores them.
// usage: gemma4a_ref mmproj.gguf clip   (after: vision_testz model.gguf mmproj.gguf clip)
#include "../src/audio.h"
#include "../src/gguf.h"
#include "../src/quants.h"
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Vec = std::vector<float>;
static const GgufFile* G = nullptr;
static bool w16 = false;

static float f16_round(float x) { return fp16_to_f32(f32_to_fp16(x)); }
static Vec tensor(const std::string& name) {
  const GgufTensor& t = G->get(name);
  Vec v(t.nelements());
  dequant_row(t.type, t.data, (int)v.size(), v.data());
  return v;
}
static bool has(const std::string& name) { return G->find(name) != nullptr; }

static inline float hsum(__m256 v) {
  const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  const __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
  return _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 1)));
}
// C[M][N] = A[M][K] W[N][K]^T (K a multiple of 8)
static void gemm(const float* A, int M, int K, const float* W, int N, float* C) {
#pragma omp parallel for schedule(dynamic, 4)
  for (int n = 0; n < N; n += 2) {
    const int nr = std::min(2, N - n);
    for (int m = 0; m < M; ++m) {
      const float* a = A + (size_t)m * K;
      for (int j = 0; j < nr; ++j) {
        const float* w = W + (size_t)(n + j) * K;
        __m256 acc = _mm256_setzero_ps();
        for (int k = 0; k < K; k += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(w + k), acc);
        C[(size_t)m * N + n + j] = hsum(acc);
      }
    }
  }
}

struct Lin {
  Vec w;
  int in = 0, out = 0;
  float in_lo = -FLT_MAX, in_hi = FLT_MAX, out_lo = -FLT_MAX, out_hi = FLT_MAX;
};
static Lin load_lin(const std::string& name) {
  Lin l;
  const GgufTensor& t = G->get(name + ".weight");
  l.in = (int)t.dims[0];
  l.out = (int)(t.nelements() / l.in);
  l.w = tensor(name + ".weight");
  if (w16)
    for (float& v : l.w) v = f16_round(v);
  if (has(name + ".input_min")) { l.in_lo = tensor(name + ".input_min")[0]; l.in_hi = tensor(name + ".input_max")[0]; }
  if (has(name + ".output_min")) { l.out_lo = tensor(name + ".output_min")[0]; l.out_hi = tensor(name + ".output_max")[0]; }
  return l;
}
static Vec apply(const Lin& l, const Vec& a, int M) {
  Vec x = a;
  for (float& v : x) v = std::clamp(v, l.in_lo, l.in_hi);
  Vec y((size_t)M * l.out);
  gemm(x.data(), M, l.in, l.w.data(), l.out, y.data());
  for (float& v : y) v = std::clamp(v, l.out_lo, l.out_hi);
  return y;
}
static void rms(float* x, size_t rows, int n, const float* w, float eps) {
#pragma omp parallel for schedule(static)
  for (long long r = 0; r < (long long)rows; ++r) {
    float* v = x + r * n;
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double)v[i] * v[i];
    const double s = 1.0 / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; ++i) v[i] = (float)(v[i] * s * (w ? w[i] : 1.0f));
  }
}
static float silu(float v) { return (float)(v / (1.0 + std::exp(-(double)v))); }

struct Block {
  Vec ff_norm, ff_post, ff_norm1, ff_post1, attn_pre, attn_post, pds, kr, conv_pre, conv_post, dw, ln2;
  Lin up, down, up1, down1, q, k, v, o, pw1, pw2;
};

// log-mel [T][128] of one clip (double)
static Vec mel(const float* x, size_t n, int n_mel, int& T) {
  const int win = 320, fft = 512, hop = 160, nb = 257, pad_left = win / 2;
  const int64_t n_left = (int64_t)n + pad_left;
  T = std::max(1, (int)((n_left - (win + 1)) / hop + 1));
  const int64_t need = (int64_t)(T - 1) * hop + fft;
  std::vector<double> p((size_t)std::max<int64_t>(need, (int64_t)n + pad_left), 0.0);
  for (size_t i = 0; i < n; ++i) p[pad_left + i] = x[i];
  auto hz2mel = [](double f) { return 2595.0 * std::log10(1.0 + f / 700.0); };
  auto mel2hz = [](double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); };
  std::vector<double> hz(n_mel + 2), filt((size_t)n_mel * nb, 0.0);
  for (int i = 0; i < n_mel + 2; ++i) hz[i] = mel2hz(hz2mel(8000.0) * i / (n_mel + 1));
  for (int m = 0; m < n_mel; ++m)
    for (int k = 0; k < nb; ++k) {
      const double f = k * 16000.0 / fft, fl = hz[m], fc = hz[m + 1], fr = hz[m + 2];
      if (f >= fl && f <= fc) filt[(size_t)m * nb + k] = (f - fl) / (fc - fl);
      else if (f > fc && f <= fr) filt[(size_t)m * nb + k] = (fr - f) / (fr - fc);
    }
  const double pi = 3.14159265358979323846;
  std::vector<double> cs((size_t)nb * win), sn((size_t)nb * win), hann(win);
  for (int i = 0; i < win; ++i) hann[i] = 0.5 - 0.5 * std::cos(2.0 * pi * i / win);
  for (int k = 0; k < nb; ++k)
    for (int i = 0; i < win; ++i) {
      cs[(size_t)k * win + i] = std::cos(2.0 * pi * ((size_t)k * i % fft) / fft);
      sn[(size_t)k * win + i] = std::sin(2.0 * pi * ((size_t)k * i % fft) / fft);
    }
  Vec out((size_t)T * n_mel);
#pragma omp parallel for
  for (int t = 0; t < T; ++t) {
    double fr[320], mag[257];
    for (int i = 0; i < win; ++i) fr[i] = hann[i] * p[(size_t)t * hop + i];
    for (int k = 0; k < nb; ++k) {
      double re = 0, im = 0;
      for (int i = 0; i < win; ++i) {
        re += fr[i] * cs[(size_t)k * win + i];
        im -= fr[i] * sn[(size_t)k * win + i];
      }
      mag[k] = std::sqrt(re * re + im * im);
    }
    for (int m = 0; m < n_mel; ++m) {
      double s = 0;
      for (int k = 0; k < nb; ++k) s += mag[k] * filt[(size_t)m * nb + k];
      out[(size_t)t * n_mel + m] = (float)std::log(std::max(s, 0.001));
    }
  }
  return out;
}
// 3x3 stride-2 conv (padding 1) over (time, freq), LayerNorm over channels (weight), ReLU; [Ci][T][F] -> [Co][T'][F']
static Vec conv(const Vec& in, int Ci, int T, int F, const Vec& w, const Vec& nw, int Co, int& To, int& Fo) {
  To = (T - 1) / 2 + 1;
  Fo = (F - 1) / 2 + 1;
  Vec out((size_t)Co * To * Fo);
#pragma omp parallel for
  for (int t = 0; t < To; ++t)
    for (int f = 0; f < Fo; ++f) {
      std::vector<double> v(Co);
      for (int co = 0; co < Co; ++co) {
        double s = 0;
        for (int ci = 0; ci < Ci; ++ci)
          for (int kt = 0; kt < 3; ++kt)
            for (int kf = 0; kf < 3; ++kf) {
              const int ti = 2 * t + kt - 1, fi = 2 * f + kf - 1;
              if (ti >= 0 && ti < T && fi >= 0 && fi < F)
                s += (double)in[((size_t)ci * T + ti) * F + fi] * w[(((size_t)co * Ci + ci) * 3 + kt) * 3 + kf];
            }
        v[co] = s;
      }
      double mean = 0, var = 0;
      for (double x : v) mean += x;
      mean /= Co;
      for (double x : v) var += (x - mean) * (x - mean);
      const double r = 1.0 / std::sqrt(var / Co + 1e-6);
      for (int co = 0; co < Co; ++co) out[((size_t)co * To + t) * Fo + f] = (float)std::max(0.0, (v[co] - mean) * r * nw[co]);
    }
  return out;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: gemma4a_ref mmproj.gguf clip\n"); return 1; }
  const auto t0 = std::chrono::steady_clock::now();
  GgufFile f(argv[1]);
  G = &f;
  w16 = getenv("REF_AW") && std::string(getenv("REF_AW")) == "f16";
  const std::string clip = argv[2];
  if (f.get_str("clip.audio.projector_type") != "gemma4a") { fprintf(stderr, "not a gemma4a mmproj\n"); return 1; }
  const int E = (int)f.get_int("clip.audio.embedding_length", 0), L = (int)f.get_int("clip.audio.block_count", 0);
  const int nh = (int)f.get_int("clip.audio.attention.head_count", 8), n_mel = (int)f.get_int("clip.audio.num_mel_bins", 128);
  const int D = E / nh, R = 13, CTX = 12;
  const float eps = 1e-6f;
  const Vec c0w = tensor("a.conv1d.0.weight"), c0n = tensor("a.conv1d.0.norm.weight");
  const Vec c1w = tensor("a.conv1d.1.weight"), c1n = tensor("a.conv1d.1.norm.weight");
  const int C0 = (int)f.get("a.conv1d.0.weight").dims[3], C1 = (int)f.get("a.conv1d.1.weight").dims[3];
  const Lin inp = load_lin("a.input_projection"), outp = load_lin("a.pre_encode.out"), mmp = load_lin("mm.a.input_projection");
  const Vec outb = tensor("a.pre_encode.out.bias");
  // sinusoidal relative positions 12 .. 0 (rows 0 .. 12), as llama.cpp's pos_emb
  Vec pe((size_t)R * E);
  {
    const int half = E / 2;
    const float inc = std::log(10000.0f) / std::max(half - 1, 1);
    for (int r = 0; r < R; ++r)
      for (int i = 0; i < half; ++i) {
        const float a = (float)(12 - r) * std::exp(-(float)i * inc);
        pe[(size_t)r * E + i] = std::sin(a);
        pe[(size_t)r * E + i + half] = std::cos(a);
      }
  }
  std::vector<Block> B(L);
  for (int il = 0; il < L; ++il) {
    const std::string p = "a.blk." + std::to_string(il) + ".";
    Block& b = B[il];
    b.ff_norm = tensor(p + "ffn_norm.weight");
    b.ff_post = tensor(p + "ffn_post_norm.weight");
    b.ff_norm1 = tensor(p + "ffn_norm_1.weight");
    b.ff_post1 = tensor(p + "ffn_post_norm_1.weight");
    b.attn_pre = tensor(p + "attn_pre_norm.weight");
    b.attn_post = tensor(p + "attn_post_norm.weight");
    b.pds = tensor(p + "per_dim_scale.weight");
    b.conv_pre = tensor(p + "conv_norm.weight");  // (swapped names in the GGUF, as llama.cpp loads them)
    b.conv_post = tensor(p + "norm_conv.weight");
    b.dw = tensor(p + "conv_dw.weight");
    b.ln2 = tensor(p + "ln2.weight");
    b.up = load_lin(p + "ffn_up");
    b.down = load_lin(p + "ffn_down");
    b.up1 = load_lin(p + "ffn_up_1");
    b.down1 = load_lin(p + "ffn_down_1");
    b.q = load_lin(p + "attn_q");
    b.k = load_lin(p + "attn_k");
    b.v = load_lin(p + "attn_v");
    b.o = load_lin(p + "attn_out");
    b.pw1 = load_lin(p + "conv_pw1");
    b.pw2 = load_lin(p + "conv_pw2");
    const Lin kr = load_lin(p + "attn_k_rel");  // the projected positions [13][E]
    b.kr = apply(kr, pe, R);
  }
  const std::vector<float> s16 = audio::decode_16k_file(clip);
  Vec ref;
  int n_tok = 0;
  const size_t chunk = 30 * 16000;
  for (size_t off = 0; off < s16.size(); off += chunk) {
    int T, T1, F1, T2, F2;
    const Vec m = mel(s16.data() + off, std::min(chunk, s16.size() - off), n_mel, T);
    const Vec a1 = conv(m, 1, T, n_mel, c0w, c0n, C0, T1, F1);
    const Vec a2 = conv(a1, C0, T1, F1, c1w, c1n, C1, T2, F2);
    const int N = T2, K0 = C1 * F2;
    Vec in((size_t)N * K0);
    for (int t = 0; t < N; ++t)
      for (int fq = 0; fq < F2; ++fq)
        for (int c = 0; c < C1; ++c) in[(size_t)t * K0 + fq * C1 + c] = a2[((size_t)c * T2 + t) * F2 + fq];
    Vec x = apply(inp, in, N);
    auto ffn = [&](const Vec& nw, const Lin& up, const Lin& down, const Vec& pw) {
      Vec h = x;
      rms(h.data(), N, E, nw.data(), eps);
      Vec u = apply(up, h, N);
      for (float& v : u) v = silu(v);
      Vec d = apply(down, u, N);
      rms(d.data(), N, E, pw.data(), eps);
      for (size_t i = 0; i < x.size(); ++i) x[i] += 0.5f * d[i];
    };
    for (const Block& b : B) {
      ffn(b.ff_norm, b.up, b.down, b.ff_post);
      {  // attention
        Vec h = x;
        rms(h.data(), N, E, b.attn_pre.data(), eps);
        const Vec q = apply(b.q, h, N), k = apply(b.k, h, N), v = apply(b.v, h, N);
        const float qs = (1.0f / std::sqrt((float)D)) / std::log(2.0f), ks = std::log(1.0f + std::exp(1.0f)) / std::log(2.0f);
        Vec o((size_t)N * E);
#pragma omp parallel for
        for (int t = 0; t < N; ++t)
          for (int hh = 0; hh < nh; ++hh) {
            float qv[256];
            for (int d = 0; d < D; ++d) qv[d] = q[(size_t)t * E + hh * D + d] * qs * b.pds[d];
            double sc[CTX], mx = -1e300;
            int nk = 0;
            for (int w = 0; w < CTX && t - w >= 0; ++w, ++nk) {
              double s = 0;
              const float* kk = &k[(size_t)(t - w) * E + hh * D];
              const float* pr = &b.kr[(size_t)(12 - w) * E + hh * D];  // (row 12 - w: relative distance w)
              for (int d = 0; d < D; ++d) s += (double)qv[d] * (kk[d] * ks) + (double)qv[d] * pr[d];
              s = 50.0 * std::tanh(s / 50.0);
              sc[w] = s;
              mx = std::max(mx, s);
            }
            double den = 0;
            for (int w = 0; w < nk; ++w) den += sc[w] = std::exp(sc[w] - mx);
            for (int d = 0; d < D; ++d) {
              double acc = 0;
              for (int w = 0; w < nk; ++w) acc += sc[w] * v[(size_t)(t - w) * E + hh * D + d];
              o[(size_t)t * E + hh * D + d] = (float)(acc / den);
            }
          }
        Vec y = apply(b.o, o, N);
        rms(y.data(), N, E, b.attn_post.data(), eps);
        for (size_t i = 0; i < x.size(); ++i) x[i] += y[i];
      }
      {  // convolution module
        Vec h = x;
        rms(h.data(), N, E, b.conv_pre.data(), eps);
        const Vec g = apply(b.pw1, h, N);  // [N][2E]
        Vec glu((size_t)N * E), c((size_t)N * E);
        for (int t = 0; t < N; ++t)
          for (int e = 0; e < E; ++e) {
            const double a = g[(size_t)t * 2 * E + e], z = g[(size_t)t * 2 * E + E + e];
            glu[(size_t)t * E + e] = (float)(a / (1.0 + std::exp(-z)));
          }
        const int K = (int)(b.dw.size() / E);
        for (int t = 0; t < N; ++t)
          for (int e = 0; e < E; ++e) {
            double s = 0;
            for (int j = 0; j < K; ++j)
              if (t - (K - 1) + j >= 0) s += (double)b.dw[(size_t)e * K + j] * glu[(size_t)(t - (K - 1) + j) * E + e];
            c[(size_t)t * E + e] = (float)s;
          }
        rms(c.data(), N, E, b.conv_post.data(), eps);
        for (float& v : c) v = silu(v);
        const Vec y = apply(b.pw2, c, N);
        for (size_t i = 0; i < x.size(); ++i) x[i] += y[i];
      }
      ffn(b.ff_norm1, b.up1, b.down1, b.ff_post1);
      rms(x.data(), N, E, b.ln2.data(), eps);
    }
    Vec y = apply(outp, x, N);
    for (int t = 0; t < N; ++t)
      for (int i = 0; i < outp.out; ++i) y[(size_t)t * outp.out + i] += outb[i];
    rms(y.data(), N, outp.out, nullptr, eps);
    const Vec z = apply(mmp, y, N);
    ref.insert(ref.end(), z.begin(), z.end());
    n_tok += N;
  }
  const int n_out = mmp.out;
  fprintf(stderr, "f32 reference: %.1f s audio -> %d tokens in %.1f s (%d threads)%s\n", s16.size() / 16000.0, n_tok,
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), omp_get_max_threads(),
          w16 ? ", weights rounded to f16" : "");
  FILE* fg = fopen((clip + ".g4a.bin").c_str(), "rb");
  int hg[3] = {0, 0, 0};
  if (!fg || fread(hg, 4, 3, fg) != 3 || hg[0] != n_tok || hg[1] != n_out) {
    fprintf(stderr, "no matching %s.g4a.bin (run vision_testz first)\n", clip.c_str());
    return 1;
  }
  Vec gpu(ref.size());
  if (fread(gpu.data(), 4, gpu.size(), fg) != gpu.size()) return 1;
  fclose(fg);
  FILE* fo = fopen((clip + (getenv("REF_AOUT") ? getenv("REF_AOUT") : ".g4a.ref.bin")).c_str(), "wb");
  fwrite(hg, 4, 3, fo);
  fwrite(ref.data(), 4, ref.size(), fo);
  fclose(fo);
  double cmin = 1, csum = 0, rsum = 0, rmax = 0;
  for (int m = 0; m < n_tok; ++m) {
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
  printf("%d tokens vs GPU: cosine min %.6f mean %.6f; relative error mean %.3f %% max %.3f %%\n", n_tok, cmin,
         csum / n_tok, rsum / n_tok * 100, rmax * 100);
  return 0;
}

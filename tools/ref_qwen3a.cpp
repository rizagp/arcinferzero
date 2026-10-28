// CPU f32 reference of the Qwen3-ASR audio encoder (mmproj qwen3a), written from Hugging Face Qwen3-ASR
// (modeling_qwen3_asr.py: Qwen3ASRAudioEncoder) with the weights dequantized to f32: the check for
// zero/audio_qwen3a.h (f16 weights on the GPU). Straight loops, no shared code with the GPU path except the log-mel
// (src/whisper.h, Hugging Face's WhisperFeatureExtractor without padding).
// usage: ref_qwen3a mmproj.gguf audio_file out.bin [seconds]   (out.bin: n_tok x 2048 f32 rows)
#include "../src/audio.h"
#include "../src/gguf.h"
#include "../src/quants.h"
#include "../src/whisper.h"
#include <cmath>
#include <cstdio>

static GgufFile* G;
static std::vector<float> W(const std::string& n) {
  const GgufTensor& t = G->get(n);
  std::vector<float> v(t.nelements());
  dequant_row(t.type, t.data, (int)v.size(), v.data());
  return v;
}
static float gelu(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.70710678f)); }
// y[M][N] = x[M][K] . w[N][K]^T + b
static std::vector<float> linear(const std::vector<float>& x, int M, int K, const std::string& name, bool bias = true) {
  const std::vector<float> w = W(name + ".weight"), b = bias ? W(name + ".bias") : std::vector<float>();
  const int N = (int)(w.size() / K);
  std::vector<float> y((size_t)M * N);
#pragma omp parallel for
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      const float* a = &x[(size_t)m * K];
      const float* r = &w[(size_t)n * K];
      float s = 0;
      for (int k = 0; k < K; ++k) s += a[k] * r[k];
      y[(size_t)m * N + n] = s + (bias ? b[n] : 0.0f);
    }
  return y;
}
static void layernorm(float* x, int E, const float* w, const float* b, float eps) {
  double m = 0, v = 0;
  for (int i = 0; i < E; ++i) m += x[i];
  m /= E;
  for (int i = 0; i < E; ++i) v += (x[i] - m) * (x[i] - m);
  const double r = 1.0 / std::sqrt(v / E + eps);
  for (int i = 0; i < E; ++i) x[i] = (float)((x[i] - m) * r) * w[i] + b[i];
}
// Conv2d(IC -> OC, 3x3, stride 2, padding 1) + GELU over x[IC][H][Wd] (H = frequency, Wd = time)
static std::vector<float> conv(const std::vector<float>& x, int IC, int H, int Wd, const std::string& name, int& Ho, int& Wo) {
  const std::vector<float> w = W(name + ".weight"), b = W(name + ".bias");  // w [OC][IC][3][3]
  const int OC = (int)b.size();
  Ho = (H - 1) / 2 + 1;
  Wo = (Wd - 1) / 2 + 1;
  std::vector<float> y((size_t)OC * Ho * Wo);
  const int ho_ = Ho, wo_ = Wo;
#pragma omp parallel for
  for (int oc = 0; oc < OC; ++oc)
    for (int i = 0; i < ho_; ++i)
      for (int j = 0; j < wo_; ++j) {
        float s = b[oc];
        for (int ic = 0; ic < IC; ++ic)
          for (int kh = 0; kh < 3; ++kh)
            for (int kw = 0; kw < 3; ++kw) {
              const int hh = 2 * i + kh - 1, ww = 2 * j + kw - 1;
              if (hh < 0 || hh >= H || ww < 0 || ww >= Wd) continue;
              s += w[(((size_t)oc * IC + ic) * 3 + kh) * 3 + kw] * x[((size_t)ic * H + hh) * Wd + ww];
            }
        y[((size_t)oc * ho_ + i) * wo_ + j] = gelu(s);
      }
  return y;
}

static int run(int argc, char** argv);
int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
static int run(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: ref_qwen3a mmproj.gguf audio out.bin [seconds]\n"); return 1; }
  GgufFile g(argv[1]);
  G = &g;
  std::vector<float> s = audio::decode_16k_file(argv[2], 0);
  if (argc > 4 && atof(argv[4]) > 0) s.resize(std::min(s.size(), (size_t)(atof(argv[4]) * 16000)));
  if (s.size() < 8000) s.resize(8000, 0.0f);
  const int E = (int)g.get_int("clip.audio.embedding_length", 0), L = (int)g.get_int("clip.audio.block_count", 0);
  const int nh = (int)g.get_int("clip.audio.attention.head_count", 0), FF = (int)g.get_int("clip.audio.feed_forward_length", 0);
  const float eps = (float)g.get_float("clip.audio.attention.layer_norm_epsilon", 1e-5);
  int F = 0, content = 0;
  const std::vector<float> mel = whisper::log_mel(s, 128, F, content, 0);  // [128][F]
  // chunks of 100 frames (the last one zero-padded), three convolutions, the kept frames of each chunk
  const int nch = (F + 99) / 100;
  const std::vector<float> pe = W("a.position_embd.weight");
  std::vector<float> x;      // [n_tok][E]
  std::vector<int> win_of;   // window (8 chunks) of each token
  for (int c = 0; c < nch; ++c) {
    const int Lc = std::min(100, F - 100 * c);
    std::vector<float> in((size_t)128 * 100, 0.0f);
    for (int m = 0; m < 128; ++m)
      for (int t = 0; t < Lc; ++t) in[(size_t)m * 100 + t] = mel[(size_t)m * F + 100 * c + t];
    int H1, W1, H2, W2, H3, W3;
    const std::vector<float> a1 = conv(in, 1, 128, 100, "a.conv2d.1", H1, W1);
    const std::vector<float> a2 = conv(a1, 480, H1, W1, "a.conv2d.2", H2, W2);
    const std::vector<float> a3 = conv(a2, 480, H2, W2, "a.conv2d.3", H3, W3);  // [480][16][13]
    int nt = Lc;
    for (int i = 0; i < 3; ++i) nt = (nt - 1) / 2 + 1;
    std::vector<float> feat((size_t)nt * 480 * H3);  // (b, t, c * f)
    for (int t = 0; t < nt; ++t)
      for (int ch = 0; ch < 480; ++ch)
        for (int f = 0; f < H3; ++f) feat[(size_t)t * 480 * H3 + ch * H3 + f] = a3[((size_t)ch * H3 + f) * W3 + t];
    std::vector<float> y = linear(feat, nt, 480 * H3, "a.conv_out", false);
    for (int t = 0; t < nt; ++t)
      for (int i = 0; i < E; ++i) y[(size_t)t * E + i] += pe[(size_t)t * E + i];
    x.insert(x.end(), y.begin(), y.end());
    for (int t = 0; t < nt; ++t) win_of.push_back(c / 8);
    fprintf(stderr, "\rconv chunk %d/%d", c + 1, nch);
  }
  const int N = (int)win_of.size(), D = E / nh;
  for (int il = 0; il < L; ++il) {
    const std::string p = "a.blk." + std::to_string(il) + ".";
    const std::vector<float> l1w = W(p + "ln1.weight"), l1b = W(p + "ln1.bias"), l2w = W(p + "ln2.weight"), l2b = W(p + "ln2.bias");
    std::vector<float> h = x;
    for (int t = 0; t < N; ++t) layernorm(&h[(size_t)t * E], E, l1w.data(), l1b.data(), eps);
    const std::vector<float> q = linear(h, N, E, p + "attn_q"), k = linear(h, N, E, p + "attn_k"), v = linear(h, N, E, p + "attn_v");
    std::vector<float> o((size_t)N * E);
#pragma omp parallel for
    for (int t = 0; t < N; ++t)
      for (int hh = 0; hh < nh; ++hh) {
        std::vector<double> sc;
        std::vector<int> keys;
        double mx = -1e300;
        for (int u = 0; u < N; ++u) {
          if (win_of[u] != win_of[t]) continue;
          double sum = 0;
          for (int i = 0; i < D; ++i) sum += (double)q[(size_t)t * E + hh * D + i] * k[(size_t)u * E + hh * D + i];
          sc.push_back(sum / std::sqrt((double)D));
          keys.push_back(u);
          mx = std::max(mx, sc.back());
        }
        double z = 0;
        for (double& a : sc) { a = std::exp(a - mx); z += a; }
        for (int i = 0; i < D; ++i) {
          double acc = 0;
          for (size_t j = 0; j < keys.size(); ++j) acc += sc[j] * v[(size_t)keys[j] * E + hh * D + i];
          o[(size_t)t * E + hh * D + i] = (float)(acc / z);
        }
      }
    const std::vector<float> ao = linear(o, N, E, p + "attn_out");
    for (size_t i = 0; i < x.size(); ++i) x[i] += ao[i];
    h = x;
    for (int t = 0; t < N; ++t) layernorm(&h[(size_t)t * E], E, l2w.data(), l2b.data(), eps);
    std::vector<float> u = linear(h, N, E, p + "ffn_up");
    for (float& a : u) a = gelu(a);
    const std::vector<float> d = linear(u, N, FF, p + "ffn_down");
    for (size_t i = 0; i < x.size(); ++i) x[i] += d[i];
    fprintf(stderr, "\rlayer %d/%d     ", il + 1, L);
  }
  const std::vector<float> pw = W("a.post_ln.weight"), pb = W("a.post_ln.bias");
  for (int t = 0; t < N; ++t) layernorm(&x[(size_t)t * E], E, pw.data(), pb.data(), eps);
  std::vector<float> y = linear(x, N, E, "mm.a.mlp.1");
  for (float& a : y) a = gelu(a);
  const std::vector<float> out = linear(y, N, (int)(y.size() / N), "mm.a.mlp.2");
  FILE* fo = fopen(argv[3], "wb");
  if (!fo) { fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
  fwrite(out.data(), 4, out.size(), fo);
  fclose(fo);
  printf("\n%d frames -> %d tokens x %d -> %s\n", F, N, (int)(out.size() / N), argv[3]);
  return 0;
}

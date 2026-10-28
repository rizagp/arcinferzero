// arcinferzero: the Qwen3-ASR audio encoder (mmproj audio projector type qwen3a), built on zero/engine.h (Level Zero).
// Matches Hugging Face Qwen3-ASR (modeling_qwen3_asr.py, the Qwen3-Omni audio encoder) and llama.cpp tools/mtmd
// (models/qwen3a.cpp, mtmd_audio_preprocessor_qwen3a):
//  - CPU: 16 kHz samples -> Whisper's log-mel without padding (src/whisper.h log_mel: 400-sample Hann window, hop 160,
//    128 Slaney bins, log10, clamped to the maximum of the whole clip - 8, (x + 4) / 4), one frame per 10 ms; cut into
//    chunks of 100 frames (the last one zero-padded to 100, as Hugging Face's pad_sequence does); the first 3x3
//    stride-2 convolution (1 -> 480 channels) and its GELU;
//  - GPU (f16 weights, f32 activations, zero/kernels/ops.cl vit_* and q3a_*): the second and third convolutions as
//    im2col + GEMM (GELU), 13 frames per chunk of which the last chunk keeps those its real frames give; conv_out
//    (480 channels x 16 frequencies -> 1024), the sinusoidal position of each frame within its chunk; 24 pre-LayerNorm
//    transformer blocks (16 heads, GELU FFN) whose attention stays inside windows of 8 chunks (n_window_infer = 800
//    frames, 104 tokens); a final LayerNorm and the projector (1024 -> GELU -> 2048). 13 tokens per second of audio.
// The windows never see each other, so the encoder runs on groups of windows one after the other.
#pragma once
#include "../src/audio_gemma4.h"
#include "../src/whisper.h"
#include "engine.h"
#include "../src/gguf.h"
#include "../src/quants.h"
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

class AudioQwen3A : public GemmaAudioEncoder {
 public:
  static constexpr int kChunk = 100;     // mel frames per convolution chunk (n_window * 2)
  static constexpr int kChunkTok = 13;   // tokens of a full chunk (100 -> 50 -> 25 -> 13)
  static constexpr int kWinChunks = 8;   // chunks per attention window (n_window_infer 800 / 100)
  static constexpr int kGroupWins = 16;  // windows encoded at once (their buffers; ~128 s of audio)
  int n_embd = 0, n_ff = 0, n_layer = 0, n_head = 0, n_mel = 128, n_out = 0, n_conv = 0;
  float eps = 1e-5f;
  double encode_ms = 0;

  AudioQwen3A(const std::string& path, Engine& eng) : f_(path), e_(eng) {
    if (f_.get_str("clip.audio.projector_type") != "qwen3a") throw std::runtime_error("mmproj: not a Qwen3-ASR audio encoder (qwen3a)");
    n_embd = (int)f_.get_int("clip.audio.embedding_length", 0);
    n_ff = (int)f_.get_int("clip.audio.feed_forward_length", 0);
    n_layer = (int)f_.get_int("clip.audio.block_count", 0);
    n_head = (int)f_.get_int("clip.audio.attention.head_count", 16);
    n_mel = (int)f_.get_int("clip.audio.num_mel_bins", 128);
    eps = (float)f_.get_float("clip.audio.attention.layer_norm_epsilon", 1e-5);
    n_out = (int)f_.get("mm.a.mlp.2.weight").dims[1];
    n_conv = (int)f_.get("a.conv2d.1.weight").dims[3];
    const GgufTensor& co = f_.get("a.conv_out.weight");
    if (n_mel != 128 || n_head <= 0 || n_embd % n_head || n_embd / n_head != 64 || n_embd % 16 || n_ff % 16 ||
        n_out % 16 || n_conv % 16 || (int)co.dims[0] != n_conv * 16 || (int)co.dims[1] != n_embd ||
        (int)f_.get("a.position_embd.weight").dims[1] < kChunkTok)
      throw std::runtime_error("mmproj: this Qwen3-ASR audio shape is not supported");
  }

  int out_dim() const override { return n_out; }

  // Tokens the encoder makes of n 16 kHz samples (frames n / 160, at least 0.5 s).
  static int tokens(size_t n) {
    const int F = (int)(std::max<size_t>(n, 8000) / whisper::kHop);
    int t = (F / kChunk) * kChunkTok;
    if (F % kChunk) t += chunk_tokens(F % kChunk);
    return t;
  }
  static int chunk_tokens(int L) {
    for (int i = 0; i < 3; ++i) L = (L - 1) / 2 + 1;
    return L;
  }

  // GPU memory of the encoder: its weights and the buffers of a group of windows.
  static size_t gpu_bytes(const GgufFile& g) {
    const size_t E = g.get_int("clip.audio.embedding_length", 0), F = g.get_int("clip.audio.feed_forward_length", 0);
    const size_t C = 480, W = kWinChunks, Mp = (size_t)kGroupWins * W * kChunkTok;
    const size_t conv = W * 50 * 64 * C * 4 + W * 800 * C * 9 * 4 + W * 800 * C * 4 + W * 208 * C * 4;
    const size_t per = C * 16 * 4 + E * 4 * 3 + std::max(3 * E, F) * 4 + E * 2 * 3 + 2048 * 4;
    return enc_weight_bytes(g, {"a.", "mm.a."}, {"a.conv2d.1.weight", "a.position_embd.weight"}) + conv + Mp * per;
  }

  // 16 kHz samples -> n_tok x n_out embeddings.
  std::vector<float> encode(const std::vector<float>& s0, int& n_tok) override {
    load();
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> s = s0;
    if (s.size() < 8000) s.resize(8000, 0.0f);  // (Qwen3-ASR: clips under 0.5 s zero-padded to it)
    int F = 0, content = 0;
    const std::vector<float> mel = whisper::log_mel(s, n_mel, F, content, 0);  // [128][F]
    const int nch = (F + kChunk - 1) / kChunk, nwin = (nch + kWinChunks - 1) / kWinChunks;
    std::vector<float> out;
    n_tok = 0;
    for (int w0 = 0; w0 < nwin; w0 += kGroupWins) {
      const int w1 = std::min(nwin, w0 + kGroupWins);
      std::vector<int> wtok;  // tokens per window of the group
      const std::vector<float> e = encode_group(mel, F, w0, w1, wtok);
      out.insert(out.end(), e.begin(), e.end());
      for (int t : wtok) n_tok += t;
    }
    encode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
  }

 private:
  GgufFile f_;
  Engine& e_;
  struct FW {
    cl_mem w = nullptr;
    int N = 0, K = 0;
  };
  struct Layer {
    cl_mem ln1w, ln1b, ln2w, ln2b, qkv_b, out_b, up_b, down_b;
    FW q, k, v, out, up, down;
  };
  bool loaded_ = false;
  std::vector<Layer> L_;
  FW c2_, c3_, cout_, mm1_, mm2_;
  cl_mem c2b_ = nullptr, c3b_ = nullptr, pos_ = nullptr, post_w_ = nullptr, post_b_ = nullptr, mm1b_ = nullptr,
         mm2b_ = nullptr;
  std::vector<float> c1w_, c1b_;  // first convolution [480][3][3], on the host
  int cap_ = 0;                    // token rows the group buffers hold
  cl_mem a1_ = nullptr, im_ = nullptr, a2_ = nullptr, a3_ = nullptr, tmap_ = nullptr, tpos_ = nullptr, g_ = nullptr,
         x_ = nullptr, hn_ = nullptr, h_ = nullptr, o_ = nullptr, q_ = nullptr, k_ = nullptr, v_ = nullptr;

  std::vector<float> host(const std::string& name) const {
    const GgufTensor& t = f_.get(name);
    std::vector<float> v(t.nelements());
    dequant_row(t.type, t.data, (int)v.size(), v.data());
    return v;
  }
  cl_mem f32(const std::vector<float>& v) { return e_.buf(v.size() * 4, v.data()); }
  cl_mem f32(const std::string& name) { return f32(host(name)); }
  FW f16(const std::vector<float>& v, int N, int K) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = f32_to_fp16(v[i]);
    FW w;
    w.w = e_.buf(h.size() * 2, h.data());
    w.N = N;
    w.K = K;
    return w;
  }
  FW f16(const std::string& name) {  // a matrix [N][K] (an [3, 3, IC, OC] convolution: K = IC * 9)
    const GgufTensor& t = f_.get(name);
    const int N = (int)t.dims.back();
    return f16(host(name), N, (int)(t.nelements() / N));
  }

  void load() {
    if (loaded_) return;
    c1w_ = host("a.conv2d.1.weight");
    c1b_ = host("a.conv2d.1.bias");
    c2_ = f16("a.conv2d.2.weight");
    c2b_ = f32("a.conv2d.2.bias");
    c3_ = f16("a.conv2d.3.weight");
    c3b_ = f32("a.conv2d.3.bias");
    cout_ = f16("a.conv_out.weight");
    {
      std::vector<float> p = host("a.position_embd.weight");
      p.resize((size_t)kChunkTok * n_embd);  // (positions 0..12 of a chunk only)
      pos_ = f32(p);
    }
    for (int il = 0; il < n_layer; ++il) {
      const std::string p = "a.blk." + std::to_string(il) + ".";
      Layer l;
      l.ln1w = f32(p + "ln1.weight");
      l.ln1b = f32(p + "ln1.bias");
      l.ln2w = f32(p + "ln2.weight");
      l.ln2b = f32(p + "ln2.bias");
      l.q = f16(p + "attn_q.weight");
      l.k = f16(p + "attn_k.weight");
      l.v = f16(p + "attn_v.weight");
      {
        std::vector<float> b = host(p + "attn_q.bias"), bk = host(p + "attn_k.bias"), bv = host(p + "attn_v.bias");
        b.insert(b.end(), bk.begin(), bk.end());
        b.insert(b.end(), bv.begin(), bv.end());
        l.qkv_b = f32(b);
      }
      l.out = f16(p + "attn_out.weight");
      l.out_b = f32(p + "attn_out.bias");
      l.up = f16(p + "ffn_up.weight");
      l.up_b = f32(p + "ffn_up.bias");
      l.down = f16(p + "ffn_down.weight");
      l.down_b = f32(p + "ffn_down.bias");
      if (l.q.N != n_embd || l.q.K != n_embd || l.up.N != n_ff || l.down.K != n_ff)
        throw std::runtime_error("mmproj: unexpected Qwen3-ASR audio block shapes");
      L_.push_back(l);
    }
    post_w_ = f32("a.post_ln.weight");
    post_b_ = f32("a.post_ln.bias");
    mm1_ = f16("mm.a.mlp.1.weight");
    mm1b_ = f32("mm.a.mlp.1.bias");
    mm2_ = f16("mm.a.mlp.2.weight");
    mm2b_ = f32("mm.a.mlp.2.bias");
    if (c2_.K != n_conv * 9 || c3_.K != n_conv * 9 || mm1_.K != n_embd || mm2_.K != mm1_.N)
      throw std::runtime_error("mmproj: unexpected Qwen3-ASR audio shapes");
    const size_t C = n_conv, W = kWinChunks;
    a1_ = e_.buf(W * 50 * 64 * C * 4);
    im_ = e_.buf(W * 800 * C * 9 * 4);
    a2_ = e_.buf(W * 800 * C * 4);
    a3_ = e_.buf(W * 208 * C * 4);
    loaded_ = true;
  }
  void reserve(int Mp) {  // token rows of a group (grown as needed: the smaller set released first)
    if (Mp <= cap_) return;
    const size_t E = n_embd;
    for (cl_mem* m : {&tmap_, &tpos_, &g_, &x_, &hn_, &h_, &o_, &q_, &k_, &v_}) e_.free_buf(*m);
    tmap_ = e_.buf((size_t)Mp * 4);
    tpos_ = e_.buf((size_t)Mp * 4);
    g_ = e_.buf((size_t)Mp * n_conv * 16 * 4);
    x_ = e_.buf((size_t)Mp * E * 4);
    hn_ = e_.buf((size_t)Mp * E * 4);
    h_ = e_.buf((size_t)Mp * std::max<size_t>(3 * E, std::max(n_ff, mm2_.N)) * 4);
    o_ = e_.buf((size_t)Mp * std::max<size_t>(E, mm1_.N) * 4);
    for (cl_mem* m : {&q_, &k_, &v_}) *m = e_.buf((size_t)Mp * E * 2 + 64);
    cap_ = Mp;
  }

  // the first convolution of chunks [c0, c0 + n) on the host: out[(c * 50 + t) * 64 + f][oc], GELU applied
  void conv1(const std::vector<float>& mel, int F, int c0, int n, std::vector<float>& out) const {
    const int C = n_conv, To = kChunk / 2, Fo = n_mel / 2;
    out.assign((size_t)n * To * Fo * C, 0.0f);
#pragma omp parallel for
    for (int r = 0; r < n * To * Fo; ++r) {
      const int f = r % Fo, t = (r / Fo) % To, c = r / (Fo * To), fr0 = (c0 + c) * kChunk;
      float in[9];
      for (int kh = 0; kh < 3; ++kh)
        for (int kw = 0; kw < 3; ++kw) {
          const int tt = 2 * t + kw - 1, ff = 2 * f + kh - 1, fr = fr0 + tt;
          // (zero padding of the convolution, and of the last chunk past the clip's frames)
          in[kh * 3 + kw] = tt >= 0 && tt < kChunk && ff >= 0 && ff < n_mel && fr < F ? mel[(size_t)ff * F + fr] : 0.0f;
        }
      float* o = &out[(size_t)r * C];
      for (int oc = 0; oc < C; ++oc) {
        const float* w = &c1w_[(size_t)oc * 9];
        float a = c1b_[oc];
        for (int i = 0; i < 9; ++i) a += w[i] * in[i];
        o[oc] = 0.5f * a * (1.0f + std::erf(a * 0.70710678f));
      }
    }
  }

  // windows [w0, w1) of the clip -> their tokens' embeddings; wtok: the tokens of each window
  std::vector<float> encode_group(const std::vector<float>& mel, int F, int w0, int w1, std::vector<int>& wtok) {
    const int nch = (F + kChunk - 1) / kChunk, E = n_embd, C = n_conv;
    // tokens: per window, per chunk its kept frames
    std::vector<int> tmap, tpos;  // (row of the third convolution's output within its window; position in the chunk)
    std::vector<int> wstart;      // first token of each window in the group
    for (int w = w0; w < w1; ++w) {
      wstart.push_back((int)tpos.size());
      for (int c = w * kWinChunks; c < std::min(nch, (w + 1) * kWinChunks); ++c) {
        const int L = std::min(kChunk, F - c * kChunk), nt = chunk_tokens(L);
        for (int t = 0; t < nt; ++t) {
          tmap.push_back((c - w * kWinChunks) * kChunkTok + t);
          tpos.push_back(t);
        }
      }
      wtok.push_back((int)tpos.size() - wstart.back());
    }
    const int N = (int)tpos.size(), Mp = (N + 15) / 16 * 16;
    reserve(Mp);
    auto& q = e_.cl_.queue;
    CL_CHECK(clEnqueueWriteBuffer(q, tmap_, CL_TRUE, 0, (size_t)N * 4, tmap.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(q, tpos_, CL_TRUE, 0, (size_t)N * 4, tpos.data(), 0, nullptr, nullptr));
    // the convolutions, window by window, into the encoder input rows g_
    std::vector<float> a1;
    for (int w = w0; w < w1; ++w) {
      const int c0 = w * kWinChunks, n = std::min(nch, c0 + kWinChunks) - c0, r0 = wstart[w - w0];
      conv1(mel, F, c0, n, a1);
      CL_CHECK(clEnqueueWriteBuffer(q, a1_, CL_TRUE, 0, a1.size() * 4, a1.data(), 0, nullptr, nullptr));
      im2col(a1_, 50, 64, 25, 32, n);
      gemm(im_, 9 * C, c2_, a2_, C, 0, false, n * 800, n * 800);
      bias_act(a2_, C, c2b_, 2, n * 800);
      im2col(a2_, 25, 32, 13, 16, n);
      gemm(im_, 9 * C, c3_, a3_, C, 0, false, n * 208, n * 208);
      bias_act(a3_, C, c3b_, 2, n * 208);
      cl_kernel k = kern("q3a_gather");
      int cc = C, ff = 16;
      set_args(k, 0, a3_, cc, ff, e_.from(tmap_, (size_t)r0 * 4), e_.from(g_, (size_t)r0 * C * 16 * 4));
      size_t g[2] = {(size_t)C * 16, (size_t)wtok[w - w0]};
      e_.run("q3a_conv", k, 2, g, nullptr);
    }
    gemm(g_, C * 16, cout_, x_, E, 0, false, Mp, N);
    {
      cl_kernel k = kern("q3a_pos");
      int e = E;
      set_args(k, 0, x_, e, pos_, tpos_);
      size_t g[2] = {(size_t)E, (size_t)N};
      e_.run("q3a_add", k, 2, g, nullptr);
    }
    const int hd = E / n_head;
    for (const Layer& l : L_) {
      ln(x_, l.ln1w, l.ln1b, N, hn_, E);
      gemm(hn_, E, l.q, h_, 3 * E, 0, false, Mp, N);
      gemm(hn_, E, l.k, h_, 3 * E, E, false, Mp, N);
      gemm(hn_, E, l.v, h_, 3 * E, 2 * E, false, Mp, N);
      bias_act(h_, 3 * E, l.qkv_b, 0, N);
      {
        cl_kernel k = kern("q3a_heads");
        int nh = n_head, d = hd, mp = Mp;
        float qs = 1.0f / std::sqrt((float)hd);
        set_args(k, 0, h_, nh, d, mp, qs, q_, k_, v_);
        size_t g[2] = {(size_t)E, (size_t)N};
        e_.run("q3a_heads", k, 2, g, nullptr);
      }
      // (non-causal attention inside each window: one launch per window, its rows from wstart)
      cl_kernel ka = kern("vit_attn_rows", "-cl-intel-256-GRF-per-thread -DHD=" + std::to_string(hd));
      for (size_t w = 0; w < wtok.size(); ++w) {
        const int m0 = wstart[w];
        int n = wtok[w], mp = Mp, nh = n_head;
        set_args(ka, 0, e_.from(q_, (size_t)m0 * hd * 2), e_.from(k_, (size_t)m0 * hd * 2), e_.from(v_, (size_t)m0 * hd * 2),
                 n, mp, nh, e_.from(o_, (size_t)m0 * E * 4));
        const size_t rows = 32;  // (4 sub-groups of 8 query rows)
        size_t g[2] = {((size_t)n + rows - 1) / rows * rows, (size_t)n_head}, lw[2] = {rows, 1};
        e_.run("q3a_attn", ka, 2, g, lw);
      }
      gemm(o_, E, l.out, x_, E, 0, true, Mp, N);
      add(x_, E, l.out_b, N);
      ln(x_, l.ln2w, l.ln2b, N, hn_, E);
      gemm(hn_, E, l.up, h_, n_ff, 0, false, Mp, N);
      bias_act(h_, n_ff, l.up_b, 2, N);
      gemm(h_, n_ff, l.down, x_, E, 0, true, Mp, N);
      add(x_, E, l.down_b, N);
    }
    ln(x_, post_w_, post_b_, N, hn_, E);
    gemm(hn_, E, mm1_, o_, mm1_.N, 0, false, Mp, N);
    bias_act(o_, mm1_.N, mm1b_, 2, N);
    gemm(o_, mm1_.N, mm2_, h_, n_out, 0, false, Mp, N);
    bias_act(h_, n_out, mm2b_, 0, N);
    std::vector<float> out((size_t)N * n_out);
    CL_CHECK(clEnqueueReadBuffer(q, h_, CL_TRUE, 0, out.size() * 4, out.data(), 0, nullptr, nullptr));
    e_.collect_profile();
    return out;
  }

  cl_kernel kern(const char* name, const std::string& defs = "") { return e_.kernel("ops", "-DVIT " + defs, name); }
  void im2col(cl_mem in, int Ti, int Fi, int To, int Fo, int chunks) {
    cl_kernel k = kern("q3a_im2col");
    int c = n_conv, ti = Ti, fi = Fi, to = To, fo = Fo;
    set_args(k, 0, in, c, ti, fi, to, fo, im_);
    size_t g[2] = {(size_t)n_conv * 9, (size_t)chunks * To * Fo};
    e_.run("q3a_conv", k, 2, g, nullptr);
  }
  void gemm(cl_mem A, int lda, const FW& w, cl_mem C, int ldc, int c_off, bool acc, int M, int n) {
    cl_kernel k = e_.kernel("ops", "-DVIT -cl-intel-256-GRF-per-thread", "vit_gemm_f16");
    int ld = lda, K = w.K, lc = ldc, co = c_off, ac = acc ? 1 : 0, mv = n;
    set_args(k, 0, A, ld, w.w, K, C, lc, co, ac, mv);
    const int tiles = M / 16;
    int wgy = 8;
    while (tiles % wgy) wgy /= 2;
    size_t g[2] = {(size_t)w.N, (size_t)tiles}, lw[2] = {16, (size_t)wgy};
    e_.run("q3a_gemm", k, 2, g, lw);
  }
  void ln(cl_mem x, cl_mem w, cl_mem b, int n, cl_mem out, int E) {
    cl_kernel k = kern("vit_ln");
    int e = E;
    float ep = eps;
    set_args(k, 0, x, e, w, b, ep, out);
    size_t g[2] = {256, (size_t)n}, lw[2] = {256, 1};
    e_.run("q3a_ln", k, 2, g, lw);
  }
  void bias_act(cl_mem x, int N, cl_mem b, int gelu, int n) {
    cl_kernel k = kern("vit_bias_act");
    int nn = N, g2 = gelu;
    set_args(k, 0, x, nn, b, g2);
    size_t g[2] = {(size_t)N, (size_t)n};
    e_.run("q3a_add", k, 2, g, nullptr);
  }
  void add(cl_mem x, int N, cl_mem a, int n) {
    cl_kernel k = kern("vit_add");
    int nn = N, pr = 0;
    set_args(k, 0, x, nn, a, pr);
    size_t g[2] = {(size_t)N, (size_t)n};
    e_.run("q3a_add", k, 2, g, nullptr);
  }
};

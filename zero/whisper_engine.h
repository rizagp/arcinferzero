// Whisper on the Arc iGPU through Level Zero (zero/clrt.h): src/whisper.h's Backend. The matrices are the engine's
// int8 Q8_0 GEMM form (gemm.cl, QTYPE 8, per-32 activation scales from ops.cl swiglu_quant), the encoder attention
// is the vision encoders' vit_attn_rows (ops.cl -DVIT, f32 with f16 q / k / v), the rest zero/kernels/whisper.cl.
// The decoder runs every token through the same GEMM kernel with the rows padded to 16 (its weights are read once
// per pass: beams and samples share it). docs/whisper-native.md.
#pragma once
#include "../src/quants.h"
#include "../src/whisper.h"
#include "clrt.h"
#include "npu/whisper_npu.h"
#include <chrono>
#include <map>

class WhisperEngine : public whisper::Backend {
 public:
  // npu: the encoder on the NPU (zero/npu/whisper_npu.h), the decoder here
  WhisperEngine(const whisper::Model& m, const std::string& kernel_dir = "zero/kernels", int max_seq = 5, bool npu = false)
      : m_(m), hp_(m.hp), max_seq_(std::max(1, max_seq)) {
    // (the NPU's driver is a second Level Zero driver: Level Zero initialized for every type before the GPU runtime,
    //  which asks for GPUs only)
    if (npu && ze_load()) zeInit(0);
    if (!cl_.init(true)) throw std::runtime_error("whisper: no Level Zero GPU");
    const auto t0 = std::chrono::steady_clock::now();
    gemm_src_ = read_file(kernel_dir + "/gemm.cl");
    ops_src_ = read_file(kernel_dir + "/ops.cl");
    wh_src_ = read_file(kernel_dir + "/whisper.cl");
    if (gemm_src_.empty() || ops_src_.empty() || wh_src_.empty())
      throw std::runtime_error("whisper: kernels not found in " + kernel_dir);
    const int D = hp_.d, A = hp_.n_audio_ctx;
    Ap_ = (A + 15) / 16 * 16;
    T1p_ = (whisper::kWindowFrames + 15) / 16 * 16;
    Vp_ = (hp_.n_vocab + 15) / 16 * 16;
    const int F = m.file->get("encoder/layer_0/ffn/linear_0/weight").shape[0];
    F_ = F;
    // weights: Q8_0 GEMM form; biases f16 (the encoder's only when it runs here)
    if (!npu) {
    conv1_ = weight_conv("encoder/conv1/weight", K1p_ = 512);
    conv1_b_ = half_buf("encoder/conv1/bias");
    conv2_ = weight_conv("encoder/conv2/weight", 3 * D);
    conv2_b_ = half_buf("encoder/conv2/bias");
    enc_pos_ = half_buf("encoder/position_encodings/encodings");
    for (int l = 0; l < hp_.enc_layers; ++l) {
      const std::string p = "encoder/layer_" + std::to_string(l) + "/";
      EncL L;
      L.ln1g = half_buf(p + "self_attention/layer_norm/gamma");
      L.ln1b = half_buf(p + "self_attention/layer_norm/beta");
      L.qkv = weight(p + "self_attention/linear_0/weight");
      L.qkv_b = half_buf(p + "self_attention/linear_0/bias");
      L.out = weight(p + "self_attention/linear_1/weight");
      L.out_b = half_buf(p + "self_attention/linear_1/bias");
      L.ln2g = half_buf(p + "ffn/layer_norm/gamma");
      L.ln2b = half_buf(p + "ffn/layer_norm/beta");
      L.fc1 = weight(p + "ffn/linear_0/weight");
      L.fc1_b = half_buf(p + "ffn/linear_0/bias");
      L.fc2 = weight(p + "ffn/linear_1/weight");
      L.fc2_b = half_buf(p + "ffn/linear_1/bias");
      enc_.push_back(L);
    }
    enc_lng_ = half_buf("encoder/layer_norm/gamma");
    enc_lnb_ = half_buf("encoder/layer_norm/beta");
    }
    for (int l = 0; l < hp_.dec_layers; ++l) {
      const std::string p = "decoder/layer_" + std::to_string(l) + "/";
      DecL L;
      L.ln1g = half_buf(p + "self_attention/layer_norm/gamma");
      L.ln1b = half_buf(p + "self_attention/layer_norm/beta");
      L.qkv = weight(p + "self_attention/linear_0/weight");
      L.qkv_b = half_buf(p + "self_attention/linear_0/bias");
      L.out = weight(p + "self_attention/linear_1/weight");
      L.out_b = half_buf(p + "self_attention/linear_1/bias");
      L.ln2g = half_buf(p + "attention/layer_norm/gamma");
      L.ln2b = half_buf(p + "attention/layer_norm/beta");
      L.cq = weight(p + "attention/linear_0/weight");
      L.cq_b = half_buf(p + "attention/linear_0/bias");
      L.ckv = weight(p + "attention/linear_1/weight");
      L.ckv_b = half_buf(p + "attention/linear_1/bias");
      L.cout = weight(p + "attention/linear_2/weight");
      L.cout_b = half_buf(p + "attention/linear_2/bias");
      L.ln3g = half_buf(p + "ffn/layer_norm/gamma");
      L.ln3b = half_buf(p + "ffn/layer_norm/beta");
      L.fc1 = weight(p + "ffn/linear_0/weight");
      L.fc1_b = half_buf(p + "ffn/linear_0/bias");
      L.fc2 = weight(p + "ffn/linear_1/weight");
      L.fc2_b = half_buf(p + "ffn/linear_1/bias");
      L.ckv_buf = buf((size_t)Ap_ * 2 * D * 4);
      L.ck16 = buf((size_t)hp_.n_head * Ap_ * 64 * 2 + 64);  // (the window's keys / values, f16, head-major)
      L.cv16 = buf((size_t)hp_.n_head * Ap_ * 64 * 2 + 64);
      dec_.push_back(L);
    }
    dec_lng_ = half_buf("decoder/layer_norm/gamma");
    dec_lnb_ = half_buf("decoder/layer_norm/beta");
    emb_ = half_buf("decoder/embeddings/weight");
    dec_pos_ = half_buf("decoder/position_encodings/encodings");
    head_ = weight("decoder/embeddings/weight", Vp_);  // (tied: the output head, rows padded to a multiple of 16)
    // working buffers: the encoder's largest (3,008 rows of conv1, 1,504 rows x 5,120 for the FFN)
    const size_t maxMK = std::max({(size_t)T1p_ * K1p_, (size_t)Ap_ * 3 * D, (size_t)Ap_ * F, (size_t)T1p_ * D});
    // (twice: the second half holds the residual quantization of the two-pass encoder layers, ARC_WH_Q2)
    aq_ = buf(2 * maxMK);
    asc_ = buf(2 * maxMK / 32 * 4);
    absum_ = buf(2 * maxMK / 16 * 2);
    q2defs_ = " -DQ2_AQ=" + std::to_string(maxMK) + " -DQ2_SC=" + std::to_string(maxMK / 32) + " -DQ2_AB=" +
              std::to_string(maxMK / 16);
    aq2_ = view(aq_, maxMK, maxMK);
    asc2_ = view(asc_, maxMK / 32 * 4, maxMK / 32 * 4);
    absum2_ = view(absum_, maxMK / 16 * 2, maxMK / 16 * 2);
    mel_ = buf((size_t)hp_.n_mels * whisper::kWindowFrames * 4);
    col_ = buf(std::max((size_t)T1p_ * K1p_, (size_t)Ap_ * 3 * D) * 4);
    h1_ = buf((size_t)T1p_ * D * 4);
    x_ = buf((size_t)Ap_ * D * 4);
    hn_ = buf((size_t)Ap_ * D * 4);
    qkv_ = buf((size_t)Ap_ * 3 * D * 4);
    att_ = buf((size_t)Ap_ * D * 4);
    y_ = buf((size_t)Ap_ * D * 4);
    ff_ = buf((size_t)Ap_ * F * 4);
    q16_ = buf((size_t)hp_.n_head * Ap_ * 64 * 2 + 4096);
    k16_ = buf((size_t)hp_.n_head * Ap_ * 64 * 2 + 4096);
    v16_ = buf((size_t)hp_.n_head * Ap_ * 64 * 2 + 4096);
    // the decoder: a KV cache per sequence, row inputs, logits
    kc_ = buf((size_t)max_seq_ * hp_.dec_layers * hp_.n_text_ctx * D * 4);
    vc_ = buf((size_t)max_seq_ * hp_.dec_layers * hp_.n_text_ctx * D * 4);
    const int maxR = (hp_.n_text_ctx + 15) / 16 * 16;
    rtok_ = buf((size_t)maxR * 4);
    rseq_ = buf((size_t)maxR * 4);
    rpos_ = buf((size_t)maxR * 4);
    logits_ = buf((size_t)maxR * Vp_ * 4);
    {  // the split cross-attention's partial results (whisper.cl wh_cross_part)
      const size_t parts = (size_t)hp_.n_head * ((A + 191) / 192) * maxR;
      pm_ = buf(parts * 4);
      pl_ = buf(parts * 4);
      po_ = buf(parts * 64 * 4);
    }
    hmap_ = buf((size_t)hp_.n_head * 4);
    no_save_ = buf((size_t)hp_.n_head * 4);
    {
      std::vector<int> none(hp_.n_head, -1);
      clEnqueueWriteBuffer(cl_.queue, no_save_, CL_TRUE, 0, none.size() * 4, none.data(), 0, nullptr, nullptr);
    }
    lens_.assign(max_seq_, 0);
    if (npu) {
      npu_ = std::make_unique<WhisperNpuEncoder>(m);
      h16_ = buf((size_t)A * D * 2 + 64);
    }
    load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  ~WhisperEngine() override {
    clFinish(cl_.queue);
    for (auto& [k, v] : kc_map_) clReleaseKernel(v);
    for (cl_program p : progs_) clReleaseProgram(p);
    for (cl_mem b : mems_) clReleaseMemObject(b);
  }

  double load_ms = 0, npu_ms = 0;  // (npu_ms: the NPU encoder's time)
  size_t gpu_bytes = 0;
  const WhisperNpuEncoder* npu() const { return npu_.get(); }
  std::map<std::string, double> prof_ms;  // (ARC_PROF=1: GPU time per kernel name)

  // ---------------------------------------------------------------- Backend
  void encode(const float* mel) override {
    const int D = hp_.d, A = hp_.n_audio_ctx, T = whisper::kWindowFrames, C = hp_.n_mels;
    if (npu_) {  // the encoder on the NPU; its output here for the cross-attention keys / values
      const auto t0 = std::chrono::steady_clock::now();
      const uint16_t* h = npu_->encode(mel);
      npu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      clEnqueueWriteBuffer(cl_.queue, h16_, CL_FALSE, 0, (size_t)A * D * 2, h, 0, nullptr, nullptr);
      cl_kernel k = wk("wh_h2f");
      int d = D, a = A;
      set_args(k, 0, h16_, d, a, hn_);
      size_t g[2] = {(size_t)D, (size_t)Ap_};
      run("h2f", k, 2, g, nullptr);
      cross_kv();
      return;
    }
    clEnqueueWriteBuffer(cl_.queue, mel_, CL_FALSE, 0, (size_t)C * T * 4, mel, 0, nullptr, nullptr);
    {  // conv1 + GELU
      cl_kernel k = wk("wh_im2col1");
      int c = C, t = T, kp = K1p_;
      set_args(k, 0, mel_, c, t, kp, col_);
      size_t g[2] = {(size_t)K1p_, (size_t)T1p_};
      run("im2col1", k, 2, g, nullptr);
      quant(col_, K1p_, T1p_, T);
      gemm(conv1_, T1p_, h1_, D);
      bias_gelu(h1_, D, conv1_b_, T1p_, nullptr);
    }
    {  // conv2 + GELU + positions
      cl_kernel k = wk("wh_im2col2");
      int t = T, d = D, a = A;
      set_args(k, 0, h1_, t, d, a, col_);
      size_t g[2] = {(size_t)3 * D, (size_t)Ap_};
      run("im2col2", k, 2, g, nullptr);
      quant(col_, 3 * D, Ap_, A);
      gemm(conv2_, Ap_, x_, D);
      bias_gelu(x_, D, conv2_b_, A, enc_pos_);
    }
    // (checks: WH_DUMP_DIR=<folder> writes the residual stream after each layer, L<n>.bin, L-1: after the convolutions)
    auto dump = [&](int at) {
      const char* dir = getenv("WH_DUMP_DIR");
      if (!dir) return;
      std::vector<float> o((size_t)A * D);
      clEnqueueReadBuffer(cl_.queue, x_, CL_TRUE, 0, o.size() * 4, o.data(), 0, nullptr, nullptr);
      std::ofstream(std::string(dir) + "/L" + std::to_string(at) + ".bin", std::ios::binary).write((const char*)o.data(), o.size() * 4);
    };
    dump(-1);
    int li = 0;
    for (const EncL& L : enc_) {
      two_ = li >= q2_from_;
      layernorm(x_, D, A, L.ln1g, L.ln1b, hn_, Ap_);
      quant(hn_, D, Ap_, A);
      gemm(L.qkv, Ap_, qkv_, 3 * D);
      {
        cl_kernel k = wk("wh_qkv_split");
        int d = D, mp = Ap_;
        set_args(k, 0, qkv_, L.qkv_b, d, mp, q16_, k16_, v16_);
        size_t g[2] = {(size_t)D, (size_t)Ap_};
        run("qkv_split", k, 2, g, nullptr);
      }
      {
        static const std::string ft = getenv("ARC_WH_FT") ? std::string(" -DFT=") + getenv("ARC_WH_FT") : std::string();
        cl_kernel k = ok("vit_attn_rows", "-cl-intel-256-GRF-per-thread -DHD=64" + ft);
        int n = A, mp = Ap_, nh = hp_.n_head;
        set_args(k, 0, q16_, k16_, v16_, n, mp, nh, att_);
        static const int arsg = getenv("ARC_WH_ARSG") ? std::max(1, atoi(getenv("ARC_WH_ARSG"))) : 4;
        const size_t rows = (size_t)8 * arsg;
        size_t g[2] = {((size_t)A + rows - 1) / rows * rows, (size_t)hp_.n_head}, lw[2] = {rows, 1};
        run("enc_attn", k, 2, g, lw);
      }
      quant(att_, D, Ap_, A);
      gemm(L.out, Ap_, y_, D);
      bias_res(x_, y_, D, L.out_b, A);
      layernorm(x_, D, A, L.ln2g, L.ln2b, hn_, Ap_);
      quant(hn_, D, Ap_, A);
      gemm(L.fc1, Ap_, ff_, F_);
      bias_gelu(ff_, F_, L.fc1_b, Ap_, nullptr);
      quant(ff_, F_, Ap_, A);
      gemm(L.fc2, Ap_, y_, D);
      bias_res(x_, y_, D, L.fc2_b, A);
      dump(++li);
    }
    two_ = q2_from_ <= hp_.enc_layers;  // (the cross-attention keys / values from the last layer's output too)
    layernorm(x_, D, A, enc_lng_, enc_lnb_, hn_, Ap_);
    cross_kv();
  }
  // the decoder layers' cross-attention keys / values of this window, from the encoder output in hn_
  void cross_kv() {
    const int D = hp_.d, A = hp_.n_audio_ctx;
    quant(hn_, D, Ap_, A);
    for (DecL& L : dec_) {
      gemm(L.ckv, Ap_, L.ckv_buf, 2 * D);
      cl_kernel k = wk("wh_ckv_pack");
      int d = D, ap = Ap_;
      set_args(k, 0, L.ckv_buf, L.ckv_b, d, ap, L.ck16, L.cv16);
      size_t g[2] = {(size_t)D, (size_t)A};
      run("ckv_pack", k, 2, g, nullptr);
    }
    two_ = false;
    collect();
  }
  // the encoder output (after the final LayerNorm) of the last window, [n_audio_ctx][d] f32 (checks)
  std::vector<float> encoder_output() {
    std::vector<float> o((size_t)hp_.n_audio_ctx * hp_.d);
    clEnqueueReadBuffer(cl_.queue, hn_, CL_TRUE, 0, o.size() * 4, o.data(), 0, nullptr, nullptr);
    return o;
  }

  void reset(int n) override {
    if (n > max_seq_) throw std::runtime_error("whisper: more decoder sequences than the engine holds");
    n_seq_ = n;
    std::fill(lens_.begin(), lens_.end(), 0);
  }
  void copy(int dst, int src) override {
    if (dst == src) return;
    cl_kernel k = wk("wh_kv_copy");
    int d = dst, s = src, L = hp_.dec_layers, ctx = hp_.n_text_ctx, len = lens_[src], D = hp_.d;
    set_args(k, 0, d, s, L, ctx, len, D, kc_, vc_);
    size_t g[2] = {(size_t)D, (size_t)ctx * L};
    run("kv_copy", k, 2, g, nullptr);
    lens_[dst] = lens_[src];
  }
  void append(int s, const int* tok, int n, int rows, float* logits) override {
    std::vector<int> seq(n, s), pos(n);
    for (int i = 0; i < n; ++i) pos[i] = lens_[s] + i;
    forward(tok, seq.data(), pos.data(), n, rows, logits, nullptr, nullptr);
    lens_[s] += n;
  }
  void append_batch(const int* seqs, const int* tok, int k, float* logits, int n_vocab) override {
    (void)n_vocab;
    std::vector<int> pos(k);
    for (int i = 0; i < k; ++i) pos[i] = lens_[seqs[i]];
    forward(tok, seqs, pos.data(), k, k, logits, nullptr, nullptr);
    for (int i = 0; i < k; ++i) ++lens_[seqs[i]];
  }
  void cross_attention(const int* tok, int n, const std::vector<std::pair<int, int>>& heads, std::vector<float>& scores,
                       std::vector<float>& p) override {
    reset(1);
    std::vector<float> lg((size_t)n * hp_.n_vocab);
    std::vector<int> seq(n, 0), pos(n);
    for (int i = 0; i < n; ++i) pos[i] = i;
    forward(tok, seq.data(), pos.data(), n, n, lg.data(), &heads, &scores);
    lens_[0] = n;
    p.assign(n > 0 ? n - 1 : 0, 0.0f);
    std::vector<float> lp;
    for (int i = 0; i + 1 < n; ++i) {
      whisper::log_softmax(&lg[(size_t)i * hp_.n_vocab], hp_.n_vocab, lp);
      p[i] = std::exp(lp[tok[i + 1]]);
    }
  }

 private:
  struct GW {
    cl_mem wq = nullptr, wh = nullptr;
    int N = 0, K = 0;
  };
  struct EncL {
    cl_mem ln1g, ln1b, qkv_b, out_b, ln2g, ln2b, fc1_b, fc2_b;
    GW qkv, out, fc1, fc2;
  };
  struct DecL {
    cl_mem ln1g, ln1b, qkv_b, out_b, ln2g, ln2b, cq_b, ckv_b, cout_b, ln3g, ln3b, fc1_b, fc2_b, ckv_buf, ck16, cv16;
    GW qkv, out, cq, ckv, cout, fc1, fc2;
  };
  const whisper::Model& m_;
  whisper::Hparams hp_;
  int max_seq_, n_seq_ = 1, Ap_ = 0, T1p_ = 0, Vp_ = 0, F_ = 0, K1p_ = 512;
  ClRt cl_;
  std::string gemm_src_, ops_src_, wh_src_;
  std::map<std::string, cl_kernel> kc_map_;
  std::vector<cl_program> progs_;
  std::vector<cl_mem> mems_;
  std::vector<EncL> enc_;
  std::vector<DecL> dec_;
  GW conv1_, conv2_, head_;
  cl_mem conv1_b_, conv2_b_, enc_pos_, enc_lng_, enc_lnb_, dec_lng_, dec_lnb_, emb_, dec_pos_;
  std::unique_ptr<WhisperNpuEncoder> npu_;
  cl_mem h16_ = nullptr;
  cl_mem aq2_, asc2_, absum2_;
  std::string q2defs_;
  bool two_ = false;
  // encoder layers from this one on run their matrix products in two int8 passes (the value, then its rounding
  // residual): the last layers carry rows with very large activations that one pass rounds too coarsely
  const int q2_from_ = getenv("ARC_WH_Q2") ? atoi(getenv("ARC_WH_Q2")) : 1 << 30;
  cl_mem aq_, asc_, absum_, mel_, col_, h1_, x_, hn_, qkv_, att_, y_, ff_, q16_, k16_, v16_;
  cl_mem kc_, vc_, rtok_, rseq_, rpos_, logits_, hmap_, no_save_, pm_, pl_, po_;
  cl_mem save_ = nullptr;
  size_t save_bytes_ = 0;
  std::vector<int> lens_;
  std::vector<std::pair<std::string, cl_event>> events_;
  const bool profile_ = getenv("ARC_PROF") && atoi(getenv("ARC_PROF")) != 0;

  cl_mem buf(size_t bytes, const void* host = nullptr) {
    gpu_bytes += bytes;
    mems_.push_back(cl_.buffer(CL_MEM_READ_WRITE, bytes, host));
    return mems_.back();
  }
  cl_mem half_buf(const std::string& name) {
    const whisper::Ct2Var& v = m_.file->get(name);
    if (v.dtype != 4) throw std::runtime_error("whisper: " + name + " is not f16");
    return buf(v.bytes + 64, v.data);
  }
  // an f16 matrix [N][K] as the Q8_0 GEMM form, rows padded to Np with zeros
  GW weight(const std::string& name, int Np = 0) {
    const whisper::Ct2Var& v = m_.file->get(name);
    const int N = v.shape[0], K = (int)(v.count() / N);
    return upload_q8(name, v, N, K, Np ? Np : N, K, [&](int n, int k) { return whisper::h2f(((const uint16_t*)v.data)[(size_t)n * K + k]); });
  }
  // a convolution weight [N][C][3] as [N][C * 3] (K padded to Kp)
  GW weight_conv(const std::string& name, int Kp) {
    const whisper::Ct2Var& v = m_.file->get(name);
    const int N = v.shape[0], K = v.shape[1] * v.shape[2];
    return upload_q8(name, v, N, K, N, Kp, [&](int n, int k) { return whisper::h2f(((const uint16_t*)v.data)[(size_t)n * K + k]); });
  }
  template <typename F>
  GW upload_q8(const std::string& name, const whisper::Ct2Var& v, int N, int K, int Np, int Kp, F at) {
    if (v.dtype != 4) throw std::runtime_error("whisper: " + name + " is not f16");
    if (Np % 16 || Kp % 256) throw std::runtime_error("whisper: " + name + " does not fit the GEMM tiles");
    std::vector<float> f((size_t)Np * Kp, 0.0f);
#pragma omp parallel for
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k) f[(size_t)n * Kp + k] = at(n, k);
    const std::vector<uint8_t> q8 = quantize_rows_q8_0(f.data(), Np, Kp);
    RepackedWeight r;
    repack_rows(GGML_Q8_0, q8.data(), Kp, 0, Np, r);
    GW w;
    w.N = Np;
    w.K = Kp;
    w.wq = buf(r.wq.size() * 4, r.wq.data());
    w.wh = buf(r.wh.size() * 4, r.wh.data());
    return w;
  }

  cl_kernel get(const std::string& which, const std::string& defs, const char* name) {
    const std::string key = which + defs + "|" + name;
    auto it = kc_map_.find(key);
    if (it != kc_map_.end()) return it->second;
    const std::string& src = which == "gemm" ? gemm_src_ : which == "ops" ? ops_src_ : wh_src_;
    cl_program p = cl_.build(src, "-cl-std=CL3.0 -cl-mad-enable " + defs + (cl_.sg8 ? "" : " -DNO_SG8"));
    progs_.push_back(p);
    return kc_map_[key] = cl_.kernel(p, name);
  }
  cl_kernel wk(const char* name) { return get("whisper", "", name); }
  cl_kernel ok(const char* name, const std::string& defs) { return get("ops", "-DVIT " + defs, name); }

  void run(const char* name, cl_kernel k, int dims, const size_t* gws, const size_t* lws) {
    cl_event e = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(cl_.queue, k, dims, nullptr, gws, lws, 0, nullptr, profile_ ? &e : nullptr));
    if (profile_) events_.push_back({name, e});
  }
  void collect() {
    if (!profile_) return;
    clFinish(cl_.queue);
    for (auto& [n, e] : events_) {
      prof_ms[n] += ClRt::event_ms(e);
      clReleaseEvent(e);
    }
    events_.clear();
  }

  // rows of src [Mp][K] f32 -> the int8 activations (aq_ / asc_ / absum_, per-32 scales); rows >= valid as zeros
  // (two_: also the residual of the rounding, quantized the same way, for the GEMM's second pass)
  void quant(cl_mem src, int K, int Mp, int valid) {
    cl_kernel k = get("ops", two_ ? q2defs_ : std::string(), "swiglu_quant");
    int KK = K, MM = Mp, TT = valid, a32 = 1, fused = 1, gelu = 0;
    set_args(k, 0, src, KK, MM, TT, aq_, asc_, absum_, a32, fused, gelu);
    size_t g[2] = {256, (size_t)Mp}, l[2] = {256, 1};
    run("quant", k, 2, g, l);
  }
  // C[Mp][N] = act . W^T (the engine's GEMM launch: split-K beyond 8192, work-group rows by Mp)
  void gemm(const GW& w, int Mp, cl_mem C, int ldc) {
    cl_kernel k = get("gemm", "-cl-intel-256-GRF-per-thread -DQTYPE=8 -DACT32", "gemm");
    int M = Mp, N = w.N, K = w.K, lc = ldc, co = 0;
    const int nsb = K / 256, splits = K > 8192 ? (nsb + 16) / 17 : 1, rgroup = 4;
    int wgy = 1;
    while (wgy < 8 && (Mp / 16) % (wgy * 2) == 0) wgy *= 2;
    size_t g[2] = {(size_t)N * (Mp / 16 / wgy), (size_t)wgy}, l[2] = {16, (size_t)wgy};
    for (int pass = 0; pass < (two_ ? 2 : 1); ++pass)  // (two_: the residual's pass accumulates)
      for (int sp = 0; sp < splits; ++sp) {
        int s0 = nsb * sp / splits, s1 = nsb * (sp + 1) / splits, ac = sp == 0 && pass == 0 ? 0 : 1;
        if (pass == 0) set_args(k, 0, w.wq, w.wh, aq_, asc_, absum_, C, M, N, K, lc, co, ac, rgroup, s0, s1);
        else set_args(k, 0, w.wq, w.wh, aq2_, asc2_, absum2_, C, M, N, K, lc, co, ac, rgroup, s0, s1);
        run(Mp >= 256 ? "enc_gemm" : "dec_gemm", k, 2, g, l);
      }
  }
  cl_mem view(cl_mem b, size_t off, size_t bytes) {
    cl_buffer_region r{off, bytes};
    cl_int err;
    cl_mem v = clCreateSubBuffer(b, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &err);
    CL_CHECK(err);
    mems_.push_back(v);
    return v;
  }
  void bias_gelu(cl_mem y, int N, cl_mem b, int M, cl_mem pos) {
    cl_kernel k = wk("wh_bias_gelu");
    int n = N, wp = pos ? 1 : 0;
    cl_mem p = pos ? pos : b;
    set_args(k, 0, y, n, b, p, wp);
    size_t g[2] = {(size_t)N, (size_t)M};
    run("bias_gelu", k, 2, g, nullptr);
  }
  void bias(cl_mem y, int N, cl_mem b, int M) {
    cl_kernel k = wk("wh_bias");
    int n = N;
    set_args(k, 0, y, n, b);
    size_t g[2] = {(size_t)N, (size_t)M};
    run("bias", k, 2, g, nullptr);
  }
  void bias_res(cl_mem x, cl_mem y, int N, cl_mem b, int M) {
    cl_kernel k = wk("wh_bias_res");
    int n = N;
    set_args(k, 0, x, y, n, b);
    size_t g[2] = {(size_t)N, (size_t)M};
    run("bias_res", k, 2, g, nullptr);
  }
  void layernorm(cl_mem x, int D, int valid, cl_mem g, cl_mem b, cl_mem out, int Mp) {
    cl_kernel k = wk("wh_layernorm");
    int d = D, mv = valid;
    set_args(k, 0, x, d, mv, g, b, out);
    size_t gw[2] = {256, (size_t)Mp}, lw[2] = {256, 1};
    run("layernorm", k, 2, gw, lw);
  }

  // n decoder rows (token tok[r] of sequence seq[r] at position pos[r]); logits of the last `rows` rows to the host;
  // heads: also the alignment heads' cross-attention scores of every row into scores [heads][n][A]
  void forward(const int* tok, const int* seq, const int* pos, int n, int rows, float* logits,
               const std::vector<std::pair<int, int>>* heads, std::vector<float>* scores) {
    const int D = hp_.d, A = hp_.n_audio_ctx, L = hp_.dec_layers, ctx = hp_.n_text_ctx, F = F_, V = hp_.n_vocab;
    const int Mp = (n + 15) / 16 * 16;
    for (int i = 0; i < n; ++i)
      if (pos[i] >= ctx) throw std::runtime_error("whisper: more than " + std::to_string(ctx) + " text positions");
    clEnqueueWriteBuffer(cl_.queue, rtok_, CL_FALSE, 0, (size_t)n * 4, tok, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(cl_.queue, rseq_, CL_FALSE, 0, (size_t)n * 4, seq, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(cl_.queue, rpos_, CL_FALSE, 0, (size_t)n * 4, pos, 0, nullptr, nullptr);
    if (heads) {
      const size_t need = heads->size() * (size_t)n * A * 4;
      if (need > save_bytes_) {
        if (save_) {
          clFinish(cl_.queue);
          mems_.erase(std::find(mems_.begin(), mems_.end(), save_));
          clReleaseMemObject(save_);
        }
        save_ = buf(need);
        save_bytes_ = need;
      }
    }
    {
      cl_kernel k = wk("wh_embed");
      int nn = n, d = D;
      set_args(k, 0, rtok_, rpos_, nn, emb_, dec_pos_, d, x_);
      size_t g[2] = {(size_t)D, (size_t)Mp};
      run("embed", k, 2, g, nullptr);
    }
    for (int l = 0; l < L; ++l) {
      const DecL& W = dec_[l];
      layernorm(x_, D, n, W.ln1g, W.ln1b, hn_, Mp);
      quant(hn_, D, Mp, n);
      gemm(W.qkv, Mp, qkv_, 3 * D);
      {
        cl_kernel k = wk("wh_kv_store");
        int d = D, LL = L, c = ctx, ll = l;
        set_args(k, 0, qkv_, W.qkv_b, d, rseq_, rpos_, LL, c, ll, kc_, vc_);
        size_t g[2] = {(size_t)D, (size_t)n};
        run("kv_store", k, 2, g, nullptr);
      }
      {
        cl_kernel k = wk("wh_self_attn");
        int d = D, LL = L, c = ctx, ll = l;
        set_args(k, 0, qkv_, W.qkv_b, d, rseq_, rpos_, LL, c, ll, kc_, vc_, att_);
        size_t g[2] = {(size_t)64 * hp_.n_head, (size_t)n}, lw[2] = {64, 1};
        run("self_attn", k, 2, g, lw);
      }
      quant(att_, D, Mp, n);
      gemm(W.out, Mp, y_, D);
      bias_res(x_, y_, D, W.out_b, n);
      layernorm(x_, D, n, W.ln2g, W.ln2b, hn_, Mp);
      quant(hn_, D, Mp, n);
      gemm(W.cq, Mp, y_, D);
      bias(y_, D, W.cq_b, n);
      {
        cl_mem hm = no_save_;
        if (heads) {
          std::vector<int> map(hp_.n_head, -1);
          bool any = false;
          for (size_t i = 0; i < heads->size(); ++i)
            if ((*heads)[i].first == l) { map[(*heads)[i].second] = (int)i; any = true; }
          if (any) {
            clEnqueueWriteBuffer(cl_.queue, hmap_, CL_TRUE, 0, map.size() * 4, map.data(), 0, nullptr, nullptr);
            hm = hmap_;
          }
        }
        // (split over the positions: CP = 192 per work-group, then combined; whisper.cl wh_cross_part)
        const int S = (A + 191) / 192;
        cl_kernel k = wk("wh_cross_part");
        int d = D, nn = n, a = A, ap = Ap_, s = S;
        cl_mem sv = save_ ? save_ : att_;
        set_args(k, 0, y_, d, nn, W.ck16, W.cv16, a, ap, s, pm_, pl_, po_, hm, sv);
        size_t g[3] = {(size_t)256 * hp_.n_head, (size_t)S, (size_t)(n + 3) / 4}, lw[3] = {256, 1, 1};
        run("cross_attn", k, 3, g, lw);
        cl_kernel kc = wk("wh_cross_comb");
        set_args(kc, 0, d, nn, s, pm_, pl_, po_, att_);
        size_t g2[2] = {(size_t)64 * hp_.n_head, (size_t)n}, lw2[2] = {64, 1};
        run("cross_comb", kc, 2, g2, lw2);
      }
      quant(att_, D, Mp, n);
      gemm(W.cout, Mp, y_, D);
      bias_res(x_, y_, D, W.cout_b, n);
      layernorm(x_, D, n, W.ln3g, W.ln3b, hn_, Mp);
      quant(hn_, D, Mp, n);
      gemm(W.fc1, Mp, ff_, F);
      bias_gelu(ff_, F, W.fc1_b, n, nullptr);
      quant(ff_, F, Mp, n);
      gemm(W.fc2, Mp, y_, D);
      bias_res(x_, y_, D, W.fc2_b, n);
    }
    layernorm(x_, D, n, dec_lng_, dec_lnb_, hn_, Mp);
    quant(hn_, D, Mp, n);
    gemm(head_, Mp, logits_, Vp_);
    std::vector<float> all((size_t)rows * Vp_);
    clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, (size_t)(n - rows) * Vp_ * 4, all.size() * 4, all.data(), 0, nullptr, nullptr);
    for (int r = 0; r < rows; ++r) memcpy(logits + (size_t)r * V, &all[(size_t)r * Vp_], (size_t)V * 4);
    if (heads && scores) {
      scores->resize(heads->size() * (size_t)n * A);
      clEnqueueReadBuffer(cl_.queue, save_, CL_TRUE, 0, scores->size() * 4, scores->data(), 0, nullptr, nullptr);
    }
    collect();
  }
};

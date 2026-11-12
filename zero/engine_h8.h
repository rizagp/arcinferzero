// H8 prompt GEMMs (QUEUE 31.12; zero/kernels/h8.cl, gemm.cl QTYPE 110): a part of class Engine (zero/engine.h),
// included inside the class body like zero/xe2/engine_xe2.h, not a header of its own.
//
// The text model's prompt GEMMs with weights and activations rotated by a Walsh-Hadamard transform in blocks of up to
// 1024 values along K, both rounded to int8 with one scale per 256-superblock: the GEMM is then pure int8 dot
// products, one float multiply-add per superblock (DP4a on the iGPU, int8 DPAS with -DXMX on Xe2 cards; the same
// kernel and launch as the W8 GEMM). Accuracy simulated against f32 (tools/ref_forward REF_HAD, per-row scales, so
// coarser than these per-256 ones): Qwen3.8 27B KL 0.00057 (the engine's own level), Gemma 4 E4B 0.00198.
//
// The weights come from the W8 / Q8_0 forms the engine already makes (decode_w8 just in time, or resident), converted
// in place once per decode (h8_weights); their H8 superblock scales go to a header buffer of their own (a resident
// weight's in h8_res_, a staged one's per role in stage_hh_). The activations stay as the fused norm / quantization
// kernels write them (aq_ / asc_, per-32 or per-256 scales) and are rotated into h8_aq_ / h8_asc_ before each GEMM
// (h8_act: 14 % of the GEMMs' time on a B60, 2026-11-06).
//
// H8R, per-column and per-token scales (gemm.cl QTYPE 111, kernels h8r_* of h8.cl, the default within ARC_H8=1;
// ARC_H8_SB=1 keeps the per-256 ones, QTYPE 110): a weight's column scales are computed once from its first decode
// (h8_col_, N floats) and each token's scale in the rotation's one pass (h8_asc_[token]), the rotations in registers,
// and a GEMM's dot products over the whole of K are one integer sum. See h8_float for the activations' source.
//
// Off unless ARC_H8=1 until it is measured; never with ARC_NPU (shared weight buffers); never with a two-pass model (q2mm_: Gemma 4 26B / 12B, Qwen3 ASR's text
// model), never for a mixture of experts' expert GEMMs (their own kernels) or the vision / audio encoders (only the
// text model's named matrix roles). On a split model each shard rotates in blocks dividing its own K: the rotation is
// per matrix input, so the shards' parts still sum to the product. ARC_H8_HB=256 / 512: smaller blocks (tests).

  bool h8_ = false;
  std::string h8_src_, x8_src_;
  cl_mem h8_aq_ = nullptr, h8_asc_ = nullptr;   // (one GEMM input rotated: int8 [M][K], per-256 scales [K / 256][M])
  cl_mem stage_hh_[kRoles] = {};                // (the H8 scales of the weight staged per role)
  bool stage_h8_[kRoles] = {};                  // (the staged weight is in its rotated H8 form, not W8: the 140T preset mixes both)
  size_t stage_hh_bytes_[kRoles] = {};
  std::map<const GpuWeight*, cl_mem> h8_res_;   // (a resident weight converted in place: its H8 scales)

  static bool h8_env() {
    static const bool on = getenv("ARC_H8") && atoi(getenv("ARC_H8")) != 0;
    return on;
  }
  static bool h8_row() {  // (one scale per column and per token; ARC_H8_SB=1: per 256 values)
    static const bool sb = getenv("ARC_H8_SB") && atoi(getenv("ARC_H8_SB")) != 0;
    return !sb;
  }
  static int h8_qtype() { return h8_row() ? 111 : 110; }
  // H8R rotates a Qwen model's activations as they are before their first rounding: quant_store16 keeps a float copy
  // after the int8 rows of aq_, in the four kernels of build_kernels (rmsnorm_quant, swiglu_quant, gated_norm_quant,
  // attn_gate_quant; not the encoders', whose aq is a buffer of their own). Their int8 rows stay for the GEMMs H8 does
  // not cover and the generation path. Gemma and Flash Next have quantization kernels of their own and rotate the
  // int8 form. ARC_H8_INT=1: the int8 form here too.
  std::string h8f_defs_;
  size_t h8f_off_ = 0;
  bool h8_float() const {
    static const bool i8 = getenv("ARC_H8_INT") && atoi(getenv("ARC_H8_INT")) != 0;
    return h8_row() && !i8 && !hp.hc && !hp.gemma;
  }
  // (q2_init has run: q2mm_ known)
  void h8_init() {
    // (never with the NPU: its shares copy a weight (N.gate = L.gate) over the same buffers, so the in-place conversion
    //  ran twice on the shared rows, and the NPU read its own rows as W8 after they had become H8; kernel audit 2026-11-03)
    // (p140_: the Arc 140T preset; h8_auto_: the default on discrete cards for Qwen dense models; zero/xe2/engine_xe2.h)
    h8_ = (h8_env() || p140_ || h8_auto_) && !q2mm_ && !npu_wanted();
    if (h8_ && h8_float()) {
      h8f_off_ = (size_t)max_chunk * aq_max_k();
      h8f_defs_ = " -DH8F=" + std::to_string(h8f_off_);
    }
    if (h8_env() && q2mm_ && !opt.dry)
      zlog::eprintf("inferzero: ARC_H8 is not for a two-pass model: its prompt GEMMs stay as they are\n");
    if (h8_env() && !q2mm_ && npu_wanted() && !opt.dry)
      zlog::eprintf("inferzero: ARC_H8 is off with ARC_NPU (both convert the same weights)\n");
    // (once per process: a split model's shards each init; bench/gemm_sweep.ps1 reads this line)
    static std::once_flag said;
    if (h8_ && !opt.dry) std::call_once(said, [] {
      zlog::eprintf("inferzero: H8 on for the prompt GEMMs (Hadamard-rotated int8, one scale per %s)\n",
              h8_row() ? "column and token" : "256 values");
    });
    if (h8_ && h8f_off_ && !opt.dry) { static std::once_flag f; std::call_once(f, [] { zlog::eprintf("inferzero: H8 rotates the activations from f32\n"); }); }
  }
  // the rotation block for a row length K: the largest power of two up to 1024 (ARC_H8_HB) dividing it; K is a whole
  // number of 256-superblocks, so at least 256
  static int h8_block(int K) {
    static const int cap = [] {
      const int v = getenv("ARC_H8_HB") ? atoi(getenv("ARC_H8_HB")) : 1024;
      return v == 256 || v == 512 ? v : 1024;
    }();
    int hb = 256;
    while (hb * 2 <= cap && K % (hb * 2) == 0) hb *= 2;
    return hb;
  }
  bool h8_ok(const GpuWeight& w) const {
    return h8_ && w.role >= 0 && w.role <= 10 &&  // (weight_role's named matrices of the text model)
           (w.gemm_qtype == (int)GGML_Q8_0 || (w.gemm_qtype >= 100 && w.gemm_qtype <= 103)) && w.K % 256 == 0 && w.N % 16 == 0 &&
           (!h8_row() || (w.K / h8_block(w.K) <= 32 && w.K <= 130000));  // (H8R: red[32]; the sum over K within int32)
  }
  // its buffers with the others (alloc_buffers), so the memory plan counts them; a dry run too
  void h8_reserve(size_t M, int maxK) {
    if (!h8_) return;
    h8_aq_ = buf(M * (size_t)maxK);
    h8_asc_ = buf(M * (size_t)(maxK / 256) * 4);
    if (h8_row()) return;
    for (int r = 0; r < kRoles; ++r)  // (a role's H8 scales: N * K / 256 floats, an eighth of its W8 header at most)
      if (stage_need_h_[r]) stage_hh_[r] = buf(stage_hh_bytes_[r] = stage_need_h_[r] / 8 + 64);
  }
  cl_kernel h8_kernel(int hb, int src, const char* name) {
    return kernel("h8", "-DHB=" + std::to_string(hb) + (src ? " -DSRC=" + std::to_string(src) : std::string()), name);
  }
  // (H8R: 64 floats x 16 lanes in registers; ARC_H8_D=1: with urakozz's random signs, which gained nothing on our models)
  cl_kernel h8r_kernel(int hb, int src, const char* name, const std::string& extra = std::string()) {
    static const bool d = getenv("ARC_H8_D") && atoi(getenv("ARC_H8_D")) != 0;
    return kernel("h8", std::string("-cl-intel-256-GRF-per-thread ") + (d ? "-DH8_D " : "") + "-DHB=" + std::to_string(hb) +
                            (src ? " -DSRC=" + std::to_string(src) : std::string()) + extra, name);
  }
  // H8R: a weight's column scales (N floats), computed once from its decoded form, then the weight rotated in place
  std::map<const GpuWeight*, cl_mem> h8_col_;
  cl_mem h8r_weights(const GpuWeight& w, cl_mem wq, cl_mem wh) {
    const int hb = h8_block(w.K);
    int K = w.K;
    size_t gws[2] = {(size_t)w.N / 16 * 256, (size_t)(w.K / hb)}, lws[2] = {256, 1};
    auto it = h8_col_.find(&w);
    if (it == h8_col_.end()) {
      const std::vector<uint32_t> zero((size_t)w.N, 0u);
      cl_mem col = buf(zero.size() * 4, zero.data());
      cl_kernel km = h8r_kernel(hb, w.gemm_qtype, "h8r_colmax");
      set_args(km, 0, wq, wh, col, K);
      run("h8_colmax", km, 2, gws, lws);
      it = h8_col_.emplace(&w, col).first;
    }
    cl_kernel k = h8r_kernel(hb, w.gemm_qtype, "h8r_weights");
    set_args(k, 0, wq, wh, it->second, K);
    run("h8_weights", k, 2, gws, lws);
    return it->second;
  }
  // H8R straight from a staged weight's raw rows (kernels/decode.cl h8r_decode): decode_w8 + h8r_weights in one pass,
  // the W8 copy neither written nor read again (a B60's 27B 4K prompt: decode_w8 189 ms + h8_weights 372 ms). The
  // same values bit for bit; for the types decode_w8 stores without requantizing and with int8 scales, rotation
  // blocks of 512 / 1024, once the column scales are known (the first decode keeps the two passes). ARC_H8_FUSE=0: off.
  bool h8_dec_ok(const GpuWeight& w) const {
    static const bool on = !getenv("ARC_H8_FUSE") || atoi(getenv("ARC_H8_FUSE")) != 0;
    return on && w.raw && w.gemm_qtype >= 100 && w.gemm_qtype <= 103 && !w8_requant(w.type) && w8_variant(w.type) != 4 &&
           w.type != GGML_IQ1_M && w.type != GGML_IQ1_S && h8_block(w.K) >= 512 && w.N % 16 == 0;
  }
  void h8r_decode(const GpuWeight& w, cl_mem wq, cl_mem col) {
    static const bool d = getenv("ARC_H8_D") && atoi(getenv("ARC_H8_D")) != 0;
    const int hb = h8_block(w.K);
    cl_kernel k = kernel("decode", std::string("-cl-intel-256-GRF-per-thread -DH8RDEC") + (d ? " -DH8_D" : "") + " -DQT=" +
                                       std::to_string(w.type) + " -DHB=" + std::to_string(hb) + " -DSRC=" +
                                       std::to_string(w.gemm_qtype), "h8r_decode");
    int N = w.N, K = w.K;
    set_args(k, 0, w.raw, N, K, tabs_, wq, col);
    size_t gws[2] = {(size_t)N / 16 * 256, (size_t)(K / hb)}, lws[2] = {256, 1};
    run("h8_decode", k, 2, gws, lws);
  }
  // W8 / Q8_0 weights (wq, wh) into H8 in place, their scales into hh
  void h8_weights(const GpuWeight& w, cl_mem wq, cl_mem wh, cl_mem hh) {
    const int hb = h8_block(w.K);
    cl_kernel k = h8_kernel(hb, w.gemm_qtype, "h8_weights");
    int N = w.N, K = w.K;
    set_args(k, 0, wq, wh, hh, N, K);
    size_t gws[2] = {(size_t)N * (hb / 4), (size_t)(K / hb)}, lws[2] = {(size_t)hb / 4, 1};
    run("h8_weights", k, 2, gws, lws);
  }
  // GEMMs in a row on the same input (q / k / v, gate / up, qkv / z) rotate it once (QUEUE 33.3): the caller holds an
  // H8Share around them, nothing in its scope writes aq_ / asc_, and a later GEMM with the same rows, K and scale
  // form finds h8_aq_ / h8_asc_ as the first one left them. The same kernel on the same input: bit-identical.
  // ARC_H8_REUSE=0 rotates before each GEMM as before.
  int h8_share_ = 0, h8_last_M_ = -1, h8_last_K_ = -1, h8_last_a32_ = -1;
  cl_mem h8_last_aq_ = nullptr;  // (the input it was rotated from: a two-pass model swaps aq_)
  static bool h8_reuse_env() {
    static const bool on = !getenv("ARC_H8_REUSE") || atoi(getenv("ARC_H8_REUSE")) != 0;
    return on;
  }
  struct H8Share {
    Engine* e;
    explicit H8Share(Engine* en) : e(en) {
      e->h8_share_ = h8_reuse_env() ? 1 : 0;
      e->h8_last_K_ = -1;
    }
    ~H8Share() {
      e->h8_share_ = 0;
      e->h8_last_K_ = -1;
    }
    H8Share(const H8Share&) = delete;
    H8Share& operator=(const H8Share&) = delete;
  };
  // The FFN's down GEMM in H8 from f32 (QUEUE 33.3): when the up GEMM's epilogue wrote silu(gate) * up to big_
  // ([rows][F] f32), those are the values swiglu_quant would only copy to aq_ + H8F (fused: v = the GEMM's value, the
  // same row stride F), and an H8 GEMM reads neither aq_'s int8 rows nor asc_ / absum_ (no weight minimum): the
  // rotation reads big_ itself and swiglu_quant is left out. Bit-identical; ARC_H8_SWI=0 as before.
  // h8_takes: matmul1's prompt path takes the H8 GEMM for w (its first branch past the GEMVs)
  bool h8_takes(const GpuWeight& w, int Mp) const {
    return cur_T_ > 8 && !q2mm_ && h8_ok(w) && (!p140_ || h8_env() || p140_take(w, Mp));
  }
  bool h8_swi_ok(const GpuWeight& down, int Mp) const {
    static const bool on = !getenv("ARC_H8_SWI") || atoi(getenv("ARC_H8_SWI")) != 0;
    return on && h8_row() && !h8f_defs_.empty() && h8_takes(down, Mp);
  }
  cl_mem h8_fsrc_ = nullptr;  // (set: the next rotations read f32 rows of K from here, not aq_ + H8F; H8Src)
  struct H8Src {
    Engine* e;
    H8Src(Engine* en, cl_mem src) : e(en) { e->h8_fsrc_ = src; }
    ~H8Src() { e->h8_fsrc_ = nullptr; }
    H8Src(const H8Src&) = delete;
    H8Src& operator=(const H8Src&) = delete;
  };
  // this GEMM's input (aq_ / asc_, per-32 scales when act32) rotated into h8_aq_ / h8_asc_
  void h8_act(int Mp, int K, bool act32) {
    const cl_mem src = h8_fsrc_ ? h8_fsrc_ : aq_;
    if (h8_share_) {
      if (h8_last_K_ == K && h8_last_M_ == Mp && h8_last_a32_ == (act32 ? 1 : 0) && h8_last_aq_ == src) return;
      h8_last_K_ = K, h8_last_M_ = Mp, h8_last_a32_ = act32 ? 1 : 0, h8_last_aq_ = src;
    }
    const int hb = h8_block(K);
    int a32 = act32 ? 1 : 0, M = Mp, KK = K;
    if (h8_row()) {
      cl_kernel k = h8r_kernel(hb, 0, "h8r_act", h8_fsrc_ ? std::string(" -DH8F=0") : h8f_defs_);
      set_args(k, 0, src, asc_, a32, h8_aq_, h8_asc_, M, KK);
      const size_t wg = 16 * (size_t)(K / hb);
      size_t gws[1] = {wg * Mp}, lws[1] = {wg};
      run("h8_act", k, 1, gws, lws);
      return;
    }
    if (h8_fsrc_) throw std::logic_error("h8_act: an f32 source without H8R");
    cl_kernel k = h8_kernel(hb, 0, "h8_act");
    set_args(k, 0, aq_, asc_, a32, h8_aq_, h8_asc_, M, KK);
    size_t gws[3] = {(size_t)hb / 4, (size_t)Mp, (size_t)(K / hb)}, lws[3] = {(size_t)hb / 4, 1, 1};
    run("h8_act", k, 3, gws, lws);
  }
  // the H8 form of a weight for this GEMM: resident ones converted once, staged ones after each decode (stage_owner_
  // stays the weight, so a layer's next chunks find it converted)
  void h8_prepare(const GpuWeight& w, cl_mem& wq, cl_mem& hh) {
    if (w.wq && h8_row()) {
      auto it = h8_col_.find(&w);
      hh = it != h8_col_.end() ? it->second : h8r_weights(w, w.wq, w.wh);
      wq = w.wq;
      return;
    }
    if (h8_row()) {
      wq = stage_q_[w.role];
      if (stage_owner_[w.role] != &w || !stage_h8_[w.role]) {
        auto it = h8_col_.find(&w);
        if (it != h8_col_.end() && h8_dec_ok(w)) {
          h8r_decode(w, wq, it->second);  // (decode + rotation in one pass, bit-identical)
          hh = it->second;
        } else {
          decode_w8(w, wq, stage_h_[w.role]);
          hh = h8r_weights(w, wq, stage_h_[w.role]);
        }
        stage_owner_[w.role] = &w;
        stage_h8_[w.role] = true;
      } else {
        hh = h8_col_.at(&w);
      }
      return;
    }
    if (w.wq) {
      auto it = h8_res_.find(&w);
      if (it == h8_res_.end()) {
        cl_mem h = buf((size_t)w.N * (w.K / 256) * 4);
        h8_weights(w, w.wq, w.wh, h);
        it = h8_res_.emplace(&w, h).first;
      }
      wq = w.wq;
      hh = it->second;
      return;
    }
    const size_t need = (size_t)w.N * (w.K / 256) * 4;
    if (stage_hh_bytes_[w.role] < need) {  // (a matrix made after alloc_buffers: the encoders', MTP)
      stage_hh_[w.role] = buf(stage_hh_bytes_[w.role] = need);
    }
    wq = stage_q_[w.role];
    hh = stage_hh_[w.role];
    if (stage_owner_[w.role] != &w || !stage_h8_[w.role]) {
      decode_w8(w, wq, stage_h_[w.role]);
      h8_weights(w, wq, stage_h_[w.role], hh);
      stage_owner_[w.role] = &w;
      stage_h8_[w.role] = true;
    }
  }
  // the split-K pieces of an H8 GEMM: ~17 superblocks each for rows longer than 8192 (ARC_GEMM_SPLIT), one with the
  // fused SwiGLU (silu(gate) * (u1 + u2) is not silu(gate) * u1 + u2). Each piece is rounded on its own before it is
  // added to C, so the pieces decide the result's bits: gemm_xw (zero/xe2/engine_xw.h) takes them from here too
  static int h8_splits(int K, int a) {
    static const int sk_sb = getenv("ARC_GEMM_SPLIT") ? std::max(0, atoi(getenv("ARC_GEMM_SPLIT"))) : 17;
    const int nsb = K / 256;
    return a != 2 && K > 8192 && sk_sb > 0 ? (nsb + sk_sb - 1) / sk_sb : 1;
  }
  // the GEMM (matmul1's prompt path) in H8: the same grid, split-K and epilogues as the W8 GEMM
  void h8_matmul(const char* name, const GpuWeight& w, int Mp, cl_mem C, int ldc, int c_off, int a, bool act32) {
    cl_mem wq = nullptr, hh = nullptr;
    h8_prepare(w, wq, hh);
    h8_act(Mp, w.K, act32);
    // (the 140T preset: 32 tokens, 2 column groups, 128 registers on XMX)
    const bool pt = p140_take(w, Mp) && !h8_env();
    // (xw_: wider XMX tiles, the same results, by default in the 140T preset and with h8_auto_ (discrete cards), elsewhere ARC_GEMM_XW=1; zero/xe2/engine_xw.h. On 8-lane GPUs the 140T preset's
    //  chunks (pt) too: gemm_xw8 instead of gemm_x8. A shape it does not take: below)
    if (xw_ && (cl_.sg8 || !pt) && xw_matmul(name, wq, hh, Mp, w.N, w.K, C, ldc, c_off, a)) return;
    const int cg = pt ? 2 : h8_cg_for(w.N);
    cl_kernel k = pt ? gemm_kernel(h8_qtype(), false, 1, 2, 32, 1) : gemm_kernel(h8_qtype(), false, -1, cg);
    int M = Mp, N = w.N, K = w.K;
    static const int rg_env = getenv("ARC_GEMM_RG") ? std::max(1, atoi(getenv("ARC_GEMM_RG"))) : 4;
    static const int wgy_max = getenv("ARC_GEMM_WGY") ? atoi(getenv("ARC_GEMM_WGY")) : 8;
    const int tm = pt ? 32 : gemm_tm();
    if (Mp % tm || N % 16 || K % 256)
      throw std::runtime_error(std::string("gemm ") + name + ": shape not a multiple of " + std::to_string(tm) + " x 16 x 256");
    const int nsb = K / 256, splits = h8_splits(K, a), rgroup = rg_env;
    int wgy = 1;
    while (wgy < wgy_max && (Mp / tm) % (wgy * 2) == 0) wgy *= 2;
    size_t gws[2] = {(size_t)N * (Mp / tm / wgy), (size_t)wgy}, lws[2] = {16 * (size_t)cg, (size_t)wgy};
    for (int sp = 0; sp < splits; ++sp) {
      int s0 = nsb * sp / splits, s1 = nsb * (sp + 1) / splits, ac = sp == 0 ? a : 1;
      set_args(k, 0, wq, hh, h8_aq_, h8_asc_, absum_, C, M, N, K, ldc, c_off, ac, rgroup, s0, s1);
      run(name, k, 2, gws, lws);
    }
  }

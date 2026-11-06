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
// and a GEMM's dot products over the whole of K are one integer sum. ARC_H8_NOD=1: without the random signs.
//
// Off unless ARC_H8=1 until it is measured; never with ARC_NPU (shared weight buffers); never with a two-pass model (q2mm_: Gemma 4 26B / 12B, Qwen3 ASR's text
// model), never for a mixture of experts' expert GEMMs (their own kernels) or the vision / audio encoders (only the
// text model's named matrix roles). On a split model each shard rotates in blocks dividing its own K: the rotation is
// per matrix input, so the shards' parts still sum to the product. ARC_H8_HB=256 / 512: smaller blocks (tests).

  bool h8_ = false;
  std::string h8_src_;
  cl_mem h8_aq_ = nullptr, h8_asc_ = nullptr;   // (one GEMM input rotated: int8 [M][K], per-256 scales [K / 256][M])
  cl_mem stage_hh_[kRoles] = {};                // (the H8 scales of the weight staged per role)
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
  // (q2_init has run: q2mm_ known)
  void h8_init() {
    // (never with the NPU: its shares copy a weight (N.gate = L.gate) over the same buffers, so the in-place conversion
    //  ran twice on the shared rows, and the NPU read its own rows as W8 after they had become H8; kernel audit 2026-11-03)
    h8_ = h8_env() && !q2mm_ && !npu_wanted();
    if (h8_env() && q2mm_ && !opt.dry)
      fprintf(stderr, "arcinferzero: ARC_H8 is not for a two-pass model: its prompt GEMMs stay as they are\n");
    if (h8_env() && !q2mm_ && npu_wanted() && !opt.dry)
      fprintf(stderr, "arcinferzero: ARC_H8 is off with ARC_NPU (both convert the same weights)\n");
    // (once per process: a split model's shards each init; bench/gemm_sweep.ps1 reads this line)
    static std::once_flag said;
    if (h8_ && !opt.dry) std::call_once(said, [] {
      fprintf(stderr, "arcinferzero: H8 on for the prompt GEMMs (Hadamard-rotated int8, one scale per %s)\n",
              h8_row() ? "column and token" : "256 values");
    });
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
  // (H8R: 64 floats x 16 lanes in registers; ARC_H8_NOD=1: without the signs)
  cl_kernel h8r_kernel(int hb, int src, const char* name) {
    static const bool nod = getenv("ARC_H8_NOD") && atoi(getenv("ARC_H8_NOD")) != 0;
    return kernel("h8", std::string("-cl-intel-256-GRF-per-thread ") + (nod ? "" : "-DH8_D ") + "-DHB=" + std::to_string(hb) +
                            (src ? " -DSRC=" + std::to_string(src) : std::string()), name);
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
  // W8 / Q8_0 weights (wq, wh) into H8 in place, their scales into hh
  void h8_weights(const GpuWeight& w, cl_mem wq, cl_mem wh, cl_mem hh) {
    const int hb = h8_block(w.K);
    cl_kernel k = h8_kernel(hb, w.gemm_qtype, "h8_weights");
    int N = w.N, K = w.K;
    set_args(k, 0, wq, wh, hh, N, K);
    size_t gws[2] = {(size_t)N * (hb / 4), (size_t)(K / hb)}, lws[2] = {(size_t)hb / 4, 1};
    run("h8_weights", k, 2, gws, lws);
  }
  // this GEMM's input (aq_ / asc_, per-32 scales when act32) rotated into h8_aq_ / h8_asc_
  void h8_act(int Mp, int K, bool act32) {
    const int hb = h8_block(K);
    int a32 = act32 ? 1 : 0, M = Mp, KK = K;
    if (h8_row()) {
      cl_kernel k = h8r_kernel(hb, 0, "h8r_act");
      set_args(k, 0, aq_, asc_, a32, h8_aq_, h8_asc_, M, KK);
      const size_t wg = 16 * (size_t)(K / hb);
      size_t gws[1] = {wg * Mp}, lws[1] = {wg};
      run("h8_act", k, 1, gws, lws);
      return;
    }
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
      if (stage_owner_[w.role] != &w) {
        decode_w8(w, wq, stage_h_[w.role]);
        hh = h8r_weights(w, wq, stage_h_[w.role]);
        stage_owner_[w.role] = &w;
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
    if (stage_owner_[w.role] != &w) {
      decode_w8(w, wq, stage_h_[w.role]);
      h8_weights(w, wq, stage_h_[w.role], hh);
      stage_owner_[w.role] = &w;
    }
  }
  // the GEMM (matmul1's prompt path) in H8: the same grid, split-K and epilogues as the W8 GEMM
  void h8_matmul(const char* name, const GpuWeight& w, int Mp, cl_mem C, int ldc, int c_off, int a, bool act32) {
    cl_mem wq = nullptr, hh = nullptr;
    h8_prepare(w, wq, hh);
    h8_act(Mp, w.K, act32);
    const int cg = xe2_cg_for(w.N);
    cl_kernel k = gemm_kernel(h8_qtype(), false, -1, cg);
    int M = Mp, N = w.N, K = w.K;
    static const int rg_env = getenv("ARC_GEMM_RG") ? std::max(1, atoi(getenv("ARC_GEMM_RG"))) : 4;
    static const int wgy_max = getenv("ARC_GEMM_WGY") ? atoi(getenv("ARC_GEMM_WGY")) : 8;
    static const int sk_sb = getenv("ARC_GEMM_SPLIT") ? std::max(0, atoi(getenv("ARC_GEMM_SPLIT"))) : 17;
    const int tm = gemm_tm();
    if (Mp % tm || N % 16 || K % 256)
      throw std::runtime_error(std::string("gemm ") + name + ": shape not a multiple of " + std::to_string(tm) + " x 16 x 256");
    const int nsb = K / 256, splits = a != 2 && K > 8192 && sk_sb > 0 ? (nsb + sk_sb - 1) / sk_sb : 1, rgroup = rg_env;
    int wgy = 1;
    while (wgy < wgy_max && (Mp / tm) % (wgy * 2) == 0) wgy *= 2;
    size_t gws[2] = {(size_t)N * (Mp / tm / wgy), (size_t)wgy}, lws[2] = {16 * (size_t)cg, (size_t)wgy};
    for (int sp = 0; sp < splits; ++sp) {
      int s0 = nsb * sp / splits, s1 = nsb * (sp + 1) / splits, ac = sp == 0 ? a : 1;
      set_args(k, 0, wq, hh, h8_aq_, h8_asc_, absum_, C, M, N, K, ldc, c_off, ac, rgroup, s0, s1);
      run(name, k, 2, gws, lws);
    }
  }

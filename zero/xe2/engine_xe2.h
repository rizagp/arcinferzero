// Tuning switches for GPUs with XMX (Xe2 / Xe3: Arc B580, Pro B60), zero/xe2/: A/B settings that only the owners of
// these cards can measure (docs/xe2-tuning.md). A part of class Engine (zero/engine.h), included inside the class
// body like zero/mgpu/engine_tp_shard.h: not a header of its own.
// On only for discrete GPUs with XMX (one card or the multi-GPU shards). Off on every iGPU, with XMX or not (Xe-LPG,
// Lunar / Arrow Lake): there every kernel is built and launched exactly as without this file (xe2_ stays false, so
// the GEMM keeps 256 registers per thread, one column group per work-group, the GEMV its default register file).
// ARC_XMX_TRY turns it on anywhere, for the tests of the switches themselves.

  bool xe2_ = false;  // (set before the XMX check: the GPU offers XMX, or ARC_XMX_TRY)

  // the dense prompt GEMM with the default register file (128) instead of 256 registers per thread: more threads per
  // core; on a Pro B60 the 27B's prompt went 450 -> 617 tok/s (issue #4, 2026-11-02). ARC_GEMM_GRF=256 as before.
  // (On the 125H iGPU 128 made the 35B's prompt 3.4x slower: xe2_ is off there, which keeps 256.)
  std::string xe2_gemm_grf() const {
    static const bool grf256 = [] {
      const char* e = getenv("ARC_GEMM_GRF");
      return e != nullptr && std::string(e) == "256";
    }();
    return xe2_ && !grf256 ? std::string() : std::string("-cl-intel-256-GRF-per-thread ");
  }
  // ARC_MOE_GRF=128: the same for the MoE expert GEMMs (256 until measured on a B-series card: their 128 build is the
  // one that spilled on the iGPU)
  std::string xe2_moe_grf() const {
    static const bool grf128 = [] {
      const char* e = getenv("ARC_MOE_GRF");
      return e != nullptr && std::string(e) == "128";
    }();
    return xe2_ && grf128 ? std::string() : std::string("-cl-intel-256-GRF-per-thread ");
  }
  // ARC_GEMM_CG=2 / 4: 16-column groups per work-group of the prompt GEMM (a matrix whose column groups it does not
  // divide keeps 1)
  int xe2_cg_for(int N) const {
    static const int cg = [] {
      const int c = getenv("ARC_GEMM_CG") ? atoi(getenv("ARC_GEMM_CG")) : 1;
      if (c == 1 || c == 2 || c == 4) return c;
      fprintf(stderr, "arcinferzero: ARC_GEMM_CG=%d not supported (1, 2 or 4): 1\n", c);
      return 1;
    }();
    return xe2_ && (N / 16) % cg == 0 ? cg : 1;
  }
  static std::string xe2_cg_defs(int cg) { return cg != 1 ? " -DCG=" + std::to_string(cg) : std::string(); }
  // ARC_GEMV_GRF=256: the single-token GEMV with the large register file
  std::string xe2_gemv_grf() const {
    static const std::string g = grf_env("ARC_GEMV_GRF");
    return xe2_ ? g : std::string();
  }
  // ARC_GEMV_MT_GRF=128: the 2-8 token GEMV (gemv_mt / gemv_mt_v: MTP and DFlash2 verification, batched slots) with
  // the default register file instead of 256 registers per thread (256 everywhere else, and always on the iGPUs)
  std::string xe2_gemv_mt_grf() const {
    static const bool g128 = getenv("ARC_GEMV_MT_GRF") && std::string(getenv("ARC_GEMV_MT_GRF")) == "128";
    return xe2_ && g128 ? std::string() : std::string("-cl-intel-256-GRF-per-thread ");
  }

  // ---- the f16 XMX prompt GEMM (kernels/gemm_x16.cl) for the W8 layouts (the 27B's IQ3_S, decoded just in time).
  // Off until measured on a B60: ARC_X16=1 on a discrete GPU with XMX. ARC_X16_TRY=1 anywhere, DPAS emulated where
  // the GPU has none (tests of the kernel's layouts on this laptop). Taken only once its results match the DP4a
  // kernel's on random data within f16 rounding (not bit for bit: f16 weights, f32 sums in another order).
  // ARC_X16_TM=8 / 16 / 32: tokens per sub-group (default 32); ARC_X16_GRF=128: the default register file (256 else)
  bool x16_ = false, x16_emu_ = false;
  std::string x16_src_;
  cl_mem x16_ah_ = nullptr, x16_rs_ = nullptr;  // (the f16 activations and each row's scale, grown on use)
  size_t x16_ah_bytes_ = 0, x16_rs_bytes_ = 0;
  static int x16_tm() {
    static const int tm = [] {
      const int t = getenv("ARC_X16_TM") ? atoi(getenv("ARC_X16_TM")) : 32;
      if (t == 8 || t == 16 || t == 32) return t;
      fprintf(stderr, "arcinferzero: ARC_X16_TM=%d not supported (8, 16 or 32): 32\n", t);
      return 32;
    }();
    return tm;
  }
  cl_kernel x16_kernel(int qtype) {
    static const bool g128 = getenv("ARC_X16_GRF") && std::string(getenv("ARC_X16_GRF")) == "128";
    return kernel("x16", std::string(g128 ? "" : "-cl-intel-256-GRF-per-thread ") + "-DQTYPE=" + std::to_string(qtype) +
                             " -DX16_TM=" + std::to_string(x16_tm()) + (x16_emu_ ? " -DX16_EMU" : ""),
                  "gemm_x16");
  }
  void x16_init() {
    const bool tr = getenv("ARC_X16_TRY") && atoi(getenv("ARC_X16_TRY")) != 0;
    const bool on = getenv("ARC_X16") && atoi(getenv("ARC_X16")) != 0 && cl_.xmx && !zrt::integrated();
    if (!tr && !on) return;
    x16_emu_ = !cl_.xmx;
    static std::mutex mu;  // (once per process, as xmx_check: a split model's GPUs are all the same device)
    static int known = -1;
    std::lock_guard<std::mutex> g(mu);
    if (known < 0) {
      known = x16_check() ? 1 : 0;
      if (known)
        fprintf(stderr, "arcinferzero: f16 XMX on for the W8 prompt GEMMs%s (results checked against DP4a)\n",
                x16_emu_ ? " (DPAS emulated: tests)" : "");
    }
    x16_ = known == 1;
  }
  // aq_ / asc_ (act32: scales per 32 values, else per 256) of Mp rows to f16 in x16_ah_, the rows' scales in x16_rs_
  void x16_act(cl_mem aq, cl_mem asc, int Mp, int K, int act32) {
    const size_t ab = (size_t)Mp * K * 2, rb = (size_t)Mp * 4;
    if (x16_ah_bytes_ < ab) {
      free_buf(x16_ah_);
      x16_ah_ = buf(ab);
      x16_ah_bytes_ = ab;
    }
    if (x16_rs_bytes_ < rb) {
      free_buf(x16_rs_);
      x16_rs_ = buf(rb);
      x16_rs_bytes_ = rb;
    }
    cl_kernel k = kernel("x16", x16_emu_ ? "-DX16_EMU -DQTYPE=100" : "-DQTYPE=100", "act_half");
    set_args(k, 0, aq, asc, Mp, K, act32, x16_ah_, x16_rs_);
    size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
    run("x16_act", k, 2, gws, lws);
  }
  // one GEMM, as matmul1 launches gemm.cl (row groups, split-K over ~17 superblocks of long rows); false: not a
  // shape or layout it takes (the caller runs gemm.cl)
  bool x16_launch(const char* name, int qtype, cl_mem wq, cl_mem wh, cl_mem aq, cl_mem asc, int Mp, int N, int K,
                  cl_mem C, int ldc, int c_off, int a, int act32) {
    const int tm = x16_tm();
    if (qtype < 100 || qtype > 103 || Mp % tm || N % 16 || K % 256) return false;
    x16_act(aq, asc, Mp, K, act32);
    cl_kernel k = x16_kernel(qtype);
    static const int rg_env = getenv("ARC_GEMM_RG") ? std::max(1, atoi(getenv("ARC_GEMM_RG"))) : 4;
    static const int wgy_max = getenv("ARC_GEMM_WGY") ? atoi(getenv("ARC_GEMM_WGY")) : 8;
    static const int sk_sb = getenv("ARC_GEMM_SPLIT") ? std::max(0, atoi(getenv("ARC_GEMM_SPLIT"))) : 17;
    const int nsb = K / 256, splits = K > 8192 && sk_sb > 0 ? (nsb + sk_sb - 1) / sk_sb : 1, rgroup = rg_env;
    int wgy = 1;
    while (wgy < wgy_max && (Mp / tm) % (wgy * 2) == 0) wgy *= 2;
    size_t gws[2] = {(size_t)N * (Mp / tm / wgy), (size_t)wgy}, lws[2] = {16, (size_t)wgy};
    for (int sp = 0; sp < splits; ++sp) {
      int s0 = nsb * sp / splits, s1 = nsb * (sp + 1) / splits, ac = sp == 0 ? a : 1, M = Mp;
      set_args(k, 0, wq, wh, x16_ah_, x16_rs_, C, M, N, K, ldc, c_off, ac, rgroup, s0, s1);
      run(name, k, 2, gws, lws);
    }
    return true;
  }
  bool x16_matmul(const char* name, const GpuWeight& w, cl_mem wq, cl_mem wh, int Mp, cl_mem C, int ldc, int c_off,
                  int a, bool act32) {
    return x16_launch(name, w.gemm_qtype, wq, wh, aq_, asc_, Mp, w.N, w.K, C, ldc, c_off, a, act32 ? 1 : 0);
  }
  // random W8 weights and int8 activations (with their true per-16 sums, which the DP4a kernel's min term reads):
  // every W8 layout, per-32 and per-256 activation scales, stored / split-K / added / fused SwiGLU, against gemm.cl
  // without XMX. A case passes when its error is below 0.5 % of the result's size (f16 rounding is ~0.03 %; a wrong
  // layout gives ~100 %)
  bool x16_check() {
    const int M = 4 * 32, N = 64, K = 2048, nsb = K / 256, ng = N / 16;
    uint32_t seed = 4242;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    auto rndf = [&](float lo, float span) { return lo + (float)(rnd() % 10000) * (span / 10000.0f); };
    std::vector<uint32_t> wq((size_t)ng * nsb * 4 * 16 * 16), wh((size_t)ng * nsb * 10 * 16);
    for (auto& v : wq) v = rnd();
    for (auto& v : wh) {
      const float f = rndf(0.001f, 0.01f);
      memcpy(&v, &f, 4);
    }
    std::vector<int8_t> aq((size_t)M * K);
    for (auto& v : aq) v = (int8_t)(rnd() >> 24);
    std::vector<float> asc((size_t)(K / 32) * M), c0((size_t)M * N);
    for (auto& v : asc) v = rndf(0.01f, 0.01f);
    for (auto& v : c0) v = rndf(-2.0f, 4.0f);
    std::vector<uint16_t> ab((size_t)(K / 16) * M);
    for (int m = 0; m < M; ++m)
      for (int b = 0; b < K / 16; ++b) {
        int sum = 0;
        for (int i = 0; i < 16; ++i) sum += aq[(size_t)m * K + b * 16 + i];
        ab[(size_t)b * M + m] = (uint16_t)(int16_t)sum;
      }
    cl_mem bq = buf(wq.size() * 4, wq.data()), bh = buf(wh.size() * 4, wh.data()), ba = buf(aq.size(), aq.data()),
           bs = buf(asc.size() * 4, asc.data()), bb = buf(ab.size() * 2, ab.data()), bc = buf((size_t)M * N * 4);
    std::string bad;
    double worst = 0;
    struct Soft { bool& f; ~Soft() { f = false; } } soft{cl_.soft_fail};
    cl_.soft_fail = true;
    try {
      for (int q : {101, 100, 103, 102})
        for (int a32 = 1; a32 >= 0; --a32)
          for (int mode = 0; mode < 4; ++mode) {  // store, split-K pieces, add, fused SwiGLU
            std::vector<float> out[2];
            for (int x = 0; x < 2; ++x) {
              CL_CHECK(clEnqueueWriteBuffer(cl_.queue, bc, CL_TRUE, 0, c0.size() * 4, c0.data(), 0, nullptr, nullptr));
              const int acc = mode == 2 ? 1 : mode == 3 ? 2 : 0, pieces = mode == 1 ? 3 : 1;
              for (int sp = 0; sp < pieces; ++sp) {
                int s0 = nsb * sp / pieces, s1 = nsb * (sp + 1) / pieces, ac = sp == 0 ? acc : 1;
                int MM = M, NN = N, KK = K, ldc = N, coff = 0, rg = 2;
                if (x == 0) {
                  cl_kernel k = gemm_kernel(q, a32 != 0, 0, 1);
                  size_t gws[2] = {(size_t)N * (M / gemm_tm()), 1}, lws[2] = {16, 1};
                  set_args(k, 0, bq, bh, ba, bs, bb, bc, MM, NN, KK, ldc, coff, ac, rg, s0, s1);
                  run("x16_check", k, 2, gws, lws);
                } else {
                  x16_act(ba, bs, M, K, a32);
                  cl_kernel k = x16_kernel(q);
                  size_t gws[2] = {(size_t)N * (M / x16_tm() / 2), 2}, lws[2] = {16, 2};
                  set_args(k, 0, bq, bh, x16_ah_, x16_rs_, bc, MM, NN, KK, ldc, coff, ac, rg, s0, s1);
                  run("x16_check", k, 2, gws, lws);
                }
              }
              out[x].resize(c0.size());
              CL_CHECK(clEnqueueReadBuffer(cl_.queue, bc, CL_TRUE, 0, out[x].size() * 4, out[x].data(), 0, nullptr, nullptr));
            }
            double e2 = 0, r2 = 0;
            bool fin = true;
            for (size_t i = 0; i < c0.size(); ++i) {
              fin = fin && std::isfinite(out[0][i]) && std::isfinite(out[1][i]);
              e2 += (double)(out[1][i] - out[0][i]) * (out[1][i] - out[0][i]);
              r2 += (double)(out[0][i] - (mode == 2 ? c0[i] : 0.0f)) * (out[0][i] - (mode == 2 ? c0[i] : 0.0f));
            }
            const double rel = r2 > 0 ? std::sqrt(e2 / r2) : 1.0;
            worst = std::max(worst, rel);
            if (!fin || !(rel < 5e-3))
              bad += " " + std::to_string(q) + (a32 ? "/act32" : "") + "/" + std::to_string(mode) + " (" + std::to_string(rel) + ")";
          }
    } catch (const std::exception& e) {
      fprintf(stderr, "arcinferzero: the f16 XMX kernels do not build here (%s): not used\n", e.what());
      return false;
    }
    for (cl_mem* m : {&bq, &bh, &ba, &bs, &bb, &bc}) free_buf(*m);
    if (!bad.empty()) fprintf(stderr, "arcinferzero: f16 XMX results differ from DP4a (%s ): not used (please report)\n", bad.c_str());
    else if (getenv("ARC_X16_TRY")) fprintf(stderr, "arcinferzero: f16 XMX check: largest error %.2e of the result\n", worst);
    return bad.empty();
  }

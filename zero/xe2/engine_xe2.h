// Tuning switches for GPUs with XMX (Xe2 / Xe3: Arc B580, Pro B60), zero/xe2/: A/B settings that only the owners of
// these cards can measure (docs/xe2-tuning.md). A part of class Engine (zero/engine.h), included inside the class
// body like zero/mgpu/engine_tp_shard.h: not a header of its own.
// On for every GPU that offers XMX: discrete cards (one or the multi-GPU shards) and iGPUs with it (Lunar Lake, Panther
// Lake, Arrow Lake H). Off on the others, as Xe-LPG: there every kernel is built and launched exactly as without this
// file (xe2_ stays false, so the GEMM keeps 256 registers per thread, one column group per work-group, the GEMV its
// default register file). Only discrete B60s have measured these defaults. ARC_XMX_TRY turns it on anywhere, for the
// tests of the switches themselves.

  bool xe2_ = false;  // (set before the XMX check: the GPU offers XMX, or ARC_XMX_TRY)
  // (128 registers only where a register is 64 bytes: on a GPU with 8-lane sub-groups, 32-byte registers, ocloc shows
  //  the XMX GEMM spilling with 128)
  bool xe2_grf128() const { return xe2_ && (!cl_.sg8 || getenv("ARC_XMX_TRY")); }

  // the dense prompt GEMM with the default register file (128) instead of 256 registers per thread: more threads per
  // core; on a Pro B60 the 27B's prompt went 450 -> 617 tok/s (issue #4, 2026-11-02). ARC_GEMM_GRF=256 as before.
  // (On the 125H iGPU 128 made the 35B's prompt 3.4x slower: xe2_ is off there, which keeps 256.)
  // (ARC_GEMM_GRF=128 asked for: 128 on any GPU, the 8-lane ones and the DP4a GEMM of iGPUs without the xe2_ switches
  //  too, for their owners' sweeps (bench\igpu_grf_sweep.ps1); the same sums, other occupancy. The 125H's DP4a GEMM
  //  timed 4.7x slower with it, 2026-11-07, so it stays opt-in)
  std::string xe2_gemm_grf() const {
    static const std::string e = getenv("ARC_GEMM_GRF") ? getenv("ARC_GEMM_GRF") : "";
    if (e == "128") return std::string();
    return xe2_grf128() && e != "256" ? std::string() : std::string("-cl-intel-256-GRF-per-thread ");
  }
  // ARC_X8_W=1: gemm_x8.cl -DX8W, one 8-lane sub-group for both halves of a 16-column group (activations loaded once)
  static std::string x8_defs() {
    static const bool w = getenv("ARC_X8_W") && atoi(getenv("ARC_X8_W")) != 0;
    return w ? " -DX8W" : "";
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
    // (asked for with ARC_GEMM_CG: on any GPU, the DP4a GEMM of iGPUs without the xe2_ switches too)
    static const bool asked = getenv("ARC_GEMM_CG") != nullptr;
    return (xe2_ || asked) && (N / 16) % cg == 0 ? cg : 1;
  }
  static std::string xe2_cg_defs(int cg) { return cg != 1 ? " -DCG=" + std::to_string(cg) : std::string(); }
  // ARC_GEMV_GRF=256: the single-token GEMV with the large register file
  // (2026-11-08: the GEMV switches act wherever they are set, not only with xe2_ on: the 140T keeps DP4a for prompts,
  //  so xe2_ is off there, and its owner's generation runs had them silently ignored. Unset, nothing changes)
  std::string xe2_gemv_grf() const {
    static const std::string g = grf_env("ARC_GEMV_GRF");
    return g;
  }
  // ARC_GEMV_PF=1: the single-token GEMV (gemv, codebook / K types) issues every row's reads of a step before decoding
  // the first (GROWS x more loads in flight: a B60's GEMV reached ~55 % of its bandwidth; QUEUE F31.4). Bit-identical
  std::string xe2_gemv_pf() const {
    static const bool pf = getenv("ARC_GEMV_PF") && atoi(getenv("ARC_GEMV_PF")) != 0;
    return pf ? std::string(" -DGEMV_PF") : std::string();
  }
  // the 2-8 token GEMV (gemv_mt / gemv_mt_v: MTP and DFlash2 verification, batched slots) with the default register
  // file (128) on discrete cards: a Pro B60's 27B MTP went 43.3 -> 47.1 tok/s (issue #4, 2026-11-02); ARC_GEMV_MT_GRF=256
  // as before. The iGPUs keep 256 (the 125H forced to 128: 27B MTP 11.78 -> 9.57 tok/s, 2026-11-03)
  std::string xe2_gemv_mt_grf() const {
    static const std::string e = getenv("ARC_GEMV_MT_GRF") ? getenv("ARC_GEMV_MT_GRF") : "";
    if (e == "128") return std::string();  // (asked for: any GPU, for its owner's sweep)
    return xe2_grf128() && e != "256" ? std::string() : std::string("-cl-intel-256-GRF-per-thread ");
  }

  // QUEUE 31.8: the XMX prompt GEMM's activations with 2D block loads (cl_intel_subgroup_2d_block_io, gemm.cl -DX2D): a
  // step's 8 token rows x 64 bytes in one 8b_8r32x2c load, the DPAS A operands as the hardware lays them out, instead of
  // 16 one-row block reads; the prefetch (-DX2D_PF) also asks for the next 64 bytes of those rows. Dense GEMMs only (the
  // MoE tile kernel reads rows through a row map). On by default wherever the driver offers them with XMX: a Pro B60's
  // 27B prompt went 608 -> 752 tok/s, 763 with the prefetch (issue #4, 2026-11-06). ARC_GEMM_2D=0 / ARC_GEMM_2D_PF=0 turn
  // them off. The start-time XMX check runs with them, and if they do not build or differ, XMX goes on without them
  bool x2d_ = false;
  static bool x2d_switch(const char* name) {
    const char* e = getenv(name);
    return e != nullptr ? atoi(e) != 0 : true;
  }
  std::string xe2_x2d() const {
    static const bool pf = x2d_switch("ARC_GEMM_2D_PF");
    return x2d_ ? std::string(pf ? " -DX2D -DX2D_PF" : " -DX2D") : std::string();
  }
  bool x2d_wanted() const { return xe2_ && cl_.block2d && x2d_switch("ARC_GEMM_2D"); }
  // ARC_GEMM_WPF=n (1..16): the XMX prompt GEMM prefetches its weights n 64-value chunks ahead (4: the next
  // superblock); off by default, a B60 to measure. Bit-identical
  std::string xe2_wpf() const {
    static const int n = getenv("ARC_GEMM_WPF") ? atoi(getenv("ARC_GEMM_WPF")) : 0;
    return xe2_ && !cl_.sg8 && n >= 1 && n <= 16 ? " -DWPF=" + std::to_string(n) : std::string();
  }

  // ---- the f16 XMX prompt GEMM (kernels/gemm_x16.cl) for the W8 layouts (the 27B's IQ3_S, decoded just in time).
  // Off until measured on a B60: ARC_X16=1 on a discrete GPU with XMX. ARC_X16_TRY=1 anywhere, DPAS emulated where
  // the GPU has none (tests of the kernel's layouts on this laptop). Taken only once its results match the DP4a
  // kernel's on random data within f16 rounding (not bit for bit: f16 weights, f32 sums in another order).
  // ARC_X16_TM=8 / 16 / 32: tokens per sub-group (default 32); ARC_X16_GRF=128: the default register file (256 else)
  bool x16_ = false, x16_emu_ = false;
  std::string x16_src_;
  cl_mem x16_ah_ = nullptr, x16_rs_ = nullptr;  // (the f16 activations and each row's scale: x16_reserve)
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
  // tokens per sub-group for a chunk of Mp rows: x16_tm(), or fewer where Mp is not a multiple of it (short chunks
  // are padded to 16 rows only); 0: none fits (gemm.cl then)
  static int x16_tm_for(int Mp) {
    for (int t = x16_tm(); t >= 8; t /= 2)
      if (Mp % t == 0) return t;
    return 0;
  }
  // ARC_X16_CG=2 / 4: 16-column groups per sub-group (default 1). Each 8-token read of the activations then feeds 2 or
  // 4 DPAS instead of 1 (sycl-tla's BMG GEMM: 32 x 64 per sub-group); a matrix whose groups it does not divide keeps 1
  static int x16_cg_for(int N) {
    static const int cg = [] {
      const int c = getenv("ARC_X16_CG") ? atoi(getenv("ARC_X16_CG")) : 1;
      if (c == 1 || c == 2 || c == 4) return c;
      fprintf(stderr, "arcinferzero: ARC_X16_CG=%d not supported (1, 2 or 4): 1\n", c);
      return 1;
    }();
    return (N / 16) % cg == 0 ? cg : 1;
  }
  cl_kernel x16_kernel(int qtype, int cg, int tm) {
    static const bool g128 = getenv("ARC_X16_GRF") && std::string(getenv("ARC_X16_GRF")) == "128";
    return kernel("x16", std::string(g128 ? "" : "-cl-intel-256-GRF-per-thread ") + "-DQTYPE=" + std::to_string(qtype) +
                             " -DX16_TM=" + std::to_string(tm) + (cg != 1 ? " -DX16_CG=" + std::to_string(cg) : "") +
                             (x16_emu_ ? " -DX16_EMU" : "") + (cl_.sg8 ? " -DX16_SG8" : ""),
                  "gemm_x16");
  }
  bool x16_wanted() const {
    const bool tr = getenv("ARC_X16_TRY") && atoi(getenv("ARC_X16_TRY")) != 0;
    // (iGPUs: only those with 8-lane DPAS, the 140T, where the int8 XMX GEMM lost to DP4a in its rescaling: gemm_x16.cl
    //  -DX16_SG8, 2026-11-08; the 16-lane iGPUs have not been measured with it)
    return tr || (getenv("ARC_X16") && atoi(getenv("ARC_X16")) != 0 && cl_.xmx && (!zrt::integrated() || cl_.sg8));
  }
  void x16_init() {
    if (!x16_wanted()) return;
    x16_emu_ = !cl_.xmx;
    // (once per device and kind of DPAS: a split model's GPUs are all one device (ClRt::open_devices) and need not
    //  check again; an engine on another device in the same process checks for itself)
    static std::mutex mu;
    static std::map<std::string, bool> known;
    const std::string key = cl_.device_id + (x16_emu_ ? "|emulated" : "");
    std::lock_guard<std::mutex> g(mu);
    auto it = known.find(key);
    if (it == known.end()) {
      const bool ok = x16_check();
      it = known.emplace(key, ok).first;
      if (ok)
        fprintf(stderr, "arcinferzero: f16 XMX on for the W8 prompt GEMMs%s (results checked against DP4a)\n",
                x16_emu_ ? " (DPAS emulated: tests)" : "");
    }
    x16_ = it->second;
  }
  // its buffers at their full size with the others (alloc_buffers), so the memory plan counts them (27B, 4096-token
  // chunks: 143 MB) and a prompt never allocates mid-pass; a dry run (--ctx auto's estimate) counts them too
  void x16_reserve(size_t M, int maxK) {
    if (!x16_ && !(opt.dry && x16_wanted())) return;
    free_buf(x16_ah_);
    free_buf(x16_rs_);
    x16_ah_bytes_ = M * (size_t)maxK * 2;
    x16_rs_bytes_ = M * 4;
    x16_ah_ = buf(x16_ah_bytes_);
    x16_rs_ = buf(x16_rs_bytes_);
  }
  // aq / asc (act32: scales per 32 values, else per 256) of Mp rows to f16 in x16_ah_, the rows' scales in x16_rs_
  void x16_act(cl_mem aq, cl_mem asc, int Mp, int K, int act32) {
    const size_t ab = (size_t)Mp * K * 2, rb = (size_t)Mp * 4;
    if (x16_ah_bytes_ < ab) {  // (only the start-up check: the engine's are reserved whole)
      free_buf(x16_ah_);
      x16_ah_ = buf(ab);
      x16_ah_bytes_ = ab;
    }
    if (x16_rs_bytes_ < rb) {
      free_buf(x16_rs_);
      x16_rs_ = buf(rb);
      x16_rs_bytes_ = rb;
    }
    cl_kernel k = kernel("x16", "-DACT_ONLY", "act_half");
    set_args(k, 0, aq, asc, Mp, K, act32, x16_ah_, x16_rs_);
    size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
    run("x16_act", k, 2, gws, lws);
  }
  // one GEMM, as matmul1 launches gemm.cl (row groups, split-K over ~17 superblocks of long rows); false: not a
  // shape or layout it takes (the caller runs gemm.cl)
  bool x16_launch(const char* name, int qtype, cl_mem wq, cl_mem wh, cl_mem aq, cl_mem asc, int Mp, int N, int K,
                  cl_mem C, int ldc, int c_off, int a, int act32) {
    const int tm = x16_tm_for(Mp);
    if (qtype < 100 || qtype > 103 || !tm || N % 16 || K % 256) return false;
    x16_act(aq, asc, Mp, K, act32);
    const int cg = cl_.sg8 ? 1 : x16_cg_for(N);
    cl_kernel k = x16_kernel(qtype, cg, tm);
    static const int rg_env = getenv("ARC_GEMM_RG") ? std::max(1, atoi(getenv("ARC_GEMM_RG"))) : 4;
    static const int wgy_max = getenv("ARC_GEMM_WGY") ? atoi(getenv("ARC_GEMM_WGY")) : 8;
    static const int sk_sb = getenv("ARC_GEMM_SPLIT") ? std::max(0, atoi(getenv("ARC_GEMM_SPLIT"))) : 17;
    // (the fused SwiGLU needs the whole sum in one piece: silu(gate) * (u1 + u2), not silu(gate) * u1 + u2)
    const int nsb = K / 256, splits = a != 2 && K > 8192 && sk_sb > 0 ? (nsb + sk_sb - 1) / sk_sb : 1, rgroup = rg_env;
    int wgy = 1;
    while (wgy < wgy_max && (Mp / tm) % (wgy * 2) == 0) wgy *= 2;
    size_t gws[2] = {(size_t)N / cg * (Mp / tm / wgy), (size_t)wgy}, lws[2] = {16, (size_t)wgy};
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
  // random W8 weights and int8 activations (with their true per-16 sums, which the DP4a kernel's min term reads),
  // activation scales spread over three decades (each row's largest divides the rest): every W8 layout, per-32 and
  // per-256 activation scales; stored / added / fused SwiGLU through x16_launch (the engine's grid: a partial row
  // group, a column offset and a row stride wider than N; 96 x 64 at x16_tm(), 80 x 48 at 16 tokens with one column
  // group) and split-K pieces; against gemm.cl without XMX. A case passes when its error is below 0.5 % of the
  // result's size (f16 rounding ~0.03 %; a wrong layout ~100 %)
  bool x16_check() {
    const int K = 2048, nsb = K / 256;
    uint32_t seed = 4242;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    auto rndf = [&](float lo, float span) { return lo + (float)(rnd() % 10000) * (span / 10000.0f); };
    std::string bad;
    double worst = 0;
    std::vector<cl_mem> bufs;
    struct Soft { bool& f; bool was; ~Soft() { f = was; } } soft{cl_.soft_fail, cl_.soft_fail};
    cl_.soft_fail = true;
    bool built = true;
    try {
      for (auto [M, N] : {std::pair{96, 64}, std::pair{80, 48}}) {
        const int ng = N / 16, ldc = N + 32, coff = 16;
        std::vector<uint32_t> wq((size_t)ng * nsb * 4 * 16 * 16), wh((size_t)ng * nsb * 10 * 16);
        for (auto& v : wq) v = rnd();
        for (auto& v : wh) {
          const float f = rndf(0.001f, 0.01f);
          memcpy(&v, &f, 4);
        }
        std::vector<int8_t> aq((size_t)M * K);
        for (auto& v : aq) v = (int8_t)(rnd() >> 24);
        std::vector<float> asc((size_t)(K / 32) * M), c0((size_t)M * ldc);
        for (auto& v : asc) v = 0.02f * std::pow(10.0f, -rndf(0.0f, 3.0f));
        for (auto& v : c0) v = rndf(-2.0f, 4.0f);
        std::vector<uint16_t> ab((size_t)(K / 16) * M);
        for (int m = 0; m < M; ++m)
          for (int b = 0; b < K / 16; ++b) {
            int sum = 0;
            for (int i = 0; i < 16; ++i) sum += aq[(size_t)m * K + b * 16 + i];
            ab[(size_t)b * M + m] = (uint16_t)(int16_t)sum;
          }
        cl_mem bq = buf(wq.size() * 4, wq.data()), bh = buf(wh.size() * 4, wh.data()), ba = buf(aq.size(), aq.data()),
               bs = buf(asc.size() * 4, asc.data()), bb = buf(ab.size() * 2, ab.data()), bc = buf(c0.size() * 4);
        bufs.insert(bufs.end(), {bq, bh, ba, bs, bb, bc});
        for (int q : {101, 100, 103, 102})
          for (int a32 = 1; a32 >= 0; --a32)
            for (int mode = 0; mode < 4; ++mode) {  // store, split-K pieces, add, fused SwiGLU
              std::vector<float> out[2];
              const int acc = mode == 2 ? 1 : mode == 3 ? 2 : 0, pieces = mode == 1 ? 3 : 1;
              for (int x = 0; x < 2; ++x) {
                CL_CHECK(clEnqueueWriteBuffer(cl_.queue, bc, CL_TRUE, 0, c0.size() * 4, c0.data(), 0, nullptr, nullptr));
                if (x == 1 && pieces == 1) {
                  if (!x16_launch("x16_check", q, bq, bh, ba, bs, M, N, K, bc, ldc, coff, acc, a32)) throw std::runtime_error("shape not taken");
                } else {
                  for (int sp = 0; sp < pieces; ++sp) {
                    int s0 = nsb * sp / pieces, s1 = nsb * (sp + 1) / pieces, ac = sp == 0 ? acc : 1;
                    int MM = M, NN = N, KK = K, ld = ldc, co = coff, rg = 2;
                    if (x == 0) {
                      cl_kernel k = gemm_kernel(q, a32 != 0, 0, 1);
                      size_t gws[2] = {(size_t)N * (M / gemm_tm()), 1}, lws[2] = {16, 1};
                      set_args(k, 0, bq, bh, ba, bs, bb, bc, MM, NN, KK, ld, co, ac, rg, s0, s1);
                      run("x16_check", k, 2, gws, lws);
                    } else {
                      x16_act(ba, bs, M, K, a32);
                      const int cg = cl_.sg8 ? 1 : x16_cg_for(N), tm = x16_tm_for(M);
                      cl_kernel k = x16_kernel(q, cg, tm);
                      size_t gws[2] = {(size_t)N / cg * (M / tm), 1}, lws[2] = {16, 1};
                      set_args(k, 0, bq, bh, x16_ah_, x16_rs_, bc, MM, NN, KK, ld, co, ac, rg, s0, s1);
                      run("x16_check", k, 2, gws, lws);
                    }
                  }
                }
                out[x].resize(c0.size());
                CL_CHECK(clEnqueueReadBuffer(cl_.queue, bc, CL_TRUE, 0, out[x].size() * 4, out[x].data(), 0, nullptr, nullptr));
              }
              double e2 = 0, r2 = 0;
              bool fin = true;
              for (size_t i = 0; i < c0.size(); ++i) {  // (the columns outside [coff, coff + N) must stay as they were)
                fin = fin && std::isfinite(out[0][i]) && std::isfinite(out[1][i]);
                e2 += (double)(out[1][i] - out[0][i]) * (out[1][i] - out[0][i]);
                r2 += (double)(out[0][i] - (mode == 2 ? c0[i] : 0.0f)) * (out[0][i] - (mode == 2 ? c0[i] : 0.0f));
              }
              const double rel = r2 > 0 ? std::sqrt(e2 / r2) : 1.0;
              worst = std::max(worst, rel);
              if (!fin || !(rel < 5e-3))
                bad += " " + std::to_string(M) + "x" + std::to_string(N) + "/" + std::to_string(q) + (a32 ? "/act32" : "") +
                       "/" + std::to_string(mode) + " (" + std::to_string(rel) + ")";
            }
      }
    } catch (const zrt::GpuWaitError&) {
      for (cl_mem& m : bufs) free_buf(m);
      throw;  // (a split model's GPU that hangs: not a verdict on the kernels, which is kept per device)
    } catch (const std::exception& e) {
      fprintf(stderr, "arcinferzero: the f16 XMX kernels do not build here (%s): not used\n", e.what());
      built = false;
    }
    for (cl_mem& m : bufs) free_buf(m);
    if (!built) return false;
    if (!bad.empty()) fprintf(stderr, "arcinferzero: f16 XMX results differ from DP4a (%s ): not used (please report)\n", bad.c_str());
    else if (getenv("ARC_X16_TRY")) fprintf(stderr, "arcinferzero: f16 XMX check: largest error %.2e of the result\n", worst);
    return bad.empty();
  }

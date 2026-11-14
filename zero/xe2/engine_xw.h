// The H8R prompt GEMM on XMX with a wider tile per sub-group (QUEUE 33.4 / 33.4b, kernels/gemm_xw.cl and
// gemm_xw8.cl): a part of class Engine (zero/engine.h), included inside the class body after zero/xe2/engine_xe2.h
// and engine_h8.h; not a header of its own. On by default in the Arc 140T preset and where H8 is the default on a
// discrete card (h8_auto_, below); elsewhere off unless asked for, every path unchanged without the switch.
//
// ARC_GEMM_XW=1: where H8R runs on XMX (h8_ with per-column scales: the Qwen dense models on discrete cards by
// default, the Arc 140T preset's chunks of 512+ tokens, ARC_H8=1 elsewhere), the H8 GEMMs whose shape it takes (M a
// multiple of XW_TM, N of 16 * XW_NC) go through the wide-tile kernel instead of gemm.cl / gemm_x8.cl; the others
// stay. On by default in the Arc 140T preset since 2026-11-11 (32 x 16 tiles, 256 registers: its owner's sweep, issue
// #5, 27B 512 / 4K / 16K prompts 147.6 / 214.1 / 205.4 -> 204.9 / 298.6 / 283.1 tok/s, logits identical; ARC_GEMM_XW=0:
// off). On by default with h8_auto_ (the Qwen dense models on discrete cards with 16-lane XMX) since 2026-11-13: a Pro
// B60's sweep (issue #4, bench\xw_sweep.ps1), 27B 512 / 4K / 16K prompts 691.1 / 1,066.5 / 978.9 -> 716.8 / 1,291.7 /
// 1,169.5 tok/s with 32 x 64 tiles (32 x 32 739.5 / 1,267.5 / 1,147.6; 16 x 64 and 32 x 16 slower than the default),
// logits identical at every shape.
//   16-lane XMX (Xe2 / Xe3): gemm_xw.cl, XW_TM tokens x XW_NC 16-column groups (32 x 4 = 32 x 64).
//   8-lane XMX (the 140T, Alchemist-kind): gemm_xw8.cl, coalesced loads (a group's weights in 4 block reads, an
//   activation row in one), XW_TM x XW_NC (32 x 1 = 32 x 16, both halves, 256 registers; before 2026-11-11 16 x 16 at
//   128, which the 140T measured 26 % slower than 32 x 16 at 256).
//   ARC_GEMM_XW_TM=16 / 32 / 64 (32 by default): tokens per sub-group; ARC_GEMM_XW_NC=1 / 2 / 4, on 16 lanes also 6 / 8
//   (4 on 16 lanes, 1 on 8); ARC_GEMM_XW_GRF=128 / 256: the register file (256 by default; on 8 lanes 32 tokens spill
//   at 128). 64 tokens and 6 / 8 groups (2026-11-13) are for sweeps beyond 32 x 64 (bench\xw_sweep.ps1 -Wide): at
//   256 registers ocloc shows 32 x 96 ~2 bytes spilled, 16 x 128 and 64 x 32 none, 32 x 128 and 64 x 64 heavy spills
//   (B60); NC 6 runs a width's 96-wide part and its rest 64 columns wide (xw_piece).
//   ARC_GEMM_XW_TRY=1: on any GPU with H8R, DPAS emulated where the GPU has none of its kind (tests; no speed
//   meaning; on an iGPU H8 needs ARC_H8=1, or the 140T preset with ARC_140T=1 ARC_XMX_TRY=2).
//   ARC_GEMM_XW_LANES=16: the 16-lane kernel on an 8-lane GPU, DPAS emulated (tests).
// Checked once per device at start against gemm.cl without XMX, bit for bit (xw_check); a difference or a build
// that fails leaves the old kernels and says so.

  bool xw_ = false, xw_emu_ = false;
  std::string xw_src_, xw8_src_;
  // 8 or 16: which kernel (8: gemm_xw8 on GPUs with 8-lane sub-groups, the 140T and the 125H)
  int xw_lanes() const {
    static const int e = getenv("ARC_GEMM_XW_LANES") ? atoi(getenv("ARC_GEMM_XW_LANES")) : 0;
    return cl_.sg8 && e != 16 ? 8 : 16;
  }
  // (8 lanes: 32 x 16 at 256 registers by default: the 140T's sweep (issue #5, 2026-11-11), 27B 4K prompt 16 x 16 at
  //  128 273.7, 32 x 16 at 256 298.6, 16 x 32 at 256 275.9, 32 x 32 at 256 257.9 tok/s; 16 lanes: 32 x 64 at 256)
  int xw_tm() const {
    static const int e = getenv("ARC_GEMM_XW_TM") ? atoi(getenv("ARC_GEMM_XW_TM")) : 0;
    const int def = 32;
    if (e == 16 || e == 32 || e == 64) return e;
    if (e) { static std::once_flag said; std::call_once(said, [&] { zlog::eprintf("inferzero: ARC_GEMM_XW_TM=%d not supported (16, 32 or 64): %d\n", e, def); }); }
    return def;
  }
  bool xw_grf256() const {
    static const char* e = getenv("ARC_GEMM_XW_GRF");
    if (e && std::string(e) == "128") return false;
    if (e && std::string(e) == "256") return true;
    return true;
  }
  int xw_nc() const {
    static const int e = getenv("ARC_GEMM_XW_NC") ? atoi(getenv("ARC_GEMM_XW_NC")) : 0;
    const int def = xw_lanes() == 8 ? 1 : 4;
    if (e == 0) return def;
    if (e == 1 || e == 2 || e == 4 || (xw_lanes() == 16 && (e == 6 || e == 8))) return e;
    static std::once_flag said;
    std::call_once(said, [&] { zlog::eprintf("inferzero: ARC_GEMM_XW_NC=%d not supported here: %d\n", e, def); });
    return def;
  }
  static bool xw_try() {
    static const bool t = getenv("ARC_GEMM_XW_TRY") && atoi(getenv("ARC_GEMM_XW_TRY")) != 0;
    return t;
  }
  // (h8_ and xmx_ known: after the XMX and H8 checks of the constructor)
  // (the 140T preset and h8_auto_ (discrete cards): on unless ARC_GEMM_XW=0; elsewhere ARC_GEMM_XW=1)
  bool xw_wanted() const {
    const char* e = getenv("ARC_GEMM_XW");
    const bool on = e ? atoi(e) != 0 : (p140_ || h8_auto_);
    return h8_ && h8_row() && (xw_try() || (on && xmx_));
  }
  // ARC_GEMM_XW_PF=0 / 1 / 2 / 3 (16 lanes; B60 prefill plan 13.2): the weights (1), the activations (2, with 2D
  // loads) or both (3) of the chunk ARC_GEMM_XW_PFD (1 / 2 / 4) chunks ahead prefetched into L1; the same results bit
  // for bit (checked at start like every shape). On by default since 2026-11-14 as weights, 1 chunk ahead: a Pro B60's
  // xw_sweep (manoreken2, issue #4, release 2026.11.14.05), the 27B 4K prompt 1,321.7 -> 1,529.0 tok/s and 512 tokens
  // 737.3 -> 867.2 (pf3 d1 867.4 / 1,478.8; pf2 no gain; deeper is slower), logits identical. =0: off, as before
  static int xw_pf() {
    static const int v = [] {
      const int e = getenv("ARC_GEMM_XW_PF") ? atoi(getenv("ARC_GEMM_XW_PF")) : 1;
      return e >= 0 && e <= 3 ? e : 1;
    }();
    return v;
  }
  static int xw_pfd() {
    static const int v = [] {
      const int e = getenv("ARC_GEMM_XW_PFD") ? atoi(getenv("ARC_GEMM_XW_PFD")) : 1;
      return e == 1 || e == 2 || e == 4 ? e : 1;
    }();
    return v;
  }
  // (the activations' part needs 2D loads: without them, or with DPAS emulated, only the weights' part is built)
  int xw_pf_eff() const {
    if (xw_lanes() != 16) return 0;
    return (!x2d_ || xw_emu_) ? (xw_pf() & 1) : xw_pf();
  }
  std::string xw_pf_defs() const {
    const int pf = xw_pf_eff();
    return pf ? " -DXW_PF=" + std::to_string(pf) + " -DXW_PFD=" + std::to_string(xw_pfd()) : "";
  }
  // ARC_GEMM_XW_PACK (8 lanes: the Arc 140T; issue #5, its xw_peak of 2026-11-13: packAW 32 x 16 against the default,
  // the 27B's FFN shapes, 4K 23.4 -> 25.0 TOPS, 512 21.6 -> 24.5): the activations (1), the weights (2) or both (3)
  // written in the DPAS operand order gemm_xw8 -DXW_PACKA / -DXW_PACKW reads, by the kernels that make them (h8r_act;
  // h8r_decode or h8r_weights), so without a repacking pass. Both by default in the 140T preset, 0 off; only where
  // gemm_xw8 takes the GEMM (its shapes; resident weights, converted once in place, stay in the plain order). The same
  // values in another order: the same results, checked at start for every form used (xw_check).
  static int xw_pack_env() {
    static const int v = [] {
      const char* e = getenv("ARC_GEMM_XW_PACK");
      return e ? std::clamp(atoi(e), 0, 3) : -1;
    }();
    return v;
  }
  int xw_pack_cfg() const {  // (the forms this engine may use; xw_ itself is decided by the check)
    // (one column group a sub-group only, the default on 8 lanes: there every H8 weight's N fits the tile, so a packed
    //  input is never left to a GEMM gemm_xw8 declines; wider tiles were slower on the 140T anyway. Review 2026-11-14)
    if (xw_lanes() != 8 || !h8_row() || xw_nc() != 1) return 0;
    const int e = xw_pack_env();
    return e >= 0 ? e : (p140_ ? 3 : 0);
  }
  bool xw_pack_a(int Mp, int K) const { return xw_ && (xw_pack_cfg() & 1) && Mp % xw_tm() == 0 && K % 256 == 0; }
  bool xw_pack_w(const GpuWeight& w, int Mp) const {
    return xw_ && (xw_pack_cfg() & 2) && !w.wq && Mp % xw_tm() == 0 && w.N % (16 * xw_nc()) == 0 && w.K % 256 == 0;
  }
  // (nc: column groups per sub-group, 0 = xw_nc(); the 32 x 96 tiles' tail runs the nc 4 kernel. pack: gemm_xw8's
  //  packed operands, bit 1 the activations, bit 2 the weights)
  cl_kernel xw_kernel(int nc = 0, int pack = 0) {
    const std::string defs = std::string(xw_grf256() ? "-cl-intel-256-GRF-per-thread " : "") + "-DXW_TM=" +
                             std::to_string(xw_tm()) + " -DXW_NC=" + std::to_string(nc ? nc : xw_nc()) +
                             (xw_emu_ ? " -DXW_EMU" : "") + (xw_lanes() == 16 ? xw_pf_defs() : std::string()) +
                             (xw_lanes() == 8 && (pack & 1) ? " -DXW_PACKA" : "") +
                             (xw_lanes() == 8 && (pack & 2) ? " -DXW_PACKW" : "");
    if (xw_lanes() == 8) return kernel("xw8", defs, "gemm_xw8");
    return kernel("xw", defs + (x2d_ && !xw_emu_ ? " -DX2D" : ""), "gemm_xw");
  }
  // one piece [s0, s1) of K over N columns from 16-column group g_base (16 lanes; 0 on 8): the grid of gemm_xw.cl's /
  // gemm_xw8.cl's header (a sub-group of 16 or 8 work-items)
  void xw_run(const char* name, cl_kernel k, cl_mem wq, cl_mem wh, cl_mem aq, cl_mem asc, int M, int N, int K, cl_mem C,
              int ldc, int c_off, int a, int rgroup, int s0, int s1, int wgy, int nc, int g_base) {
    const int tm = xw_tm(), nb = N / (16 * nc), ln = xw_lanes();
    size_t gws[2] = {(size_t)ln * nb * (M / tm / wgy), (size_t)wgy}, lws[2] = {(size_t)ln, (size_t)wgy};
    if (ln == 8) set_args(k, 0, wq, wh, aq, asc, C, M, N, K, ldc, c_off, a, rgroup, s0, s1);
    else set_args(k, 0, wq, wh, aq, asc, C, M, N, K, ldc, c_off, a, rgroup, s0, s1, g_base);
    run(name, k, 2, gws, lws);
  }
  // ARC_GEMM_XW_NC=6 (96 columns, 16 lanes): N's first columns in 96-wide tiles, as many as leave a multiple of 64 for
  // the NC 4 kernel (the 27B's widths 1024, 5120, 17408 are not multiples of 96); -1: N has no such split
  static int xw_cols96(int N) {
    for (int a = N / 96; a >= 0; --a)
      if ((N - 96 * a) % 64 == 0) return 96 * a;
    return -1;
  }
  bool xw_takes_n(int N) const { return xw_nc() == 6 ? xw_cols96(N) >= 0 : N % (16 * xw_nc()) == 0; }
  // one piece of K over all N columns: one launch, or with NC 6 the 96-wide part and the rest in 64-wide tiles
  void xw_piece(const char* name, cl_mem wq, cl_mem wh, cl_mem aq, cl_mem asc, int M, int N, int K, cl_mem C, int ldc,
                int c_off, int a, int rgroup, int s0, int s1, int wgy, int pack = 0) {
    const int nc = xw_nc();
    if (nc != 6) return xw_run(name, xw_kernel(0, pack), wq, wh, aq, asc, M, N, K, C, ldc, c_off, a, rgroup, s0, s1, wgy, nc, 0);
    if (pack) throw std::logic_error("gemm_xw: packed operands with NC 6");
    const int n96 = xw_cols96(N);
    if (n96 > 0) xw_run(name, xw_kernel(6), wq, wh, aq, asc, M, n96, K, C, ldc, c_off, a, rgroup, s0, s1, wgy, 6, 0);
    if (n96 < N)
      xw_run(name, xw_kernel(4), wq, wh, aq, asc, M, N - n96, K, C, ldc, c_off, a, rgroup, s0, s1, wgy, 4, n96 / 16);
  }
  // h8_matmul's GEMM through gemm_xw / gemm_xw8, with gemm.cl's split-K pieces and row groups; false: a shape it does
  // not take (the old kernel runs it)
  bool xw_matmul(const char* name, cl_mem wq, cl_mem hh, int Mp, int N, int K, cl_mem C, int ldc, int c_off, int a,
                 int pack = 0) {
    const int tm = xw_tm();
    if (Mp % tm || !xw_takes_n(N) || K % 256) return false;
    static const int rg_env = getenv("ARC_GEMM_RG") ? std::max(1, atoi(getenv("ARC_GEMM_RG"))) : 4;
    static const int wgy_max = getenv("ARC_GEMM_WGY") ? atoi(getenv("ARC_GEMM_WGY")) : 8;
    // (the pieces gemm.cl's H8 GEMM takes, h8_matmul's own helper: the same sums per piece, so the same results)
    const int nsb = K / 256, splits = h8_splits(K, a);
    int wgy = 1;
    while (wgy < wgy_max && (Mp / tm) % (wgy * 2) == 0) wgy *= 2;
    for (int sp = 0; sp < splits; ++sp)
      xw_piece(name, wq, hh, h8_aq_, h8_asc_, Mp, N, K, C, ldc, c_off, sp == 0 ? a : 1, rg_env, nsb * sp / splits,
               nsb * (sp + 1) / splits, wgy, pack);
    return true;
  }
  void xw_init() {
    if (!xw_wanted()) return;
    // (8 lanes: emulated where the GPU has no XMX at all (the 125H, ARC_XMX_TRY=2); 16 lanes: where it has no 16-lane
    //  XMX, the 140T's 8-lane DPAS included)
    xw_emu_ = xw_lanes() == 8 ? !cl_.xmx : (!xmx_ || cl_.sg8);
    static std::mutex mu;
    static std::map<std::string, bool> known;
    const std::string key = cl_.device_id + "|" + std::to_string(xw_lanes()) + "|" + std::to_string(xw_tm()) + "x" +
                            std::to_string(xw_nc()) + (xw_emu_ ? "|emulated" : "") + (x2d_ ? "|2d" : "") + xw_pf_defs() +
                            "|pack" + std::to_string(xw_pack_cfg());
    std::lock_guard<std::mutex> g(mu);
    auto it = known.find(key);
    if (it == known.end()) {
      const bool ok = xw_check();
      it = known.emplace(key, ok).first;
      if (ok) {
        zlog::eprintf("inferzero: the H8 GEMMs on XMX with %d x %d tiles (%s, results checked bit for bit)%s%s\n",
                xw_tm(), 16 * xw_nc(), xw_lanes() == 8 ? "gemm_xw8, 8 lanes" : "gemm_xw", xw_emu_ ? " (DPAS emulated: tests)" : "",
                xw_nc() == 6 ? "; a width's rest after its 96s 64 columns wide" : "");
        if (!xw_pf_defs().empty())
          zlog::eprintf("inferzero: gemm_xw prefetches the %s of the chunk %d ahead (ARC_GEMM_XW_PF=%d)\n",
                        xw_pf_eff() == 1 ? "weights" : xw_pf_eff() == 2 ? "activations" : "weights and activations", xw_pfd(), xw_pf());
      } else
        zlog::eprintf("inferzero: %s differs from gemm.cl or does not build: not used (please report)\n",
                xw_lanes() == 8 ? "gemm_xw8" : "gemm_xw");
    }
    xw_ = it->second;
  }
  // random H8R weights (int8 values, per-column scales), int8 activations with per-token scales over three decades;
  // stored, in 3 split-K pieces, added and with the fused SwiGLU; two shapes (a partial row group, two and three
  // column blocks, a column offset and a row stride wider than N), against gemm.cl without XMX: bit for bit
  bool xw_check() {
    const int K = 2048, nsb = K / 256;
    uint32_t seed = 777;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    auto rndf = [&](float lo, float span) { return lo + (float)(rnd() % 10000) * (span / 10000.0f); };
    std::string bad;
    std::vector<cl_mem> bufs;
    struct Soft { bool& f; bool was; ~Soft() { f = was; } } soft{cl_.soft_fail, cl_.soft_fail};
    cl_.soft_fail = true;
    bool built = true;
    try {
      const int tm = xw_tm(), cols = 16 * xw_nc();
      std::vector<std::pair<int, int>> shapes = {{3 * tm, 2 * cols}, {2 * tm, 3 * cols}};
      // (NC 6: 96 + 64 parts, 96 + 128 (the 27B's widths have a 128-wide rest), 96 only, 64 only)
      if (xw_nc() == 6) shapes = {{3 * tm, 256}, {2 * tm, 224}, {2 * tm, 288}, {2 * tm, 128}};
      for (auto [M, N] : shapes) {
        const int ng = N / 16, ldc = N + 32, coff = 16;
        std::vector<uint32_t> wq((size_t)ng * nsb * 4 * 16 * 16), wh((size_t)ng * 16);
        for (auto& v : wq) v = rnd();
        for (auto& v : wh) {
          const float f = rndf(0.001f, 0.01f);
          memcpy(&v, &f, 4);
        }
        std::vector<int8_t> aq((size_t)M * K);
        for (auto& v : aq) v = (int8_t)(rnd() >> 24);
        std::vector<float> asc((size_t)M), c0((size_t)M * ldc);
        for (auto& v : asc) v = 0.02f * std::pow(10.0f, -rndf(0.0f, 3.0f));
        for (auto& v : c0) v = rndf(-2.0f, 4.0f);
        cl_mem bq = buf(wq.size() * 4, wq.data()), bh = buf(wh.size() * 4, wh.data()), ba = buf(aq.size(), aq.data()),
               bs = buf(asc.size() * 4, asc.data()), bb = buf((size_t)(K / 16) * M * 2), bc = buf(c0.size() * 4);
        bufs.insert(bufs.end(), {bq, bh, ba, bs, bb, bc});
        // (ARC_GEMM_XW_PACK: the same data in gemm_xw8's packed orders, made here as h8r_act / h8r_decode / h8r_weights
        //  make them, and each form used checked too: plain, activations, weights, both)
        const int pcfg = xw_pack_cfg(), nch = K / 64;
        cl_mem bqp = nullptr, bap = nullptr;
        if (pcfg & 2) {  // ([group][chunk][lo0, lo1, hi0, hi1][8 words][8 lanes] from [group][chunk][16 words][16 columns])
          std::vector<uint32_t> wp(wq.size());
          for (size_t gc = 0; gc < (size_t)ng * nch; ++gc)
            for (int j = 0; j < 16; ++j)
              for (int cl = 0; cl < 16; ++cl)
                wp[gc * 256 + ((j / 8) * 2 + cl / 8) * 64 + (j % 8) * 8 + cl % 8] = wq[gc * 256 + j * 16 + cl];
          bqp = buf(wp.size() * 4, wp.data());
          bufs.push_back(bqp);
        }
        if (pcfg & 1) {  // ([M / 8][chunk][2 halves][8 rows][8 words] from [M][K])
          std::vector<uint32_t> aw((size_t)M * K / 4), ap(aw.size());
          memcpy(aw.data(), aq.data(), aq.size());
          for (int m = 0; m < M; ++m)
            for (int c = 0; c < nch; ++c)
              for (int h = 0; h < 2; ++h)
                for (int l = 0; l < 8; ++l)
                  ap[((size_t)(m / 8) * nch + c) * 128 + h * 64 + (m % 8) * 8 + l] = aw[(size_t)m * (K / 4) + c * 16 + h * 8 + l];
          bap = buf(ap.size() * 4, ap.data());
          bufs.push_back(bap);
        }
        std::vector<int> forms = {0};
        for (int p = 1; p <= 3; ++p)
          if ((p & pcfg) == p) forms.push_back(p);
        for (int mode = 0; mode < 4; ++mode) {  // store, split-K pieces, add, fused SwiGLU
          const int acc = mode == 2 ? 1 : mode == 3 ? 2 : 0, pieces = mode == 1 ? 3 : 1;
          for (int wgy : {1, 2}) {
            if ((M / tm) % wgy) continue;
            for (int form : forms) {
            std::vector<float> out[2];
            for (int x = 0; x < 2; ++x) {
              CL_CHECK(clEnqueueWriteBuffer(cl_.queue, bc, CL_TRUE, 0, c0.size() * 4, c0.data(), 0, nullptr, nullptr));
              for (int sp = 0; sp < pieces; ++sp) {
                int s0 = nsb * sp / pieces, s1 = nsb * (sp + 1) / pieces, ac = sp == 0 ? acc : 1;
                int MM = M, NN = N, KK = K, ld = ldc, co = coff, rg = 2;
                if (x == 0) {  // (16 tokens a sub-group, whatever ARC_GEMM_TM: M is a multiple of 16, not of 32)
                  cl_kernel k = gemm_kernel(111, false, 0, 1, 16);
                  size_t gws[2] = {(size_t)N * (M / 16), 1}, lws[2] = {16, 1};
                  set_args(k, 0, bq, bh, ba, bs, bb, bc, MM, NN, KK, ld, co, ac, rg, s0, s1);
                  run("xw_check", k, 2, gws, lws);
                } else {
                  xw_piece("xw_check", (form & 2) ? bqp : bq, bh, (form & 1) ? bap : ba, bs, M, N, K, bc, ld, co, ac, rg, s0,
                           s1, wgy, form);
                }
              }
              out[x].resize(c0.size());
              CL_CHECK(clEnqueueReadBuffer(cl_.queue, bc, CL_TRUE, 0, out[x].size() * 4, out[x].data(), 0, nullptr, nullptr));
            }
            bool sane = false;  // (finite, and some output changed: else the comparison proves nothing)
            for (size_t i = 0; i < c0.size(); ++i) {
              if (!std::isfinite(out[0][i])) { sane = false; break; }
              sane = sane || out[0][i] != c0[i];
            }
            if (!sane || memcmp(out[0].data(), out[1].data(), c0.size() * 4)) {
              size_t diff = 0;
              for (size_t i = 0; i < c0.size(); ++i) diff += memcmp(&out[0][i], &out[1][i], 4) != 0;
              bad += " " + std::to_string(M) + "x" + std::to_string(N) + "/" +
                     (mode == 0 ? "store" : mode == 1 ? "split" : mode == 2 ? "add" : "swiglu") + "/wgy" +
                     std::to_string(wgy) + (form ? "/pack" + std::to_string(form) : std::string()) + "(" +
                     std::to_string(diff) + " differ)";
            }
            }
          }
        }
      }
    } catch (const std::exception& e) {
      built = false;
      bad += std::string(" build: ") + e.what();
    }
    for (cl_mem& b : bufs) free_buf(b);
    if (!bad.empty()) zlog::eprintf("inferzero: gemm_xw check:%s\n", bad.c_str());
    return built && bad.empty();
  }

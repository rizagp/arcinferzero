// The H8R prompt GEMM on XMX with a wider tile per sub-group (QUEUE 33.4 / 33.4b, kernels/gemm_xw.cl and
// gemm_xw8.cl): a part of class Engine (zero/engine.h), included inside the class body after zero/xe2/engine_xe2.h
// and engine_h8.h; not a header of its own. Off unless asked for: every path is unchanged without the switch.
//
// ARC_GEMM_XW=1: where H8R runs on XMX (h8_ with per-column scales: the Qwen dense models on discrete cards by
// default, the Arc 140T preset's chunks of 512+ tokens, ARC_H8=1 elsewhere), the H8 GEMMs whose shape it takes (M a
// multiple of XW_TM, N of 16 * XW_NC) go through the wide-tile kernel instead of gemm.cl / gemm_x8.cl; the others
// stay. Off until a B60 / a 140T measures it.
//   16-lane XMX (Xe2 / Xe3): gemm_xw.cl, XW_TM tokens x XW_NC 16-column groups (32 x 4 = 32 x 64).
//   8-lane XMX (the 140T, Alchemist-kind): gemm_xw8.cl, coalesced loads (a group's weights in 4 block reads, an
//   activation row in one), XW_TM x XW_NC (16 x 1 = 16 x 16, both halves, 128 registers).
//   ARC_GEMM_XW_TM=16 / 32 (32 on 16 lanes, 16 on 8): tokens per sub-group; ARC_GEMM_XW_NC=1 / 2 / 4 (4 on 16 lanes, 1 on 8; 4 is
//   16-lane only); ARC_GEMM_XW_GRF=128 / 256: the register file (256 on 16 lanes, 128 on 8, where 32 tokens spill).
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
  // (8 lanes: 16 x 16 at the default register file by default, the shape that does not spill there (ocloc arl-h,
  //  2026-11-11): the 140T's XMX GEMM gained 108 -> 165 tok/s from 128 registers' occupancy; 16 lanes: 32 x 64 at 256)
  int xw_tm() const {
    static const int e = getenv("ARC_GEMM_XW_TM") ? atoi(getenv("ARC_GEMM_XW_TM")) : 0;
    const int def = xw_lanes() == 8 ? 16 : 32;
    if (e == 16 || e == 32) return e;
    if (e) { static std::once_flag said; std::call_once(said, [&] { fprintf(stderr, "inferzero: ARC_GEMM_XW_TM=%d not supported (16 or 32): %d\n", e, def); }); }
    return def;
  }
  bool xw_grf256() const {
    static const char* e = getenv("ARC_GEMM_XW_GRF");
    if (e && std::string(e) == "128") return false;
    if (e && std::string(e) == "256") return true;
    return xw_lanes() == 16;
  }
  int xw_nc() const {
    static const int e = getenv("ARC_GEMM_XW_NC") ? atoi(getenv("ARC_GEMM_XW_NC")) : 0;
    const int def = xw_lanes() == 8 ? 1 : 4;
    if (e == 0) return def;
    if (e == 1 || e == 2 || (e == 4 && xw_lanes() == 16)) return e;
    static std::once_flag said;
    std::call_once(said, [&] { fprintf(stderr, "inferzero: ARC_GEMM_XW_NC=%d not supported here: %d\n", e, def); });
    return def;
  }
  static bool xw_try() {
    static const bool t = getenv("ARC_GEMM_XW_TRY") && atoi(getenv("ARC_GEMM_XW_TRY")) != 0;
    return t;
  }
  // (h8_ and xmx_ known: after the XMX and H8 checks of the constructor)
  bool xw_wanted() const {
    const bool on = getenv("ARC_GEMM_XW") && atoi(getenv("ARC_GEMM_XW")) != 0;
    return h8_ && h8_row() && (xw_try() || (on && xmx_));
  }
  cl_kernel xw_kernel() {
    const std::string defs = std::string(xw_grf256() ? "-cl-intel-256-GRF-per-thread " : "") + "-DXW_TM=" +
                             std::to_string(xw_tm()) + " -DXW_NC=" + std::to_string(xw_nc()) + (xw_emu_ ? " -DXW_EMU" : "");
    if (xw_lanes() == 8) return kernel("xw8", defs, "gemm_xw8");
    return kernel("xw", defs + (x2d_ && !xw_emu_ ? " -DX2D" : ""), "gemm_xw");
  }
  // one piece [s0, s1) of K: the grid of gemm_xw.cl's / gemm_xw8.cl's header (a sub-group of 16 or 8 work-items)
  void xw_run(const char* name, cl_kernel k, cl_mem wq, cl_mem wh, cl_mem aq, cl_mem asc, int M, int N, int K, cl_mem C,
              int ldc, int c_off, int a, int rgroup, int s0, int s1, int wgy) {
    const int tm = xw_tm(), nb = N / (16 * xw_nc()), ln = xw_lanes();
    size_t gws[2] = {(size_t)ln * nb * (M / tm / wgy), (size_t)wgy}, lws[2] = {(size_t)ln, (size_t)wgy};
    set_args(k, 0, wq, wh, aq, asc, C, M, N, K, ldc, c_off, a, rgroup, s0, s1);
    run(name, k, 2, gws, lws);
  }
  // h8_matmul's GEMM through gemm_xw / gemm_xw8, with gemm.cl's split-K pieces and row groups; false: a shape it does
  // not take (the old kernel runs it)
  bool xw_matmul(const char* name, cl_mem wq, cl_mem hh, int Mp, int N, int K, cl_mem C, int ldc, int c_off, int a) {
    const int tm = xw_tm();
    if (Mp % tm || N % (16 * xw_nc()) || K % 256) return false;
    static const int rg_env = getenv("ARC_GEMM_RG") ? std::max(1, atoi(getenv("ARC_GEMM_RG"))) : 4;
    static const int wgy_max = getenv("ARC_GEMM_WGY") ? atoi(getenv("ARC_GEMM_WGY")) : 8;
    // (the pieces gemm.cl's H8 GEMM takes, h8_matmul's own helper: the same sums per piece, so the same results)
    const int nsb = K / 256, splits = h8_splits(K, a);
    int wgy = 1;
    while (wgy < wgy_max && (Mp / tm) % (wgy * 2) == 0) wgy *= 2;
    cl_kernel k = xw_kernel();
    for (int sp = 0; sp < splits; ++sp)
      xw_run(name, k, wq, hh, h8_aq_, h8_asc_, Mp, N, K, C, ldc, c_off, sp == 0 ? a : 1, rg_env, nsb * sp / splits,
             nsb * (sp + 1) / splits, wgy);
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
                            std::to_string(xw_nc()) + (xw_emu_ ? "|emulated" : "") + (x2d_ ? "|2d" : "");
    std::lock_guard<std::mutex> g(mu);
    auto it = known.find(key);
    if (it == known.end()) {
      const bool ok = xw_check();
      it = known.emplace(key, ok).first;
      if (ok)
        fprintf(stderr, "inferzero: the H8 GEMMs on XMX with %d x %d tiles (%s, results checked bit for bit)%s\n",
                xw_tm(), 16 * xw_nc(), xw_lanes() == 8 ? "gemm_xw8, 8 lanes" : "gemm_xw", xw_emu_ ? " (DPAS emulated: tests)" : "");
      else
        fprintf(stderr, "inferzero: %s differs from gemm.cl or does not build: not used (please report)\n",
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
      for (auto [M, N] : {std::pair{3 * tm, 2 * cols}, std::pair{2 * tm, 3 * cols}}) {
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
        for (int mode = 0; mode < 4; ++mode) {  // store, split-K pieces, add, fused SwiGLU
          const int acc = mode == 2 ? 1 : mode == 3 ? 2 : 0, pieces = mode == 1 ? 3 : 1;
          for (int wgy : {1, 2}) {
            if ((M / tm) % wgy) continue;
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
                  xw_run("xw_check", xw_kernel(), bq, bh, ba, bs, M, N, K, bc, ld, co, ac, rg, s0, s1, wgy);
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
                     std::to_string(wgy) + "(" + std::to_string(diff) + " differ)";
            }
          }
        }
      }
    } catch (const std::exception& e) {
      built = false;
      bad += std::string(" build: ") + e.what();
    }
    for (cl_mem& b : bufs) free_buf(b);
    if (!bad.empty()) fprintf(stderr, "inferzero: gemm_xw check:%s\n", bad.c_str());
    return built && bad.empty();
  }

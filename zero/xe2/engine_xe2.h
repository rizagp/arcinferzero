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

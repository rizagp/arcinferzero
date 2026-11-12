// DeepSeek-V4.1-Flash (deepseek41): hyperparameters, weights, state and the layer (docs/deepseek41-spec.md).
// A part of class Engine (zero/engine.h), included inside the class body: not a header of its own.
//
// What it reuses: the int8 GEMMs (matmul) for every projection, the MoE path of engine_moe.h for the routed and shared
// experts (expert streaming included) with the router in f32 (gemv_bf16 on the f32 FFN input in hco_), a
// sqrt(softplus) routing with the selection bias (moe.cl -DROUTE_DS) and clamped SwiGLU (-DSWI_CLAMP).
// New (zero/kernels/ops_ds41.cl): the hyper-connection mixes with Sinkhorn, collapse / expand of the 4 streams,
// Engram, the window + compressed-row attention with sinks, the compressor, the indexer and its top-k.
// First version: f32 caches and plain kernels, checked against tools/ref_forward; the server's snapshots (ds_save) and
// saved conversations (ds_save_state); not yet slots, drafters, images or multi-GPU for this architecture.

  struct DsHp {
    int QL = 0, G = 0, OL = 0, W = 0, IH = 0, ID = 0, TOPK = 0, HC = 4, MIX = 24, iters = 20;
    int cand_src = -1, cand_k = 0, cand_bs = 1;
    int ngram = 0, hpn = 0, PD = 0, ncol = 0, pad_c = 0;
    float hc_eps = 1e-6f, clamp = 0.0f, wscale = 1.0f;
    double base0 = 1e4, base1 = 1.6e5, yfac = 1, yorig = 65536, bfast = 32, bslow = 1;
    std::vector<int> ratio, kv_src, ix_src, ple_layers;
    std::vector<int64_t> mult, offs, primes, tmap;
  } ds_;
  struct DsLayer {
    int rt = 0, kvs = -1, ixs = -1, eng = -1;  // compress ratio; its KV source and index source layers; Engram index
    bool kv_own = false, ix_own = false;
    GpuWeight q_a, q_b, kv, out_b, comp_kv, comp_gate, ix_k, ix_qb, ix_proj, eng_kv;
    std::vector<GpuWeight> out_a;  // (one per output group: rows g*OL .. of attn_output_a)
    cl_mem q_a_norm = nullptr, kv_norm = nullptr, sinks = nullptr, hc_fn[2] = {}, hc_base[2] = {}, hc_scale[2] = {},
           comp_norm = nullptr, ix_knorm = nullptr, eng_gate = nullptr, ebias = nullptr;
    cl_mem ring = nullptr;                         // window keys [W][D] (slot p % W)
    cl_mem cc = nullptr, ic = nullptr, pend = nullptr;  // KV source: compressed rows [cap][D], index keys [cap][ID],
    int cap = 0;                                   // the pending tokens' values / scores [rt - 1][2][D]
  };
  std::vector<DsLayer> dsl_;
  static constexpr int kDsPad = 8;  // (rope table rows before the chunk: the compressor's groups that started there)
  cl_mem ds_rt_[2] = {}, ds_mixA_ = nullptr, ds_mixF_ = nullptr, ds_qa_ = nullptr, ds_q_ = nullptr, ds_kv_ = nullptr,
         ds_cv_ = nullptr, ds_cg_ = nullptr, ds_lat_ = nullptr, ds_lat2_ = nullptr, ds_ik_ = nullptr, ds_iq_ = nullptr,
         ds_iw_ = nullptr, ds_sc_ = nullptr, ds_bsc_ = nullptr, ds_sel_ = nullptr, ds_nsel_ = nullptr,
         ds_bsel_ = nullptr, ds_nbsel_ = nullptr, ds_cand_ = nullptr, ds_att_ = nullptr, ds_oa_ = nullptr,
         ds_f_ = nullptr, ds_ekv_ = nullptr, ds_eng_[2] = {}, ds_kvh_ = nullptr;
  int ds_sc_q_ = 1, ds_nbmax_ = 1;  // (indexer scores: queries per pass; blocks of the candidate map)
  std::vector<int> ds_hist_;        // (the token ids before the current position, newest first, for the Engram hash)
  std::vector<float> ds_eng_host_;

  // the clamped SwiGLU's define (empty for every other model)
  std::string ds_swi_defs() const { return hp.ds41 && ds_.clamp > 0 ? " -DSWI_CLAMP=" + flt_def(ds_.clamp) : std::string(); }
  static std::string flt_def(float v) {
    char b[48];
    snprintf(b, sizeof b, "%.9gf", v);
    std::string s = b;
    if (s.find_first_of(".e") == std::string::npos) s.insert(s.size() - 1, ".0");
    return s;
  }

  void load_hparams_ds41() {
    const std::string a = hp.arch;
    auto I = [&](const std::string& k, int64_t def = -1) {
      const int64_t v = f_.get_int(a + "." + k, def);
      if (v < 0) throw std::runtime_error("missing hparam " + k);
      return (int)v;
    };
    auto Fl = [&](const std::string& k, double def) { return f_.get_float(a + "." + k, def); };
    auto ints = [&](const std::string& k) {
      const std::vector<int64_t>* v = f_.get_ints(a + "." + k);
      if (!v) throw std::runtime_error("missing hparam " + k);
      return *v;
    };
    hp.ds41 = hp.moe = hp.moe_std = true;
    hp.full_attn_interval = 1;  // (no recurrent layers)
    hp.n_layer = I("block_count");
    hp.n_embd = I("embedding_length");
    hp.n_head = I("attention.head_count");
    hp.n_head_kv = I("attention.head_count_kv");
    hp.head_dim = I("attention.key_length");
    hp.n_rot = I("rope.dimension_count");
    hp.eps = (float)Fl("attention.layer_norm_rms_epsilon", 1e-20);
    hp.rope_base = (float)Fl("rope.freq_base", 1e4);
    hp.hc_n = I("hyper_connection.count");
    hp.n_expert = I("expert_count");
    hp.n_expert_used = I("expert_used_count");
    hp.n_ff_exp = I("expert_feed_forward_length");
    hp.n_ff_shexp = hp.n_ff_exp * I("expert_shared_count", 1);
    hp.n_ff_exp_pad = padded_k(hp.n_ff_exp);
    DsHp& d = ds_;
    d.QL = I("attention.q_lora_rank");
    d.G = I("attention.output_group_count");
    d.OL = I("attention.output_lora_rank");
    d.W = I("attention.sliding_window");
    d.IH = I("attention.indexer.head_count");
    d.ID = I("attention.indexer.key_length");
    d.TOPK = I("attention.indexer.top_k");
    d.HC = hp.hc_n;
    d.MIX = (2 + d.HC) * d.HC;
    d.iters = I("hyper_connection.sinkhorn_iterations", 20);
    d.hc_eps = (float)Fl("hyper_connection.epsilon", 1e-6);
    d.wscale = (float)Fl("expert_weights_scale", 1.0);
    d.base0 = Fl("rope.freq_base", 1e4);
    d.base1 = Fl("attention.compress_rope_freq_base", 1.6e5);
    d.yfac = Fl("rope.scaling.factor", 1);
    d.yorig = I("rope.scaling.original_context_length", 65536);
    d.bfast = Fl("rope.scaling.yarn_beta_fast", 32);
    d.bslow = Fl("rope.scaling.yarn_beta_slow", 1);
    const auto cr = ints("attention.compress_ratios");
    if ((int)cr.size() < hp.n_layer) throw std::runtime_error("deepseek41: compress_ratios shorter than block_count");
    for (int il = 0; il < hp.n_layer; ++il) d.ratio.push_back((int)cr[il]);  // (the trailing entries: the drafter's layers)
    for (int64_t v : ints("attention.kv_source_layers")) d.kv_src.push_back((int)v);
    for (int64_t v : ints("attention.indexer.source_layers")) d.ix_src.push_back((int)v);
    d.cand_src = (int)f_.get_int(a + ".attention.indexer.candidate_source_layer", -1);
    d.cand_k = I("attention.indexer.candidate_top_k_blocks", 0);
    d.cand_bs = I("attention.indexer.candidate_block_size", 1);
    for (int64_t v : ints("ple.layers")) d.ple_layers.push_back((int)v);
    d.ngram = I("ple.ngram_size");
    d.hpn = I("ple.heads_per_ngram");
    d.PD = I("embedding_length_per_layer_input");
    d.ncol = (d.ngram - 1) * d.hpn;
    d.mult = ints("ple.layer_multipliers");
    d.offs = ints("ple.head_offsets");
    d.primes = ints("ple.head_vocab_sizes");
    d.tmap = ints("engram.token_map");
    const int pad_id = I("engram.pad_token_id", 2);
    d.pad_c = pad_id < (int)d.tmap.size() ? (int)d.tmap[pad_id] : 0;
    if (const auto* ce = f_.get_floats(a + ".swiglu_clamp_exp")) {
      d.clamp = ce->empty() ? 0.0f : (float)(*ce)[0];
      for (double v : *ce)
        if ((float)v != d.clamp) throw std::runtime_error("deepseek41: per-layer SwiGLU clamps that differ");
    }
    if (const auto* cs = f_.get_floats(a + ".swiglu_clamp_shexp"))
      for (double v : *cs)
        if ((float)v != d.clamp) throw std::runtime_error("deepseek41: a shared-expert SwiGLU clamp unlike the experts'");
    tok_embd_ = &f_.get("token_embd.weight");
    if (tok_embd_->dims.size() < 2 || tok_embd_->dims[0] != (uint64_t)hp.n_embd) throw std::runtime_error("deepseek41: bad token_embd");
    hp.n_vocab = (int)tok_embd_->dims[1];
    // (the kernels' assumptions: ops_ds41.cl ds_attn holds 512-wide heads in 16 lanes, ds_mix 4 streams, the GEMMs K % 256)
    if (hp.head_dim != 512 || hp.n_head_kv != 1 || hp.n_rot != 64 || d.HC != 4 || hp.n_embd % 256 || d.QL % 256 ||
        hp.n_head % d.G || (hp.n_head / d.G * hp.head_dim) % 256 || (d.G * d.OL) % 256 || d.OL % 16 || d.ID != 128 ||
        d.IH % 16 || d.IH * d.ID % 256 || d.TOPK > 2048 || d.cand_k > 2048 || hp.n_expert % 16 ||
        hp.n_expert_used > 16 || hp.n_ff_exp % 256 || d.W < 1 || (d.ncol * d.PD) % 256 || d.ngram > 8 ||
        (int)d.mult.size() < d.ngram * (int)d.ple_layers.size() || (int)d.offs.size() < d.ncol * (int)d.ple_layers.size() ||
        (int)d.primes.size() < d.ncol * (int)d.ple_layers.size() || d.ple_layers.size() > 2)
      throw std::runtime_error("unsupported DeepSeek-V4.1 dimensions");
    // per layer: KV source and index source (the most recent at or before it), Engram
    dsl_.assign(hp.n_layer, {});
    auto has = [](const std::vector<int>& v, int il) { return std::find(v.begin(), v.end(), il) != v.end(); };
    for (int il = 0; il < hp.n_layer; ++il) {
      DsLayer& L = dsl_[il];
      L.rt = d.ratio[il];
      if (L.rt < 0 || L.rt > 2) throw std::runtime_error("deepseek41: compress ratio " + std::to_string(L.rt));
      L.kv_own = L.rt > 0 && has(d.kv_src, il);
      L.ix_own = L.rt > 0 && has(d.ix_src, il);
      if (L.rt)
        for (int s : d.kv_src) if (s <= il) L.kvs = s;
      if (L.rt)
        for (int s : d.ix_src) if (s <= il) L.ixs = s;
      if (L.rt && (L.kvs < 0 || L.ixs < 0 || d.ratio[L.kvs] != L.rt || dsl_[L.ixs].kvs != L.kvs))
        throw std::runtime_error("deepseek41: layer " + std::to_string(il) + " without its KV / index source");
      for (size_t e = 0; e < d.ple_layers.size(); ++e)
        if (d.ple_layers[e] == il) L.eng = (int)e;
    }
    if (d.cand_k > 0 && (d.cand_src < 0 || d.cand_src >= hp.n_layer || !dsl_[d.cand_src].ix_own))
      throw std::runtime_error("deepseek41: the candidate source layer is not an index layer");
    if (opt.slots > 1 || opt.mtp || !opt.dflash.empty() || opt.tp_size > 1)
      throw std::runtime_error("DeepSeek-V4.1: slots, drafters and multi-GPU are not supported yet");
  }

  void load_weights_ds41() {
    layers_.resize(hp.n_layer);
    const DsHp& d = ds_;
    const int E = hp.n_embd, D = hp.head_dim, H = hp.n_head, GK = H / d.G * D;
    for (int il = 0; il < hp.n_layer; ++il) {
      Layer& L = layers_[il];
      DsLayer& S = dsl_[il];
      L.il = il;
      const std::string p = "blk." + std::to_string(il) + ".";
      L.attn_norm = load_vec(p + "attn_norm.weight", E);
      L.post_norm = load_vec(p + "ffn_norm.weight", E);
      S.q_a = load_w(p + "attn_q_a.weight");
      S.q_a_norm = load_vec(p + "attn_q_a_norm.weight", d.QL);
      S.q_b = load_w(p + "attn_q_b.weight");
      S.kv = load_w(p + "attn_kv.weight");
      S.kv_norm = load_vec(p + "attn_kv_a_norm.weight", D);
      S.sinks = load_vec(p + "attn_sinks.weight", H);
      expect(S.q_a, d.QL, E, p + "attn_q_a.weight");
      expect(S.q_b, H * D, d.QL, p + "attn_q_b.weight");
      expect(S.kv, D, E, p + "attn_kv.weight");
      {  // the grouped output LoRA: one matrix per group of its rows
        const GgufTensor& t = f_.get(p + "attn_output_a.weight");
        if (t.nelements() != (uint64_t)GK * d.G * d.OL || t.dims[0] != (uint64_t)GK)
          throw std::runtime_error("unexpected tensor shape: " + t.name);
        const size_t rb = t.nbytes / ((size_t)d.G * d.OL);
        for (int g = 0; g < d.G; ++g) {
          const uint8_t* src = t.data + (size_t)g * d.OL * rb;
          if (t.type == GGML_F32 || t.type == GGML_F16 || t.type == GGML_BF16) {
            std::vector<float> v((size_t)d.OL * GK);
            dequant_row(t.type, src, (int)v.size(), v.data());
            auto q8 = quantize_rows_q8_0(v.data(), d.OL, GK);
            S.out_a.push_back(make_weight(GGML_Q8_0, q8.data(), q8.size(), d.OL, GK));
          } else {
            S.out_a.push_back(make_weight(t.type, src, (size_t)d.OL * rb, d.OL, GK));
          }
        }
      }
      S.out_b = load_w(p + "attn_output_b.weight");
      expect(S.out_b, E, d.G * d.OL, p + "attn_output_b.weight");
      for (int b = 0; b < 2; ++b) {
        const std::string q = p + (b ? "hc_ffn" : "hc_attn");
        S.hc_fn[b] = load_vec(q + "_fn.weight", d.MIX * d.HC * E);
        S.hc_base[b] = load_vec(q + "_base.weight", d.MIX);
        S.hc_scale[b] = load_vec(q + "_scale.weight", 3);
      }
      S.ebias = load_vec(p + "exp_probs_b.bias", hp.n_expert);
      if (S.kv_own) {
        S.comp_kv = load_w(p + "attn_compressor_kv.weight");
        expect(S.comp_kv, D, E, p + "attn_compressor_kv.weight");
        if (S.rt > 1) {
          S.comp_gate = load_w(p + "attn_compressor_gate.weight");
          expect(S.comp_gate, D, E, p + "attn_compressor_gate.weight");
        }
        S.comp_norm = load_vec(p + "attn_compressor_norm.weight", D);
        S.ix_k = load_w(p + "indexer.attn_k.weight");
        expect(S.ix_k, d.ID, D, p + "indexer.attn_k.weight");
        S.ix_knorm = load_vec(p + "indexer.k_norm.weight", d.ID);
        S.cap = ctx / S.rt + 2;
        S.cc = buf((size_t)S.cap * D * ds_kvb());
        S.ic = buf((size_t)S.cap * d.ID * ds_kvb());
        S.pend = buf(ds_pend_bytes(S));
      }
      if (S.ix_own) {
        S.ix_qb = load_w(p + "indexer.attn_q_b.weight");
        expect(S.ix_qb, d.IH * d.ID, d.QL, p + "indexer.attn_q_b.weight");
        S.ix_proj = load_w(p + "indexer.proj.weight");
        expect(S.ix_proj, d.IH, E, p + "indexer.proj.weight");
      }
      if (S.eng >= 0) {
        S.eng_kv = load_w(p + "engram_kv.weight");
        expect(S.eng_kv, (d.HC + 1) * E, d.ncol * d.PD, p + "engram_kv.weight");
        S.eng_gate = load_vec(p + "engram_gate.weight", d.HC * E);
      }
      S.ring = buf((size_t)d.W * D * ds_kvb());
      load_moe(L, p);
      clFinish(cl_.queue);
    }
    ple_table_ = &f_.get("per_layer_token_embd.weight");
    if ((int)ple_table_->dims[0] != d.PD) throw std::runtime_error("unexpected Engram table shape");
    for (size_t e = 0; e < d.ple_layers.size(); ++e)
      for (int c = 0; c < d.ncol; ++c)
        if (d.primes[e * d.ncol + c] < 2 || d.offs[e * d.ncol + c] < 0 ||
            (uint64_t)(d.offs[e * d.ncol + c] + d.primes[e * d.ncol + c]) > ple_table_->dims[1])
          throw std::runtime_error("Engram hash columns past the table");
  }

  void alloc_ds41() {
    const DsHp& d = ds_;
    const size_t M = max_chunk;
    const int E = hp.n_embd, D = hp.head_dim, H = hp.n_head;
    if (!hco_) hco_ = buf(M * E * 4);
    for (int i = 0; i < 2; ++i) ds_rt_[i] = buf((M + kDsPad) * (hp.n_rot / 2) * 8);
    ds_mixA_ = buf(M * d.MIX * 4);
    ds_mixF_ = buf(M * d.MIX * 4);
    ds_qa_ = buf(M * d.QL * 4);
    ds_q_ = buf(M * H * D * 4);
    ds_kv_ = buf(M * D * 4);
    ds_kvh_ = buf(M * D * ds_kvb());  // (the chunk's keys as the caches hold them)
    ds_cv_ = buf(M * D * 4);
    ds_cg_ = buf(M * D * 4);
    ds_lat_ = buf((M + 16) * D * 4);
    ds_lat2_ = buf((M + 16) * D * 4);
    ds_ik_ = buf((M + 16) * d.ID * 4);
    ds_iq_ = buf(M * d.IH * d.ID * 4);
    ds_iw_ = buf(M * d.IH * 4);
    // (indexer scores: queries per pass so that a pass's scores stay within 64M floats)
    int min_rt = 2;
    for (const DsLayer& L : dsl_) if (L.rt) min_rt = std::min(min_rt, L.rt);
    const size_t rows = (size_t)ctx / min_rt + 64;
    ds_sc_q_ = (int)std::max<size_t>(1, std::min<size_t>(M, ((size_t)64 << 20) / rows));
    ds_sc_ = buf((size_t)ds_sc_q_ * rows * 4);
    ds_nbmax_ = (int)((rows + d.cand_bs - 1) / d.cand_bs);
    ds_bsc_ = buf((size_t)ds_sc_q_ * ds_nbmax_ * 4);
    ds_sel_ = buf(M * std::max(1, d.TOPK) * 4);
    ds_nsel_ = buf(M * 4);
    ds_bsel_ = buf(M * std::max(1, d.cand_k) * 4);
    ds_nbsel_ = buf(M * 4);
    ds_cand_ = buf((M * ds_nbmax_ + 3) / 4 * 4);
    ds_att_ = buf(M * H * D * 4);
    ds_oa_ = buf(M * d.G * d.OL * 4);
    ds_f_ = buf(M * E * 4);
    ds_ekv_ = buf(M * (d.HC + 1) * E * 4);
    for (size_t e = 0; e < d.ple_layers.size(); ++e) ds_eng_[e] = buf(M * d.ncol * d.PD * 4);
    ds_eng_host_.resize(M * d.ncol * d.PD);
  }

  // Per chunk (from embed): the RoPE tables of positions pos - kDsPad .. pos + T - 1 (window layers plain, compressed
  // layers with YaRN's frequency interpolation), and the Engram rows of the chunk's tokens.
  void ds_prepare(const int* tokens, int T) {
    const DsHp& d = ds_;
    const int R = hp.n_rot, n = T + kDsPad;
    for (int ty = 0; ty < 2; ++ty) {
      const double theta = ty ? d.base1 : d.base0;
      std::vector<double> f(R / 2);
      for (int i = 0; i < R / 2; ++i) f[i] = std::pow(theta, -2.0 * i / R);
      if (ty) {
        auto cd = [&](double r) { return R * std::log(d.yorig / (2 * 3.14159265358979323846 * r)) / (2 * std::log(theta)); };
        const double lo = std::max(std::floor(cd(d.bfast)), 0.0), hi = std::min(std::ceil(cd(d.bslow)), (double)R - 1);
        for (int i = 0; i < R / 2; ++i) {
          const double ramp = std::clamp((i - lo) / std::max(hi - lo, 1e-3), 0.0, 1.0);
          f[i] = f[i] / d.yfac * ramp + f[i] * (1 - ramp);
        }
      }
      std::vector<float> tab((size_t)n * R);
      for (int r = 0; r < n; ++r) {
        const double p = (double)pos - kDsPad + r;
        for (int i = 0; i < R / 2; ++i) {
          tab[(size_t)r * R + 2 * i] = (float)std::cos(p * f[i]);
          tab[(size_t)r * R + 2 * i + 1] = (float)std::sin(p * f[i]);
        }
      }
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, ds_rt_[ty], CL_TRUE, 0, tab.size() * 4, tab.data(), 0, nullptr, nullptr));
    }
    // Engram: per token, its compressed id and those of the ngram - 1 before it (missing or out of range: the pad id)
    const int ng = d.ngram, nc = d.ncol, PD = d.PD;
    auto cid = [&](int id) -> int64_t { return id >= 0 && id < (int)d.tmap.size() && d.tmap[id] >= 0 ? d.tmap[id] : d.pad_c; };
    const size_t prb = ple_table_->nbytes / ple_table_->dims[1];
    for (size_t e = 0; e < d.ple_layers.size(); ++e) {
#pragma omp parallel for if (T > 16)
      for (int t = 0; t < T; ++t) {
        uint64_t tk[8];
        for (int k = 0; k < ng; ++k) {
          const int j = t - k;
          const int id = j >= 0 ? tokens[j] : (-j - 1 < (int)ds_hist_.size() ? ds_hist_[-j - 1] : -1);
          tk[k] = (uint64_t)(id >= 0 || j >= 0 ? cid(id) : d.pad_c);
        }
        uint64_t r = tk[0] * (uint64_t)d.mult[e * ng];
        for (int o = 1; o < ng; ++o) {
          r ^= tk[o] * (uint64_t)d.mult[e * ng + o];
          for (int h = 0; h < d.hpn; ++h) {
            const int col = (o - 1) * d.hpn + h;
            const uint64_t row = r % (uint64_t)d.primes[e * nc + col] + (uint64_t)d.offs[e * nc + col];
            dequant_row(ple_table_->type, ple_table_->data + row * prb, PD, &ds_eng_host_[((size_t)t * nc + col) * PD]);
          }
        }
      }
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, ds_eng_[e], CL_TRUE, 0, (size_t)T * nc * PD * 4, ds_eng_host_.data(), 0,
                                    nullptr, nullptr));
    }
    // the history after the chunk (newest first)
    std::vector<int> h;
    for (int k = 0; k < ng - 1; ++k) {
      const int j = T - 1 - k;
      h.push_back(j >= 0 ? tokens[j] : (-j - 1 < (int)ds_hist_.size() ? ds_hist_[-j - 1] : -1));
    }
    ds_hist_ = h;
  }

  // A snapshot of what a step leaves besides the position-indexed compressed rows and index keys (rows below
  // pos / ratio stay as they are until a later pass at those positions writes them again, which the server's dropping
  // of later snapshots on a restore covers, as for a KV cache): every layer's window ring, every KV source's pending
  // compressor rows, the Engram token history.
  void ds_save(std::vector<std::vector<uint8_t>>& v, std::vector<int>& h) {
    clFinish(cl_.queue);
    v.clear();
    for (DsLayer& L : dsl_) {
      v.push_back(read_buf(L.ring, (size_t)ds_.W * hp.head_dim * ds_kvb()));
      if (L.kv_own) v.push_back(read_buf(L.pend, ds_pend_bytes(L)));
    }
    h = ds_hist_;
  }
  void ds_load(const std::vector<std::vector<uint8_t>>& v, const std::vector<int>& h) {
    size_t k = 0;
    for (DsLayer& L : dsl_) {
      write_buf(L.ring, v.at(k++));
      if (L.kv_own) write_buf(L.pend, v.at(k++));
    }
    ds_hist_ = h;
  }
  // The key caches in f16 (the window ring and the chunk's keys, the compressed rows, the index keys: half the
  // memory, ~325 MB at 128K of context on the real model; its reference simulates fp8 / fp4 there); ARC_DS_KV32=1: f32
  static bool ds_kv32() {
    static const bool v = getenv("ARC_DS_KV32") && atoi(getenv("ARC_DS_KV32")) != 0;
    return v;
  }
  static int ds_kvb() { return ds_kv32() ? 4 : 2; }
  static std::string ds_kv_defs() { return ds_kv32() ? std::string() : std::string(" -DDS_KVH"); }
  size_t ds_pend_bytes(const DsLayer& L) const { return (size_t)2 * hp.head_dim * 4 * std::max(1, L.rt - 1); }

  // A saved conversation (save_state / load_state, format version 3): header [magic, 3, layers, embd, vocab, the
  // model's DeepSeek shape (window, KV sources' ratios), pos, the Engram history], the snapshots [count, then per
  // snapshot: pos, its history], the host snapshot's [pos, history]; then the device ranges (every layer's ring, every
  // KV source's compressed rows and index keys below the highest saved position, its pending rows), then the
  // snapshots' and the host snapshot's copies (ds_save's vectors, in order).
  int ds_shape_code() const {  // (unsigned arithmetic: it wraps on 40 layers; stored as its 32 bits)
    uint32_t c = (uint32_t)ds_.W * 31u + (uint32_t)ds_.ID;
    for (const DsLayer& L : dsl_) c = c * 7u + (L.kv_own ? 3u + (uint32_t)L.rt : 0u);
    // (f16 caches: another code, so a file saved with f32 caches does not load into them; f32: as before 2026-11-12)
    return (int)(ds_kvb() == 4 ? c : c * 5u + 2u);
  }
  int ds_saved_hi() const {
    int hi = pos;
    for (const Snap& s : snaps_) hi = std::max(hi, s.pos);
    return std::max(hi, hsnap_.pos);
  }
  std::vector<Seg> ds_segs(int hi) {
    std::vector<Seg> v;
    for (DsLayer& L : dsl_) {
      v.push_back({L.ring, 0, (size_t)ds_.W * hp.head_dim * ds_kvb()});
      if (!L.kv_own) continue;
      const size_t n = (size_t)(hi / L.rt);
      v.push_back({L.cc, 0, n * hp.head_dim * ds_kvb()});
      v.push_back({L.ic, 0, n * ds_.ID * ds_kvb()});
      v.push_back({L.pend, 0, ds_pend_bytes(L)});
    }
    return v;
  }
  std::vector<int32_t> ds_header(bool host_snapshot) const {
    std::vector<int32_t> h = {kStateMagic, 3, hp.n_layer, hp.n_embd, hp.n_vocab, ds_shape_code(), pos,
                              (int)ds_hist_.size()};
    h.insert(h.end(), ds_hist_.begin(), ds_hist_.end());
    h.push_back((int)snaps_.size());
    for (const Snap& s : snaps_) {
      h.push_back(s.pos);
      h.push_back(s.pos >= 0 ? (int)s.dsh.size() : 0);
      if (s.pos >= 0) h.insert(h.end(), s.dsh.begin(), s.dsh.end());
    }
    const bool hs = host_snapshot && hsnap_.pos >= 0;
    h.push_back(hs ? hsnap_.pos : -1);
    h.push_back(hs ? (int)hsnap_.dsh.size() : 0);
    if (hs) h.insert(h.end(), hsnap_.dsh.begin(), hsnap_.dsh.end());
    return h;
  }
  size_t ds_saved_size() {
    size_t n = ds_header(true).size() * 4;
    for (const Seg& s : ds_segs(ds_saved_hi())) n += s.n;
    for (const Snap& s : snaps_)
      if (s.pos >= 0)
        for (const auto& v : s.ds) n += v.size();
    if (hsnap_.pos >= 0)
      for (const auto& v : hsnap_.ds) n += v.size();
    return n;
  }
  bool ds_save_state(const Sink& out, bool host_snapshot) {
    clFinish(cl_.queue);
    const std::vector<int32_t> h = ds_header(host_snapshot);
    const int hi = std::max({pos, host_snapshot ? hsnap_.pos : -1, [&] { int m = -1; for (const Snap& s : snaps_) m = std::max(m, s.pos); return m; }()});
    bool ok = out(h.data(), h.size() * 4) && segs_out(ds_segs(hi), out);
    for (const Snap& s : snaps_)
      if (s.pos >= 0)
        for (const auto& v : s.ds) ok = ok && out(v.data(), v.size());
    if (host_snapshot && hsnap_.pos >= 0)
      for (const auto& v : hsnap_.ds) ok = ok && out(v.data(), v.size());
    return ok;
  }
  // (reset() and the snapshot drops have run: load_state)
  bool ds_load_state(FILE* f) {
    auto fail = [&] {
      reset();
      for (int i = 0; i < (int)snaps_.size(); ++i) drop_snapshot(i);
      drop_host_snapshot();
      return false;
    };
    const int nh = ds_.ngram - 1;
    auto hist = [&](std::vector<int>& hv) {  // (a history: its count, then the ids; -1 = none)
      int32_t c = -1;
      if (!get(f, &c, 4) || c < 0 || c > nh) return false;
      hv.assign(c, -1);
      if (c && !get(f, hv.data(), (size_t)c * 4)) return false;
      for (int id : hv)
        if (id < -1 || id >= hp.n_vocab) return false;
      return true;
    };
    int32_t h[7];
    if (!get(f, h, sizeof h) || h[0] != kStateMagic || h[1] != 3 || h[2] != hp.n_layer || h[3] != hp.n_embd ||
        h[4] != hp.n_vocab || h[5] != ds_shape_code() || h[6] < 0 || h[6] > ctx)
      return false;
    std::vector<int> hcur;
    if (!hist(hcur)) return false;
    int32_t ns = 0;
    if (!get(f, &ns, 4) || ns < 0 || ns > 8) return false;
    snaps_.assign(ns, Snap());
    int hi = h[6];
    for (Snap& s : snaps_) {
      int32_t sp = -1;
      if (!get(f, &sp, 4) || sp < -1 || sp > ctx) return fail();
      if (!hist(s.dsh)) return fail();
      s.pos = sp;
      hi = std::max(hi, sp);
    }
    int32_t hpos = -1;
    if (!get(f, &hpos, 4) || hpos < -1 || hpos > ctx) return fail();
    std::vector<int> hh;
    if (!hist(hh)) return fail();
    hi = std::max(hi, (int)hpos);
    if (!segs_in(ds_segs(hi), f)) return fail();
    // the snapshots' copies: sizes as ds_save makes them on this model
    std::vector<size_t> sizes;
    for (const DsLayer& L : dsl_) {
      sizes.push_back((size_t)ds_.W * hp.head_dim * ds_kvb());
      if (L.kv_own) sizes.push_back(ds_pend_bytes(L));
    }
    auto copies = [&](std::vector<std::vector<uint8_t>>& v) {
      v.clear();
      for (size_t n : sizes) {
        v.emplace_back(n);
        if (!get(f, v.back().data(), n)) return false;
      }
      return true;
    };
    for (Snap& s : snaps_)
      if (s.pos >= 0 && !copies(s.ds)) return fail();
    if (hpos >= 0) {
      if (!copies(hsnap_.ds)) return fail();
      hsnap_.dsh = hh;
      hsnap_.pos = hpos;
    }
    ds_hist_ = hcur;
    pos = h[6];
    return true;
  }

  // (two kernels, 16 work-groups a token: ops_ds41.cl ds_mix_part / ds_mix_fin; ARC_DS_MIX1=1: the one-work-group
  //  ds_mix of before, 0.78 ms a site and token on the laptop at the real widths)
  void ds_mix(cl_mem fn, cl_mem base, cl_mem scale, int T, cl_mem out) {
    int hcd = hp.hce(), it = ds_.iters, TT = T;
    float eps = hp.eps, he = ds_.hc_eps;
    static const bool one = getenv("ARC_DS_MIX1") && atoi(getenv("ARC_DS_MIX1")) != 0;
    if (one || hcd % 16) {
      cl_kernel k = kernel("ops", "", "ds_mix");
      set_args(k, 0, x_, hcd, fn, base, scale, eps, he, it, out);
      size_t gws[2] = {256, (size_t)T}, lws[2] = {256, 1};
      run("ds_mix", k, 2, gws, lws);
      return;
    }
    if (!ds_mixp_) ds_mixp_ = buf((size_t)max_chunk * 16 * (ds_.MIX + 1) * 4);
    cl_kernel k1 = kernel("ops", "", "ds_mix_part");
    set_args(k1, 0, x_, hcd, fn, ds_mixp_);
    size_t g1[2] = {256 * 16, (size_t)T}, l1[2] = {256, 1};
    run("ds_mix", k1, 2, g1, l1);
    cl_kernel k2 = kernel("ops", "", "ds_mix_fin");
    set_args(k2, 0, ds_mixp_, hcd, TT, base, scale, eps, he, it, out);
    size_t g2 = (size_t)(T + 63) / 64 * 64, l2 = 64;
    run("ds_mix", k2, 1, &g2, &l2);
  }
  cl_mem ds_mixp_ = nullptr;  // (ds_mix_part's sums: [max_chunk][16][25])
  cl_mem ds_apart_ = nullptr;  // (ds_attn_split's partials: [8 rows][heads][8][D + 2])
  // collapse + norm of rows r0 .. r0 + n - 1 (mix rows alike): f32 into hco_, quantized into aq_ (Mp rows)
  void ds_collapse(cl_mem mix, int first, cl_mem normw, int r0, int n, int Mp) {
    cl_kernel k = kernel("ops", "", "ds_collapse");
    int ro = r0, fi = first, E = hp.n_embd, MM = Mp, nn = n, a32 = 1;
    float eps = hp.eps;
    set_args(k, 0, x_, ro, mix, fi, E, normw, eps, MM, nn, aq_, asc_, absum_, hco_, a32);
    size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
    run("ds_collapse", k, 2, gws, lws);
  }
  void ds_expand(cl_mem f, cl_mem mix, int T) {
    cl_kernel k = kernel("ops", "", "ds_expand");
    int E = hp.n_embd;
    set_args(k, 0, x_, f, mix, E);
    size_t gws[2] = {(size_t)E, (size_t)T};
    run("ds_expand", k, 2, gws, nullptr);
  }
  // (drow0: the first destination row: a cache's next rows without a view per position; to_kv: dst is a key cache,
  //  f16 or f32 as ds_kv_defs says)
  void ds_norm_rope(cl_mem src, cl_mem dst, int K, int nrows, cl_mem w, int ty, int rpp, int pmul, int poff, float sign,
                    int drow0 = 0, bool to_kv = false) {
    if (nrows <= 0) return;
    // (audit 2026-11-12 #6: the kernel holds a row in float v[32] per lane, 16 lanes, in pairs: K / 16 values a lane,
    //  at most 32 and even. Callers pass 512 and 128, both checked at load; this keeps a new caller from overflowing)
    if (K <= 0 || K > 512 || K % 32) throw std::logic_error("ds_norm_rope: rows of " + std::to_string(K) + " (must be <= 512, a multiple of 32)");
    cl_kernel k = kernel("ops", to_kv ? ds_kv_defs() + " -DDS_DST_KV" : std::string(), "ds_norm_rope");
    int d0 = drow0, KK = K, un = w ? 1 : src == dst && !to_kv ? 2 : 0, R = hp.n_rot, rp = rpp, pm = pmul, po = poff;
    float eps = hp.eps, sg = sign;
    set_args(k, 0, src, dst, d0, KK, w ? w : src, un, eps, ds_rt_[ty], R, rp, pm, po, sg);
    size_t gws[2] = {16, (size_t)nrows}, lws[2] = {16, 1};
    run("ds_rope", k, 2, gws, lws);
  }
  // a matmul over n rows of aq_ (n may differ from the chunk's: the compressed rows)
  void ds_matmul_rows(const char* name, const GpuWeight& w, int n, cl_mem C, int ldc) {
    const int saved = cur_T_;
    cur_T_ = n;
    matmul(name, w, (n + 15) / 16 * 16, C, ldc, 0, 0, true);
    cur_T_ = saved;
  }

  void ds_layer(Layer& L0, int T, int Mp) {
    DsLayer& L = dsl_[L0.il];
    const DsHp& d = ds_;
    const int E = hp.n_embd, D = hp.head_dim, H = hp.n_head, il = L0.il, ty = L.rt ? 1 : 0;
    // Engram before the block
    if (L.eng >= 0) {
      quant_rows(ds_eng_[L.eng], d.ncol * d.PD, Mp, T, 1);
      matmul("ds_engram_kv", L.eng_kv, Mp, ds_ekv_, (d.HC + 1) * E, 0, 0, true);
      cl_kernel k = kernel("ops", "", "ds_engram");
      float eps = hp.eps;
      int EE = E;
      set_args(k, 0, x_, ds_ekv_, L.eng_gate, EE, eps);
      size_t gws[2] = {256, (size_t)T * d.HC}, lws[2] = {256, 1};
      run("ds_engram", k, 2, gws, lws);
    }
    // attention site
    ds_mix(L.hc_fn[0], L.hc_base[0], L.hc_scale[0], T, ds_mixA_);
    ds_collapse(ds_mixF_, il == 0 ? 1 : 0, L0.attn_norm, 0, T, Mp);
    matmul("ds_q_a", L.q_a, Mp, ds_qa_, d.QL, 0, 0, true);
    matmul("ds_kv", L.kv, Mp, ds_kv_, D, 0, 0, true);
    if (L.kv_own) {
      matmul("ds_comp_kv", L.comp_kv, Mp, ds_cv_, D, 0, 0, true);
      if (L.rt > 1) matmul("ds_comp_gate", L.comp_gate, Mp, ds_cg_, D, 0, 0, true);
    }
    if (L.ix_own) matmul("ds_ix_proj", L.ix_proj, Mp, ds_iw_, d.IH, 0, 0, true);
    rmsnorm_quant(ds_qa_, 0, L.q_a_norm, d.QL, Mp, T, 1);
    matmul("ds_q_b", L.q_b, Mp, ds_q_, H * D, 0, 0, true);
    if (L.ix_own) matmul("ds_ix_q_b", L.ix_qb, Mp, ds_iq_, d.IH * d.ID, 0, 0, true);
    ds_norm_rope(ds_kv_, ds_kvh_, D, T, L.kv_norm, ty, 1, 1, kDsPad, 1.0f, 0, true);
    ds_norm_rope(ds_q_, ds_q_, D, T * H, nullptr, ty, H, 1, kDsPad, 1.0f);
    // the compressor: the groups completed in this chunk become compressed rows n0 .. n1 - 1 of this layer's caches
    if (L.kv_own) {
      const int rt = L.rt, n0 = pos / rt, n1 = (pos + T) / rt, nn = n1 - n0;
      if (n1 > L.cap) throw std::runtime_error("deepseek41: compressed cache full");
      if (nn > 0) {
        {
          cl_kernel k = kernel("ops", "", "ds_compress");
          int DD = D, r = rt, p0 = pos, a = n0;
          set_args(k, 0, ds_cv_, ds_cg_, L.pend, DD, r, p0, a, ds_lat_);
          size_t gws[2] = {(size_t)D, (size_t)nn};
          run("ds_compress", k, 2, gws, nullptr);
        }
        // normed latent (f32 into ds_lat2_, int8 for the index key's GEMM)
        {
          const int Mn = (nn + 15) / 16 * 16, saved = cur_T_;
          cur_T_ = nn;
          float eps = hp.eps;
          int r0 = 0, K = D, MM = Mn, NN = nn, wh = 1, a32 = 1;
          set_args(k_rmsnorm_, 0, ds_lat_, r0, L.comp_norm, K, MM, NN, eps, aq_, asc_, absum_, ds_lat2_, wh, a32);
          size_t gws[2] = {256, (size_t)Mn}, lws[2] = {256, 1};
          run("rmsnorm_quant", k_rmsnorm_, 2, gws, lws);
          cur_T_ = saved;
        }
        ds_matmul_rows("ds_ix_k", L.ix_k, nn, ds_ik_, d.ID);
        const int po = n0 * rt - pos + kDsPad;
        ds_norm_rope(ds_ik_, L.ic, d.ID, nn, L.ix_knorm, 1, 1, rt, po, 1.0f, n0, true);
        ds_norm_rope(ds_lat2_, L.cc, D, nn, nullptr, 1, 1, rt, po, 1.0f, n0, true);
      }
      if (rt > 1 && (pos + T) % rt) {  // the tokens of the group not complete yet
        cl_kernel k = kernel("ops", "", "ds_pend_store");
        int DD = D, r = rt, p0 = pos, b = n1;
        set_args(k, 0, ds_cv_, ds_cg_, L.pend, DD, r, p0, b);
        size_t gws[2] = {(size_t)D, (size_t)((pos + T) % rt)};
        run("ds_pend", k, 2, gws, nullptr);
      }
    }
    // the indexer: per query, the top-k of the visible compressed rows of its KV source
    if (L.ix_own) {
      const DsLayer& K = dsl_[L.kvs];
      const int rt = L.rt, nrows = (pos + T) / rt, ld = (nrows + 63) / 64 * 64;
      const bool is_cand = il == d.cand_src && d.cand_k > 0;
      const bool use_cand = !is_cand && d.cand_src >= 0 && il > d.cand_src && d.cand_k > 0 && L.kvs == dsl_[d.cand_src].kvs;
      ds_norm_rope(ds_iq_, ds_iq_, d.ID, T * d.IH, nullptr, ty, d.IH, 1, kDsPad, 1.0f);
      if (is_cand) fill(ds_cand_, ((size_t)T * ds_nbmax_ + 3) / 4 * 4);
      for (int q0 = 0; q0 < T; q0 += ds_sc_q_) {
        const int Tq = std::min(ds_sc_q_, T - q0);
        if (nrows > 0) {
          // (ds_ix_scores16: a lane per row, the key read once for all heads; ARC_DS_IX1=1: one work-item per score)
          static const bool ix1 = getenv("ARC_DS_IX1") && atoi(getenv("ARC_DS_IX1")) != 0;
          const bool ix16 = !ix1 && d.IH <= 32 && d.ID % 16 == 0;
          cl_kernel k = ix16 ? kernel("ops", ds_kv_defs() + " -DDS_IH=" + std::to_string(d.IH), "ds_ix_scores16")
                             : kernel("ops", ds_kv_defs(), "ds_ix_scores");
          int ih = d.IH, id = d.ID, nr = nrows, p0 = pos, r = rt, tq = q0, l2 = ld, uc = use_cand ? 1 : 0,
              nb = ds_nbmax_, bs = d.cand_bs;
          set_args(k, 0, ds_iq_, ds_iw_, K.ic, ih, id, nr, p0, r, tq, ds_sc_, l2, ds_cand_, uc, nb, bs);
          size_t gws[2] = {(size_t)ld, (size_t)Tq}, lws[2] = {64, 1};
          run(ix16 ? "ds_ix_scores16" : "ds_ix_scores", k, 2, gws, lws);
        }
        auto topk = [&](cl_mem sc, int ldsc, int kk, int bs_div, cl_mem sel, cl_mem nsel, int ldsel) {
          cl_kernel k = kernel("ops", "", "ds_topk");
          int l2 = ldsc, n = 0, k2 = kk, p0 = pos, r = rt, tq = q0, pq = 1, bd = bs_div, ls = ldsel;
          set_args(k, 0, sc, l2, n, k2, p0, r, tq, pq, bd, sel, nsel, ls);
          size_t gws[2] = {256, (size_t)Tq}, lws[2] = {256, 1};
          run("ds_topk", k, 2, gws, lws);
        };
        if (is_cand && nrows > 0) {
          {
            cl_kernel k = kernel("ops", "", "ds_block_max");
            int l2 = ld, p0 = pos, r = rt, tq = q0, bs = d.cand_bs, nb = ds_nbmax_;
            set_args(k, 0, ds_sc_, l2, p0, r, tq, bs, ds_bsc_, nb);
            size_t gws[2] = {(size_t)ds_nbmax_, (size_t)Tq};
            run("ds_block_max", k, 2, gws, nullptr);
          }
          topk(ds_bsc_, ds_nbmax_, d.cand_k, d.cand_bs, ds_bsel_, ds_nbsel_, d.cand_k);
          {
            cl_kernel k = kernel("ops", "", "ds_cand_map");
            int ls = d.cand_k, tq = q0, nb = ds_nbmax_;
            set_args(k, 0, ds_bsel_, ds_nbsel_, ls, tq, ds_cand_, nb);
            size_t gws[2] = {(size_t)d.cand_k, (size_t)Tq};
            run("ds_cand_map", k, 2, gws, nullptr);
          }
        }
        if (nrows > 0) topk(ds_sc_, ld, d.TOPK, 0, ds_sel_, ds_nsel_, d.TOPK);
        else fill(ds_nsel_, (size_t)T * 4);
      }
    }
    // attention over the window and the selected compressed rows
    // (1-8 rows: the keys of each (row, head) split over 8 sub-groups, combined after: ds_attn_split / ds_attn_comb;
    //  ARC_DS_ATTN1=1: ds_attn, one sub-group per (row, head), for every pass; prompt chunks: ds_attn_gh below)
    static const bool attn1 = getenv("ARC_DS_ATTN1") && atoi(getenv("ARC_DS_ATTN1")) != 0;
    static const bool attn_gh = !getenv("ARC_DS_ATTN_GH") || atoi(getenv("ARC_DS_ATTN_GH")) != 0;
    constexpr int kRows = 8, S = 8;  // (rows of the split path, splits per (row, head))
    // (heads per work-group: the 64 heads share one KV head, so a work-group of several heads of a row keeps its key
    //  reads on one Xe core; ARC_DS_ATTN_HPW=n, default 16; the same arithmetic per head at any n)
    static const int hpw0 = getenv("ARC_DS_ATTN_HPW") ? atoi(getenv("ARC_DS_ATTN_HPW")) : 16;
    int hpw = std::clamp(hpw0, 1, 32);  // (at most 512 work-items)
    while (H % hpw) --hpw;
    if (T <= kRows && !attn1) {
      if (!ds_apart_) ds_apart_ = buf((size_t)kRows * H * S * (D + 2) * 4);
      const DsLayer* K = L.rt ? &dsl_[L.kvs] : nullptr;
      {
        cl_kernel k = kernel("ops", ds_kv_defs(), "ds_attn_split");
        int W = d.W, us = L.rt ? 1 : 0, ls = std::max(1, d.TOPK), p0 = pos, HH = H, SS = S;
        set_args(k, 0, ds_q_, ds_kvh_, L.ring, W, K ? K->cc : L.ring, ds_sel_, ds_nsel_, us, ls, L.sinks, p0, HH, SS,
                 ds_apart_);
        size_t gws[3] = {(size_t)16 * H, (size_t)T, (size_t)S}, lws[3] = {(size_t)16 * hpw, 1, 1};
        run("ds_attn", k, 3, gws, lws);
      }
      {
        cl_kernel k = kernel("ops", ds_kv_defs(), "ds_attn_comb");
        int SS = S, R = hp.n_rot, po = kDsPad, HH = H, hpg = H / d.G, MM = Mp;
        set_args(k, 0, ds_apart_, SS, ds_rt_[ty], R, po, HH, hpg, MM, ds_att_);
        size_t gws[2] = {(size_t)16 * H, (size_t)T}, lws[2] = {16, 1};
        run("ds_attn", k, 2, gws, lws);
      }
    } else if (H % 16 == 0 && hp.n_rot % 4 == 0 && !attn1 && attn_gh) {
      // (prompt chunks: 16 heads a work-group, lane = key, ops_ds41.cl ds_attn_gh; ARC_DS_ATTN_GH=0: ds_attn)
      cl_kernel k = kernel("ops", ds_kv_defs() + " -cl-intel-256-GRF-per-thread", "ds_attn_gh");
      const DsLayer* K = L.rt ? &dsl_[L.kvs] : nullptr;
      int W = d.W, us = L.rt ? 1 : 0, ls = std::max(1, d.TOPK), R = hp.n_rot, po = kDsPad, p0 = pos, HH = H,
          hpg = H / d.G, MM = Mp;
      set_args(k, 0, ds_q_, ds_kvh_, L.ring, W, K ? K->cc : L.ring, ds_sel_, ds_nsel_, us, ls, L.sinks, ds_rt_[ty], R, po,
               p0, HH, hpg, MM, ds_att_);
      size_t gws[3] = {(size_t)16 * 8, (size_t)H / 16, (size_t)T}, lws[3] = {(size_t)16 * 8, 1, 1};
      run("ds_attn", k, 3, gws, lws);
    } else {
      cl_kernel k = kernel("ops", ds_kv_defs(), "ds_attn");
      const DsLayer* K = L.rt ? &dsl_[L.kvs] : nullptr;
      int W = d.W, us = L.rt ? 1 : 0, ls = std::max(1, d.TOPK), R = hp.n_rot, po = kDsPad, p0 = pos, HH = H,
          hpg = H / d.G, MM = Mp;
      set_args(k, 0, ds_q_, ds_kvh_, L.ring, W, K ? K->cc : L.ring, ds_sel_, ds_nsel_, us, ls, L.sinks, ds_rt_[ty], R, po,
               p0, HH, hpg, MM, ds_att_);
      size_t gws[2] = {(size_t)16 * H, (size_t)T}, lws[2] = {(size_t)16 * hpw, 1};
      run("ds_attn", k, 2, gws, lws);
    }
    {
      cl_kernel k = kernel("ops", ds_kv_defs(), "ds_ring_store");
      int W = d.W, DD = D, p0 = pos, TT = T;
      set_args(k, 0, ds_kvh_, L.ring, W, DD, p0, TT);
      size_t gws[2] = {(size_t)D, (size_t)std::min(T, d.W)};
      run("ds_ring_store", k, 2, gws, nullptr);
    }
    // grouped output LoRA, then the output projection
    {
      const int GK = H / d.G * D;
      for (int g = 0; g < d.G; ++g) {
        quant_rows(from(ds_att_, (size_t)g * Mp * GK * 4), GK, Mp, T, 1);
        matmul("ds_out_a", L.out_a[g], Mp, ds_oa_, d.G * d.OL, g * d.OL, 0, true);
      }
      quant_rows(ds_oa_, d.G * d.OL, Mp, T, 1);
      matmul("ds_out_b", L.out_b, Mp, ds_f_, E, 0, 0, true);
    }
    ds_expand(ds_f_, ds_mixA_, T);
    if (dbg_rows) dbg_row(T - 1);
    // FFN site: its mixes from the post-attention streams, its input collapsed with the attention site's pre
    ds_mix(L.hc_fn[1], L.hc_base[1], L.hc_scale[1], T, ds_mixF_);
    ds_collapse(ds_mixA_, 0, L0.post_norm, 0, T, Mp);
    layer_moe(L0, T, Mp);
    ds_expand(hco_, ds_mixF_, T);
    if (dbg_rows) dbg_row(T - 1);
  }

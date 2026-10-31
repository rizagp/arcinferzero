// Tensor parallel: one shard of a model split over GPUs (zero/mgpu/engine_tp.h runs the shards, one thread and GPU
// each). A part of class Engine (zero/engine.h), included inside the class body: not a header of its own. The
// multi-GPU code stays in zero/mgpu/ and its kernels in zero/kernels/tp.cl; the one-GPU code only calls the hooks
// below (T, tp_head, tp_out, gdn_ab, tp_conv_lws, tp_loaded), which do what it did before when the model is not split.
//
// The split (Qwen3.5 / 3.6 / 3.8 dense, qwen35): every shard holds the whole residual stream x_ and computes the
// norms itself; the heads and FFN columns are divided, so each GPU reads 1 / tp_size of the weights per token:
//  - DeltaNet: k-heads [k0, k1) and the v-heads that read them, h = k + r * nk (the kernels pair v-head h with k-head
//    h % nk), stored r-major so that within the shard v-head j still reads k-head j % (k1 - k0): the kernels are
//    unchanged. Rows of qkv / z / alpha / beta / conv1d / ssm_a / dt_bias, columns of ssm_out.
//  - attention: the kv-head groups [g0, g1) with their q-heads (rows of q / k / v, columns of attn_output) and their
//    part of the KV cache.
//  - FFN: columns [f0, f1) of gate / up, the same columns of down.
//  - output head: vocabulary rows [v0, v1); the logits are joined on the host.
// A count that does not divide by tp_size is split as evenly as its unit allows (k-heads in pairs, kv-heads one by
// one, FFN columns 256 at a time, vocabulary rows 16 at a time; the first GPUs take one unit more): the 27B over 3
// GPUs holds 6 / 6 / 4 k-heads, 2 / 1 / 1 kv-heads.
// The output projections (ssm_out, attn_output, ffn_down) write partial sums that tp_out adds over the shards
// (TpComm::reduce) before they reach x_. MTP, DFlash2, slots and the NPU are off in a split model.

 public:
  TpComm* tp_ = nullptr;  // (set by TpEngine before the first forward)
  int n_logits() const { return lm_head_.N; }  // this engine's logits per row (its vocabulary rows when split)
  const std::vector<float>& logits_host() const { return logits_host_; }

 private:
  struct TpGlobal { int nk = 0, nv = 0, nh = 0, nkv = 0, ff = 0, vocab = 0; };
  TpGlobal tpg_;
  using TpRange = std::pair<int64_t, int64_t>;
  TpRange tpk_, tpkv_, tpf_, tpv_;  // this shard's k-heads, kv-heads, FFN columns, vocabulary rows: [begin, end)
  // rank r's part of `total` in whole units (the first total / unit % N ranks one unit more)
  static TpRange tp_range(int64_t total, int64_t unit, int N, int r) {
    const int64_t units = total / unit, base = units / N, extra = units % N;
    const int64_t b = r * base + std::min<int64_t>(r, extra), e = b + base + (r < extra ? 1 : 0);
    return {b * unit, e * unit};
  }
  std::deque<std::vector<uint8_t>> tp_keep_;               // (the cut tensors' bytes, freed after the load: tp_loaded)
  std::unordered_map<std::string, GgufTensor> tp_cut_;     // (name -> this shard's part)
  using TpRuns = std::vector<std::pair<int64_t, int64_t>>;  // [begin, end) ranges, in order
  cl_kernel k_tp_add_ = nullptr, k_tp_pack_ = nullptr;
  cl_mem tp_ab_ = nullptr;  // (a split model's padded alpha / beta rows, see gdn_ab)
  cl_mem tp_host_ = nullptr;  // (the shared host memory as this GPU's kernel argument)
  cl_mem tp_part_ = nullptr;                                // (the partial output of a projection, [max_chunk][E])
  bool tp_on() const { return opt.tp_size > 1; }

  // This shard's dimensions in hp (the global ones in tpg_); everything sized from hp is then this shard's.
  void tp_setup() {
    const int N = opt.tp_size, r = opt.tp_rank;
    if (hp.arch != "qwen35" || hp.moe || hp.hc || hp.gemma || hp.qwen3)
      throw std::runtime_error("multi-GPU (--gpus) runs dense Qwen3.5 / 3.6 / 3.8 models only for now (this is " + hp.arch + ")");
    if (r < 0 || r >= N) throw std::runtime_error("tensor parallel: bad rank");
    tpg_ = {hp.ssm_nk, hp.ssm_nv, hp.n_head, hp.n_head_kv, hp.n_ff, hp.n_vocab};
    auto need = [&](bool ok, const char* what) {
      if (!ok)
        throw std::runtime_error("this model cannot be split over " + std::to_string(N) + " GPUs (" + what +
                                 "); try " + (N > 2 ? "2" : "1"));
    };
    // (k-heads in pairs: a pair's 2 x 128 v-values are one 256-value block of ssm_out's columns)
    need(hp.ssm_nk > 0 && hp.ssm_nk % 2 == 0 && hp.ssm_nk / 2 >= N && hp.ssm_nv % hp.ssm_nk == 0, "DeltaNet heads");
    need(hp.n_head_kv >= N && hp.n_head % hp.n_head_kv == 0 && (hp.n_head / hp.n_head_kv * hp.head_dim) % 256 == 0, "attention heads");
    need(hp.n_ff % 256 == 0 && hp.n_ff / 256 >= N, "FFN width");
    need(hp.n_vocab % 16 == 0, "vocabulary");
    tpk_ = tp_range(hp.ssm_nk, 2, N, r);
    tpkv_ = tp_range(hp.n_head_kv, 1, N, r);
    tpf_ = tp_range(hp.n_ff, 256, N, r);
    tpv_ = tp_range(hp.n_vocab, 16, N, r);
    const int hpg = hp.n_head / hp.n_head_kv;
    hp.ssm_nk = (int)(tpk_.second - tpk_.first);
    hp.ssm_nv = hp.ssm_nk * (tpg_.nv / tpg_.nk);
    hp.ssm_d_inner = hp.ssm_nv * hp.ssm_d_state;
    hp.n_head_kv = (int)(tpkv_.second - tpkv_.first);
    hp.n_head = hp.n_head_kv * hpg;
    hp.n_ff = (int)(tpf_.second - tpf_.first);
    opt.mtp = false;
    opt.dflash.clear();
    opt.npu = 0;
    opt.slots = 1;
    fprintf(stderr, "arcinferzero: GPU %d of %d: %d of %d DeltaNet k-heads, %d of %d kv-heads, %d of %d FFN columns, %lld of %d vocabulary rows\n",
            r, N, hp.ssm_nk, tpg_.nk, hp.n_head_kv, tpg_.nkv, hp.n_ff, tpg_.ff, (long long)(tpv_.second - tpv_.first), tpg_.vocab);
  }

  // rows `rows` (empty: all) and columns `cols` (empty: all; whole blocks of the type) of a 1-D or 2-D tensor
  const GgufTensor& tp_slice(const GgufTensor& t, const TpRuns& rows, const TpRuns& cols) {
    const int64_t K = (int64_t)t.dims[0], N = t.dims.size() > 1 ? (int64_t)t.dims[1] : 1;
    int blk = 1, bytes = 4;
    ggml_block_info(t.type, blk, bytes);
    const size_t rb = (size_t)(K / blk) * bytes;
    const TpRuns R = rows.empty() ? TpRuns{{0, N}} : rows, Cc = cols.empty() ? TpRuns{{0, K}} : cols;
    int64_t n2 = 0, k2 = 0;
    for (auto [a, b] : R) {
      if (a < 0 || b > N || a >= b) throw std::runtime_error("tensor parallel: bad rows of " + t.name);
      n2 += b - a;
    }
    for (auto [a, b] : Cc) {
      if (a < 0 || b > K || a >= b || a % blk || b % blk) throw std::runtime_error("tensor parallel: bad columns of " + t.name);
      k2 += b - a;
    }
    tp_keep_.emplace_back((size_t)n2 * (size_t)(k2 / blk) * bytes);
    uint8_t* o = tp_keep_.back().data();
    for (auto [a, b] : R)
      for (int64_t n = a; n < b; ++n)
        for (auto [c0, c1] : Cc) {
          const size_t len = (size_t)((c1 - c0) / blk) * bytes;
          memcpy(o, t.data + (size_t)n * rb + (size_t)(c0 / blk) * bytes, len);
          o += len;
        }
    GgufTensor c = t;
    c.dims[0] = (uint64_t)k2;
    if (c.dims.size() > 1) c.dims[1] = (uint64_t)n2;
    c.data = tp_keep_.back().data();
    c.nbytes = tp_keep_.back().size();
    return tp_cut_[t.name] = c;
  }
  // ranges of the v-heads this shard keeps (unit = values per head), r-major: see the top of this file
  TpRuns tp_vheads(int64_t unit) const {
    const int64_t nk = tpg_.nk, k0 = tpk_.first, k1 = tpk_.second;
    TpRuns v;
    for (int64_t r = 0; r < tpg_.nv / tpg_.nk; ++r) v.push_back({(k0 + r * nk) * unit, (k1 + r * nk) * unit});
    return v;
  }
  // The tensor `name` as this shard holds it (the whole one when the model is not split or the tensor is not).
  const GgufTensor& T(const std::string& name) {
    const GgufTensor& t = f_.get(name);
    if (!tp_on() || name.rfind("blk.", 0) != 0) return t;
    if (auto it = tp_cut_.find(name); it != tp_cut_.end()) return it->second;
    const size_t dot = name.find('.', 4);
    const int il = atoi(name.c_str() + 4);
    const std::string s = name.substr(dot + 1);
    const int64_t D = hp.ssm_d_state, hd = hp.head_dim;
    if (il < hp.n_layer && hp.recurrent(il)) {
      const int64_t nk = tpg_.nk, k0 = tpk_.first, k1 = tpk_.second;
      if (s == "attn_qkv.weight" || s == "ssm_conv1d.weight") {  // channels [q | k | v]
        TpRuns rows{{k0 * D, k1 * D}, {(nk + k0) * D, (nk + k1) * D}};
        for (auto [a, b] : tp_vheads(D)) rows.push_back({2 * nk * D + a, 2 * nk * D + b});
        return tp_slice(t, rows, {});
      }
      if (s == "attn_gate.weight") return tp_slice(t, tp_vheads(D), {});
      if (s == "ssm_alpha.weight" || s == "ssm_beta.weight") return tp_slice(t, tp_vheads(1), {});
      if (s == "ssm_a" || s == "ssm_dt.bias") return tp_slice(t, {}, tp_vheads(1));
      if (s == "ssm_out.weight") return tp_slice(t, {}, tp_vheads(D));
    } else if (il < hp.n_layer) {
      const int64_t hpg = tpg_.nh / tpg_.nkv, g0 = tpkv_.first, g1 = tpkv_.second, q0 = g0 * hpg, q1 = g1 * hpg;
      if (s == "attn_q.weight") {  // (each head's rows: query then gate, 2 * head_dim)
        const int64_t per = (int64_t)t.dims[1] / tpg_.nh;
        return tp_slice(t, {{q0 * per, q1 * per}}, {});
      }
      if (s == "attn_k.weight" || s == "attn_v.weight") return tp_slice(t, {{g0 * hd, g1 * hd}}, {});
      if (s == "attn_output.weight") return tp_slice(t, {}, {{q0 * hd, q1 * hd}});
    }
    if (il < hp.n_layer) {
      const int64_t f0 = tpf_.first, f1 = tpf_.second;
      if (s == "ffn_gate.weight" || s == "ffn_up.weight") return tp_slice(t, {{f0, f1}}, {});
      if (s == "ffn_down.weight") return tp_slice(t, {}, {{f0, f1}});
    }
    return t;
  }
  // the output head's vocabulary rows of this shard
  const GgufTensor& tp_head(const GgufTensor& t) {
    if (!tp_on()) return t;
    return tp_slice(t, {tpv_}, {});
  }
  // An output projection into the residual stream: x_ += act * W^T, the shards' partial sums added (tp_reduce in
  // zero/engine_tp.h) when the model is split.
  void tp_out(const char* name, const GpuWeight& w, int Mp, int act32) {
    if (!tp_on()) return matmul(name, w, Mp, x_, hp.n_embd, 0, 1, act32);
    if (!tp_part_) tp_part_ = buf((size_t)max_chunk * hp.n_embd * 4);
    matmul(name, w, Mp, tp_part_, hp.n_embd, 0, 0, act32);
    tp_reduce(cur_T_);
  }
  void tp_reduce(int rows);  // (zero/engine_tp.h)
  // rows of a matrix the GEMM must see in whole tiles of 16: a split model's DeltaNet alpha / beta (2 x its v-heads:
  // 24 for the 27B's 12 over 4 GPUs) padded with rows of zeros; unchanged in one piece
  int tp_rows(int n) const { return tp_on() ? (n + 15) / 16 * 16 : n; }
  // DeltaNet's alpha / beta projection into gb_ ([rows][2 nv], as gdn_gates / gdn_recur read it). Padded rows go
  // to a scratch buffer and are packed back, the kernels stay as they are.
  void gdn_ab(const Layer& L, int Mp, int act32) {
    const int nab = 2 * hp.ssm_nv;
    if (L.ab.N == nab) return matmul("gemm_ab", L.ab, Mp, gb_, nab, 0, 0, act32);
    if (!tp_ab_) tp_ab_ = buf((size_t)max_chunk * L.ab.N * 4);
    matmul("gemm_ab", L.ab, Mp, tp_ab_, L.ab.N, 0, 0, act32);
    tp_build();
    int ld = L.ab.N, n = nab, rows = cur_T_;
    set_args(k_tp_pack_, 0, tp_ab_, ld, gb_, n, rows);
    size_t g2[2] = {(size_t)(n + 15) / 16 * 16, (size_t)rows}, l2[2] = {16, 1};
    run("tp_pack", k_tp_pack_, 2, g2, l2);
  }
  // gdn_conv's work-group: 4 sub-groups of 16 lanes, each 128 channels; a shard can hold a number of channel groups
  // that 4 does not divide (the 27B over 3 GPUs: 30): one sub-group per work-group then (the kernel needs no more)
  void tp_conv_lws(int C, size_t& lws0) const {
    if (tp_on() && (C / 128) % 4) lws0 = 16;
  }
  // the split's kernels, zero/kernels/tp.cl (built on this shard's GPU)
  void tp_build() {
    if (k_tp_add_) return;
    const std::string src = read_file(opt.kernel_dir + "/tp.cl");
    cl_program p = cl_.build(src, "-cl-std=CL3.0");
    progs_.push_back(p);
    k_tp_add_ = cl_.kernel(p, "tp_add");
    k_tp_pack_ = cl_.kernel(p, "tp_pack");
  }
  // after load_weights: the cut tensors are on the GPU; their host copies go (27B over 2 GPUs: ~7.5 GB of RAM each)
  void tp_loaded() {
    tp_cut_.clear();
    tp_keep_.clear();
    tp_keep_.shrink_to_fit();
  }

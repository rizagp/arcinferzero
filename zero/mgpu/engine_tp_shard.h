// Tensor parallel: one shard of a model split over GPUs (zero/engine_tp.h runs the shards, one thread and GPU each).
// A part of class Engine (zero/engine.h), included inside the class body: not a header of its own.
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
// The output projections (ssm_out, attn_output, ffn_down) write partial sums that tp_out adds over the shards
// (TpComm::reduce) before they reach x_. MTP, DFlash2, slots and the NPU are off in a split model.

 public:
  TpComm* tp_ = nullptr;  // (set by TpEngine before the first forward)
  int n_logits() const { return lm_head_.N; }  // this engine's logits per row (its vocabulary rows when split)
  const std::vector<float>& logits_host() const { return logits_host_; }

 private:
  struct TpGlobal { int nk = 0, nv = 0, nh = 0, nkv = 0, ff = 0, vocab = 0; };
  TpGlobal tpg_;
  std::deque<std::vector<uint8_t>> tp_keep_;               // (the cut tensors' bytes, alive while loading)
  std::unordered_map<std::string, GgufTensor> tp_cut_;     // (name -> this shard's part)
  using TpRuns = std::vector<std::pair<int64_t, int64_t>>;  // [begin, end) ranges, in order
  cl_kernel k_tp_add_ = nullptr;
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
    need(hp.ssm_nk % N == 0 && (hp.ssm_nk / N) % 2 == 0 && hp.ssm_nv % hp.ssm_nk == 0, "DeltaNet heads");
    need(hp.n_head_kv % N == 0 && (hp.n_head / N * hp.head_dim) % 256 == 0, "attention heads");
    need(hp.n_ff % (256 * N) == 0, "FFN width");
    need(hp.n_vocab % (16 * N) == 0, "vocabulary");
    hp.ssm_nk /= N;
    hp.ssm_nv /= N;
    hp.ssm_d_inner = hp.ssm_nv * hp.ssm_d_state;
    hp.n_head /= N;
    hp.n_head_kv /= N;
    hp.n_ff /= N;
    opt.mtp = false;
    opt.dflash.clear();
    opt.npu = 0;
    opt.slots = 1;
    if (r == 0)
      fprintf(stderr, "arcinferzero: split over %d GPUs: each %d of %d DeltaNet k-heads, %d of %d kv-heads, %d of %d FFN columns, %d of %d vocabulary rows\n",
              N, hp.ssm_nk, tpg_.nk, hp.n_head_kv, tpg_.nkv, hp.n_ff, tpg_.ff, tpg_.vocab / N, tpg_.vocab);
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
    const int64_t nk = tpg_.nk, k0 = (int64_t)opt.tp_rank * hp.ssm_nk, k1 = k0 + hp.ssm_nk;
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
    const int64_t D = hp.ssm_d_state, hd = hp.head_dim, r = opt.tp_rank;
    if (il < hp.n_layer && hp.recurrent(il)) {
      const int64_t nk = tpg_.nk, k0 = r * hp.ssm_nk, k1 = k0 + hp.ssm_nk;
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
      const int64_t q0 = r * hp.n_head, q1 = q0 + hp.n_head, g0 = r * hp.n_head_kv, g1 = g0 + hp.n_head_kv;
      if (s == "attn_q.weight") {  // (each head's rows: query then gate, 2 * head_dim)
        const int64_t per = (int64_t)t.dims[1] / tpg_.nh;
        return tp_slice(t, {{q0 * per, q1 * per}}, {});
      }
      if (s == "attn_k.weight" || s == "attn_v.weight") return tp_slice(t, {{g0 * hd, g1 * hd}}, {});
      if (s == "attn_output.weight") return tp_slice(t, {}, {{q0 * hd, q1 * hd}});
    }
    if (il < hp.n_layer) {
      const int64_t f0 = r * hp.n_ff, f1 = f0 + hp.n_ff;
      if (s == "ffn_gate.weight" || s == "ffn_up.weight") return tp_slice(t, {{f0, f1}}, {});
      if (s == "ffn_down.weight") return tp_slice(t, {}, {{f0, f1}});
    }
    return t;
  }
  // the output head's vocabulary rows of this shard
  const GgufTensor& tp_head(const GgufTensor& t) {
    if (!tp_on()) return t;
    const int64_t v = tpg_.vocab / opt.tp_size;
    return tp_slice(t, {{opt.tp_rank * v, (opt.tp_rank + 1) * v}}, {});
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

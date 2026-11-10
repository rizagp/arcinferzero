// A part of class Engine (zero/engine.h), included inside the class body at the place it was cut from (the
// class is the same text), not a header of its own: saved conversations (the server's session cache: snapshots in memory and on disk).

  // ---------------------------------------------------------------- saved conversations (the server's session cache)
  // Everything a conversation leaves in the engine after `pos` positions: the K/V blocks and scales of every layer
  // that owns a KV cache (full-attention layers: positions [0, pos); sliding-window layers only from n_swa positions
  // before `keep_from`, the earliest position the conversation should be resumable at), the MTP block's cache and
  // hidden state, the DeltaNet states and all snapshots. Device data is copied in batches of up to 256 MB.
  // load_state() returns false (the engine left reset) for a stream from another model or an incompatible layout.
  // swa_rings: the first position a conversation saved with keep_from can be resumed at (see rewind()).
  int saved_floor(int keep_from) const {
    return hp.swa_rings ? std::max(rewind_floor(), swa_save_lo(keep_from) + hp.n_swa) : 0;
  }
  // save_state() writes through `out` (a file, or memory for a write in the background); saved_size() is its length.
  using Sink = std::function<bool(const void*, size_t)>;
  // host_snapshot = false: without the host snapshot (a state saved exactly where it was taken holds it already)
  bool save_state(FILE* f, int keep_from, bool host_snapshot = true) {
    return save_state([f](const void* p, size_t n) { return fwrite(p, 1, n, f) == n; }, keep_from, host_snapshot);
  }
  size_t saved_size(int keep_from) {
    if (!opt.attn_int8) return 0;
    std::vector<int32_t> hdr;
    std::vector<Seg> segs;
    save_layout(keep_from, hdr, segs);
    size_t n = hdr.size() * 4;
    for (const Seg& sg : segs) n += sg.n;
    if (hsnap_.pos >= 0) {
      for (const auto& v : hsnap_.S) n += v.size();
      if (has_mtp) n += hsnap_.h.size();
    }
    return n;
  }
  bool save_state(const Sink& out, int keep_from, bool host_snapshot = true) {
    if (!opt.attn_int8) return false;
    flush_mtp_pending();
    std::vector<int32_t> hdr;
    std::vector<Seg> segs;
    save_layout(keep_from, hdr, segs, host_snapshot);
    bool ok = out(hdr.data(), hdr.size() * 4) && segs_out(segs, out);
    if (ok && host_snapshot && hsnap_.pos >= 0) {
      for (const auto& v : hsnap_.S) ok = ok && out(v.data(), v.size());
      if (has_mtp) ok = ok && out(hsnap_.h.data(), hsnap_.h.size());
    }
    return ok;
  }
  bool load_state(FILE* f) {
    reset();
    for (int i = 0; i < (int)snaps_.size(); ++i) drop_snapshot(i);
    drop_host_snapshot();
    if (!opt.attn_int8) return false;
    const auto kl = kv_layers();
    std::vector<int32_t> hdr(11);
    if (!get(f, hdr.data(), hdr.size() * 4)) return false;
    const int spos = hdr[7], swh = hdr[8], slo = hdr[9];
    if (hdr[0] != kStateMagic || hdr[1] != 2 || hdr[2] != hp.n_layer || hdr[3] != hp.n_embd || hdr[4] != hp.n_vocab ||
        hdr[5] != state_flags() || hdr[6] != (has_mtp ? 1 : 0) || hdr[10] != (int)kl.size() || spos < 0 ||
        spos > ctx || swh > ctx || slo < 0)
      return false;
    std::vector<int32_t> geo(kl.size() * 5);
    if (!get(f, geo.data(), geo.size() * 4)) return false;
    std::vector<std::pair<int, int>> br;
    for (size_t i = 0; i < kl.size(); ++i) {
      const int* q = &geo[i * 5];
      // same heads and scale layout; the blocks must fit this cache (a sliding-window ring: its capacity)
      if (q[0] != kl[i].nkv || q[1] != kl[i].hd || q[2] != kl[i].sb || q[3] < 0 || q[4] < q[3] ||
          q[4] - q[3] > kl[i].cap / 16 || (!kl[i].L->swa && (q[3] != 0 || q[4] > kl[i].cap / 16)))
        return false;
      br.push_back({q[3], q[4]});
    }
    int32_t ns = 0, hpos = -1;
    if (!get(f, &ns, 4) || ns < 0 || ns > 8) return false;
    std::vector<int> sp(ns);
    if (!get(f, sp.data(), (size_t)ns * 4) || !get(f, &hpos, 4) || hpos < -1 || hpos > ctx) return false;
    for (int i = 0; i < ns; ++i)  // (snapshot positions: -1 (none) or within the context, else a damaged file)
      if (sp[i] < -1 || sp[i] > ctx) return false;
    for (int i = 0; i < ns; ++i)
      if (sp[i] >= 0) snap_slot(i);
    bool ok = segs_in(state_segs(kl, br, sp), f);
    if (ok && hpos >= 0) {
      for (auto& L : layers_)
        if (hp.recurrent(L.il)) {
          hsnap_.S.emplace_back(state_bytes());
          hsnap_.S.emplace_back(conv_bytes());
        }
      for (auto& v : hsnap_.S) ok = ok && get(f, v.data(), v.size());
      if (has_mtp) {
        hsnap_.h.resize((size_t)hp.n_embd * 4);
        ok = ok && get(f, hsnap_.h.data(), hsnap_.h.size());
      }
    }
    if (!ok) {
      reset();
      for (int i = 0; i < (int)snaps_.size(); ++i) drop_snapshot(i);
      drop_host_snapshot();
      return false;
    }
    for (int i = 0; i < ns; ++i)
      if (sp[i] >= 0) snaps_[i].pos = sp[i];
    for (Snap& s : snaps_) s.ring_ok = s.ring_want = false;  // (Gemma: a saved conversation holds no ring copies)
    hsnap_.pos = hpos;
    pos = spos;
    written_hi_ = swh;
    swa_lo_ = slo;
    pend_toks_.clear();
    last_drafts_.clear();
    attn_in_ready_ = ffn_in_ready_ = false;
    rt_pos_[0] = rt_pos_[1] = -1;
    return true;
  }

 private:
  struct Layer {
    int il = 0;
    cl_mem attn_norm = nullptr, post_norm = nullptr;
    // Gated DeltaNet
    GpuWeight qkv, z, ssm_out, ab;
    // single tokens: qkv + z (+ ab when they are Q8_0) rows stacked, one GEMV (qkv / z / ab are views of it)
    GpuWeight qz;
    bool qz_ab = false;
    cl_mem dt_bias = nullptr, ssm_a = nullptr, conv_w = nullptr, ssm_norm = nullptr;
    cl_mem conv_state = nullptr, S = nullptr;
    // speculative verification (MTP): the pass writes its states to S_next and keeps S, its inputs and the conv history
    // before it, so verify_done() takes S_next (every draft accepted) or replays the accepted tokens from S
    cl_mem S_next = nullptr, conv_pre = nullptr, xin = nullptr, abin = nullptr, qkvin = nullptr;
    // attention
    GpuWeight wq, wk, wv, wo;
    cl_mem q_norm = nullptr, k_norm = nullptr, kc = nullptr, vc = nullptr, kq = nullptr, ksc = nullptr,
           vq = nullptr, vsc = nullptr;
    // FFN
    GpuWeight gate, up, down;
    // MoE FFN: router (+ shared-expert gate row), shared expert, routed experts (gate/up/down codes and scales)
    GpuWeight router, gate_sh, up_sh, down_sh;
    // single tokens: router rows (padded to rgu_rows()) + shared-expert gate + up rows, one Q8_0 GEMV (they read
    // the same input; three small GEMVs each paid a launch's fixed cost)
    GpuWeight rgu;
    cl_mem ex_q[3] = {}, ex_s[3] = {};
    uint32_t ex_type[3] = {};  // (standard expert tensors: their GGUF types; ex_q = the GPU raw form)
    std::vector<cl_mem> ex_sub[3];  // views of ex_q per group of moe_group_ experts (prompt GEMM path, made on use)
    // Gemma 4
    bool swa = false, has_v = true;
    int nkv = 0, hd = 0, cap = 0;  // kv heads, head size, KV-cache capacity (positions; ring for sliding window)
    float out_scale = 1.0f;
    cl_mem post_attn = nullptr, ffn_norm = nullptr, post_ffw = nullptr, rope_ff = nullptr;
    int kv_src = -1;            // shared KV: the layer whose cache this layer reads (no K/V of its own)
    bool i8 = true;             // Gemma 4: keeps the int8 K / V copy (only the layers the drafter reads)
    cl_mem router_bf16 = nullptr;  // Flash Next precise: the router rows in BF16 (read by gemv_bf16 with f32 inputs)
    // Gemma 4 with experts: the dense FFN's output norm, the experts' input norm and output norm, the router's input
    // norm weights (its scale / sqrt(E)), the scale of each expert
    cl_mem post_ffw1 = nullptr, ffn_norm2 = nullptr, post_ffw2 = nullptr, rnorm = nullptr, escale = nullptr,
           router_f32 = nullptr;
    GpuWeight ple_gate, ple_proj;
    cl_mem ple_post = nullptr;  // per-layer-embedding block
    // Flash Next hyper-connection mixers before the attention / DeltaNet block (0) and the FFN (1): stream norm,
    // low-rank down projection stacked with the inject rows (hc_lr + hc_n, padded), up projection (K padded to 512)
    cl_mem hc_norm[2] = {};
    GpuWeight hc_down[2], hc_up[2];
    // Flash Next PLE block (its layer only): key / value projections, norms, depthwise conv weights [HCE][taps]
    GpuWeight ple_key, ple_value;
    cl_mem ple_nk = nullptr, ple_nq = nullptr, ple_nc = nullptr, ple_cw = nullptr;
    // Flash Next QSA (attention layers): indexer key + query rows stacked, their norms, the pooled block keys
    // ([ctx / qr][dim], by block: position-indexed like the KV cache), the raw keys of the tail block (state)
    int qr = 0, qsa_i = -1;
    GpuWeight ix;
    cl_mem ix_kn = nullptr, ix_qn = nullptr, pk = nullptr, qtail = nullptr;
    cl_mem ixsv = nullptr;  // (a verification pass: its indexer rows, for verify_done)
  };

  struct Snap {
    int pos = -1;
    std::vector<cl_mem> S, conv;
    cl_mem h = nullptr;  // MTP: main hidden state of the last position
    std::vector<std::vector<uint8_t>> ring;  // Gemma: the sliding-window rings' n_swa positions before pos (ring_segs)
    bool ring_ok = false;    // (that copy taken)
    bool ring_want = false;  // (to be taken before the rings overwrite it: ring_guard)
  };
  std::vector<Snap> snaps_;
  static constexpr int32_t kStateMagic = 0x53435241;  // "ARCS"
  struct HostSnap {
    int pos = -1;
    std::vector<std::vector<uint8_t>> S;  // per recurrent layer: state, conv state
    std::vector<uint8_t> h;
  };
  HostSnap hsnap_;
  // ---- conversation slots (see bind): the state buffers' fields, each slot's device addresses for them, the views
  // inside them, the addresses the buffer objects were created with, the host state of the slots not bound
  struct SlotHost {
    int pos = 0, rope_shift = 0, written_hi = 0, swa_lo = 0, ple_prev[2] = {-1, -1};
    std::vector<int> pend;
    std::vector<Snap> snaps;
    HostSnap hsnap;
    double dfl_ema = 4.0;
    int dfl_wait = 0, dfl_gap = 8;
    bool dfl_probe = false, dfl_used = false, vrows = false;
    std::vector<int> last_drafts;
  };
  struct SlotView { ZMem* m; int role; size_t off; };
  std::vector<cl_mem*> roles_;
  std::vector<std::vector<char*>> slot_ptrs_;
  std::vector<SlotView> subs_;
  std::vector<std::pair<ZMem*, char*>> orig_ptrs_;
  std::vector<SlotHost> slot_host_;
  int slot_ = 0;
  // ---- cancellation (see cancel)
  std::vector<std::vector<uint8_t>> blk_;  // DeltaNet states at the start of the current block
  bool cancelled() {
    if (tp_ && tp_served()) return tp_cancelled();  // (serverz --gpus: shard 0 decides for every shard)
    if (!cancel) return false;
    clFinish(cl_.queue);  // the check follows the work done, not just enqueued
    return cancel();
  }
  void keep_block() {
    size_t k = 0;
    for (auto& L : layers_)
      if (hp.recurrent(L.il))
        for (auto [b, n] : {std::pair{L.S, state_bytes()}, std::pair{L.conv_state, conv_bytes()}}) {
          if (k == blk_.size()) blk_.emplace_back(n);
          CL_CHECK(clEnqueueReadBuffer(cl_.queue, b, CL_FALSE, 0, n, blk_[k++].data(), 0, nullptr, nullptr));
        }
    clFinish(cl_.queue);
  }
  // back to p0, the start of the block being processed, then throw Cancelled
  [[noreturn]] void cancel_block(int p0) {
    clFinish(cl_.queue);
    size_t k = 0;
    for (auto& L : layers_)
      if (hp.recurrent(L.il)) {
        write_buf(L.S, blk_[k++]);
        write_buf(L.conv_state, blk_[k++]);
      }
    prog_base_ = prog_total_ = 0;
    if (hp.swa_rings && !rewind(p0)) reset();  // the rings no longer hold the n_swa positions before p0
    else pos = p0;
    throw Cancelled();
  }
  int swa_lo_ = 0;  // sliding-window layers hold no K/V below this position (a loaded conversation, trimmed)
  size_t state_bytes() const { return (size_t)hp.ssm_nv * hp.ssm_d_state * hp.ssm_d_state * 4; }
  // (DeltaNet conv history of 3 rows; Flash Next: every recurrent layer's state also has room for the PLE block's,
  // used at its layer: ple_hist() rows of the conv input and the last two token ids, so snapshots, saved
  // conversations and cancellation carry it like the other recurrent state)
  int ple_hist() const { return hp.ple_layer >= 0 ? (hp.ple_taps - 1) * hp.ple_ng : 0; }
  size_t ple_hist_off() const { return (size_t)3 * hp.conv_dim() * 4; }
  size_t ple_tok_off() const { return ple_hist_off() + (size_t)ple_hist() * hp.hce() * 4; }
  // (QSA: the raw keys of each QSA layer's tail block, 4 KB apart)
  size_t qsa_tail_off(int i) const { return (ple_tok_off() + 16 + 4095) / 4096 * 4096 + (size_t)i * 4096; }
  size_t conv_bytes() const {
    return hp.n_qsa ? qsa_tail_off(hp.n_qsa) : ple_tok_off() + (hp.ple_layer >= 0 ? 16 : 0);
  }
  int qsa_blocks(int c) const { return hp.n_qsa ? (c + 15) / 16 * 16 / hp.qsa_ratio[hp.full_attn_interval - 1] : 0; }
  int qsa_mask_words(int c) const { return (qsa_blocks(c) + 31) / 32; }
  std::vector<uint8_t> read_buf(cl_mem b, size_t n, size_t off = 0) {
    std::vector<uint8_t> v(n);
    if (n) CL_CHECK(clEnqueueReadBuffer(cl_.queue, b, CL_TRUE, off, n, v.data(), 0, nullptr, nullptr));
    return v;
  }
  void write_buf(cl_mem b, const std::vector<uint8_t>& v, size_t off = 0) {
    if (!v.empty()) CL_CHECK(clEnqueueWriteBuffer(cl_.queue, b, CL_TRUE, off, v.size(), v.data(), 0, nullptr, nullptr));
  }
  static bool put(FILE* f, const void* p, size_t n) { return fwrite(p, 1, n, f) == n; }
  static bool put(FILE* f, const std::vector<uint8_t>& v) { return put(f, v.data(), v.size()); }
  static bool get(FILE* f, void* p, size_t n) { return fread(p, 1, n, f) == n; }
  Snap& snap_slot(int slot) {  // GPU snapshot slot with its buffers allocated (as snapshot() does)
    if (slot >= (int)snaps_.size()) snaps_.resize(slot + 1);
    Snap& s = snaps_[slot];
    if (s.S.empty())
      for (auto& L : layers_)
        if (hp.recurrent(L.il)) { s.S.push_back(buf(state_bytes())); s.conv.push_back(buf(conv_bytes())); }
    if (has_mtp && !s.h) s.h = buf(mtp_hw() * 4);
    return s;
  }
  // layers with a KV cache of their own and its geometry (int8 cache: K/V blocks of 16 keys x hd bytes per kv head,
  // sb scales per key and block)
  struct KvGeom {
    Layer* L;
    int nkv, hd, cap, sb;
  };
  std::vector<KvGeom> kv_layers() {
    std::vector<KvGeom> v;
    for (auto& L : layers_) {
      if (hp.swa_rings) {
        if (L.kv_src < 0) v.push_back({&L, L.nkv, L.hd, L.cap, L.hd / 64});
      } else if (!hp.recurrent(L.il)) {
        v.push_back({&L, hp.n_head_kv, hp.head_dim, ctx, 1});
      }
    }
    if (mtp_block_) v.push_back({&mtp_, hp.n_head_kv, hp.head_dim, ctx, 1});
    return v;
  }
  int kv_hi() const { return hp.swa_rings ? std::max(pos, written_hi_) : pos; }  // positions the caches hold
  // first position the sliding-window layers keep when saving: n_swa before keep_from (block-aligned), no older than
  // what the ring still holds
  int swa_save_lo(int keep_from) const {
    const int need = std::max(0, std::min(keep_from, pos) - hp.n_swa) / 16 * 16;
    const int ring = (std::max(0, kv_hi() - swa_cap_) + 15) / 16 * 16;
    return std::max({need, ring, (swa_lo_ + 15) / 16 * 16});
  }
  struct Seg {
    cl_mem b;
    size_t off, n;
  };
  // device ranges of a saved state, in stream order: per KV layer and kv head K, V, K scales, V scales (blocks
  // [b0, b1) at their ring slots), the DeltaNet states, the GPU snapshots that hold a position, the MTP hidden state
  std::vector<Seg> state_segs(const std::vector<KvGeom>& kl, const std::vector<std::pair<int, int>>& br,
                              const std::vector<int>& sp) {
    std::vector<Seg> v;
    for (size_t i = 0; i < kl.size(); ++i) {
      const KvGeom& k = kl[i];
      const int nbc = k.cap / 16, b0 = br[i].first, n = br[i].second - br[i].first, s0 = b0 % nbc;
      const int first = std::min(n, nbc - s0);
      for (int g = 0; g < k.nkv; ++g)
        for (int t = 0; t < (k.L->i8 ? 4 : 0); ++t) {  // (Gemma layers without the int8 copy: the float rows only)
          cl_mem b = t == 0 ? k.L->kq : t == 1 ? k.L->vq : t == 2 ? k.L->ksc : k.L->vsc;
          const size_t unit = t < 2 ? (size_t)16 * k.hd : (size_t)k.sb * 16 * 4;
          const size_t base = (size_t)g * nbc;
          if (first > 0) v.push_back({b, (base + s0) * unit, (size_t)first * unit});
          if (n > first) v.push_back({b, base * unit, (size_t)(n - first) * unit});
        }
      if (frows_)  // (the float K / V rows attn_fa_g reads, [kv head][cap][hd], the same 16-row blocks)
        for (int g = 0; g < k.nkv; ++g)
          for (int t = 0; t < 2; ++t) {
            const size_t unit = (size_t)16 * k.hd * (t == 0 ? kc_bytes() : 2), base = (size_t)g * nbc;
            cl_mem b = t == 0 ? k.L->kc : k.L->vc;
            if (first > 0) v.push_back({b, (base + s0) * unit, (size_t)first * unit});
            if (n > first) v.push_back({b, base * unit, (size_t)(n - first) * unit});
          }
    }
    if (hp.n_qsa && !br.empty())  // QSA pooled keys of the blocks below the saved positions (the KV blocks' range)
      for (auto& L : layers_)
        if (L.pk) v.push_back({L.pk, 0, (size_t)std::min(qsa_blocks(ctx), br[0].second * 16 / L.qr) * hp.qsa_dim * 4});
    for (auto& L : layers_)
      if (hp.recurrent(L.il)) {
        v.push_back({L.S, 0, state_bytes()});
        v.push_back({L.conv_state, 0, conv_bytes()});
      }
    for (size_t i = 0; i < sp.size(); ++i) {
      if (sp[i] < 0) continue;
      const Snap& sn = snaps_[i];
      for (size_t k = 0; k < sn.S.size(); ++k) {
        v.push_back({sn.S[k], 0, state_bytes()});
        v.push_back({sn.conv[k], 0, conv_bytes()});
      }
      if (has_mtp) v.push_back({sn.h, 0, (size_t)hp.n_embd * 4});
    }
    if (has_mtp) v.push_back({hlast_, 0, mtp_hw() * 4});
    return v;
  }
  // Gemma: the device ranges of the sliding-window layers' ring slots for positions [pos - n_swa, pos) (16-key
  // blocks), in state_segs' order; the global layers keep every position and need none
  std::vector<Seg> ring_segs(int p) {
    const auto kl = kv_layers();
    const int lo = std::max({0, p - hp.n_swa, swa_lo_}) / 16;
    std::vector<std::pair<int, int>> br;
    for (const KvGeom& k : kl) br.push_back(k.L->swa ? std::pair{lo, std::max(lo, (p + 15) / 16)} : std::pair{0, 0});
    auto v = state_segs(kl, br, {});
    std::erase_if(v, [&](const Seg& s) { return s.b == hlast_; });  // (the MTP hidden state: Snap::h)
    return v;
  }
  // (a saved state loads only into the same layout: ring caches (swa_rings), the 4-bit V copy, the two-pass mode's float K / V rows)
  int state_flags() const { return (hp.swa_rings ? 1 : 0) | (frows_ ? 4 | 8 : 0) | (q2_ && !q2mm_ ? 16 : 0) | (g4_att8_ ? 32 : 0); }  // (2: the removed
  //  --v4; 8: int8 copies of the drafter's layers only; 16: Gemma's matmuls in one pass, E2B / E4B)
  // header and device ranges of save_state()
  void save_layout(int keep_from, std::vector<int32_t>& hdr, std::vector<Seg>& segs, bool host_snapshot = true) {
    const auto kl = kv_layers();
    // positions [0, pos) (beyond pos a Gemma cache may still hold a longer, rewound conversation: not needed)
    const int hi = pos, slo = hp.swa_rings ? swa_save_lo(keep_from) : 0;
    hdr = {kStateMagic, 2, hp.n_layer, hp.n_embd, hp.n_vocab, state_flags(), has_mtp ? 1 : 0, pos, pos, slo,
           (int)kl.size()};
    std::vector<std::pair<int, int>> br;  // blocks [b0, b1) saved per KV layer
    for (const KvGeom& k : kl) {
      const int b0 = k.L->swa ? slo / 16 : 0, b1 = std::max(b0, (hi + 15) / 16);
      br.push_back({b0, b1});
      hdr.insert(hdr.end(), {k.nkv, k.hd, k.sb, b0, b1});
    }
    std::vector<int> sp;
    for (const Snap& sn : snaps_) sp.push_back(sn.pos);
    hdr.push_back((int)sp.size());
    hdr.insert(hdr.end(), sp.begin(), sp.end());
    hdr.push_back(host_snapshot ? hsnap_.pos : -1);
    segs = state_segs(kl, br, sp);
  }
  // device ranges -> sink / stream -> device ranges, through a host buffer of up to 256 MB: non-blocking copies, one
  // wait per batch
  template <class Fn>
  static void batches(const std::vector<Seg>& segs, Fn fn) {  // fn(first, last, bytes) per batch
    const size_t cap = (size_t)256 << 20;
    for (size_t i = 0; i < segs.size();) {
      size_t j = i, n = 0;
      while (j < segs.size() && (j == i || n + segs[j].n <= cap)) n += segs[j++].n;
      if (!fn(i, j, n)) return;
      i = j;
    }
  }
  bool segs_out(const std::vector<Seg>& segs, const Sink& out) {
    std::vector<uint8_t> hb;
    bool ok = true;
    batches(segs, [&](size_t i, size_t j, size_t n) {
      hb.resize(n);
      size_t o = 0;
      for (size_t k = i; k < j; o += segs[k++].n)
        if (segs[k].n)
          CL_CHECK(clEnqueueReadBuffer(cl_.queue, segs[k].b, CL_FALSE, segs[k].off, segs[k].n, hb.data() + o, 0, nullptr, nullptr));
      CL_CHECK(clFinish(cl_.queue));
      return ok = out(hb.data(), n);
    });
    return ok;
  }
  bool segs_in(const std::vector<Seg>& segs, FILE* f) {
    std::vector<uint8_t> hb;
    bool ok = true;
    batches(segs, [&](size_t i, size_t j, size_t n) {
      hb.resize(n);
      if (!(ok = get(f, hb.data(), n))) return false;
      size_t o = 0;
      for (size_t k = i; k < j; o += segs[k++].n)
        if (segs[k].n)
          CL_CHECK(clEnqueueWriteBuffer(cl_.queue, segs[k].b, CL_FALSE, segs[k].off, segs[k].n, hb.data() + o, 0, nullptr, nullptr));
      CL_CHECK(clFinish(cl_.queue));
      return true;
    });
    return ok;
  }
  GgufFile f_;
  std::string path_;
  ClRt cl_;
  std::string gemm_src_, asr_src_;
  KernelParts ops_, dec_;  // (zero/kernels/ops*.cl, decode*.cl: a kernel built from its part and the ones before it)
  std::map<std::string, cl_kernel> kcache_;
  std::vector<cl_program> progs_;  // released with the engine
  std::unique_ptr<XCache> xc_;     // streamed experts (null: experts on the GPU, as in the former OpenCL engine)
  bool stream_ = false;            // experts streamed (also set in dry runs, where xc_ stays null)
  uint64_t stream_bytes_ = 0;      // all routed experts in the GPU form
  std::vector<cl_mem> mems_;       // every buffer and sub-buffer, released with the engine
  cl_kernel k_rmsnorm_ = nullptr, k_swiglu_ = nullptr, k_gdn_conv_ = nullptr, k_conv_upd_ = nullptr,
            k_gdn_recur_ = nullptr, k_gnorm_ = nullptr, k_attn_prep_ = nullptr, k_attn_ = nullptr,
            k_attn_gate_ = nullptr, k_gdn_gates_ = nullptr, k_attn_q8_ = nullptr,
            k_attn_comb_ = nullptr;
  // (DeltaNet with fewer launches, bit-identical, ARC_GDN_FUSE=0 off: gdn_conv storing the conv history for one
  //  token, gdn_recur computing the gates; not in verification passes, whose replay needs the separate results)
  cl_kernel k_gdn_conv1_ = nullptr, k_gdn_recur_g_ = nullptr;
  const bool gdn_fuse_ = !getenv("ARC_GDN_FUSE") || atoi(getenv("ARC_GDN_FUSE")) != 0;
  // (ARC_GDN_CHUNK=1 / 2, QUEUE F33.1: the chunked DeltaNet kernel gdn_chunk for prompt passes of kGdnChunkMin+
  //  tokens, 2 with its local rows read 16 floats at a time (-DGDN_VEC); off until measured: not bit-identical to
  //  gdn_recur)
  static constexpr int kGdnChunkMin = 32;
  //  3: gdn_chunk_x, its products on XMX (16-lane DPAS; elsewhere gdn_chunk as 1); ARC_GDN_XEMU=1 with 3: DPAS
  //  emulated on any GPU (tests)
  static int gdn_chunk_mode() {
    static const int m = getenv("ARC_GDN_CHUNK") ? std::max(0, std::min(3, atoi(getenv("ARC_GDN_CHUNK")))) : 0;
    return m;
  }
  static bool gdn_chunk_on() { return gdn_chunk_mode() != 0; }
  // (ARC_GDN_XEMU: 1 = DPAS emulated in the GPU's own form (8 lanes where its sub-groups have 8, else 16); 16 / 8 =
  //  that form)
  static int gdn_xemu() {
    static const int e = getenv("ARC_GDN_XEMU") ? atoi(getenv("ARC_GDN_XEMU")) : 0;
    return e == 1 || e == 8 || e == 16 ? e : 0;
  }
  std::vector<Layer> layers_;
  cl_kernel k_add_norm_ = nullptr, k_ple_gate_ = nullptr, k_ple_comb_ = nullptr, k_add_nq_ = nullptr;
  cl_kernel k_rope_tab_ = nullptr;
  // Gemma 4 RoPE tables (sliding, global) and the rows they hold (pos, T)
  cl_mem rope_tab_[2] = {nullptr, nullptr};
  int rt_pos_[2] = {-1, -1}, rt_T_[2] = {0, 0};
  // single-token passes: the next layer's attention input was already normed and quantized (add_norm_quant)
  bool attn_in_ready_ = false, ffn_in_ready_ = false;
  const GgufTensor* ple_tok_embd_ = nullptr;
  GpuWeight ple_model_proj_;
  cl_mem ple_proj_norm_ = nullptr, ple_ = nullptr, ple_tok_ = nullptr, ple_g_ = nullptr;
  std::vector<float> ple_host_;
  int nc_lo_ = -1, nc_hi_ = -1;  // positions processed non-causally in the current forward (an image span)
  cl_mem rpos_ = nullptr;          // per-row (t, h, w) rotary positions of an image chunk (Qwen M-RoPE)
  int use_rpos_ = 0;
  // set_media: image rows inside a token prompt (base[i] -> rows[i], rpos 3 per row)
  const int* media_base_ = nullptr;
  size_t media_n_ = 0;
  const std::vector<const float*>* media_ = nullptr;
  const std::vector<int>* media_rpos_ = nullptr;
  // the media rows of tokens[0 .. T) when they lie in the set_media prompt (nullptr: none there)
  cl_mem rpos_all_ = nullptr;  // (forward_layer_major's segment of positions)
  const float* const* media_rows(const int* tokens, int T) const {
    if (!media_ || tokens < media_base_ || tokens + T > media_base_ + media_n_) return nullptr;
    const float* const* r = media_->data() + (tokens - media_base_);
    for (int t = 0; t < T; ++t)
      if (r[t]) return r;
    return nullptr;
  }
  // the chunk tokens[0 .. T)'s rotary positions from the set_media prompt, for the next run_layer calls
  void media_pos(const int* tokens, int T) {
    if (!media_rpos_ || tokens < media_base_ || tokens + T > media_base_ + media_n_) return;
    if (!rpos_) rpos_ = buf((size_t)max_chunk * 3 * 4);
    CL_CHECK(clEnqueueWriteBuffer(cl_.queue, rpos_, CL_TRUE, 0, (size_t)T * 3 * 4,
                                  media_rpos_->data() + (size_t)(tokens - media_base_) * 3, 0, nullptr, nullptr));
    use_rpos_ = 1;
  }
  // Gemma 4: the layer whose KV cache layer il uses (its own, or llama.cpp's KV reuse rule)
  int g4_kv_layer(int il) const { return il >= hp.n_layer_kv ? hp.n_layer_kv - (hp.swa[il] ? 2 : 1) : il; }
  int swa_cap_ = 0, written_hi_ = 0;  // Gemma: sliding-window ring capacity, highest position written + 1
  // ---- a dense model that nearly fills the GPU share (iGPU, weights over 85 % of it: Gemma 4 31B, 17.7 GB in a 19.2 GiB
  // share, left ~4K tokens of context): part of its weights in host memory (the same DRAM; ~91 % of device memory's
  // read speed, measured 64.4 vs 70.8 GB/s; not counted in the share), prompt chunks of 1024, and the sliding-window
  // rings' margin for moving back one chunk instead of 4096 rows (the rings were 1.2 MB per token per 1K rows: 11 GB
  // at the full 9,216). Models that fit with room (Qwen3.8 27B: 68 % of this laptop's share): unchanged (tight_ false,
  // host_left_ 0)
  bool tight_ = false;
  size_t host_left_ = 0, host_bytes_ = 0;  // (weights still to put in host memory; put there)
  std::vector<void*> host_ptrs_;           // (freed by release_all)
  int swa_margin() const { return tight_ ? std::min(max_chunk, 4096) : 4096; }
  void plan_tight() {
    if (opt.tp_size > 1 || hp.moe || stream_) return;
    double total = 0;
    const bool tied = !f_.find("output.weight");
    for (const GgufTensor& t : f_.tensors)
      if (!(t.name == "token_embd.weight" && !tied) && t.name != "per_layer_token_embd.weight") total += (double)t.nbytes;
    const double budget = (double)device_mem() - driver_slack(), GiB = 1073741824.0;
    tight_ = zrt::integrated() && total > 0.85 * budget;
    double want = opt.host_weights_gb >= 0 ? opt.host_weights_gb * GiB : 0;
    if (opt.host_weights_gb < 0 && tight_) {
      // (what leaves 45 % of the share for the rest, but no more than the RAM free now beyond the share, less 4 GiB:
      //  host memory is RAM all the same. Gemma 4 31B, 24.8 GB free: 5.3 GiB in host memory and a share filled with
      //  context left 2.1 GB free, and generation fell from 3.1 to 1.2 tok/s (paging))
      MEMORYSTATUSEX ms{sizeof ms};
      GlobalMemoryStatusEx(&ms);
      want = std::max(0.0, std::min(total - 0.55 * budget, (double)ms.ullAvailPhys - budget - 4.0 * GiB));
    }
    if (const char* e = getenv("ARC_HOST_WEIGHTS_GB")) want = std::max(0.0, atof(e)) * GiB;  // (tests: 0 turns it off)
    host_left_ = (size_t)want;
  }
  // MTP block and speculative-decoding state
  Layer mtp_;
  GpuWeight eh_proj_;
  cl_mem enorm_ = nullptr, hnorm_ = nullptr, shnorm_ = nullptr;
  cl_kernel k_mtp_concat_ = nullptr;
  std::vector<int> last_drafts_, pend_toks_;
  cl_mem mtp_x_ = nullptr, mtp_e_ = nullptr, hfin_ = nullptr, hfin1_ = nullptr, hlast_ = nullptr, hpend_ = nullptr,
         hver_ = nullptr, mtp_h_ = nullptr;
  bool spec_ = false;  // verification pass: DeltaNet states to S_next, the inputs kept (verify_done)
  std::vector<float> mtp_logits_host_, spec_logits_host_;
  int conv_ct_ = 1;  // tokens per sub-group in gdn_conv (1 is fastest: more sub-groups in flight)
  GpuWeight lm_head_;
  cl_mem out_norm_ = nullptr, tabs_ = nullptr;
  // Flash Next: the final mixer (the output norm), the mixers' buffers (normed streams, gates, low-rank + inject
  // outputs, the block output / mixed input), the PLE block's n-gram embeddings (per chunk view in layer-major
  // prefill) and the token ids before the current position (host copy of the ones kept in the PLE layer's state)
  cl_mem hc_rinv_ = nullptr;  // (single-token mixers: the streams' 1 / rms)
  cl_mem out_hc_norm_ = nullptr, hc_xn_ = nullptr, hc_gate_ = nullptr, hc_lo_ = nullptr, hco_ = nullptr,
         ple_emb_ = nullptr, ple_hist_ = nullptr;
  GpuWeight out_hc_down_, out_hc_up_;
  std::vector<cl_mem> ple_views_;
  const GgufTensor* ple_table_ = nullptr;
  int ple_prev_[2] = {-1, -1};
  // QSA: indexer rows of the pass, normed / rotated queries, block scores of up to 256 rows, the selection masks
  cl_mem qsa_ixo_ = nullptr, qsa_qx_ = nullptr, qsa_sc_ = nullptr, qsa_mask_ = nullptr;
  int qsa_n_ = 0;
  // (a verification pass on Flash Next: the PLE / QSA state before it, the pass's PLE conv rows and the token ids
  //  before it, so that verify_done() rebuilds the state after the accepted tokens)
  cl_mem hc_bak_ = nullptr, ple_sv_ = nullptr;
  int spec_prev_[2] = {-1, -1}, spec_tok_ = -1;
  const GgufTensor* tok_embd_ = nullptr;
  // activations
  cl_mem asct_ = nullptr;  // (MoE prompt tiles: the FFN input's scales token by token, see moe gateup_q)
  cl_mem x_ = nullptr, aq_ = nullptr, asc_ = nullptr, absum_ = nullptr, big_ = nullptr, z_ = nullptr,
         conv_out_ = nullptr, gb_ = nullptr, o_ = nullptr, qrope_ = nullptr, logits_ = nullptr, qi_ = nullptr,
         qsc_ = nullptr, apart_ = nullptr;
  // just-in-time W8 staging, one slot per matrix role (attn_qkv, ffn_up, ...): a layer's matrices stay decoded
  // while several chunks go through it (layer-major prefill)
  static constexpr int kRoles = 16;
  cl_mem stage_q_[kRoles] = {}, stage_h_[kRoles] = {};
  bool has_rgu_ = false;      // MoE: stacked router + shared-expert gate / up weights (Layer::rgu)
  cl_mem rgu_out_ = nullptr, rgu_gu_ = nullptr;
  cl_mem qz_out_ = nullptr, qz_z_ = nullptr, qz_ab_ = nullptr;  // stacked DeltaNet input projections' output, its views
  bool has_qz_ = false;
  size_t mem_mtp_ = 0, mem_bufs_ = 0, mem_load_ = 0;  // GPU bytes: MTP block / drafter, working buffers, after loading
  // router rows padded to 128 (512 bytes: the shared-expert part of rgu_out_ starts at a sub-buffer alignment)
  int rgu_rows() const { return (hp.router_rows() + 127) / 128 * 128; }
  size_t stage_need_q_[kRoles] = {}, stage_need_h_[kRoles] = {};
  size_t stage_have_q_[kRoles] = {}, stage_have_h_[kRoles] = {};  // (bytes allocated: ensure_stage)
  // just-in-time W8 staging large enough for every matrix made since alloc_buffers (the vision encoder's, made later)
  void ensure_stage() {
    for (int r = 0; r < kRoles; ++r)
      if (stage_need_q_[r] > stage_have_q_[r] || stage_need_h_[r] > stage_have_h_[r]) {
        // (the old pair freed, and the role's "already decoded" mark dropped: the new buffers hold nothing yet, and a
        //  weight decoded into the old ones would otherwise be read from them uninitialized)
        free_buf(stage_q_[r]);
        free_buf(stage_h_[r]);
        stage_q_[r] = buf(stage_have_q_[r] = std::max(stage_need_q_[r], stage_have_q_[r]));
        stage_h_[r] = buf(stage_have_h_[r] = std::max(stage_need_h_[r], stage_have_h_[r]));
        stage_owner_[r] = nullptr;
        stage_h8_[r] = false;
      }
  }
  const GpuWeight* stage_owner_[kRoles] = {};
  // Gemma 4 assistant drafter (speculative decoding): small layers of its own whose attention reads the main model's
  // KV cache (its last sliding-window and last global layer); input [embedding(tok) * sqrt(E) | main hidden state]
  struct DraftLayer {
    bool swa = false;
    int hd = 0;
    float out_scale = 1.0f;
    GpuWeight wq, wo, gate, up, down;
    cl_mem attn_norm = nullptr, q_norm = nullptr, post_attn = nullptr, ffn_norm = nullptr, post_ffw = nullptr,
           rope_ff = nullptr;
  };
  std::unique_ptr<GgufFile> df_;
  std::vector<DraftLayer> dl_;
  GpuWeight d_pre_, d_post_, d_lm_;
  cl_mem d_norm_ = nullptr, d_x_ = nullptr, d_cat_ = nullptr, d_h_ = nullptr;
  int d_E_ = 0, d_F_ = 0, d_nh_ = 0, d_swa_ = 0;  // drafter width, FFN size, q heads, sliding window
  float d_base_ = 1e6f, d_base_swa_ = 1e4f;
  bool gemma_mtp_ = false;
  bool mtp_block_ = false;  // the model's own MTP (nextn) block is loaded (Qwen)
  // Qwen3.8 Flash Next's MTP head from a file of its own (mtp-*.gguf, llama.cpp's form; QUEUE F34): its blk.N.*
  // tensors are adopted into f_ (data in this file's mapping); its experts pinned in the expert cache; its head mixer
  std::unique_ptr<GgufFile> mtp_file_;
  bool mtp_ext_ = false;
  cl_mem mtp_hc_norm_ = nullptr;
  GpuWeight mtp_hc_down_, mtp_hc_up_;
  void* mtp_xbuf_ = nullptr;  // (its experts, pinned: pin_mtp_experts; freed in release_all)
  size_t mtp_xbytes_ = 0;
  // floats of a hidden row the MTP block reads (hlast_ / hfin_ / hver_ / hpend_ / mtp_h_) and of its residual rows
  // (mtp_x_): Flash Next's hc streams side by side (the residual before the head mixer), else n_embd after the norm
  size_t mtp_hw() const { return hp.hc && mtp_block_ ? (size_t)hp.hce() : (size_t)hp.n_embd; }
  // MoE: router logits, routing, slots (token, expert pairs grouped by expert), expert activations and outputs
  std::string moe_src_;
  cl_mem gmlp_ = nullptr, gmoe_ = nullptr;  // (Gemma 4 experts, see layer_gemma_ffn)
  cl_mem rl_ = nullptr, mids_ = nullptr, mwts_ = nullptr, sgate_ = nullptr, mcount_ = nullptr, moffs_ = nullptr,
         mcursor_ = nullptr, mtiles_ = nullptr, stok_ = nullptr, swt_ = nullptr, minv_ = nullptr, mh_ = nullptr,
         mhq_ = nullptr, mhsc_ = nullptr, my_ = nullptr, msh_ = nullptr, mtstart_ = nullptr;
  int slots_ = 0;  // slot capacity (max_chunk * n_expert_used): row stride of mhsc_
  // prompt chunks of standard-type experts: a group of moe_group_ experts at a time decoded into W8 (moe_wq_ /
  // moe_wh_), then the int8 GEMM over that group's slot tiles (gemm.cl MOE_TILES)
  cl_mem moe_wq_ = nullptr, moe_wh_ = nullptr;
  // one token, streamed experts: the tiles of its cached and of its missing experts (layer_moe, ARC_XSPLIT)
  cl_mem xt_hit_ = nullptr, xt_miss_ = nullptr;
  std::vector<int> xt_hit_h_, xt_miss_h_;
  int moe_group_ = 0;
  // residual stream of a prompt segment (up to lm_seg_ tokens) for layer-major prefill, and its per-chunk views
  cl_mem x_all_ = nullptr;
  int lm_seg_ = 16384, prog_base_ = 0, prog_total_ = 0;
  std::vector<cl_mem> x_views_;
  std::vector<float> logits_host_, x_host_;
  std::vector<std::pair<std::string, cl_event>> events_;
  int cur_T_ = 0;


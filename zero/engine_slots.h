// A part of class Engine (zero/engine.h), included inside the class body at the place it was cut from (the
// class is the same text), not a header of its own: sizing the context (server --ctx auto), conversation slots (server --slots) and batched decoding over
// slots (QUEUE 20.4).

  // ---------------------------------------------------------------- sizing the context (server --ctx auto)
  // GPU bytes that depend on the context size (same formulas as the allocations: the KV caches of load_attn and
  // load_weights_gemma4, the decode partials and the layer-major residual stream of alloc_buffers); the rest of
  // gpu_bytes does not change with ctx.
  // (the part of the context's memory that is working buffers, allocated with the others: decode partials, the
  // residual stream of layer-major prompt segments)
  size_t ctx_buf_bytes(int c) const {
    c = (c + 15) / 16 * 16;
    if (hp.ds41) return ds_ctx_buf_bytes(c);  // (DeepSeek-V4.1: its own caches and buffers, zero/engine_ds41.h)
    size_t n = 4 * (size_t)std::max(c / 256 + 1, 512) * hp.n_head * (hp.head_dim + 2) * 4;  // apart_
    if (hp.n_qsa) n += (size_t)256 * qsa_blocks(c) * 4 + (size_t)max_chunk * qsa_mask_words(c) * 4;  // QSA scores, masks
    if (lm_prompts()) {
      const int seg = lm_seg_for(c);
      n += (size_t)(seg + max_chunk) * (hp.hce() + (hp.ple_layer >= 0 ? hp.n_embd : 0)) * 4;  // x_all_ (+ PLE rows)
    }
    return n;
  }
  // KV cache bytes of the MTP block's attention layer (allocated with the block)
  // Qwen KV bytes per position and kv head: K and V int8 + their scales
  size_t kv_pos_bytes() const {
    if (!opt.attn_int8) return 4 * (size_t)hp.head_dim;  // (--attn-f32: f16 K and V rows, load_attn's kc / vc)
    if (hp.qwen3) return (size_t)(kc_bytes() + 2) * hp.head_dim;  // (Qwen3 dense: f16 (or f32) K and f16 V rows only)
    return 2 * (size_t)hp.head_dim + 8;
  }
  size_t mtp_kv_bytes(int c) const {
    c = (c + 15) / 16 * 16;
    return mtp_block_ ? (size_t)hp.n_head_kv * c * kv_pos_bytes() : 0;
  }
  // GPU memory by part for a context of c tokens (as loaded, this engine's ctx; a dry load gives the same numbers):
  // weights (incl. DeltaNet states), MTP block / drafter (with its state copies, without its KV), working buffers,
  // the context (KV caches + the buffers that grow with it). Encoders loaded later: gpu_bytes - loaded_bytes().
  struct MemParts { double weights = 0, mtp = 0, buffers = 0, context = 0; };
  MemParts mem_parts(int c) const {
    MemParts m;
    const double here = (double)ctx_bytes(ctx), bufc = (double)ctx_buf_bytes(ctx), mkv = (double)mtp_kv_bytes(ctx);
    m.mtp = std::max(0.0, (double)mem_mtp_ - mkv);
    m.buffers = std::max(0.0, (double)mem_bufs_ - bufc);
    m.weights = std::max(0.0, (double)mem_load_ - here - m.mtp - m.buffers);
    m.context = (double)ctx_bytes(c);
    return m;
  }
  size_t loaded_bytes() const { return mem_load_; }
  bool tight() const { return tight_; }                  // (plan_tight: weights over 85 % of an iGPU's share)
  size_t host_weight_bytes() const { return host_bytes_; }  // (weights in host memory: plan_tight)
  size_t ctx_bytes(int c) const {
    c = (c + 15) / 16 * 16;
    size_t n = ctx_buf_bytes(c);
    if (hp.ds41) return n + ds_ctx_cache_bytes(c);
    if (hp.swa_rings) {
      const int swa_cap = std::min(c, (hp.n_swa + max_chunk + swa_margin() + 255) / 256 * 256);
      for (const Layer& L : layers_)
        if (L.kv_src < 0) {
          const size_t cap = L.swa ? swa_cap : c;
          if (L.i8) n += 2 * (size_t)L.nkv * cap * L.hd + 2 * (size_t)L.nkv * cap * (L.hd / 64) * 4;
          if (frows_) n += (kc_bytes() + 2) * (size_t)L.nkv * cap * L.hd;  // (attn_fa_g's float K / V rows: f32 (or f16) K, f16 V)
        }
    } else {
      int nl = mtp_block_ ? 1 : 0;
      for (const Layer& L : layers_) nl += !hp.recurrent(L.il);
      n += (size_t)nl * hp.n_head_kv * c * kv_pos_bytes();
      n += (size_t)hp.n_qsa * qsa_blocks(c) * hp.qsa_dim * 4;  // QSA pooled keys
    }
    return n;
  }
  // GPU bytes of the server's two recurrent-state snapshots
  size_t snapshot_bytes() const {
    size_t n = 0;
    for (const Layer& L : layers_)
      if (hp.recurrent(L.il)) n += state_bytes() + conv_bytes();
    return 2 * (n + (has_mtp ? (size_t)hp.n_embd * 4 : 0));
  }
  uint64_t device_mem() const {
    cl_ulong v = 0;
    clGetDeviceInfo(cl_.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(v), &v, nullptr);
    return v;
  }

  // The first position rewind() accepts (swa_rings): the sliding-window rings must still hold the n_swa positions before it.
  int rewind_floor() const {
    return hp.swa_rings ? std::max({0, written_hi_ - swa_cap_ + hp.n_swa, swa_lo_ > 0 ? swa_lo_ + hp.n_swa : 0}) : 0;
  }

  // A recurrent-state snapshot kept in host memory (no GPU memory): the server takes it where every conversation
  // of a client shares the prompt (the end of the system prompt), so a new conversation resumes there.
  void snapshot_host() {
    TP_MIRROR(snapshot_host(), TpMirror::kNone, 0);
    hsnap_.S.clear();
    if (hp.ds41) {  // (DeepSeek-V4.1: see snapshot())
      ds_save(hsnap_.ds, hsnap_.dsh);
      hsnap_.pos = pos;
      return;
    }
    for (auto& L : layers_)
      if (hp.recurrent(L.il)) {
        hsnap_.S.push_back(read_buf(L.S, state_bytes()));
        hsnap_.S.push_back(read_buf(L.conv_state, conv_bytes()));
      }
    if (has_mtp) {
      flush_mtp_pending();
      hsnap_.h = read_buf(hlast_, mtp_hw() * 4);
    }
    hsnap_.pos = pos;
  }
  int host_snapshot_pos() const { return hsnap_.pos; }
  void drop_host_snapshot() {
    TP_MIRROR(drop_host_snapshot(), TpMirror::kNone, 0);
    hsnap_ = HostSnap();
  }
  bool restore_host() {
    TP_MIRROR(restore_host(), TpMirror::kNone, 0);
    if (hsnap_.pos < 0) return false;
    if (hp.ds41) {
      ds_load(hsnap_.ds, hsnap_.dsh);
      pos = hsnap_.pos;
      return true;
    }
    size_t k = 0;
    for (auto& L : layers_)
      if (hp.recurrent(L.il)) {
        write_buf(L.S, hsnap_.S[k++]);
        write_buf(L.conv_state, hsnap_.S[k++]);
      }
    if (has_mtp) {
      write_buf(hlast_, hsnap_.h);
      pend_toks_.clear();
    }
    pos = hsnap_.pos;
    // verify() embeds directly, without forward()'s PLE predecessor refresh.
    // The restored convolution state owns the history for this checkpoint.
    if (hp.ple_layer >= 0) ple_sync();
    return true;
  }

  // ---------------------------------------------------------------- conversation slots (server --slots)
  // Several conversations held at once, one step at a time: each slot has its own KV caches, DeltaNet and conv
  // states (with Flash Next's PLE history and QSA tails inside), QSA pooled keys, the MTP block's KV cache and
  // hidden states, DFlash2's context rings, snapshots, position and drafter choice. bind(s) points the engine's state
  // buffers at slot s's memory without copying: the buffer objects stay the same, only their device addresses
  // change, and kernel arguments are set at every launch. Weights, working buffers and the expert cache are shared.
  // Bind only between steps (a verification pass settled with verify_done()).
  int slot_count() const { return std::max(1, (int)slot_ptrs_.size()); }
  // the drafter choice back to its start (DFlash2 first): a conversation that starts over in a slot
  void reset_drafter_choice() {
    dfl_ema_ = 4.0;
    dfl_wait_ = 0;
    dfl_gap_ = 8;
    dfl_probe_ = false;
  }
  int bound_slot() const { return slot_; }
  static constexpr int kMaxSlots = 8;
  static constexpr int kSlotCtxFloor = 32768;
  // GPU share bytes left to the driver (the user's choice): 0.5 GiB by default on iGPUs and discrete cards alike (1.2 GiB
  // until 2026-11-08); ARC_DRIVER_RESERVE_GB=N, or serverz / runz --driver-reserve-gb N (zero-server -DriverReserveGb N),
  // sets it, 0 to 8 GiB. Read once per process
  static double driver_slack() {
    static const double v = [] {
      const char* e = getenv("ARC_DRIVER_RESERVE_GB");
      double g = e ? atof(e) : 0.5;
      if (!(g >= 0.0 && g <= 8.0)) {
        zlog::eprintf("inferzero: ARC_DRIVER_RESERVE_GB=%s not in 0..8: 0.5\n", e);
        g = 0.5;
      }
      return g * 1073741824.0;
    }();
    return v;
  }
  void add_slots(int n) {
    if (n > kMaxSlots) throw std::out_of_range("add_slots: at most " + std::to_string(kMaxSlots) + " slots");
    if (n <= 1 || !slot_ptrs_.empty() || opt.dry) return;
    std::set<cl_mem> seen;
    auto add = [&](cl_mem& f) {
      if (f && seen.insert(f).second) roles_.push_back(&f);
    };
    // (with the verification pass's state, so that a slot's drafts can wait for verify_done while other slots run:
    //  S_next and the inputs verify_done replays, the pass's hidden states, its DFlash2 rows)
    auto add_layer = [&](Layer& L) {
      for (cl_mem* f : {&L.kq, &L.vq, &L.ksc, &L.vsc, &L.conv_state, &L.S, &L.pk, &L.S_next, &L.conv_pre,
                        &L.xin, &L.qkvin, &L.abin})
        add(*f);
      if (!opt.attn_int8 || frows_) { add(L.kc); add(L.vc); }
    };
    for (auto& L : layers_) add_layer(L);
    if (mtp_block_) add_layer(mtp_);
    add(hlast_);
    add(hpend_);
    add(hver_);
    for (auto& D : dfl_l_) { add(D.rk); add(D.rv); }
    if (dfl_) {
      dfl_vrows_rows_ = batch_rows_;
      dfl_vrows_ = buf((size_t)dfl_vrows_rows_ * hp.n_embd * 4);
      add(dfl_vrows_);
    }
    // views into those buffers (the PLE history and QSA tails in a recurrent state) follow their buffer
    for (ZMem* m : zrt::D().mems) {
      if (m->owner) continue;
      for (size_t i = 0; i < roles_.size(); ++i) {
        const ZMem* r = *roles_[i];
        if (m != r && m->ptr >= r->ptr && m->ptr < r->ptr + r->size) {
          subs_.push_back({m, (int)i, (size_t)(m->ptr - r->ptr)});
          break;
        }
      }
    }
    // the addresses the buffer objects had (put back before they are freed); S_next trades places with S
    for (cl_mem* r : roles_) orig_ptrs_.push_back({*r, (*r)->ptr});
    for (auto& v : subs_) orig_ptrs_.push_back({v.m, v.m->ptr});
    {  // (slot_bytes, which --ctx auto counts with, must be what a slot really takes)
      size_t per = 0;
      for (cl_mem* r : roles_) per += (*r)->size;
      const size_t est = slot_bytes(ctx) - snapshot_bytes();
      slot_real_bytes_ = per;
      slot_est_bytes_ = est;
      if (per > est + est / 100 || getenv("ARC_SLOT_BYTES"))
        zlog::eprintf("inferzero: a slot takes %.1f MiB, slot_bytes() counts %.1f MiB\n", per / 1048576.0, est / 1048576.0);
    }
    slot_ptrs_.assign(n, {});
    slot_host_.assign(n, SlotHost());
    for (cl_mem* r : roles_) slot_ptrs_[0].push_back((*r)->ptr);
    for (int s = 1; s < n; ++s)
      for (cl_mem* r : roles_) {
        cl_mem b = buf((*r)->size);
        fill(b, (*r)->size);
        slot_ptrs_[s].push_back(b->ptr);
      }
    clFinish(cl_.queue);
  }
  void bind(int s) {
    TP_MIRROR(bind(s), TpMirror::kNone, 0);
    if (s < 0 || s >= slot_count()) throw std::out_of_range("bind: slot " + std::to_string(s) + " out of range");
    if (s == slot_) return;
    if (!last_drafts_.empty() && !vrows_) throw std::runtime_error("bind: a verification pass is not settled");
    clFinish(cl_.queue);  // (cancellation reads and host snapshots work on the bound buffers)
    park();
    unpark(s);
  }

  // ---------------------------------------------------------------- batched decoding over slots (QUEUE 20.4)
  // One generation step for several slots at once: row b is the next token of slot slots[b] (at that slot's position).
  // The parts that work row by row (norms, every projection and FFN / MoE GEMV, the output head) run once for all rows,
  // so the weights are read once per step instead of once per slot; the parts that follow a sequence (DeltaNet conv
  // and recurrence, RoPE and the K/V write, attention) run per slot on its rows with its state (point_to). Then each
  // slot's MTP block KV / hidden state and DFlash2 context take their row. Returns the logits rows (row b: slot b).
  // Dense and MoE Qwen models and Flash Next (not Gemma; Flash Next without a drafter: its PLE token ids, conv and
  // history and its QSA blocks per sequence too); every slot's drafts settled; up to 1 + max(drafts, slots - 1) rows.
  // The bound slot stays bound.
  bool batch_ok() const { return (!hp.hc || !has_mtp) && opt.attn_int8 && slot_ptrs_.size() > 1; }
  bool batch_verify_ok() const { return batch_ok() && has_mtp && !gemma_mtp_ && !hp.hc; }
  const std::vector<float>& forward_batch(const std::vector<int>& slots, const std::vector<int>& toks) {
    TP_MIRROR(forward_batch(slots, toks), TpMirror::kBatch, (int)slots.size());
    const int B = (int)slots.size();
    if (!batch_ok() || B < 1 || B != (int)toks.size() || B > batch_rows_)
      throw std::runtime_error("forward_batch: not for this model or too many rows");
    check_batch(slots, std::vector<int>(B, 1));
    BatchGuard guard{*this, slot_};
    for (int b = 0; b < B; ++b) {  // (pending MTP rows of accepted drafts first, each in its slot)
      bind(slots[b]);
      if (has_mtp) flush_mtp_pending();
      ring_guard(1);
    }
    bind(guard.bound);
    clFinish(cl_.queue);
    park();
    guard.parked = true;
    batch_.clear();
    for (int b = 0; b < B; ++b) {
      const SlotHost& h = slot_host_[slots[b]];
      batch_.push_back({slots[b], b, 1, h.pos, h.rope_shift});
    }
    const int Mp = 16;
    embed(toks.data(), B, Mp);
    // (DFlash2: every row's features go to the first accumulator; each slot's ring takes its row below)
    pos = 0;
    dfl_cap_lo_ = dfl_cap_base_ = 0;
    dfl_cap_end_ = B;
    for (auto& L : layers_) run_layer(L, B, Mp);
    fail_point();
    const std::vector<BatchSeq> seqs = batch_;
    batch_.clear();
    head_rows(0, B, out_norm_, nullptr);
    if (batch_logits_host_.size() < (size_t)hp.n_vocab * B) batch_logits_host_.resize((size_t)hp.n_vocab * batch_rows_);
    // (rows of n_logits(): the vocabulary, or a split model's shard of it)
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, (size_t)n_logits() * B * 4, batch_logits_host_.data(), 0,
                                 nullptr, nullptr));
    for (const BatchSeq& q : seqs) {
      point_to(q.slot);
      pos = q.pos;
      rope_shift = q.rs;
      if (mtp_block_ && hp.hc) {  // (Flash Next, QUEUE F34: the row's residual streams as they are)
        const size_t W4 = mtp_hw() * 4;
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hlast_, hfin_, 0, 0, W4, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, x_, hfin_, (size_t)q.row * W4, W4, W4, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hfin_, hlast_, W4, 0, W4, 0, nullptr, nullptr));
        mtp_rows(&toks[q.row], hfin_, 1, q.pos, false);
      } else if (mtp_block_) {  // as mtp_prefill for its row: hlast_, then the row's final hidden state, then the MTP block
        const size_t E4 = (size_t)hp.n_embd * 4;
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hlast_, hfin_, 0, 0, E4, 0, nullptr, nullptr));
        float eps = hp.eps;
        int E = hp.n_embd, M = Mp, TT = 1, r0 = q.row, wh = 1, a32 = 1;
        set_args(k_rmsnorm_, 0, x_, r0, out_norm_, E, M, TT, eps, aq_, asc_, absum_, hfin1_, wh, a32);
        size_t gws[2] = {256, 1}, lws[2] = {256, 1};
        run("mtp_concat", k_rmsnorm_, 2, gws, lws);
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hfin_, hlast_, E4, 0, E4, 0, nullptr, nullptr));
        mtp_rows(&toks[q.row], hfin_, 1, q.pos, false);
      }
      if (has_mtp && hlast_ && mtp_block_ && hp.hc) {
        // (done above: hlast_ holds the row's streams)
      } else if (has_mtp && hlast_) {  // (as head() leaves it: the normed hidden state of the row; the drafters start there)
        float eps = hp.eps;
        int E = hp.n_embd, M = Mp, TT = 1, r0 = q.row, wh = 1, a32 = 1;
        set_args(k_rmsnorm_, 0, x_, r0, out_norm_, E, M, TT, eps, aq_, asc_, absum_, hlast_, wh, a32);
        size_t gws[2] = {256, 1}, lws[2] = {256, 1};
        run("rmsnorm_quant", k_rmsnorm_, 2, gws, lws);
      }
      if (dfl_) dfl_commit(0, q.row, 1, q.pos);
    }
    clFinish(cl_.queue);
    for (const BatchSeq& q : seqs) slot_host_[q.slot].pos = q.pos + 1;  // (the slots advance once the pass is done)
    dfl_cap_lo_ = dfl_cap_end_ = 0;
    guard.parked = false;
    unpark(guard.bound);
    collect_profile();
    return batch_logits_host_;
  }
  int batch_rows() const { return batch_rows_; }
  // (tests of the error paths: the fail_after-th verification or batched pass from now throws after its layers have
  //  written their state; 0 = never. The server sets it from ARC_FAIL_PASS)
  long fail_after = 0;
  void fail_point() {
    if (fail_after > 0 && --fail_after == 0) throw std::runtime_error("injected failure (ARC_FAIL_PASS)");
  }
  size_t slot_real_bytes_ = 0, slot_est_bytes_ = 0;  // (add_slots: one slot's buffers, and what slot_bytes() counted)
 private:
  // a batch's slots: in range, each once, none with a verification pass not settled, n[b] more positions within ctx
  void check_batch(const std::vector<int>& slots, const std::vector<int>& n) const {
    std::vector<char> seen(slot_count(), 0);
    for (size_t b = 0; b < slots.size(); ++b) {
      const int s = slots[b];
      if (s < 0 || s >= slot_count()) throw std::out_of_range("batch: slot " + std::to_string(s) + " out of range");
      if (seen[s]++) throw std::runtime_error("batch: slot " + std::to_string(s) + " twice");
      const bool bound = s == slot_;
      const bool pending = bound ? !last_drafts_.empty() || vrows_ : !slot_host_[s].last_drafts.empty() || slot_host_[s].vrows;
      if (pending) throw std::runtime_error("batch: slot " + std::to_string(s) + " has a verification pass not settled");
      if ((bound ? pos : slot_host_[s].pos) + n[b] > ctx) throw std::runtime_error("context overflow");
    }
  }
  // a batched pass's state, cleared however the pass ends, and the slot bound before it bound again. A pass that
  // threw leaves its slots' positions and pending drafts as they were (they change only once it is done), but their
  // device state (KV rows past the position, DeltaNet states) may be partly written: the caller starts those
  // conversations over (reset).
  struct BatchGuard {
    Engine& e;
    int bound;
    bool parked = false;
    ~BatchGuard() {
      e.batch_.clear();
      e.spec_ = false;
      e.dfl_cap_lo_ = e.dfl_cap_base_ = e.dfl_cap_end_ = 0;
      if (parked) e.unpark(bound);
      else if (e.slot_ != bound) {  // (bind or flush_mtp_pending threw in the loop before park: host work only,
        e.park();                   // no queue wait or pending-drafts check that could throw again)
        e.unpark(bound);
      }
    }
  };
 public:
  // verify() for several slots in one pass: rows[b] = {token, drafts...} of slot slots[b] (drafts from mtp_draft in
  // that slot). Returns the logits rows of every slot one after the other; then each slot settles its own with
  // verify_done() (bound to it, in any order, other slots' passes may run in between). At most batch_rows() rows
  // (8 or fewer keep the single-token kernels: the rows bit-identical to verify() of each slot alone).
  const std::vector<float>& verify_batch(const std::vector<int>& slots, const std::vector<std::vector<int>>& rows) {
    const int B = (int)slots.size();
    int R = 0;
    for (const auto& r : rows) R += (int)r.size();
    if (!batch_verify_ok() || B < 1 || B != (int)rows.size() || R > batch_rows_ || (dfl_ && R > dfl_vrows_rows_))
      throw std::runtime_error("verify_batch: not for this model or too many rows");
    std::vector<int> n(B);
    for (int b = 0; b < B; ++b) {
      if (rows[b].size() < 2 || (int)rows[b].size() > 1 + max_drafts()) throw std::runtime_error("verify_batch: drafts");
      n[b] = (int)rows[b].size();
    }
    check_batch(slots, n);
    BatchGuard guard{*this, slot_};
    for (int b = 0; b < B; ++b) {
      bind(slots[b]);
      flush_mtp_pending();
      ring_guard(n[b]);
    }
    bind(guard.bound);
    clFinish(cl_.queue);
    park();
    guard.parked = true;
    batch_.clear();
    std::vector<int> toks;
    for (int b = 0; b < B; ++b) {
      const SlotHost& h = slot_host_[slots[b]];
      batch_.push_back({slots[b], (int)toks.size(), (int)rows[b].size(), h.pos, h.rope_shift});
      toks.insert(toks.end(), rows[b].begin(), rows[b].end());
    }
    const int Mp = 16;
    embed(toks.data(), R, Mp);
    pos = 0;
    dfl_cap_lo_ = dfl_cap_base_ = 0;
    dfl_cap_end_ = R;
    spec_ = true;
    for (auto& L : layers_) run_layer(L, R, Mp);
    fail_point();
    spec_ = false;
    const std::vector<BatchSeq> seqs = batch_;
    batch_.clear();
    head_rows(0, R, out_norm_, nullptr);
    const size_t V = (size_t)n_logits();  // (= hp.n_vocab; a split model's shard: its vocabulary rows)
    if (batch_logits_host_.size() < V * R) batch_logits_host_.resize(V * batch_rows_);
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, V * R * 4, batch_logits_host_.data(), 0,
                                 nullptr, nullptr));
    const size_t E4 = (size_t)hp.n_embd * 4;
    for (int b = 0; b < B; ++b) {  // each slot's hidden states (verify_done) and DFlash2 rows of the pass
      const BatchSeq& q = seqs[b];
      point_to(q.slot);
      if (mtp_block_ && hp.hc) {  // (Flash Next's MTP head reads the rows' residual streams, QUEUE F34)
        const size_t W4 = mtp_hw() * 4;
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, x_, hver_, (size_t)q.row * W4, 0, (size_t)q.n * W4, 0, nullptr, nullptr));
      } else {
        float eps = hp.eps;
        int E = hp.n_embd, M = Mp, nn = q.n, r0 = q.row, wh = 1, a32 = 1;
        set_args(k_rmsnorm_, 0, x_, r0, out_norm_, E, M, nn, eps, aq_, asc_, absum_, hver_, wh, a32);
        size_t gws[2] = {256, (size_t)q.n}, lws[2] = {256, 1};
        run("rmsnorm_quant", k_rmsnorm_, 2, gws, lws);
      }
      if (dfl_)
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, dfl_acc_[0], dfl_vrows_, (size_t)q.row * E4, 0, (size_t)q.n * E4, 0, nullptr, nullptr));
    }
    std::vector<std::vector<int>> pend(B);  // (allocated before any slot changes: the commit below cannot throw)
    for (int b = 0; b < B; ++b) pend[b].assign(rows[b].begin() + 1, rows[b].end());
    clFinish(cl_.queue);
    for (int b = 0; b < B; ++b) {  // (the slots advance once the pass is done)
      SlotHost& h = slot_host_[seqs[b].slot];
      h.pos = seqs[b].pos + seqs[b].n;
      h.last_drafts.swap(pend[b]);  // (now pending: verify_done settles it)
      h.vrows = true;
    }
    dfl_cap_lo_ = dfl_cap_end_ = 0;
    guard.parked = false;
    unpark(guard.bound);
    collect_profile();
    return batch_logits_host_;
  }

 private:
  // the bound slot's host state and device addresses back into its records
  void park() {
    SlotHost& h = slot_host_[slot_];
    h.pos = pos;
    h.rope_shift = rope_shift;
    h.written_hi = written_hi_;
    h.swa_lo = swa_lo_;
    h.pend = std::move(pend_toks_);
    h.ple_prev[0] = ple_prev_[0];
    h.ple_prev[1] = ple_prev_[1];
    h.snaps = std::move(snaps_);
    h.hsnap = std::move(hsnap_);
    h.dfl_ema = dfl_ema_;
    h.dfl_wait = dfl_wait_;
    h.dfl_gap = dfl_gap_;
    h.dfl_probe = dfl_probe_;
    h.dfl_used = dfl_used_;
    h.vrows = vrows_;
    h.last_drafts = std::move(last_drafts_);
    for (size_t i = 0; i < roles_.size(); ++i) slot_ptrs_[slot_][i] = (*roles_[i])->ptr;
  }
  // slot s's host state and device addresses into the engine
  void unpark(int s) {
    slot_ = s;
    SlotHost& g = slot_host_[s];
    pos = g.pos;
    rope_shift = g.rope_shift;
    written_hi_ = g.written_hi;
    swa_lo_ = g.swa_lo;
    pend_toks_ = std::move(g.pend);
    ple_prev_[0] = g.ple_prev[0];
    ple_prev_[1] = g.ple_prev[1];
    snaps_ = std::move(g.snaps);
    hsnap_ = std::move(g.hsnap);
    dfl_ema_ = g.dfl_ema;
    dfl_wait_ = g.dfl_wait;
    dfl_gap_ = g.dfl_gap;
    dfl_probe_ = g.dfl_probe;
    dfl_used_ = g.dfl_used;
    vrows_ = g.vrows;
    last_drafts_ = std::move(g.last_drafts);
    point_to(s);
    attn_in_ready_ = ffn_in_ready_ = false;
    rt_pos_[0] = rt_pos_[1] = -1;
  }
  // the state buffers at slot s's memory (device addresses only; inside a batched pass, every slot parked)
  void point_to(int s) {
    for (size_t i = 0; i < roles_.size(); ++i) (*roles_[i])->ptr = slot_ptrs_[s][i];
    for (auto& v : subs_) v.m->ptr = (*roles_[v.role])->ptr + v.off;
  }
  // a batched pass: each sequence's rows (slot, first row, rows, position, rotary shift); empty otherwise
  struct BatchSeq { int slot, row, n, pos, rs; };
  std::vector<BatchSeq> batch_;
  bool vrows_ = false;            // the pending verification is a batched one: its DFlash2 rows in dfl_vrows_
  int dfl_vrows_rows_ = 0;
  cl_mem dfl_vrows_ = nullptr;    // (slots with DFlash2: a slot's rows of a batched verification pass)
  int batch_rows_ = 1;
  std::vector<float> batch_logits_host_;
  // a working buffer from byte `off` on (rows of a sequence inside a batched pass), made once
  std::map<std::pair<ZMem*, size_t>, cl_mem> views_;
  cl_mem from(cl_mem b, size_t off) {
    if (!off || !b) return b;
    cl_mem& v = views_[{b, off}];
    if (!v) {
      if (off >= b->size) throw std::runtime_error("from: offset past the buffer");
      for (cl_mem* role : roles_) {  // (a view of a slot's buffer would not follow point_to: working buffers only)
        const ZMem* r = *role;
        if (b->ptr >= r->ptr && b->ptr < r->ptr + r->size) throw std::logic_error("from: a view of a slot's state buffer");
      }
      cl_buffer_region r{off, b->size - off};
      cl_int err;
      v = clCreateSubBuffer(b, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &err);
      CL_CHECK(err);
      mems_.push_back(v);
    }
    return v;
  }

 public:
  // GPU bytes of one more slot with a context of c tokens: the KV caches (and QSA pooled keys), the recurrent states,
  // the MTP block's hidden states, DFlash2's rings, the snapshots
  // (the roles add_slots copies: the KV caches and QSA keys of ctx_bytes; per recurrent layer S and conv_state, with
  //  drafts also S_next, conv_pre, xin, qkvin, abin; hlast_ / hpend_ / hver_; DFlash2's rings and rows)
  size_t slot_bytes(int c) const {
    size_t n = ctx_bytes(c) - ctx_buf_bytes(c);
    const bool spec = opt.mtp && !hp.gemma;
    for (const Layer& L : layers_)
      if (hp.recurrent(L.il)) {
        n += state_bytes() + conv_bytes();
        if (spec)
          n += state_bytes() + conv_bytes() + 2 * (size_t)(1 + kMaxDrafts) * hp.conv_dim() * 4 +
               (size_t)(1 + kMaxDrafts) * 2 * hp.ssm_nv * 4;
      }
    if (has_mtp) n += mtp_hw() * 4 * (1 + max_drafts() + 1 + max_drafts());  // (hlast_, hpend_, hver_)
    n += dfl_l_.size() * 2 * (size_t)dfl_R_ * dfl_nkv_ * 128 * 4;
    if (dfl_) n += (size_t)batch_rows_ * hp.n_embd * 4;
    return n + snapshot_bytes();
  }


// A part of class Engine (zero/engine.h), included inside the class body at the place it was cut from (the
// class is the same text), not a header of its own: the activation buffers and the memory plan (alloc_buffers).

  void alloc_buffers() {
    const size_t M = max_chunk;
    const int maxK = aq_max_k();
    const int ldq = hp.ds41 ? 2 * hp.n_ff_shexp : hp.n_head * hp.head_dim * 2 + 2 * hp.n_head_kv * hp.head_dim;
    const int big_w = std::max({hp.conv_dim(), 2 * hp.n_ff, ldq});
    x_ = buf(M * hp.hce() * 4);
    aq_ = buf(M * maxK * (q2mm_ ? 2 : 1) + h8f_off_ * 4);  // (h8f_off_: ARC_H8_FLOAT's float copy, zero/engine_h8.h)
    x16_reserve(M, maxK);  // (the f16 XMX GEMM's activations, when it is on: zero/xe2/engine_xe2.h)
    h8_reserve(M, maxK);   // (the H8 GEMM's rotated activations and staged scales, ARC_H8=1: zero/engine_h8.h)
    if (hp.hc) {
      hc_xn_ = buf(M * hp.hce() * 4);
      hc_gate_ = buf(M * hp.hce() * 4);
      hc_lo_ = buf(M * (size_t)std::max(layers_[0].hc_down[0].N, out_hc_down_.N) * 4);
      hco_ = buf(M * hp.n_embd * 4);
      if (hp.n_qsa) {
        qsa_ixo_ = buf(M * (size_t)(1 + hp.qsa_heads) * hp.qsa_dim * 4);
        qsa_qx_ = buf(M * (size_t)hp.qsa_heads * hp.qsa_dim * 4);
        qsa_sc_ = buf((size_t)256 * qsa_blocks(ctx) * 4 + 64);
        qsa_mask_ = buf(M * (size_t)qsa_mask_words(ctx) * 4 + 64);
      }
      if (hp.ple_layer >= 0) {
        ple_emb_ = buf(M * hp.n_embd * 4);
        hc_bak_ = buf(conv_bytes() - ple_hist_off());
        ple_sv_ = buf((size_t)(1 + kMaxDrafts) * hp.hce() * 4);
        ple_host_.resize(M * hp.n_embd);
        if (!opt.dry) {  // the PLE conv history inside the PLE layer's recurrent state (see conv_bytes)
          cl_buffer_region r{ple_hist_off(), (size_t)ple_hist() * hp.hce() * 4};
          cl_int err;
          ple_hist_ = clCreateSubBuffer(layers_[hp.ple_layer].conv_state, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                                        &r, &err);
          CL_CHECK(err);
          mems_.push_back(ple_hist_);
        }
      }
    }
    asc_ = buf(M * (maxK / 32) * 4 * (q2mm_ ? 2 : 1));
    if (hp.moe) asct_ = buf(M * (hp.n_embd / 32) * 4);
    absum_ = buf(M * (maxK / 16) * 2 * (q2mm_ ? 2 : 1));
    big_ = buf(M * big_w * 4);
    // MoE: the expert activations / outputs (mh_ [slots][F], my_ [slots][E] f32) share memory with buffers that are
    // only used inside the attention / DeltaNet part of a layer (z_; conv_out_, o_, qrope_): never live at the same time
    // (Qwen3.6 35B, 2048-token chunks: 167 MB, ~18K tokens of context)
    const size_t z_bytes = std::max<size_t>(M * hp.ssm_d_inner * 4, 64);
    const int nhd = hp.ds41 ? 16 : hp.n_head * hp.head_dim;  // (DeepSeek-V4.1: no Qwen attention buffers)
    const size_t cv_bytes = (std::max<size_t>(M * hp.conv_dim() * 4, 64) + 4095) / 4096 * 4096,
                 o_bytes = (M * std::max(hp.ssm_d_inner, nhd) * 4 + 4095) / 4096 * 4096,
                 qr_bytes = M * nhd * 4;
    const bool share = hp.moe;
    auto sub = [&](cl_mem parent, size_t off, size_t bytes) -> cl_mem {
      if (opt.dry) return nullptr;
      cl_int err;
      cl_buffer_region r{off, bytes};
      cl_mem m = clCreateSubBuffer(parent, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &err);
      CL_CHECK(err);
      mems_.push_back(m);
      return m;
    };
    if (share) {
      const size_t S = M * hp.n_expert_used, hb = S * hp.n_ff_exp_pad * 4, yb = S * hp.n_embd * 4;
      cl_mem a = buf(std::max(z_bytes, hb));
      z_ = sub(a, 0, z_bytes);
      mh_ = sub(a, 0, hb);
      cl_mem b = buf(std::max(cv_bytes + o_bytes + qr_bytes, yb));
      conv_out_ = sub(b, 0, cv_bytes);
      o_ = sub(b, cv_bytes, o_bytes);
      qrope_ = sub(b, cv_bytes + o_bytes, qr_bytes);
      my_ = sub(b, 0, yb);
    } else {
      z_ = buf(z_bytes);
    }
    if (has_qz_) {
      const size_t C = hp.conv_dim(), Di = hp.ssm_d_inner, nab = 2 * hp.ssm_nv;
      qz_out_ = buf((C + Di + nab + 64) * 4);
      if (!opt.dry) {
        cl_int err;
        cl_buffer_region rz{C * 4, Di * 4}, ra{(C + Di) * 4, (nab + 64) * 4};
        qz_z_ = clCreateSubBuffer(qz_out_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rz, &err);
        CL_CHECK(err);
        mems_.push_back(qz_z_);
        qz_ab_ = clCreateSubBuffer(qz_out_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &ra, &err);
        CL_CHECK(err);
        mems_.push_back(qz_ab_);
      }
    }
    if (!share) {
      conv_out_ = buf(std::max<size_t>(M * hp.conv_dim() * 4, 64));
      o_ = buf(M * std::max(hp.ssm_d_inner, nhd) * 4);
      qrope_ = buf(M * nhd * 4);
    }
    gb_ = buf(std::max<size_t>(M * 2 * hp.ssm_nv * 4, 64));
    qi_ = buf(M * nhd);
    if (frows_) qf_ = buf(M * hp.n_head * hp.head_dim * 4);  // (the f32 queries of attn_fa_g)
    qsc_ = buf(M * hp.n_head * (hp.head_dim / 64) * 4);  // q scales per 64-dim block
    if (hp.swa_rings) {  // (RoPE tables per layer type: sliding / global)
      rope_tab_[0] = buf(M * (hp.hd_swa / 2) * 8);
      rope_tab_[1] = buf(M * (hp.hd_full / 2) * 8);
    }
    if (hp.qwen3) rope_tab_[1] = buf(M * (hp.head_dim / 2) * 8);  // (Qwen3 dense: attn_prep_g with the table)
    // decode key-split partials: up to 4 rows x splits (small splits only while splits * sub-groups < 512)
    apart_ = hp.ds41 ? buf(64) : buf((size_t)4 * std::max(ctx / 256 + 1, 512) * hp.n_head * (hp.head_dim + 2) * 4);
    // rows of a speculative verification pass or a batched pass over the slots
    // (DFlash2's head writes block - 1 draft rows whatever the drafts kept)
    batch_rows_ = std::min(16, std::max({1 + max_drafts(), opt.slots, 1, dfl_ ? dfl_B_ : 1}));
    logits_ = buf((size_t)n_logits() * 4 * batch_rows_);
    if (hp.moe) {
      const size_t S = M * hp.n_expert_used, F = hp.n_ff_exp_pad, NE = hp.n_expert;
      slots_ = (int)S;
      rl_ = buf(M * hp.router_rows() * 4);
      if (has_rgu_) {  // stacked router + shared gate / up output; the gate / up part as its own (aligned) buffer
        rgu_out_ = buf((size_t)(rgu_rows() + 2 * hp.n_ff_shexp) * 4);
        if (!opt.dry) {
          cl_buffer_region reg{(size_t)rgu_rows() * 4, (size_t)2 * hp.n_ff_shexp * 4};
          cl_int err;
          rgu_gu_ = clCreateSubBuffer(rgu_out_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err);
          CL_CHECK(err);
          mems_.push_back(rgu_gu_);
        }
      }
      mids_ = buf(S * 4);
      mwts_ = buf(S * 4);
      sgate_ = buf(M * 4);
      mcount_ = buf(NE * 4);
      moffs_ = buf((NE + 1) * 4);
      mcursor_ = buf(NE * 4);
      mtiles_ = buf((1 + 3 * (S / 16 + NE + S)) * 4);  // enough for 1-slot tiles too
      stok_ = buf(S * 4);
      swt_ = buf(S * 4);
      minv_ = buf(S * 4);
      if (!share) {
        mh_ = buf(S * F * 4);
        my_ = buf(S * hp.n_embd * 4);
      }
      mhq_ = buf(S * F * (q2_ ? 2 : 1));
      mhsc_ = buf(S * (F / 32) * 4 * (q2_ ? 2 : 1));
      if (q2_) {  // (two-pass mode: the down input's residual at these offsets; the up projection apart from the gate)
        q2m_hq_ = S * F;
        q2m_sc_ = S * (F / 32) * 4;
        mh2_ = buf(S * F * 4);
      }
      msh_ = buf(M * hp.n_embd * 4);
      if (hp.gemma) {  // (Gemma 4 experts: msh_ stays zero; the dense FFN's normed output; the experts' sum)
        fill(msh_, M * hp.n_embd * 4);
        gmlp_ = buf(M * hp.n_embd * 4);
        gmoe_ = buf(M * hp.n_embd * 4);
      }
      mtstart_ = buf((NE + 1) * 4);
      fill(mcount_, NE * 4);
      if (moe_gemm_ok()) {
        static const int g_env = getenv("ARC_MOE_G") ? atoi(getenv("ARC_MOE_G")) : 64;
        moe_group_ = std::clamp(g_env, 1, (int)NE);
        while (NE % moe_group_) --moe_group_;
        const size_t G = moe_group_, rows = std::max(F, (size_t)hp.n_embd);
        moe_wq_ = buf(G * F * hp.n_embd);
        moe_wh_ = buf(w8_bytes_h((int)(G * rows), (int)(F * hp.n_embd / rows)));
      }
    }
    if (hp.n_ple) {
      const size_t np = (size_t)hp.n_ple * hp.n_layer;
      ple_ = buf(M * np * 4);
      ple_tok_ = buf(M * np * 4);
      ple_g_ = buf(M * hp.n_ple * 4);
      ple_host_.resize(M * np);
    }
    if (gemma_mtp_) {
      d_x_ = buf((size_t)16 * d_E_ * 4);
      d_cat_ = buf((size_t)2 * hp.n_embd * 4);
      d_h_ = buf((size_t)hp.n_embd * 4);
      fill(d_x_, (size_t)16 * d_E_ * 4);
    }
    if (dfl_) dfl_buffers();
    if (has_mtp) {
      const size_t mb = mtp_block_ ? M : 1;  // (DFlash alone: no MTP block rows)
      const size_t hw = mtp_hw();  // (Flash Next: hc streams a row)
      mtp_x_ = buf(mb * hw * 4);
      mtp_e_ = buf(mb * hp.n_embd * 4);
      hfin_ = buf((mb + 1) * hw * 4);
      cl_buffer_region reg{hw * 4, mb * hw * 4};
      if (!opt.dry) {
        cl_int err;
        hfin1_ = clCreateSubBuffer(hfin_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err);
        CL_CHECK(err);
        mems_.push_back(hfin1_);
      }
      hlast_ = buf(hw * 4);
      hpend_ = buf((size_t)max_drafts() * hw * 4);  // (hidden states of the accepted drafts)
      hver_ = buf((size_t)(1 + max_drafts()) * hw * 4);
      mtp_h_ = buf(hw * 4);
      fill(hlast_, hw * 4);
      mtp_logits_host_.resize(hp.n_vocab);
    }
    for (int r = 0; r < kRoles; ++r)
      if (stage_need_q_[r]) {
        stage_q_[r] = buf(stage_have_q_[r] = stage_need_q_[r]);
        stage_h_[r] = buf(stage_have_h_[r] = stage_need_h_[r]);
      }
    if (hp.ds41) alloc_ds41();
    if ((!w8_resident || stream_) && !hp.ds41) {
      // residual stream of a layer-major segment (+ one chunk of padding rows)
      lm_seg_ = lm_seg_for(ctx);
      const size_t rows = (size_t)lm_seg_ + max_chunk;
      x_all_ = buf(rows * hp.hce() * 4);
      cl_mem ple_all = hp.ple_layer >= 0 ? buf(rows * hp.n_embd * 4) : nullptr;
      for (size_t r0 = 0; r0 + max_chunk <= rows; r0 += max_chunk) {
        cl_buffer_region reg{r0 * hp.hce() * 4, (size_t)max_chunk * hp.hce() * 4};
        cl_int err;
        if (opt.dry) continue;
        x_views_.push_back(clCreateSubBuffer(x_all_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err));
        mems_.push_back(x_views_.back());
        CL_CHECK(err);
        if (ple_all) {
          cl_buffer_region rp{r0 * hp.n_embd * 4, (size_t)max_chunk * hp.n_embd * 4};
          ple_views_.push_back(clCreateSubBuffer(ple_all, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rp, &err));
          mems_.push_back(ple_views_.back());
          CL_CHECK(err);
        }
      }
    }
    fill(x_, M * hp.hce() * 4);
    logits_host_.resize(n_logits());
    if (has_mtp) spec_logits_host_.resize((size_t)hp.n_vocab * (1 + max_drafts()));
    x_host_.resize(M * hp.hce());
  }
  // tokens per layer-major prompt segment (Flash Next: 8192, its residual rows are 4 streams wide)
  int lm_seg_for(int c) const { return std::max(max_chunk, std::min(c, hp.hc ? 8192 : 16384) / max_chunk * max_chunk); }


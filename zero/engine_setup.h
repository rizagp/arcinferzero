// A part of class Engine (zero/engine.h), included inside the class body at the place it was cut from (the
// class is the same text), not a header of its own: setup: the model's hyperparameters per architecture, the weight plan, kernel builds (kernel()).

  // ---------------------------------------------------------------- setup
  void load_hparams() {
    const std::string a = f_.get_str("general.architecture");
    hp.arch = a;
    if (a == "gemma4") return load_hparams_gemma4();
    if (a == "qwen3" || a == "qwen3vl") return load_hparams_qwen3();
    if (a == "deepseek41") return load_hparams_ds41();
    if (a != "qwen35" && a != "qwen35moe" && a != "qwen4exp") throw std::runtime_error("unsupported architecture: " + a);
    hp.moe = a == "qwen35moe" || a == "qwen4exp";
    hp.hc = a == "qwen4exp";
    auto I = [&](const std::string& k, int64_t def = -1) {
      int64_t v = f_.get_int(a + "." + k, def);
      if (v < 0) throw std::runtime_error("missing hparam " + k);
      return (int)v;
    };
    {
      const int blocks = I("block_count"), nextn = (int)f_.get_int(a + ".nextn_predict_layers", 0);
      if (blocks <= 0 || blocks > 4096 || nextn < 0 || nextn >= blocks)
        throw std::runtime_error("invalid block_count / nextn_predict_layers in the GGUF");
      hp.n_layer = blocks - nextn;
    }
    hp.n_embd = I("embedding_length");
    if (hp.n_embd <= 0) throw std::runtime_error("invalid embedding_length in the GGUF");
    hp.n_ff = hp.moe ? 0 : I("feed_forward_length");
    if (hp.moe) {
      hp.n_expert = I("expert_count");
      hp.n_expert_used = I("expert_used_count");
      hp.n_ff_exp = I("expert_feed_forward_length");
      hp.n_ff_shexp = I("expert_shared_feed_forward_length");
      // experts: GSQ 2-bit (tools/convert_gsq.py, zero/kernels/moe.cl) or standard GGUF types (decode.cl moe_*_q)
      // (the key is inferzero.expert_format; a file converted before 2026-11-10 is refused when opened, src/gguf.h)
      hp.moe_std = f_.get_str("inferzero.expert_format", "") != "gsq2-v1";
      hp.n_ff_exp_pad = padded_k(hp.n_ff_exp);
      // (Flash Next: FFN widths of 640, its down projections padded to 768 and its shared expert to 768 rows)
      if (hp.hc) hp.n_ff_shexp = padded_k(hp.n_ff_shexp);
      if ((hp.n_expert != 256 && !(hp.hc && hp.n_expert == 512)) || hp.n_expert_used < 1 || hp.n_expert_used > 16 ||
          (hp.n_ff_exp % 256 && !hp.hc) || hp.n_ff_shexp % 256 || (hp.hc && !hp.moe_std))
        throw std::runtime_error("unsupported MoE dimensions");
    }
    if (hp.hc) {
      hp.hc_n = I("hyper_connection.count");
      hp.hc_lr = I("hyper_connection.low_rank");
      if (const auto* pl = f_.get_ints(a + ".ple.layers"))
        if (!pl->empty()) hp.ple_layer = (int)(*pl)[0];
      if (hp.ple_layer >= 0) {
        hp.ple_dim = I("embedding_length_per_layer_input");
        hp.ple_ng = I("ple.ngram_size");
        hp.ple_per = I("ple.heads_per_ngram");
        hp.ple_taps = I("ple.conv_kernel");
        hp.ple_eos = f_.get_int(a + ".ple.eos_token_id", -1);
        hp.ple_img = (int)f_.get_int(a + ".ple.image_token_id", -1);
        for (auto [key, v] : {std::pair{"ple.layer_multipliers", &hp.ple_mult}, std::pair{"ple.head_offsets", &hp.ple_off},
                              std::pair{"ple.head_vocab_sizes", &hp.ple_voc}})
          if (const auto* x = f_.get_ints(a + "." + key)) v->assign(x->begin(), x->end());
        const int nph = (hp.ple_ng - 1) * hp.ple_per;
        if ((int)hp.ple_mult.size() < hp.ple_ng || (int)hp.ple_off.size() < nph || (int)hp.ple_voc.size() < nph ||
            nph * hp.ple_dim != I("embedding_length") || hp.ple_taps != 4 || (hp.ple_taps - 1) * hp.ple_ng > 16)
          throw std::runtime_error("unsupported PLE n-gram block");
      }
      hp.full_attn_interval = I("full_attention_interval", 4);
      if (const auto* cr = f_.get_ints(a + ".attention.compress_ratios")) {
        const int nl = I("block_count");
        hp.qsa_ratio.assign(nl, 0);
        for (int il = 0; il < nl && il < (int)cr->size(); ++il)
          if (f_.find("blk." + std::to_string(il) + ".indexer.k_proj.weight")) hp.qsa_ratio[il] = (int)(*cr)[il];
        for (int r : hp.qsa_ratio) hp.n_qsa += r > 0;
        if (hp.n_qsa) {
          hp.qsa_heads = I("attention.indexer.head_count");
          hp.qsa_dim = I("attention.indexer.key_length");
          hp.qsa_topk = I("attention.indexer.top_k");
          if (const char* e = getenv("ARC_QSA_TOPK")) hp.qsa_topk = atoi(e);  // (tests: fewer keys, sparse sooner)
          // (the interval indexes the ratios: checked here, before the model's other dimensions are)
          if (hp.full_attn_interval < 1 || hp.full_attn_interval > (int)hp.qsa_ratio.size())
            throw std::runtime_error("invalid full_attention_interval in the GGUF");
          for (int r : hp.qsa_ratio)
            if (r > 0 && (r < 2 || r > 8 || hp.qsa_topk % r || r != hp.qsa_ratio[hp.full_attn_interval - 1]))
              throw std::runtime_error("unsupported QSA block size");
          if (hp.qsa_dim != 128 || hp.qsa_heads < 1 || hp.qsa_heads > 16 || hp.ple_layer < 0)
            throw std::runtime_error("unsupported QSA indexer");
          // Only precise mode enables the new float QSA path. Keep the historical
          // normal-mode --attn-f32 rejection until that configuration is validated.
          if (!opt.attn_int8 && !fn_precise())
            throw std::runtime_error("f32 QSA attention requires Flash Next precise mode");
        }
      }
    }
    hp.n_head = I("attention.head_count");
    hp.n_head_kv = I("attention.head_count_kv");
    hp.head_dim = I("attention.key_length");
    hp.n_rot = I("rope.dimension_count");
    hp.ssm_d_state = I("ssm.state_size");
    hp.ssm_nk = I("ssm.group_count");
    hp.ssm_nv = I("ssm.time_step_rank");
    hp.ssm_d_inner = I("ssm.inner_size");
    hp.conv_k = I("ssm.conv_kernel");
    hp.full_attn_interval = I("full_attention_interval", 4);
    hp.eps = (float)f_.get_float(a + ".attention.layer_norm_rms_epsilon", 1e-6);
    hp.rope_base = (float)f_.get_float(a + ".rope.freq_base", 1e7);
    if (const auto* rs = f_.get_ints(a + ".rope.dimension_sections"))
      for (int i = 0; i < 4 && i < (int)rs->size(); ++i) hp.rope_sect[i] = (int)(*rs)[i];
    if (hp.rope_sect[0] + hp.rope_sect[1] + hp.rope_sect[2] + hp.rope_sect[3] == 0) hp.rope_sect[0] = hp.n_rot / 2;
    tok_embd_ = &f_.get("token_embd.weight");
    // (a broken or hostile file: divisors and shapes checked before anything uses them)
    if (tok_embd_->dims.size() < 2 || tok_embd_->dims[1] <= 0 || hp.n_head <= 0 || hp.n_head_kv <= 0 ||
        hp.full_attn_interval <= 0 || hp.n_layer <= 0)
      throw std::runtime_error("invalid model dimensions in the GGUF");
    if (tok_embd_->dims[0] != (uint64_t)hp.n_embd)  // (embed() reads n_embd values a row: a narrower table is read past)
      throw std::runtime_error("token_embd.weight is not n_embd wide in the GGUF");
    hp.n_vocab = (int)tok_embd_->dims[1];
    // (n_rot exactly 64: attn_prep pairs RoPE elements across lanes 16 apart, right for 64 only)
    if (hp.head_dim != 256 || hp.ssm_d_state != 128 || hp.conv_k != 4 || hp.n_rot != 64 || hp.n_embd % 256 ||
        hp.n_ff % 256 || hp.ssm_d_inner % 256 || hp.n_head % hp.n_head_kv)
      throw std::runtime_error("unsupported Qwen3.5 dimensions");
    // (attn_q8 / attn_split copy each key tile in 2048 / lsz words per work-item, lsz = 16 x sub-groups x rows: a q
    //  head count per kv head that is not a power of two times attn_hs (5, 7, 10, ...) left words of the tile unloaded)
    if (const int nsp = attn_nsp(); nsp & (nsp - 1))
      throw std::runtime_error("unsupported Qwen3.5 attention: " + std::to_string(hp.n_head / hp.n_head_kv) + " q heads per kv head");
    // (hc_down1 / hc_up1: hc_n divides the 8 K slices and fits hc_rinv_'s 8 rows x 8; the inject logits within the
    //  padded low rank, which hc_up1 writes)
    // (hc_n x n_embd = 10240: kernels/ops_flashnext.cl HCK, the size of hc_down1 / hc_up1's local rows; another shape
    //  would write past them, audit 2026-11-14)
    if (hp.hc && (hp.hc_n < 1 || 8 % hp.hc_n || hp.hc_n * hp.n_embd != 10240 || hp.hc_lr <= 0 || hp.hc_lr + hp.hc_n > 512 ||
                  hp.hc_lr + hp.hc_n > padded_k(hp.hc_lr) ||
                  (hp.ple_layer >= 0 && !hp.recurrent(hp.ple_layer))))
      throw std::runtime_error("unsupported hyper-connection shape");
  }

  // Qwen3 dense. qwen3vl (Qwen3-VL / Qwen3-ASR) has M-RoPE sections, but its text and audio rows take one position
  // for all three components, which is plain NEOX RoPE over the head: no image rows go to this path.
  void load_hparams_qwen3() {
    const std::string a = hp.arch;
    auto I = [&](const std::string& k, int64_t def = -1) {
      int64_t v = f_.get_int(a + "." + k, def);
      if (v < 0) throw std::runtime_error("missing hparam " + k);
      return (int)v;
    };
    hp.qwen3 = true;
    kf16_ = !(getenv("ARC_Q3_KF32") && atoi(getenv("ARC_Q3_KF32")) != 0);
    hp.n_layer = I("block_count");
    if (hp.n_layer <= 0 || hp.n_layer > 4096) throw std::runtime_error("invalid block_count in the GGUF");
    hp.n_embd = I("embedding_length");
    hp.n_ff = I("feed_forward_length");
    hp.n_head = I("attention.head_count");
    hp.n_head_kv = I("attention.head_count_kv");
    hp.head_dim = I("attention.key_length");
    hp.n_rot = I("rope.dimension_count", hp.head_dim);
    hp.full_attn_interval = 1;  // (recurrent() is false for every layer)
    hp.eps = (float)f_.get_float(a + ".attention.layer_norm_rms_epsilon", 1e-6);
    hp.rope_base = (float)f_.get_float(a + ".rope.freq_base", 1e6);
    tok_embd_ = &f_.get("token_embd.weight");
    if (tok_embd_->dims.size() < 2 || tok_embd_->dims[1] <= 0 || hp.n_head <= 0 || hp.n_head_kv <= 0)
      throw std::runtime_error("invalid model dimensions in the GGUF");
    if (tok_embd_->dims[0] != (uint64_t)hp.n_embd)  // (embed() reads n_embd values a row: a narrower table is read past)
      throw std::runtime_error("token_embd.weight is not n_embd wide in the GGUF");
    hp.n_vocab = (int)tok_embd_->dims[1];
    if (hp.head_dim != 128 || hp.n_rot != hp.head_dim || I("attention.value_length", 128) != 128 || hp.n_embd % 256 ||
        hp.n_ff % 256 || hp.n_head % hp.n_head_kv || (hp.n_head * hp.head_dim) % 256 ||
        f_.get_int(a + ".n_deepstack_layers", 0) > 0)
      throw std::runtime_error("unsupported Qwen3 dimensions");
    if (!opt.attn_int8) throw std::runtime_error("f32 attention (--attn-f32) does not support Qwen3 dense models");
  }

  void load_hparams_gemma4() {
    const std::string a = "gemma4";
    auto I = [&](const std::string& k, int64_t def = -1) {
      int64_t v = f_.get_int(a + "." + k, def);
      if (v < 0) throw std::runtime_error("missing hparam " + k);
      return (int)v;
    };
    hp.gemma = true;
    hp.swa_rings = true;
    // the float attention's K rows in f16 instead of f32 (as Qwen3 dense has them; ARC_G4_KF16=0: f32 as before
    // 2026-11-11): half the K memory, so more context. The same speed on the laptop (its prompt attention is bound by
    // instructions, not bytes); KL against f32 as before: 20 tokens 12B 0.00000, E4B 0.00395 (0.00389), 26B 0.01013
    // (0.01014); after an 8,839-token prompt the 12B's next 33 tokens KL 0.00007 against f32 K, every top token equal
    kf16_ = !(getenv("ARC_G4_KF16") && atoi(getenv("ARC_G4_KF16")) == 0);
    hp.n_layer = I("block_count");
    if (hp.n_layer <= 0 || hp.n_layer > 4096) throw std::runtime_error("invalid block_count in the GGUF");
    hp.n_embd = I("embedding_length");
    hp.n_ff = I("feed_forward_length");
    hp.n_head = I("attention.head_count");
    hp.hd_full = I("attention.key_length");
    hp.hd_swa = I("attention.key_length_swa", hp.hd_full);
    hp.head_dim = std::max(hp.hd_full, hp.hd_swa);
    hp.n_swa = I("attention.sliding_window", 0);
    hp.eps = (float)f_.get_float(a + ".attention.layer_norm_rms_epsilon", 1e-6);
    hp.rope_base = (float)f_.get_float(a + ".rope.freq_base", 1e6);
    hp.rope_base_swa = (float)f_.get_float(a + ".rope.freq_base_swa", 1e4);
    hp.softcap = (float)f_.get_float(a + ".final_logit_softcapping", 0.0);
    hp.n_ple = (int)f_.get_int(a + ".embedding_length_per_layer_input", 0);
    hp.n_layer_kv = hp.n_layer - (int)f_.get_int(a + ".attention.shared_kv_layers", 0);
    if (hp.n_ple != 0 && hp.n_ple != 256) throw std::runtime_error("unsupported Gemma 4 per-layer embedding size");
    if (hp.n_layer_kv < 2) throw std::runtime_error("unsupported Gemma 4 KV sharing");
    if (hp.n_layer <= 0) throw std::runtime_error("invalid Gemma 4 shape in the GGUF");
    hp.swa.assign(hp.n_layer, 0);
    hp.kv_heads.assign(hp.n_layer, 0);
    const auto* pat = f_.get_ints(a + ".attention.sliding_window_pattern");
    const auto* kvh = f_.get_ints(a + ".attention.head_count_kv");
    for (int il = 0; il < hp.n_layer; ++il) {
      hp.swa[il] = pat && il < (int)pat->size() ? (int)(*pat)[il] : 0;
      hp.kv_heads[il] = kvh && il < (int)kvh->size() ? (int)(*kvh)[il] : I("attention.head_count_kv");
    }
    hp.n_head_kv = *std::max_element(hp.kv_heads.begin(), hp.kv_heads.end());
    if (hp.n_embd <= 0 || hp.n_head <= 0 || hp.n_head_kv <= 0 || hp.n_layer_kv > hp.n_layer)
      throw std::runtime_error("invalid Gemma 4 shape in the GGUF");
    tok_embd_ = &f_.get("token_embd.weight");
    if (tok_embd_->dims.size() < 2 || tok_embd_->dims[1] <= 0) throw std::runtime_error("invalid Gemma 4 shape in the GGUF");
    if (tok_embd_->dims[0] != (uint64_t)hp.n_embd)  // (embed() reads n_embd values a row: a narrower table is read past)
      throw std::runtime_error("token_embd.weight is not n_embd wide in the GGUF");
    hp.n_vocab = (int)tok_embd_->dims[1];
    for (int il = 0; il < hp.n_layer; ++il) {
      const int hd = hp.swa[il] ? hp.hd_swa : hp.hd_full;
      if ((hd != 256 && hd != 512) || hp.kv_heads[il] <= 0 || hp.n_head <= 0 || hp.n_head % hp.kv_heads[il])
        throw std::runtime_error("unsupported Gemma 4 attention shape");
    }
    // (26B-A4B: the dense FFN is 2112 wide; its gate / up get zero rows and its down rows zero columns up to whole
    //  superblocks, 2304: the extra values are gelu(0) * 0 = 0, the results the same)
    hp.n_ff = padded_k(hp.n_ff);
    if (hp.n_embd % 256) throw std::runtime_error("unsupported Gemma 4 dimensions");
    // Gemma 4 26B-A4B: every layer has the dense FFN and routed experts next to it (expert gate and up in one tensor,
    // GELU, a router of its own input norm, a scale per expert)
    hp.n_expert = (int)f_.get_int(a + ".expert_count", 0);
    if (hp.n_expert > 0) {
      hp.moe = hp.moe_std = true;
      hp.n_expert_used = I("expert_used_count");
      hp.n_ff_exp = I("expert_feed_forward_length");
      hp.n_ff_exp_pad = padded_k(hp.n_ff_exp);
      hp.n_ff_shexp = 0;
      if (hp.n_expert % 16 || hp.n_expert_used < 1 || hp.n_expert_used > std::min(16, hp.n_expert) || hp.n_ff_exp % 32 || hp.n_ple)
        throw std::runtime_error("unsupported Gemma 4 expert shape");
    }
  }

  // q heads per attention sub-group: the largest of 4,3,2,1 dividing the GQA ratio.
  // int8 attention tiling: each sub-group handles hs of the q heads sharing a kv head (nsp = hpg / hs sub-groups
  // per token, a power of two), and a work-group covers tpw tokens with 512 work-items.
  // Qwen prompt attention (attn_q8, 256-dim heads): 4 q heads per sub-group where the GQA ratio allows, despite a few
  // spilled registers: 16384-token prompt (bench/attn_bench), 8 q heads per kv head (Qwen3.6 35B) 544 -> 479 ms per
  // layer, 4 per kv head (Qwen3.5 9B) 531 -> 468 ms; 6 per kv head (27B) stays at 3 (6 per sub-group spills: slower)
  int attn_hs() const {
    const int hpg = hp.n_head / hp.n_head_kv;
    if (!hp.gemma && hpg % 4 == 0 && hpg / 4 <= 32 && ((hpg / 4) & (hpg / 4 - 1)) == 0) return 4;
    return attn_hs_for(hpg);
  }
  static int attn_hs_for(int hpg) {
    for (int h : {3, 2, 1})
      if (hpg % h == 0 && ((hpg / h) & (hpg / h - 1)) == 0 && hpg / h <= 32) return h;
    return 1;
  }
  int attn_nsp() const { return hp.n_head / hp.n_head_kv / attn_hs(); }
  // Qwen prompt attention with the head dims split over sub-groups (attn_ps) for chunks ending past `from`: dim
  // groups, q heads and tokens per sub-group, tokens per work-group. pns = 0: attn_q8 everywhere. ARC_QWEN_PS =
  // "pns,phg,ptt,tpw[,from]" overrides (bench/attn_bench ATT_PS / ATT_TPW for the search).
  struct PsCfg { int pns = 0, phg = 0, ptt = 0, tpw = 0, from = 0; };
  PsCfg qwen_ps(int hd, int hpg) const {
    PsCfg c;
    if (hp.gemma || hd != 256) return c;
    if (const char* e = getenv("ARC_QWEN_PS")) {
      // (all four fields given and positive, else off: a missing or 0 phg / ptt divided by zero, BUGS.md 8)
      if (sscanf(e, "%d,%d,%d,%d,%d", &c.pns, &c.phg, &c.ptt, &c.tpw, &c.from) < 4 || c.phg <= 0 || c.ptt <= 0 || c.tpw <= 0 ||
          (c.pns && (hpg % c.phg || c.tpw % c.ptt)))
        c.pns = 0;
      return c;
    }
    (void)hpg;
    return c;  // (defaults per GQA ratio: set from the measurements)
  }
  int attn_tpw() const { return 32 / attn_nsp(); }
  // int8 attention scale blocks per row: Gemma 4 one per 64 dims (its V rows have a few large dims, which a single
  // row scale lets swamp the others: measured against an f32 reference), Qwen one per row (as accurate there, faster)
  std::string sb_opt(int hd) const { return " -DSB=" + std::to_string(hp.swa_rings ? hd / 64 : 1); }

  int gq() const { return gq_div(hp.n_head / hp.n_head_kv, 4); }
  // the largest number of q heads <= cap that divides hpg: a sub-group of the attention kernels holds GQ consecutive
  // q heads and reads the kv head of the first (hpg 5 with GQ 4 gave heads 5-7 kv head 0)
  static int gq_div(int hpg, int cap) {
    for (int g = std::min(hpg, cap); g > 1; --g)
      if (hpg % g == 0) return g;
    return 1;
  }

  // Decides whether decoded W8 weights stay resident.
  void plan_weights() {
    if (hp.moe) {  // Q8_0 matrices (repacked at load) and 2-bit experts: nothing is decoded just in time
      w8_resident = true;
      // streamed experts that cannot all be cached (more than the GPU memory share; Qwen3.8 Flash Next: 52 GB): the
      // dense weights' W8 copies (~3 GB) go to the expert cache instead, decoded just in time per prompt segment
      uint64_t ex = 0;
      for (auto& t : f_.tensors)
        if (t.name.find("_exps.weight") != std::string::npos) ex += t.nbytes;
      static const bool keep = getenv("ARC_W8_RESIDENT") != nullptr;  // (A/B)
      if (hp.moe_std && opt.stream_experts && !keep && zrt::D().global_mem && ex > zrt::D().global_mem) w8_resident = false;
      return;
    }
    size_t raw = 0, w8 = 0;
    double raw_s = 0, w8_s = 0;  // (a split model: this shard's part, tp_weight_share in zero/mgpu/engine_tp_shard.h)
    for (auto& t : f_.tensors) {
      int il = -1;
      if (t.dims.size() != 2 || sscanf(t.name.c_str(), "blk.%d.", &il) != 1 || il >= hp.n_layer) continue;
      raw += t.nbytes;
      w8 += (size_t)t.dims[0] * t.dims[1] * 41 / 40;
      if (tp_on()) {
        const double sh = tp_weight_share(t.name);
        raw_s += (double)t.nbytes * sh;
        w8_s += (double)((size_t)t.dims[0] * t.dims[1] * 41 / 40) * sh;
      }
    }
    // (ARC_TP_SIM: every shard on the one GPU, so the budget is theirs together, as device_mem is divided there)
    const double budget =
        opt.w8_budget_gb / (tp_on() && getenv("ARC_TP_SIM") && atoi(getenv("ARC_TP_SIM")) != 0 ? opt.tp_size : 1);
    w8_resident = (tp_on() ? raw_s + w8_s : (double)(raw + w8)) / 1073741824.0 <= budget;
  }

  cl_kernel kernel(const std::string& which, const std::string& defs, const char* name) {
    const std::string key = which + defs + "|" + name;
    auto it = kcache_.find(key);
    if (it != kcache_.end()) return it->second;
    if (which == "asr" && asr_src_.empty()) {  // (the Qwen3-ASR encoder's kernels, zero/kernels/asr.cl: read when first used)
      asr_src_ = read_file(opt.kernel_dir + "/asr.cl");
      if (asr_src_.empty()) throw std::runtime_error("kernel file not found: " + opt.kernel_dir + "/asr.cl");
    }
    if (which == "x16" && x16_src_.empty()) {  // (the f16 XMX prompt GEMM of discrete GPUs, zero/xe2/engine_xe2.h)
      x16_src_ = read_file(opt.kernel_dir + "/gemm_x16.cl");
      if (x16_src_.empty()) throw std::runtime_error("kernel file not found: " + opt.kernel_dir + "/gemm_x16.cl");
    }
    if (which == "xw" && xw_src_.empty()) {  // (the H8R GEMM on XMX with wider tiles, zero/xe2/engine_xw.h)
      xw_src_ = read_file(opt.kernel_dir + "/gemm_xw.cl");
      if (xw_src_.empty()) throw std::runtime_error("kernel file not found: " + opt.kernel_dir + "/gemm_xw.cl");
    }
    if (which == "xw8" && xw8_src_.empty()) {  // (its 8-lane form, the 140T: zero/xe2/engine_xw.h)
      xw8_src_ = read_file(opt.kernel_dir + "/gemm_xw8.cl");
      if (xw8_src_.empty()) throw std::runtime_error("kernel file not found: " + opt.kernel_dir + "/gemm_xw8.cl");
    }
    if (which == "x8" && x8_src_.empty()) {  // (the XMX prompt GEMM of GPUs with 8-lane sub-groups, gemm_kernel)
      x8_src_ = read_file(opt.kernel_dir + "/gemm_x8.cl");
      if (x8_src_.empty()) throw std::runtime_error("kernel file not found: " + opt.kernel_dir + "/gemm_x8.cl");
    }
    if (which == "h8" && h8_src_.empty()) {  // (the H8 rotations, zero/engine_h8.h)
      h8_src_ = read_file(opt.kernel_dir + "/h8.cl");
      if (h8_src_.empty()) throw std::runtime_error("kernel file not found: " + opt.kernel_dir + "/h8.cl");
    }
    if (which == "gdnx" && gdnx_src_.empty()) {  // (the chunked DeltaNet on XMX, zero/xe2/engine_xe2.h)
      gdnx_src_ = read_file(opt.kernel_dir + "/gdn_x.cl");
      if (gdnx_src_.empty()) throw std::runtime_error("kernel file not found: " + opt.kernel_dir + "/gdn_x.cl");
    }
    if (which == "attnx" && attnx_src_.empty()) {  // (attn_q8 on XMX: attn_x.cl in front of the ops parts, engine_xe2.h)
      const std::string ax = read_file(opt.kernel_dir + "/attn_x.cl");
      if (ax.empty()) throw std::runtime_error("kernel file not found: " + opt.kernel_dir + "/attn_x.cl");
      attnx_src_ = ax + "\n" + ops_.for_kernel("attn_q8");
    }
    const std::string& src = which == "gemm" ? gemm_src_ : which == "decode" ? dec_.for_kernel(name) : which == "moe" ? moe_src_
                             : which == "asr" ? asr_src_ : which == "x16" ? x16_src_ : which == "h8" ? h8_src_ : which == "x8" ? x8_src_
                             : which == "gdnx" ? gdnx_src_ : which == "attnx" ? attnx_src_ : which == "xw" ? xw_src_ : which == "xw8" ? xw8_src_
                             : ops_.for_kernel(name);
    auto p = cl_.build(src, "-cl-std=CL3.0 -cl-mad-enable " + defs + (which == "ops" || which == "attnx" ? q2_defs_ : std::string()) +
                            (cl_.sg8 ? "" : " -DNO_SG8"));
    progs_.push_back(p);
    return kcache_[key] = cl_.kernel(p, name);
  }
  // (experiments: ARC_GEMM_DEFS adds build options, ARC_GEMM_TM sets the tokens per sub-group)
  static const std::string& gemm_extra() {
    static const std::string e = getenv("ARC_GEMM_DEFS") ? std::string(" ") + getenv("ARC_GEMM_DEFS") : std::string();
    return e;
  }
  static int gemm_tm() {  // (16 or 32: gemm.cl keeps TM / 16 activation vectors per lane)
    static const int tm = [] {
      const int t = getenv("ARC_GEMM_TM") ? atoi(getenv("ARC_GEMM_TM")) : 16;
      if (t == 16 || t == 32) return t;
      zlog::eprintf("inferzero: ARC_GEMM_TM=%d not supported (16 or 32): 16\n", t);
      return 16;
    }();
    return tm;
  }
  static const char* xmx_defs() {
    static const int t = getenv("ARC_XMX_TRY") ? atoi(getenv("ARC_XMX_TRY")) : 0;
    return t == 2 ? " -DTP=8" : t == 3 ? " -DTP=8 -DDBG_NOBCAST" : " -DXMX -DTP=8";
  }
  // (xe2_gemm_grf / moe_grf / cg: zero/xe2/engine_xe2.h, the defaults on GPUs without XMX)
  // tm: tokens per sub-group (0: gemm_tm()); g128: 1 the default register file, 0 256 registers, -1 xe2_gemm_grf()
  // (the 140T preset, zero/xe2/engine_xe2.h p140_, picks both per GEMM)
  cl_kernel gemm_kernel(int qtype, bool act32, int xmx = -1, int cg = 1, int tm = 0, int g128 = -1) {
    const int tmv = tm ? tm : gemm_tm();
    const std::string grf = g128 < 0 ? xe2_gemm_grf() : g128 ? std::string() : std::string("-cl-intel-256-GRF-per-thread ");
    // (a GPU with 8-lane sub-groups, Arrow Lake H: its DPAS needs a kernel of its own, for the W8 and H8 layouts; the
    //  native types stay on DP4a there. ARC_XMX_TRY=2 / 3: the same kernel with DP4a for DPAS, right and wrong)
    if (cl_.sg8) {
      if ((xmx < 0 ? xmx_ : xmx != 0) && qtype >= 100) {
        static const int t = getenv("ARC_XMX_TRY") ? atoi(getenv("ARC_XMX_TRY")) : 0;
        return kernel("x8", grf + "-DQTYPE=" + std::to_string(qtype) + (act32 ? " -DACT32" : "") +
                                (tmv != 16 ? " -DTM=" + std::to_string(tmv) : std::string()) + xe2_cg_defs(cg) +
                                x8_defs() + (t == 2 ? " -DX8_EMU" : t == 3 ? " -DX8_EMU -DDBG_NOBCAST" : ""),
                      "gemm_x8");
      }
      xmx = 0;
    }
    return kernel("gemm", grf + "-DQTYPE=" + std::to_string(qtype) + (act32 ? " -DACT32" : "") +
                              (tmv != 16 ? " -DTM=" + std::to_string(tmv) : std::string()) +
                              xe2_cg_defs(cg) + gemm_extra() +
                              ((xmx < 0 ? xmx_ : xmx != 0) ? xmx_defs() + xe2_x2d() + xe2_wpf() : std::string()),
                  "gemm");
  }
  // the mixture-of-experts prompt GEMM over slot tiles (gemm.cl MOE_TILES; W8 layouts 100..103, per-32 scales)
  cl_kernel moe_gemm_kernel(int qtype, int xmx = -1) {
    return kernel("gemm", xe2_moe_grf() + "-DACT32 -DMOE_TILES -DQTYPE=" + std::to_string(qtype) +
                              ((xmx < 0 ? xmx_ : xmx != 0) && !cl_.sg8 ? xmx_defs() : ""),  // (8-lane GPUs: DP4a, gemm_x8 has no tiles)
                  "gemm");
  }
  // XMX for the prompt GEMMs (gemm.cl -DXMX): taken only when the GPU offers 16-lane DPAS (ClRt::xmx) and the XMX
  // kernels give the DP4a kernels' results bit for bit on random data: every weight layout with and without per-32
  // activation scales, as the engine launches them (store; split-K pieces accumulating; the fused SwiGLU; several
  // work-group shapes and row groups) and the mixture-of-experts tile kernel (two experts, partial tiles, a row map
  // and routing weights). A GPU or driver that differs, or a kernel that does not build, keeps DP4a. Written without
  // an XMX GPU at hand (this laptop's Xe-LPG has none): the check is what guards it on the GPUs that have one.
  // (ARC_XMX_TRY=1: the check on any GPU; =2: a DP4a variant in the XMX slot, which must pass; =3: a wrong one, which
  //  must fail)
  bool xmx_ = false;
  bool xmx_slow_ = false;  // (the check failed on speed, the results being right: no retry without 2D loads)
  // mode 0: the plain XMX kernels and the MoE tiles; mode 1: the H8 kernels this engine runs (its column groups) and the
  // 140T preset's, checked apart and kept per kind by the caller (zero/engine.h): a failing H8 kernel then turns H8
  // off, not XMX (self audit 2026-11-08)
  bool xmx_check(int mode = 0) {
    // (M: 4 tiles of tm tokens, so every work-group shape below covers at least one; 64 with the default tm 16)
    // (N: 4 column groups, so ARC_GEMM_CG up to 4 is checked too)
    const int M = 4 * gemm_tm(), N = 64, K = 1024, nsb = K / 256, ng = 2 * N / 16;  // (weights for 2 * N rows: two experts)
    uint32_t seed = 12345;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    auto rndf = [&](float lo, float span) { return lo + (float)(rnd() % 10000) * (span / 10000.0f); };
    std::vector<uint32_t> wq((size_t)ng * nsb * 4 * 16 * 16), wh((size_t)ng * nsb * 10 * 16);
    for (auto& v : wq) v = rnd();
    for (auto& v : wh) {  // (floats in [0.001, 0.011): headers and scales of every layout read sensible values)
      const float f = rndf(0.001f, 0.01f);
      memcpy(&v, &f, 4);
    }
    std::vector<int8_t> aq((size_t)M * K);
    for (auto& v : aq) v = (int8_t)(rnd() >> 24);
    std::vector<float> asc((size_t)(K / 32) * M), c0((size_t)M * N);
    for (auto& v : asc) v = rndf(0.01f, 0.01f);
    for (auto& v : c0) v = rndf(-2.0f, 4.0f);  // (C before an accumulating launch: the gate of the SwiGLU)
    std::vector<uint16_t> ab((size_t)(K / 16) * M);
    for (auto& v : ab) v = (uint16_t)(int16_t)((int)(rnd() % 4001) - 2000);
    // MoE tiles: tiles[0] = count, then (expert, first slot, rows); expert 0: slots 0-15, 16-20; expert 1: 21-36, 37-39
    const std::vector<int> tiles{4, 0, 0, 16, 0, 16, 5, 1, 21, 16, 1, 37, 3}, tstart{0, 2, 4};
    std::vector<int> rowmap(M);
    std::vector<float> rowscale(M);
    for (auto& v : rowmap) v = (int)(rnd() % M);
    for (auto& v : rowscale) v = rndf(0.05f, 0.9f);
    cl_mem bq = buf(wq.size() * 4, wq.data()), bh = buf(wh.size() * 4, wh.data()), ba = buf(aq.size(), aq.data()),
           bs = buf(asc.size() * 4, asc.data()), bb = buf(ab.size() * 2, ab.data()), bc = buf((size_t)M * N * 4),
           bt = buf(tiles.size() * 4, tiles.data()), bts = buf(tstart.size() * 4, tstart.data()),
           bm = buf(rowmap.size() * 4, rowmap.data()), bw = buf(rowscale.size() * 4, rowscale.data());
    std::string bad;
    // one case: C from c0, then launch(kernel) with the DP4a and the XMX kernel; their C must be equal bit for bit
    // (and the DP4a result finite and not all zero, or the comparison proves nothing)
    auto check = [&](const std::string& name, const std::function<cl_kernel(int)>& make,
                     const std::function<void(cl_kernel)>& launch) {
      std::vector<float> out[2];
      for (int x = 0; x < 2; ++x) {
        cl_kernel k = make(x);
        CL_CHECK(clEnqueueWriteBuffer(cl_.queue, bc, CL_TRUE, 0, c0.size() * 4, c0.data(), 0, nullptr, nullptr));
        launch(k);
        out[x].resize(c0.size());
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, bc, CL_TRUE, 0, out[x].size() * 4, out[x].data(), 0, nullptr, nullptr));
      }
      bool sane = false;
      for (size_t i = 0; i < out[0].size(); ++i) {
        if (!std::isfinite(out[0][i])) { sane = false; break; }
        sane = sane || out[0][i] != c0[i];
      }
      if (!sane || memcmp(out[0].data(), out[1].data(), out[0].size() * 4)) bad += " " + name;
    };
    struct Soft { bool& f; ~Soft() { f = false; } } soft{cl_.soft_fail};
    cl_.soft_fail = true;
    try {
      for (int q : {101, 100, 103, 102, 8, 12, 13, 14, 110, 111})  // (110, 111: H8, zero/engine_h8.h, never per-32 activations)
        for (int a32 = q >= 110 ? 0 : 1; a32 >= 0; --a32) {
          // (only when asked for: an H8 kernel that fails its check must not take XMX away from everything else)
          if (mode == 0 ? q >= 110 : (q < 110 || !(h8_ && (!p140_ || h8_env()) && q == h8_qtype()))) continue;  // (H8: mode 1 only)
          if (cl_.sg8 && q < 100) continue;  // (the native types stay on DP4a on 8-lane GPUs: nothing to compare)
          const std::string nm = std::to_string(q) + (a32 ? "/act32" : "");
          const int cg = q >= 110 ? h8_cg_for(N) : xe2_cg_for(N);  // (H8: its default column groups, 4 on discrete cards)
          auto make = [&](int x) { return gemm_kernel(q, a32 != 0, x, cg); };
          // (wgy rows of tm tokens per work-group, rg row groups; split: superblocks per piece, 0 = one piece)
          auto dense = [&](int acc, int wgy, int rg, int split) {
            return [&, acc, wgy, rg, split](cl_kernel k) {
              int MM = M, NN = N, KK = K, ldc = N, coff = 0, rgg = rg;
              const int pieces = split ? (nsb + split - 1) / split : 1;
              size_t gws[2] = {(size_t)N * (M / gemm_tm() / wgy), (size_t)wgy}, lws[2] = {16 * (size_t)cg, (size_t)wgy};
              for (int sp = 0; sp < pieces; ++sp) {
                int s0 = nsb * sp / pieces, s1 = nsb * (sp + 1) / pieces, ac = sp == 0 ? acc : 1;
                set_args(k, 0, bq, bh, ba, bs, bb, bc, MM, NN, KK, ldc, coff, ac, rgg, s0, s1);
                run("xmx_check", k, 2, gws, lws);
              }
            };
          };
          check(nm + "/store", make, dense(0, 2, 4, 0));
          check(nm + "/split", make, dense(0, 4, 1, 1));
          check(nm + "/add", make, dense(1, 1, 2, 2));
          check(nm + "/swiglu", make, dense(2, 2, 4, 0));
        }
      for (int q : {100, 101})  // (the tile kernel has no layouts with mins)
        for (int acc : {0, 2}) {
          if (cl_.sg8 || mode != 0) continue;  // (DP4a on 8-lane GPUs: moe_gemm_kernel)
          auto make = [&](int x) { return moe_gemm_kernel(q, x); };
          check(std::to_string(q) + "/moe" + (acc ? "/swiglu" : ""), make, [&](cl_kernel k) {
            int MM = M, NN = N, KK = K, ldc = N, ac = acc, e0 = 0, e1 = 2, um = acc ? 1 : 0, us = acc ? 0 : 1;
            set_args(k, 0, bq, bh, ba, bs, bb, bc, MM, NN, KK, ldc, ac, bt, bts, e0, e1, bm, um, bw, us);
            size_t gws[2] = {(size_t)N, 3}, lws[2] = {16, 1};  // (3 tile groups for 4 tiles: one loops)
            run("xmx_check", k, 2, gws, lws);
          });
          // (ARC_MOE_CG / ARC_MOE_WY shapes: 2 column groups x 2 tiles per work-group, one tile left to loop)
          check(std::to_string(q) + "/moe2x2" + (acc ? "/swiglu" : ""), make, [&](cl_kernel k) {
            int MM = M, NN = N, KK = K, ldc = N, ac = acc, e0 = 0, e1 = 2, um = acc ? 1 : 0, us = acc ? 0 : 1;
            set_args(k, 0, bq, bh, ba, bs, bb, bc, MM, NN, KK, ldc, ac, bt, bts, e0, e1, bm, um, bw, us);
            size_t gws[2] = {(size_t)N, 2}, lws[2] = {32, 2};
            run("xmx_check", k, 2, gws, lws);
          });
        }
      // (the Arc 140T preset, zero/xe2/engine_xe2.h p140_: its own kernels, XMX at 32 tokens, 2 column groups and 128
      //  registers, against the DP4a default (16 tokens, 256 registers) bit for bit: the W8 layouts with and without
      //  per-32 activation scales and H8R, stored, in two split-K pieces and with the fused SwiGLU)
      if (mode == 1 && p140_ && bad.empty()) {
        const int wgy = 2;
        for (int q : {101, 100, 103, 102, 111})
          for (int a32 = q >= 110 ? 0 : 1; a32 >= 0; --a32)
            for (int mode2 = 0; mode2 < 3; ++mode2) {  // (store; split-K: a second piece accumulating; the fused SwiGLU)
              const int acc = mode2 == 2 ? 2 : 0, pieces = mode2 == 1 ? 2 : 1;
              std::vector<float> out[2];
              for (int x = 0; x < 2; ++x) {
                const int tmx = x ? 32 : 16, cgx = x ? 2 : 1;
                cl_kernel k = x ? gemm_kernel(q, a32 != 0, 1, 2, 32, 1) : gemm_kernel(q, a32 != 0, 0, 1, 16, 0);
                CL_CHECK(clEnqueueWriteBuffer(cl_.queue, bc, CL_TRUE, 0, c0.size() * 4, c0.data(), 0, nullptr, nullptr));
                size_t gws[2] = {(size_t)N * (M / tmx / wgy), (size_t)wgy}, lws[2] = {16 * (size_t)cgx, (size_t)wgy};
                for (int sp = 0; sp < pieces; ++sp) {
                  int MM = M, NN = N, KK = K, ldc = N, coff = 0, ac = sp == 0 ? acc : 1, rg = 4, s0 = nsb * sp / pieces,
                      s1 = nsb * (sp + 1) / pieces;
                  set_args(k, 0, bq, bh, ba, bs, bb, bc, MM, NN, KK, ldc, coff, ac, rg, s0, s1);
                  run("xmx_check", k, 2, gws, lws);
                }
                out[x].resize(c0.size());
                CL_CHECK(clEnqueueReadBuffer(cl_.queue, bc, CL_TRUE, 0, out[x].size() * 4, out[x].data(), 0, nullptr, nullptr));
              }
              bool sane = false;
              for (size_t i = 0; i < out[0].size(); ++i) {
                if (!std::isfinite(out[0][i])) { sane = false; break; }
                sane = sane || out[0][i] != c0[i];
              }
              if (!sane || memcmp(out[0].data(), out[1].data(), out[0].size() * 4))
                bad += " 140T/" + std::to_string(q) + (a32 ? "/act32" : "") + (mode2 == 1 ? "/split" : mode2 == 2 ? "/swiglu" : "");
            }
      }
      // (a GPU with 8-lane sub-groups, Arrow Lake H, comes here only with ARC_XMX_FORCE=1 (zero/engine.h). The speed
      //  check that stood here timed a 64 x 64 x 1024 GEMM: on a 140T it found XMX 17 % faster while the 27B's real
      //  GEMMs were 1.6x slower, so it was removed, 2026-11-07)
      // iGPUs with XMX (Lunar / Panther Lake): their memory is shared and slower than a discrete card's, so XMX may not
      // pay there either. A GEMM of a model's size (512 tokens x 17408 x 5120, the 27B's gate: 89 MB of weights, past
      // the caches), as the engine launches it (2D loads too where on), best of 3 each; XMX only if 3 % faster.
      // ARC_XMX_FORCE=1: no check; ARC_XMX_SPEED=1: the check also under ARC_XMX_TRY (tests) and on discrete cards
      const bool spd_env = getenv("ARC_XMX_SPEED") && atoi(getenv("ARC_XMX_SPEED")) != 0;
      const bool force = getenv("ARC_XMX_FORCE") && atoi(getenv("ARC_XMX_FORCE")) != 0;
      // (also discrete cards with 8-lane DPAS, the Alchemist A-series: gemm_x8 lost 1.6x on a 140T and is unmeasured there)
      if (mode == 0 && bad.empty() && !force && !p140_ && (spd_env || ((zrt::integrated() || cl_.sg8) && !getenv("ARC_XMX_TRY")))) {
        const int SM = 512, SN = 17408, SK = 5120, sng = SN / 16, snsb = SK / 256;
        std::vector<uint32_t> sw((size_t)sng * snsb * 4 * 16 * 16), sh((size_t)sng * snsb * 10 * 16);
        for (auto& v : sw) v = rnd();
        for (auto& v : sh) {
          const float f = rndf(0.001f, 0.01f);
          memcpy(&v, &f, 4);
        }
        std::vector<int8_t> sa((size_t)SM * SK);
        for (auto& v : sa) v = (int8_t)(rnd() >> 24);
        std::vector<float> ss((size_t)(SK / 32) * SM);
        for (auto& v : ss) v = rndf(0.01f, 0.01f);
        cl_mem q8 = buf(sw.size() * 4, sw.data()), h8 = buf(sh.size() * 4, sh.data()), a8 = buf(sa.size(), sa.data()),
               s8 = buf(ss.size() * 4, ss.data()), b8 = buf((size_t)(SK / 16) * SM * 2), c8 = buf((size_t)SM * SN * 4);
        // (DP4a as it would run without XMX: the GPU's own defaults, not the discrete cards' switches (xe2_), which
        //  are dropped too if XMX loses: zero/engine.h)
        const int wgy = 8;
        int cgs[2];
        cl_kernel kk[2];
        {
          const bool sv = xe2_, sx = x2d_;
          xe2_ = false, x2d_ = false;
          cgs[0] = xe2_cg_for(SN);
          kk[0] = gemm_kernel(101, true, 0, cgs[0]);
          xe2_ = sv, x2d_ = sx;
        }
        cgs[1] = xe2_cg_for(SN);
        kk[1] = gemm_kernel(101, true, 1, cgs[1]);
        int MM = SM, NN = SN, KK = SK, ldc = SN, coff = 0, ac = 0, rg = 4, s0 = 0, s1 = snsb;
        double best[2] = {1e30, 1e30};
        float back = 0;
        for (int r = 0; r < 4; ++r)  // (round 0 warms up: not counted)
          for (int x = 0; x < 2; ++x) {
            size_t gws[2] = {(size_t)SN * (SM / gemm_tm() / wgy), (size_t)wgy}, lws[2] = {16 * (size_t)cgs[x], (size_t)wgy};
            set_args(kk[x], 0, q8, h8, a8, s8, b8, c8, MM, NN, KK, ldc, coff, ac, rg, s0, s1);
            const auto t0 = std::chrono::steady_clock::now();
            run("xmx_check", kk[x], 2, gws, lws);
            CL_CHECK(clEnqueueReadBuffer(cl_.queue, c8, CL_TRUE, 0, 4, &back, 0, nullptr, nullptr));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (r > 0) best[x] = std::min(best[x], ms);
          }
        for (cl_mem* m : {&q8, &h8, &a8, &s8, &b8, &c8}) free_buf(*m);
        const bool faster = best[1] < 0.97 * best[0];
        zlog::eprintf("inferzero: XMX speed check (a 512 x 17408 x 5120 GEMM): DP4a %.1f ms, XMX %.1f ms: %s\n",
                best[0], best[1], faster ? "XMX" : "DP4a");
        if (!faster) {
          xmx_slow_ = true;
          return false;
        }
      }
    } catch (const zrt::GpuWaitError&) {
      throw;  // (a split model's GPU that hangs: not a verdict on XMX, which is kept for the process)
    } catch (const std::exception& e) {
      zlog::eprintf("inferzero: XMX kernels do not build here (%s): DP4a\n", e.what());
      return false;
    }
    if (!bad.empty()) zlog::eprintf("inferzero: %s results differ from DP4a (%s ): %s (please report)\n", mode ? "H8 XMX" : "XMX", bad.c_str(), mode ? "H8 off" : "DP4a");
    return bad.empty();
  }


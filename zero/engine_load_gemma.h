// A part of class Engine (zero/engine.h), included inside the class body at the place it was cut from (the
// class is the same text), not a header of its own: the Gemma 4 weights (load_weights_gemma4).

  void load_weights_gemma4() {
    layers_.resize(hp.n_layer);
    cl_mem rope_ff = f_.find("rope_freqs.weight") ? load_f32("rope_freqs.weight") : nullptr;
    // sliding-window layers keep a ring of the last n_swa + chunk (+ room to rewind an agent's reply) positions. The
    // layers that share another layer's KV (E2B / E4B) read its ring later in the same pass over the same key range,
    // so the ring is enough for them too (they kept full caches before 2026-10-22: E4B 13,921-token prompt in a
    // 16K context, ring vs full, logits bit-identical with and without the drafter; ~115 KB less per token)
    swa_cap_ = std::min(ctx, (hp.n_swa + max_chunk + swa_margin() + 255) / 256 * 256);
    if (hp.n_ple) {
      ple_tok_embd_ = &f_.get("per_layer_token_embd.weight");
      if (ple_tok_embd_->dims.size() < 2 || ple_tok_embd_->dims[0] != (uint64_t)hp.n_ple * hp.n_layer ||
          ple_tok_embd_->dims[1] < (uint64_t)hp.n_vocab)  // (a row: n_ple values for every layer, read for each token)
        throw std::runtime_error("per_layer_token_embd.weight has an unexpected shape in the GGUF");
      ple_model_proj_ = load_w("per_layer_model_proj.weight");
      ple_proj_norm_ = load_f32("per_layer_proj_norm.weight");
    }
    for (int il = 0; il < hp.n_layer; ++il) {
      Layer& L = layers_[il];
      L.il = il;
      const std::string p = "blk." + std::to_string(il) + ".";
      L.swa = hp.swa[il] != 0;
      L.nkv = hp.kv_heads[il];
      L.hd = L.swa ? hp.hd_swa : hp.hd_full;
      L.cap = L.swa ? swa_cap_ : ctx;
      L.attn_norm = load_f32(p + "attn_norm.weight");
      L.post_attn = load_f32(p + "post_attention_norm.weight");
      L.ffn_norm = load_f32(p + "ffn_norm.weight");
      L.post_ffw = load_f32(p + "post_ffw_norm.weight");
      L.wq = load_w(p + "attn_q.weight");
      if (il >= hp.n_layer_kv) L.kv_src = hp.n_layer_kv - (L.swa ? 2 : 1);  // llama.cpp's KV reuse rule
      if (L.kv_src < 0) {
        L.wk = load_w(p + "attn_k.weight");
        L.has_v = f_.find(p + "attn_v.weight") != nullptr;
        if (L.has_v) L.wv = load_w(p + "attn_v.weight");
      }
      L.wo = load_w(p + "attn_output.weight");
      L.q_norm = load_f32(p + "attn_q_norm.weight");
      // (shared-KV layers compute no K: Google's GGUFs omit their attn_k_norm; attn_prep_g then reads only Q heads)
      L.k_norm = L.kv_src >= 0 && !f_.find(p + "attn_k_norm.weight") ? L.q_norm : load_f32(p + "attn_k_norm.weight");
      if (!L.swa) L.rope_ff = rope_ff;
      if (f_.find(p + "layer_output_scale.weight")) {
        const GgufTensor& t = f_.get(p + "layer_output_scale.weight");
        float v = 1.0f;
        dequant_row(t.type, t.data, 1, &v);
        L.out_scale = v;
      }
      if (L.kv_src >= 0) {
        const Layer& S = layers_[L.kv_src];
        if (S.hd != L.hd || S.nkv != L.nkv || S.swa != L.swa) throw std::runtime_error("Gemma 4: incompatible shared KV layer");
        L.kq = S.kq; L.vq = S.vq; L.ksc = S.ksc; L.vsc = S.vsc; L.cap = S.cap;
        L.kc = S.kc; L.vc = S.vc;
        L.i8 = S.i8;
      } else {
        const size_t kvb = (size_t)L.nkv * L.cap * L.hd, scb = (size_t)L.nkv * L.cap * (L.hd / 64) * 4;
        // (the int8 copy only where the assistant drafter reads it: the sources of the last sliding and the last
        //  global layer, and only with the drafter; the two-pass attention reads the float rows. 26B: ~20 % less
        //  per token, the sliding rings ~0.7 GB less)
        L.i8 = !q2_ || (opt.mtp && (il == g4_kv_layer(hp.n_layer - 2) || il == g4_kv_layer(hp.n_layer - 1))) ||
               (g4_att8_ && !L.swa);  // (E2B / E4B: the global layers' prompt attention reads it, see g4_att8_)
        const size_t ib = L.i8 ? kvb : 64, is = L.i8 ? scb : 64;
        L.kq = buf(ib);
        L.vq = buf(ib);
        L.ksc = buf(is);
        L.vsc = buf(is);
        for (cl_mem m : {L.kq, L.vq}) fill(m, ib);
        for (cl_mem m : {L.ksc, L.vsc}) fill(m, is);
        if (q2_) {  // (two-pass mode: f32 K / f16 V rows for attn_fa_g; the int8 cache stays for the drafter)
          L.kc = buf(kvb * 4);
          L.vc = buf(kvb * 2);
          fill(L.kc, kvb * 4);
          fill(L.vc, kvb * 2);
        }
      }
      if (hp.n_ple) {
        L.ple_gate = load_w(p + "inp_gate.weight");
        L.ple_proj = load_w(p + "proj.weight");
        L.ple_post = load_f32(p + "post_norm.weight");
      }
      L.gate = load_w_pad(p + "ffn_gate.weight", hp.n_ff);
      L.up = load_w_pad(p + "ffn_up.weight", hp.n_ff);
      L.down = load_w_pad(p + "ffn_down.weight", hp.n_embd);
      if (hp.moe) load_moe_gemma(L, p);
      clFinish(cl_.queue);
    }
    if (opt.mtp) {
      std::string dpath = opt.mtp_model;
      if (dpath.empty()) dpath = find_gemma_drafter();
      if (!dpath.empty()) {
        const size_t g0 = gpu_bytes;
        load_gemma_drafter(dpath);
        mem_mtp_ += gpu_bytes - g0;
      }
    }
  }

  // An assistant drafter for this model next to it: mtp-*.gguf of arch gemma4-assistant with the same hidden size.
  std::string find_gemma_drafter() const {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::path(path_).parent_path();
    for (const auto& e : fs::directory_iterator(dir.empty() ? fs::path(".") : dir, ec)) {
      const std::string n = e.path().filename().string();
      if (n.rfind("mtp-", 0) != 0 || e.path().extension() != ".gguf") continue;
      try {
        GgufFile g(e.path().string());
        const std::string a = g.get_str("general.architecture");
        if (a == "gemma4-assistant" && g.get_int(a + ".embedding_length_out", -1) == hp.n_embd) return e.path().string();
      } catch (...) {
      }
    }
    return "";
  }

  // GEMV-only matrix (raw rows; single tokens only), requantized to Q8_0 when the GEMV has no kernel for its type
  GpuWeight gemv_weight(const GgufFile& g, const std::string& name) {
    const GgufTensor& t = g.get(name);
    GpuWeight w;
    w.K = (int)t.dims[0];
    w.N = (int)t.dims[1];
    if (t.type == GGML_Q8_0 || is_w8_type(t.type)) {
      w.type = t.type;
      w.raw = upload_padded(t.type, t.data, t.nbytes);
      return w;
    }
    std::vector<uint8_t> q8;
    const size_t rb = t.nbytes / w.N;
    std::vector<float> v;
    for (int r0 = 0; r0 < w.N; r0 += 8192) {  // in row chunks (the drafter's lm head has 262144 rows)
      const int n = std::min(8192, w.N - r0);
      v.resize((size_t)n * w.K);
#pragma omp parallel for
      for (int r = 0; r < n; ++r) dequant_row(t.type, t.data + (size_t)(r0 + r) * rb, w.K, &v[(size_t)r * w.K]);
      auto q = quantize_rows_q8_0(v.data(), n, w.K);
      q8.insert(q8.end(), q.begin(), q.end());
    }
    w.type = GGML_Q8_0;
    w.raw = upload_padded(GGML_Q8_0, q8.data(), q8.size());
    return w;
  }
  cl_mem f32_from(const GgufFile& g, const std::string& name) {
    const GgufTensor& t = g.get(name);
    std::vector<float> v(t.nelements());
    dequant_row(t.type, t.data, (int)v.size(), v.data());
    return buf(v.size() * 4, v.data());
  }

  void load_gemma_drafter(const std::string& path) {
    df_ = std::make_unique<GgufFile>(path);
    const GgufFile& d = *df_;
    const std::string a = d.get_str("general.architecture");
    auto I = [&](const std::string& k) { return (int)d.get_int(a + "." + k, -1); };
    const int nl = I("block_count");
    d_E_ = I("embedding_length");
    d_F_ = I("feed_forward_length");
    d_nh_ = I("attention.head_count");
    d_swa_ = (int)d.get_int(a + ".attention.sliding_window", hp.n_swa);
    d_base_ = (float)d.get_float(a + ".rope.freq_base", 1e6);
    d_base_swa_ = (float)d.get_float(a + ".rope.freq_base_swa", 1e4);
    const auto* pat = d.get_ints(a + ".attention.sliding_window_pattern");
    const int Ls = hp.n_layer - 2, Lg = hp.n_layer - 1;  // main layers whose KV the drafter reads (llama.cpp rule)
    if (a != "gemma4-assistant" || I("embedding_length_out") != hp.n_embd || d_nh_ <= 0 ||
        d_nh_ % layers_[Ls].nkv || d_nh_ % layers_[Lg].nkv || d_swa_ > hp.n_swa || !pat || (int)pat->size() < nl ||
        d_E_ % 256 || d_F_ % 256 || !hp.swa[Ls] || hp.swa[Lg])
      throw std::runtime_error("unsupported Gemma 4 drafter " + path);
    d_pre_ = gemv_weight(d, "nextn.pre_projection.weight");
    d_post_ = gemv_weight(d, "nextn.post_projection.weight");
    d_lm_ = gemv_weight(d, "token_embd.weight");
    d_norm_ = f32_from(d, "output_norm.weight");
    cl_mem rope_ff = d.find("rope_freqs.weight") ? f32_from(d, "rope_freqs.weight") : nullptr;
    dl_.resize(nl);
    for (int il = 0; il < nl; ++il) {
      DraftLayer& D = dl_[il];
      const std::string p = "blk." + std::to_string(il) + ".";
      D.swa = (*pat)[il] != 0;
      D.hd = D.swa ? hp.hd_swa : hp.hd_full;
      D.wq = gemv_weight(d, p + "attn_q.weight");
      D.wo = gemv_weight(d, p + "attn_output.weight");
      D.gate = gemv_weight(d, p + "ffn_gate.weight");
      D.up = gemv_weight(d, p + "ffn_up.weight");
      D.down = gemv_weight(d, p + "ffn_down.weight");
      D.attn_norm = f32_from(d, p + "attn_norm.weight");
      D.q_norm = f32_from(d, p + "attn_q_norm.weight");
      D.post_attn = f32_from(d, p + "post_attention_norm.weight");
      D.ffn_norm = f32_from(d, p + "ffn_norm.weight");
      D.post_ffw = f32_from(d, p + "post_ffw_norm.weight");
      if (!D.swa) D.rope_ff = rope_ff;
      if (d.find(p + "layer_output_scale.weight")) {
        const GgufTensor& t = d.get(p + "layer_output_scale.weight");
        dequant_row(t.type, t.data, 1, &D.out_scale);
      }
      if (D.wq.N != d_nh_ * D.hd || D.hd != layers_[D.swa ? Ls : Lg].hd)
        throw std::runtime_error("Gemma 4 drafter: attention shape differs from the main model");
    }
    has_mtp = gemma_mtp_ = true;
    clFinish(cl_.queue);
  }

  void load_head() {
    if (hp.hc) {
      out_hc_norm_ = load_f32("output_hc_norm.weight");
      out_hc_down_ = load_hc_down("output_hc_down.weight", "");
      out_hc_up_ = load_hc_up("output_hc_up.weight");
      if (hp.ple_layer >= 0) {
        ple_table_ = &f_.get("per_layer_token_embd.weight");
        if ((int)ple_table_->dims[0] != hp.ple_dim) throw std::runtime_error("unexpected PLE table shape");
      }
    } else {
      out_norm_ = load_f32("output_norm.weight");
    }
    // lm_head is only ever a GEMV on the last token: keep just the raw rows.
    const GgufTensor& lt = tp_head(f_.find("output.weight") ? f_.get("output.weight") : *tok_embd_);
    lm_head_.N = (int)lt.dims[1];
    lm_head_.K = (int)lt.dims[0];
    if (lt.type == GGML_F32 || lt.type == GGML_F16 || lt.type == GGML_BF16) {
      std::vector<float> v((size_t)lm_head_.N * lm_head_.K);
      dequant_row(lt.type, lt.data, (int)v.size(), v.data());
      const bool q4 = getenv("ARC_HEAD_Q4") != nullptr;
      auto q8 = q4 ? quantize_rows_q4_0(v.data(), lm_head_.N, lm_head_.K) : quantize_rows_q8_0(v.data(), lm_head_.N, lm_head_.K);
      lm_head_.type = q4 ? GGML_Q4_0 : GGML_Q8_0;
      lm_head_.raw = upload_padded(lm_head_.type, q8.data(), q8.size());
    } else {
      lm_head_.type = lt.type;
      lm_head_.raw = upload_padded(lt.type, lt.data, lt.nbytes);
    }
  }


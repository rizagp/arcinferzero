// A part of class Engine (zero/engine.h), included inside the class body at the place it was cut from (the
// class is the same text), not a header of its own: the kernels the forward passes use (build_kernels) and the Qwen weights (load_weights).

  void build_kernels() {
    const std::string dir = opt.kernel_dir;
    gemm_src_ = read_file(dir + "/gemm.cl");
    ops_.load(dir, KernelParts::kOps);
    dec_.load(dir, KernelParts::kDecode);
    if (hp.moe) moe_src_ = read_file(dir + "/moe.cl");
    auto p = [&](const char* n) { return kernel("ops", "", n); };
    // (h8f_defs_: H8's float copy of the activations, zero/engine_h8.h; empty otherwise)
    k_rmsnorm_ = kernel("ops", h8f_defs_, "rmsnorm_quant");
    if (hp.ds41) {  // (DeepSeek-V4.1: its shared expert's SwiGLU clamped; none of the Qwen attention / DeltaNet kernels)
      k_swiglu_ = kernel("ops", h8f_defs_ + ds_swi_defs(), "swiglu_quant");
      auto tabs = build_iq_tables();
      tabs_ = buf(tabs.size() * 4, tabs.data());
      return;
    }
    k_swiglu_ = kernel("ops", h8f_defs_, "swiglu_quant");
    if (getenv("ARC_CT")) conv_ct_ = std::max(1, atoi(getenv("ARC_CT")));
    {
      const char* g = getenv("ARC_CONV_GRF");
      const bool big = g != nullptr && std::string(g) == "256";
      const std::string co = (big ? std::string("-cl-intel-256-GRF-per-thread ") : std::string()) + "-DCT=" +
                             std::to_string(conv_ct_);
      k_gdn_conv_ = kernel("ops", co, "gdn_conv");
      k_gdn_conv1_ = kernel("ops", co + " -DCONV_UPD", "gdn_conv");
    }
    k_conv_upd_ = p("conv_state_update");
    k_gnorm_ = hp.hc ? kernel("ops", "-DGATE_SIGMOID", "gated_norm_quant") : kernel("ops", h8f_defs_, "gated_norm_quant");
    k_attn_prep_ = kernel("ops", sb_opt(256), "attn_prep");
    k_attn_gate_ = kernel("ops", h8f_defs_, "attn_gate_quant");
    k_gdn_gates_ = p("gdn_gates");
    const std::string big = "-cl-intel-256-GRF-per-thread -DGQ=" + std::to_string(gq());
    k_attn_ = kernel("ops", big, "attn_fa");
    k_gdn_recur_ = kernel("ops", big, "gdn_recur");
    k_gdn_recur_g_ = kernel("ops", big + " -DFUSE_GATES", "gdn_recur");
    const std::string att = "-cl-intel-256-GRF-per-thread -DHPG=" + std::to_string(hp.n_head / hp.n_head_kv) +
                            " -DHS=" + std::to_string(attn_hs()) + " -DTPW=" + std::to_string(attn_tpw()) + sb_opt(256) +
                            (hp.qwen3 ? " -DHD=" + std::to_string(hp.head_dim) : std::string());
    k_attn_q8_ = kernel("ops", att, "attn_q8");
    attn_q8_defs_ = att;  // (attnx_init builds the XMX form with the same options: zero/xe2/engine_xe2.h)
    k_attn_comb_ = kernel("ops", "", "attn_combine");
    k_mtp_concat_ = kernel("ops", "", "mtp_concat");
    auto tabs = build_iq_tables();
    tabs_ = buf(tabs.size() * 4, tabs.data());
  }

  cl_mem buf(size_t bytes, const void* host = nullptr) {
    gpu_bytes += bytes;
    if (opt.dry) return nullptr;
    mems_.push_back(cl_.buffer(CL_MEM_READ_WRITE, bytes, host));
    return mems_.back();
  }
  // releases a buffer of buf() before the engine ends (an encoder growing its buffers; its from() views go with it)
  void free_buf(cl_mem& m) {
    if (!m) return;
    // (its views made by from() go with it: a later buffer at the same address must not find them)
    for (auto v = views_.begin(); v != views_.end();) {
      if (v->first.first != m) { ++v; continue; }
      auto vi = std::find(mems_.begin(), mems_.end(), v->second);
      if (vi != mems_.end()) mems_.erase(vi);
      clReleaseMemObject(v->second);
      v = views_.erase(v);
    }
    auto it = std::find(mems_.begin(), mems_.end(), m);
    if (it != mems_.end()) mems_.erase(it);
    gpu_bytes -= std::min<size_t>(gpu_bytes, m->size);
    clReleaseMemObject(m);
    m = nullptr;
  }
  // Raw GGUF rows uploaded with each superblock zero-padded to a multiple of 8 bytes (zero/kernels/decode.cl BBP),
  // written in parallel straight into a host-visible buffer (zero-copy on an integrated GPU).
  // Zero bytes after the last superblock: gemv reads two superblocks per step (rounded up to 64-byte block reads),
  // so with an odd superblock count its last row reads up to one superblock + 63 bytes past the data (lanes whose
  // block is past the row contribute 0 x those bytes, which must be finite: zeros). 1024 covers every type.
  static constexpr size_t kRawSlack = 1024;
  cl_mem upload_padded(uint32_t type, const uint8_t* data, size_t nbytes) {
    const size_t bb = gguf_sb_bytes(type), bp = raw_padded_sb_bytes(type), nsb = nbytes / bb;
    const size_t total = nsb * bp + kRawSlack;
    // (in host memory while plan_tight's share of the weights lasts: the first matrices loaded; the same bytes and
    //  kernels, so the logits are bit-identical)
    const bool host = host_left_ >= total;
    if (host) {
      host_left_ -= total;
      host_bytes_ += total;
    }
    if (opt.dry) {
      if (!host) gpu_bytes += total;
      return nullptr;
    }
    loadp::add((double)nbytes);
    cl_int err;
    cl_mem m;
    uint8_t* dst;
    if (host) {
      void* p = zrt::host_alloc(total);
      host_ptrs_.push_back(p);
      m = zrt::wrap(p, total);
      mems_.push_back(m);
      dst = (uint8_t*)p;
    } else {
      m = clCreateBuffer(cl_.ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, total, nullptr, &err);
      CL_CHECK(err);
      mems_.push_back(m);
      gpu_bytes += total;
      dst = (uint8_t*)clEnqueueMapBuffer(cl_.queue, m, CL_TRUE, CL_MAP_WRITE, 0, total, 0, nullptr, nullptr, &err);
      CL_CHECK(err);
    }
    const int64_t chunks = (int64_t)((nsb + 4095) / 4096);
#pragma omp parallel for schedule(static)
    for (int64_t ch = 0; ch < chunks; ++ch) {
      const size_t s0 = (size_t)ch * 4096, s1 = std::min(nsb, s0 + 4096);
      if (bb == bp && !gpu_regrouped(type)) {
        memcpy(dst + s0 * bp, data + s0 * bb, (s1 - s0) * bb);
      } else {
        for (size_t i = s0; i < s1; ++i) gpu_raw_sb(type, data + i * bb, dst + i * bp);
      }
    }
    memset(dst + nsb * bp, 0, kRawSlack);
    if (!host) CL_CHECK(clEnqueueUnmapMemObject(cl_.queue, m, dst, 0, nullptr, nullptr));
    return m;
  }

  void fill(cl_mem b, size_t bytes) {
    if (!b) return;  // (a dry load)
    const uint32_t zero = 0;
    CL_CHECK(clEnqueueFillBuffer(cl_.queue, b, &zero, 4, 0, bytes, 0, nullptr, nullptr));
  }

  // Uploads a matrix: raw rows always, plus its prompt-processing form when resident.
  // (view: the raw rows already on the GPU, a sub-buffer of a stacked upload)
  // ARC_W8_PARTIAL_GB=x (B60 prefill plan 13.6, opt-in until a discrete card measures it): when the weights are
  // decoded just in time (raw + W8 above --budget), the layer matrices loaded first, up to x GB of decoded W8, keep
  // that copy resident, so a prompt chunk decodes only the rest. The same values as decoded just in time (one decoder),
  // so the logits are the same bit for bit. Not with a split model or MoE (their own plans); the head stays raw.
  double w8_part_left_ = -1, w8_part_bytes_ = 0;
  int w8_part_n_ = 0;
  bool w8_part_take(int role, size_t bytes) {
    if (w8_part_left_ < 0) {
      const char* e = getenv("ARC_W8_PARTIAL_GB");
      w8_part_left_ = e && !tp_on() && !hp.moe ? std::max(0.0, atof(e)) * 1073741824.0 : 0.0;
    }
    if (role >= kRoles - 1 || (double)bytes > w8_part_left_) return false;
    w8_part_left_ -= (double)bytes;
    w8_part_bytes_ += (double)bytes;
    ++w8_part_n_;
    return true;
  }
  GpuWeight make_weight(uint32_t type, const uint8_t* data, size_t nbytes, int N, int K, int role = kRoles - 1,
                        const cl_mem* view = nullptr) {
    GpuWeight w;
    w.role = role;
    w.type = type;
    w.N = N;
    w.K = K;
    w.raw = view ? *view : upload_padded(type, data, nbytes);
    // Q8_0 (per-32 float scales) is fastest in its own repacked kernel; everything else goes through W8. The
    // repacked copy is kept only with resident weights of a dense model; otherwise (MoE: ~1.5 GB of Q8_0 matrices
    // stored twice, or just-in-time models) prompt chunks decode the raw rows into the same form (decode_w8, PER32).
    if (type == GGML_Q8_0 && (!w8_resident || hp.moe)) {
      w.gemm_qtype = (int)type;
      stage_need_q_[w.role] = std::max(stage_need_q_[w.role], w8_bytes_q(N, K));
      stage_need_h_[w.role] = std::max(stage_need_h_[w.role], w8_bytes_h(N, K));
      return w;
    }
    if (type == GGML_Q8_0) {
      RepackedWeight r;
      repack_rows(type, data, K, 0, N, r);
      w.wq = buf(r.wq.size() * 4, r.wq.data());
      w.wh = buf(r.wh.size() * 4, r.wh.data());
      w.gemm_qtype = (int)type;
      return w;
    }
    if (!is_w8_type(type)) throw std::runtime_error(std::string("unsupported weight type ") + ggml_type_name(type));
    // per-32-scale types (Q4_0, IQ4_NL, Q2_0) decode into the native Q8_0 GEMM form (per-32 f32 scales; there is no
    // such W8 variant)
    w.gemm_qtype = w8_variant(type) == 4 ? (int)GGML_Q8_0 : 100 + w8_variant(type);
    const size_t qb = w8_bytes_q(N, K), hb = w8_bytes_h(N, K);
    if (w8_resident || w8_part_take(role, qb + hb)) {
      w.wq = buf(qb);
      w.wh = buf(hb);
      if (!opt.dry) decode_w8(w, w.wq, w.wh);
    } else {
      stage_need_q_[w.role] = std::max(stage_need_q_[w.role], qb);
      stage_need_h_[w.role] = std::max(stage_need_h_[w.role], hb);
    }
    return w;
  }
  static int weight_role(const std::string& name) {
    static const char* roles[] = {"attn_qkv.", "attn_gate.", "ssm_out.", "attn_q.", "attn_k.", "attn_v.",
                                  "attn_output.", "ffn_gate.", "ffn_up.", "ffn_down.", "nextn.eh_proj."};
    for (int r = 0; r < (int)(sizeof(roles) / sizeof(roles[0])); ++r)
      if (name.find(roles[r]) != std::string::npos) return r;
    return kRoles - 1;
  }
  // a matrix the loaders read as rows: two dimensions or more, sizes an int holds, bytes present (a damaged or hostile
  // GGUF: an error here, not a read past the tensor or at address 0)
  static const GgufTensor& mat(const GgufTensor& t) {
    if (t.dims.size() < 2 || t.dims[0] == 0 || t.dims[1] == 0 || t.dims[0] > (uint64_t)INT32_MAX ||
        t.dims[1] > (uint64_t)INT32_MAX || t.dims[0] * t.dims[1] > (uint64_t)INT32_MAX || (!t.data && t.nbytes))
      throw std::runtime_error("unexpected tensor shape: " + t.name);
    return t;
  }
  GpuWeight load_w(const std::string& name) {
    const GgufTensor& t = mat(T(name));
    const int K = (int)t.dims[0], N = (int)t.dims[1];
    if (t.type == GGML_F32 || t.type == GGML_F16 || t.type == GGML_BF16) {
      std::vector<float> v((size_t)N * K);
      dequant_row(t.type, t.data, N * K, v.data());
      auto q8 = quantize_rows_q8_0(v.data(), N, K);
      return make_weight(GGML_Q8_0, q8.data(), q8.size(), N, K, weight_role(name));
    }
    return make_weight(t.type, t.data, t.nbytes, N, K, weight_role(name));
  }
  // DeltaNet input projections (qkv, z = attn_gate, ab = alpha + beta). When qkv and z have the same quantized type
  // they are uploaded as one stack of rows (with ab, already Q8_0, when they are Q8_0 too): a single token runs one GEMV
  // over all of them (Layer::qz); the matrices are views of it, stored once.
  static bool float_type(uint32_t t) { return t == GGML_F32 || t == GGML_F16 || t == GGML_BF16; }
  // a float matrix as the Q8_0 rows load_w would make of it (kept in `keep`), so that it can be stacked with others
  const GgufTensor& as_q8(const GgufTensor& t, GgufTensor& out, std::vector<uint8_t>& keep) {
    mat(t);
    std::vector<float> v((size_t)t.dims[0] * t.dims[1]);
    dequant_row(t.type, t.data, (int)v.size(), v.data());
    keep = quantize_rows_q8_0(v.data(), (int)t.dims[1], (int)t.dims[0]);
    out = t;
    out.type = GGML_Q8_0;
    out.data = keep.data();
    out.nbytes = keep.size();
    return out;
  }
  void load_gdn_in(Layer& L, const std::string& p) {
    if (hp.hc && fn_precise() && !tp_on()) {
      const GgufTensor &alpha = T(p + "ssm_alpha.weight"), &beta = T(p + "ssm_beta.weight");
      const int E = hp.n_embd, N = hp.ssm_nv;
      if (alpha.nelements() != (uint64_t)N * E || beta.nelements() != (uint64_t)N * E)
        throw std::runtime_error("unexpected DeltaNet gate projection shape");
      std::vector<float> values((size_t)2 * N * E);
      dequant_row(alpha.type, alpha.data, N * E, values.data());
      dequant_row(beta.type, beta.data, N * E, values.data() + (size_t)N * E);
      L.ab_f32 = buf(values.size() * sizeof(float), values.data());
    }
    GgufTensor q8q, q8z;
    std::vector<uint8_t> kq, kz;
    const GgufTensor *pq = &T(p + "attn_qkv.weight"), *pz = &T(p + "attn_gate.weight");
    // (float qkv / z (a BF16 GGUF: Xiaomi-OCR-0): the same Q8_0 rows load_w makes, stacked as quantized ones are)
    if (float_type(pq->type) && float_type(pz->type) && !getenv("ARC_NO_QZ_FLT")) {
      pq = &as_q8(*pq, q8q, kq);
      pz = &as_q8(*pz, q8z, kz);
    }
    const GgufTensor &tq = *pq, &tz = *pz;
    const int E = hp.n_embd, C = (int)tq.dims[1], Di = (int)tz.dims[1], nab = 2 * hp.ssm_nv;
    const bool quant = tq.type != GGML_F32 && tq.type != GGML_F16 && tq.type != GGML_BF16;
    if (!quant || tq.type != tz.type || tq.dims[0] != E || tz.dims[0] != E || E % 256 || C % 64 || Di % 64 ||
        C != hp.conv_dim() || Di != hp.ssm_d_inner) {
      L.qkv = load_w(p + "attn_qkv.weight");
      L.z = load_w(p + "attn_gate.weight");
      L.ab = load_q8_concat(p + "ssm_alpha.weight", p + "ssm_beta.weight");
      return;
    }
    const uint32_t type = tq.type;
    std::vector<uint8_t> ab;
    {
      const GgufTensor &ta = T(p + "ssm_alpha.weight"), &tb = T(p + "ssm_beta.weight");
      if (mat(ta).nelements() < (uint64_t)nab / 2 * E || mat(tb).nelements() < (uint64_t)nab / 2 * E)
        throw std::runtime_error("unexpected tensor shape: " + ta.name);
      std::vector<float> v((size_t)tp_rows(nab) * E, 0.0f);  // (a split model: padded rows of zeros, see gdn_ab)
      dequant_row(ta.type, ta.data, (int64_t)nab / 2 * E, v.data());
      dequant_row(tb.type, tb.data, (int64_t)nab / 2 * E, v.data() + (size_t)nab / 2 * E);
      ab = quantize_rows_q8_0(v.data(), tp_rows(nab), E);
    }
    L.qz_ab = !(hp.hc && fn_precise() && !tp_on()) && type == GGML_Q8_0 && nab % 16 == 0 && !getenv("ARC_NO_QZAB16");
    std::vector<uint8_t> cat(tq.nbytes + tz.nbytes + (L.qz_ab ? ab.size() : 0));
    memcpy(cat.data(), tq.data, tq.nbytes);
    memcpy(cat.data() + tq.nbytes, tz.data, tz.nbytes);
    if (L.qz_ab) memcpy(cat.data() + tq.nbytes + tz.nbytes, ab.data(), ab.size());
    L.qz.type = type;
    L.qz.N = C + Di + (L.qz_ab ? nab : 0);
    L.qz.K = E;
    L.qz.role = kRoles - 1;
    L.qz.raw = upload_padded(type, cat.data(), cat.size());
    has_qz_ = true;
    const size_t sbr = (size_t)E / 256 * raw_padded_sb_bytes(type);
    auto view = [&](int row0, int N) -> cl_mem {
      if (!L.qz.raw) return nullptr;  // (dry load)
      cl_buffer_region reg{(size_t)row0 * sbr, (size_t)N * sbr + kRawSlack};  // (the slack upload_padded leaves: the
                                                                               //  GEMV reads up to BBP + 63 past a row)
      cl_int err;
      cl_mem m = clCreateSubBuffer(L.qz.raw, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err);
      CL_CHECK(err);
      mems_.push_back(m);
      return m;
    };
    const cl_mem vq = view(0, C), vz = view(C, Di);
    L.qkv = make_weight(type, tq.data, tq.nbytes, C, E, weight_role(p + "attn_qkv.weight"), &vq);
    L.z = make_weight(type, tz.data, tz.nbytes, Di, E, weight_role(p + "attn_gate.weight"), &vz);
    if (L.qz_ab) {
      const cl_mem va = view(C + Di, nab);
      L.ab = make_weight(GGML_Q8_0, ab.data(), ab.size(), nab, E, kRoles - 1, &va);  // (nab % 64 == 0: not padded)
    } else {
      L.ab = make_weight(GGML_Q8_0, ab.data(), ab.size(), tp_rows(nab), E);
    }
  }
  // Two matrices with the same K stacked by rows, requantized to Q8_0 (the tiny DeltaNet gate projections).
  // Matrices of one quantized type and row length as one stack of rows, their GGUF blocks copied as they are (the
  // values unchanged; a row's result does not depend on the others): one GEMV / GEMM instead of one per matrix.
  // BF16 / F16 / F32 matrices (Qwen3-ASR's BF16 file) are requantized to Q8_0 row by row, as a single one is.
  GpuWeight load_w_stack(std::initializer_list<std::string> names) {
    const GgufTensor& t0 = T(*names.begin());
    const bool flt = t0.type == GGML_F32 || t0.type == GGML_F16 || t0.type == GGML_BF16;
    const int K = (int)t0.dims[0];
    std::vector<uint8_t> cat;
    std::vector<float> v;
    int N = 0;
    for (const std::string& n : names) {
      const GgufTensor& t = T(n);
      if (t.type != t0.type || t.dims[0] != t0.dims[0])
        throw std::runtime_error("load_w_stack: matrices of different shapes or types: " + n);
      if (flt) {
        v.resize((size_t)(N + t.dims[1]) * K);
        dequant_row(t.type, t.data, (int)(t.dims[1] * K), v.data() + (size_t)N * K);
      } else {
        cat.insert(cat.end(), t.data, t.data + t.nbytes);
      }
      N += (int)t.dims[1];
    }
    if (flt) {
      auto q8 = quantize_rows_q8_0(v.data(), N, K);
      return make_weight(GGML_Q8_0, q8.data(), q8.size(), N, K, weight_role(*names.begin()));
    }
    return make_weight(t0.type, cat.data(), cat.size(), N, K, weight_role(*names.begin()));
  }
  GpuWeight load_q8_concat(const std::string& a, const std::string& b) {
    const GgufTensor &ta = mat(T(a)), &tb = mat(T(b));
    if (tb.dims[0] != ta.dims[0]) throw std::runtime_error("unexpected tensor shape: " + b);
    const int K = (int)ta.dims[0], Na = (int)ta.dims[1], Nb = (int)tb.dims[1], Np = tp_rows(Na + Nb);
    std::vector<float> v((size_t)Np * K, 0.0f);  // (a split model: rows of zeros up to the GEMM's 16, see gdn_ab)
    dequant_row(ta.type, ta.data, Na * K, v.data());
    dequant_row(tb.type, tb.data, Nb * K, v.data() + (size_t)Na * K);
    auto q8 = quantize_rows_q8_0(v.data(), Np, K);
    return make_weight(GGML_Q8_0, q8.data(), q8.size(), Np, K);
  }
  // a weight matrix of the size the model's dimensions give (N rows, rows of K; -1: any): one wider than the buffer
  // its product writes, or longer rows than the activations hold, would go past them on the GPU (a broken or
  // mismatched GGUF); refused when it is loaded
  static void expect(const GpuWeight& w, int N, int K, const std::string& name) {
    if ((N >= 0 && w.N != N) || (K >= 0 && w.K != K && w.K != padded_k(K)))
      throw std::runtime_error("unexpected tensor shape: " + name + " is " + std::to_string(w.N) + " x " + std::to_string(w.K) +
                               ", the model's dimensions give " + std::to_string(N) + " x " + std::to_string(K));
  }
  // a vector of n values (a norm's weights; the kernels read n of them)
  cl_mem load_vec(const std::string& name, int n) {
    if (f_.get(name).nelements() != (uint64_t)n)
      throw std::runtime_error("unexpected tensor shape: " + name + " (" + std::to_string(n) + " values expected)");
    return load_f32(name);
  }
  cl_mem load_f32(const std::string& name) {
    const GgufTensor& t = T(name);
    if ((!t.data && t.nbytes) || t.nelements() > (uint64_t)INT32_MAX) throw std::runtime_error("unexpected tensor shape: " + name);
    std::vector<float> v(t.nelements());
    dequant_row(t.type, t.data, (int)v.size(), v.data());
    return buf(v.size() * 4, v.data());
  }
  // A matrix with its rows padded to Np rows of zeros and, for a type of small blocks (32 / 64: Q8_0, IQ4_NL, Q2_0)
  // whose K is not whole superblocks, its columns to padded_k(K) (zero blocks): Flash Next's 640-wide shared expert as 768 (the extra outputs are 0).
  GpuWeight load_w_pad(const std::string& name, int Np) {
    const GgufTensor& t = mat(f_.get(name));
    const int K = (int)t.dims[0], N = (int)t.dims[1];
    // (more rows than the model's dimension: the padded copy below has room for Np rows only)
    if (N > Np) throw std::runtime_error("unexpected tensor shape: " + name + " has " + std::to_string(N) + " rows, the model " + std::to_string(Np));
    if (K % 256 == 0 && N == Np) return load_w(name);
    int blk = 0, bytes = 0;
    ggml_block_info(t.type, blk, bytes);
    if (K % 256 && (blk >= 256 || K % blk)) throw std::runtime_error("unsupported row length: " + name);
    const int Kp = padded_k(K);
    const size_t prb = (size_t)Kp / blk * bytes;
    std::vector<uint8_t> v((size_t)Np * prb, 0);
    if (K % 256) pad_small_block_rows(t.type, t.data, N, K, v.data());
    else memcpy(v.data(), t.data, (size_t)N * prb);
    return make_weight(t.type, v.data(), v.size(), Np, Kp, weight_role(name));
  }
  // Flash Next mixer: the low-rank down rows [hc_lr][HCE] and the inject rows [hc_n][HCE] stacked (padded to 16 rows)
  // as one Q8_0 matrix: the GEMM gives both from the same normed input.
  GpuWeight load_hc_down(const std::string& down, const std::string& inject) {
    const int K = hp.hce(), LR = hp.hc_lr, NI = inject.empty() ? 0 : hp.hc_n, N = (LR + NI + 15) / 16 * 16;
    std::vector<float> v((size_t)N * K, 0.0f);
    const GgufTensor& td = f_.get(down);
    if ((int)td.dims[0] != K || (int)td.dims[1] != LR) throw std::runtime_error("unexpected tensor shape: " + down);
    dequant_row(td.type, td.data, (int64_t)LR * K, v.data());
    if (NI) {
      const GgufTensor& ti = f_.get(inject);
      if ((int)ti.dims[0] != K || (int)ti.dims[1] != NI) throw std::runtime_error("unexpected tensor shape: " + inject);
      dequant_row(ti.type, ti.data, (int64_t)NI * K, v.data() + (size_t)LR * K);
    }
    if (hc_bf16()) return hc_exact_weight(v, N, K);
    auto q8 = quantize_rows_q8_0(v.data(), N, K);
    return make_weight(GGML_Q8_0, q8.data(), q8.size(), N, K);
  }
  // Flash Next's precise mode, ARC_FN_PRECISE=1 (QUEUE 2, 2026-10-24; 363 tokens of README text against the f32
  // reference, KL prompt / decode): the mixer rows as the file's BF16 values, exact (hc_down1 / hc_up1 / hc_gemm_f32
  // -DHC_BF16, f32 inputs in prompts too: with Q8_0 rows every block's residual was ~0.5 % off), the router and shared
  // gate rows BF16 too (gemv_bf16), every matmul input on two int8 passes, the experts included: 0.0143 / 0.0147 ->
  // 0.0117 / 0.0096, 4K prompt 180 -> 87 tok/s, generation 6.1 -> 5.4. The BF16 rows alone, on one pass: 0.0137 / 0.0154
  // at 139 tok/s (the int8 inputs dominate). What is left is mostly the int8 attention (q / K / V per row: simulated
  // in the f32 reference, +0.11 % at layer 3 as in the engine; per-32 scales +0.06 %): float attention with the QSA
  // masks is the next step.
  // (ARC_FN_PRECISE=1 / serverz --precise for every Flash Next; a model with AR4_128 experts turns it on for its own
  //  engine, fn_precise_hold_)
  bool fn_precise() const {
    static const bool env = getenv("ARC_FN_PRECISE") && atoi(getenv("ARC_FN_PRECISE")) != 0;
    return env || fn_precise_hold_.on;
  }
  bool hc_bf16() const { return fn_precise(); }
  // Q8 and F32 GGUF rows need not be exactly representable as BF16 after dequantization.
  // Preserve them in F32; BF16 source values retain the compact existing path.
  static bool hc_bf16_exact(const std::vector<float>& v) {
    for (float x : v) {
      uint32_t bits;
      memcpy(&bits, &x, sizeof bits);
      if (bits & 0xffffu) return false;
    }
    return true;
  }
  GpuWeight hc_exact_weight(const std::vector<float>& v, int N, int K) {
    if (hc_bf16_exact(v)) return bf16_weight(v, N, K);
    GpuWeight w;
    w.type = GGML_F32;
    w.N = N;
    w.K = K;
    w.raw = buf(v.size() * sizeof(float) + kRawSlack, nullptr);
    if (w.raw) CL_CHECK(clEnqueueWriteBuffer(cl_.queue, w.raw, CL_TRUE, 0, v.size() * sizeof(float), v.data(), 0, nullptr, nullptr));
    return w;
  }
  static std::string hc_weight_defs(const GpuWeight& w) {
    return w.type == GGML_F32 ? " -DHC_F32" : w.type == GGML_BF16 ? " -DHC_BF16" : "";
  }
  // rows of f32 values that came from BF16 (exact) stored back as BF16, for the -DHC_BF16 kernels
  GpuWeight bf16_weight(const std::vector<float>& v, int N, int K) {
    std::vector<uint16_t> bf(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
      uint32_t u;
      memcpy(&u, &v[i], 4);
      bf[i] = (uint16_t)(u >> 16);
    }
    GpuWeight w;
    w.type = GGML_BF16;
    w.N = N;
    w.K = K;
    w.raw = buf(bf.size() * 2 + kRawSlack, nullptr);
    if (w.raw) CL_CHECK(clEnqueueWriteBuffer(cl_.queue, w.raw, CL_TRUE, 0, bf.size() * 2, bf.data(), 0, nullptr, nullptr));
    return w;
  }
  int hc_lr_pad() const { return padded_k(hp.hc_lr); }
  // Flash Next mixer up projection [HCE][hc_lr], its K padded with zeros to a multiple of 256
  GpuWeight load_hc_up(const std::string& name) {
    const GgufTensor& t = f_.get(name);
    const int K = hp.hc_lr, N = hp.hce(), KP = hc_lr_pad();
    if ((int)t.dims[0] != K || (int)t.dims[1] != N) throw std::runtime_error("unexpected tensor shape: " + name);
    std::vector<float> a((size_t)N * K), v((size_t)N * KP, 0.0f);
    dequant_row(t.type, t.data, (int64_t)N * K, a.data());
    for (int n = 0; n < N; ++n) memcpy(&v[(size_t)n * KP], &a[(size_t)n * K], (size_t)K * 4);
    if (hc_bf16()) return hc_exact_weight(v, N, KP);
    auto q8 = quantize_rows_q8_0(v.data(), N, KP);
    return make_weight(GGML_Q8_0, q8.data(), q8.size(), N, KP);
  }

  // Qwen3.8 Flash Next's MTP head in a file of its own (QUEUE F34, GitHub #7): --mtp-model FILE, else an mtp-*.gguf next
  // to the model of the same architecture whose blocks are this model's plus its nextn block (llama.cpp's form: the
  // base model's head, e.g. mtp-Qwen3.8-Flash-Next-Q4_0.gguf). Its blk.N.* tensors join f_'s list, so the loaders read
  // them as the model's own; its output / token_embd are not taken (the model's are used: drafts are verified anyway).
  // Runs before load_hparams (no pointer into f_'s tensor list is kept yet). Without --mtp: nothing.
  void adopt_mtp_file() {
    const std::string a = f_.get_str("general.architecture");
    // (--draft dflash without a DFlash2 folder still drafts with the block, as init() decides after this)
    if (a != "qwen4exp" || !opt.mtp || (!opt.mtp_block && !opt.dflash.empty())) return;
    const int64_t nb = f_.get_int(a + ".block_count", 0);
    if (nb <= 0 || f_.get_int(a + ".nextn_predict_layers", 0) > 0) return;  // (a block of its own)
    const std::string mp = "blk." + std::to_string(nb) + ".";
    auto fits = [&](const std::string& p) {
      try {
        auto g = std::make_unique<GgufFile>(p);
        if (g->get_str("general.architecture") != a || g->get_int(a + ".block_count", 0) != nb + 1 ||
            g->get_int(a + ".nextn_predict_layers", 0) != 1 ||
            g->get_int(a + ".embedding_length", -1) != f_.get_int(a + ".embedding_length", -2) ||
            g->get_int(a + ".expert_count", -1) != f_.get_int(a + ".expert_count", -2) ||
            !g->find(mp + "nextn.eh_proj.weight"))
          return false;
        mtp_file_ = std::move(g);
        return true;
      } catch (const std::exception&) {
        return false;
      }
    };
    if (!opt.mtp_model.empty()) {
      if (!fits(opt.mtp_model))
        throw std::runtime_error("--mtp-model " + opt.mtp_model + ": not an MTP head of this model (" + a + ", " +
                                 std::to_string(nb) + " blocks + 1)");
    } else {
      namespace fs = std::filesystem;
      std::error_code ec;
      const fs::path dir = fs::path(path_).parent_path();
      for (const auto& e : fs::directory_iterator(dir.empty() ? fs::path(".") : dir, ec)) {
        const std::string n = e.path().filename().string();
        if (n.rfind("mtp-", 0) == 0 && e.path().extension() == ".gguf" && fits(e.path().string())) break;
      }
    }
    if (!mtp_file_) return;
    const int n = f_.adopt(*mtp_file_, mp);
    mtp_ext_ = true;
    zlog::eprintf("inferzero: MTP head %s (%d tensors)\n", mtp_file_->path.c_str(), n);
  }

  void load_weights() {
    // The output head first: the GPU reads a large buffer allocated late in the load (after gigabytes of other
    // weights) at about half the speed (Gemma 4 12B's 826 MB head: 14 ms per token first, 21-28 ms last).
    load_head();
    if (hp.gemma) return load_weights_gemma4();
    if (hp.ds41) return load_weights_ds41();
    layers_.resize(hp.n_layer);
    for (int il = 0; il < hp.n_layer; ++il) {
      Layer& L = layers_[il];
      L.il = il;
      const std::string p = "blk." + std::to_string(il) + ".";
      if (hp.hc) {
        for (int b = 0; b < 2; ++b) {
          const std::string hp_ = p + (b ? "hc_ffn" : "hc_attn");
          L.hc_norm[b] = load_f32(hp_ + "_norm.weight");
          L.hc_down[b] = load_hc_down(hp_ + "_down.weight", hp_ + "_inject.weight");
          L.hc_up[b] = load_hc_up(hp_ + "_up.weight");
        }
        if (il == hp.ple_layer) {
          L.ple_key = load_w(p + "ple_key.weight");
          L.ple_value = load_w(p + "ple_value.weight");
          L.ple_nk = load_f32(p + "ple_norm_key.weight");
          L.ple_nq = load_f32(p + "ple_norm_query.weight");
          L.ple_nc = load_f32(p + "ple_norm_conv.weight");
          L.ple_cw = load_f32(p + "ple_conv1d.weight");
          if (L.ple_key.N != hp.hce() || L.ple_key.K != hp.n_embd || L.ple_value.N != hp.n_embd)
            throw std::runtime_error("unexpected PLE block shape");
        }
      } else {
        L.attn_norm = load_vec(p + "attn_norm.weight", hp.n_embd);
        L.post_norm = load_vec(p + (hp.qwen3 ? "ffn_norm.weight" : "post_attention_norm.weight"), hp.n_embd);
      }
      if (hp.recurrent(il)) {
        load_gdn_in(L, p);
        L.ssm_out = load_w(p + "ssm_out.weight");
        expect(L.ssm_out, hp.n_embd, -1, p + "ssm_out.weight");
        L.dt_bias = load_f32(p + "ssm_dt.bias");
        L.ssm_a = load_f32(p + "ssm_a");
        L.conv_w = load_f32(p + "ssm_conv1d.weight");
        L.ssm_norm = load_f32(p + "ssm_norm.weight");
        L.conv_state = buf(conv_bytes());
        L.S = buf((size_t)hp.ssm_nv * hp.ssm_d_state * hp.ssm_d_state * 4);
        if (opt.mtp && !hp.gemma) {
          const size_t g0 = gpu_bytes, rows = 1 + kMaxDrafts;
          L.S_next = buf((size_t)hp.ssm_nv * hp.ssm_d_state * hp.ssm_d_state * 4);
          L.conv_pre = buf(conv_bytes());
          L.xin = buf(rows * hp.conv_dim() * 4);
          L.qkvin = buf(rows * hp.conv_dim() * 4);
          L.abin = buf(rows * 2 * hp.ssm_nv * 4);
          mem_mtp_ += gpu_bytes - g0;  // (the state of a verification pass, for rejected drafts)
        }
      } else {
        load_attn(L, p);
      }
      if (hp.moe) {
        load_moe(L, p);
      } else if ((hp.qwen3 || (float_type(f_.get(p + "ffn_gate.weight").type) && !getenv("ARC_NO_GU_FLT"))) &&
                 f_.get(p + "ffn_gate.weight").type == f_.get(p + "ffn_up.weight").type) {
        L.gate = load_w_stack({p + "ffn_gate.weight", p + "ffn_up.weight"});  // (rows [gate | up], L.up empty: layer_ffn)
        L.down = load_w(p + "ffn_down.weight");
        expect(L.gate, 2 * hp.n_ff, hp.n_embd, p + "ffn_gate.weight + ffn_up.weight");
      } else {
        L.gate = load_w(p + "ffn_gate.weight");
        L.up = load_w(p + "ffn_up.weight");
        L.down = load_w(p + "ffn_down.weight");
        expect(L.gate, hp.n_ff, hp.n_embd, p + "ffn_gate.weight");
        expect(L.up, hp.n_ff, hp.n_embd, p + "ffn_up.weight");
      }
      if (!hp.moe) expect(L.down, hp.n_embd, hp.n_ff, p + "ffn_down.weight");
      clFinish(cl_.queue);
    }
    // MTP (nextn) block: eh_proj + one gated-attention layer + FFN, sharing the output head
    const std::string mp = "blk." + std::to_string(hp.n_layer) + ".";
    const size_t g_mtp = gpu_bytes;
    struct MtpBytes { Engine& e; size_t g0; ~MtpBytes() { e.mem_mtp_ += e.gpu_bytes - g0; } } mtp_bytes{*this, g_mtp};
    if (opt.mtp && opt.mtp_block && f_.find(mp + "nextn.eh_proj.weight")) {
      has_mtp = mtp_block_ = true;
      mtp_.il = hp.n_layer;
      if (hp.hc) {  // (Flash Next, QUEUE F34: a layer as its others, with the mixers; no PLE)
        for (int b = 0; b < 2; ++b) {
          const std::string hp_ = mp + (b ? "hc_ffn" : "hc_attn");
          mtp_.hc_norm[b] = load_f32(hp_ + "_norm.weight");
          mtp_.hc_down[b] = load_hc_down(hp_ + "_down.weight", hp_ + "_inject.weight");
          mtp_.hc_up[b] = load_hc_up(hp_ + "_up.weight");
        }
      } else {
        mtp_.attn_norm = load_f32(mp + "attn_norm.weight");
        mtp_.post_norm = load_f32(mp + "post_attention_norm.weight");
      }
      load_attn(mtp_, mp);
      if (hp.moe) {
        const uint64_t sb0 = stream_bytes_;
        load_moe(mtp_, mp);
        if (mtp_ext_ && stream_) {  // (not in the model's expert file: all of them on the GPU)
          stream_bytes_ = sb0;
          if (!opt.dry) pin_mtp_experts(mtp_);
          else gpu_bytes += pin_bytes(mtp_.il);
        }
      } else {
        mtp_.gate = load_w(mp + "ffn_gate.weight");
        mtp_.up = load_w(mp + "ffn_up.weight");
        mtp_.down = load_w(mp + "ffn_down.weight");
      }
      eh_proj_ = load_w(mp + "nextn.eh_proj.weight");
      enorm_ = load_f32(mp + "nextn.enorm.weight");
      hnorm_ = load_f32(mp + "nextn.hnorm.weight");  // (Flash Next: one row of weights per stream, [hc][E])
      if (hp.hc) {
        mtp_hc_norm_ = load_f32(mp + "nextn.hc_head_norm.weight");
        mtp_hc_down_ = load_hc_down(mp + "nextn.hc_head_down.weight", "");
        mtp_hc_up_ = load_hc_up(mp + "nextn.hc_head_up.weight");
      } else {
        shnorm_ = load_f32(mp + "nextn.shared_head_norm.weight");
      }
      clFinish(cl_.queue);
    }
    // DFlash2 (with or without the MTP block: with both, mtp_draft picks one per step; Flash Next: the features are the
    // means of the residual streams, the PLE / QSA state rolled back in verify_done)
    if (!opt.dflash.empty()) load_dflash(opt.dflash);
  }

  // MoE FFN: the router and the shared-expert gate requantized together to Q8_0 (one GEMM gives both), the shared
  // expert's Q8_0 matrices, and the routed experts' 2-bit codes and bf16 scales as they are in the file.

// Included inside class Engine (zero/engine.h): the NPU experiment (branch npu-experiment, zero/npu/), opt-in with
// ARC_NPU=1, off by default (every function here returns at once then). Qwen dense models only: in prompt chunks of at
// least ARC_NPU_MIN rows (default 2048) the NPU computes the last share of the FFN's gate and up columns
// (ARC_NPU_SHARE, default 1/3) while the GPU computes the rest:
//   - the FFN norm writes its f32 rows once more (rmsnorm_quant with its f32 output on), a small kernel quantizes them
//     per row into memory both Level Zero contexts map (external memmap, zero/npu/npu_rt.h);
//   - the NPU runs the "fqi" graph (int8 rows and weight rows, FakeQuantize with constant bounds, then the row and
//     column scales: exact per-row int8 on its int8 units), one graph per chunk height (blob cache), the layer's
//     weights as its inputs (per-channel int8, a resident copy made at load);
//   - the GPU meanwhile runs gate / up on its columns (the same GEMMs and fused SwiGLU epilogue, on copies of the
//     weights with N = its share), waits for the NPU, and writes silu(gate) * up of the NPU's columns into big_.
// KL against f32 (tools/ref_forward REF_NPU, 27B-AP, 128 tokens): 1/3 of gate / up 0.0022 against 0.00046 without.

static bool npu_wanted() {
  const char* e = getenv("ARC_NPU");
  return e && atoi(e) != 0;
}
// (before the GPU runtime: it initializes Level Zero for GPUs only, which would hide the NPU driver)
static void npu_preinit() {
  if (npu_wanted() && ze_load()) zeInit(0);
}

// A matrix whose last columns the NPU computes in f16 ("w8g": its weight rows int8 with one f16 scale per 32
// values, dequantized in the graph; the activations f16): ffn_down and the DeltaNet layers' attn_qkv
struct NpuMat {
  GpuWeight gpu;                // (the GPU's columns: a persistent copy with N = its share)
  int8_t* bq = nullptr;         // (the NPU's rows, int8)
  uint16_t* bs = nullptr;       // (their f16 scales, one per 32 values)
  int ng = 0, nn = 0, K = 0;
  bool on = false;
};
struct NpuLayer {
  GpuWeight gate, up;              // (the GPU's columns: copies with N = its share; persistent, the staging keys on them)
  int8_t* bq[2] = {nullptr, nullptr};  // (the NPU's columns of gate / up: int8 rows)
  uint16_t* bs[2] = {nullptr, nullptr};  // (their f16 row scales, absmax)
  bool on = false;
  NpuMat down, qkv;
};
std::unique_ptr<npu::Rt> npu_;
std::vector<NpuLayer> npu_layers_;
std::map<int, ze_graph_handle_t> npu_graphs_;  // (by chunk height)
int npu_ng_ = 0, npu_nn_ = 0, npu_min_ = 2048;
uint8_t* npu_shm_ = nullptr;
size_t npu_a_ = 0, npu_sa_ = 0, npu_cg_ = 0, npu_cu_ = 0;
cl_mem npu_xn_ = nullptr, npu_gaq_ = nullptr, npu_gsa_ = nullptr, npu_gcg_ = nullptr, npu_gcu_ = nullptr;
cl_kernel npu_rowq_ = nullptr, npu_merge_ = nullptr, npu_to16_ = nullptr, npu_norm16_ = nullptr, npu_put_ = nullptr;
// (the f16 part: one input block A16 for the matrix in flight (Mp x K f16, K up to n_ff), an output block per kind)
size_t npu_a16_ = 0, npu_cd_ = 0, npu_cq_ = 0;
cl_mem npu_ga16_ = nullptr, npu_gcd_ = nullptr, npu_gcq_ = nullptr;
std::map<std::pair<int, int>, ze_graph_handle_t> npu_graphs16_;  // (by kind (0 down, 1 qkv) and chunk height)
double npu_wait_ms_ = 0;
int npu_runs_ = 0;
size_t npu_shm_bytes_ = 0;

// everything of the NPU path (the engine's destructor; a server switching models frees its NPU memory with it): the
// shared block's mappings in both contexts, then the NPU runtime (graphs, weight rows, blobs, context)
void npu_teardown() {
  if (!npu_) return;
  npu_->wait();
  if (npu_shm_) {
    zeMemFree(npu_->ctx, npu_shm_);
    zeMemFree(zrt::ctx, npu_shm_);
  }
  npu_graphs_.clear();
  npu_graphs16_.clear();
  npu_layers_.clear();
  npu_.reset();
  if (npu_shm_) VirtualFree(npu_shm_, 0, MEM_RELEASE);
  npu_shm_ = nullptr;
}

void npu_setup() {
  if (!npu_wanted() || opt.dry) return;
  if (hp.gemma || hp.hc || (hp.arch != "qwen35" && hp.arch != "qwen35moe")) {
    fprintf(stderr, "ARC_NPU: Qwen3.5 / 3.6 / 3.8 dense and MoE models only; off for this one\n");
    return;
  }
  // (measured 2026-10-25: the NPU's share pays off where the GPU decodes the weights per chunk (W8 just in time: the
  //  27B, +15-20 % prompt); with resident W8 the GPU's GEMMs are faster than the NPU makes its share (4B +2 %, 9B +4-6 %
  //  at KL 0.008-0.013, 35B +0.5 %): off there unless ARC_NPU=2)
  if (w8_resident && atoi(getenv("ARC_NPU")) < 2) {
    fprintf(stderr, "ARC_NPU: off for this model (its W8 weights stay resident: no gain measured); ARC_NPU=2 forces it\n");
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  npu_ = std::make_unique<npu::Rt>();
  if (!npu_->init()) {
    fprintf(stderr, "ARC_NPU: no NPU (Level Zero graph extension) found; off\n");
    npu_.reset();
    return;
  }
  const int E = hp.n_embd, F = hp.n_ff, M = max_chunk;
  const double share = getenv("ARC_NPU_SHARE") ? atof(getenv("ARC_NPU_SHARE")) : 1.0 / 3.0;
  if (getenv("ARC_NPU_MIN")) npu_min_ = std::max(16, atoi(getenv("ARC_NPU_MIN")));
  // (gate / up: dense FFNs only; ARC_NPU_SHARE=0 turns that part off)
  npu_nn_ = hp.moe || F < 256 || share <= 0 ? 0 : std::clamp((int)std::lround(F * share / 64.0) * 64, 64, F - 64);
  npu_ng_ = F - npu_nn_;
  // the f16 shares (ARC_NPU_DOWN / ARC_NPU_QKV, 0: off): about where the NPU (w8g, ~2.9-3.1 TFLOPS at 4096 rows) and
  // the GPU finish together on the 27B
  auto cols = [](int N, double s) { return s <= 0 ? 0 : std::clamp((int)std::lround(N * s / 64.0) * 64, 64, N - 64); };
  const int Cq = hp.conv_dim();
  const int nd = cols(E, getenv("ARC_NPU_DOWN") ? atof(getenv("ARC_NPU_DOWN")) : 0.275);
  const int nq = cols(Cq, getenv("ARC_NPU_QKV") ? atof(getenv("ARC_NPU_QKV")) : 0.29);
  // shared memory: int8 rows, their scales, the NPU's two f16 outputs, for max_chunk rows; then the f16 part's input
  // (the widest K: n_ff) and its outputs
  npu_a_ = 0;
  npu_sa_ = (size_t)M * E;
  npu_cg_ = (npu_sa_ + (size_t)M * 2 + 4095) / 4096 * 4096;
  npu_cu_ = npu_cg_ + (size_t)M * npu_nn_ * 2;
  npu_a16_ = (npu_cu_ + (size_t)M * npu_nn_ * 2 + 4095) / 4096 * 4096;
  npu_cd_ = npu_a16_ + (size_t)M * std::max(F, E) * 2;
  npu_cq_ = (npu_cd_ + (size_t)M * std::max(nd, 64) * 2 + 4095) / 4096 * 4096;
  const size_t total = (npu_cq_ + (size_t)M * std::max(nq, 64) * 2 + 65535) / 65536 * 65536;
  npu_shm_ = (uint8_t*)VirtualAlloc(nullptr, total, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  npu_shm_bytes_ = total;
  if (!npu_shm_ || npu::import_memory(zrt::ctx, npu_shm_, total) != npu_shm_ ||
      npu::import_memory(npu_->ctx, npu_shm_, total) != npu_shm_)
    throw std::runtime_error("ARC_NPU: cannot map the shared memory into the GPU's and the NPU's contexts");
  auto view = [&](size_t off, size_t bytes) {
    ZMem* m = new ZMem{(char*)npu_shm_ + off, bytes, false};
    zrt::mems.insert(m);
    mems_.push_back(m);
    return (cl_mem)m;
  };
  npu_gaq_ = view(npu_a_, (size_t)M * E);
  npu_gsa_ = view(npu_sa_, (size_t)M * 2);
  npu_gcg_ = view(npu_cg_, (size_t)M * npu_nn_ * 2);
  npu_gcu_ = view(npu_cu_, (size_t)M * npu_nn_ * 2);
  npu_ga16_ = view(npu_a16_, (size_t)M * std::max(F, E) * 2);
  npu_gcd_ = view(npu_cd_, (size_t)M * std::max(nd, 64) * 2);
  npu_gcq_ = view(npu_cq_, (size_t)M * std::max(nq, 64) * 2);
  npu_xn_ = buf((size_t)M * E * 4);
  static const char* src = R"CLC(
__kernel void npu_rowq(__global const float* x, int K, __global char* q, __global half* s) {
  const int r = get_group_id(0), l = get_local_id(0);
  __local float red[256];
  float m = 0.0f;
  for (int k = l; k < K; k += 256) m = fmax(m, fabs(x[(size_t)r * K + k]));
  red[l] = m;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = 128; o > 0; o >>= 1) {
    if (l < o) red[l] = fmax(red[l], red[l + o]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float amax = red[0], inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  for (int k = l; k < K; k += 256) q[(size_t)r * K + k] = (char)clamp((int)rint(x[(size_t)r * K + k] * inv), -127, 127);
  if (l == 0) vstore_half(amax, r, s);
}
// h[m][ng + j] = silu(gate) * up of the NPU's columns, as the GEMM's fused SwiGLU epilogue computes it
__kernel void npu_merge(__global const half* cg, __global const half* cu, int nn, __global float* h, int ldh, int ng) {
  const int j = get_global_id(0), m = get_global_id(1);
  const float g = vload_half((size_t)m * nn + j, cg), u = vload_half((size_t)m * nn + j, cu);
  h[(size_t)m * ldh + ng + j] = u * (g / (1.0f + exp(-g)));
}
// the f16 copy of rows of f32 (the NPU's f16 input)
__kernel void npu_to16(__global const float* x, int ldx, int K, __global half* y) {
  const int k = get_global_id(0), m = get_global_id(1);
  if (k < K) vstore_half(x[(size_t)m * ldx + k], (size_t)m * K + k, y);
}
// RMS norm of rows (rmsnorm_quant's formula: x * rsqrt(mean(x^2) + eps) * w, padding rows 0) written as f16
__kernel void npu_norm16(__global const float* x, __global const float* w, int K, int M_valid, float eps, __global half* y) {
  const int m = get_group_id(0), l = get_local_id(0);
  __local float red[256];
  float ss = 0.0f;
  if (m < M_valid)
    for (int k = l; k < K; k += 256) ss += x[(size_t)m * K + k] * x[(size_t)m * K + k];
  red[l] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = 128; o > 0; o >>= 1) {
    if (l < o) red[l] += red[l + o];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float r = rsqrt(red[0] / K + eps);
  for (int k = l; k < K; k += 256) vstore_half(m < M_valid ? x[(size_t)m * K + k] * r * w[k] : 0.0f, (size_t)m * K + k, y);
}
// out[m][coff + j] (+)= c16[m][j]: the NPU's f16 columns into an f32 result
__kernel void npu_put(__global const half* c16, int nn, __global float* out, int ldo, int coff, int add) {
  const int j = get_global_id(0), m = get_global_id(1);
  const float v = vload_half((size_t)m * nn + j, c16);
  __global float* o = out + (size_t)m * ldo + coff + j;
  *o = add ? *o + v : v;
}
)CLC";
  cl_program pr = cl_.build(src, "-cl-std=CL3.0 -cl-mad-enable");
  progs_.push_back(pr);
  npu_rowq_ = cl_.kernel(pr, "npu_rowq");
  npu_merge_ = cl_.kernel(pr, "npu_merge");
  npu_to16_ = cl_.kernel(pr, "npu_to16");
  npu_norm16_ = cl_.kernel(pr, "npu_norm16");
  npu_put_ = cl_.kernel(pr, "npu_put");
  // per layer: the GPU's weight copies and the NPU's int8 rows (dequantized from the GGUF on the CPU); the MTP block
  // (index n_layer, blk.<n_layer>.) too: its prompt pass (mtp_prefill) runs the same FFN over every chunk
  npu_layers_.resize(layers_.size() + 1);
  size_t nbytes = 0;
  std::vector<Layer*> all;
  for (auto& L : layers_) all.push_back(&L);
  if (mtp_block_) all.push_back(&mtp_);
  for (Layer* Lp : all) {
    Layer& L = *Lp;
    if ((size_t)L.il >= npu_layers_.size()) continue;
    NpuLayer& N = npu_layers_[L.il];
    // (the int8 share of a dense FFN's gate / up; MoE layers have none: their attention projections only)
    const bool gu = npu_nn_ > 0 && L.gate.raw && L.up.raw && L.gate.N == F && L.up.N == F && L.gate.K == E;
    if (gu) {
    N.gate = L.gate;
    N.up = L.up;
    N.gate.N = N.up.N = npu_ng_;
    for (int t = 0; t < 2; ++t) {
      const GgufTensor& tt = f_.get("blk." + std::to_string(L.il) + (t == 0 ? ".ffn_gate.weight" : ".ffn_up.weight"));
      const size_t rb = tt.nbytes / tt.dims[1];
      N.bq[t] = (int8_t*)npu_->host((size_t)npu_nn_ * E);
      N.bs[t] = (uint16_t*)npu_->host((size_t)npu_nn_ * 2);
      if (!N.bq[t] || !N.bs[t]) throw std::runtime_error("ARC_NPU: out of memory for the NPU's weight rows");
      nbytes += (size_t)npu_nn_ * (E + 2);
      int8_t* bq = N.bq[t];
      uint16_t* bs = N.bs[t];
#pragma omp parallel
      {
        std::vector<float> w(E);
#pragma omp for schedule(dynamic, 64)
        for (int r = 0; r < npu_nn_; ++r) {
          dequant_row(tt.type, tt.data + (size_t)(npu_ng_ + r) * rb, E, w.data());
          float amax = 0;
          for (int k = 0; k < E; ++k) amax = std::max(amax, std::fabs(w[k]));
          const float inv = amax > 0 ? 127.0f / amax : 0.0f;
          for (int k = 0; k < E; ++k) bq[(size_t)r * E + k] = (int8_t)std::clamp((int)std::lrint(w[k] * inv), -127, 127);
          bs[r] = npu::f2h(amax);
        }
      }
    }
    N.on = true;
    }
    // the f16 shares: the last nn rows of the matrix as int8 with one f16 scale per 32 values (w8g)
    auto w8g = [&](NpuMat& m, const GpuWeight& gw, const std::string& tname, int nn) {
      if (!nn || !gw.raw || gw.K % 32 || gw.N <= nn) return;
      const GgufTensor& tt = f_.get("blk." + std::to_string(L.il) + "." + tname);
      const size_t rb = tt.nbytes / tt.dims[1];
      const int K = gw.K, ng = gw.N - nn;
      m.gpu = gw;
      m.gpu.N = ng;
      m.ng = ng;
      m.nn = nn;
      m.K = K;
      m.bq = (int8_t*)npu_->host((size_t)nn * K);
      m.bs = (uint16_t*)npu_->host((size_t)nn * (K / 32) * 2);
      if (!m.bq || !m.bs) throw std::runtime_error("ARC_NPU: out of memory for the NPU's f16-part weight rows");
      nbytes += (size_t)nn * K + (size_t)nn * (K / 32) * 2;
      int8_t* bq = m.bq;
      uint16_t* bs = m.bs;
#pragma omp parallel
      {
        std::vector<float> w(K);
#pragma omp for schedule(dynamic, 64)
        for (int r = 0; r < nn; ++r) {
          dequant_row(tt.type, tt.data + (size_t)(ng + r) * rb, K, w.data());
          for (int b = 0; b < K / 32; ++b) {
            float amax = 0;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(w[b * 32 + i]));
            const uint16_t sh = npu::f2h(amax / 127.0f);
            const float s = npu::h2f(sh), inv = s > 0 ? 1.0f / s : 0.0f;
            bs[(size_t)r * (K / 32) + b] = sh;
            for (int i = 0; i < 32; ++i)
              bq[(size_t)r * K + b * 32 + i] = (int8_t)std::clamp((int)std::lrint(w[b * 32 + i] * inv), -127, 127);
          }
        }
      }
      m.on = true;
    };
    if (!hp.moe && L.down.N == E && L.down.K == F) w8g(N.down, L.down, "ffn_down.weight", nd);  // (MoE: layer_moe)
    if (L.il < hp.n_layer && hp.recurrent(L.il) && L.qkv.N == Cq && L.qkv.K == E) w8g(N.qkv, L.qkv, "attn_qkv.weight", nq);
  }
  fprintf(stderr, "ARC_NPU: FFN gate / up of %zu layers split GPU %d + NPU %d columns (chunks of >= %d rows), NPU weights "
                  "%.2f GiB, set up in %.1f s\n",
          (size_t)std::count_if(npu_layers_.begin(), npu_layers_.end(), [](const NpuLayer& n) { return n.on; }), npu_ng_,
          npu_nn_, npu_min_, nbytes / 1073741824.0,
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  fprintf(stderr, "ARC_NPU: f16 shares (w8g): ffn_down %d of %d columns, attn_qkv %d of %d (ARC_NPU_DOWN / ARC_NPU_QKV)\n", nd, E,
          nq, Cq);
  // (this memory is system RAM the GPU's budget (--ctx auto, the expert cache) does not count: on an iGPU the same
  //  RAM as the GPU share; with a tight share, lower the context or the shares)
  fprintf(stderr, "ARC_NPU: %.2f GiB of system RAM outside the GPU budget (NPU weight rows %.2f GiB, shared block %.2f GiB)\n",
          (nbytes + npu_shm_bytes_) / 1073741824.0, nbytes / 1073741824.0, npu_shm_bytes_ / 1073741824.0);
}

// the f16 (w8g) graph of kind 0 (ffn_down) or 1 (attn_qkv) for chunks of Mp rows: C[Mp][nn] = A16 . dq(B)^T
ze_graph_handle_t npu_graph16(int kind, int Mp, const NpuMat& m) {
  auto it = npu_graphs16_.find({kind, Mp});
  if (it != npu_graphs16_.end()) return it->second;
  npu::Ir ir;
  const int K = m.K, nn = m.nn;
  const auto A = ir.param("A", {"f16", {Mp, K}});
  const auto qb = ir.param("B", {"i8", {nn, K}});
  const auto sb = ir.param("SB", {"f16", {nn, K / 32, 1}});
  const auto b3 = ir.convert("B_f", ir.reshape("B_3", qb, {nn, K / 32, 32}), "f16");
  const auto B = ir.reshape("B_dq", ir.multiply("B_s", b3, sb), {nn, K});
  ir.result(ir.matmul_tb("C", A, B), "C_out");
  double ms = 0;
  bool cached = false;
  ze_graph_handle_t gr = npu_->compile(ir, ms, cached);
  if (!gr) throw std::runtime_error("ARC_NPU: the NPU's f16 graph did not compile");
  const std::vector<int> outs = npu_->outputs(gr);
  if (outs.size() != 1 || !npu_->set(gr, "A", npu_shm_ + npu_a16_) ||
      npu_->g->pfnSetArgumentValue(gr, outs[0], npu_shm_ + (kind == 0 ? npu_cd_ : npu_cq_)) != 0 || !npu_->set(gr, "B", m.bq) ||
      !npu_->set(gr, "SB", m.bs))
    throw std::runtime_error("ARC_NPU: cannot set the NPU f16 graph's arguments");
  if (!npu_->initialize(gr)) throw std::runtime_error("ARC_NPU: NPU f16 graph initialize failed");
  fprintf(stderr, "ARC_NPU: f16 graph %s for %d rows %s in %.0f ms\n", kind == 0 ? "ffn_down" : "attn_qkv", Mp,
          cached ? "loaded" : "compiled", ms);
  return npu_graphs16_[{kind, Mp}] = gr;
}
// starts matrix m's NPU share on the f16 rows already in A16 (the GPU's writes finished first)
void npu_start16(int kind, int Mp, NpuMat& m) {
  ze_graph_handle_t gr = npu_graph16(kind, npu_height(Mp), m);
  clFinish(cl_.queue);
  if (!npu_->set(gr, "B", m.bq) || !npu_->set(gr, "SB", m.bs) || !npu_->start(gr)) throw std::runtime_error("ARC_NPU: NPU start failed");
}
// waits for it and writes (add: adds) its columns into out[Mp][ldo] at column m.ng
void npu_finish16(int kind, int Mp, const NpuMat& m, cl_mem out, int ldo, int add) {
  const auto tw = std::chrono::steady_clock::now();
  if (!npu_->wait()) throw std::runtime_error("ARC_NPU: NPU run failed");
  npu_wait_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count();
  ++npu_runs_;
  int nn = m.nn, ld = ldo, coff = m.ng, ad = add;
  set_args(npu_put_, 0, kind == 0 ? npu_gcd_ : npu_gcq_, nn, out, ld, coff, ad);
  size_t gws[2] = {(size_t)m.nn, (size_t)Mp};
  run("npu_put", npu_put_, 2, gws, nullptr);
}
// attn_qkv of a DeltaNet layer with the NPU's share (false: not here, layer_gdn's own matmul); the z and ab GEMMs of
// the layer run on the GPU while the NPU works, then `finish` writes its columns before the conv reads them
bool npu_qkv_start(Layer& L, int T, int Mp) {
  if (!npu_ || cur_T_ <= 8 || Mp < npu_min_ || q2mm_ || (size_t)L.il >= npu_layers_.size() || !npu_layers_[L.il].qkv.on)
    return false;
  NpuMat& m = npu_layers_[L.il].qkv;
  {
    int K = hp.n_embd, mv = T;
    float eps = hp.eps;
    set_args(npu_norm16_, 0, x_, L.attn_norm, K, mv, eps, npu_ga16_);
    size_t gws[1] = {(size_t)Mp * 256}, lws[1] = {256};
    run("npu_norm16", npu_norm16_, 1, gws, lws);
  }
  npu_start16(1, Mp, m);
  matmul("gemm_qkv", m.gpu, Mp, big_, hp.conv_dim(), 0, false, act32_for(false));
  return true;
}
void npu_qkv_finish(Layer& L, int Mp) { npu_finish16(1, Mp, npu_layers_[L.il].qkv, big_, hp.conv_dim(), 0); }

// the height of the graphs for a chunk of Mp rows: a multiple of 1024 (the last chunk of a prompt has any height; one
// graph per height, so a few heights instead of one per length; the rows past Mp are computed and never merged)
int npu_height(int Mp) const { return std::min((Mp + 1023) / 1024 * 1024, max_chunk); }
// the NPU graph for chunks of Mp rows (compiled once, later loaded from the blob cache), its fixed arguments set
ze_graph_handle_t npu_graph(int Mp) {
  auto it = npu_graphs_.find(Mp);
  if (it != npu_graphs_.end()) return it->second;
  const int E = hp.n_embd, Nn = npu_nn_;
  npu::Ir ir;
  const auto ilo = ir.const_f16("ilo", {1, 1}, {-127.0f}), ihi = ir.const_f16("ihi", {1, 1}, {127.0f});
  const auto olo = ir.const_f16("olo", {1, 1}, {-1.0f}), ohi = ir.const_f16("ohi", {1, 1}, {1.0f});
  auto fq = [&](const char* nm, int rows) {
    const auto pin = ir.param(nm, {"i8", {rows, E}});
    return ir.fake_quantize(std::string(nm) + "_fq", ir.convert(std::string(nm) + "_f", pin, "f16"), ilo, ihi, olo, ohi, 255);
  };
  const auto A = fq("A", Mp);
  const auto SA = ir.param("SA", {"f16", {Mp, 1}});
  for (int t = 0; t < 2; ++t) {
    const std::string nm = t == 0 ? "G" : "U";
    const auto B = fq(t == 0 ? "BG" : "BU", Nn);
    const auto SB = ir.param(t == 0 ? "SBG" : "SBU", {"f16", {1, Nn}});
    // (the weight row's scale first, then the activation row's: the f16 product of the [-1, 1] values times a large
    //  activation absmax first could pass f16's 65504 on outlier rows)
    ir.result(ir.multiply(nm + "_sa", ir.multiply(nm + "_sb", ir.matmul_tb(nm, A, B), SB), SA), nm + "_out");
  }
  double ms = 0;
  bool cached = false;
  ze_graph_handle_t gr = npu_->compile(ir, ms, cached);
  if (!gr) throw std::runtime_error("ARC_NPU: the NPU graph did not compile");
  // (the outputs by their names, G_... / U_...; by their order only when the driver reports other names)
  const std::vector<int> outs = npu_->outputs(gr);
  const int og = npu_->output_named(gr, "G"), ou = npu_->output_named(gr, "U");
  const bool named = og >= 0 && ou >= 0 && og != ou;
  if (outs.size() != 2 || !npu_->set(gr, "A", npu_shm_ + npu_a_) || !npu_->set(gr, "SA", npu_shm_ + npu_sa_) ||
      npu_->g->pfnSetArgumentValue(gr, named ? og : outs[0], npu_shm_ + npu_cg_) != 0 ||
      npu_->g->pfnSetArgumentValue(gr, named ? ou : outs[1], npu_shm_ + npu_cu_) != 0)
    throw std::runtime_error("ARC_NPU: cannot set the NPU graph's arguments");
  // (initialize needs every argument: the first layer's weights, set again per layer before each run)
  for (const NpuLayer& N : npu_layers_)
    if (N.on) {
      npu_->set(gr, "BG", N.bq[0]);
      npu_->set(gr, "BU", N.bq[1]);
      npu_->set(gr, "SBG", N.bs[0]);
      npu_->set(gr, "SBU", N.bs[1]);
      break;
    }
  if (!npu_->initialize(gr)) throw std::runtime_error("ARC_NPU: NPU graph initialize failed");
  fprintf(stderr, "ARC_NPU: graph for %d rows %s in %.0f ms\n", Mp, cached ? "loaded" : "compiled", ms);
  return npu_graphs_[Mp] = gr;
}

// The FFN of a prompt chunk with the NPU's share; false: not here (off, another model, a short chunk), layer_ffn's own
bool npu_ffn(Layer& L, int T, int Mp) {
  if (!npu_ || cur_T_ <= 8 || Mp < npu_min_ || q2mm_ || hp.n_embd > 8192 || (size_t)L.il >= npu_layers_.size() ||
      (!npu_layers_[L.il].on && !npu_layers_[L.il].down.on))
    return false;
  NpuLayer& N = npu_layers_[L.il];
  const int E = hp.n_embd, F = hp.n_ff;
  const int a_in = act32_for(false);
  if (!N.on) {  // (no gate / up share, ARC_NPU_SHARE=0: layer_ffn's gate / up; the down share below)
    rmsnorm_quant(x_, 0, L.post_norm, E, Mp, T, a_in);
    matmul("gemm_gate", L.gate, Mp, big_, F, 0, 0, a_in);
    matmul("gemm_up", L.up, Mp, big_, F, 0, 2, a_in);
  } else {
  {  // the FFN norm: int8 per 32 for the GPU's GEMMs as always, and its f32 rows for the NPU's quantization
    float eps = hp.eps;
    int r0 = 0, EE = E, MM = Mp, TT = T, wh = 1, a32 = a_in;
    set_args(k_rmsnorm_, 0, x_, r0, L.post_norm, EE, MM, TT, eps, aq_, asc_, absum_, npu_xn_, wh, a32);
    size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
    run("rmsnorm_quant", k_rmsnorm_, 2, gws, lws);
  }
  {
    int KK = E;
    set_args(npu_rowq_, 0, npu_xn_, KK, npu_gaq_, npu_gsa_);
    size_t gws[1] = {(size_t)Mp * 256}, lws[1] = {256};
    run("npu_rowq", npu_rowq_, 1, gws, lws);
  }
  ze_graph_handle_t gr = npu_graph(npu_height(Mp));
  clFinish(cl_.queue);  // (the int8 rows written before the NPU reads them)
  if (!npu_->set(gr, "BG", N.bq[0]) || !npu_->set(gr, "BU", N.bq[1]) || !npu_->set(gr, "SBG", N.bs[0]) ||
      !npu_->set(gr, "SBU", N.bs[1]) || !npu_->start(gr))
    throw std::runtime_error("ARC_NPU: NPU start failed");
  matmul("gemm_gate", N.gate, Mp, big_, F, 0, 0, a_in);
  matmul("gemm_up", N.up, Mp, big_, F, 0, 2, a_in);
  const auto tw = std::chrono::steady_clock::now();
  if (!npu_->wait()) throw std::runtime_error("ARC_NPU: NPU run failed");
  npu_wait_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count();
  ++npu_runs_;
  {
    int nn = npu_nn_, ld = F, ng = npu_ng_;
    set_args(npu_merge_, 0, npu_gcg_, npu_gcu_, nn, big_, ld, ng);
    size_t gws[2] = {(size_t)npu_nn_, (size_t)Mp};
    run("npu_merge", npu_merge_, 2, gws, nullptr);
  }
  }
  const int a_dn = act32_for(false);
  if (N.down.on) {  // (the NPU's f16 share of ffn_down: silu(gate) * up as f16 rows, started before the GPU's share)
    int ldx = F, KK = F;
    set_args(npu_to16_, 0, big_, ldx, KK, npu_ga16_);
    size_t gws[2] = {(size_t)F, (size_t)Mp};
    run("npu_to16", npu_to16_, 2, gws, nullptr);
    npu_start16(0, Mp, N.down);
  }
  {
    int gelu = 0, fused = 1;
    set_args(k_swiglu_, 0, big_, F, Mp, T, aq_, asc_, absum_, a_dn, fused, gelu);
    size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
    run("swiglu_quant", k_swiglu_, 2, gws, lws);
  }
  if (N.down.on) {
    matmul("gemm_down", N.down.gpu, Mp, x_, E, 0, true, a_dn);
    npu_finish16(0, Mp, N.down, x_, E, 1);
  } else {
    matmul("gemm_down", L.down, Mp, x_, E, 0, true, a_dn);
  }
  return true;
}

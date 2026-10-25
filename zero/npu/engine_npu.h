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

struct NpuLayer {
  GpuWeight gate, up;              // (the GPU's columns: copies with N = its share; persistent, the staging keys on them)
  int8_t* bq[2] = {nullptr, nullptr};  // (the NPU's columns of gate / up: int8 rows)
  uint16_t* bs[2] = {nullptr, nullptr};  // (their f16 row scales, absmax)
  bool on = false;
};
std::unique_ptr<npu::Rt> npu_;
std::vector<NpuLayer> npu_layers_;
std::map<int, ze_graph_handle_t> npu_graphs_;  // (by chunk height)
int npu_ng_ = 0, npu_nn_ = 0, npu_min_ = 2048;
uint8_t* npu_shm_ = nullptr;
size_t npu_a_ = 0, npu_sa_ = 0, npu_cg_ = 0, npu_cu_ = 0;
cl_mem npu_xn_ = nullptr, npu_gaq_ = nullptr, npu_gsa_ = nullptr, npu_gcg_ = nullptr, npu_gcu_ = nullptr;
cl_kernel npu_rowq_ = nullptr, npu_merge_ = nullptr;
double npu_wait_ms_ = 0;
int npu_runs_ = 0;

void npu_setup() {
  if (!npu_wanted() || opt.dry) return;
  if (hp.moe || hp.gemma || hp.hc || hp.arch != "qwen35") {
    fprintf(stderr, "ARC_NPU: Qwen dense models only; off for this one\n");
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
  npu_nn_ = std::clamp((int)std::lround(F * share / 64.0) * 64, 64, F - 64);
  npu_ng_ = F - npu_nn_;
  // shared memory: int8 rows, their scales, the NPU's two f16 outputs, for max_chunk rows
  npu_a_ = 0;
  npu_sa_ = (size_t)M * E;
  npu_cg_ = (npu_sa_ + (size_t)M * 2 + 4095) / 4096 * 4096;
  npu_cu_ = npu_cg_ + (size_t)M * npu_nn_ * 2;
  const size_t total = (npu_cu_ + (size_t)M * npu_nn_ * 2 + 65535) / 65536 * 65536;
  npu_shm_ = (uint8_t*)VirtualAlloc(nullptr, total, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
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
)CLC";
  cl_program pr = cl_.build(src, "-cl-std=CL3.0 -cl-mad-enable");
  progs_.push_back(pr);
  npu_rowq_ = cl_.kernel(pr, "npu_rowq");
  npu_merge_ = cl_.kernel(pr, "npu_merge");
  // per layer: the GPU's weight copies and the NPU's int8 rows (dequantized from the GGUF on the CPU); the MTP block
  // (index n_layer, blk.<n_layer>.) too: its prompt pass (mtp_prefill) runs the same FFN over every chunk
  npu_layers_.resize(layers_.size() + 1);
  size_t nbytes = 0;
  std::vector<Layer*> all;
  for (auto& L : layers_) all.push_back(&L);
  if (mtp_block_) all.push_back(&mtp_);
  for (Layer* Lp : all) {
    Layer& L = *Lp;
    if ((size_t)L.il >= npu_layers_.size() || !L.gate.raw || !L.up.raw || L.gate.N != F || L.up.N != F || L.gate.K != E) continue;
    NpuLayer& N = npu_layers_[L.il];
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
  fprintf(stderr, "ARC_NPU: FFN gate / up of %zu layers split GPU %d + NPU %d columns (chunks of >= %d rows), NPU weights "
                  "%.2f GiB, set up in %.1f s\n",
          (size_t)std::count_if(npu_layers_.begin(), npu_layers_.end(), [](const NpuLayer& n) { return n.on; }), npu_ng_,
          npu_nn_, npu_min_, nbytes / 1073741824.0,
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

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
    ir.result(ir.multiply(nm + "_sb", ir.multiply(nm + "_sa", ir.matmul_tb(nm, A, B), SA), SB), nm + "_out");
  }
  double ms = 0;
  bool cached = false;
  ze_graph_handle_t gr = npu_->compile(ir, ms, cached);
  if (!gr) throw std::runtime_error("ARC_NPU: the NPU graph did not compile");
  const std::vector<int> outs = npu_->outputs(gr);
  if (outs.size() != 2 || !npu_->set(gr, "A", npu_shm_ + npu_a_) || !npu_->set(gr, "SA", npu_shm_ + npu_sa_) ||
      npu_->g->pfnSetArgumentValue(gr, outs[0], npu_shm_ + npu_cg_) != 0 ||
      npu_->g->pfnSetArgumentValue(gr, outs[1], npu_shm_ + npu_cu_) != 0)
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
      !npu_layers_[L.il].on)
    return false;
  NpuLayer& N = npu_layers_[L.il];
  const int E = hp.n_embd, F = hp.n_ff;
  const int a_in = act32_for(false);
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
  ze_graph_handle_t gr = npu_graph(Mp);
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
  const int a_dn = act32_for(false);
  {
    int gelu = 0, fused = 1;
    set_args(k_swiglu_, 0, big_, F, Mp, T, aq_, asc_, absum_, a_dn, fused, gelu);
    size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
    run("swiglu_quant", k_swiglu_, 2, gws, lws);
  }
  matmul("gemm_down", L.down, Mp, x_, E, 0, true, a_dn);
  return true;
}

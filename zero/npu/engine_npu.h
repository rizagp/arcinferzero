// Included inside class Engine (zero/engine.h): the NPU next to the GPU (zero/npu/), opt-in with --npu on
// (EngineOptions::npu; or ARC_NPU=1), off by default (every function here returns at once then). Qwen dense models only: in prompt chunks of at
// least ARC_NPU_MIN rows (default 2048) the NPU computes the last share of the FFN's gate and up columns
// (ARC_NPU_SHARE, default 1/3) while the GPU computes the rest:
//   - the FFN norm writes its f32 rows once more (rmsnorm_quant with its f32 output on), a small kernel quantizes them
//     per row into memory both Level Zero contexts map (external memmap, zero/npu/npu_rt.h);
//   - the NPU runs the "fqi" graph (int8 rows and weight rows, FakeQuantize with constant bounds, then the row and
//     column scales: exact per-row int8 on its int8 units), one graph per chunk height (blob cache), the layer's
//     weights as its inputs (per-row int8, decoded per layer by the GPU from its W8 form into the shared memory);
//   - the GPU meanwhile runs gate / up on its columns (the same GEMMs and fused SwiGLU epilogue, on copies of the
//     weights with N = its share), waits for the NPU, and writes silu(gate) * up of the NPU's columns into big_.
// KL against f32 (tools/ref_forward REF_NPU, 27B-AP, 128 tokens): 1/3 of gate / up 0.0022 against 0.00046 without.

int npu_level() const {
  if (opt.npu >= 0) return opt.npu;
  const char* e = getenv("ARC_NPU");
  return e ? atoi(e) : 0;
}
bool npu_wanted() const { return npu_level() > 0; }
// (before the GPU runtime: it initializes Level Zero for GPUs only, which would hide the NPU driver)
void npu_preinit() {
  if (npu_wanted() && ze_load()) zeInit(0);
}

// A matrix whose last columns the NPU computes in f16 ("w8g": its weight rows int8 with one f16 scale per 32
// values, dequantized in the graph; the activations f16): ffn_down and the DeltaNet layers' attn_qkv
// (the NPU's weight rows are not kept: per layer the GPU decodes them from its own form into the shared block, npu_rows)
struct NpuMat {
  GpuWeight gpu;                // (the GPU's columns: a persistent copy with N = its share)
  const GpuWeight* src = nullptr;  // (the whole matrix: the NPU's rows [ng, ng + nn) come from it)
  int ng = 0, nn = 0, K = 0;
  bool on = false;
};
struct NpuLayer {
  GpuWeight gate, up;              // (the GPU's columns: copies with N = its share; persistent, the staging keys on them)
  const GpuWeight* src[2] = {nullptr, nullptr};  // (the whole gate / up matrices)
  bool on = false;
  NpuMat down, qkv;
};
// the NPU's weight rows in the shared block, one region per kind (0 gate, 1 up, 2 ffn_down, 3 attn_qkv): int8 rows and
// their f16 scales (per row for gate / up, per 32 values for down / qkv), the layer they hold (-1: none)
size_t npu_wq_[4] = {}, npu_ws_[4] = {};
cl_mem npu_gwq_[4] = {}, npu_gws_[4] = {};
int npu_own_[4] = {-1, -1, -1, -1};
cl_mem npu_stq_ = nullptr, npu_sth_ = nullptr;  // (GPU staging: the W8 decode of the NPU's rows, just-in-time models)
cl_kernel npu_w8rows_ = nullptr;
std::unique_ptr<npu::Rt> npu_;
std::vector<NpuLayer> npu_layers_;
std::map<int, ze_graph_handle_t> npu_graphs_;  // (by chunk height)
int npu_ng_ = 0, npu_nn_ = 0, npu_min_ = 2048;
uint8_t* npu_shm_ = nullptr;
size_t npu_a_ = 0, npu_sa_ = 0, npu_cg_ = 0, npu_cu_ = 0;
cl_mem npu_xn_ = nullptr, npu_gaq_ = nullptr, npu_gsa_ = nullptr, npu_gcg_ = nullptr, npu_gcu_ = nullptr;
cl_kernel npu_rowq_ = nullptr, npu_merge_ = nullptr, npu_to16_ = nullptr, npu_norm16_ = nullptr, npu_put_ = nullptr, npu_gelu16_ = nullptr;
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
  clFinish(cl_.queue);  // (the GPU done with the shared block and the NPU kernels)
  npu_->wait();
  if (npu_shm_) {
    zeMemFree(npu_->ctx, npu_shm_);
    zeMemFree(zrt::ctx, npu_shm_);
  }
  npu_graphs_.clear();
  npu_graphs16_.clear();
  npu_layers_.clear();
  for (int& o : npu_own_) o = -1;
  for (cl_kernel* k : {&npu_rowq_, &npu_merge_, &npu_to16_, &npu_norm16_, &npu_put_, &npu_w8rows_, &npu_gelu16_})
    if (*k) {
      clReleaseKernel(*k);
      *k = nullptr;
    }
  npu_.reset();
  if (npu_shm_) VirtualFree(npu_shm_, 0, MEM_RELEASE);
  npu_shm_ = nullptr;
}

void npu_setup() {
  if (!npu_wanted() || opt.dry) return;
  // (Qwen3.5 / 3.6 / 3.8 dense and MoE, Flash Next (its attn_qkv shares), Gemma 4 (its FFN, f16))
  if (!hp.gemma && !hp.hc && hp.arch != "qwen35" && hp.arch != "qwen35moe") {
    fprintf(stderr, "ARC_NPU: Qwen3.5 / 3.6 / 3.8 and Gemma 4 models only; off for this one\n");
    return;
  }
  // (measured 2026-10-25: the NPU's share pays off where the GPU decodes the weights per chunk (W8 just in time: the
  //  27B, +15-20 % prompt); with resident W8 the GPU's GEMMs are faster than the NPU makes its share (4B -6 %, 9B
  //  +4-6 % at KL 0.008-0.013, E4B 0 %), and MoE models' few dense matrices do not pay for the NPU's overhead (35B -5 %,
  //  Gemma 4 26B -1-8 %): off there unless --npu force. On: Qwen3.8 27B +15 %, Gemma 4 12B +10 %)
  if ((w8_resident || hp.moe) && npu_level() < 2) {
    fprintf(stderr, "ARC_NPU: off for this model (%s: no gain measured); --npu force uses it anyway\n",
            hp.moe ? "a mixture of experts" : "its W8 weights stay resident");
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
  // (gate / up: dense FFNs only (Qwen MoE layers have none; Gemma 4 26B its dense MLP); ARC_NPU_SHARE=0 turns that part
  //  off. Gemma: f16 like down (its KL is sensitive: two int8 passes on the GPU), a smaller share by default)
  const double gdef = q2mm_ ? 0.4 : 0.27;  // (Gemma 4: the f16 shares of gate / up and down, by default)
  const double gshare = getenv("ARC_NPU_SHARE") ? share : hp.gemma ? gdef : share;
  npu_nn_ = F < 256 || gshare <= 0 ? 0 : std::clamp((int)std::lround(F * gshare / 64.0) * 64, 64, F - 64);
  npu_ng_ = F - npu_nn_;
  // the f16 shares (ARC_NPU_DOWN / ARC_NPU_QKV, 0: off): about where the NPU (w8g, ~2.9-3.1 TFLOPS at 4096 rows) and
  // the GPU finish together on the 27B
  auto cols = [](int N, double s) {
    return s <= 0 || N < 256 ? 0 : std::clamp((int)std::lround(N * s / 64.0) * 64, 64, N - 64);
  };
  const int Cq = hp.gemma ? 0 : hp.conv_dim();  // (Gemma: no DeltaNet layers)
  const int nd = cols(E, getenv("ARC_NPU_DOWN") ? atof(getenv("ARC_NPU_DOWN")) : hp.gemma ? gdef : 0.275);
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
  // then the weight regions (one layer's NPU rows per kind: gate, up, ffn_down, attn_qkv)
  size_t end = npu_cq_ + (size_t)M * std::max(nq, 64) * 2;
  const int wn[4] = {npu_nn_, npu_nn_, nd, nq}, wk[4] = {E, E, F, E};
  for (int t = 0; t < 4; ++t) {
    npu_wq_[t] = (end + 4095) / 4096 * 4096;
    npu_ws_[t] = (npu_wq_[t] + (size_t)wn[t] * wk[t] + 4095) / 4096 * 4096;
    end = npu_ws_[t] + (size_t)std::max(wn[t], 64) * (wk[t] / 32) * 2;
  }
  const size_t total = (end + 65535) / 65536 * 65536;
  npu_shm_ = (uint8_t*)VirtualAlloc(nullptr, total, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  npu_shm_bytes_ = total;
  const bool gm = npu_shm_ && npu::import_memory(zrt::ctx, npu_shm_, total) == npu_shm_;
  if (!gm || npu::import_memory(npu_->ctx, npu_shm_, total) != npu_shm_) {
    if (gm) zeMemFree(zrt::ctx, npu_shm_);
    if (npu_shm_) VirtualFree(npu_shm_, 0, MEM_RELEASE);
    npu_shm_ = nullptr;
    throw std::runtime_error("ARC_NPU: cannot map the shared memory into the GPU's and the NPU's contexts");
  }
  auto view = [&](size_t off, size_t bytes) {
    ZMem* m = new ZMem{(char*)npu_shm_ + off, bytes, false};
    zrt::D().mems.insert(m);
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
  for (int t = 0; t < 4; ++t) {
    npu_gwq_[t] = view(npu_wq_[t], std::max<size_t>((size_t)wn[t] * wk[t], 4096));
    npu_gws_[t] = view(npu_ws_[t], (size_t)std::max(wn[t], 64) * (wk[t] / 32) * 2);
  }
  npu_xn_ = buf((size_t)M * E * 4);
  static const char* src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// The NPU's weight rows from the GPU's W8 form (zero/kernels/gemm.cl, QTYPE >= 100: per 16-row group g and superblock
// s, the int8 words of chunk j at ((g * nsb + s) * 4 + j) * 256 + word * 16 + lane, word = (k & 63) >> 2, byte k & 3;
// the header at (g * nsb + s) * 160: d, dmin, the scale words (2 + j), the min words (6 + j); value d * sc * q -
// dmin * mn). One work-group per group of 16 rows, a superblock at a time through local memory, rows written whole.
// per_row: int8 with one scale per row, s[r] = f16(absmax) (the "fqi" graph); else one f16 scale per 32 values,
// s[r][b] = f16(absmax / 127) (the "w8g" graph). q8: the GEMM's Q8_0 form (Q8_0, Q4_0, IQ4_NL, Q2_0: the same words,
// a header of 8 f32 scales per superblock, one per 32-block, at (g * nsb + s) * 128 + b * 16; value d[b] * q).
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void npu_w8rows(__global const uint* wq, __global const uint* wh, int K, int sc32, int has_min, int per_row,
                __global char* oq, __global half* os, int q8) {
  const int g = get_group_id(0), t = get_local_id(0), nsb = K / 256;
  const int lane = t & 15, w = t >> 4;   // (loading: this thread's row in the group and word of each chunk)
  const int row = t >> 4, c = t & 15;    // (storing: its row and 16 values of the superblock)
  const size_t r = (size_t)g * 16 + row;
  __local float tile[16][260];
  __local float red[256];
  float rmax = 0.0f, inv = 0.0f;
  for (int pass = per_row ? 0 : 1; pass < 2; ++pass) {
    if (pass == 1 && per_row) {  // (the row's absmax from pass 0)
      red[t] = rmax;
      barrier(CLK_LOCAL_MEM_FENCE);
      float m = 0.0f;
      for (int i = 0; i < 16; ++i) m = fmax(m, red[row * 16 + i]);
      const half h = convert_half_rte(m);  // (quantized with the scale the NPU multiplies by, f16)
      inv = (float)h > 0.0f ? 127.0f / (float)h : 0.0f;
      if (c == 0) os[r] = h;
    }
    for (int s = 0; s < nsb; ++s) {
      __global const uint* hb = wh + ((size_t)g * nsb + s) * (q8 ? 128 : 160);
      const float d = as_float(hb[lane]), dmin = as_float(hb[16 + lane]);
      for (int j = 0; j < 4; ++j) {
        const uint qw = wq[(((size_t)g * nsb + s) * 4 + j) * 256 + w * 16 + lane];
        const float ds = q8 ? as_float(hb[(2 * j + (w >> 3)) * 16 + lane])
                            : d * (float)(char)(hb[(2 + j) * 16 + lane] >> (8 * (sc32 ? 2 * (w >> 3) : w >> 2)));
        const float mn = has_min && !q8 ? dmin * (float)((hb[(6 + j) * 16 + lane] >> (8 * (w >> 2))) & 255u) : 0.0f;
        for (int b = 0; b < 4; ++b) tile[lane][j * 64 + 4 * w + b] = ds * (float)(char)(qw >> (8 * b)) - mn;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (pass == 0) {
        for (int i = 0; i < 16; ++i) rmax = fmax(rmax, fabs(tile[row][c * 16 + i]));
      } else {
        float sinv = inv;
        if (!per_row) {  // (the scale of this 32-block, as the CPU form computed it: f16(absmax / 127))
          float m = 0.0f;
          for (int i = 0; i < 32; ++i) m = fmax(m, fabs(tile[row][(c >> 1) * 32 + i]));
          // (at least f16's smallest normal, 2^-14: a subnormal scale could be flushed to zero by the NPU; the values
          //  of such a small block then stay under 127)
          const half h = m > 0.0f ? convert_half_rte(fmax(m / 127.0f, 6.103515625e-05f)) : (half)0.0f;
          const float hs = (float)h;
          sinv = hs > 0.0f ? 1.0f / hs : 0.0f;
          if (!(c & 1)) os[r * (K / 32) + s * 8 + (c >> 1)] = h;
        }
        uint p[4] = {0u, 0u, 0u, 0u};
        for (int i = 0; i < 16; ++i)
          p[i >> 2] |= (uint)(uchar)(char)clamp((int)rint(tile[row][c * 16 + i] * sinv), -127, 127) << (8 * (i & 3));
        vstore4((uint4)(p[0], p[1], p[2], p[3]), (r * K + (size_t)s * 256 + c * 16) / 16, (__global uint*)oq);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}
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
  // (quantized with the f16 scale the NPU multiplies by)
  const half ha = convert_half_rte(red[0]);
  const float amax = (float)ha, inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  for (int k = l; k < K; k += 256) q[(size_t)r * K + k] = (char)clamp((int)rint(x[(size_t)r * K + k] * inv), -127, 127);
  if (l == 0) s[r] = ha;
}
// h[m][ng + j] = silu(gate) * up of the NPU's columns, as the GEMM's fused SwiGLU epilogue computes it
__kernel void npu_merge(__global const half* cg, __global const half* cu, int nn, __global float* h, int ldh, int ng) {
  const int j = get_global_id(0), m = get_global_id(1);
  const float g = clamp(vload_half((size_t)m * nn + j, cg), -65504.0f, 65504.0f),
              u = clamp(vload_half((size_t)m * nn + j, cu), -65504.0f, 65504.0f);  // (no inf)
  h[(size_t)m * ldh + ng + j] = u * (g / (1.0f + exp(-g)));
}
// the f16 copy of rows of f32 (the NPU's f16 input)
__kernel void npu_to16(__global const float* x, int ldx, int K, __global half* y) {
  const int k = get_global_id(0), m = get_global_id(1);
  if (k < K) vstore_half(clamp(x[(size_t)m * ldx + k], -65504.0f, 65504.0f), (size_t)m * K + k, y);
}
// gelu_tanh(gate) * up as f16 (Gemma 4: the rows of big_ are [gate (F) | up (F)], zero/kernels/ops.cl's formula), the
// NPU's f16 input of ffn_down
__kernel void npu_gelu16(__global const float* gu, int F, __global half* y) {
  const int k = get_global_id(0), m = get_global_id(1);
  const float g = gu[(size_t)m * 2 * F + k], u = gu[(size_t)m * 2 * F + F + k];
  const float v = 0.5f * g * (1.0f + tanh(0.7978845608f * (g + 0.044715f * g * g * g))) * u;
  vstore_half(clamp(v, -65504.0f, 65504.0f), (size_t)m * F + k, y);
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
  const float v = clamp(vload_half((size_t)m * nn + j, c16), -65504.0f, 65504.0f);  // (an f16 overflow: no inf)
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
  npu_w8rows_ = cl_.kernel(pr, "npu_w8rows");
  npu_gelu16_ = cl_.kernel(pr, "npu_gelu16");
  // per layer: the GPU's weight copies (N = its share) and where the NPU's rows come from; the MTP block (index
  // n_layer, blk.<n_layer>.) too: its prompt pass (mtp_prefill) runs the same FFN over every chunk. Only matrices in
  // the W8 form (gemm_qtype 100..103: the NPU's rows are decoded from it; Q8_0-form types stay on the GPU)
  npu_layers_.resize(layers_.size() + 1);
  // (the GEMM forms npu_w8rows reads: W8 (100..103) and Q8_0 (Q8_0, Q4_0, IQ4_NL, Q2_0))
  auto w8 = [](const GpuWeight& w) {
    return ((w.gemm_qtype >= 100 && w.gemm_qtype <= 103) || w.gemm_qtype == (int)GGML_Q8_0) && (w.wq || w.raw) &&
           w.K % 256 == 0 && w.N % 16 == 0;
  };
  size_t stq = 0, sth = 0;
  auto stage = [&](int nn, int K) {
    stq = std::max(stq, w8_bytes_q(nn, K));
    sth = std::max(sth, w8_bytes_h(nn, K));
  };
  std::vector<Layer*> all;
  for (auto& L : layers_) all.push_back(&L);
  if (mtp_block_) all.push_back(&mtp_);
  for (Layer* Lp : all) {
    Layer& L = *Lp;
    if ((size_t)L.il >= npu_layers_.size()) continue;
    NpuLayer& N = npu_layers_[L.il];
    // (the int8 share of a dense FFN's gate / up; MoE layers have none: their attention projections only)
    if (npu_nn_ > 0 && w8(L.gate) && w8(L.up) && L.gate.N == F && L.up.N == F && L.gate.K == E && L.up.K == E) {
      N.gate = L.gate;
      N.up = L.up;
      N.gate.N = N.up.N = npu_ng_;
      N.src[0] = &L.gate;
      N.src[1] = &L.up;
      N.on = true;
      if (!L.gate.wq) stage(npu_nn_, E);
    }
    // the f16 shares: the last nn rows of the matrix as int8 with one f16 scale per 32 values (w8g)
    auto w8g = [&](NpuMat& m, const GpuWeight& gw, int nn) {
      if (!nn || !w8(gw) || gw.N <= nn) return;
      m.gpu = gw;
      m.gpu.N = gw.N - nn;
      m.src = &gw;
      m.ng = gw.N - nn;
      m.nn = nn;
      m.K = gw.K;
      m.on = true;
      if (!gw.wq) stage(nn, gw.K);
    };
    // (dense FFNs and Gemma 4 26B's dense MLP; Qwen MoE layers have no ffn_down: layer_moe)
    if ((!hp.moe || hp.gemma) && L.down.N == E && L.down.K == F) w8g(N.down, L.down, nd);
    if (L.il < hp.n_layer && hp.recurrent(L.il) && L.qkv.N == Cq && L.qkv.K == E) w8g(N.qkv, L.qkv, nq);
  }
  if (stq) {
    npu_stq_ = buf(stq);
    npu_sth_ = buf(sth);
  }
  auto count = [&](auto pred) { return (int)std::count_if(npu_layers_.begin(), npu_layers_.end(), pred); };
  fprintf(stderr, "ARC_NPU: NPU shares (chunks of >= %d rows): ffn_gate / up %d of %d columns %s in %d layers, "
                  "ffn_down %d of %d f16 in %d, attn_qkv %d of %d f16 in %d (ARC_NPU_SHARE / _DOWN / _QKV); set up in %.1f s\n",
          npu_min_, npu_nn_, F, hp.gemma ? "f16" : "int8", count([](const NpuLayer& n) { return n.on; }), nd, E,
          count([](const NpuLayer& n) { return n.down.on; }), nq, Cq, count([](const NpuLayer& n) { return n.qkv.on; }),
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  // (the shared block is system RAM the GPU's budget (--ctx auto, the expert cache) does not count: on an iGPU the
  //  same RAM as the GPU share. The NPU's weight rows are decoded per layer by the GPU into it: no copy of them stays)
  fprintf(stderr, "ARC_NPU: %.2f GiB of system RAM outside the GPU budget (the shared block); %.0f MiB GPU staging\n",
          npu_shm_bytes_ / 1073741824.0, (stq + sth) / 1048576.0);
}

// the NPU's rows [ng, ng + nn) of matrix w (kind: its region, npu_wq_ / npu_ws_) for layer il, unless the region holds
// them already (layer-major prompt segments run one layer's chunks back to back): the W8 form of those rows (just in
// time: decode_w8 of their raw rows into the staging, as the GPU's own share; resident: a view of the W8 matrix), then
// npu_w8rows into the NPU's int8 form (per_row: gate / up; else per 32 values)
void npu_rows(int kind, int il, const GpuWeight& w, int ng, int nn, int per_row) {
  if (npu_own_[kind] == il) return;
  npu_own_[kind] = il;
  const int K = w.K, q8 = w.gemm_qtype == (int)GGML_Q8_0;
  cl_mem dq = npu_stq_, dh = npu_sth_;
  if (w.wq) {  // (header words per row and superblock: 10 in W8, 8 in the Q8_0 form)
    dq = from(w.wq, (size_t)ng * K);
    dh = from(w.wh, (size_t)ng * (K / 256) * (q8 ? 8 : 10) * 4);
  } else {
    decode_w8(from(w.raw, (size_t)ng * (K / 256) * raw_padded_sb_bytes(w.type)), w.type, nn, K, dq, dh);
  }
  int KK = K, sc32 = w.gemm_qtype & 1, hm = !q8 && w.gemm_qtype >= 102, pr = per_row, qq = q8;
  set_args(npu_w8rows_, 0, dq, dh, KK, sc32, hm, pr, npu_gwq_[kind], npu_gws_[kind], qq);
  size_t gws[1] = {(size_t)nn / 16 * 256}, lws[1] = {256};
  run("npu_w8rows", npu_w8rows_, 1, gws, lws);
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
      npu_->g->pfnSetArgumentValue(gr, outs[0], npu_shm_ + (kind == 0 ? npu_cd_ : npu_cq_)) != 0 ||
      !npu_->set(gr, "B", npu_shm_ + npu_wq_[kind + 2]) || !npu_->set(gr, "SB", npu_shm_ + npu_ws_[kind + 2]))
    throw std::runtime_error("ARC_NPU: cannot set the NPU f16 graph's arguments");
  if (!npu_->initialize(gr)) throw std::runtime_error("ARC_NPU: NPU f16 graph initialize failed");
  fprintf(stderr, "ARC_NPU: f16 graph %s for %d rows %s in %.0f ms\n", kind == 0 ? "ffn_down" : "attn_qkv", Mp,
          cached ? "loaded" : "compiled", ms);
  return npu_graphs16_[{kind, Mp}] = gr;
}
// Gemma 4's gate / up graph for chunks of Mp rows: the NPU's columns of both in f16 (w8g weights) on one f16 input,
// G[Mp][nn] = A16 . dq(BG)^T and U likewise, into the gate / up output blocks (cg / cu)
ze_graph_handle_t npu_graph_gu16(int Mp) {
  auto it = npu_graphs16_.find({2, Mp});
  if (it != npu_graphs16_.end()) return it->second;
  npu::Ir ir;
  const int K = hp.n_embd, nn = npu_nn_;
  const auto A = ir.param("A", {"f16", {Mp, K}});
  for (int t = 0; t < 2; ++t) {
    const std::string nm = t == 0 ? "G" : "U";
    const auto qb = ir.param("B" + nm, {"i8", {nn, K}});
    const auto sb = ir.param("SB" + nm, {"f16", {nn, K / 32, 1}});
    const auto b3 = ir.convert(nm + "_Bf", ir.reshape(nm + "_B3", qb, {nn, K / 32, 32}), "f16");
    const auto B = ir.reshape(nm + "_Bdq", ir.multiply(nm + "_Bs", b3, sb), {nn, K});
    ir.result(ir.matmul_tb(nm, A, B), nm + "_out");
  }
  double ms = 0;
  bool cached = false;
  ze_graph_handle_t gr = npu_->compile(ir, ms, cached);
  if (!gr) throw std::runtime_error("ARC_NPU: the NPU's f16 gate / up graph did not compile");
  const std::vector<int> outs = npu_->outputs(gr);
  const int og = npu_->output_named(gr, "G"), ou = npu_->output_named(gr, "U");
  const bool named = og >= 0 && ou >= 0 && og != ou;
  if (outs.size() != 2 || !npu_->set(gr, "A", npu_shm_ + npu_a16_) ||
      npu_->g->pfnSetArgumentValue(gr, named ? og : outs[0], npu_shm_ + npu_cg_) != 0 ||
      npu_->g->pfnSetArgumentValue(gr, named ? ou : outs[1], npu_shm_ + npu_cu_) != 0 ||
      !npu_->set(gr, "BG", npu_shm_ + npu_wq_[0]) || !npu_->set(gr, "BU", npu_shm_ + npu_wq_[1]) ||
      !npu_->set(gr, "SBG", npu_shm_ + npu_ws_[0]) || !npu_->set(gr, "SBU", npu_shm_ + npu_ws_[1]))
    throw std::runtime_error("ARC_NPU: cannot set the NPU f16 gate / up graph's arguments");
  if (!npu_->initialize(gr)) throw std::runtime_error("ARC_NPU: NPU f16 gate / up graph initialize failed");
  fprintf(stderr, "ARC_NPU: f16 graph ffn_gate / up for %d rows %s in %.0f ms\n", Mp, cached ? "loaded" : "compiled", ms);
  return npu_graphs16_[{2, Mp}] = gr;
}

// Gemma 4's FFN matmuls of a prompt chunk with the NPU's shares (gate / up and down in f16), after its norm; false:
// not here (layer_gemma_ffn's own). The GPU's share keeps its two int8 passes (q2mm_); the rest of the FFN (the 26B's
// experts, the norms) follows in layer_gemma_ffn.
bool npu_gemma_ffn(Layer& L, int T, int Mp, int a_in) {
  if (!npu_ || !hp.gemma || cur_T_ <= 8 || Mp < npu_min_ || (size_t)L.il >= npu_layers_.size() ||
      (!npu_layers_[L.il].on && !npu_layers_[L.il].down.on))
    return false;
  NpuLayer& N = npu_layers_[L.il];
  if (npu_->pending) npu_->wait();  // (a run left by an exception: done before its memory is written again)
  const int E = hp.n_embd, F = hp.n_ff;
  if (N.on) {
    {  // (the FFN norm's rows as f16: x_ is still the norm's input)
      int K = E, mv = T;
      float eps = hp.eps;
      set_args(npu_norm16_, 0, x_, L.ffn_norm, K, mv, eps, npu_ga16_);
      size_t gws[1] = {(size_t)Mp * 256}, lws[1] = {256};
      run("npu_norm16", npu_norm16_, 1, gws, lws);
    }
    npu_rows(0, L.il, *N.src[0], npu_ng_, npu_nn_, 0);
    npu_rows(1, L.il, *N.src[1], npu_ng_, npu_nn_, 0);
    ze_graph_handle_t gr = npu_graph_gu16(npu_height(Mp));
    clFinish(cl_.queue);
    if (!npu_->start(gr)) throw std::runtime_error("ARC_NPU: NPU start failed");
    matmul("gemm_gate", N.gate, Mp, big_, 2 * F, 0, 0, a_in);
    matmul("gemm_up", N.up, Mp, big_, 2 * F, F, 0, a_in);
    const auto tw = std::chrono::steady_clock::now();
    if (!npu_->wait()) throw std::runtime_error("ARC_NPU: NPU run failed");
    npu_wait_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count();
    ++npu_runs_;
    for (int t = 0; t < 2; ++t) {  // (the NPU's columns of gate / up into big_'s [gate | up] rows)
      int nn = npu_nn_, ld = 2 * F, coff = (t ? F : 0) + npu_ng_, ad = 0;
      set_args(npu_put_, 0, t ? npu_gcu_ : npu_gcg_, nn, big_, ld, coff, ad);
      size_t gws[2] = {(size_t)npu_nn_, (size_t)Mp};
      run("npu_put", npu_put_, 2, gws, nullptr);
    }
  } else {
    matmul("gemm_gate", L.gate, Mp, big_, 2 * F, 0, 0, a_in);
    matmul("gemm_up", L.up, Mp, big_, 2 * F, F, 0, a_in);
  }
  if (N.down.on) {  // (gelu(gate) * up as f16 rows for the NPU's share of ffn_down, started before the GPU's)
    int FF = F;
    set_args(npu_gelu16_, 0, big_, FF, npu_ga16_);
    size_t gws[2] = {(size_t)F, (size_t)Mp};
    run("npu_gelu16", npu_gelu16_, 2, gws, nullptr);
    npu_start16(0, Mp, N.down, L.il);
  }
  const int a_dn = act32_for(false);
  {
    int gelu = 1, FF = F, MM = Mp, TT = T, fused = 0;
    set_args(k_swiglu_, 0, big_, FF, MM, TT, aq_, asc_, absum_, a_dn, fused, gelu);
    run_rows_quant("swiglu_quant", F, T, Mp);
  }
  if (N.down.on) {
    matmul("gemm_down", N.down.gpu, Mp, big_, E, 0, 0, a_dn);
    npu_finish16(0, Mp, N.down, big_, E, 0);
  } else {
    matmul("gemm_down", L.down, Mp, big_, E, 0, 0, a_dn);
  }
  return true;
}

// starts matrix m's (layer il's) NPU share on the f16 rows already in A16, its weight rows decoded first (the GPU's
// writes finished before the NPU reads them)
void npu_start16(int kind, int Mp, NpuMat& m, int il) {
  npu_rows(kind + 2, il, *m.src, m.ng, m.nn, 0);
  ze_graph_handle_t gr = npu_graph16(kind, npu_height(Mp), m);
  clFinish(cl_.queue);
  if (!npu_->start(gr)) throw std::runtime_error("ARC_NPU: NPU start failed");
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
  if (npu_->pending) npu_->wait();  // (a run left by an exception: done before its memory is written again)
  if (hp.hc) {  // (Flash Next: the block input is the mixer's output, hco_, already normed)
    int ldx = hp.n_embd, KK = hp.n_embd;
    set_args(npu_to16_, 0, hco_, ldx, KK, npu_ga16_);
    size_t gws[2] = {(size_t)KK, (size_t)Mp};
    run("npu_to16", npu_to16_, 2, gws, nullptr);
  } else {
    int K = hp.n_embd, mv = T;
    float eps = hp.eps;
    set_args(npu_norm16_, 0, x_, L.attn_norm, K, mv, eps, npu_ga16_);
    size_t gws[1] = {(size_t)Mp * 256}, lws[1] = {256};
    run("npu_norm16", npu_norm16_, 1, gws, lws);
  }
  npu_start16(1, Mp, m, L.il);
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
      npu_->g->pfnSetArgumentValue(gr, named ? ou : outs[1], npu_shm_ + npu_cu_) != 0 ||
      !npu_->set(gr, "BG", npu_shm_ + npu_wq_[0]) || !npu_->set(gr, "BU", npu_shm_ + npu_wq_[1]) ||
      !npu_->set(gr, "SBG", npu_shm_ + npu_ws_[0]) || !npu_->set(gr, "SBU", npu_shm_ + npu_ws_[1]))
    throw std::runtime_error("ARC_NPU: cannot set the NPU graph's arguments");
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
  if (npu_->pending) npu_->wait();  // (a run left by an exception: done before its memory is written again)
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
  npu_rows(0, L.il, *N.src[0], npu_ng_, npu_nn_, 1);
  npu_rows(1, L.il, *N.src[1], npu_ng_, npu_nn_, 1);
  ze_graph_handle_t gr = npu_graph(npu_height(Mp));
  clFinish(cl_.queue);  // (the int8 rows and weight rows written before the NPU reads them)
  if (!npu_->start(gr)) throw std::runtime_error("ARC_NPU: NPU start failed");
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
    npu_start16(0, Mp, N.down, L.il);
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

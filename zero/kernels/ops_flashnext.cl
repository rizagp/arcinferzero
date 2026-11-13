// Part of the non-GEMM kernels (zero/kernels/ops*.cl, KernelParts in zero/clrt.h): built after the parts before
// it, whose definitions it uses. Qwen3.8 Flash Next: hyper-connections, the PLE n-gram table, QSA sparse attention, and the fused
// single-token mixers (hc_down1 / hc_up1).

// ---- Qwen3.8 Flash Next (qwen4exp) hyper-connections (docs/flash-next-spec.md). The residual of a token is HC
// streams of E floats ([M][HC*E], stream c at c*E); E is a multiple of 256, so every superblock of a row lies in
// one stream.

// xn = per-stream RMS norm of res times w ([HC][E]), stored as f32 and quantized for the low-rank down projection
// (K = HC*E); source rows r0 + m. grid (256, M), local (256, 1); rows >= M_valid store zeros.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void hc_norm_quant(__global const float* res, int r0, __global const float* w, int E, int HC, int M,
                            int M_valid, float eps, __global float* xn, __global char* aq, __global float* asc,
                            __global ushort* absum, int act32) {
  const int m = get_group_id(1), HCE = HC * E, lane = get_sub_group_local_id(), sg = get_sub_group_id();
  const bool valid = m < M_valid;
  __global const float* xr = res + (size_t)(r0 + m) * HCE;
  __local float rs[16];
  for (int c = 0; c < HC; ++c) {
    float ss = 0.0f;
    if (valid)
      for (int i = get_local_id(0); i < E; i += get_local_size(0)) ss += xr[c * E + i] * xr[c * E + i];
    ss = work_group_reduce_add(ss);
    if (get_local_id(0) == 0) rs[c] = rsqrt(ss / E + eps);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = sg; s < HCE / 256; s += get_num_sub_groups()) {
    const int base = s * 256 + 16 * lane, c = (s * 256) / E;
    const float r = rs[c];
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = valid ? xr[base + i] * r * w[base + i] : 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) xn[(size_t)m * HCE + base + i] = v[i];
    quant_store16(v, s, m, M, HCE, aq, asc, absum, act32);
  }
}

// out[t][n] = W[n] . x[t] in f32 (rows of K floats; K a multiple of 64): a router whose choice int8 inputs would
// change (Gemma 4 26B-A4B: int8 router inputs alone move the reference by KL 0.035). One sub-group per (row, token).
// grid (16 * N, T), local (16, 1). (Defined after gemv_bf16.)
// A router in f32: f32 input rows x [T][K] against BF16 weight rows (Flash Next's precise mode) (the file's own, exact in f32), RT rows x TT tokens
// per sub-group so that each weight and input load serves several products (one row x one token per sub-group, as
// gemv_f32, re-read the weights for every token: 744 ms of a 4K prompt). grid (16 * N / RT, ceil(T / TT)), local (16, 1)
// -DW_F32: f32 weight rows (Gemma 4 26B's router, F32 in its file)
#ifdef W_F32
#define GW_T float
#else
#define GW_T ushort
#endif
#ifndef RT
#define RT 4
#endif
#ifndef TT
#define TT 4
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_bf16(__global const float* x, int K, __global const GW_T* w, int N, int T, __global float* out,
                        int ldo) {
  const int n0 = get_global_id(0) / 16 * RT, t0 = get_global_id(1) * TT, lane = get_sub_group_local_id();
  float acc[RT][TT];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int t = 0; t < TT; ++t) acc[r][t] = 0.0f;
  for (int i = lane; i < K / 4; i += 16) {
    float4 a[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) a[t] = t0 + t < T ? vload4(i, x + (size_t)(t0 + t) * K) : (float4)0.0f;
#pragma unroll
    for (int r = 0; r < RT; ++r) {
#ifdef W_F32
      const float4 b = vload4(i, w + (size_t)min(n0 + r, N - 1) * K);
#else
      const ushort4 u = vload4(i, w + (size_t)min(n0 + r, N - 1) * K);
      const float4 b = (float4)(as_float((uint)u.x << 16), as_float((uint)u.y << 16), as_float((uint)u.z << 16),
                                as_float((uint)u.w << 16));
#endif
#pragma unroll
      for (int t = 0; t < TT; ++t) acc[r][t] += a[t].x * b.x + a[t].y * b.y + a[t].z * b.z + a[t].w * b.w;
    }
  }
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int t = 0; t < TT; ++t) {
      const float s = sub_group_reduce_add(acc[r][t]);
      if (lane == 0 && n0 + r < N && t0 + t < T) out[(size_t)(t0 + t) * ldo + n0 + r] = s;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))  // (gemv_f32: one 16-lane sub-group per row, see above gemv_bf16)
__kernel void gemv_f32(__global const float* x, int K, __global const float* w, int N, __global float* out, int ldo) {
  const int n = get_global_id(0) / 16, t = get_global_id(1), lane = get_sub_group_local_id();
  if (n >= N) return;
  __global const float4* xr = (__global const float4*)(x + (size_t)t * K);
  __global const float4* wr = (__global const float4*)(w + (size_t)n * K);
  float acc = 0.0f;
  for (int i = lane; i < K / 4; i += 16) {
    const float4 a = xr[i], b = wr[i];
    acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  }
  acc = sub_group_reduce_add(acc);
  if (lane == 0) out[(size_t)t * ldo + n] = acc;
}

// silu(lo / HC) of the low-rank activations ([M][LR], row stride LD), padded with zeros to KP (a multiple of 256) and quantized
// for the up projection. grid (16 * KP / 256, M), local (16, 1).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void hc_lo_quant(__global const float* lo, int LR, int LD, int KP, int HC, int M, int M_valid, __global char* aq,
                          __global float* asc, __global ushort* absum, int act32) {
  const int m = get_global_id(1), s = get_group_id(0), lane = get_sub_group_local_id();
  const float inv_hc = 1.0f / (float)HC;
  float v[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int k = s * 256 + 16 * lane + i;
    v[i] = m < M_valid && k < LR ? silu(lo[(size_t)m * LD + k] * inv_hc) : 0.0f;
  }
  quant_store16(v, s, m, M, KP, aq, asc, absum, act32);
}

// The rounding residual of hc_lo_quant (same values and scales: v - q * amax / 127), quantized for a second up pass
// that accumulates (ARC_HC_F32=4: two-pass int8 up input). grid (16 * KP / 256, M), local (16, 1).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void hc_lo_quant_res(__global const float* lo, int LR, int LD, int KP, int HC, int M, int M_valid,
                              __global char* aq, __global float* asc, __global ushort* absum, int act32) {
  const int m = get_global_id(1), s = get_group_id(0), lane = get_sub_group_local_id();
  const float inv_hc = 1.0f / (float)HC;
  float v[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int k = s * 256 + 16 * lane + i;
    v[i] = m < M_valid && k < LR ? silu(lo[(size_t)m * LD + k] * inv_hc) : 0.0f;
  }
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmax(amax, fabs(v[i]));
  if (act32) amax = fmax(amax, intel_sub_group_shuffle_xor(amax, 1));
  else amax = sub_group_reduce_max(amax);
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f, d = amax / 127.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] -= (float)convert_int_rte(v[i] * inv) * d;
  quant_store16(v, s, m, M, KP, aq, asc, absum, act32);
}

// The block input: mixed = mean over the streams of xn * sigmoid(gate) ([M][E], f32 and quantized for the block's
// projections, K = E). grid (256, M), local (256, 1).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void hc_gate_mean_quant(__global const float* xn, __global const float* gate, int E, int HC, int M,
                                 int M_valid, __global float* mixed, __global char* aq, __global float* asc,
                                 __global ushort* absum, int act32) {
  const int m = get_group_id(1), HCE = HC * E, lane = get_sub_group_local_id(), sg = get_sub_group_id();
  const bool valid = m < M_valid;
  const float inv_hc = 1.0f / (float)HC;
  for (int s = sg; s < E / 256; s += get_num_sub_groups()) {
    const int base = s * 256 + 16 * lane;
    // (each lane's 16 values with one vector load per stream: read value by value, a load instruction touched 16
    //  cache lines for 4 bytes each; Flash Next, 2048-row chunk: 45 -> 3.7 ms, bit-identical)
    float16 acc = (float16)0.0f;
    if (valid)
      for (int c = 0; c < HC; ++c) {
        const size_t j = (size_t)m * HCE + c * E + base;
        acc += vload16(0, xn + j) / (1.0f + exp(-vload16(0, gate + j)));
      }
    float v[16];
    vstore16(acc * inv_hc, 0, v);
    vstore16(acc * inv_hc, 0, mixed + (size_t)m * E + base);
    quant_store16(v, s, m, M, E, aq, asc, absum, act32);
  }
}

// The mixers of a prompt chunk with f32 activations (ARC_HC_F32; single tokens always take f32: hc_down1 / hc_up1):
// C[m][n] = W[n] . x_m for the Q8 superblock rows of hc_down / hc_up (272 bytes per 256 values: 8 f16 scales, one
// per 32 values, then the 256 int8 values in order; q8sb_dot reads them through the HC_XT order of its inputs). A lane per output column n, a
// work-group 256 columns x HCF_TM rows: the rows' 256 inputs of a superblock go through local memory once per
// work-group. lo_in: the inputs are silu(lo / HC) of the low-rank values (k < LR of rows of stride ld_in, zero up to
// K), else x rows of stride ld_in. grid (256 * ceil(N / 256), ceil(M / HCF_TM)), local (256, 1).
#ifndef HCF_TM
// -DHC_BF16 (Flash Next): the mixer rows as the file's BF16 values, exact, in their natural order (256 per
// superblock, lane l the 16 values 16 l ..): the same lane / input pairing as q8sb_dot, no scale
inline float bf16_at(uint w, int odd) { return as_float(odd ? (w & 0xFFFF0000u) : (w << 16)); }
inline float bf16sb_dot(__global const ushort* p, __local const float* x) {
  const int lane = get_sub_group_local_id();
  const uint8 q = vload8(lane, (__global const uint*)p);
  const uint w[8] = {q.s0, q.s1, q.s2, q.s3, q.s4, q.s5, q.s6, q.s7};
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) acc += bf16_at(w[j / 2], j % 2) * x[16 * j + lane];
  return acc;
}
inline float bf16sb_dot_v(__global const ushort* p, __local const float* x, int nv) {
  const int lane = get_sub_group_local_id();
  const uint8 q = 16 * lane < nv ? vload8(lane, (__global const uint*)p) : (uint8)0;
  const uint w[8] = {q.s0, q.s1, q.s2, q.s3, q.s4, q.s5, q.s6, q.s7};
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) acc += bf16_at(w[j / 2], j % 2) * x[16 * j + lane];
  return acc;
}

inline float f32sb_dot(__global const float* p, __local const float* x, int nv) {
  const int lane = get_sub_group_local_id();
  const float16 q = 16 * lane < nv ? vload16(lane, p) : (float16)0.0f;
  const float w[16] = {q.s0, q.s1, q.s2, q.s3, q.s4, q.s5, q.s6, q.s7,
                       q.s8, q.s9, q.sa, q.sb, q.sc, q.sd, q.se, q.sf};
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) acc += w[j] * x[16 * j + lane];
  return acc;
}

#define HCF_TM 16
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hc_gemm_f32(__global const float* x, int ld_in, int K, int lo_in, int LR, int HC,
                          __global const uchar* wq, int N, __global float* C, int ldc, int M) {
  __local float xs[HCF_TM * 256];
  const int li = get_local_id(0), n = get_global_id(0), m0 = get_group_id(1) * HCF_TM, nsb = K / 256;
  const float inv_hc = 1.0f / (float)HC;
  float acc[HCF_TM];
#pragma unroll
  for (int t = 0; t < HCF_TM; ++t) acc[t] = 0.0f;
  for (int sb = 0; sb < nsb; ++sb) {
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = li; i < HCF_TM * 256; i += 256) {
      const int t = i / 256, k = sb * 256 + i % 256, m = m0 + t;
      float v = 0.0f;
      if (m < M) {
        if (lo_in) v = k < LR ? silu(x[(size_t)m * ld_in + k] * inv_hc) : 0.0f;
        else v = x[(size_t)m * ld_in + k];
      }
      xs[i] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (n < N) {
#if defined(HC_F32)
      __global const float* p = (__global const float*)wq + ((size_t)n * nsb + sb) * 256;
#elif defined(HC_BF16)
      __global const uint* p = (__global const uint*)((__global const ushort*)wq + ((size_t)n * nsb + sb) * 256);
#else
      __global const uchar* p = wq + ((size_t)n * nsb + sb) * 272;
#endif
      for (int l = 0; l < 16; ++l) {
        if (lo_in && sb * 256 + 16 * l >= LR) break;  // (the up projection's zero padding past LR)
#if defined(HC_F32)
        const float16 q = vload16(l, p);
        const float wf[16] = {q.s0, q.s1, q.s2, q.s3, q.s4, q.s5, q.s6, q.s7,
                              q.s8, q.s9, q.sa, q.sb, q.sc, q.sd, q.se, q.sf};
#elif defined(HC_BF16)
        const uint8 q = vload8(l, p);
        const uint w8[8] = {q.s0, q.s1, q.s2, q.s3, q.s4, q.s5, q.s6, q.s7};
#else
        const float d = vload_half(l / 2, (__global const half*)p);
        const uint4 q = vload4(0, (__global const uint*)(p + 16 + 16 * l));
        const uint w4[4] = {q.x, q.y, q.z, q.w};
#endif
#pragma unroll
        for (int j = 0; j < 16; ++j) {
#if defined(HC_F32)
          const float w = wf[j];
#elif defined(HC_BF16)
          const float w = bf16_at(w8[j / 2], j % 2);
#else
          const float w = d * (float)(char)((w4[j / 4] >> (8 * (j % 4))) & 0xFF);
#endif
          const int k = 16 * l + j;
#pragma unroll
          for (int t = 0; t < HCF_TM; ++t) acc[t] += w * xs[t * 256 + k];
        }
      }
    }
  }
  if (n < N)
#pragma unroll
    for (int t = 0; t < HCF_TM; ++t)
      if (m0 + t < M) C[(size_t)(m0 + t) * ldc + n] = acc[t];
}

// res[m][c] += 2 sigmoid(inject[m][c] / HC) * out[m] for rows < M_valid (inject logits: column off of rows of stride
// ld). grid (HC * E, M_valid).
__kernel void hc_combine(__global float* res, __global const float* out, __global const float* inject, int ld,
                         int off, int E, int HC) {
  const int j = get_global_id(0), m = get_global_id(1), c = j / E, i = j % E;
  const float w = 2.0f / (1.0f + exp(-inject[(size_t)m * ld + off + c] / (float)HC));
  res[(size_t)m * HC * E + j] += w * out[(size_t)m * E + i];
}

// per-32 activation scales [K/32][M] -> [M][K/32] (the per-tile expert gate / up kernel, -DASCT, reads a token's
// scales side by side). grid (K/32 rounded up to 16, M).
__kernel void asc_transpose(__global const float* asc, int M, int KB, __global float* asct) {
  const int b = get_global_id(0), m = get_global_id(1);
  if (b < KB) asct[(size_t)m * KB + b] = asc[(size_t)b * M + m];
}

// the mean of the HC streams per row (Flash Next's DFlash2 features). grid (E, M_valid).
__kernel void hc_mean(__global const float* res, int E, int HC, __global float* out) {
  const int i = get_global_id(0), m = get_global_id(1);
  float s = 0.0f;
  for (int c = 0; c < HC; ++c) s += res[((size_t)m * HC + c) * E + i];
  out[(size_t)m * E + i] = s / (float)HC;
}

// ---- Flash Next PLE (n-gram embedding block of one layer; docs/flash-next-spec.md, llama.cpp qwen4exp)
// Per row m < M_valid and stream c: s = <rmsnorm(key_c) * wk_c, rmsnorm(res_c) * wq_c> / sqrt(E), gate =
// sigmoid(sign(s) * sqrt(max(|s|, 1e-6))), gated_c = value * gate, nrm_c = rmsnorm(gated_c) * wc_c.
// key [M][HC*E], value [M][E], res [M][HC*E]; gated, nrm [M][HC*E]. grid (256, M_valid), local (256, 1).
__kernel void ple_gate(__global const float* key, __global const float* value, __global const float* res,
                       __global const float* wk, __global const float* wq, __global const float* wc, int E, int HC,
                       float eps, __global float* gated, __global float* nrm) {
  const int m = get_group_id(1), HCE = HC * E, li = get_local_id(0), ls = get_local_size(0);
  __global const float* kr = key + (size_t)m * HCE;
  __global const float* qr = res + (size_t)m * HCE;
  __global const float* vr = value + (size_t)m * E;
  for (int c = 0; c < HC; ++c) {
    float kk = 0.0f, qq = 0.0f, kq = 0.0f;
    for (int i = li; i < E; i += ls) {
      const float a = kr[c * E + i], b = qr[c * E + i];
      kk += a * a;
      qq += b * b;
      kq += a * wk[c * E + i] * b * wq[c * E + i];
    }
    kk = work_group_reduce_add(kk);
    qq = work_group_reduce_add(qq);
    kq = work_group_reduce_add(kq);
    const float s = kq * rsqrt(kk / E + eps) * rsqrt(qq / E + eps) / sqrt((float)E);
    const float g = 1.0f / (1.0f + exp(-(s < 0.0f ? -1.0f : s > 0.0f ? 1.0f : 0.0f) * sqrt(fmax(fabs(s), 1e-6f))));
    float gg = 0.0f;
    for (int i = li; i < E; i += ls) {
      const float v = vr[i] * g;
      gated[(size_t)m * HCE + c * E + i] = v;
      gg += v * v;
    }
    gg = work_group_reduce_add(gg);
    const float r = rsqrt(gg / E + eps);
    for (int i = li; i < E; i += ls) nrm[(size_t)m * HCE + c * E + i] = vr[i] * g * r * wc[c * E + i];
  }
}

// res[m][ch] += gated[m][ch] + silu(sum_k w[ch][k] * nrm(m - (NTAP-1-k) * DIL)[ch]) for rows < T: a depthwise causal
// conv over the rows of this chunk and, before them, the history hist [HIST][HCE] (the last HIST rows of earlier
// positions, oldest first; zeros at the start of a sequence). grid (HCE, T).
__kernel void ple_conv_add(__global float* res, __global const float* gated, __global const float* nrm,
                           __global const float* hist, __global const float* w, int HCE, int NTAP, int DIL, int HIST) {
  const int ch = get_global_id(0), m = get_global_id(1);
  float acc = 0.0f;
  for (int k = 0; k < NTAP; ++k) {
    const int t = m - (NTAP - 1 - k) * DIL;
    const float v = t >= 0 ? nrm[(size_t)t * HCE + ch] : t + HIST >= 0 ? hist[(size_t)(t + HIST) * HCE + ch] : 0.0f;
    acc += w[(size_t)ch * NTAP + k] * v;
  }
  res[(size_t)m * HCE + ch] += gated[(size_t)m * HCE + ch] + acc / (1.0f + exp(-acc));
}

// hist = the last HIST rows of [hist; nrm[0..T)], in place (each work-item owns one channel). grid (HCE).
__kernel void ple_hist_update(__global float* hist, __global const float* nrm, int HCE, int T, int HIST) {
  const int ch = get_global_id(0);
  float h[16];
  for (int j = 0; j < HIST; ++j) {
    const int t = T - HIST + j;  // row of the combined sequence relative to the chunk start
    h[j] = t >= 0 ? nrm[(size_t)t * HCE + ch] : hist[(size_t)(t + HIST) * HCE + ch];
  }
  for (int j = 0; j < HIST; ++j) hist[(size_t)j * HCE + ch] = h[j];
}

// ---- Flash Next QSA (sparse attention, llama.cpp qwen4exp build_qsa_sel; docs/flash-next-spec.md). The indexer keys
// of blocks of QR consecutive positions are pooled (mean of the raw keys, RMS norm, RoPE at the block's first
// position); a query scores every complete block it can see (sum over the indexer heads of relu(q . k) / sqrt(dim))
// and attends to the keys of its top blocks and of its incomplete tail block. Indexer rows ixo [M][ldx]: raw key
// (QD floats) at column 0, the QH query heads after it. A lane holds dims lane + 16 j (QD = 128): RoPE pairs (i, i +
// n_rot / 2) are then in one lane.
#ifndef QD
#define QD 128
#endif
#ifndef QR  // (block size and indexer heads: the engine passes the model's)
#define QR 4
#endif
#ifndef QH
#define QH 4
#endif
#define QJ (QD / 16)
inline void qsa_norm_rope(float* v, __global const float* w, float eps, int p, float theta_scale, int n_rot) {
  const int lane = get_sub_group_local_id();
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < QJ; ++j) ss += v[j] * v[j];
  const float r = rsqrt(sub_group_reduce_add(ss) / QD + eps);
#pragma unroll
  for (int j = 0; j < QJ; ++j) v[j] *= r * w[lane + 16 * j];
  const int jh = n_rot / 32;  // (pairs (i, i + n_rot / 2): lane values j, j + jh; n_rot a multiple of 32)
#pragma unroll
  for (int j = 0; j < QJ; ++j)
    if (j < jh) {
      const float theta = (float)p * pow(theta_scale, (float)(lane + 16 * j));
      const float c = cos(theta), sn = sin(theta), a = v[j], b = v[j + jh];
      v[j] = a * c - b * sn;
      v[j + jh] = a * sn + b * c;
    }
}
// pooled keys of the blocks b0 .. b0 + get_num_groups(0) - 1 (complete within positions < pos0 + T); positions
// before pos0 come from the tail state (positions pos0 / QR * QR .. pos0 - 1). grid (16 * blocks), local (16).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qsa_pool(__global const float* ixo, int ldx, int pos0, __global const float* tail, __global const float* kn,
                       int b0, float eps, float theta_scale, int n_rot, __global float* pk) {
  const int b = b0 + get_group_id(0), lane = get_sub_group_local_id(), t0 = pos0 / QR * QR;
  float v[QJ];
#pragma unroll
  for (int j = 0; j < QJ; ++j) v[j] = 0.0f;
  for (int i = 0; i < QR; ++i) {
    const int p = b * QR + i;
#pragma unroll
    for (int j = 0; j < QJ; ++j) v[j] += p >= pos0 ? ixo[(size_t)(p - pos0) * ldx + lane + 16 * j] : tail[(p - t0) * QD + lane + 16 * j];
  }
#pragma unroll
  for (int j = 0; j < QJ; ++j) v[j] *= 1.0f / QR;
  qsa_norm_rope(v, kn, eps, b * QR, theta_scale, n_rot);
#pragma unroll
  for (int j = 0; j < QJ; ++j) pk[(size_t)b * QD + lane + 16 * j] = v[j];
}
// the tail state after the chunk: raw keys of positions (pos0 + T) / QR * QR .. pos0 + T - 1. grid (QD), one group.
__kernel void qsa_tail(__global const float* ixo, int ldx, int pos0, int T, __global float* tail) {
  const int d = get_global_id(0), t0 = pos0 / QR * QR, n0 = (pos0 + T) / QR * QR;
  float h[QR];
  for (int i = 0; i < QR - 1; ++i) {
    const int p = n0 + i;
    h[i] = p >= pos0 + T ? 0.0f : p >= pos0 ? ixo[(size_t)(p - pos0) * ldx + d] : tail[(p - t0) * QD + d];
  }
  for (int i = 0; i < QR - 1; ++i) tail[i * QD + d] = h[i];
}
// indexer queries: RMS norm + RoPE per head. grid (16 * QH, T), local (16, 1).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qsa_q(__global const float* ixo, int ldx, int pos0, __global const float* qn, float eps, float theta_scale,
                    int n_rot, __global float* qx) {
  const int m = get_global_id(1), hh = get_group_id(0), lane = get_sub_group_local_id();
  float v[QJ];
#pragma unroll
  for (int j = 0; j < QJ; ++j) v[j] = ixo[(size_t)m * ldx + QD + hh * QD + lane + 16 * j];
  qsa_norm_rope(v, qn, eps, pos0 + m, theta_scale, n_rot);
#pragma unroll
  for (int j = 0; j < QJ; ++j) qx[((size_t)m * QH + hh) * QD + lane + 16 * j] = v[j];
}
// scores of rows r0 .. r0 + R - 1 against the blocks they see (else -1). grid (blocks rounded to 64, R), local (64, 1).
__kernel void qsa_score(__global const float* qx, __global const float* pk, int r0, int pos0, __global float* sc, int lds) {
  const int b = get_global_id(0), i = get_global_id(1), m = r0 + i, nv = (pos0 + m + 1) / QR;
  __local float q[QH * QD];
  for (int k = get_local_id(0); k < QH * QD; k += get_local_size(0)) q[k] = qx[(size_t)m * QH * QD + k];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (b >= lds) return;
  float s = -1.0f;
  if (b < nv) {
    __global const float4* kp = (__global const float4*)(pk + (size_t)b * QD);
    s = 0.0f;
    for (int hh = 0; hh < QH; ++hh) {
      float d = 0.0f;
      for (int k = 0; k < QD / 4; ++k) {
        const float4 kv = kp[k];
        d += q[hh * QD + 4 * k] * kv.x + q[hh * QD + 4 * k + 1] * kv.y + q[hh * QD + 4 * k + 2] * kv.z + q[hh * QD + 4 * k + 3] * kv.w;
      }
      s += fmax(d, 0.0f);
    }
    s *= rsqrt((float)QD);
  }
  sc[(size_t)i * lds + b] = s;
}
// qsa_score for QRT rows at a time: each block's keys are read once for the QRT rows (qsa_score reads them once per
// row: the whole pooled key array per row, ~64 GB per layer of a 32K prompt). Per row and head the same sum in the
// same order (k ascending, then the heads), so the scores are the same. grid (blocks rounded to 64, ceil(R / QRT)),
// local (64, 1).
#ifndef QRT
#define QRT 8
#endif
__kernel void qsa_score_rt(__global const float* qx, __global const float* pk, int r0, int pos0, int R,
                           __global float* sc, int lds) {
  const int b = get_global_id(0), i0 = get_global_id(1) * QRT;
  __local float q[QRT * QH * QD];
  for (int k = get_local_id(0); k < QRT * QH * QD; k += get_local_size(0)) {
    const int r = k / (QH * QD);
    q[k] = i0 + r < R ? qx[(size_t)(r0 + i0 + r) * QH * QD + k % (QH * QD)] : 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (b >= lds) return;
  float d[QRT][QH];
#pragma unroll
  for (int r = 0; r < QRT; ++r)
#pragma unroll
    for (int hh = 0; hh < QH; ++hh) d[r][hh] = 0.0f;
  __global const float4* kp = (__global const float4*)(pk + (size_t)b * QD);
  for (int k = 0; k < QD / 4; ++k) {
    const float4 kv = kp[k];
#pragma unroll
    for (int r = 0; r < QRT; ++r)
#pragma unroll
      for (int hh = 0; hh < QH; ++hh) {
        __local const float* qq = q + (r * QH + hh) * QD + 4 * k;
        d[r][hh] += qq[0] * kv.x + qq[1] * kv.y + qq[2] * kv.z + qq[3] * kv.w;
      }
  }
#pragma unroll
  for (int r = 0; r < QRT; ++r) {
    const int i = i0 + r, nv = (pos0 + r0 + i + 1) / QR;
    if (i >= R) break;
    float s = -1.0f;
    if (b < nv) {
      s = 0.0f;
#pragma unroll
      for (int hh = 0; hh < QH; ++hh) s += fmax(d[r][hh], 0.0f);
      s *= rsqrt((float)QD);
    }
    sc[(size_t)i * lds + b] = s;
  }
}
// the mask of row r0 + i: its top `ntop` visible blocks (ties: the lower blocks), all of them when it sees no more.
// Exact selection by a radix select on the score bits (scores >= 0, so their bits sort as the values). mask
// [M][qmw] words, bit b = block b. grid (256, R), local (256, 1).
__kernel void qsa_select(__global const float* sc, int lds, int r0, int pos0, int ntop, __global uint* mask, int qmw) {
  const int i = get_group_id(1), m = r0 + i, lid = get_local_id(0), nv = (pos0 + m + 1) / QR;
  __global const uint* su = (__global const uint*)(sc + (size_t)i * lds);
  __local uint hist[256];
  __local uint sel[2];  // prefix, rank left
  uint prefix = 0, pmask = 0, k = (uint)ntop;
  const bool all = nv <= ntop;
  if (!all)
    for (int shift = 24; shift >= 0; shift -= 8) {
      hist[lid] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int b = lid; b < nv; b += 256) {
        const uint u = su[b];
        if ((u & pmask) == prefix) atomic_inc(&hist[(u >> shift) & 255]);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (lid == 0) {
        uint cum = 0;
        int bin = 255;
        for (; bin > 0; --bin) {
          if (cum + hist[bin] >= k) break;
          cum += hist[bin];
        }
        sel[0] = prefix | ((uint)bin << shift);
        sel[1] = k - cum;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      prefix = sel[0];
      k = sel[1];
      pmask |= 255u << shift;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  // prefix = the ntop-th largest score; k = how many of the scores equal to it are taken (the lowest blocks)
  const int nw = (nv + 31) / 32, per = (nw + 255) / 256, w0 = lid * per, w1 = min(nw, w0 + per);
  uint eq = 0;
  if (!all)
    for (int b = w0 * 32; b < min(nv, w1 * 32); ++b) eq += su[b] == prefix;
  uint before = work_group_scan_exclusive_add(eq);
  for (int w = w0; w < w1; ++w) {
    uint bits = 0;
    for (int j = 0; j < 32; ++j) {
      const int b = w * 32 + j;
      if (b >= nv) break;
      bool take = all;
      if (!all) {
        const uint u = su[b];
        take = u > prefix;
        if (u == prefix) { take = before < k; ++before; }
      }
      bits |= (take ? 1u : 0u) << j;
    }
    mask[(size_t)m * qmw + w] = bits;
  }
}


// ---- Flash Next mixers of a single-token pass, fused, with f32 activations (the prompt path quantizes them for the
// int8 GEMMs). Weights in the Q8_0 GPU raw form: superblocks of 8 f16 scales + 256 int8 (272 bytes). HCK = HC * E.
#ifndef HCK
#define HCK 10240
#endif
// The activations in local memory are stored per superblock transposed, element 16 l + j at 16 j + l (HC_XT), so that
// the lanes' reads of one step are consecutive: read as x[16 l + j] all 16 lanes hit the same bank (hc_up1 100 us).
#define HC_XT(k) (((k) & ~255) | (((k) & 15) << 4) | (((k) >> 4) & 15))
// dot of 256 weights (one superblock at p) with x[0..256) (HC_XT layout): lane l takes elements 16 l .. 16 l + 15
// (block l / 2)
inline float q8sb_dot(__global const uchar* p, __local const float* x) {
  const int lane = get_sub_group_local_id();
  const float d = vload_half(lane / 2, (__global const half*)p);
  const uint4 q = vload4(0, (__global const uint*)(p + 16 + 16 * lane));
  const uint w[4] = {q.x, q.y, q.z, q.w};
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) acc += (float)(char)((w[j / 4] >> (8 * (j % 4))) & 0xFF) * x[16 * j + lane];
  return d * acc;
}
// q8sb_dot for a superblock holding nv < 256 values (the rest zero padding): lanes past them load nothing (the up
// projection's 320 values in 512; hc_up1 905 -> 843 ms per 96 Flash Next tokens, bit-identical; returning 0 for
// those lanes instead was 731 ms but changed the compiled arithmetic and with it the results)
inline float q8sb_dot_v(__global const uchar* p, __local const float* x, int nv) {
  const int lane = get_sub_group_local_id();
  const bool on = 16 * lane < nv;  // (the same arithmetic for every lane, only the loads predicated)
  const float d = on ? vload_half(lane / 2, (__global const half*)p) : 0.0f;
  const uint4 q = on ? vload4(0, (__global const uint*)(p + 16 + 16 * lane)) : (uint4)0;
  const uint w[4] = {q.x, q.y, q.z, q.w};
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) acc += (float)(char)((w[j / 4] >> (8 * (j % 4))) & 0xFF) * x[16 * j + lane];
  return d * acc;
}
// Partial sums of lo[n] = W_down[n] . xn over K slice ks (HCK / KS values; n < N: the low-rank and the inject rows),
// xn = per-stream RMS norm of res row r0 times w; part [KS][N]; the streams' 1 / rms to rinv. grid (256 * ceil(N / 16),
// KS), local (256, 1): a sub-group per (row, slice), so enough threads run (one row per sub-group over all of K: 21
// work-groups, 3x slower than the int8 GEMV path).
#ifndef HC_KS
#define HC_KS 8
#endif
#ifndef HC_RT
#define HC_RT 1  // rows (single tokens: 1; the 2-8 rows of a verification pass, each computed as a single token is)
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hc_down1(__global const float* res, int r0, __global const float* w, int E, int HC, float eps,
                       __global const uchar* wd, int N, __global float* part, __global float* rinv) {
  __local float xs[HC_RT][HCK / HC_KS];
  __local float rs[HC_RT];
  const int li = get_local_id(0), ls = get_local_size(0), ks = get_group_id(1), SL = HC * E / HC_KS;
  const int k0 = ks * SL, c = k0 / E;  // (a slice lies in one stream: E is a multiple of SL)
  for (int r = 0; r < HC_RT; ++r) {
    __global const float* xr = res + (size_t)(r0 + r) * HC * E;
    float ss = 0.0f;
    for (int i = li; i < E; i += ls) {
      const float v = xr[c * E + i];
      ss += v * v;
    }
    ss = work_group_reduce_add(ss);
    if (li == 0) rs[r] = rsqrt(ss / E + eps);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int r = 0; r < HC_RT; ++r) {
    __global const float* xr = res + (size_t)(r0 + r) * HC * E;
    const float rr = rs[r];
    for (int i = li; i < SL; i += ls) xs[r][HC_XT(i)] = xr[k0 + i] * rr * w[k0 + i];
    if (get_group_id(0) == 0 && li == 0 && (k0 % E) == 0) rinv[r * HC + c] = rr;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int n = get_group_id(0) * 16 + get_sub_group_id(), nsb = HC * E / 256, s0 = k0 / 256;
  if (n >= N) return;
  float acc[HC_RT];
#pragma unroll
  for (int r = 0; r < HC_RT; ++r) acc[r] = 0.0f;
  for (int sb = 0; sb < SL / 256; ++sb)
#pragma unroll
    for (int r = 0; r < HC_RT; ++r)
#if defined(HC_F32)
      acc[r] += f32sb_dot((__global const float*)wd + ((size_t)n * nsb + s0 + sb) * 256, xs[r] + sb * 256, 256);
#elif defined(HC_BF16)
      acc[r] += bf16sb_dot((__global const ushort*)wd + ((size_t)n * nsb + s0 + sb) * 256, xs[r] + sb * 256);
#else
      acc[r] += q8sb_dot(wd + ((size_t)n * nsb + s0 + sb) * 272, xs[r] + sb * 256);
#endif
#pragma unroll
  for (int r = 0; r < HC_RT; ++r) {
    const float a = sub_group_reduce_add(acc[r]);
    if (get_sub_group_local_id() == 0) part[((size_t)r * HC_KS + ks) * N + n] = a;
  }
}
// mixed[i] = mean over the streams c of xn_c[i] * sigmoid(W_up[c E + i] . silu(lo / HC)), lo = the sum of the partial
// sums (LR values, the rest of the KP-wide input zero). grid (256 * ceil(E / 16)), local (256): a sub-group per i.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hc_up1(__global const float* res, int r0, __global const float* w, __global const float* rinv,
                     __global const float* part, int N, __global float* lo, int LR, int KP, int E, int HC,
                     __global const uchar* wu, __global float* mixed) {
  __local float xs[HC_RT][1024];
  const int li = get_local_id(0);
  for (int r = 0; r < HC_RT; ++r)
    for (int k = li; k < KP; k += get_local_size(0)) {
      float v = 0.0f;
      if (k < N)
        for (int q = 0; q < HC_KS; ++q) v += part[((size_t)r * HC_KS + q) * N + k];
      if (get_group_id(0) == 0 && k < N) lo[(size_t)r * N + k] = v;  // (the inject logits for hc_combine)
      v = k < LR ? v / (float)HC : 0.0f;
      xs[r][HC_XT(k)] = v / (1.0f + exp(-v));
    }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int i = get_group_id(0) * 16 + get_sub_group_id(), nsb = KP / 256;
  if (i >= E) return;
  float m[HC_RT];
#pragma unroll
  for (int r = 0; r < HC_RT; ++r) m[r] = 0.0f;
  for (int c = 0; c < HC; ++c) {
    const int n = c * E + i;
    float acc[HC_RT];
#pragma unroll
    for (int r = 0; r < HC_RT; ++r) acc[r] = 0.0f;
    for (int sb = 0; sb < nsb; ++sb)
#pragma unroll
      for (int r = 0; r < HC_RT; ++r)
#if defined(HC_F32)
        acc[r] += f32sb_dot((__global const float*)wu + ((size_t)n * nsb + sb) * 256, xs[r] + sb * 256, LR - sb * 256);
#elif defined(HC_BF16)
        acc[r] += bf16sb_dot_v((__global const ushort*)wu + ((size_t)n * nsb + sb) * 256, xs[r] + sb * 256, LR - sb * 256);
#else
        acc[r] += q8sb_dot_v(wu + ((size_t)n * nsb + sb) * 272, xs[r] + sb * 256, LR - sb * 256);
#endif
#pragma unroll
    for (int r = 0; r < HC_RT; ++r) {
      const float g = sub_group_reduce_add(acc[r]);
      __global const float* xr = res + (size_t)(r0 + r) * HC * E;
      m[r] += xr[n] * rinv[r * HC + c] * w[n] / (1.0f + exp(-g));
    }
  }
  if (get_sub_group_local_id() == 0)
#pragma unroll
    for (int r = 0; r < HC_RT; ++r) mixed[(size_t)r * E + i] = m[r] / (float)HC;
}

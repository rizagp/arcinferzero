#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// Non-GEMM kernels of the forward passes (norms, activation quantization, Gated DeltaNet, RoPE, attention,
// Gemma 4 and MTP helpers). Activation quantization layout matches kernels/gemm.cl.
// The first part of them (KernelParts in zero/clrt.h): the common definitions, the norms and activation
// quantization, Whisper's helpers and Gated DeltaNet; the parts after it are ops_attn.cl, ops_attn_g.cl, ops_vit.cl,
// ops_flashnext.cl and ops_dflash.cl, each built with the ones before it.

#define SG 16
// int8 attention: scale blocks per q/K/V row (see attn_q8); the engine passes -DSB to the attention kernels
#ifndef SB
#define SB 1
#endif
// Gemma 4's float attention: the K rows in f32 (-DK_F16: in f16, Qwen3 dense: its float attention reads half the K
// bytes. Qwen3-ASR, 72-token prompt against f32: KL < 0.00001 either way (max 0.00004); attention -10 %, one token
// at 2048 of context 26.8 -> 27.9 tok/s; Gemma 4 too since 2026-11-11, ARC_G4_KF16=0: f32)
#ifdef K_F16
#define KC_T half
#define KLD16(p) vload_half16(0, p)
#define KST1(x, p) vstore_half(x, 0, p)
#else
#define KC_T float
#define KLD16(p) vload16(0, p)
#define KST1(x, p) (*(p) = (x))
#endif
// and the V rows in f16
#define VC_T half
#define VC_RAW ushort
#define VLD4(p) vload_half4(0, p)
#define VLD8(p) vload_half8(0, p)
#define VLD16(p) vload_half16(0, p)
#define VST1(x, p) vstore_half(x, 0, p)

inline float silu(float x) { return x / (1.0f + exp(-x)); }

// Quantizes superblock s of row m to int8. Lane l holds v[i] = element s*256 + 16*l + i.
// Scale per 256 elements (asc [K/256][M]) or, with act32, per 32 elements (asc [K/32][M]).
// absum [K/16][M] holds the int8 sum of each 16-element block (for weight types with mins).
inline void quant_store16(const float* v, int s, int m, int M, int K, __global char* aq, __global float* asc,
                          __global ushort* absum, int act32) {
  const int lane = get_sub_group_local_id();
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmax(amax, fabs(v[i]));
  if (act32) amax = fmax(amax, intel_sub_group_shuffle_xor(amax, 1));
  else amax = sub_group_reduce_max(amax);
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  int sum = 0;
  uint pk[4] = {0, 0, 0, 0};
#ifdef Q2_AQ
  float res[16];  // (two passes: what the int8 values miss)
#endif
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int q = convert_int_rte(v[i] * inv);
    sum += q;
    pk[i / 4] |= (uint)(q & 0xFF) << (8 * (i % 4));
#ifdef Q2_AQ
    res[i] = v[i] - (float)q * (amax / 127.0f);
#endif
  }
  vstore4((uint4)(pk[0], pk[1], pk[2], pk[3]), 0, (__global uint*)(aq + (size_t)m * K + s * 256 + 16 * lane));
#ifdef H8F
  // (zero/engine_h8.h h8_float: the values before rounding too, H8F bytes further on, for the H8 rotation)
  vstore16((float16)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]), 0,
           (__global float*)(aq + H8F) + (size_t)m * K + s * 256 + 16 * lane);
#endif
  absum[((size_t)s * 16 + lane) * M + m] = (ushort)(short)sum;
  if (act32) {
    if ((lane & 1) == 0) asc[((size_t)s * 8 + lane / 2) * M + m] = amax / 127.0f;
  } else if (lane == 0) {
    asc[(size_t)s * M + m] = amax / 127.0f;
  }
#ifdef Q2_AQ
  // the residual rounded the same way, in the same layout, Q2_AQ bytes / Q2_SC floats / Q2_AB shorts further on
  {
    float rmax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) rmax = fmax(rmax, fabs(res[i]));
    if (act32) rmax = fmax(rmax, intel_sub_group_shuffle_xor(rmax, 1));
    else rmax = sub_group_reduce_max(rmax);
    const float rinv = rmax > 0.0f ? 127.0f / rmax : 0.0f;
    int rsum = 0;
    uint rk[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = convert_int_rte(res[i] * rinv);
      rsum += q;
      rk[i / 4] |= (uint)(q & 0xFF) << (8 * (i % 4));
    }
    __global char* aq2 = aq + Q2_AQ;
    __global float* asc2 = asc + Q2_SC;
    __global ushort* ab2 = absum + Q2_AB;
    vstore4((uint4)(rk[0], rk[1], rk[2], rk[3]), 0, (__global uint*)(aq2 + (size_t)m * K + s * 256 + 16 * lane));
    ab2[((size_t)s * 16 + lane) * M + m] = (ushort)(short)rsum;
    if (act32) {
      if ((lane & 1) == 0) asc2[((size_t)s * 8 + lane / 2) * M + m] = rmax / 127.0f;
    } else if (lane == 0) {
      asc2[(size_t)s * M + m] = rmax / 127.0f;
    }
  }
#endif
}

// y = rmsnorm(x[row_off + m]) * w, quantized for the next GEMM. Optionally also stores y as f32.
// Work-group: 256 items per row; grid (256, M), or (256 * G, T) for a 1-8 token pass: each of the G work-groups
// sums the whole row's squares the same way (bit-identical r) and quantizes its share of the superblocks.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void rmsnorm_quant(__global const float* x, int row_off, __global const float* w, int K, int M, int M_valid,
                            float eps, __global char* aq, __global float* asc, __global ushort* absum,
                            __global float* hout, int write_h, int act32) {
  const int m = get_group_id(1);
  const bool valid = m < M_valid;
  __global const float* xr = x + (size_t)(m + row_off) * K;
  float ss = 0.0f;
  if (valid)
    for (int i = get_local_id(0); i < K; i += get_local_size(0)) ss += xr[i] * xr[i];
  ss = work_group_reduce_add(ss);
  const float r = rsqrt(ss / K + eps);
  const int lane = get_sub_group_local_id();
  // (superblocks over every sub-group of the grid: grid (256, M) is one work-group per row as before; a 1-8 token
  //  pass may launch several along dim 0 (zero/engine.h wide_rows), the same arithmetic per superblock)
  for (int s = get_group_id(0) * get_num_sub_groups() + get_sub_group_id(); s < K / 256;
       s += get_num_groups(0) * get_num_sub_groups()) {
    const int base = s * 256 + 16 * lane;
    float v[16];
    if (valid) {  // (one vector load per lane and array, the row flag tested once; the same arithmetic)
      const float16 xa = vload16(0, xr + base), wa = vload16(0, w + base);
      const float xv[16] = {xa.s0, xa.s1, xa.s2, xa.s3, xa.s4, xa.s5, xa.s6, xa.s7, xa.s8, xa.s9, xa.sa, xa.sb, xa.sc, xa.sd, xa.se, xa.sf};
      const float wv[16] = {wa.s0, wa.s1, wa.s2, wa.s3, wa.s4, wa.s5, wa.s6, wa.s7, wa.s8, wa.s9, wa.sa, wa.sb, wa.sc, wa.sd, wa.se, wa.sf};
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = xv[i] * r * wv[i];
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = 0.0f;
    }
    if (write_h)
#pragma unroll
      for (int i = 0; i < 16; ++i) hout[(size_t)m * K + base + i] = v[i];
    quant_store16(v, s, m, M, K, aq, asc, absum, act32);
  }
}

#ifdef H8NORM
// ---- rmsnorm_quant and h8.cl's h8r_act in one kernel (B60 prefill plan 13.5, zero/engine_h8.h rmsnorm_h8r;
// -DH8NORM -DHB=<rotation block> -DH8N_K=<K>, built without -DH8F): rmsnorm_quant's body unchanged (the same
// work-group of 256, the same reduction for r, the same int8 rows, scales and sums), but the f32 values go to local
// memory instead of the H8F copy, and the H8R rotation of the row is made from them as h8r_act makes it from that copy:
// the same floats, the same transform, the same scale, so h8_aq_ / h8_asc_ are the same bit for bit. Saves the f32
// copy's write and read and one launch per GEMM input. K / HB rotation blocks, one sub-group each (at most 16).
#define HN_NV (HB / 16)
inline float h8n_signed(float v, uint i) {  // (h8.cl h8_signed)
#ifdef H8_D
  uint x = i * 0x9E3779B1u;
  x ^= x >> 15;
  x *= 0x85EBCA77u;
  return as_float(as_uint(v) ^ (x & 0x80000000u));
#else
  return v;
#endif
}
inline void h8n_fwht_regs(float* v) {  // (h8.cl h8_fwht_regs)
  const uint l = get_sub_group_local_id();
#pragma unroll
  for (uint h = 1; h < HN_NV; h <<= 1)
#pragma unroll
    for (uint j = 0; j < HN_NV; ++j)
      if ((j & h) == 0) {
        const float a = v[j], b = v[j + h];
        v[j] = a + b;
        v[j + h] = a - b;
      }
#pragma unroll
  for (uint h = 1; h < 16; h <<= 1)
#pragma unroll
    for (uint j = 0; j < HN_NV; ++j) {
      const float o = intel_sub_group_shuffle_xor(v[j], h);
      v[j] = (l & h) ? (o - v[j]) : (v[j] + o);
    }
}
inline uint h8n_pack_q(const float v[4], float inv) {  // (h8.cl pack_q)
  uint w = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int q = clamp((int)rint(v[i] * inv), -127, 127);
    w |= ((uint)(q & 0xFF)) << (8 * i);
  }
  return w;
}
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void rmsnorm_h8r(__global const float* x, int row_off, __global const float* w, int K, int M, int M_valid,
                          float eps, __global char* aq, __global float* asc, __global ushort* absum,
                          __global float* hout, int write_h, int act32, __global uint* restrict aqh,
                          __global float* restrict asch) {
  __local float rowv[H8N_K];
  __local float red[16];
  const int m = get_group_id(1);
  const bool valid = m < M_valid;
  __global const float* xr = x + (size_t)(m + row_off) * K;
  float ss = 0.0f;
  if (valid)
    for (int i = get_local_id(0); i < K; i += get_local_size(0)) ss += xr[i] * xr[i];
  ss = work_group_reduce_add(ss);
  const float r = rsqrt(ss / K + eps);
  const int lane = get_sub_group_local_id();
  for (int s = get_group_id(0) * get_num_sub_groups() + get_sub_group_id(); s < K / 256;
       s += get_num_groups(0) * get_num_sub_groups()) {
    const int base = s * 256 + 16 * lane;
    float v[16];
    if (valid) {
      const float16 xa = vload16(0, xr + base), wa = vload16(0, w + base);
      const float xv[16] = {xa.s0, xa.s1, xa.s2, xa.s3, xa.s4, xa.s5, xa.s6, xa.s7, xa.s8, xa.s9, xa.sa, xa.sb, xa.sc, xa.sd, xa.se, xa.sf};
      const float wv[16] = {wa.s0, wa.s1, wa.s2, wa.s3, wa.s4, wa.s5, wa.s6, wa.s7, wa.s8, wa.s9, wa.sa, wa.sb, wa.sc, wa.sd, wa.se, wa.sf};
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = xv[i] * r * wv[i];
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = 0.0f;
    }
    if (write_h)
#pragma unroll
      for (int i = 0; i < 16; ++i) hout[(size_t)m * K + base + i] = v[i];
    quant_store16(v, s, m, M, K, aq, asc, absum, act32);
#pragma unroll
    for (int i = 0; i < 16; ++i) rowv[base + i] = v[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // ---- h8r_act on the row in local memory: sub-group bk rotates block bk, lane l holds values 64 r + 4 l + b
  const int nblk = K / HB, bk = get_sub_group_id();
  const uint l = lane;
  float hv[HN_NV];
  float amax = 0.0f;
  if (bk < nblk) {
#pragma unroll
    for (int rr = 0; rr < HN_NV / 4; ++rr)
#pragma unroll
      for (int b = 0; b < 4; ++b)
        hv[4 * rr + b] = h8n_signed(rowv[bk * HB + 64 * rr + 4 * l + b], 64 * rr + 4 * l + b);
    h8n_fwht_regs(hv);
#pragma unroll
    for (int j = 0; j < HN_NV; ++j) amax = fmax(amax, fabs(hv[j]));
    amax = sub_group_reduce_max(amax);
    if (l == 0) red[bk] = amax;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (bk < nblk) {
    amax = 0.0f;
    for (int i = 0; i < nblk; ++i) amax = fmax(amax, red[i]);
    if (bk == 0 && l == 0) asch[m] = amax / (127.0f * sqrt((float)HB));
    const float iv = amax > 0.0f ? 127.0f / amax : 0.0f;
#pragma unroll
    for (int rr = 0; rr < HN_NV / 4; ++rr) {
      const float q[4] = {hv[4 * rr], hv[4 * rr + 1], hv[4 * rr + 2], hv[4 * rr + 3]};
      intel_sub_group_block_write(aqh + (size_t)m * (K / 4) + (size_t)bk * (HB / 4) + rr * SG, h8n_pack_q(q, iv));
    }
  }
}
#endif  // H8NORM

inline float gelu_tanh(float x) { return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x))); }

// y = silu(gate) * up (or gelu_tanh(gate) * up) from gu[m] = [gate(F) | up(F)], quantized (K = F).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void swiglu_quant(__global const float* gu, int F, int M, int M_valid, __global char* aq,
                           __global float* asc, __global ushort* absum, int act32, int fused, int gelu) {
  const int m = get_group_id(1);
  const bool valid = m < M_valid;
  const int lane = get_sub_group_local_id();
  // gu rows: [gate | up] (2F), or with `fused` the GEMM already wrote silu(gate) * up (F)
  __global const float* gr = gu + (size_t)m * (fused ? F : 2 * F);
  // superblocks over every sub-group along dim 0: grid (256, M) = one work-group per row; for a few rows the engine
  // launches (16 * F/256, M) with local (16, 1), one sub-group per superblock
  const int nsg = get_num_groups(0) * get_num_sub_groups();
  for (int s = get_group_id(0) * get_num_sub_groups() + get_sub_group_id(); s < F / 256; s += nsg) {
    const int base = s * 256 + 16 * lane;
    // (the row's flags tested once, the 16 values of a lane as one vector load each: the per-element branches kept
    //  every load apart; the same arithmetic per element)
    float v[16];
    if (!valid) {
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = 0.0f;
    } else {
      const float16 g = vload16(0, gr + base);
      const float gv[16] = {g.s0, g.s1, g.s2, g.s3, g.s4, g.s5, g.s6, g.s7, g.s8, g.s9, g.sa, g.sb, g.sc, g.sd, g.se, g.sf};
      if (fused) {
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = gv[i];
      } else {
        const float16 u = vload16(0, gr + F + base);
        const float uv[16] = {u.s0, u.s1, u.s2, u.s3, u.s4, u.s5, u.s6, u.s7, u.s8, u.s9, u.sa, u.sb, u.sc, u.sd, u.se, u.sf};
        if (gelu) {
#pragma unroll
          for (int i = 0; i < 16; ++i) v[i] = gelu_tanh(gv[i]) * uv[i];
        } else {
#ifdef SWI_CLAMP  // (DeepSeek-V4.1's shared expert: silu(min(g, L)) * clamp(u, -L, L))
#pragma unroll
          for (int i = 0; i < 16; ++i) v[i] = silu(fmin(gv[i], SWI_CLAMP)) * clamp(uv[i], -SWI_CLAMP, SWI_CLAMP);
#else
#pragma unroll
          for (int i = 0; i < 16; ++i) v[i] = silu(gv[i]) * uv[i];
#endif
        }
      }
    }
    quant_store16(v, s, m, M, F, aq, asc, absum, act32);
  }
}

// ---- Whisper (zero/whisper_engine.h): its LayerNorms and the GELU of its FFN together with the next GEMM's
// quantization, one launch instead of two or three (the values as zero/kernels/whisper.cl wh_bias_res /
// wh_layernorm / wh_bias_gelu compute them, quantized as swiglu_quant does; no f32 copy written in between)
inline float wh_gelu_erf(float x) { return 0.5f * x * (1.0f + erf(x * 0.70710678118654752f)); }
// with_res: x[m] += y[m] + rb first (the residual after a projection; rows < M_valid); then the LayerNorm (eps 1e-5,
// g, b) of row m, quantized; rows >= M_valid quantize as zeros. grid (256, Mp), local (256, 1).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void wh_ln_quant(__global float* x, __global const float* y, __global const half* rb, int with_res, int D,
                          int M, int M_valid, __global const half* g, __global const half* b, __global char* aq,
                          __global float* asc, __global ushort* absum, int act32) {
  const int m = get_group_id(1), l = get_local_id(0), lane = get_sub_group_local_id();
  const bool valid = m < M_valid;
  __global float* r = x + (size_t)m * D;
  __local float red[256];
  float mu = 0.0f, inv = 0.0f;
  if (valid) {  // (the work-group's own row: the branch is the same for all of it)
    if (with_res) {
      __global const float* yr = y + (size_t)m * D;
      for (int i = l; i < D; i += 256) r[i] += yr[i] + vload_half(i, rb);
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
    float s = 0.0f;
    for (int i = l; i < D; i += 256) s += r[i];
    red[l] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 128; k > 0; k >>= 1) {
      if (l < k) red[l] += red[l + k];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    mu = red[0] / D;
    barrier(CLK_LOCAL_MEM_FENCE);
    float q = 0.0f;
    for (int i = l; i < D; i += 256) { const float d = r[i] - mu; q += d * d; }
    red[l] = q;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 128; k > 0; k >>= 1) {
      if (l < k) red[l] += red[l + k];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    inv = rsqrt(red[0] / D + 1e-5f);
  }
  for (int s = get_sub_group_id(); s < D / 256; s += get_num_sub_groups()) {
    const int base = s * 256 + 16 * lane;
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = valid ? (r[base + i] - mu) * inv * vload_half(base + i, g) + vload_half(base + i, b) : 0.0f;
    quant_store16(v, s, m, M, D, aq, asc, absum, act32);
  }
}
// gelu(ff[m] + b) quantized (the FFN's hidden rows); rows >= M_valid as zeros. grid (256, Mp), local (256, 1).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void wh_gelu_quant(__global const float* ff, int F, __global const half* b, int M, int M_valid,
                            __global char* aq, __global float* asc, __global ushort* absum, int act32) {
  const int m = get_group_id(1), lane = get_sub_group_local_id();
  const bool valid = m < M_valid;
  __global const float* fr = ff + (size_t)m * F;
  for (int s = get_sub_group_id(); s < F / 256; s += get_num_sub_groups()) {
    const int base = s * 256 + 16 * lane;
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = valid ? wh_gelu_erf(fr[base + i] + vload_half(base + i, b)) : 0.0f;
    quant_store16(v, s, m, M, F, aq, asc, absum, act32);
  }
}

// Causal depthwise conv (kernel 4) + SiLU over qkv channels, then L2-norm of each q/k head (128).
// q is additionally scaled by 1/sqrt(128) (the delta-net output scale). One sub-group per (128-channel
// group, CT consecutive tokens; CT = 1 measured fastest), with vector loads; groups [0, nqk) are q heads,
// [nqk, 2nqk) k heads, the rest v heads. grid (C/128*16, ceil(M/CT)).
#ifndef CT
#define CT 1
#endif
// -DCONV_UPD (one token, not a verification pass): conv_state_update fused in. With M_valid = 1 each lane already
// holds its channels' new history (h after the step), so it stores it: the same values, one launch fewer.
#ifdef CONV_UPD
#define CSTATE_T __global float*
#else
#define CSTATE_T __global const float*
#endif
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void gdn_conv(__global const float* qkv, CSTATE_T cstate, __global const float* cw, int C,
                       int nqk, int M_valid, float eps, __global float* out) {
  const int m0 = get_global_id(1) * CT, hg = get_global_id(0) / SG, lane = get_sub_group_local_id();
  const int c0 = hg * 128 + lane * 8;
  if (m0 >= M_valid) return;
  // taps: cw [c][4] -> wt[t] = weights of tap t for channels c0..c0+7
  const float16 wa = vload16(0, cw + (size_t)c0 * 4), wb = vload16(1, cw + (size_t)c0 * 4);
  const float8 wt0 = (float8)(wa.s0, wa.s4, wa.s8, wa.sc, wb.s0, wb.s4, wb.s8, wb.sc);
  const float8 wt1 = (float8)(wa.s1, wa.s5, wa.s9, wa.sd, wb.s1, wb.s5, wb.s9, wb.sd);
  const float8 wt2 = (float8)(wa.s2, wa.s6, wa.sa, wa.se, wb.s2, wb.s6, wb.sa, wb.se);
  const float8 wt3 = (float8)(wa.s3, wa.s7, wa.sb, wa.sf, wb.s3, wb.s7, wb.sb, wb.sf);
  float8 h[3];
#pragma unroll
  for (int t = 0; t < 3; ++t) {
    const int tt = m0 - 3 + t;
    h[t] = tt >= 0 ? vload8(0, qkv + (size_t)tt * C + c0) : vload8(0, cstate + (size_t)(tt + 3) * C + c0);
  }
  const bool qk = hg < 2 * nqk;
  const float qscale = hg < nqk ? rsqrt(128.0f) : 1.0f;
  float8 xs[CT];
#pragma unroll
  for (int j = 0; j < CT; ++j) xs[j] = m0 + j < M_valid ? vload8(0, qkv + (size_t)(m0 + j) * C + c0) : (float8)0.0f;
#pragma unroll
  for (int j = 0; j < CT; ++j) {
    const int m = m0 + j;
    if (m >= M_valid) break;
    const float8 x = xs[j];
    const float8 acc = wt0 * h[0] + wt1 * h[1] + wt2 * h[2] + wt3 * x;
    float8 v = acc / (1.0f + exp(-acc));
    if (qk) {
      const float ss = sub_group_reduce_add(dot(v.lo, v.lo) + dot(v.hi, v.hi));
      v *= rsqrt(ss + eps) * qscale;
    }
    vstore8(v, 0, out + (size_t)m * C + c0);
    h[0] = h[1];
    h[1] = h[2];
    h[2] = x;
  }
#ifdef CONV_UPD
  if (M_valid == 1) {
    vstore8(h[0], 0, cstate + c0);
    vstore8(h[1], 0, cstate + (size_t)C + c0);
    vstore8(h[2], 0, cstate + (size_t)2 * C + c0);
  }
#endif
}
#undef CSTATE_T

// Keeps the last 3 raw qkv rows as conv history. grid (C).
__kernel void conv_state_update(__global const float* qkv, __global float* cstate, int C, int M_valid) {
  const int c = get_global_id(0);
  if (c >= C) return;
  const float old[3] = {cstate[c], cstate[C + c], cstate[2 * C + c]};
  float nw[3];
  for (int i = 0; i < 3; ++i) {
    const int t = M_valid - 3 + i;
    nw[i] = t >= 0 ? qkv[(size_t)t * C + c] : old[i + M_valid];
  }
  for (int i = 0; i < 3; ++i) cstate[(size_t)i * C + c] = nw[i];
}

// In place: ab[m] = [alpha | beta] (raw projections) -> [decay | beta] with
// decay = exp(softplus(alpha + dt_bias) * ssm_a), beta = sigmoid(beta). n = M_valid * 2 * nv.
__kernel void gdn_gates(__global float* ab, __global const float* dt_bias, __global const float* ssm_a, int nv, int n) {
  const int i = get_global_id(0);
  if (i >= n) return;
  const int c = i % (2 * nv);
  const float x = ab[i];
  if (c < nv) {
    const float a = x + dt_bias[c];
    ab[i] = exp((a > 20.0f ? a : log(1.0f + exp(a))) * ssm_a[c]);
  } else {
    ab[i] = 1.0f / (1.0f + exp(-x));
  }
}

// Gated delta rule, sequential over tokens. State S[h][j][i] (i = key dim) f32.
// v-head h uses k-head h % nk (ggml repeat semantics). SIMD8: each lane owns one full state column j (128
// floats in registers). q and k are block-read across the 8 lanes and used through constant-index lane
// broadcasts (free register regioning), so no reductions or redundant loads are needed. Next-token
// operands are prefetched one step ahead. Needs the 256-GRF build.
// x: conv output [M][C] = [q (nk*128) | k (nk*128) | v (nv*128)]; ab: [M][2*nv] = [decay | beta].
// grid: nv*128 work-items. The state is read from S and written to Sout (the same buffer: in place; a speculative
// verification pass writes elsewhere and keeps S for the replay of its accepted tokens).
// (-DNO_SG8, GPUs without 8-lane sub-groups (Xe2 / Xe3): 16 lanes, the same arithmetic in the same order; their
//  64-byte registers hold the 128-float column as Xe-LPG's 32-byte ones do at 8 lanes)
#ifdef NO_SG8
#define GW 16
#else
#define GW 8
#endif
#define BCW(x, i) intel_sub_group_shuffle((x), (i))
// -DFUSE_GATES (not a verification pass): gdn_gates fused in; ab holds the raw [alpha | beta] projections, loaded
// as before (one token ahead), and each work-item turns its head's two into [decay | beta] where a token uses them,
// with gdn_gates' expressions: the same values, one launch fewer.
#ifdef FUSE_GATES
#define GDN_DECAY(x) gdn_decay((x), dtb, ssa)
#define GDN_BETA(x) (1.0f / (1.0f + exp(-(x))))
inline float gdn_decay(float x, float b, float sa) {
  const float a = x + b;
  return exp((a > 20.0f ? a : log(1.0f + exp(a))) * sa);
}
#else
#define GDN_DECAY(x) (x)
#define GDN_BETA(x) (x)
#endif
__attribute__((intel_reqd_sub_group_size(GW)))
__kernel void gdn_recur(__global const float* x, int C, __global const float* ab, int nk, int nv,
                        __global const float* S, int M_valid, __global float* o, __global float* Sout
#ifdef FUSE_GATES
                        , __global const float* dt_bias, __global const float* ssm_a
#endif
                        ) {
  const int col = get_global_id(0), lane = get_sub_group_local_id();
  const int h = col / 128, j = col % 128, kh = h % nk, jb = j - lane;
  // the first token's operands are requested before the state, so their latency hides under the state loads
  const int qo = kh * 128, ko = nk * 128 + kh * 128, vo = 2 * nk * 128 + h * 128 + jb;
  uint8 kn0 = 0, kn1 = 0, qn0 = 0, qn1 = 0;
  uint vn = 0;
  float dn = 0.0f, bn = 0.0f;
  if (M_valid > 0) {
    kn0 = intel_sub_group_block_read8((__global const uint*)(x + ko));
    qn0 = intel_sub_group_block_read8((__global const uint*)(x + qo));
#if GW == 8
    kn1 = intel_sub_group_block_read8((__global const uint*)(x + ko + 64));
    qn1 = intel_sub_group_block_read8((__global const uint*)(x + qo + 64));
#endif
    vn = intel_sub_group_block_read((__global const uint*)(x + vo));
    dn = ab[h];
    bn = ab[nv + h];
  }
#ifdef FUSE_GATES
  const float dtb = dt_bias[h], ssa = ssm_a[h];
#endif
  float s[128];
#ifdef GDN_LINE_SWAP
  // lanes 2c and 2c+1 move the two 32-byte halves of one 64-byte line (message 0: the even lane's row, message 1:
  // the odd lane's row) and swap 8 floats with each other
  const int hf = lane & 1;
  const float8 hm = (float8)(as_float(-(uint)hf));  // all ones on odd lanes: bitselect(a, b, hm) = hf ? b : a
  __global const float* sp0 = S + ((size_t)h * 128 + (j & ~1)) * 128 + 8 * hf;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const float8 e0 = vload8(0, sp0 + 16 * q), e1 = vload8(0, sp0 + 128 + 16 * q);
    const float8 r = intel_sub_group_shuffle_xor(bitselect(e1, e0, hm), 1);
    const float8 lo = bitselect(e0, r, hm), hi = bitselect(r, e1, hm);
    s[16 * q] = lo.s0; s[16 * q + 1] = lo.s1; s[16 * q + 2] = lo.s2; s[16 * q + 3] = lo.s3;
    s[16 * q + 4] = lo.s4; s[16 * q + 5] = lo.s5; s[16 * q + 6] = lo.s6; s[16 * q + 7] = lo.s7;
    s[16 * q + 8] = hi.s0; s[16 * q + 9] = hi.s1; s[16 * q + 10] = hi.s2; s[16 * q + 11] = hi.s3;
    s[16 * q + 12] = hi.s4; s[16 * q + 13] = hi.s5; s[16 * q + 14] = hi.s6; s[16 * q + 15] = hi.s7;
  }
#else
  // 32 bytes per lane and message (vload16 = two d32x8 messages): all 16 state loads fit the 16 send tokens
  __global const float* sp = S + ((size_t)h * 128 + j) * 128;
#pragma unroll
  for (int i = 0; i < 128; i += 16) {
    const float16 v16 = vload16(i / 16, sp);
    s[i] = v16.s0; s[i + 1] = v16.s1; s[i + 2] = v16.s2; s[i + 3] = v16.s3;
    s[i + 4] = v16.s4; s[i + 5] = v16.s5; s[i + 6] = v16.s6; s[i + 7] = v16.s7;
    s[i + 8] = v16.s8; s[i + 9] = v16.s9; s[i + 10] = v16.sa; s[i + 11] = v16.sb;
    s[i + 12] = v16.sc; s[i + 13] = v16.sd; s[i + 14] = v16.se; s[i + 15] = v16.sf;
  }
#endif
  for (int t = 0; t < M_valid; ++t) {
    // (element i of q / k: register i / GW of lane i % GW, as the sub-group block reads lay them out)
    const float kr[16] = {as_float(kn0.s0), as_float(kn0.s1), as_float(kn0.s2), as_float(kn0.s3),
                          as_float(kn0.s4), as_float(kn0.s5), as_float(kn0.s6), as_float(kn0.s7),
                          as_float(kn1.s0), as_float(kn1.s1), as_float(kn1.s2), as_float(kn1.s3),
                          as_float(kn1.s4), as_float(kn1.s5), as_float(kn1.s6), as_float(kn1.s7)};
    const float qr[16] = {as_float(qn0.s0), as_float(qn0.s1), as_float(qn0.s2), as_float(qn0.s3),
                          as_float(qn0.s4), as_float(qn0.s5), as_float(qn0.s6), as_float(qn0.s7),
                          as_float(qn1.s0), as_float(qn1.s1), as_float(qn1.s2), as_float(qn1.s3),
                          as_float(qn1.s4), as_float(qn1.s5), as_float(qn1.s6), as_float(qn1.s7)};
    const float v = as_float(vn), decay = GDN_DECAY(dn), beta = GDN_BETA(bn);
    if (t + 1 < M_valid) {
      __global const float* nrow = x + (size_t)(t + 1) * C;
      kn0 = intel_sub_group_block_read8((__global const uint*)(nrow + ko));
      qn0 = intel_sub_group_block_read8((__global const uint*)(nrow + qo));
#if GW == 8
      kn1 = intel_sub_group_block_read8((__global const uint*)(nrow + ko + 64));
      qn1 = intel_sub_group_block_read8((__global const uint*)(nrow + qo + 64));
#endif
      vn = intel_sub_group_block_read((__global const uint*)(nrow + vo));
      dn = ab[(size_t)(t + 1) * 2 * nv + h];
      bn = ab[(size_t)(t + 1) * 2 * nv + nv + h];
    }
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
    for (int i = 0; i < 128; i += 4) {
      s[i] *= decay; s[i + 1] *= decay; s[i + 2] *= decay; s[i + 3] *= decay;
      a0 = fma(s[i], BCW(kr[i / GW], i % GW), a0);
      a1 = fma(s[i + 1], BCW(kr[(i + 1) / GW], (i + 1) % GW), a1);
      a2 = fma(s[i + 2], BCW(kr[(i + 2) / GW], (i + 2) % GW), a2);
      a3 = fma(s[i + 3], BCW(kr[(i + 3) / GW], (i + 3) % GW), a3);
    }
    const float d = (v - ((a0 + a1) + (a2 + a3))) * beta;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
#pragma unroll
    for (int i = 0; i < 128; i += 4) {
      s[i] = fma(BCW(kr[i / GW], i % GW), d, s[i]);
      s[i + 1] = fma(BCW(kr[(i + 1) / GW], (i + 1) % GW), d, s[i + 1]);
      s[i + 2] = fma(BCW(kr[(i + 2) / GW], (i + 2) % GW), d, s[i + 2]);
      s[i + 3] = fma(BCW(kr[(i + 3) / GW], (i + 3) % GW), d, s[i + 3]);
      b0 = fma(s[i], BCW(qr[i / GW], i % GW), b0);
      b1 = fma(s[i + 1], BCW(qr[(i + 1) / GW], (i + 1) % GW), b1);
      b2 = fma(s[i + 2], BCW(qr[(i + 2) / GW], (i + 2) % GW), b2);
      b3 = fma(s[i + 3], BCW(qr[(i + 3) / GW], (i + 3) % GW), b3);
    }
    intel_sub_group_block_write((__global uint*)(o + (size_t)t * nv * 128 + h * 128 + jb),
                                as_uint((b0 + b1) + (b2 + b3)));
  }
#ifdef GDN_LINE_SWAP
  __global float* so0 = Sout + ((size_t)h * 128 + (j & ~1)) * 128 + 8 * hf;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const float8 lo = (float8)(s[16 * q], s[16 * q + 1], s[16 * q + 2], s[16 * q + 3], s[16 * q + 4], s[16 * q + 5],
                               s[16 * q + 6], s[16 * q + 7]);
    const float8 hi = (float8)(s[16 * q + 8], s[16 * q + 9], s[16 * q + 10], s[16 * q + 11], s[16 * q + 12],
                               s[16 * q + 13], s[16 * q + 14], s[16 * q + 15]);
    const float8 r = intel_sub_group_shuffle_xor(bitselect(hi, lo, hm), 1);
    vstore8(bitselect(lo, r, hm), 0, so0 + 16 * q);
    vstore8(bitselect(r, hi, hm), 0, so0 + 128 + 16 * q);
  }
#else
  __global float* so = Sout + ((size_t)h * 128 + j) * 128;
#pragma unroll
  for (int i = 0; i < 128; i += 16)
    vstore16((float16)(s[i], s[i + 1], s[i + 2], s[i + 3], s[i + 4], s[i + 5], s[i + 6], s[i + 7], s[i + 8], s[i + 9],
                       s[i + 10], s[i + 11], s[i + 12], s[i + 13], s[i + 14], s[i + 15]), i / 16, so);
#endif
}
#undef GW
#undef GDN_DECAY
#undef GDN_BETA

// y = rmsnorm_head(o) * w * silu(z) per 128-dim head, quantized (K = nv*128). grid (256, M).
__attribute__((intel_reqd_sub_group_size(SG)))
#ifdef GATE_SIGMOID  // (Qwen3.8 Flash Next: the DeltaNet output gate is a sigmoid)
#define GDN_GATE(z) (1.0f / (1.0f + exp(-(z))))
#else
#define GDN_GATE(z) silu(z)
#endif
__kernel void gated_norm_quant(__global const float* o, __global const float* z, __global const float* w, int K,
                               int M, int M_valid, float eps, __global char* aq, __global float* asc,
                               __global ushort* absum, int act32) {
  const int m = get_group_id(1);
  const bool valid = m < M_valid;
  const int lane = get_sub_group_local_id();
  // (superblocks over every sub-group of the grid: grid (256, M) is one work-group per row as before; a 1-8 token
  //  pass may launch several along dim 0 (zero/engine.h wide_rows), the same arithmetic per superblock)
  for (int s = get_group_id(0) * get_num_sub_groups() + get_sub_group_id(); s < K / 256;
       s += get_num_groups(0) * get_num_sub_groups()) {
    const int base = s * 256 + 16 * lane, d0 = (16 * lane) & 127;
    float v[16];
    if (valid) {  // (the row flag tested once, vector loads; the same arithmetic. `valid` is the work-group's)
      const float16 oa = vload16(0, o + (size_t)m * K + base);
      const float ov[16] = {oa.s0, oa.s1, oa.s2, oa.s3, oa.s4, oa.s5, oa.s6, oa.s7, oa.s8, oa.s9, oa.sa, oa.sb, oa.sc, oa.sd, oa.se, oa.sf};
      float ss = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        v[i] = ov[i];
        ss += v[i] * v[i];
      }
      ss += intel_sub_group_shuffle_xor(ss, 1);
      ss += intel_sub_group_shuffle_xor(ss, 2);
      ss += intel_sub_group_shuffle_xor(ss, 4);
      const float r = rsqrt(ss / 128.0f + eps);
      const float16 wa = vload16(0, w + d0), za = vload16(0, z + (size_t)m * K + base);
      const float wv[16] = {wa.s0, wa.s1, wa.s2, wa.s3, wa.s4, wa.s5, wa.s6, wa.s7, wa.s8, wa.s9, wa.sa, wa.sb, wa.sc, wa.sd, wa.se, wa.sf};
      const float zv[16] = {za.s0, za.s1, za.s2, za.s3, za.s4, za.s5, za.s6, za.s7, za.s8, za.s9, za.sa, za.sb, za.sc, za.sd, za.se, za.sf};
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = v[i] * r * wv[i] * GDN_GATE(zv[i]);
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = 0.0f;
    }
    quant_store16(v, s, m, M, K, aq, asc, absum, act32);
  }
}



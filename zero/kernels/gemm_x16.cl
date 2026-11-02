// The prompt GEMM in f16 on the XMX engines of discrete Xe2 / Xe3 GPUs (Arc B580, Pro B60): zero/xe2/engine_xe2.h.
// Not built on the iGPUs. C[M][ldc] (+)= A[M][K] * W[N][K]^T (at column offset c_off) for the W8 weight layouts
// (100..103: int8 q + per-16 / per-32 int8 scales (+ per-16 mins), kernels/gemm.cl).
//
// Why another kernel: gemm.cl -DXMX multiplies int8 by int8, but every 16- or 32-value block of a weight row has its
// own scale, so each DPAS (8 tokens x 16 columns x 32 values) starts from zero and its 8 results are scaled one by
// one on the vector unit, about as much work as the DP4a dot products it replaces (B60: XMX 450 vs DP4a 406 tok/s).
// Here the weights are decoded to f16 with their scales applied, the activations are f16 too, and one DPAS
// accumulator per 8 tokens runs over the whole reduction: the vector unit only decodes weights, once per 16 columns
// for X16_TM tokens.
//
// act_half: the int8 activations (aq, scales asc per 32 or per 256 values) to f16, divided by the row's largest scale
// (rs[m]) so every value is within +-127: no f16 overflow whatever the activations. The GEMM multiplies rs[m] back.
// Weights are decoded times 256 (WSCALE; values of LLM weights are far below 255) so small ones stay normal f16.
// -DX16_EMU: DPAS emulated with shuffles (tests on GPUs without XMX: same products, f32 sums in another order).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SG 16
#ifndef X16_TM
#define X16_TM 32  // tokens per sub-group: X16_TM / 8 accumulators
#endif
#define NRG (X16_TM / 8)
#define WSCALE 256.0f

#define SC32 (QTYPE & 1)
#define HAS_MIN (QTYPE >= 102)
#define DPJ 16
#define HDR 10

// grid (256, M), local (256, 1): one work-group per row
__kernel void act_half(__global const char* restrict aq, __global const float* restrict asc, int M, int K, int act32,
                       __global half* restrict ah, __global float* restrict rs) {
  const int m = get_group_id(1), lid = get_local_id(0);
  const int bs = act32 ? 32 : 256, nb = K / bs;
  float mx = 0.0f;
  for (int b = lid; b < nb; b += 256) mx = fmax(mx, fabs(asc[(size_t)b * M + m]));
  mx = work_group_reduce_max(mx);
  const float inv = mx > 0.0f ? 1.0f / mx : 0.0f;
  if (lid == 0) rs[m] = mx;
  __global const char* ap = aq + (size_t)m * K;
  __global half* hp = ah + (size_t)m * K;
  for (int k = lid * 4; k < K; k += 1024) {
    const float s = asc[(size_t)(k / bs) * M + m] * inv;
    vstore_half4_rte(convert_float4(vload4(0, ap + k)) * s, 0, hp + k);
  }
}

#ifdef X16_EMU
// one DPAS 8 x 16 x 16 (f16, f32 sums): a: lane l holds A[r][l] in component r; b: lane n = column n, its 16 values
// as 8 pairs (k = 2i low half, 2i + 1 high); result: component r = row r, lane = column
inline float8 x16_mad(short8 a, int8 b, float8 acc) {
  const int bw[8] = {b.s0, b.s1, b.s2, b.s3, b.s4, b.s5, b.s6, b.s7};
  float c[8] = {acc.s0, acc.s1, acc.s2, acc.s3, acc.s4, acc.s5, acc.s6, acc.s7};
  const uint ar[8] = {as_ushort(a.s0), as_ushort(a.s1), as_ushort(a.s2), as_ushort(a.s3),
                      as_ushort(a.s4), as_ushort(a.s5), as_ushort(a.s6), as_ushort(a.s7)};
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const half2 bh = as_half2(bw[k >> 1]);
    const float bk = (k & 1) ? (float)bh.s1 : (float)bh.s0;
#pragma unroll
    for (int r = 0; r < 8; ++r) c[r] = fma((float)as_half((ushort)intel_sub_group_shuffle(ar[r], k)), bk, c[r]);
  }
  return (float8)(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
}
#else
#define x16_mad(a, b, acc) intel_sub_group_f16_f16_matrix_mad_k16((a), (b), (acc))
#endif

// 4 int8 weights of a word, times s, less mn, as 2 words of f16 pairs (lower half: the lower k)
inline int2 x16_dec(uint w, float s, float mn) {
  return as_int2(convert_half4_rte(fma(convert_float4(as_char4(w)), (float4)(s), (float4)(-mn))));
}

// grid (N * rows, WGY), local (16, WGY), rows = M / (X16_TM * WGY): the work-group and row-group order of gemm.cl
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void gemm_x16(__global const uint* restrict wq, __global const uint* restrict wh,
                       __global const half* restrict ah, __global const float* restrict rs, __global float* restrict C,
                       int M, int N, int K, int ldc, int c_off, int accumulate, int rgroup, int s_begin, int s_end) {
  const int NC = N / SG, rows = M / (X16_TM * (int)get_local_size(1));
  const int w = get_group_id(0), grp = w / (rgroup * NC), rem = w % (rgroup * NC);
  const int rin = min(rgroup, rows - grp * rgroup);
  const int g = rem / rin, row = grp * rgroup + rem % rin;
  const int m0 = (row * (int)get_local_size(1) + (int)get_local_id(1)) * X16_TM;
  const int nsb = K / 256, lane = get_sub_group_local_id();
  __global const uint* wqp = wq + (size_t)g * nsb * (4 * DPJ * SG);
  __global const uint* whp = wh + (size_t)g * nsb * (HDR * SG);
  __global const ushort* ap = (__global const ushort*)ah + (size_t)m0 * K;

  float8 acc[NRG];
#pragma unroll
  for (int i = 0; i < NRG; ++i) acc[i] = (float8)(0.0f);

  for (int s = s_begin; s < s_end; ++s) {
    __global const uint* hp = whp + s * HDR * SG;
    const uint2 hd = intel_sub_group_block_read2(hp);
    const float d = as_float(hd.s0) * WSCALE;
    const uint4 hs4 = intel_sub_group_block_read4(hp + 2 * SG);
    const uint hsc[4] = {hs4.s0, hs4.s1, hs4.s2, hs4.s3};
#if HAS_MIN
    const float dmin = as_float(hd.s1) * WSCALE;
    const uint4 hm4 = intel_sub_group_block_read4(hp + 6 * SG);
    const uint hmn[4] = {hm4.s0, hm4.s1, hm4.s2, hm4.s3};
#endif
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // ---- this lane's column, values 64j .. 64j + 63 of the superblock, as 4 x 16 f16 (16-blocks c = 0..3)
      __global const uint* dp = wqp + (s * 4 + j) * DPJ * SG;
      const uint8 r0 = intel_sub_group_block_read8(dp), r1 = intel_sub_group_block_read8(dp + 8 * SG);
      const uint wd[16] = {r0.s0, r0.s1, r0.s2, r0.s3, r0.s4, r0.s5, r0.s6, r0.s7,
                           r1.s0, r1.s1, r1.s2, r1.s3, r1.s4, r1.s5, r1.s6, r1.s7};
      int8 B[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
#if SC32
        const float sc = d * (float)(char)((hsc[j] >> (8 * (c & 2))) & 0xFF);  // (bytes 0 and 2: the two 32-blocks)
#else
        const float sc = d * (float)(char)((hsc[j] >> (8 * c)) & 0xFF);
#endif
#if HAS_MIN
        const float mn = dmin * (float)((hmn[j] >> (8 * c)) & 0xFF);
#else
        const float mn = 0.0f;
#endif
        const int2 p0 = x16_dec(wd[4 * c], sc, mn), p1 = x16_dec(wd[4 * c + 1], sc, mn);
        const int2 p2 = x16_dec(wd[4 * c + 2], sc, mn), p3 = x16_dec(wd[4 * c + 3], sc, mn);
        B[c] = (int8)(p0.s0, p0.s1, p1.s0, p1.s1, p2.s0, p2.s1, p3.s0, p3.s1);
      }
      // ---- 8 tokens at a time: lane l of row r holds its values l, 16 + l, 32 + l, 48 + l (one per 16-block)
      const int k0 = s * 256 + j * 64;
#pragma unroll
      for (int i = 0; i < NRG; ++i) {
        ushort4 a[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) a[r] = intel_sub_group_block_read_us4(ap + (size_t)(8 * i + r) * K + k0);
        acc[i] = x16_mad(as_short8((ushort8)(a[0].s0, a[1].s0, a[2].s0, a[3].s0, a[4].s0, a[5].s0, a[6].s0, a[7].s0)), B[0], acc[i]);
        acc[i] = x16_mad(as_short8((ushort8)(a[0].s1, a[1].s1, a[2].s1, a[3].s1, a[4].s1, a[5].s1, a[6].s1, a[7].s1)), B[1], acc[i]);
        acc[i] = x16_mad(as_short8((ushort8)(a[0].s2, a[1].s2, a[2].s2, a[3].s2, a[4].s2, a[5].s2, a[6].s2, a[7].s2)), B[2], acc[i]);
        acc[i] = x16_mad(as_short8((ushort8)(a[0].s3, a[1].s3, a[2].s3, a[3].s3, a[4].s3, a[5].s3, a[6].s3, a[7].s3)), B[3], acc[i]);
      }
    }
  }

  // ---- C: each row's result times its activation scale (rs) and 1 / WSCALE; accumulate as gemm.cl
#pragma unroll
  for (int i = 0; i < NRG; ++i) {
    const float res[8] = {acc[i].s0, acc[i].s1, acc[i].s2, acc[i].s3, acc[i].s4, acc[i].s5, acc[i].s6, acc[i].s7};
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int m = m0 + 8 * i + r;
      __global uint* cp = (__global uint*)(C + c_off + (size_t)m * ldc + g * SG);
      float v = res[r] * (rs[m] * (1.0f / WSCALE));
      if (accumulate == 1) {
        v += as_float(intel_sub_group_block_read(cp));
      } else if (accumulate == 2) {  // fused SwiGLU: C holds the gate projection, v the up projection
        const float gt = as_float(intel_sub_group_block_read(cp));
        v *= gt / (1.0f + exp(-gt));
      }
      intel_sub_group_block_write(cp, as_uint(v));
    }
  }
}

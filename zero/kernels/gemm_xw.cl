// The H8R prompt GEMM on XMX with a wider tile per sub-group (QUEUE 33.4; zero/xe2/engine_xw.h, ARC_GEMM_XW=1):
// C[M][ldc] (+)= A[M][K] * W[N][K]^T at column offset c_off, for the H8R layout only (gemm.cl QTYPE 111: int8
// weights in the W8 values layout with one scale per column, int8 activations with one scale per token, the dot
// product over a split's whole K one integer sum).
//
// gemm.cl's XMX kernel gives a sub-group 16 or 32 tokens x 16 columns: each 64-value chunk of a column group's
// weights (1 KB) is loaded for 16 / 32 tokens only, and the activation rows are loaded again by every column group
// (shared through L1 at best). Here a sub-group holds XW_TM tokens x XW_NC column groups (default 32 x 64): the
// activations of a chunk are loaded once for XW_NC * 16 columns, a column group's weights once for XW_TM tokens,
// twice the DPAS work per byte loaded at 32 x 64. The accumulators are TM / 8 x NC int8 vectors (128 registers at
// 32 x 64: the 256-register build).
//
// Bit for bit gemm.cl's: the integer sums are exact in any order (|sum| <= 127 * 127 * K, as there), split-K pieces
// are the same (the host passes s_begin / s_end as gemm.cl gets them), and the epilogue is gemm.cl's expression,
// (token scale * column scale) * (float)sum, then the same accumulate modes. Checked at start against gemm.cl
// without XMX (engine_xw.h xw_check), bit for bit.
//
// -DXW_EMU: DPAS emulated with DP4a and shuffles (tests on GPUs without XMX: ARC_GEMM_XW_TRY=1).
// -DX2D: the activations by 2D block loads, as gemm.cl -DX2D (8 rows x 64 bytes a load).
// Grid: (16 * (N / (16 * XW_NC)) * rows, WGY), local (16, WGY), rows = M / (XW_TM * WGY); the work-groups sweep
// `rgroup` token rows together column block by column block, as gemm.cl, so a weight slice is reused while in L3.

#define SG 16
#ifndef XW_TM
#define XW_TM 32
#endif
#ifndef XW_NC
#define XW_NC 4
#endif
#if XW_TM != 16 && XW_TM != 32
#error "XW_TM must be 16 or 32"
#endif
#if XW_NC != 1 && XW_NC != 2 && XW_NC != 4
#error "XW_NC must be 1, 2 or 4"
#endif
#define XW_TG (XW_TM / 8)  // DPAS row groups of 8 tokens
#define BCAST(x, i) intel_sub_group_shuffle((x), (i))

// c + A (8 rows x 32 int8: lane l holds values 2l, 2l + 1 of each row) x B (32 x 16: lane n = column n, its 32 values
// as 8 words of 4), int32, component r = row r, lane = column
inline int8 xw_mad(short8 a, uint8 b, int8 c) {
#ifdef XW_EMU
  const int av[8] = {a.s0, a.s1, a.s2, a.s3, a.s4, a.s5, a.s6, a.s7};
  const uint bw[8] = {b.s0, b.s1, b.s2, b.s3, b.s4, b.s5, b.s6, b.s7};
  int cr[8] = {c.s0, c.s1, c.s2, c.s3, c.s4, c.s5, c.s6, c.s7};
#pragma unroll
  for (int w = 0; w < 8; ++w)
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      // (values 4w .. 4w + 3 of row r: lanes 2w and 2w + 1, two bytes each)
      const uint lo = (uint)intel_sub_group_shuffle(av[r], 2 * w) & 0xFFFFu;
      const uint hi = (uint)intel_sub_group_shuffle(av[r], 2 * w + 1) & 0xFFFFu;
      cr[r] += dot_4x8packed_ss_int(lo | (hi << 16), bw[w]);
    }
  return (int8)(cr[0], cr[1], cr[2], cr[3], cr[4], cr[5], cr[6], cr[7]);
#else
  return intel_sub_group_i8_i8_matrix_mad_k32(a, as_int8(b), c);
#endif
}

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void gemm_xw(__global const uint* restrict wq, __global const uint* restrict wh,
                      __global const char* restrict aq, __global const float* restrict asc,
                      __global float* restrict C, int M, int N, int K, int ldc, int c_off, int accumulate,
                      int rgroup, int s_begin, int s_end) {
  const int NB = N / (SG * XW_NC), rows = M / (XW_TM * (int)get_local_size(1));
  const int w = get_group_id(0), grp = w / (rgroup * NB), rem = w % (rgroup * NB);
  const int rin = min(rgroup, rows - grp * rgroup);
  const int cb = rem / rin, row = grp * rgroup + rem % rin;
  const int m0 = (row * (int)get_local_size(1) + (int)get_local_id(1)) * XW_TM;
  const int nsb = K / 256, g0 = cb * XW_NC;
#ifndef X2D
  __global const char* ap = aq + (size_t)m0 * K;
#endif

  int8 xacc[XW_TG][XW_NC];
#pragma unroll
  for (int t = 0; t < XW_TG; ++t)
#pragma unroll
    for (int c = 0; c < XW_NC; ++c) xacc[t][c] = (int8)(0);

  for (int s = s_begin; s < s_end; ++s) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // ---- this 64-value chunk of the XW_TM tokens' rows: al values 0..31, ah 32..63 of 8 rows each
      short8 al[XW_TG], ah[XW_TG];
#pragma unroll
      for (int t = 0; t < XW_TG; ++t) {
#ifdef X2D
        ushort d[16];
        intel_sub_group_2d_block_read_8b_8r32x2c(aq, K, M, K, (int2)(s * 256 + j * 64, m0 + 8 * t), d);
        al[t] = as_short8((ushort8)(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]));
        ah[t] = as_short8((ushort8)(d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]));
#else
        short ta[8], tb[8];
#pragma unroll
        for (int p = 0; p < 8; ++p) {
          __global const char* rp = ap + (size_t)(8 * t + p) * K + s * 256 + j * 64;
          ta[p] = as_short(intel_sub_group_block_read_us((__global const ushort*)rp));
          tb[p] = as_short(intel_sub_group_block_read_us((__global const ushort*)(rp + 32)));
        }
        al[t] = (short8)(ta[0], ta[1], ta[2], ta[3], ta[4], ta[5], ta[6], ta[7]);
        ah[t] = (short8)(tb[0], tb[1], tb[2], tb[3], tb[4], tb[5], tb[6], tb[7]);
#endif
      }
      // ---- each column group's 64 values (lo 0..31, hi 32..63), used by every row group
#pragma unroll
      for (int c = 0; c < XW_NC; ++c) {
        __global const uint* dp = wq + ((size_t)(g0 + c) * nsb * 4 + s * 4 + j) * (16 * SG);
        const uint8 lo = intel_sub_group_block_read8(dp), hi = intel_sub_group_block_read8(dp + 8 * SG);
#pragma unroll
        for (int t = 0; t < XW_TG; ++t) xacc[t][c] = xw_mad(ah[t], hi, xw_mad(al[t], lo, xacc[t][c]));
      }
    }
  }

  // ---- the one epilogue, gemm.cl's: the token's scale x the column's x the sum, then the accumulate mode
  float av[XW_TM / 16];
#pragma unroll
  for (int v = 0; v < XW_TM / 16; ++v)
    av[v] = as_float(intel_sub_group_block_read((__global const uint*)(asc + m0 + 16 * v)));
#pragma unroll
  for (int c = 0; c < XW_NC; ++c) {
    const int g = g0 + c;
    const float dcol = as_float(intel_sub_group_block_read(wh + (size_t)g * SG));
#pragma unroll
    for (int t = 0; t < XW_TG; ++t) {
      const int8 x = xacc[t][c];
      const int is[8] = {x.s0, x.s1, x.s2, x.s3, x.s4, x.s5, x.s6, x.s7};
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int m = 8 * t + r;
        const float res = BCAST(av[m / 16], m % 16) * dcol * (float)is[r];
        __global uint* cp = (__global uint*)(C + c_off + (size_t)(m0 + m) * ldc + g * SG);
        float v = res;
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
}

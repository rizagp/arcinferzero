// The H8R prompt GEMM on 8-lane XMX (Arrow Lake H's Arc 140T, Alchemist-kind DPAS) with a wider tile and coalesced
// loads (QUEUE 33.4b; zero/xe2/engine_xw.h, ARC_GEMM_XW=1 on a GPU with 8-lane sub-groups): C[M][ldc] (+)= A * W^T at
// column offset c_off, for gemm.cl's H8R layout (QTYPE 111: int8 weights with one scale per column, int8
// activations with one scale per token, a split's dot product one integer sum).
//
// gemm_x8.cl (the 140T's XMX GEMM) reads gemm.cl's 16-lane weight layout with 8 lanes one word a lane at a time: 16
// reads of 32 bytes per column half and 64 values, and each activation row in two 32-byte reads; ocloc arl-h counts
// ~110 instructions and ~13 sends per DPAS (docs/xe2-tuning.md section 12). Here:
//   - a 16-column group's 64 values of a chunk in 4 block reads of 8 words: lane l gets word w of column l (half 0)
//     in component 2k and of column 8 + l (half 1) in 2k + 1, as the layout interleaves them every 16 words;
//   - an activation row's 64 bytes of a chunk in one 2-word block read (values 0..31, 32..63);
//   - a sub-group holds XW_TM tokens x XW_NC column groups (2 * XW_NC DPAS column blocks of 8): each activation word
//     serves 16 * XW_NC columns, each weight word XW_TM tokens.
// The same values reach the same DPAS operands as in gemm_x8 (A: component r = row r's word, lane l = values 4l..4l+3
// of the 32; B: lane n = column n, component w = its values 4w..4w+3), the integer sums are exact, and the epilogue is
// gemm_x8's (token scale * column scale) * (float)sum with the same accumulate modes: bit for bit gemm.cl's DP4a
// results (checked at start, engine_xw.h xw_check). -DXW_EMU: DPAS as the DP4a dot products the extension's text
// gives (tests on a GPU without XMX).
// Grid: (8 * (N / (16 * XW_NC)) * rows, WGY), local (8, WGY), rows = M / (XW_TM * WGY); the row-group sweep of gemm.cl.

#define SG 8
#define LS 16  // lanes of the weight layout (gemm.cl's)
#ifndef XW_TM
#define XW_TM 16
#endif
#ifndef XW_NC
#define XW_NC 1
#endif
#if XW_TM != 16 && XW_TM != 32
#error "XW_TM must be 16 or 32"
#endif
#if XW_NC != 1 && XW_NC != 2
#error "XW_NC must be 1 or 2"
#endif
#define XW_TG (XW_TM / 8)   // DPAS row groups of 8 tokens
#define XW_NH (2 * XW_NC)   // DPAS column blocks of 8
#define BCAST(x, i) intel_sub_group_shuffle((x), (i))

#ifdef XW_EMU
inline int xw8_row(uint a, int8 b, int acc) {
  acc += dot_4x8packed_ss_int(BCAST(a, 0), (uint)b.s0);
  acc += dot_4x8packed_ss_int(BCAST(a, 1), (uint)b.s1);
  acc += dot_4x8packed_ss_int(BCAST(a, 2), (uint)b.s2);
  acc += dot_4x8packed_ss_int(BCAST(a, 3), (uint)b.s3);
  acc += dot_4x8packed_ss_int(BCAST(a, 4), (uint)b.s4);
  acc += dot_4x8packed_ss_int(BCAST(a, 5), (uint)b.s5);
  acc += dot_4x8packed_ss_int(BCAST(a, 6), (uint)b.s6);
  acc += dot_4x8packed_ss_int(BCAST(a, 7), (uint)b.s7);
  return acc;
}
inline int8 xw8_mad(int8 a, int8 b, int8 c) {
  return (int8)(xw8_row((uint)a.s0, b, c.s0), xw8_row((uint)a.s1, b, c.s1), xw8_row((uint)a.s2, b, c.s2),
                xw8_row((uint)a.s3, b, c.s3), xw8_row((uint)a.s4, b, c.s4), xw8_row((uint)a.s5, b, c.s5),
                xw8_row((uint)a.s6, b, c.s6), xw8_row((uint)a.s7, b, c.s7));
}
#else
#define xw8_mad(a, b, c) intel_sub_group_i8_i8_matrix_mad_k32((a), (b), (c))
#endif

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void gemm_xw8(__global const uint* restrict wq, __global const uint* restrict wh,
                       __global const char* restrict aq, __global const float* restrict asc,
                       __global float* restrict C, int M, int N, int K, int ldc, int c_off, int accumulate,
                       int rgroup, int s_begin, int s_end) {
  const int NB = N / (LS * XW_NC), rows = M / (XW_TM * (int)get_local_size(1));
  const int w = get_group_id(0), grp = w / (rgroup * NB), rem = w % (rgroup * NB);
  const int rin = min(rgroup, rows - grp * rgroup);
  const int cb = rem / rin, row = grp * rgroup + rem % rin;
  const int m0 = (row * (int)get_local_size(1) + (int)get_local_id(1)) * XW_TM;
  const int nsb = K / 256, g0 = cb * XW_NC;
  __global const char* ap = aq + (size_t)m0 * K;

  int8 xacc[XW_TG][XW_NH];
#pragma unroll
  for (int t = 0; t < XW_TG; ++t)
#pragma unroll
    for (int h = 0; h < XW_NH; ++h) xacc[t][h] = (int8)(0);

  for (int s = s_begin; s < s_end; ++s) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // ---- the XW_TM tokens' rows of this 64-value chunk: al values 0..31, ah 32..63 (component p = row p's word)
      int8 al[XW_TG], ah[XW_TG];
#pragma unroll
      for (int t = 0; t < XW_TG; ++t) {
        uint2 r[8];
#pragma unroll
        for (int p = 0; p < 8; ++p)
          r[p] = intel_sub_group_block_read2((__global const uint*)(ap + (size_t)(8 * t + p) * K + s * 256 + j * 64));
        al[t] = (int8)(r[0].s0, r[1].s0, r[2].s0, r[3].s0, r[4].s0, r[5].s0, r[6].s0, r[7].s0);
        ah[t] = (int8)(r[0].s1, r[1].s1, r[2].s1, r[3].s1, r[4].s1, r[5].s1, r[6].s1, r[7].s1);
      }
      // ---- each column group's 16 words (lo 0..7, hi 8..15) of both halves in 4 reads, used by every row group
#pragma unroll
      for (int c = 0; c < XW_NC; ++c) {
        __global const uint* dp = wq + ((size_t)(g0 + c) * nsb * 4 + s * 4 + j) * (16 * LS);
        const uint8 q0 = intel_sub_group_block_read8(dp), q1 = intel_sub_group_block_read8(dp + 4 * LS);
        const uint8 q2 = intel_sub_group_block_read8(dp + 8 * LS), q3 = intel_sub_group_block_read8(dp + 12 * LS);
        const int8 lo0 = as_int8((uint8)(q0.s0, q0.s2, q0.s4, q0.s6, q1.s0, q1.s2, q1.s4, q1.s6));
        const int8 lo1 = as_int8((uint8)(q0.s1, q0.s3, q0.s5, q0.s7, q1.s1, q1.s3, q1.s5, q1.s7));
        const int8 hi0 = as_int8((uint8)(q2.s0, q2.s2, q2.s4, q2.s6, q3.s0, q3.s2, q3.s4, q3.s6));
        const int8 hi1 = as_int8((uint8)(q2.s1, q2.s3, q2.s5, q2.s7, q3.s1, q3.s3, q3.s5, q3.s7));
#pragma unroll
        for (int t = 0; t < XW_TG; ++t) {
          xacc[t][2 * c] = xw8_mad(ah[t], hi0, xw8_mad(al[t], lo0, xacc[t][2 * c]));
          xacc[t][2 * c + 1] = xw8_mad(ah[t], hi1, xw8_mad(al[t], lo1, xacc[t][2 * c + 1]));
        }
      }
    }
  }

  // ---- the one epilogue, gemm_x8's: the token's scale x the column's x the sum, then the accumulate mode
  float av[XW_TM / SG];
#pragma unroll
  for (int v = 0; v < XW_TM / SG; ++v)
    av[v] = as_float(intel_sub_group_block_read((__global const uint*)(asc + m0 + SG * v)));
#pragma unroll
  for (int h = 0; h < XW_NH; ++h) {
    const int g = g0 + h / 2, hg = h & 1;
    const float dcol = as_float(intel_sub_group_block_read(wh + (size_t)g * LS + hg * SG));
#pragma unroll
    for (int t = 0; t < XW_TG; ++t) {
      const int8 x = xacc[t][h];
      const int is[8] = {x.s0, x.s1, x.s2, x.s3, x.s4, x.s5, x.s6, x.s7};
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int m = 8 * t + r;
        const float res = BCAST(av[m / SG], m % SG) * dcol * (float)is[r];
        __global uint* cp = (__global uint*)(C + c_off + (size_t)(m0 + m) * ldc + (size_t)g * LS + hg * SG);
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

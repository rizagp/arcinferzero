// The XMX prompt GEMM for GPUs whose smallest sub-group is 8 lanes (Arrow Lake H, Arc 140T; zero/engine_setup.h
// gemm_kernel): cl_intel_subgroup_matrix_multiply_accumulate must be called from an 8-lane sub-group there, with an int8
// operand a (the extension's text; gemm.cl's 16-lane form is undefined on them, and its results differed on a 140T,
// issue #5). The same arithmetic as gemm.cl in the same order, so the results are its DP4a kernel's bit for bit; the
// start-time XMX check compares them.
// W8 and H8 layouts only: QTYPE 100..103, 110, 111. The native types stay on DP4a on these GPUs.
// A 16-column group is two sub-groups: half 0 owns columns 0-7 of the group, half 1 columns 8-15. The weight and output
// layouts stay gemm.cl's, one word of a column every 16 uints.
// -DX8W (ARC_X8_W=1): one sub-group takes both halves, so the activations of 8 tokens are loaded once for 16 columns
// (two DPAS each) instead of once per half; the sub-groups of the second halves exit at once. The same sums per
// column, bit for bit.
// -DX8_EMU (tests on a GPU without XMX, ARC_XMX_TRY=2): the DPAS as the DP4a dot products the extension's text gives.
// With -DDBG_NOBCAST (ARC_XMX_TRY=3) a wrong one, which the check must refuse.
#define SG 8
#define LS 16  // lanes in the layouts
#ifndef TM
#define TM 16
#endif
#if TM != 16 && TM != 32
#error "TM must be 16 or 32"
#endif
#define TMV (TM / SG)
#ifndef CG
#define CG 1
#endif
#ifdef X8W
#define NH 2  // column halves per sub-group
#else
#define NH 1
#endif
#define BCAST(x, i) intel_sub_group_shuffle((x), (i))

#if QTYPE == 110 || QTYPE == 111
#define H8 1
#define SC32 1
#define HAS_MIN 0
#define HDR 1
#if QTYPE == 111
#define H8R 1
#endif
#ifdef ACT32
#error "H8 takes per-256 activation scales"
#endif
#elif QTYPE >= 100 && QTYPE <= 103
#define SC32 (QTYPE & 1)
#define HAS_MIN (QTYPE >= 102)
#define HDR 10
#else
#error "gemm_x8: W8 and H8 layouts only"
#endif
#define DPJ 16

#ifdef X8_EMU
#ifdef DBG_NOBCAST
#define X8A(x, w) (x)
#else
#define X8A(x, w) BCAST(x, w)
#endif
inline int x8_row(uint a, int8 b, int acc) {
  acc += dot_4x8packed_ss_int(X8A(a, 0), (uint)b.s0);
  acc += dot_4x8packed_ss_int(X8A(a, 1), (uint)b.s1);
  acc += dot_4x8packed_ss_int(X8A(a, 2), (uint)b.s2);
  acc += dot_4x8packed_ss_int(X8A(a, 3), (uint)b.s3);
  acc += dot_4x8packed_ss_int(X8A(a, 4), (uint)b.s4);
  acc += dot_4x8packed_ss_int(X8A(a, 5), (uint)b.s5);
  acc += dot_4x8packed_ss_int(X8A(a, 6), (uint)b.s6);
  acc += dot_4x8packed_ss_int(X8A(a, 7), (uint)b.s7);
  return acc;
}
inline int8 x8_mad(int8 a, int8 b, int8 c) {
  return (int8)(x8_row((uint)a.s0, b, c.s0), x8_row((uint)a.s1, b, c.s1), x8_row((uint)a.s2, b, c.s2), x8_row((uint)a.s3, b, c.s3),
                x8_row((uint)a.s4, b, c.s4), x8_row((uint)a.s5, b, c.s5), x8_row((uint)a.s6, b, c.s6), x8_row((uint)a.s7, b, c.s7));
}
#else
#define x8_mad(a, b, c) intel_sub_group_i8_i8_matrix_mad_k32((a), (b), (c))
#endif

// W8: as gemm.cl, one instruction each where a 32x32-bit multiply is emulated; the same products
#define MULS(x, sc) ((x) * (short)(sc))
#define MULMIN(b, mn) ((short)(b) * (short)(mn))
#define SBYTE(u, k) ((int)(char)(((u) >> (8 * (k))) & 0xFF))
#define UBYTE(u, k) ((int)(((u) >> (8 * (k))) & 0xFF))

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void gemm_x8(__global const uint* restrict wq, __global const uint* restrict wh,
                      __global const char* restrict aq, __global const float* restrict asc,
                      __global const ushort* restrict absum, __global float* restrict C,
                      int M, int N, int K, int ldc, int c_off, int accumulate, int rgroup, int s_begin, int s_end) {
  // the work-groups of gemm.cl: (16 * CG) x WGY lanes, grid (N * rows, WGY)
  const int NC = N / (LS * CG), rows = M / (TM * (int)get_local_size(1));
  const int w = get_group_id(0), grp = w / (rgroup * NC), rem = w % (rgroup * NC);
  const int rin = min(rgroup, rows - grp * rgroup);
  const int g = (rem / rin) * CG + (int)get_local_id(0) / LS, row = grp * rgroup + rem % rin;
  const int hg = ((int)get_local_id(0) / SG) & 1;  // (which half of the group's columns)
#ifdef X8W
  if (hg) return;  // (the first half's sub-group takes both)
#endif
  const int m0 = (row * (int)get_local_size(1) + (int)get_local_id(1)) * TM;
  const int nsb = K / 256;

  // column half h of this sub-group: its weights at + h * SG (one word of a column every 16 uints)
  __global const uint* wqp = wq + (size_t)g * nsb * (4 * DPJ * LS) + hg * SG;
  __global const uint* whp = wh + (size_t)g * nsb * (HDR * LS) + hg * SG;
  __global const char* ap = aq + (size_t)m0 * K;

  float res[NH][TM];
#pragma unroll
  for (int h = 0; h < NH; ++h)
#pragma unroll
    for (int m = 0; m < TM; ++m) res[h][m] = 0.0f;
#ifdef H8R
  int8 xacc[NH][TM / 8];
#pragma unroll
  for (int h = 0; h < NH; ++h)
#pragma unroll
    for (int v = 0; v < TM / 8; ++v) xacc[h][v] = (int8)(0);
#endif

  for (int s = s_begin; s < s_end; ++s) {
#if defined(H8R)
    // (one scale per column, read after the loop)
#elif defined(H8)
    float d[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) d[h] = as_float(intel_sub_group_block_read(whp + s * HDR * LS + h * SG));
#else
    float d[NH], dmin[NH];
    uint hsc[NH][4];
#if HAS_MIN
    uint hmn[NH][4];
#endif
#pragma unroll
    for (int h = 0; h < NH; ++h) {
      __global const uint* hp = whp + s * HDR * LS + h * SG;
      d[h] = as_float(intel_sub_group_block_read(hp));
      dmin[h] = as_float(intel_sub_group_block_read(hp + LS));
#pragma unroll
      for (int q = 0; q < 4; ++q) hsc[h][q] = intel_sub_group_block_read(hp + (2 + q) * LS);
#if HAS_MIN
#pragma unroll
      for (int q = 0; q < 4; ++q) hmn[h][q] = intel_sub_group_block_read(hp + (6 + q) * LS);
#endif
    }
#endif

#ifdef ACT32
    float tq[NH][TM];
#pragma unroll
    for (int h = 0; h < NH; ++h)
#pragma unroll
      for (int m = 0; m < TM; ++m) tq[h][m] = 0.0f;
#if HAS_MIN
    float tmn[NH][TM];
#pragma unroll
    for (int h = 0; h < NH; ++h)
#pragma unroll
      for (int m = 0; m < TM; ++m) tmn[h][m] = 0.0f;
#endif
#elif !defined(H8R)
    int isum[NH][TM];
#pragma unroll
    for (int h = 0; h < NH; ++h)
#pragma unroll
      for (int m = 0; m < TM; ++m) isum[h][m] = 0;
#endif

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // values 0..31 and 32..63 of this column's 64, a word of 4 each (per half)
      int8 lo[NH], hi[NH];
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        __global const uint* dp = wqp + (s * 4 + j) * DPJ * LS + h * SG;
        lo[h] = (int8)(intel_sub_group_block_read(dp), intel_sub_group_block_read(dp + LS), intel_sub_group_block_read(dp + 2 * LS),
                       intel_sub_group_block_read(dp + 3 * LS), intel_sub_group_block_read(dp + 4 * LS), intel_sub_group_block_read(dp + 5 * LS),
                       intel_sub_group_block_read(dp + 6 * LS), intel_sub_group_block_read(dp + 7 * LS));
        hi[h] = (int8)(intel_sub_group_block_read(dp + 8 * LS), intel_sub_group_block_read(dp + 9 * LS), intel_sub_group_block_read(dp + 10 * LS),
                       intel_sub_group_block_read(dp + 11 * LS), intel_sub_group_block_read(dp + 12 * LS), intel_sub_group_block_read(dp + 13 * LS),
                       intel_sub_group_block_read(dp + 14 * LS), intel_sub_group_block_read(dp + 15 * LS));
      }
#ifdef ACT32
      float avl[TMV], avh[TMV];
#pragma unroll
      for (int v = 0; v < TMV; ++v) {
        avl[v] = as_float(intel_sub_group_block_read((__global const uint*)(asc + ((size_t)s * 8 + 2 * j) * M + m0 + SG * v)));
        avh[v] = as_float(intel_sub_group_block_read((__global const uint*)(asc + ((size_t)s * 8 + 2 * j + 1) * M + m0 + SG * v)));
      }
#if HAS_MIN
      int b0[TMV], b1[TMV], b2[TMV], b3[TMV];
#pragma unroll
      for (int v = 0; v < TMV; ++v) {
        b0[v] = (short)intel_sub_group_block_read_us(absum + ((size_t)s * 16 + 4 * j + 0) * M + m0 + SG * v);
        b1[v] = (short)intel_sub_group_block_read_us(absum + ((size_t)s * 16 + 4 * j + 1) * M + m0 + SG * v);
        b2[v] = (short)intel_sub_group_block_read_us(absum + ((size_t)s * 16 + 4 * j + 2) * M + m0 + SG * v);
        b3[v] = (short)intel_sub_group_block_read_us(absum + ((size_t)s * 16 + 4 * j + 3) * M + m0 + SG * v);
      }
#endif
#endif
#pragma unroll
      for (int mg = 0; mg < TM; mg += 8) {
        // this chunk's values 0..31 and 32..63 of 8 tokens: a lane's word is 4 values of a row (loaded once for NH halves)
#define X8_ROW(p, o) intel_sub_group_block_read((__global const uint*)(ap + (size_t)(mg + (p)) * K + s * 256 + j * 64 + (o)))
        const int8 al = (int8)(X8_ROW(0, 0), X8_ROW(1, 0), X8_ROW(2, 0), X8_ROW(3, 0), X8_ROW(4, 0), X8_ROW(5, 0), X8_ROW(6, 0), X8_ROW(7, 0));
        const int8 ah = (int8)(X8_ROW(0, 32), X8_ROW(1, 32), X8_ROW(2, 32), X8_ROW(3, 32), X8_ROW(4, 32), X8_ROW(5, 32), X8_ROW(6, 32), X8_ROW(7, 32));
#pragma unroll
        for (int h = 0; h < NH; ++h) {
#ifdef H8R
          xacc[h][mg / 8] = x8_mad(ah, hi[h], x8_mad(al, lo[h], xacc[h][mg / 8]));
#else
#if !defined(H8)
          const int sc0 = SBYTE(hsc[h][j], 0), sc1 = SBYTE(hsc[h][j], 1), sc2 = SBYTE(hsc[h][j], 2), sc3 = SBYTE(hsc[h][j], 3);
#if HAS_MIN
          const int mn0 = UBYTE(hmn[h][j], 0), mn1 = UBYTE(hmn[h][j], 1), mn2 = UBYTE(hmn[h][j], 2), mn3 = UBYTE(hmn[h][j], 3);
#endif
#endif
          int c0[8], c1[8];
#if SC32
          {
            const int8 r0 = x8_mad(al, lo[h], (int8)(0)), r1 = x8_mad(ah, hi[h], (int8)(0));
            vstore8(r0, 0, c0);
            vstore8(r1, 0, c1);
          }
#else
          int c2[8], c3[8];
          {
            // (per-16 scales: each half of a 32-block alone, the other half's weights zero)
            const int8 r0 = x8_mad(al, (int8)(lo[h].s0, lo[h].s1, lo[h].s2, lo[h].s3, 0, 0, 0, 0), (int8)(0));
            const int8 r1 = x8_mad(al, (int8)(0, 0, 0, 0, lo[h].s4, lo[h].s5, lo[h].s6, lo[h].s7), (int8)(0));
            const int8 r2 = x8_mad(ah, (int8)(hi[h].s0, hi[h].s1, hi[h].s2, hi[h].s3, 0, 0, 0, 0), (int8)(0));
            const int8 r3 = x8_mad(ah, (int8)(0, 0, 0, 0, hi[h].s4, hi[h].s5, hi[h].s6, hi[h].s7), (int8)(0));
            vstore8(r0, 0, c0);
            vstore8(r1, 0, c1);
            vstore8(r2, 0, c2);
            vstore8(r3, 0, c3);
          }
#endif
#pragma unroll
          for (int p = 0; p < 8; ++p) {
            const int m = mg + p;
#if defined(H8)
            const int plo = c0[p], phi = c1[p];
#elif SC32
            const int plo = MULS(c0[p], sc0), phi = MULS(c1[p], sc2);
#else
            const int plo = MULS(c0[p], sc0) + MULS(c1[p], sc1), phi = MULS(c2[p], sc2) + MULS(c3[p], sc3);
#endif
#ifdef ACT32
            tq[h][m] = fma(BCAST(avl[m / SG], m % SG), (float)plo, fma(BCAST(avh[m / SG], m % SG), (float)phi, tq[h][m]));
#if HAS_MIN
            const int mlo = MULMIN(BCAST(b0[m / SG], m % SG), mn0) + MULMIN(BCAST(b1[m / SG], m % SG), mn1);
            const int mhi = MULMIN(BCAST(b2[m / SG], m % SG), mn2) + MULMIN(BCAST(b3[m / SG], m % SG), mn3);
            tmn[h][m] = fma(BCAST(avl[m / SG], m % SG), (float)mlo, fma(BCAST(avh[m / SG], m % SG), (float)mhi, tmn[h][m]));
#endif
#else
            isum[h][m] += plo + phi;
#endif
          }
#endif  // H8R
        }
      }
    }

#ifdef ACT32
#pragma unroll
    for (int h = 0; h < NH; ++h)
#pragma unroll
      for (int m = 0; m < TM; ++m) {
#if HAS_MIN
        res[h][m] = fma(d[h], tq[h][m], fma(-dmin[h], tmn[h][m], res[h][m]));
#else
        res[h][m] = fma(d[h], tq[h][m], res[h][m]);
#endif
      }
#elif !defined(H8R)
    // ---- superblock epilogue: activation scale (and the min term)
    float av[TMV];
#pragma unroll
    for (int v = 0; v < TMV; ++v) av[v] = as_float(intel_sub_group_block_read((__global const uint*)(asc + (size_t)s * M + m0 + SG * v)));
#if HAS_MIN
    int bs[16][TMV];
#pragma unroll
    for (int b = 0; b < 16; ++b)
#pragma unroll
      for (int v = 0; v < TMV; ++v) bs[b][v] = (short)intel_sub_group_block_read_us(absum + ((size_t)s * 16 + b) * M + m0 + SG * v);
#endif
#pragma unroll
    for (int h = 0; h < NH; ++h) {
#if HAS_MIN
      int mnv[16];
#pragma unroll
      for (int b = 0; b < 16; ++b) mnv[b] = UBYTE(hmn[h][b / 4], b & 3);
#endif
#pragma unroll
      for (int m = 0; m < TM; ++m) {
        const float a_s = BCAST(av[m / SG], m % SG);
        res[h][m] = fma(a_s * d[h], (float)isum[h][m], res[h][m]);
#if HAS_MIN
        int msum = 0;
#pragma unroll
        for (int b = 0; b < 16; ++b) msum += mnv[b] * BCAST(bs[b][m / SG], m % SG);
        res[h][m] = fma(-a_s * dmin[h], (float)msum, res[h][m]);
#endif
      }
    }
#endif
  }

#ifdef H8R
  // ---- the one epilogue: the column's scale x each token's
  float av[TMV];
#pragma unroll
  for (int v = 0; v < TMV; ++v)
    av[v] = as_float(intel_sub_group_block_read((__global const uint*)(asc + m0 + SG * v)));
#pragma unroll
  for (int h = 0; h < NH; ++h) {
    const float dcol = as_float(intel_sub_group_block_read(wh + (size_t)g * LS + hg * SG + h * SG));
    int isum[TM];
#pragma unroll
    for (int v = 0; v < TM / 8; ++v) vstore8(xacc[h][v], 0, isum + 8 * v);
#pragma unroll
    for (int m = 0; m < TM; ++m) res[h][m] = BCAST(av[m / SG], m % SG) * dcol * (float)isum[m];
  }
#endif

#pragma unroll
  for (int h = 0; h < NH; ++h)
#pragma unroll
    for (int m = 0; m < TM; ++m) {
      __global uint* cp = (__global uint*)(C + c_off + (size_t)(m0 + m) * ldc + (size_t)g * LS + hg * SG + h * SG);
      float v = res[h][m];
      if (accumulate == 1) {
        v += as_float(intel_sub_group_block_read(cp));
      } else if (accumulate == 2) {  // fused SwiGLU: C holds the gate projection, v the up projection
        const float gt = as_float(intel_sub_group_block_read(cp));
        v *= gt / (1.0f + exp(-gt));
      }
      intel_sub_group_block_write(cp, as_uint(v));
    }
}

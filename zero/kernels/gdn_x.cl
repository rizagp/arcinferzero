// The chunked DeltaNet rule on XMX (zero/xe2/engine_xe2.h, ARC_GDN_CHUNK=3): gdn_chunk_x, a kernel file of its own
// (discrete Xe2 / Xe3 cards: 16-lane DPAS). The rule and its symbols are gdn_chunk's (zero/kernels/ops.cl).
// Built with -DGDN_XMX (and -DGDN_XMX_EMU for DPAS emulated); FUSE_GATES as gdn_chunk.

#define GCH 32
// gdn_chunk's rule with its matrix products on XMX (DPAS 8 x 16 x 16, f16 in, f32 sums; QUEUE F33.1; ARC_GDN_CHUNK=3,
// discrete Xe2 / Xe3 cards with 16-lane DPAS). Built only with -DGDN_XMX (the builtin does not exist elsewhere);
// -DGDN_XMX_EMU emulates DPAS with shuffles (tests on GPUs and CPUs without XMX: the same products, sums in another
// order). A work-group of 16 sub-groups of 16 per v-head; sub-group g holds state rows 8g..8g+7 as DPAS accumulators,
// one float8 per 16 key dims (component r = row 8g + r, lane = key dim): the layout of a DPAS result and, converted to
// f16, of its A operand. Per chunk (lane = token below), with Th = T = (I + A)^-1 made by sub-group 0:
//   PK^T = S K^T, PQ^T = S Q^T                       (A: the state; B: a token's k / q row, lane = token)
//   E^T  = beta (v - G PK^T), D^T = E^T T^T            (the deltas; T from local memory)
//   O^T  = G PQ^T + D^T P^T                            (outputs, written 8 rows per token)
//   S    = G_n S + (exp(lg_n - lg) D)^T K              (B: k transposed, lane = key dim)
// Every f16 operand is split into a high and a low half (x = hi + lo, ~22 bits) and each product takes hi*hi + hi*lo +
// lo*hi: f16 alone (11 bits) is too coarse for v - S k, a difference of close values. DPAS is cheap here (~170 per
// sub-group and chunk); the vector work (A, P, T, the splits) is the rest.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef GDN_XMX_EMU
// one DPAS 8 x 16 x 16 (as gemm_x16.cl's X16_EMU): a: lane l = A[r][l] in component r; b: lane n = column n, its 16
// values as 8 half pairs (k = 2i low, 2i + 1 high); result: component r = row r, lane = column
inline float8 gx_mad(short8 a, int8 b, float8 acc) {
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
#define gx_mad(a, b, acc) intel_sub_group_f16_f16_matrix_mad_k16((a), (b), (acc))
#endif
// x = hi + lo in f16 (A operands: 8 values a lane; B operands: 16 values a lane as 8 pairs)
inline short8 gx_hi8(float8 x) { return as_short8(convert_half8_rte(x)); }
inline short8 gx_lo8(float8 x) { return as_short8(convert_half8_rte(x - convert_float8(convert_half8_rte(x)))); }
inline int8 gx_hi16(float16 x) { return as_int8(convert_half16_rte(x)); }
inline int8 gx_lo16(float16 x) { return as_int8(convert_half16_rte(x - convert_float16(convert_half16_rte(x)))); }
// acc += a b with both split: hi*hi + hi*lo + lo*hi
#define GX_MAD3(acc, ah, al, bh, bl) acc = gx_mad((al), (bh), gx_mad((ah), (bl), gx_mad((ah), (bh), (acc))))
// Operand scales, powers of two (exact, undone on the f32 sums): k (unit rows, elements ~0.09), q (also / sqrt(128),
// ~0.008), P (~0.05) and the deltas are scaled into f16's normal range before the hi / lo split, so the low halves are
// not subnormal (~4e-6 for q: a DPAS that flushes f16 subnormals would drop them, leaving ~11 bits)
#define GX_KS 16.0f
#define GX_QS 1024.0f
#define GX_PS 64.0f
#define GX_ES 16.0f

#ifndef GDN_XMX8  // (the 16-lane form; the 8-lane one, gdn_chunk_x8, is below)
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void gdn_chunk_x(__global const float* x, int C, __global const float* ab, int nk, int nv,
                          __global const float* S, int M_valid, __global float* o, __global float* Sout
#ifdef FUSE_GATES
                          , __global const float* dt_bias, __global const float* ssm_a
#endif
                          ) {
  // k and q rows (hi / lo f16, row-major: B operands with lane = token), k transposed (lane = key dim); A, P, T f32
  __local half Kh[GCH][128], Kl[GCH][128], Qh[GCH][128], Ql[GCH][128], KTh[128][GCH], KTl[128][GCH];
  __local float A[GCH][GCH], P[GCH][GCH], T[GCH][GCH], lg[GCH], bt[GCH];
  const int lid = get_local_id(0), h = get_group_id(0), sg = lid / 16, ln = lid % 16, j0 = sg * 8;
  const int kh = h % nk, qo = kh * 128, ko = nk * 128 + kh * 128, vo = 2 * nk * 128 + h * 128 + j0;
#ifdef FUSE_GATES
  const float dtb = dt_bias[h], ssa = ssm_a[h];
#endif
  // the state rows j0..j0+7: st[kb] component r = S[j0 + r][16 kb + lane]
  float8 st[8];
  __global const float* sp = S + ((size_t)h * 128 + j0) * 128;
#pragma unroll
  for (int kb = 0; kb < 8; ++kb) {
    float v[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) v[r] = as_float(intel_sub_group_block_read((__global const uint*)(sp + r * 128 + kb * 16)));
    st[kb] = (float8)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
  }
  for (int c0 = 0; c0 < M_valid; c0 += GCH) {
    const int n = min(GCH, M_valid - c0);
    for (int e = lid; e < GCH * 128; e += 256) {
      const int i = e / 128, k = e % 128;
      const bool in = i < n;
      __global const float* row = x + (size_t)(c0 + (in ? i : 0)) * C;
      const float kv = in ? row[ko + k] * GX_KS : 0.0f, qv = in ? row[qo + k] * GX_QS : 0.0f;
      const half kvh = convert_half_rte(kv), qvh = convert_half_rte(qv);
      const half kvl = convert_half_rte(kv - (float)kvh), qvl = convert_half_rte(qv - (float)qvh);
      Kh[i][k] = kvh; Kl[i][k] = kvl; Qh[i][k] = qvh; Ql[i][k] = qvl;
      KTh[k][i] = kvh; KTl[k][i] = kvl;
    }
    if (lid < GCH) {
      float ld = 0.0f, b = 0.0f;
      if (lid < n) {
        __global const float* g = ab + (size_t)(c0 + lid) * 2 * nv;
#ifdef FUSE_GATES
        const float a = g[h] + dtb;
        ld = (a > 20.0f ? a : log(1.0f + exp(a))) * ssa;
        b = 1.0f / (1.0f + exp(-g[nv + h]));
#else
        ld = log(max(g[h], 1e-30f));
        b = g[nv + h];
#endif
      }
      lg[lid] = ld;
      bt[lid] = b;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0)
      for (int i = 1; i < GCH; ++i) lg[i] += lg[i - 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    // A = K K^T and P = Q K^T on DPAS, an 8 x 16 tile per sub-group (0..7: A, 8..15: P; rows tokens 8 rb.., columns
    // tokens 16 cb..; A operand: the rows' 16 dims of a step, lane = dim; B: a token's row, lane = token), scaled back,
    // masked and weighted as gdn_chunk's
    {
      const int t = sg & 7, rb = t >> 1, cb = t & 1, m = 16 * cb + ln;
      const bool isp = sg >= 8;  // (uniform in a sub-group)
      float8 acc = (float8)0.0f;
#pragma unroll
      for (int kb = 0; kb < 8; ++kb) {
        short ah_[8], al_[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          ah_[r] = as_short(isp ? Qh[8 * rb + r][16 * kb + ln] : Kh[8 * rb + r][16 * kb + ln]);
          al_[r] = as_short(isp ? Ql[8 * rb + r][16 * kb + ln] : Kl[8 * rb + r][16 * kb + ln]);
        }
        const short8 a_h = (short8)(ah_[0], ah_[1], ah_[2], ah_[3], ah_[4], ah_[5], ah_[6], ah_[7]);
        const short8 a_l = (short8)(al_[0], al_[1], al_[2], al_[3], al_[4], al_[5], al_[6], al_[7]);
        const int8 bh = as_int8(vload8(0, (__local const uint*)&Kh[m][16 * kb])), bl = as_int8(vload8(0, (__local const uint*)&Kl[m][16 * kb]));
        GX_MAD3(acc, a_h, a_l, bh, bl);
      }
      acc *= isp ? 1.0f / (GX_QS * GX_KS) : 1.0f / (GX_KS * GX_KS);
      const float av[8] = {acc.s0, acc.s1, acc.s2, acc.s3, acc.s4, acc.s5, acc.s6, acc.s7};
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int l = 8 * rb + r;
        const float rr = m <= l && l < n ? exp(lg[l] - lg[m]) : 0.0f;
        if (isp) P[l][m] = rr * av[r];
        else A[l][m] = m < l ? bt[l] * rr * av[r] : 0.0f;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // T = (I + A)^-1, unit lower triangular, by sub-group 0: row i from the rows before it (lane = column, two each)
    if (sg == 0) {
      for (int i = 0; i < GCH; ++i) {
#pragma unroll
        for (int hc = 0; hc < 2; ++hc) {
          const int m = ln + 16 * hc;
          float t = m == i ? 1.0f : 0.0f;
          if (m < i)
            for (int p = m; p < i; ++p) t = fma(-A[i][p], T[p][m], t);
          T[i][m] = t;
        }
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // PK^T and PQ^T: [ib] component r = row j0 + r, lane = token 16 ib + lane
    float8 pk[2] = {(float8)0.0f, (float8)0.0f}, pq[2] = {(float8)0.0f, (float8)0.0f};
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      const short8 sh = gx_hi8(st[kb]), sl = gx_lo8(st[kb]);
#pragma unroll
      for (int ib = 0; ib < 2; ++ib) {
        const int i = 16 * ib + ln;
        const int8 kbh = as_int8(vload8(0, (__local const uint*)&Kh[i][16 * kb])), kbl = as_int8(vload8(0, (__local const uint*)&Kl[i][16 * kb]));
        const int8 qbh = as_int8(vload8(0, (__local const uint*)&Qh[i][16 * kb])), qbl = as_int8(vload8(0, (__local const uint*)&Ql[i][16 * kb]));
        GX_MAD3(pk[ib], sh, sl, kbh, kbl);
        GX_MAD3(pq[ib], sh, sl, qbh, qbl);
      }
    }
#pragma unroll
    for (int ib = 0; ib < 2; ++ib) {  // (k and q were scaled)
      pk[ib] *= 1.0f / GX_KS;
      pq[ib] *= 1.0f / GX_QS;
    }
    // E^T, then D^T = E^T T^T (E scaled by GX_ES for its split, undone after)
    float8 et[2], dt[2] = {(float8)0.0f, (float8)0.0f};
#pragma unroll
    for (int ib = 0; ib < 2; ++ib) {
      const int i = 16 * ib + ln;
      const float8 v = i < n ? vload8(0, x + (size_t)(c0 + i) * C + vo) : (float8)0.0f;
      et[ib] = bt[i] * (v - exp(lg[i]) * pk[ib]) * GX_ES;
    }
#pragma unroll
    for (int lb = 0; lb < 2; ++lb) {
      const short8 eh = gx_hi8(et[lb]), el = gx_lo8(et[lb]);
#pragma unroll
      for (int ib = 0; ib < 2; ++ib) {
        const float16 tr = vload16(0, &T[16 * ib + ln][16 * lb]);
        GX_MAD3(dt[ib], eh, el, gx_hi16(tr), gx_lo16(tr));
      }
    }
#pragma unroll
    for (int ib = 0; ib < 2; ++ib) dt[ib] *= 1.0f / GX_ES;
    // O^T = G PQ^T + D^T P^T (D and P scaled for the products); component r of lane i -> o[c0 + i][h][j0 + r]
    float8 ot[2], o2[2] = {(float8)0.0f, (float8)0.0f};
#pragma unroll
    for (int lb = 0; lb < 2; ++lb) {
      const float8 ds = dt[lb] * GX_ES;
      const short8 dh = gx_hi8(ds), dl = gx_lo8(ds);
#pragma unroll
      for (int ib = 0; ib < 2; ++ib) {
        const float16 pr = vload16(0, &P[16 * ib + ln][16 * lb]) * GX_PS;
        GX_MAD3(o2[ib], dh, dl, gx_hi16(pr), gx_lo16(pr));
      }
    }
#pragma unroll
    for (int ib = 0; ib < 2; ++ib) {
      const int i = 16 * ib + ln;
      ot[ib] = exp(lg[i]) * pq[ib] + o2[ib] * (1.0f / (GX_ES * GX_PS));
      if (i < n) vstore8(ot[ib], 0, o + (size_t)(c0 + i) * nv * 128 + h * 128 + j0);
    }
    // S = G_n S + (exp(lg_n - lg) D)^T K (the deltas scaled by GX_ES, k by GX_KS: undone on each 16-dim block's sum)
    const float lgn = lg[n - 1], gn = exp(lgn);
    short8 ddh[2], ddl[2];
#pragma unroll
    for (int lb = 0; lb < 2; ++lb) {
      const int l = 16 * lb + ln;
      const float8 dd = dt[lb] * (l < n ? exp(lgn - lg[l]) * GX_ES : 0.0f);
      ddh[lb] = gx_hi8(dd);
      ddl[lb] = gx_lo8(dd);
    }
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      const int k = 16 * kb + ln;
      float8 up = (float8)0.0f;
#pragma unroll
      for (int lb = 0; lb < 2; ++lb) {
        const int8 bh = as_int8(vload8(0, (__local const uint*)&KTh[k][16 * lb])), bl = as_int8(vload8(0, (__local const uint*)&KTl[k][16 * lb]));
        GX_MAD3(up, ddh[lb], ddl[lb], bh, bl);
      }
      st[kb] = gn * st[kb] + up * (1.0f / (GX_ES * GX_KS));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  __global float* so = Sout + ((size_t)h * 128 + j0) * 128;
#pragma unroll
  for (int kb = 0; kb < 8; ++kb) {
    const float v[8] = {st[kb].s0, st[kb].s1, st[kb].s2, st[kb].s3, st[kb].s4, st[kb].s5, st[kb].s6, st[kb].s7};
#pragma unroll
    for (int r = 0; r < 8; ++r) intel_sub_group_block_write((__global uint*)(so + r * 128 + kb * 16), as_uint(v[r]));
  }
}
#endif  // !GDN_XMX8

#ifdef GDN_XMX8
// gdn_chunk_x at 8 lanes (-DGDN_XMX8): GPUs whose DPAS takes 8-lane sub-groups (Arrow Lake H, Arc 140T; Xe1 cards). The
// extension's 8-lane form: one DPAS 8 x 8 x 16, a: int8 (lane l holds values 2l, 2l + 1 of each row as a half pair),
// b: lane n = column n (8 columns), its 16 values as 8 pairs; result: component r = row r, lane = column. A work-group
// of 16 sub-groups of 8 per v-head (128 work-items); sub-group g holds state rows 8g..8g+7 as 16 accumulators of 8 key
// dims. An A operand of 16 values comes from two such accumulators (lane l < 4: values 2l, 2l + 1 of the first,
// l >= 4: of the second, through shuffles). The same products and splits as gdn_chunk_x; 4 token blocks of 8.
#ifdef GDN_XMX_EMU
inline float8 gx8_mad(int8 a, int8 b, float8 acc) {  // (as gemm_x16.cl's X16_EMU with X16_SG8)
  const int bw[8] = {b.s0, b.s1, b.s2, b.s3, b.s4, b.s5, b.s6, b.s7};
  float c[8] = {acc.s0, acc.s1, acc.s2, acc.s3, acc.s4, acc.s5, acc.s6, acc.s7};
  const uint ar[8] = {(uint)a.s0, (uint)a.s1, (uint)a.s2, (uint)a.s3, (uint)a.s4, (uint)a.s5, (uint)a.s6, (uint)a.s7};
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const half2 bh = as_half2(bw[k >> 1]);
    const float bk = (k & 1) ? (float)bh.s1 : (float)bh.s0;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const half2 ah = as_half2(intel_sub_group_shuffle(ar[r], k >> 1));
      c[r] = fma((k & 1) ? (float)ah.s1 : (float)ah.s0, bk, c[r]);
    }
  }
  return (float8)(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
}
#else
#define gx8_mad(a, b, acc) intel_sub_group_f16_f16_matrix_mad_k16((a), (b), (acc))
#endif
#define GX8_MAD3(acc, ah, al, bh, bl) acc = gx8_mad((al), (bh), gx8_mad((ah), (bl), gx8_mad((ah), (bh), (acc))))
// two half8 (values 2l and 2l + 1 of each row) as the int8 A operand
inline int8 gx8_pair(half8 a, half8 b) {
  return (int8)(as_int((half2)(a.s0, b.s0)), as_int((half2)(a.s1, b.s1)), as_int((half2)(a.s2, b.s2)),
                as_int((half2)(a.s3, b.s3)), as_int((half2)(a.s4, b.s4)), as_int((half2)(a.s5, b.s5)),
                as_int((half2)(a.s6, b.s6)), as_int((half2)(a.s7, b.s7)));
}
// the A operand of 16 values from two 8-value accumulators lo (values 0..7) and hi (8..15), split into hi / lo halves
inline void gx8_a(float8 lo, float8 hi, int8* ah, int8* al) {
  const int l = get_sub_group_local_id(), s0 = (2 * l) & 7;
  const float8 lo0 = intel_sub_group_shuffle(lo, s0), lo1 = intel_sub_group_shuffle(lo, s0 + 1);
  const float8 hi0 = intel_sub_group_shuffle(hi, s0), hi1 = intel_sub_group_shuffle(hi, s0 + 1);
  const float8 f0 = l < 4 ? lo0 : hi0, f1 = l < 4 ? lo1 : hi1;
  const half8 h0 = convert_half8_rte(f0), h1 = convert_half8_rte(f1);
  *ah = gx8_pair(h0, h1);
  *al = gx8_pair(convert_half8_rte(f0 - convert_float8(h0)), convert_half8_rte(f1 - convert_float8(h1)));
}

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void gdn_chunk_x8(__global const float* x, int C, __global const float* ab, int nk, int nv,
                           __global const float* S, int M_valid, __global float* o, __global float* Sout
#ifdef FUSE_GATES
                           , __global const float* dt_bias, __global const float* ssm_a
#endif
                           ) {
  __local half Kh[GCH][128], Kl[GCH][128], Qh[GCH][128], Ql[GCH][128], KTh[128][GCH], KTl[128][GCH];
  __local float A[GCH][GCH], P[GCH][GCH], T[GCH][GCH], lg[GCH], bt[GCH];
  const int lid = get_local_id(0), h = get_group_id(0), sg = lid / 8, ln = lid % 8, j0 = sg * 8;
  const int kh = h % nk, qo = kh * 128, ko = nk * 128 + kh * 128, vo = 2 * nk * 128 + h * 128 + j0;
#ifdef FUSE_GATES
  const float dtb = dt_bias[h], ssa = ssm_a[h];
#endif
  // st[kb] component r = S[j0 + r][8 kb + lane]
  float8 st[16];
  __global const float* sp = S + ((size_t)h * 128 + j0) * 128;
#pragma unroll
  for (int kb = 0; kb < 16; ++kb) {
    float v[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) v[r] = as_float(intel_sub_group_block_read((__global const uint*)(sp + r * 128 + kb * 8)));
    st[kb] = (float8)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
  }
  for (int c0 = 0; c0 < M_valid; c0 += GCH) {
    const int n = min(GCH, M_valid - c0);
    for (int e = lid; e < GCH * 128; e += 128) {
      const int i = e / 128, k = e % 128;
      const bool in = i < n;
      __global const float* row = x + (size_t)(c0 + (in ? i : 0)) * C;
      const float kv = in ? row[ko + k] * GX_KS : 0.0f, qv = in ? row[qo + k] * GX_QS : 0.0f;
      const half kvh = convert_half_rte(kv), qvh = convert_half_rte(qv);
      const half kvl = convert_half_rte(kv - (float)kvh), qvl = convert_half_rte(qv - (float)qvh);
      Kh[i][k] = kvh; Kl[i][k] = kvl; Qh[i][k] = qvh; Ql[i][k] = qvl;
      KTh[k][i] = kvh; KTl[k][i] = kvl;
    }
    if (lid < GCH) {
      float ld = 0.0f, b = 0.0f;
      if (lid < n) {
        __global const float* g = ab + (size_t)(c0 + lid) * 2 * nv;
#ifdef FUSE_GATES
        const float a = g[h] + dtb;
        ld = (a > 20.0f ? a : log(1.0f + exp(a))) * ssa;
        b = 1.0f / (1.0f + exp(-g[nv + h]));
#else
        ld = log(max(g[h], 1e-30f));
        b = g[nv + h];
#endif
      }
      lg[lid] = ld;
      bt[lid] = b;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0)
      for (int i = 1; i < GCH; ++i) lg[i] += lg[i - 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    // A = K K^T and P = Q K^T on DPAS: sub-group g makes the 8 x 8 tiles (rows tokens 8 rb.., columns 8 cb..) of both
    // (A operand: a row's dims 2 lane, 2 lane + 1 of a 16-dim step as one word; B: a token's row, lane = token)
    {
      const int rb = sg >> 2, cb = sg & 3, m = 8 * cb + ln;
      float8 ak = (float8)0.0f, aq = (float8)0.0f;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        int kh_[8], kl_[8], qh_[8], ql_[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          kh_[r] = ((__local const int*)&Kh[8 * rb + r][16 * kk])[ln];
          kl_[r] = ((__local const int*)&Kl[8 * rb + r][16 * kk])[ln];
          qh_[r] = ((__local const int*)&Qh[8 * rb + r][16 * kk])[ln];
          ql_[r] = ((__local const int*)&Ql[8 * rb + r][16 * kk])[ln];
        }
        const int8 a_kh = (int8)(kh_[0], kh_[1], kh_[2], kh_[3], kh_[4], kh_[5], kh_[6], kh_[7]);
        const int8 a_kl = (int8)(kl_[0], kl_[1], kl_[2], kl_[3], kl_[4], kl_[5], kl_[6], kl_[7]);
        const int8 a_qh = (int8)(qh_[0], qh_[1], qh_[2], qh_[3], qh_[4], qh_[5], qh_[6], qh_[7]);
        const int8 a_ql = (int8)(ql_[0], ql_[1], ql_[2], ql_[3], ql_[4], ql_[5], ql_[6], ql_[7]);
        const int8 bh = as_int8(vload8(0, (__local const uint*)&Kh[m][16 * kk])), bl = as_int8(vload8(0, (__local const uint*)&Kl[m][16 * kk]));
        GX8_MAD3(ak, a_kh, a_kl, bh, bl);
        GX8_MAD3(aq, a_qh, a_ql, bh, bl);
      }
      ak *= 1.0f / (GX_KS * GX_KS);
      aq *= 1.0f / (GX_QS * GX_KS);
      const float kv8[8] = {ak.s0, ak.s1, ak.s2, ak.s3, ak.s4, ak.s5, ak.s6, ak.s7};
      const float qv8[8] = {aq.s0, aq.s1, aq.s2, aq.s3, aq.s4, aq.s5, aq.s6, aq.s7};
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int l = 8 * rb + r;
        const float rr = m <= l && l < n ? exp(lg[l] - lg[m]) : 0.0f;
        A[l][m] = m < l ? bt[l] * rr * kv8[r] : 0.0f;
        P[l][m] = rr * qv8[r];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {  // T = (I + A)^-1 (lane = column, four each)
      for (int i = 0; i < GCH; ++i) {
#pragma unroll
        for (int hc = 0; hc < 4; ++hc) {
          const int m = ln + 8 * hc;
          float t = m == i ? 1.0f : 0.0f;
          if (m < i)
            for (int p = m; p < i; ++p) t = fma(-A[i][p], T[p][m], t);
          T[i][m] = t;
        }
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // PK^T, PQ^T: [ib] component r = row j0 + r, lane = token 8 ib + lane
    float8 pk[4], pq[4];
#pragma unroll
    for (int ib = 0; ib < 4; ++ib) pk[ib] = pq[ib] = (float8)0.0f;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {  // (16 key dims a DPAS: accumulators 2 kk, 2 kk + 1)
      int8 sh, sl;
      gx8_a(st[2 * kk], st[2 * kk + 1], &sh, &sl);
#pragma unroll
      for (int ib = 0; ib < 4; ++ib) {
        const int i = 8 * ib + ln;
        const int8 kbh = as_int8(vload8(0, (__local const uint*)&Kh[i][16 * kk])), kbl = as_int8(vload8(0, (__local const uint*)&Kl[i][16 * kk]));
        const int8 qbh = as_int8(vload8(0, (__local const uint*)&Qh[i][16 * kk])), qbl = as_int8(vload8(0, (__local const uint*)&Ql[i][16 * kk]));
        GX8_MAD3(pk[ib], sh, sl, kbh, kbl);
        GX8_MAD3(pq[ib], sh, sl, qbh, qbl);
      }
    }
    float8 et[4], dt[4], ot[4], o2[4];
#pragma unroll
    for (int ib = 0; ib < 4; ++ib) {
      const int i = 8 * ib + ln;
      pk[ib] *= 1.0f / GX_KS;
      pq[ib] *= 1.0f / GX_QS;
      const float8 v = i < n ? vload8(0, x + (size_t)(c0 + i) * C + vo) : (float8)0.0f;
      et[ib] = bt[i] * (v - exp(lg[i]) * pk[ib]) * GX_ES;
      dt[ib] = o2[ib] = (float8)0.0f;
    }
#pragma unroll
    for (int lb = 0; lb < 2; ++lb) {  // (16 tokens a DPAS: token blocks 2 lb, 2 lb + 1)
      int8 eh, el;
      gx8_a(et[2 * lb], et[2 * lb + 1], &eh, &el);
#pragma unroll
      for (int ib = 0; ib < 4; ++ib) {
        const float16 tr = vload16(0, &T[8 * ib + ln][16 * lb]);
        GX8_MAD3(dt[ib], eh, el, gx_hi16(tr), gx_lo16(tr));
      }
    }
#pragma unroll
    for (int ib = 0; ib < 4; ++ib) dt[ib] *= 1.0f / GX_ES;
#pragma unroll
    for (int lb = 0; lb < 2; ++lb) {
      int8 dh, dl;
      gx8_a(dt[2 * lb] * GX_ES, dt[2 * lb + 1] * GX_ES, &dh, &dl);
#pragma unroll
      for (int ib = 0; ib < 4; ++ib) {
        const float16 pr = vload16(0, &P[8 * ib + ln][16 * lb]) * GX_PS;
        GX8_MAD3(o2[ib], dh, dl, gx_hi16(pr), gx_lo16(pr));
      }
    }
#pragma unroll
    for (int ib = 0; ib < 4; ++ib) {
      const int i = 8 * ib + ln;
      ot[ib] = exp(lg[i]) * pq[ib] + o2[ib] * (1.0f / (GX_ES * GX_PS));
      if (i < n) vstore8(ot[ib], 0, o + (size_t)(c0 + i) * nv * 128 + h * 128 + j0);
    }
    const float lgn = lg[n - 1], gn = exp(lgn);
    float8 dd[4];
#pragma unroll
    for (int ib = 0; ib < 4; ++ib) {
      const int l = 8 * ib + ln;
      dd[ib] = dt[ib] * (l < n ? exp(lgn - lg[l]) * GX_ES : 0.0f);
    }
    int8 ddh[2], ddl[2];
#pragma unroll
    for (int lb = 0; lb < 2; ++lb) gx8_a(dd[2 * lb], dd[2 * lb + 1], &ddh[lb], &ddl[lb]);
#pragma unroll
    for (int kb = 0; kb < 16; ++kb) {
      const int k = 8 * kb + ln;
      float8 up = (float8)0.0f;
#pragma unroll
      for (int lb = 0; lb < 2; ++lb) {
        const int8 bh = as_int8(vload8(0, (__local const uint*)&KTh[k][16 * lb])), bl = as_int8(vload8(0, (__local const uint*)&KTl[k][16 * lb]));
        GX8_MAD3(up, ddh[lb], ddl[lb], bh, bl);
      }
      st[kb] = gn * st[kb] + up * (1.0f / (GX_ES * GX_KS));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  __global float* so = Sout + ((size_t)h * 128 + j0) * 128;
#pragma unroll
  for (int kb = 0; kb < 16; ++kb) {
    const float v[8] = {st[kb].s0, st[kb].s1, st[kb].s2, st[kb].s3, st[kb].s4, st[kb].s5, st[kb].s6, st[kb].s7};
#pragma unroll
    for (int r = 0; r < 8; ++r) intel_sub_group_block_write((__global uint*)(so + r * 128 + kb * 8), as_uint(v[r]));
  }
}
#undef GX8_MAD3
#endif  // GDN_XMX8
#undef GX_MAD3
#undef GX_KS
#undef GX_QS
#undef GX_PS
#undef GX_ES
#undef GCH

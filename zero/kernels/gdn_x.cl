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
      const float kv = in ? row[ko + k] : 0.0f, qv = in ? row[qo + k] : 0.0f;
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
    // A and P in f32 (from k / q as hi + lo)
    for (int e = lid; e < GCH * GCH; e += 256) {
      const int l = e / GCH, m = e % GCH;
      float kk = 0.0f, qk = 0.0f;
      if (m <= l && l < n) {
#pragma unroll 8
        for (int k = 0; k < 128; ++k) {
          const float km = (float)Kh[m][k] + (float)Kl[m][k];
          kk = fma((float)Kh[l][k] + (float)Kl[l][k], km, kk);
          qk = fma((float)Qh[l][k] + (float)Ql[l][k], km, qk);
        }
      }
      const float r = m <= l && l < n ? exp(lg[l] - lg[m]) : 0.0f;
      A[l][m] = m < l ? bt[l] * r * kk : 0.0f;
      P[l][m] = r * qk;
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
    // E^T, then D^T = E^T T^T
    float8 et[2], dt[2] = {(float8)0.0f, (float8)0.0f};
#pragma unroll
    for (int ib = 0; ib < 2; ++ib) {
      const int i = 16 * ib + ln;
      const float8 v = i < n ? vload8(0, x + (size_t)(c0 + i) * C + vo) : (float8)0.0f;
      et[ib] = bt[i] * (v - exp(lg[i]) * pk[ib]);
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
    // O^T = G PQ^T + D^T P^T; component r of lane i -> o[c0 + i][h][j0 + r]
    float8 ot[2];
#pragma unroll
    for (int ib = 0; ib < 2; ++ib) ot[ib] = exp(lg[16 * ib + ln]) * pq[ib];
#pragma unroll
    for (int lb = 0; lb < 2; ++lb) {
      const short8 dh = gx_hi8(dt[lb]), dl = gx_lo8(dt[lb]);
#pragma unroll
      for (int ib = 0; ib < 2; ++ib) {
        const float16 pr = vload16(0, &P[16 * ib + ln][16 * lb]);
        GX_MAD3(ot[ib], dh, dl, gx_hi16(pr), gx_lo16(pr));
      }
    }
#pragma unroll
    for (int ib = 0; ib < 2; ++ib) {
      const int i = 16 * ib + ln;
      if (i < n) vstore8(ot[ib], 0, o + (size_t)(c0 + i) * nv * 128 + h * 128 + j0);
    }
    // S = G_n S + (exp(lg_n - lg) D)^T K
    const float lgn = lg[n - 1], gn = exp(lgn);
    short8 ddh[2], ddl[2];
#pragma unroll
    for (int lb = 0; lb < 2; ++lb) {
      const int l = 16 * lb + ln;
      const float8 dd = dt[lb] * (l < n ? exp(lgn - lg[l]) : 0.0f);
      ddh[lb] = gx_hi8(dd);
      ddl[lb] = gx_lo8(dd);
    }
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      st[kb] *= gn;
      const int k = 16 * kb + ln;
#pragma unroll
      for (int lb = 0; lb < 2; ++lb) {
        const int8 bh = as_int8(vload8(0, (__local const uint*)&KTh[k][16 * lb])), bl = as_int8(vload8(0, (__local const uint*)&KTl[k][16 * lb]));
        GX_MAD3(st[kb], ddh[lb], ddl[lb], bh, bl);
      }
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
#undef GX_MAD3
#undef GCH

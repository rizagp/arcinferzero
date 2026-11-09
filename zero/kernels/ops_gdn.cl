// Part of the non-GEMM kernels (zero/kernels/ops*.cl, KernelParts in zero/clrt.h), the last one: built after the
// parts before it. The chunked DeltaNet of prompt passes (QUEUE F33.1, ARC_GDN_CHUNK): a part of its own so that
// the other ops kernels' source, and so their kernel cache, stay as they were.

// Chunked gated delta rule (prompts, QUEUE F33.1; ARC_GDN_CHUNK=1, zero/engine_qwen.h): gdn_recur's recurrence over
// chunks of GCH tokens. Per chunk, from the state S0 at its start, lg = the inclusive sum of log decay within it and
// G = exp(lg):
//   A[l][m] = beta_l exp(lg_l - lg_m) (k_l . k_m), m < l        P[i][l] = exp(lg_i - lg_l) (q_i . k_l), l <= i
//   d_i = beta_i (v_i - G_i S0 k_i) - sum_{m<i} A[i][m] d_m     (the token's delta, by forward substitution)
//   o_i = G_i S0 q_i + sum_{l<=i} P[i][l] d_l                    S = G_n S0 + sum_l exp(lg_n - lg_l) d_l k_l^T
// (checked against the sequential rule in f64 / f32: 1e-7 relative in f32, chunks of 16 and 64, decays 1e-6..1).
// Not bit-identical to gdn_recur (sums in another order): a KL check.
// Layout: a work-group of 256 per half v-head (64 state rows), 4 work-items per row, each holding a quarter of the row
// (32 of the 128 key dims) and adding its partial dot products with two sub-group shuffles: 4x the work-items of
// gdn_recur, whose nv*128 cannot fill a large GPU (a B60 at ~30 % with the 27B's 48 heads). The chunk's k / q rows,
// A, P and the gates sit in local memory (40 KB); v is read where used. grid: nv*2 work-groups of 256.
// x, ab, S, o, Sout as gdn_recur; -DFUSE_GATES: ab holds the raw projections (the log decay taken from them directly).
#define GCH 32
#ifdef NO_SG8
#define GW2 16
#else
#define GW2 8
#endif
__attribute__((intel_reqd_sub_group_size(GW2)))
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void gdn_chunk(__global const float* x, int C, __global const float* ab, int nk, int nv,
                        __global const float* S, int M_valid, __global float* o, __global float* Sout
#ifdef FUSE_GATES
                        , __global const float* dt_bias, __global const float* ssm_a
#endif
                        ) {
  __local float Ks[GCH][128], Qs[GCH][128], A[GCH][GCH], P[GCH][GCH], lg[GCH], bt[GCH];
  const int lid = get_local_id(0), h = get_group_id(0) / 2, j = (get_group_id(0) % 2) * 64 + lid / 4, qq = lid % 4;
  const int k0 = qq * 32, kh = h % nk, qo = kh * 128, ko = nk * 128 + kh * 128, vo = 2 * nk * 128 + h * 128 + j;
#ifdef FUSE_GATES
  const float dtb = dt_bias[h], ssa = ssm_a[h];
#endif
  float s[32];
  __global const float* sp = S + ((size_t)h * 128 + j) * 128 + k0;
#pragma unroll
  for (int k = 0; k < 32; k += 4) {
    const float4 v4 = vload4(k / 4, sp);
    s[k] = v4.x; s[k + 1] = v4.y; s[k + 2] = v4.z; s[k + 3] = v4.w;
  }
  for (int c0 = 0; c0 < M_valid; c0 += GCH) {
    const int n = min(GCH, M_valid - c0);
    // the chunk's k and q rows (zeros past its end), and each token's log decay and beta
    for (int e = lid; e < GCH * 128; e += 256) {
      const int i = e / 128, k = e % 128;
      const bool in = i < n;
      __global const float* row = x + (size_t)(c0 + (in ? i : 0)) * C;
      Ks[i][k] = in ? row[ko + k] : 0.0f;
      Qs[i][k] = in ? row[qo + k] : 0.0f;
    }
    if (lid < GCH) {
      float ld = 0.0f, b = 0.0f;
      if (lid < n) {
        __global const float* g = ab + (size_t)(c0 + lid) * 2 * nv;
#ifdef FUSE_GATES
        const float a = g[h] + dtb;
        ld = (a > 20.0f ? a : log(1.0f + exp(a))) * ssa;  // (gdn_gates' exponent: log of its decay)
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
    // A and P (zero outside their triangles and past the chunk's end)
    for (int e = lid; e < GCH * GCH; e += 256) {
      const int l = e / GCH, m = e % GCH;
      float kk = 0.0f, qk = 0.0f;
      if (m <= l && l < n) {
#pragma unroll 8
        for (int k = 0; k < 128; ++k) {
          kk = fma(Ks[l][k], Ks[m][k], kk);
          qk = fma(Qs[l][k], Ks[m][k], qk);
        }
      }
      const float r = m <= l && l < n ? exp(lg[l] - lg[m]) : 0.0f;
      A[l][m] = m < l ? bt[l] * r * kk : 0.0f;
      P[l][m] = r * qk;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // this row's deltas d_i (the 4 work-items of a row hold the same values), then its outputs and the new state.
    // (local rows read 16 floats at a time into private arrays: indexed with constants once unrolled, registers)
    // (-DGDN_VEC, ARC_GDN_CHUNK=2: a row's quarter read as two 16-float loads into r (fewer messages, a few registers
    //  spilled on Xe-LPG per ocloc); else element by element (no spills): which is faster is for a GPU to say)
#ifdef GDN_VEC
    float r[32];
#define GDN_ROW(M, i) vstore16(vload16(0, &M[i][k0]), 0, r); vstore16(vload16(1, &M[i][k0]), 0, r + 16)
#define GDN_EL(M, i, k) r[k]
#else
#define GDN_ROW(M, i)
#define GDN_EL(M, i, k) M[i][k0 + (k)]
#endif
    float d[GCH];
#pragma unroll
    for (int i = 0; i < GCH; ++i) {
      GDN_ROW(Ks, i);
      float pk = 0.0f;
#pragma unroll
      for (int k = 0; k < 32; ++k) pk = fma(GDN_EL(Ks, i, k), s[k], pk);
      pk += intel_sub_group_shuffle_xor(pk, 1);
      pk += intel_sub_group_shuffle_xor(pk, 2);
      const float v = i < n ? x[(size_t)(c0 + i) * C + vo] : 0.0f;
      float e = bt[i] * (v - exp(lg[i]) * pk);
#pragma unroll
      for (int m = 0; m < i; ++m) e = fma(-A[i][m], d[m], e);  // (one address for every work-item: a broadcast)
      d[i] = e;
    }
#pragma unroll
    for (int i = 0; i < GCH; ++i) {
      GDN_ROW(Qs, i);
      float pq = 0.0f;
#pragma unroll
      for (int k = 0; k < 32; ++k) pq = fma(GDN_EL(Qs, i, k), s[k], pq);
      pq += intel_sub_group_shuffle_xor(pq, 1);
      pq += intel_sub_group_shuffle_xor(pq, 2);
      float acc = exp(lg[i]) * pq;
#pragma unroll
      for (int l = 0; l <= i; ++l) acc = fma(P[i][l], d[l], acc);
      if (qq == 0 && i < n) o[(size_t)(c0 + i) * nv * 128 + h * 128 + j] = acc;
    }
    const float lgn = lg[n - 1], gn = exp(lgn);
#pragma unroll
    for (int l = 0; l < GCH; ++l) d[l] *= l < n ? exp(lgn - lg[l]) : 0.0f;
#ifdef GDN_VEC
#pragma unroll
    for (int k = 0; k < 32; ++k) s[k] *= gn;
#pragma unroll
    for (int l = 0; l < GCH; ++l) {
      GDN_ROW(Ks, l);
#pragma unroll
      for (int k = 0; k < 32; ++k) s[k] = fma(d[l], r[k], s[k]);
    }
#else
#pragma unroll
    for (int k = 0; k < 32; ++k) {
      float a = gn * s[k];
#pragma unroll
      for (int l = 0; l < GCH; ++l) a = fma(d[l], Ks[l][k0 + k], a);
      s[k] = a;
    }
#endif
#undef GDN_ROW
#undef GDN_EL
    barrier(CLK_LOCAL_MEM_FENCE);  // (the next chunk overwrites the local arrays)
  }
  __global float* so = Sout + ((size_t)h * 128 + j) * 128 + k0;
#pragma unroll
  for (int k = 0; k < 32; k += 4) vstore4((float4)(s[k], s[k + 1], s[k + 2], s[k + 3]), k / 4, so);
}
#undef GW2
#undef GCH

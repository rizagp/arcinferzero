// Whisper's own kernels (zero/whisper_engine.h): the convolutions as im2col, LayerNorm with bias, the bias + GELU
// (erf) and residual epilogues, the q / k / v split for the encoder attention (vit_attn_rows in ops.cl), and the
// decoder's embedding, KV cache and attentions. The matrix products are the engine's int8 GEMM (gemm.cl, Q8_0, per-32
// activation scales: ops.cl swiglu_quant quantizes the rows).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

inline float gelu_erf(float x) { return 0.5f * x * (1.0f + erf(x * 0.70710678118654752f)); }

// conv1 (kernel 3, pad 1) as im2col: col[t][c * 3 + k] = mel[c][t + k - 1], K padded to Kp with zeros; rows t >= T
// zero. grid (Kp, Mp).
__kernel void wh_im2col1(__global const float* mel, int C, int T, int Kp, __global float* col) {
  const int kk = get_global_id(0), t = get_global_id(1);
  float v = 0.0f;
  if (kk < C * 3 && t < T) {
    const int c = kk / 3, s = t + kk % 3 - 1;
    if (s >= 0 && s < T) v = mel[(size_t)c * T + s];
  }
  col[(size_t)t * Kp + kk] = v;
}
// conv2 (kernel 3, stride 2, pad 1): col[t][c * 3 + k] = h[2t + k - 1][c]; rows t >= A zero. grid (3D, Ap).
__kernel void wh_im2col2(__global const float* h, int T, int D, int A, __global float* col) {
  const int kk = get_global_id(0), t = get_global_id(1);
  float v = 0.0f;
  if (t < A) {
    const int c = kk / 3, s = 2 * t + kk % 3 - 1;
    if (s >= 0 && s < T) v = h[(size_t)s * D + c];
  }
  col[(size_t)t * 3 * D + kk] = v;
}

// y[m][n] = gelu(y[m][n] + b[n]) (+ pos[m][n] when pos is given). grid (N, M).
__kernel void wh_bias_gelu(__global float* y, int N, __global const half* b, __global const half* pos, int with_pos) {
  const int n = get_global_id(0), m = get_global_id(1);
  const size_t i = (size_t)m * N + n;
  float v = gelu_erf(y[i] + vload_half(n, b));
  if (with_pos) v += vload_half(i, pos);
  y[i] = v;
}
// y[m][n] += b[n]. grid (N, M).
__kernel void wh_bias(__global float* y, int N, __global const half* b) {
  const int n = get_global_id(0), m = get_global_id(1);
  y[(size_t)m * N + n] += vload_half(n, b);
}
// x[m][n] += y[m][n] + b[n] (the residual after a projection). grid (N, M_valid).
__kernel void wh_bias_res(__global float* x, __global const float* y, int N, __global const half* b) {
  const int n = get_global_id(0), m = get_global_id(1);
  const size_t i = (size_t)m * N + n;
  x[i] += y[i] + vload_half(n, b);
}

// LayerNorm (eps 1e-5) of row m of x into out; rows m >= M_valid are written as zeros. grid (256, Mp), local (256, 1).
__kernel void wh_layernorm(__global const float* x, int D, int M_valid, __global const half* g, __global const half* b,
                           __global float* out) {
  const int m = get_group_id(1), l = get_local_id(0);
  __local float red[256];
  __global const float* r = x + (size_t)m * D;
  __global float* o = out + (size_t)m * D;
  if (m >= M_valid) {
    for (int i = l; i < D; i += 256) o[i] = 0.0f;
    return;
  }
  float s = 0.0f;
  for (int i = l; i < D; i += 256) s += r[i];
  red[l] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int k = 128; k > 0; k >>= 1) {
    if (l < k) red[l] += red[l + k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float mu = red[0] / D;
  barrier(CLK_LOCAL_MEM_FENCE);
  float q = 0.0f;
  for (int i = l; i < D; i += 256) { const float d = r[i] - mu; q += d * d; }
  red[l] = q;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int k = 128; k > 0; k >>= 1) {
    if (l < k) red[l] += red[l + k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = rsqrt(red[0] / D + 1e-5f);
  for (int i = l; i < D; i += 256) o[i] = (r[i] - mu) * inv * vload_half(i, g) + vload_half(i, b);
}

// the encoder attention's inputs: qkv[m][3D] + bias -> q (scaled by 1 / sqrt(64)), k, v as f16 [nh][Mp][64], the
// layout vit_attn_rows reads. grid (D, Mp).
__kernel void wh_qkv_split(__global const float* qkv, __global const half* b, int D, int Mp, __global half* q,
                           __global half* k, __global half* v) {
  const int c = get_global_id(0), m = get_global_id(1), h = c / 64, d = c % 64;
  const size_t src = (size_t)m * 3 * D + c, dst = ((size_t)h * Mp + m) * 64 + d;
  vstore_half((qkv[src] + vload_half(c, b)) * 0.125f, dst, q);
  vstore_half(qkv[src + D] + vload_half(D + c, b), dst, k);
  vstore_half(qkv[src + 2 * D] + vload_half(2 * D + c, b), dst, v);
}

// ---- the decoder: rows r (one token each) of sequences seq[r] at positions pos[r]
// x[r] = emb[tok[r]] + posenc[pos[r]]; rows r >= n zero. grid (D, Mp).
__kernel void wh_embed(__global const int* tok, __global const int* pos, int n, __global const half* emb,
                       __global const half* posenc, int D, __global float* x) {
  const int d = get_global_id(0), r = get_global_id(1);
  x[(size_t)r * D + d] = r < n ? vload_half((size_t)tok[r] * D + d, emb) + vload_half((size_t)pos[r] * D + d, posenc) : 0.0f;
}
// the self-attention cache: k / v (+ bias) of row r into sequence seq[r] at position pos[r] of layer l.
// cache [seqs][layers][ctx][D] f32. grid (D, n).
__kernel void wh_kv_store(__global const float* qkv, __global const half* b, int D, __global const int* seq,
                          __global const int* pos, int L, int ctx, int l, __global float* kc, __global float* vc) {
  const int c = get_global_id(0), r = get_global_id(1);
  const size_t dst = (((size_t)seq[r] * L + l) * ctx + pos[r]) * D + c;
  kc[dst] = qkv[(size_t)r * 3 * D + D + c] + vload_half(D + c, b);
  vc[dst] = qkv[(size_t)r * 3 * D + 2 * D + c] + vload_half(2 * D + c, b);
}

// one (row, head) per work-group of 64: scores into local memory, softmax, weighted values.
// self-attention: q = qkv[r][0..D) + bias, keys 0..pos[r] of the row's sequence (causal). grid (64 * nh, n), local (64, 1).
#define MAXK 1504
__kernel void wh_self_attn(__global const float* qkv, __global const half* b, int D, __global const int* seq,
                           __global const int* pos, int L, int ctx, int l, __global const float* kc,
                           __global const float* vc, __global float* out) {
  const int h = get_group_id(0), r = get_group_id(1), t = get_local_id(0);
  __local float qv[64], sc[448], red[64];
  qv[t] = (qkv[(size_t)r * 3 * D + h * 64 + t] + vload_half(h * 64 + t, b)) * 0.125f;
  barrier(CLK_LOCAL_MEM_FENCE);
  const int nk = pos[r] + 1;
  const size_t base = ((size_t)seq[r] * L + l) * ctx * D + h * 64;
  float mx = -1e30f;
  for (int j = t; j < nk; j += 64) {
    __global const float* kj = kc + base + (size_t)j * D;
    float s = 0.0f;
    for (int d = 0; d < 64; ++d) s += qv[d] * kj[d];
    sc[j] = s;
    mx = fmax(mx, s);
  }
  red[t] = mx;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int k = 32; k > 0; k >>= 1) {
    if (t < k) red[t] = fmax(red[t], red[t + k]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  mx = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  float z = 0.0f;
  for (int j = t; j < nk; j += 64) { sc[j] = exp(sc[j] - mx); z += sc[j]; }
  red[t] = z;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int k = 32; k > 0; k >>= 1) {
    if (t < k) red[t] += red[t + k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = 1.0f / red[0];
  float o = 0.0f;
  for (int j = 0; j < nk; ++j) o += sc[j] * vc[base + (size_t)j * D + t];
  out[(size_t)r * D + h * 64 + t] = o * inv;
}
// cross-attention: q[r][D] (bias added), keys / values ckv[j][2D] = [k | v] of the window (A positions). Scores of
// the heads with hmap[h] >= 0 are saved (q.k / 8, before the softmax) into save[hmap[h]][row0 + r][A].
// grid (256 * nh, n), local (256, 1).
__kernel void wh_cross_attn(__global const float* q, int D, __global const float* ckv, int A, __global float* out,
                            __global const int* hmap, __global float* save, int row0, int rows_total) {
  const int h = get_group_id(0), r = get_group_id(1), t = get_local_id(0);
  __local float qv[64], sc[MAXK], red[256];
  if (t < 64) qv[t] = q[(size_t)r * D + h * 64 + t] * 0.125f;
  barrier(CLK_LOCAL_MEM_FENCE);
  float mx = -1e30f;
  for (int j = t; j < A; j += 256) {
    __global const float* kj = ckv + (size_t)j * 2 * D + h * 64;
    float s = 0.0f;
    for (int d = 0; d < 64; ++d) s += qv[d] * kj[d];
    sc[j] = s;
    mx = fmax(mx, s);
  }
  const int slot = hmap[h];
  if (slot >= 0)
    for (int j = t; j < A; j += 256) save[((size_t)slot * rows_total + row0 + r) * A + j] = sc[j];
  red[t] = mx;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int k = 128; k > 0; k >>= 1) {
    if (t < k) red[t] = fmax(red[t], red[t + k]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  mx = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  float z = 0.0f;
  for (int j = t; j < A; j += 256) { sc[j] = exp(sc[j] - mx); z += sc[j]; }
  red[t] = z;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int k = 128; k > 0; k >>= 1) {
    if (t < k) red[t] += red[t + k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = 1.0f / red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  // the weighted values: 4 groups of 64 threads, each over a quarter of the positions, then summed
  const int d = t & 63, part = t >> 6;
  float o = 0.0f;
  for (int j = part; j < A; j += 4) o += sc[j] * ckv[(size_t)j * 2 * D + D + h * 64 + d];
  red[t] = o;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < 64) out[(size_t)r * D + h * 64 + t] = (red[t] + red[t + 64] + red[t + 128] + red[t + 192]) * inv;
}
// the window's cross-attention keys / values as f16, head-major: ckv[j][2D] (+ bias) -> ck[h][j][64], cv[h][j][64]
// (one layer). grid (D, A).
__kernel void wh_ckv_pack(__global const float* ckv, __global const half* b, int D, int Ap, __global half* ck,
                          __global half* cv) {
  const int c = get_global_id(0), j = get_global_id(1), h = c / 64, d = c % 64;
  const size_t dst = ((size_t)h * Ap + j) * 64 + d;
  vstore_half(ckv[(size_t)j * 2 * D + c] + vload_half(c, b), dst, ck);
  vstore_half(ckv[(size_t)j * 2 * D + D + c] + vload_half(D + c, b), dst, cv);
}
// cross-attention on the packed keys / values, RB rows per work-group sharing each key / value read:
// q[r][D] (bias added), ck / cv [h][Ap][64] f16 of A positions; out[r][D]. Scores of heads with hmap[h] >= 0 saved
// (q.k / 8, before the softmax) into save[hmap[h]][r][A]. grid (256 * nh, ceil(n / RB)), local (256, 1).
#define RB 4
__kernel void wh_cross_attn2(__global const float* q, int D, int n, __global const half* ck, __global const half* cv,
                             int A, int Ap, __global float* out, __global const int* hmap, __global float* save) {
  const int h = get_group_id(0), r0 = get_group_id(1) * RB, t = get_local_id(0);
  const int nr = min(RB, n - r0);
  __local float qv[RB][64], sc[RB][MAXK];
  for (int i = t; i < RB * 64; i += 256) {
    const int r = i / 64, d = i % 64;
    qv[r][d] = r < nr ? q[(size_t)(r0 + r) * D + h * 64 + d] * 0.125f : 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __global const half* kh = ck + (size_t)h * Ap * 64;
  __global const half* vh = cv + (size_t)h * Ap * 64;
  float mx[RB];
  for (int r = 0; r < RB; ++r) mx[r] = -1e30f;
  for (int j = t; j < A; j += 256) {
    float s[RB];
    for (int r = 0; r < RB; ++r) s[r] = 0.0f;
    for (int d8 = 0; d8 < 8; ++d8) {
      const float8 k = vload_half8(d8, kh + (size_t)j * 64);
      for (int r = 0; r < RB; ++r) {
        s[r] += k.s0 * qv[r][8 * d8] + k.s1 * qv[r][8 * d8 + 1] + k.s2 * qv[r][8 * d8 + 2] + k.s3 * qv[r][8 * d8 + 3] +
                k.s4 * qv[r][8 * d8 + 4] + k.s5 * qv[r][8 * d8 + 5] + k.s6 * qv[r][8 * d8 + 6] + k.s7 * qv[r][8 * d8 + 7];
      }
    }
    for (int r = 0; r < RB; ++r) { sc[r][j] = s[r]; mx[r] = fmax(mx[r], s[r]); }
  }
  for (int r = 0; r < RB; ++r) {
    const float m = work_group_reduce_max(mx[r]);
    mx[r] = m;
  }
  for (int r = 0; r < nr; ++r) {
    const int slot = hmap[h];
    if (slot >= 0)
      for (int j = t; j < A; j += 256) save[((size_t)slot * n + r0 + r) * A + j] = sc[r][j];
  }
  float z[RB];
  for (int r = 0; r < RB; ++r) z[r] = 0.0f;
  for (int j = t; j < A; j += 256)
    for (int r = 0; r < RB; ++r) { const float e = exp(sc[r][j] - mx[r]); sc[r][j] = e; z[r] += e; }
  for (int r = 0; r < RB; ++r) z[r] = work_group_reduce_add(z[r]);
  barrier(CLK_LOCAL_MEM_FENCE);
  // values: work-item (d, part) sums positions part, part + 4, ... for every row, then the 4 parts are added
  const int d = t & 63, part = t >> 6;
  float o[RB];
  for (int r = 0; r < RB; ++r) o[r] = 0.0f;
  for (int j = part; j < A; j += 4) {
    const float v = vload_half((size_t)j * 64 + d, vh);
    for (int r = 0; r < RB; ++r) o[r] += sc[r][j] * v;
  }
  __local float acc[RB][256];
  for (int r = 0; r < RB; ++r) acc[r][t] = o[r];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < 64)
    for (int r = 0; r < nr; ++r)
      out[(size_t)(r0 + r) * D + h * 64 + t] = (acc[r][t] + acc[r][t + 64] + acc[r][t + 128] + acc[r][t + 192]) / z[r];
}

// cross-attention split over the positions (flash decoding): work-group (head, part, row group) takes positions
// [part * P, part * P + P), RB rows; writes each row's partial max, sum and weighted values; wh_cross_comb adds the
// parts. part buffers: pm / pl [nh][S][n], po [nh][S][n][64]. grid (256 * nh, S, ceil(n / RB)), local (256, 1, 1).
#define CP 192
__kernel void wh_cross_part(__global const float* q, int D, int n, __global const half* ck, __global const half* cv,
                            int A, int Ap, int S, __global float* pm, __global float* pl, __global float* po,
                            __global const int* hmap, __global float* save) {
  const int h = get_group_id(0), part = get_group_id(1), r0 = get_group_id(2) * RB, t = get_local_id(0);
  const int nr = min(RB, n - r0), j0 = part * CP, j1 = min(A, j0 + CP), nj = max(0, j1 - j0);
  __local float qv[RB][64], sc[RB][CP], acc[RB][256];
  for (int i = t; i < RB * 64; i += 256) {
    const int r = i / 64, d = i % 64;
    qv[r][d] = r < nr ? q[(size_t)(r0 + r) * D + h * 64 + d] * 0.125f : 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __global const half* kh = ck + (size_t)h * Ap * 64;
  __global const half* vh = cv + (size_t)h * Ap * 64;
  // scores: positions j0 + t (one key row per work-item; CP <= 256)
  if (t < nj) {
    float s[RB];
    for (int r = 0; r < RB; ++r) s[r] = 0.0f;
    for (int d8 = 0; d8 < 8; ++d8) {
      const float8 k = vload_half8(d8, kh + (size_t)(j0 + t) * 64);
      for (int r = 0; r < RB; ++r)
        s[r] += k.s0 * qv[r][8 * d8] + k.s1 * qv[r][8 * d8 + 1] + k.s2 * qv[r][8 * d8 + 2] + k.s3 * qv[r][8 * d8 + 3] +
                k.s4 * qv[r][8 * d8 + 4] + k.s5 * qv[r][8 * d8 + 5] + k.s6 * qv[r][8 * d8 + 6] + k.s7 * qv[r][8 * d8 + 7];
    }
    for (int r = 0; r < RB; ++r) sc[r][t] = s[r];
    const int slot = hmap[h];
    if (slot >= 0)
      for (int r = 0; r < nr; ++r) save[((size_t)slot * n + r0 + r) * A + j0 + t] = s[r];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float mx[RB], z[RB];
  for (int r = 0; r < RB; ++r) mx[r] = work_group_reduce_max(t < nj ? sc[r][t] : -1e30f);
  if (t < nj)
    for (int r = 0; r < RB; ++r) sc[r][t] = exp(sc[r][t] - mx[r]);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int r = 0; r < RB; ++r) z[r] = work_group_reduce_add(t < nj ? sc[r][t] : 0.0f);
  const int d = t & 63, quarter = t >> 6;
  float o[RB];
  for (int r = 0; r < RB; ++r) o[r] = 0.0f;
  for (int j = quarter; j < nj; j += 4) {
    const float v = vload_half((size_t)(j0 + j) * 64 + d, vh);
    for (int r = 0; r < RB; ++r) o[r] += sc[r][j] * v;
  }
  for (int r = 0; r < RB; ++r) acc[r][t] = o[r];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < 64)
    for (int r = 0; r < nr; ++r) {
      const size_t i = ((size_t)h * S + part) * n + r0 + r;
      po[i * 64 + t] = acc[r][t] + acc[r][t + 64] + acc[r][t + 128] + acc[r][t + 192];
      if (t == 0) { pm[i] = mx[r]; pl[i] = z[r]; }
    }
}
// out[r][h * 64 + d] = sum over parts of exp(m_p - M) * o_p / sum of exp(m_p - M) * l_p. grid (64 * nh, n), local (64, 1).
__kernel void wh_cross_comb(int D, int n, int S, __global const float* pm, __global const float* pl,
                            __global const float* po, __global float* out) {
  const int h = get_group_id(0), r = get_global_id(1), d = get_local_id(0);
  float M = -1e30f;
  for (int p = 0; p < S; ++p) M = fmax(M, pm[((size_t)h * S + p) * n + r]);
  float num = 0.0f, den = 0.0f;
  for (int p = 0; p < S; ++p) {
    const size_t i = ((size_t)h * S + p) * n + r;
    const float w = exp(pm[i] - M);
    num += w * po[i * 64 + d];
    den += w * pl[i];
  }
  out[(size_t)r * D + h * 64 + d] = num / den;
}

// sequence dst's cache becomes a copy of src's (all layers, positions < len). grid (D, ctx * L).
__kernel void wh_kv_copy(int dst, int src, int L, int ctx, int len, int D, __global float* kc, __global float* vc) {
  const int c = get_global_id(0), lp = get_global_id(1), l = lp / ctx, p = lp % ctx;
  if (p >= len) return;
  const size_t s = (((size_t)src * L + l) * ctx + p) * D + c, t = (((size_t)dst * L + l) * ctx + p) * D + c;
  kc[t] = kc[s];
  vc[t] = vc[s];
}

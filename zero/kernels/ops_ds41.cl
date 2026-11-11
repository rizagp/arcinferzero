// DeepSeek-V4.1-Flash (deepseek41): the kernels of zero/engine_ds41.h (docs/deepseek41-spec.md section 3).
// A part of the ops program (zero/clrt.h KernelParts): it builds after ops.cl .. ops_gdn.cl and uses their helpers
// (quant_store16). First versions: plain f32 arithmetic, written to be checked against tools/ref_forward, not fast.
//
// The residual is DS_HC (4) streams of E floats per token, stream-major (x[t][s][E]). A site's mixes are 24 floats
// per token: pre [4], post [4], comb [4][4] after Sinkhorn (comb[j][k]: from stream j to stream k).

#ifndef DS_HC
#define DS_HC 4
#endif
// The key caches (the window ring and the chunk's keys, the compressed rows, the index keys): f16 with -DDS_KVH (the
// default, engine_ds41.h ds_kv_defs), else f32
#ifdef DS_KVH
#define DSKV_T half
#define DSKV_LD(p, i) vload_half((i), (p))
#define DSKV_ST(v, p, i) vstore_half((v), (i), (p))
#else
#define DSKV_T float
#define DSKV_LD(p, i) ((p)[i])
#define DSKV_ST(v, p, i) ((p)[i] = (v))
#endif
// ds_norm_rope's destination: a cache (-DDS_DST_KV) or f32 rows
#ifdef DS_DST_KV
#define DSDST_T DSKV_T
#define DSDST_ST(v, p, i) DSKV_ST(v, p, i)
#else
#define DSDST_T float
#define DSDST_ST(v, p, i) ((p)[i] = (v))
#endif
#define DS_MIX ((2 + DS_HC) * DS_HC)

// The mixes of one site for each token: m = fn . x (24 rows of HC*E) * rsqrt(mean(x^2) + eps), then
// pre = sigmoid(m * s0 + b) + 1e-6, post = 2 sigmoid(m * s1 + b), comb = Sinkhorn(m * s2 + b) (row softmax + eps,
// a column step, then iters - 1 row / column steps). grid (256, T), local (256, 1).
__kernel void ds_mix(__global const float* x, int HCD, __global const float* fn, __global const float* base,
                     __global const float* scale, float eps, float hc_eps, int iters, __global float* mix) {
  const int t = get_group_id(1), l = get_local_id(0);
  __global const float* xr = x + (size_t)t * HCD;
  float acc[DS_MIX];
#pragma unroll
  for (int j = 0; j < DS_MIX; ++j) acc[j] = 0.0f;
  float ss = 0.0f;
  for (int i = l; i < HCD; i += 256) {
    const float v = xr[i];
    ss += v * v;
#pragma unroll
    for (int j = 0; j < DS_MIX; ++j) acc[j] += fn[(size_t)j * HCD + i] * v;
  }
  __local float red[DS_MIX + 1];
  ss = work_group_reduce_add(ss);
#pragma unroll
  for (int j = 0; j < DS_MIX; ++j) {
    const float s = work_group_reduce_add(acc[j]);
    if (l == 0) red[j] = s;
  }
  if (l != 0) return;
  const float rs = rsqrt(ss / HCD + eps);
  __global float* o = mix + (size_t)t * DS_MIX;
  for (int j = 0; j < DS_HC; ++j) {
    o[j] = 1.0f / (1.0f + exp(-(red[j] * rs * scale[0] + base[j]))) + 1e-6f;
    o[DS_HC + j] = 2.0f / (1.0f + exp(-(red[DS_HC + j] * rs * scale[1] + base[DS_HC + j])));
  }
  float c[DS_HC * DS_HC];
  for (int j = 0; j < DS_HC; ++j) {
    float mx = -INFINITY;
    for (int k = 0; k < DS_HC; ++k) {
      const int q = 2 * DS_HC + j * DS_HC + k;
      c[j * DS_HC + k] = red[q] * rs * scale[2] + base[q];
      mx = fmax(mx, c[j * DS_HC + k]);
    }
    float z = 0.0f;
    for (int k = 0; k < DS_HC; ++k) z += c[j * DS_HC + k] = exp(c[j * DS_HC + k] - mx);
    for (int k = 0; k < DS_HC; ++k) c[j * DS_HC + k] = c[j * DS_HC + k] / z + hc_eps;
  }
  for (int it = 0; it < iters; ++it) {
    if (it > 0)
      for (int j = 0; j < DS_HC; ++j) {
        float s = 0.0f;
        for (int k = 0; k < DS_HC; ++k) s += c[j * DS_HC + k];
        for (int k = 0; k < DS_HC; ++k) c[j * DS_HC + k] /= s + hc_eps;
      }
    for (int k = 0; k < DS_HC; ++k) {
      float s = 0.0f;
      for (int j = 0; j < DS_HC; ++j) s += c[j * DS_HC + k];
      for (int j = 0; j < DS_HC; ++j) c[j * DS_HC + k] /= s + hc_eps;
    }
  }
  for (int i = 0; i < DS_HC * DS_HC; ++i) o[2 * DS_HC + i] = c[i];
}

// A site's input: y = rmsnorm(sum_j pre[j] x[j]) * w for rows row_off + m (pre from mix row m, offset 0; first = 1:
// stream 0 alone, layer 0's attention), as f32 into h[m] and quantized for the next GEMM (rmsnorm_quant's layout).
// grid (256, M), local (256, 1); rows m >= M_valid as zeros.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void ds_collapse(__global const float* x, int row_off, __global const float* mix, int first, int E,
                          __global const float* w, float eps, int M, int M_valid, __global char* aq,
                          __global float* asc, __global ushort* absum, __global float* h, int act32) {
  const int m = get_group_id(1);
  const bool valid = m < M_valid;
  __global const float* xr = x + (size_t)(m + row_off) * DS_HC * E;
  __global float* hr = h + (size_t)m * E;
  float pw[DS_HC];
#pragma unroll
  for (int j = 0; j < DS_HC; ++j) pw[j] = first ? (j == 0 ? 1.0f : 0.0f) : valid ? mix[(size_t)(m + row_off) * DS_MIX + j] : 0.0f;
  float ss = 0.0f;
  if (valid)
    for (int i = get_local_id(0); i < E; i += get_local_size(0)) {
      float y = 0.0f;
#pragma unroll
      for (int j = 0; j < DS_HC; ++j) y += pw[j] * xr[(size_t)j * E + i];
      hr[i] = y;
      ss += y * y;
    }
  ss = work_group_reduce_add(ss);
  barrier(CLK_GLOBAL_MEM_FENCE);
  const float r = rsqrt(ss / E + eps);
  const int lane = get_sub_group_local_id();
  for (int s = get_sub_group_id(); s < E / 256; s += get_num_sub_groups()) {
    const int b = s * 256 + 16 * lane;
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = valid ? hr[b + i] * r * w[b + i] : 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) hr[b + i] = v[i];
    quant_store16(v, s, m, M, E, aq, asc, absum, act32);
  }
}

// x[t][k] = post[k] f[t] + sum_j comb[j][k] x[t][j] (in place: one work-item owns element i of every stream).
// grid (E, T).
__kernel void ds_expand(__global float* x, __global const float* f, __global const float* mix, int E) {
  const int i = get_global_id(0), t = get_global_id(1);
  __global float* xr = x + (size_t)t * DS_HC * E + i;
  __global const float* mx = mix + (size_t)t * DS_MIX;
  float xs[DS_HC];
#pragma unroll
  for (int j = 0; j < DS_HC; ++j) xs[j] = xr[(size_t)j * E];
  const float fv = f[(size_t)t * E + i];
#pragma unroll
  for (int k = 0; k < DS_HC; ++k) {
    float a = mx[DS_HC + k] * fv;
#pragma unroll
    for (int j = 0; j < DS_HC; ++j) a += mx[2 * DS_HC + j * DS_HC + k] * xs[j];
    xr[(size_t)k * E] = a;
  }
}

// Rows of K values (src row r at src + r * K, dst likewise): optionally rmsnorm with w, then the last R values rotated
// as adjacent pairs at the row's position (table rows of (cos, sin) for R/2 frequencies; row r's table row is
// (r / rpp) * pmul + poff), sign -1: the inverse rotation. One sub-group per row: lane l holds values l*K/16 ...
// grid (16, nrows), local (16, 1).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void ds_norm_rope(__global const float* src, __global DSDST_T* dst, int drow0, int K, __global const float* w,
                           int use_norm, float eps, __global const float2* tab, int R, int rpp, int pmul, int poff,
                           float sign) {
  const int r = get_group_id(1), l = get_sub_group_local_id(), per = K / 16;
  __global const float* s = src + (size_t)r * K + l * per;
  __global DSDST_T* d = dst + (size_t)(r + drow0) * K + l * per;  // (drow0: the first destination row)
  float v[32];
  float ss = 0.0f;
  for (int i = 0; i < per; ++i) {
    v[i] = s[i];
    ss += v[i] * v[i];
  }
  if (use_norm) {
    ss = sub_group_reduce_add(ss);
    const float rs = rsqrt(ss / K + eps);
    for (int i = 0; i < per; ++i) v[i] = v[i] * rs * w[l * per + i];
  }
  const int r0 = K - R, prow = (r / rpp) * pmul + poff;
  for (int i = 0; i < per; i += 2) {
    const int e = l * per + i;
    if (e >= r0) {
      const float2 cs = tab[(size_t)prow * (R / 2) + (e - r0) / 2];
      const float sn = sign * cs.y, a = v[i], b = v[i + 1];
      v[i] = a * cs.x - b * sn;
      v[i + 1] = a * sn + b * cs.x;
    }
  }
  for (int i = 0; i < per; ++i) DSDST_ST(v[i], d, i);
}

// The compressor: compressed row q (absolute row n0 + q) from the tokens j*rt .. j*rt + rt - 1 (j = n0 + q), each the
// chunk's row p - pos or, before the chunk, a pending row (pend [rt-1][2][D]: values, scores of tokens n0*rt ..):
// latent = sum_u softmax_u(score) * value per channel (rt = 1: the value). grid (D, nnew).
__kernel void ds_compress(__global const float* cv, __global const float* cg, __global const float* pend, int D, int rt,
                          int pos, int n0, __global float* lat) {
  const int d = get_global_id(0), q = get_global_id(1), j = n0 + q;
  float sc[8], vv[8], mx = -INFINITY;
  for (int u = 0; u < rt; ++u) {
    const int p = j * rt + u;
    if (p >= pos) {
      vv[u] = cv[(size_t)(p - pos) * D + d];
      sc[u] = rt > 1 ? cg[(size_t)(p - pos) * D + d] : 0.0f;
    } else {
      const int k = p - n0 * rt;
      vv[u] = pend[(size_t)(2 * k) * D + d];
      sc[u] = pend[(size_t)(2 * k + 1) * D + d];
    }
    mx = fmax(mx, sc[u]);
  }
  if (rt == 1) {
    lat[(size_t)q * D + d] = vv[0];
    return;
  }
  float z = 0.0f, a = 0.0f;
  for (int u = 0; u < rt; ++u) {
    const float e = exp(sc[u] - mx);
    z += e;
    a += e * vv[u];
  }
  lat[(size_t)q * D + d] = a / z;
}

// The chunk's tokens n1*rt .. pos+T-1 (the group not complete yet) into the pending rows. grid (D, npend).
__kernel void ds_pend_store(__global const float* cv, __global const float* cg, __global float* pend, int D, int rt,
                            int pos, int n1) {
  const int d = get_global_id(0), k = get_global_id(1), p = n1 * rt + k;
  pend[(size_t)(2 * k) * D + d] = cv[(size_t)(p - pos) * D + d];
  pend[(size_t)(2 * k + 1) * D + d] = rt > 1 ? cg[(size_t)(p - pos) * D + d] : 0.0f;
}

// Indexer scores of queries tq0 .. tq0 + Tq - 1 (chunk rows) against the compressed rows' index keys:
// score[t][j] = sum_h w[t][h] / sqrt(ID * IH) * relu(iq[t][h] . ik[j]) for the rows the query sees
// (j < (pos + t + 1) / rt), else -inf; with a candidate map (cand [Tq][nbmax] bytes): rows outside its kept blocks
// -inf too. grid (rows padded to 64, Tq), local (64, 1).
__kernel void ds_ix_scores(__global const float* iq, __global const float* iw, __global const DSKV_T* ik, int IH, int ID,
                           int nrows, int pos, int rt, int tq0, __global float* sc, int ld, __global const uchar* cand,
                           int use_cand, int nbmax, int bs) {
  const int j = get_global_id(0), tl = get_global_id(1), t = tq0 + tl;
  if (j >= nrows) return;
  const int nv = (pos + t + 1) / rt;
  float s = -INFINITY;
  if (j < nv && (!use_cand || cand[(size_t)t * nbmax + j / bs])) {
    s = 0.0f;
    const float wsc = rsqrt((float)(ID * IH));
    __global const DSKV_T* kr = ik + (size_t)j * ID;
    for (int h = 0; h < IH; ++h) {
      __global const float* qr = iq + ((size_t)t * IH + h) * ID;
      float d = 0.0f;
      for (int i = 0; i < ID; ++i) d += qr[i] * DSKV_LD(kr, i);
      s += iw[(size_t)t * IH + h] * wsc * fmax(d, 0.0f);
    }
  }
  sc[(size_t)tl * ld + j] = s;
}

// Block maxima of the scores (blocks of bs rows), the block of the query's newest visible row +inf.
// grid (nbmax, Tq).
__kernel void ds_block_max(__global const float* sc, int ld, int pos, int rt, int tq0, int bs, __global float* bsc, int ldb) {
  const int b = get_global_id(0), tl = get_global_id(1);
  const int nv = (pos + tq0 + tl + 1) / rt, nb = (nv + bs - 1) / bs;
  float m = -INFINITY;
  if (b < nb) {
    if (b == nb - 1) {
      m = INFINITY;
    } else {
      for (int j = b * bs; j < b * bs + bs && j < nv; ++j) m = fmax(m, sc[(size_t)tl * ld + j]);
    }
  }
  bsc[(size_t)tl * ldb + b] = m;
}

inline uint ds_key(float f) {  // (float order as unsigned order; -inf lowest of the finite ones' side)
  const uint u = as_uint(f);
  return (u & 0x80000000u) ? ~u : u | 0x80000000u;
}

#define DS_TK_MAX 2048
// Per query: the k largest of n scores (-inf never), as indices in increasing order into sel[t][0..k) and the count
// into nsel[t]; equal scores: the lower index first. A radix select over the order-preserving keys, then an ordered
// gather. grid (256, Tq), local (256, 1); k <= DS_TK_MAX.
__kernel void ds_topk(__global const float* sc, int ld, int n, int k, int pos, int rt, int tq0, int per_query_n,
                      int bs_div, __global int* sel, __global int* nsel, int ldsel) {
  const int tl = get_group_id(1), l = get_local_id(0);
  // (per_query_n: n = the query's visible rows, or, with bs_div, its blocks)
  int nn = n;
  if (per_query_n) {
    const int nv = (pos + tq0 + tl + 1) / rt;
    nn = bs_div ? (nv + bs_div - 1) / bs_div : nv;
  }
  __global const float* s = sc + (size_t)tl * ld;
  __local int hist[256];
  __local uint pre_s, need_s;
  __local int cnt_s;
  const uint ninf = ds_key(-INFINITY);
  // valid count
  int c = 0;
  for (int j = l; j < nn; j += 256) c += ds_key(s[j]) > ninf;
  c = work_group_reduce_add(c);
  const int kk = min(k, c);
  __global int* out = sel + (size_t)(tq0 + tl) * ldsel;
  if (kk == 0) {
    if (l == 0) nsel[tq0 + tl] = 0;
    return;
  }
  // the kk-th largest key, 8 bits at a time from the top
  uint prefix = 0, mask = 0;
  int need = kk;  // (how many of the keys with this prefix are still to take)
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = l; i < 256; i += 256) hist[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int j = l; j < nn; j += 256) {
      const uint key = ds_key(s[j]);
      if (key > ninf && (key & mask) == prefix) atomic_inc(&hist[(key >> shift) & 255]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l == 0) {
      int acc = 0, b = 255;
      for (; b > 0; --b) {
        if (acc + hist[b] >= need) break;
        acc += hist[b];
      }
      pre_s = prefix | ((uint)b << shift);
      need_s = (uint)(need - acc);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    prefix = pre_s;
    need = (int)need_s;
    mask |= 255u << shift;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // prefix: the threshold key; take every key above it and the first `need` equal to it (lowest indices)
  __local int idx[DS_TK_MAX];
  if (l == 0) cnt_s = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int j = l; j < nn; j += 256) {
    const uint key = ds_key(s[j]);
    if (key > prefix) idx[atomic_inc(&cnt_s)] = j;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int base = cnt_s;  // (= kk - need)
  int taken = 0;
  for (int j0 = 0; j0 < nn && taken < need; j0 += 256) {  // (taken, need: the same in every work-item)
    const int j = j0 + l;
    const int eq = j < nn && ds_key(s[j]) == prefix ? 1 : 0;
    const int o = work_group_scan_exclusive_add(eq);
    if (eq && taken + o < need) idx[base + taken + o] = j;
    taken += work_group_reduce_add(eq);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // sort the kk indices (bitonic, padded to a power of two)
  int P = 1;
  while (P < kk) P <<= 1;
  for (int i = kk + l; i < P; i += 256) idx[i] = 0x7FFFFFFF;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int size = 2; size <= P; size <<= 1)
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = l; i < P; i += 256) {
        const int pa = i ^ stride;
        if (pa > i) {
          const int a = idx[i], b = idx[pa];
          const bool up = (i & size) == 0;
          if ((a > b) == up) {
            idx[i] = b;
            idx[pa] = a;
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  for (int i = l; i < kk; i += 256) out[i] = idx[i];
  if (l == 0) nsel[tq0 + tl] = kk;
}

// The candidate map of queries tq0.. (zeroed before): the kept blocks (rows of block indices) set. grid (k, Tq).
__kernel void ds_cand_map(__global const int* bsel, __global const int* nbsel, int ldsel, int tq0, __global uchar* cand,
                          int nbmax) {
  const int i = get_global_id(0), tl = get_global_id(1);
  if (i < nbsel[tq0 + tl]) cand[(size_t)(tq0 + tl) * nbmax + bsel[(size_t)(tq0 + tl) * ldsel + i]] = 1;
}

// Attention of query row t (absolute position pos + t), head h: keys = the window positions max(0, p - W + 1) .. p
// (the chunk's kv rows from pos on, the ring before it: slot s % W), then the selected compressed rows; V = K; the
// sink adds exp(sink_h - m) to the denominator; scale 1/sqrt(D). The output's last R values rotated back (inverse
// RoPE at p) and stored group-major: att[g][t][(h % hpg) * D + i], g = h / hpg. One sub-group per (t, h): lane l holds
// values l*32 .. l*32 + 31 (D = 512). grid (16 * H, T), local (16, 1).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void ds_attn(__global const float* q, __global const DSKV_T* kv, __global const DSKV_T* ring, int W,
                      __global const DSKV_T* cc, __global const int* sel, __global const int* nsel, int use_sel, int ldsel,
                      __global const float* sinks, __global const float2* tab, int R, int poff, int pos, int H, int hpg,
                      int Mp, __global float* att) {
  const int h = get_group_id(0), t = get_group_id(1), l = get_sub_group_local_id();
  const int D = 512, p = pos + t;
  float qv[32], o[32];
  __global const float* qr = q + ((size_t)t * H + h) * D + l * 32;
  for (int i = 0; i < 32; ++i) {
    qv[i] = qr[i];
    o[i] = 0.0f;
  }
  const float scale = rsqrt((float)D);
  float m = sinks[h], z = 1.0f;
  const int s0 = max(0, p - W + 1);
  const int nc = use_sel ? nsel[t] : 0;
  for (int kk = s0; kk <= p + nc; ++kk) {
    __global const DSKV_T* kr;
    if (kk <= p) kr = kk >= pos ? kv + (size_t)(kk - pos) * D : ring + (size_t)(kk % W) * D;
    else kr = cc + (size_t)sel[(size_t)t * ldsel + (kk - p - 1)] * D;
    kr += l * 32;
    float kv32[32], d = 0.0f;
    for (int i = 0; i < 32; ++i) {
      kv32[i] = DSKV_LD(kr, i);
      d += qv[i] * kv32[i];
    }
    d = sub_group_reduce_add(d) * scale;
    const float mn = fmax(m, d), cr = exp(m - mn), pe = exp(d - mn);
    z = z * cr + pe;
    for (int i = 0; i < 32; ++i) o[i] = o[i] * cr + pe * kv32[i];
    m = mn;
  }
  const float iz = 1.0f / z;
  for (int i = 0; i < 32; ++i) o[i] *= iz;
  const int r0 = D - R;
  for (int i = 0; i < 32; i += 2) {
    const int e = l * 32 + i;
    if (e >= r0) {
      const float2 cs = tab[(size_t)(t + poff) * (R / 2) + (e - r0) / 2];
      const float a = o[i], b = o[i + 1];
      o[i] = a * cs.x + b * cs.y;
      o[i + 1] = -a * cs.y + b * cs.x;
    }
  }
  const int g = h / hpg;
  __global float* out = att + ((size_t)g * Mp + t) * (hpg * D) + (h % hpg) * D + l * 32;
  for (int i = 0; i < 32; ++i) out[i] = o[i];
}

// The window ring after a chunk: positions max(pos, pos + T - W) .. pos + T - 1 into slots p % W. grid (D, n).
__kernel void ds_ring_store(__global const DSKV_T* kv, __global DSKV_T* ring, int W, int D, int pos, int T) {
  const int d = get_global_id(0), q = get_global_id(1);
  const int p = max(pos, pos + T - W) + q;
  ring[(size_t)(p % W) * D + d] = kv[(size_t)(p - pos) * D + d];
}

// Engram on the streams of row t: per stream s, g = sigmoid(signed sqrt(sum_d x[s][d] k[s][d] gate[s][d] *
// rsqrt(mean x^2 + eps) * rsqrt(mean k^2 + eps) / sqrt(E))), x[s] += g * value (kv[t] = [k0 .. k3 | value]).
// grid (256, T * HC), local (256, 1).
__kernel void ds_engram(__global float* x, __global const float* kv, __global const float* gate, int E, float eps) {
  const int ts = get_group_id(1), t = ts / DS_HC, s = ts % DS_HC, l = get_local_id(0);
  __global float* xr = x + ((size_t)t * DS_HC + s) * E;
  __global const float* kr = kv + ((size_t)t * (DS_HC + 1) + s) * E;
  __global const float* vr = kv + ((size_t)t * (DS_HC + 1) + DS_HC) * E;
  float xx = 0.0f, kk = 0.0f, dt = 0.0f;
  for (int i = l; i < E; i += 256) {
    const float a = xr[i], b = kr[i];
    xx += a * a;
    kk += b * b;
    dt += a * b * gate[(size_t)s * E + i];
  }
  xx = work_group_reduce_add(xx);
  kk = work_group_reduce_add(kk);
  dt = work_group_reduce_add(dt);
  const float dot = dt * rsqrt(xx / E + eps) * rsqrt(kk / E + eps) * rsqrt((float)E);
  const float g = 1.0f / (1.0f + exp(-(dot >= 0.0f ? 1.0f : -1.0f) * sqrt(fmax(fabs(dot), 1e-6f))));
  for (int i = l; i < E; i += 256) xr[i] += g * vr[i];
}

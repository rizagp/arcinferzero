// Part of the non-GEMM kernels (zero/kernels/ops*.cl, KernelParts in zero/clrt.h): built after the parts before
// it, whose definitions it uses. The vision and audio encoders (ViT, Gemma 4 vision and audio; -DVIT builds only).

// ---------------------------------------------------------------- vision encoders (ViT on the GPU)
// Compiled only into the vision encoders' builds of this file (-DVIT): the language model never pays for them.
#ifdef VIT
// The matrix products go through vit_gemm_f16 (or, for the Gemma 4 encoder's int8 path, vit_quant_pad and the int8
// GEMM, kernels/gemm.cl); these kernels do what lies between them. Rows m >= M_valid are padding: quantized as zeros, not written otherwise.

// x[m][n] += a[n] (per_row 0: a bias) or a[m][n] (per_row 1), n < N. grid (N, M_valid).
__kernel void vit_add(__global float* x, int N, __global const float* a, int per_row) {
  const int n = get_global_id(0), m = get_global_id(1);
  x[(size_t)m * N + n] += a[per_row ? (size_t)m * N + n : n];
}

// Qwen3-VL vision attention input: qkv[m] = [q | k | v] (3E, E = nh * QD) plus its bias; q and k get the 2-D rotary
// positions of patch m (pos[m] = (row, column)): pair (i, i + QD/2) of each head turns by row * base^(-i/(QD/4)) for
// i < QD/4 and by column * base^(-(i - QD/4)/(QD/4)) after (Hugging Face Qwen3-VL, ggml GGML_ROPE_TYPE_VISION: two
// sections of QD/4). q is pre-scaled by 1/sqrt(QD). Output per head: [nh][Mp][QD] halves. QD = head size (-DQD, default
// 64; Qwen3.5 9B: 72). grid (nh * QD/2, M_valid).
#ifndef QD
#define QD 64
#endif
__kernel void vit_qkv_rope(__global const float* qkv, __global const float* bias, __global const int2* pos, int nh,
                           float log2_base, int Mp, __global half* q, __global half* k, __global half* v) {
  const int hf = QD / 2, qt = QD / 4;
  const int i = get_global_id(0) % hf, h = get_global_id(0) / hf, m = get_global_id(1), E = nh * QD;
  __global const float* row = qkv + (size_t)m * 3 * E + h * QD;
  __global const float* bb = bias + h * QD;
  const int2 p = pos[m];
  const float theta = (float)(i < qt ? p.x : p.y) * exp2(-log2_base * (float)(i % qt) / (float)qt);
  const float c = cos(theta), s = sin(theta), qs = rsqrt((float)QD);
  const size_t o = ((size_t)h * Mp + m) * QD;
  const float q0 = row[i] + bb[i], q1 = row[i + hf] + bb[i + hf];
  const float k0 = row[E + i] + bb[E + i], k1 = row[E + i + hf] + bb[E + i + hf];
  vstore_half((q0 * c - q1 * s) * qs, o + i, q);
  vstore_half((q0 * s + q1 * c) * qs, o + i + hf, q);
  vstore_half(k0 * c - k1 * s, o + i, k);
  vstore_half(k0 * s + k1 * c, o + i + hf, k);
  vstore_half(row[2 * E + i] + bb[2 * E + i], o + i, v);
  vstore_half(row[2 * E + i + hf] + bb[2 * E + i + hf], o + i + hf, v);
}

// LayerNorm with weight and bias of row m (any E) into out (f32). grid (256, M_valid), local (256, 1).
__kernel void vit_ln(__global const float* x, int E, __global const float* w, __global const float* b, float eps,
                     __global float* out) {
  const int m = get_group_id(1);
  __global const float* xr = x + (size_t)m * E;
  float s = 0.0f;
  for (int i = get_local_id(0); i < E; i += get_local_size(0)) s += xr[i];
  const float mean = work_group_reduce_add(s) / E;
  float qv = 0.0f;
  for (int i = get_local_id(0); i < E; i += get_local_size(0)) qv += (xr[i] - mean) * (xr[i] - mean);
  const float r = rsqrt(work_group_reduce_add(qv) / E + eps);
  for (int i = get_local_id(0); i < E; i += get_local_size(0)) out[(size_t)m * E + i] = (xr[i] - mean) * r * w[i] + b[i];
}

// Vision-encoder GEMM in floating point: C[m][c_off + n] (+)= sum_k A[m][k] * W[n][k], A f32 [M][lda], W the
// mmproj's f16 weights [N][K], f32 accumulation. The ViT is far more sensitive to int8 activations than the
// language model (Qwen3-VL, photos: 10-23 % relative error with int8 GEMM inputs, 0.6-1.8 % in f16). Lane = column
// (16 per sub-group), VTM tokens per sub-group; the sub-groups of a work-group share the weight columns through L1.
// A token row is read as 16 floats per lane (one block read) and broadcast into the FMAs. grid (N, Mp / VTM),
// local (16, WGY); K a multiple of 16, rows m >= M_valid are computed but not stored.
#ifndef VTM
#define VTM 16
#endif
#ifdef NOSHUF  // (timing ablation: wrong results)
#define VSH(a, i) (a)
#else
#define VSH(a, i) intel_sub_group_shuffle((a), (i))
#endif
// Vision rows x [M][lda] (Kv values each) quantized for the engine's int8 GEMM as quant_rows does (per-32 scales; with
// the two-pass defines also the residual), K padded with zeros to Kp, whole superblocks: the vision widths 1152 / 4304
// are not. grid (256, Mp), local (256, 1)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void vit_quant_pad(__global const float* x, int lda, int Kv, int Kp, int M, int M_valid, __global char* aq,
                        __global float* asc, __global ushort* absum) {
  const int m = get_group_id(1), lane = get_sub_group_local_id();
  const bool valid = m < M_valid;
  const int nsg = get_num_groups(0) * get_num_sub_groups();
  for (int s = get_group_id(0) * get_num_sub_groups() + get_sub_group_id(); s < Kp / 256; s += nsg) {
    const int base = s * 256 + 16 * lane;
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = valid && base + i < Kv ? x[(size_t)m * lda + base + i] : 0.0f;
    quant_store16(v, s, m, M, Kp, aq, asc, absum, 1);
  }
}

__attribute__((intel_reqd_sub_group_size(16)))  // (16 activations per block read, shuffled by lane)
__kernel void vit_gemm_f16(__global const float* A, int lda, __global const half* W, int K, __global float* C,
                           int ldc, int c_off, int acc, int M_valid) {
  const int n = get_global_id(0), m0 = get_global_id(1) * VTM;
  float r[VTM];
#pragma unroll
  for (int m = 0; m < VTM; ++m) r[m] = 0.0f;
  __global const half* wr = W + (size_t)n * K;
  for (int k0 = 0; k0 < K; k0 += 16) {
    const float16 w = vload_half16(0, wr + k0);
#pragma unroll
    for (int m = 0; m < VTM; ++m) {
      const float a = as_float(intel_sub_group_block_read((__global const uint*)(A + (size_t)(m0 + m) * lda + k0)));
      float s = r[m];
      s = fma(VSH(a, 0), w.s0, s);
      s = fma(VSH(a, 1), w.s1, s);
      s = fma(VSH(a, 2), w.s2, s);
      s = fma(VSH(a, 3), w.s3, s);
      s = fma(VSH(a, 4), w.s4, s);
      s = fma(VSH(a, 5), w.s5, s);
      s = fma(VSH(a, 6), w.s6, s);
      s = fma(VSH(a, 7), w.s7, s);
      s = fma(VSH(a, 8), w.s8, s);
      s = fma(VSH(a, 9), w.s9, s);
      s = fma(VSH(a, 10), w.sa, s);
      s = fma(VSH(a, 11), w.sb, s);
      s = fma(VSH(a, 12), w.sc, s);
      s = fma(VSH(a, 13), w.sd, s);
      s = fma(VSH(a, 14), w.se, s);
      s = fma(VSH(a, 15), w.sf, s);
      r[m] = s;
    }
  }
#pragma unroll
  for (int m = 0; m < VTM; ++m) {
    if (m0 + m >= M_valid) break;
    __global float* c = C + (size_t)(m0 + m) * ldc + c_off + n;
    *c = acc ? *c + r[m] : r[m];
  }
}


// x[m][n] = act(x[m][n] + b[n]) for the valid rows: the bias (and GELU tanh; gelu = 2: GELU erf) after an f32 GEMM.
// grid (N, M_valid).
__kernel void vit_bias_act(__global float* x, int N, __global const float* b, int gelu) {
  const int n = get_global_id(0), m = get_global_id(1);
  float v = x[(size_t)m * N + n] + b[n];
  if (gelu == 2) v = 0.5f * v * (1.0f + erf(v * 0.70710678f));
  else if (gelu) v = 0.5f * v * (1.0f + tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
  x[(size_t)m * N + n] = v;
}


// Vision attention in f32, lane = query row (SIMD8): each lane holds its row's q (64 values, pre-scaled by 1/8) and 64
// accumulators; a key's K and V rows (f16, 8 dims per lane) are read once per sub-group and broadcast into the FMAs,
// so the softmax needs no cross-lane reductions and no local memory. Tiles of FT keys, online softmax (natural
// exp, as vit_attn). Inputs as vit_qkv_rope writes them ([nh][Mp][64] f16). out[m][h*64 + d].
// grid (ceil(n / (8 * ARSG)) * 8 * ARSG, nh), local (8 * ARSG, 1); -cl-intel-256-GRF-per-thread.
// Tuned on the Arc iGPU (1008-token image, 24 layers): FT 16 -> 1118 ms (registers spill), FT 8 -> 704, FT 4 with
// native_exp -> 588 ms (2.7 TFLOPS; the error against the f32 reference unchanged at 0.2-0.6 %).
#ifndef FT
#define FT 4
#endif
// head size HD (default 64; Qwen3.5 9B's encoder: 72): lane l of the SIMD8 sub-group holds dims PL*l .. PL*l + PL-1
// (-DNO_SG8: 16 lanes, dims interleaved, ceil(HD / 16) per lane; the last block read may run past a row's HD values
//  into the next row or the buffer's slack, never used)
#ifndef HD
#define HD 64
#endif
#ifdef NO_SG8
#define VW 16
#else
#define VW 8
#endif
#define PL ((HD + VW - 1) / VW)
#if PL == 8 && VW == 8
#define VIT_QREG 1
#define KD(r, d) intel_sub_group_shuffle((r)[(d) % PL], (d) / PL)
#else
// other head sizes: interleaved, lane l holds dims l, l + VW, ... (one aligned sub-group block read per VW dims)
#define KD(r, d) intel_sub_group_shuffle((r)[(d) / VW], (d) % VW)
inline void kv_row(__global const half* p, float* r) {
#if VW == 8 && PL >= 8  // (one 8-wide block read for the first 64 dims, the same lane layout: 2 messages per row of 72, not 9)
  const ushort8 a = intel_sub_group_block_read_us8((__global const ushort*)p);
  r[0] = (float)as_half(a.s0); r[1] = (float)as_half(a.s1); r[2] = (float)as_half(a.s2); r[3] = (float)as_half(a.s3);
  r[4] = (float)as_half(a.s4); r[5] = (float)as_half(a.s5); r[6] = (float)as_half(a.s6); r[7] = (float)as_half(a.s7);
#pragma unroll
  for (int i = 8; i < PL; ++i) r[i] = (float)as_half(intel_sub_group_block_read_us((__global const ushort*)(p + VW * i)));
#else
#pragma unroll
  for (int i = 0; i < PL; ++i) r[i] = (float)as_half(intel_sub_group_block_read_us((__global const ushort*)(p + VW * i)));
#endif
}
#endif
#ifdef PRECISE_VEXP
#define AEXP exp
#else
#define AEXP native_exp
#endif
__attribute__((intel_reqd_sub_group_size(VW)))
__kernel void vit_attn_rows(__global const half* q, __global const half* k, __global const half* v, int n, int Mp,
                            int nh, __global float* out) {
  const int lane = get_sub_group_local_id(), h = get_group_id(1), m = get_global_id(0);
  const bool act = m < n;
  __global const half* qp = q + ((size_t)h * Mp + (act ? m : 0)) * HD;
#ifdef VIT_QREG
  float qr[HD];
#pragma unroll
  for (int d = 0; d < HD; ++d) qr[d] = vload_half(d, qp);
#endif
  // (other head sizes: q is read again per key from L1 instead of held: 72-dim q + accumulators spilled)
  float o[HD];
#pragma unroll
  for (int d = 0; d < HD; ++d) o[d] = 0.0f;
  float mx = -INFINITY, l = 0.0f;
  __global const half* kh = k + (size_t)h * Mp * HD;
  __global const half* vh = v + (size_t)h * Mp * HD;
  for (int j0 = 0; j0 < n; j0 += FT) {
    float s[FT], tm = -INFINITY;
#pragma unroll
    for (int jj = 0; jj < FT; ++jj) {
      const int key = min(j0 + jj, n - 1);
      float kr[PL];  // dims PL * lane ..
#ifdef VIT_QREG
      vstore8(vload_half8(lane, kh + (size_t)key * HD), 0, kr);
#else
      kv_row(kh + (size_t)key * HD, kr);
#endif
      float a0 = 0.0f, a1 = 0.0f;
#ifdef VIT_QREG
#pragma unroll
      for (int d = 0; d < HD; d += 2) {
        a0 = fma(qr[d], KD(kr, d), a0);
        a1 = fma(qr[d + 1], KD(kr, d + 1), a1);
      }
#else
#pragma unroll
      for (int c = 0; c < HD / 8; ++c) {
        const float8 qc = vload_half8(c, qp);
        a0 = fma(qc.s0, KD(kr, 8 * c), a0);
        a1 = fma(qc.s1, KD(kr, 8 * c + 1), a1);
        a0 = fma(qc.s2, KD(kr, 8 * c + 2), a0);
        a1 = fma(qc.s3, KD(kr, 8 * c + 3), a1);
        a0 = fma(qc.s4, KD(kr, 8 * c + 4), a0);
        a1 = fma(qc.s5, KD(kr, 8 * c + 5), a1);
        a0 = fma(qc.s6, KD(kr, 8 * c + 6), a0);
        a1 = fma(qc.s7, KD(kr, 8 * c + 7), a1);
      }
#endif
      s[jj] = j0 + jj < n ? a0 + a1 : -INFINITY;
      tm = fmax(tm, s[jj]);
    }
    if (tm > mx) {
      const float corr = AEXP(mx - tm);
      l *= corr;
#pragma unroll
      for (int d = 0; d < HD; ++d) o[d] *= corr;
      mx = tm;
    }
#pragma unroll
    for (int jj = 0; jj < FT; ++jj) {
      const float p = AEXP(s[jj] - mx);
      l += p;
      const int key = min(j0 + jj, n - 1);
      float vr[PL];
#ifdef VIT_QREG
      vstore8(vload_half8(lane, vh + (size_t)key * HD), 0, vr);
#else
      kv_row(vh + (size_t)key * HD, vr);
#endif
#ifndef NOPV
#pragma unroll
      for (int d = 0; d < HD; ++d) o[d] = fma(p, KD(vr, d), o[d]);
#else
      o[jj] += p * vr[jj % PL];
#endif
    }
  }
  if (act) {
    __global float* dst = out + (size_t)m * nh * HD + h * HD;
    const float inv = 1.0f / l;
#pragma unroll
    for (int d = 0; d < HD; ++d) dst[d] = o[d] * inv;
  }
}
// ---- Gemma 4 vision (mmproj gemma4v: the E2B / E4B encoder; zero/vision_gemma4v.h). RMS norms throughout, and
// "clippable" linears: each matrix product's input and output are clamped to ranges stored in the mmproj.

// out[m] = clamp(rms(x[m]) * w, lo, hi). grid (256, M_valid), local (256, 1).
__kernel void g4v_norm(__global const float* x, int ldx, int E, __global const float* w, float eps, float lo, float hi,
                       __global float* out, int ldo) {
  const int m = get_group_id(1);
  __global const float* xr = x + (size_t)m * ldx;
  float s = 0.0f;
  for (int i = get_local_id(0); i < E; i += get_local_size(0)) s += xr[i] * xr[i];
  const float r = rsqrt(work_group_reduce_add(s) / E + eps);
  for (int i = get_local_id(0); i < E; i += get_local_size(0))
    out[(size_t)m * ldo + i] = clamp(xr[i] * r * w[i], lo, hi);
}

// Sandwich norm into the residual: x[m] += rms(clamp(y[m], lo, hi)) * w. grid (256, M_valid), local (256, 1).
__kernel void g4v_norm_add(__global const float* y, int E, __global const float* w, float eps, float lo, float hi,
                           __global float* x) {
  const int m = get_group_id(1);
  __global const float* yr = y + (size_t)m * E;
  float s = 0.0f;
  for (int i = get_local_id(0); i < E; i += get_local_size(0)) {
    const float v = clamp(yr[i], lo, hi);
    s += v * v;
  }
  const float r = rsqrt(work_group_reduce_add(s) / E + eps);
  for (int i = get_local_id(0); i < E; i += get_local_size(0)) x[(size_t)m * E + i] += clamp(yr[i], lo, hi) * r * w[i];
}

// x[m][n] = clamp(x[m][n], lo, hi) for n < N (row stride ld). grid (N, M_valid).
__kernel void g4v_clamp(__global float* x, int ld, float lo, float hi) {
  const int n = get_global_id(0), m = get_global_id(1);
  x[(size_t)m * ld + n] = clamp(x[(size_t)m * ld + n], lo, hi);
}

// Attention input: h[m] = [q | k | v] (3E, E = nh * 64), each clamped to its matrix's output range (cl: q lo, hi,
// k lo, hi, v lo, hi); per head an RMS norm (q, k with weights qn / kn, v without), then 2-D rotary positions in NEOX
// order: dims (i, i + 16), i < 16, turn by column * 100^(-i/16), dims (32 + i, 48 + i) by row * 100^(-i/16)
// (llama.cpp models/gemma4v.cpp). No 1/sqrt(64): Gemma 4's vision attention is unscaled. Output [nh][Mp][64] f16 as
// vit_attn_rows reads it. Lane l of a sub-group holds dims l, l + 16, l + 32, l + 48 of one head and row.
// grid (nh * 16, M_valid), local (16, 1).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void g4v_qkv(__global const float* h, int nh, __global const float* qn, __global const float* kn,
                      __global const int2* pos, float eps, float8 cl, float log2_base, int Mp, __global half* q,
                      __global half* k, __global half* v) {
  const int l = get_sub_group_local_id(), hh = get_global_id(0) / 16, m = get_global_id(1), E = nh * 64;
  const int2 p = pos[m];  // (column, row)
  const float f = exp2(-log2_base * (float)l / 16.0f), th1 = (float)p.x * f, th2 = (float)p.y * f;
  const float c1 = cos(th1), s1 = sin(th1), c2 = cos(th2), s2 = sin(th2);
  const size_t o = ((size_t)hh * Mp + m) * 64;
  const float los[3] = {cl.s0, cl.s2, cl.s4}, his[3] = {cl.s1, cl.s3, cl.s5};
#pragma unroll
  for (int t = 0; t < 3; ++t) {
    __global const float* row = h + (size_t)m * 3 * E + t * E + hh * 64;
    float a0 = clamp(row[l], los[t], his[t]), a1 = clamp(row[l + 16], los[t], his[t]);
    float a2 = clamp(row[l + 32], los[t], his[t]), a3 = clamp(row[l + 48], los[t], his[t]);
    const float r = rsqrt(sub_group_reduce_add(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3) / 64.0f + eps);
    if (t < 2) {
      __global const float* nw = t == 0 ? qn : kn;
      a0 *= r * nw[l]; a1 *= r * nw[l + 16]; a2 *= r * nw[l + 32]; a3 *= r * nw[l + 48];
      const float b0 = a0 * c1 - a1 * s1, b1 = a0 * s1 + a1 * c1;
      const float b2 = a2 * c2 - a3 * s2, b3 = a2 * s2 + a3 * c2;
      a0 = b0; a1 = b1; a2 = b2; a3 = b3;
    } else {
      a0 *= r; a1 *= r; a2 *= r; a3 *= r;
    }
    __global half* dst = t == 0 ? q : t == 1 ? k : v;
    vstore_half(a0, o + l, dst);
    vstore_half(a1, o + l + 16, dst);
    vstore_half(a2, o + l + 32, dst);
    vstore_half(a3, o + l + 48, dst);
  }
}

// g4v_qkv for heads of G4V_HD dims (-DG4V_HD, a multiple of 4 up to 128; Gemma 4 26B-A4B: 72): the same values, lane l
// taking dims l, l + 16, ... of a head; the rotary partner of a dim (NEOX within each half: dims (i, i + G4V_HD / 4)) is
// read from the row again and normed with the same factor. Output [nh][Mp][G4V_HD] f16. grid (nh * 16, M_valid).
#ifndef G4V_HD
#define G4V_HD 64
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void g4v_qkv_hd(__global const float* h, int nh, __global const float* qn, __global const float* kn,
                         __global const int2* pos, float eps, float8 cl, float log2_base, int Mp, __global half* q,
                         __global half* k, __global half* v) {
  const int l = get_sub_group_local_id(), hh = get_global_id(0) / 16, m = get_global_id(1), E = nh * G4V_HD, Q = G4V_HD / 4;
  const int2 p = pos[m];  // (column, row)
  const size_t o = ((size_t)hh * Mp + m) * G4V_HD;
  const float los[3] = {cl.s0, cl.s2, cl.s4}, his[3] = {cl.s1, cl.s3, cl.s5};
  for (int t = 0; t < 3; ++t) {
    __global const float* row = h + (size_t)m * 3 * E + t * E + hh * G4V_HD;
    float ss = 0.0f;
    for (int d = l; d < G4V_HD; d += 16) {
      const float a = clamp(row[d], los[t], his[t]);
      ss += a * a;
    }
    const float r = rsqrt(sub_group_reduce_add(ss) / (float)G4V_HD + eps);
    __global half* dst = t == 0 ? q : t == 1 ? k : v;
    for (int d = l; d < G4V_HD; d += 16) {
      float a = clamp(row[d], los[t], his[t]) * r;
      if (t < 2) {
        __global const float* nw = t == 0 ? qn : kn;
        const int i = d % Q, second = d % (2 * Q) >= Q, pd = second ? d - Q : d + Q;
        const float b = clamp(row[pd], los[t], his[t]) * r * nw[pd];
        a *= nw[d];
        const float th = (float)(d < 2 * Q ? p.x : p.y) * exp2(-log2_base * (float)i / (float)Q);
        const float c = cos(th), s = sin(th);
        a = second ? b * s + a * c : a * c - b * s;
      }
      vstore_half(a, o + d, dst);
    }
  }
}

// x[t] = (x[t] - bias) * scale (Gemma 4 vision std_bias / std_scale, after the pooling). grid (E, n_tok).
__kernel void g4v_std(__global float* x, int E, __global const float* bias, __global const float* scale) {
  const int e = get_global_id(0), t = get_global_id(1);
  x[(size_t)t * E + e] = (x[(size_t)t * E + e] - bias[e]) * scale[e];
}

// FFN: out[m][n] = clamp(gelu_tanh(clamp(g, cl.s0, cl.s1)) * clamp(u, cl.s2, cl.s3), cl.s4, cl.s5) with
// h[m] = [gate | up] (2F); the last clamp is the down projection's input range. grid (F, M_valid).
__kernel void g4v_geglu(__global const float* h, int F, float8 cl, __global float* out) {
  const int n = get_global_id(0), m = get_global_id(1);
  const float g = clamp(h[(size_t)m * 2 * F + n], cl.s0, cl.s1), u = clamp(h[(size_t)m * 2 * F + F + n], cl.s2, cl.s3);
  const float ge = 0.5f * g * (1.0f + tanh(0.7978845608f * (g + 0.044715f * g * g * g)));
  out[(size_t)m * F + n] = clamp(ge * u, cl.s4, cl.s5);
}

// 3x3 (K x K) average pooling over the px x py patch grid (row-major patches), times sqrt(E):
// out[t][e], t over the (px / K) x (py / K) grid in row-major order. grid (E, n_tok).
__kernel void g4v_pool(__global const float* x, int E, int px, int K, float scale, __global float* out) {
  const int e = get_global_id(0), t = get_global_id(1), ox = px / K;
  const int ty = t / ox, tx = t % ox;
  float s = 0.0f;
  for (int dy = 0; dy < K; ++dy)
    for (int dx = 0; dx < K; ++dx) s += x[((size_t)(ty * K + dy) * px + tx * K + dx) * E + e];
  out[(size_t)t * E + e] = s / (float)(K * K) * scale;
}
// ---- Gemma 4 audio conformer (mmproj gemma4a: the E2B / E4B audio encoder; zero/audio_gemma4a.h)

// x[m][n] = clamp(silu(x[m][n]), lo, hi) for n < N. grid (N, M_valid).
__kernel void a4_silu(__global float* x, int N, float lo, float hi) {
  const int n = get_global_id(0), m = get_global_id(1);
  const float v = x[(size_t)m * N + n];
  x[(size_t)m * N + n] = clamp(v / (1.0f + exp(-v)), lo, hi);
}

// Half-step FFN residual: x[m] += s * rms(clamp(y[m], lo, hi)) * w. grid (256, M_valid), local (256, 1).
__kernel void a4_norm_axpy(__global const float* y, int E, __global const float* w, float eps, float lo, float hi,
                           float s, __global float* x) {
  const int m = get_group_id(1);
  __global const float* yr = y + (size_t)m * E;
  float ss = 0.0f;
  for (int i = get_local_id(0); i < E; i += get_local_size(0)) {
    const float v = clamp(yr[i], lo, hi);
    ss += v * v;
  }
  const float r = rsqrt(work_group_reduce_add(ss) / E + eps);
  for (int i = get_local_id(0); i < E; i += get_local_size(0)) x[(size_t)m * E + i] += s * clamp(yr[i], lo, hi) * r * w[i];
}

// Chunked local self-attention of the Gemma 4 audio conformer, written out per query: query t sees keys
// t - (W - 1) .. t (W = 12: llama.cpp's blocked mask allows 0 <= t - key < 12), score = q . k + q . p[12 - d]
// (d = t - key, p = the projected sinusoidal relative positions [13][E]), soft-capped (cap * tanh(s / cap)), softmax.
// qkv[m] = [q | k | v] (3E), each clamped to its output range (cl), q scaled by qs * pds[dim], k by ks.
// A sub-group per (head, query): lane l holds dims l + 16 j of the head (D = 128). out[m][h*D + d].
// grid (nh * 16, n), local (16, 1).
#define A4W 12
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void a4_attn(__global const float* qkv, int nh, int D, __global const float* p, __global const float* pds,
                      float qs, float ks, float cap, float8 cl, __global float* out) {
  const int l = get_sub_group_local_id(), h = get_global_id(0) / 16, t = get_global_id(1), E = nh * D, J = D / 16;
  float qv[8];
  for (int j = 0; j < J; ++j) {
    const int d = l + 16 * j;
    qv[j] = clamp(qkv[(size_t)t * 3 * E + h * D + d], cl.s0, cl.s1) * qs * pds[d];
  }
  float sc[A4W];
  float mx = -INFINITY;
  for (int w = 0; w < A4W; ++w) {
    const int key = t - w;  // d = w
    float s = 0.0f;
    if (key >= 0) {
      __global const float* kr = qkv + (size_t)key * 3 * E + E + h * D;
      __global const float* pr = p + (size_t)(12 - w) * E + h * D;
      for (int j = 0; j < J; ++j) {
        const int d = l + 16 * j;
        s += qv[j] * (clamp(kr[d], cl.s2, cl.s3) * ks + pr[d]);
      }
      s = sub_group_reduce_add(s);
      s = cap * tanh(s / cap);
      mx = fmax(mx, s);
    }
    sc[w] = s;
  }
  float den = 0.0f;
  for (int w = 0; w < A4W; ++w) {
    sc[w] = t - w >= 0 ? exp(sc[w] - mx) : 0.0f;
    den += sc[w];
  }
  for (int j = 0; j < J; ++j) {
    const int d = l + 16 * j;
    float o = 0.0f;
    for (int w = 0; w < A4W; ++w)
      if (t - w >= 0) o += sc[w] * clamp(qkv[(size_t)(t - w) * 3 * E + 2 * E + h * D + d], cl.s4, cl.s5);
    out[(size_t)t * E + h * D + d] = o / den;
  }
}

// Convolution module, after pointwise conv 1 (h[m] = [a | b], 2E, clamped to [lo, hi]): GLU a * sigmoid(b), then a
// causal depthwise conv of width K over time (w[c][j], j = K - 1 the current step). grid (E, n).
__kernel void a4_glu_dw(__global const float* h, int E, int K, __global const float* w, float lo, float hi,
                        __global float* out) {
  const int c = get_global_id(0), t = get_global_id(1);
  float s = 0.0f;
  for (int j = 0; j < K; ++j) {
    const int tt = t - (K - 1) + j;
    if (tt < 0) continue;
    const float a = clamp(h[(size_t)tt * 2 * E + c], lo, hi), b = clamp(h[(size_t)tt * 2 * E + E + c], lo, hi);
    s += w[(size_t)c * K + j] * (a / (1.0f + exp(-b)));
  }
  out[(size_t)t * E + c] = s;
}
#endif  // VIT


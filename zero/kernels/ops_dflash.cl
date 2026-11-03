// Part of the non-GEMM kernels (zero/kernels/ops*.cl, KernelParts in zero/clrt.h): built after the parts before
// it, whose definitions it uses. The DFlash2 block drafter (and dfl_add, which the multi-GPU path uses too).

// ---- DFlash2 block drafter (z-lab/dflash model.py; zero/dflash.h). Rows of a block of DB positions.
// Grouped dynamic causal conv over the block rows: out[t][c] = sum_k (base[k][c] + dyn[t][half][k][g(c)]) x[t-k][c]
// (x rows before 0 are zero), dyn rows [2][K][E/G] of ldd floats. grid (E, rows).
__kernel void dfl_conv(__global const float* x, __global const float* dyn, int ldd, int half_, __global const float* base,
                       int E, int G, int K, __global float* out) {
  const int c = get_global_id(0), t = get_global_id(1), N = E / G, g = c / G;
  float acc = 0.0f;
  for (int k = 0; k < K && k <= t; ++k)
    acc += (base[k * E + c] + dyn[(size_t)t * ldd + (half_ * K + k) * N + g]) * x[(size_t)(t - k) * E + c];
  out[(size_t)t * E + c] = acc;
}
// RMS norm (w) + RoPE over all QD dims of each head, in place: rows [rows][ld], heads at h * QD. grid (16 * nh, rows).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void dfl_norm_rope(__global float* x, int ld, __global const float* w, float eps, int pos0, float theta_scale) {
  const int m = get_global_id(1), hh = get_group_id(0), lane = get_sub_group_local_id();
  __global float* r = x + (size_t)m * ld + hh * QD;
  float v[QJ];
#pragma unroll
  for (int j = 0; j < QJ; ++j) v[j] = r[lane + 16 * j];
  qsa_norm_rope(v, w, eps, pos0 + m, theta_scale, QD);
#pragma unroll
  for (int j = 0; j < QJ; ++j) r[lane + 16 * j] = v[j];
}
// k, v rows [rows][KVW] (k normed and rotated) into the context rings [R][KVW] at positions pos0 + row. grid (KVW, rows).
__kernel void dfl_kv_store(__global const float* k, __global const float* v, int KVW, int pos0, int R,
                           __global float* rk, __global float* rv) {
  const int i = get_global_id(0), m = get_global_id(1);
  const size_t dst = (size_t)((pos0 + m) % R) * KVW + i;
  rk[dst] = k[(size_t)m * KVW + i];
  rv[dst] = v[(size_t)m * KVW + i];
}
// Block attention: query row i (position start + i) of head h over the context keys at positions > start + i - win
// (ring of R positions) and every block key (non-causal). q [rows][NH * QD], kb / vb [rows][NKV * QD] (k normed +
// rotated), out [rows][NH * QD]. grid (QD, rows * NH), local (QD, 1).
__kernel void dfl_attn(__global const float* q, __global const float* kb, __global const float* vb,
                       __global const float* rk, __global const float* rv, int R, int start, int rows, int win, int NH,
                       int NKV, __global float* out) {
  const int d = get_local_id(0), i = get_group_id(1) / NH, h = get_group_id(1) % NH, g = h / (NH / NKV);
  const int KVW = NKV * QD, lo = max(0, start + i - win + 1), nc = start - lo, nk = nc + rows;
  __local float qv[QD];
  __local float sc[4096 + 64];
  __local float red[QD];
  qv[d] = q[((size_t)i * NH + h) * QD + d] * rsqrt((float)QD);
  barrier(CLK_LOCAL_MEM_FENCE);
  float mx = -INFINITY;
  for (int k = d; k < nk; k += QD) {
    __global const float* kr = k < nc ? rk + (size_t)((lo + k) % R) * KVW + g * QD : kb + (size_t)(k - nc) * KVW + g * QD;
    float s = 0.0f;
    for (int e = 0; e < QD; ++e) s += qv[e] * kr[e];
    sc[k] = s;
    mx = fmax(mx, s);
  }
  mx = work_group_reduce_max(mx);
  float z = 0.0f;
  for (int k = d; k < nk; k += QD) {
    const float p = exp(sc[k] - mx);
    sc[k] = p;
    z += p;
  }
  z = work_group_reduce_add(z);
  barrier(CLK_LOCAL_MEM_FENCE);
  float acc = 0.0f;
  for (int k = 0; k < nk; ++k) {
    __global const float* vr = k < nc ? rv + (size_t)((lo + k) % R) * KVW + g * QD : vb + (size_t)(k - nc) * KVW + g * QD;
    acc += sc[k] * vr[d];
  }
  out[((size_t)i * NH + h) * QD + d] = acc / z;
}
// x += y over rows [rows][E]. grid (E, rows).
__kernel void dfl_add(__global float* x, __global const float* y, int E) {
  const size_t j = (size_t)get_global_id(1) * E + get_global_id(0);
  x[j] += y[j];
}

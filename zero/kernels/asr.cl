#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// The Qwen3-ASR audio encoder's own kernels (mmproj qwen3a, zero/audio_qwen3a.h); its matrix products, LayerNorms,
// bias / GELU and attention are the vision encoders' vit_* kernels of ops.cl (-DVIT).
// Activations of the convolutions: [chunk][time][freq][channel] rows of C floats.

// im2col of a 3x3 stride-2 convolution with zero padding 1 over (time, freq): row (chunk, to, fo) of the output holds
// in[chunk][2 to + kw - 1][2 fo + kh - 1][ic] at column ic * 9 + kh * 3 + kw (the mmproj's weight order). grid
// (C * 9, chunks * To * Fo).
__kernel void q3a_im2col(__global const float* in, int C, int Ti, int Fi, int To, int Fo, __global float* out) {
  const int col = get_global_id(0), row = get_global_id(1);
  const int ic = col / 9, kh = (col / 3) % 3, kw = col % 3;
  const int fo = row % Fo, to = (row / Fo) % To, ch = row / (Fo * To);
  const int t = 2 * to + kw - 1, f = 2 * fo + kh - 1;
  float v = 0.0f;
  if (t >= 0 && t < Ti && f >= 0 && f < Fi) v = in[(((size_t)ch * Ti + t) * Fi + f) * C + ic];
  out[(size_t)row * C * 9 + col] = v;
}
// The kept frames of the last convolution as encoder input rows, features in the order channel * F + freq:
// out[r][c * F + f] = in[tmap[r]][f][c] (tmap: the frame's row index chunk * T + t). grid (C * F, n).
__kernel void q3a_gather(__global const float* in, int C, int F, __global const int* tmap, __global float* out) {
  const int col = get_global_id(0), r = get_global_id(1), c = col / F, f = col % F;
  out[(size_t)r * C * F + col] = in[((size_t)tmap[r] * F + f) * C + c];
}
// x[r] += pos[tpos[r]] (the sinusoidal position of the frame within its chunk). grid (E, n).
__kernel void q3a_pos(__global float* x, int E, __global const float* pos, __global const int* tpos) {
  const int i = get_global_id(0), r = get_global_id(1);
  x[(size_t)r * E + i] += pos[(size_t)tpos[r] * E + i];
}
// q / k / v rows [M][3E] (biases added) into f16 head-major [nh][Mp][HD] for vit_attn_rows, q scaled by qs.
// grid (E, M_valid).
__kernel void q3a_heads(__global const float* qkv, int nh, int hd, int Mp, float qs, __global half* q, __global half* k,
                        __global half* v) {
  const int i = get_global_id(0), m = get_global_id(1), E = nh * hd, h = i / hd, d = i % hd;
  __global const float* row = qkv + (size_t)m * 3 * E;
  const size_t o = ((size_t)h * Mp + m) * hd + d;
  vstore_half(row[i] * qs, o, q);
  vstore_half(row[E + i], o, k);
  vstore_half(row[2 * E + i], o, v);
}

// Multi-GPU (tensor parallel, zero/mgpu/): the split model's own kernels, apart from the one-GPU kernels.

// x[i] += part 0 + part 1 + ... (each GPU's partial sum of an output projection, in pinned host memory), added in
// rank order so every GPU's residual stream gets the same float sum
__kernel void tp_add(__global float* x, __global const float* p, ulong off, int n, int np, ulong stride) {
  const int i = get_global_id(0);
  if (i >= n) return;
  float s = x[i];
  for (int k = 0; k < np; ++k) s += p[off + k * stride + i];
  x[i] = s;
}

// rows of `ld` values (a GEMM's output padded to its tiles of 16) packed to rows of n
__kernel void tp_pack(__global const float* src, int ld, __global float* dst, int n, int rows) {
  const int c = get_global_id(0), r = get_global_id(1);
  if (c < n && r < rows) dst[(size_t)r * n + c] = src[(size_t)r * ld + c];
}

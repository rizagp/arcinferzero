// Masked float attention against a double CPU reference, including excluded leading tiles and tail blocks.
// Linux: g++ -std=c++20 -O2 -mavx2 -mf16c -include src/cpu_check.h -include src/log.h -Ithird_party tests/test_qsa_float.cpp -ldl -o /tmp/test_qsa_float
#include "../zero/clrt.h"
#include "../src/quants.h"
#include <random>
#include <cstdio>

int main() {
#ifdef __linux__
  setenv("ARC_NO_KCACHE", "1", 1);  // Standalone tests must not leave Windows-style cache paths in the checkout.
#endif
  ClRt cl;
  if (!cl.init()) return 1;
  const std::string src = read_file("zero/kernels/ops.cl") + read_file("zero/kernels/ops_attn.cl");
  const std::string flags = "-cl-std=CL3.0 -cl-mad-enable -DSG=16 -DNO_SG8 -DGQ=2 -DFA_QSA";
  cl_program ordinary = cl.build(src, flags), split_program = cl.build(src, flags + " -DFA_QSA_SPLIT");
  std::mt19937 rng(81);
  std::normal_distribution<float> nd;
  const int nh = 4, nkv = 1, hd = 256;
  int failures = 0, checks = 0;
  for (int split : {0, 1}) for (int cap : {80, 262144}) for (int qr : {2, 4, 8}) for (int T : {1, 3, 17}) for (int mode = 0; mode < 4; ++mode) {
    if (cap > 80 && (qr != 4 || T != 3 || mode != 3)) continue;
    const int pos0 = cap > 80 ? cap - 24 : 41;
    const int words = (cap / qr + 31) / 32, qmw = mode ? words : 0;
    std::vector<float> q((size_t)T * nh * hd), out(q.size());
    std::vector<uint16_t> k(cap * hd), v(cap * hd);
    for (float& x : q) x = nd(rng) * 0.08f;
    for (auto& x : k) x = f32_to_fp16(nd(rng));
    for (auto& x : v) x = f32_to_fp16(nd(rng));
    std::vector<uint32_t> masks(T * words, 0);
    std::vector<std::vector<int>> keys(T);
    for (int t = 0; t < T; ++t) {
      const int pos = pos0 + t, complete = (pos + 1) / qr;
      for (int block = 0; block < complete; ++block) {
        const bool selected = mode == 0 || (mode == 1 && block * qr >= 32) ||
                              (mode == 2 && block % 3 == 1) ||
                              (mode == 3 && block == complete - 1);
        if (!selected) continue;
        masks[t * words + block / 32] |= 1u << (block % 32);
        for (int i = 0; i < qr; ++i) keys[t].push_back(block * qr + i);
      }
      for (int key = complete * qr; key <= pos; ++key) keys[t].push_back(key);
    }
    cl_mem bq = cl.buffer(CL_MEM_READ_ONLY, q.size() * 4, q.data());
    cl_mem bk = cl.buffer(CL_MEM_READ_ONLY, k.size() * 2, k.data());
    cl_mem bv = cl.buffer(CL_MEM_READ_ONLY, v.size() * 2, v.data());
    cl_mem bm = cl.buffer(CL_MEM_READ_ONLY, masks.size() * 4, masks.data());
    cl_mem bo = cl.buffer(CL_MEM_READ_WRITE, out.size() * 4);
    const int ns = split ? (cap > 80 ? 64 : 7) : 1;
    cl_mem parts = cl.buffer(CL_MEM_READ_WRITE, (size_t)T * nh * ns * 258 * 4);
    cl_kernel kernel = cl.kernel(split ? split_program : ordinary, "attn_fa");
    set_args(kernel, 0, bq, bk, bv, cap, nh, nkv, pos0, T, bo, bm, qmw, qr);
    if (split) set_args(kernel, 12, ns, parts);
    size_t global[3] = {16, nh / 2, (size_t)((T * ns + 15) / 16 * 16)}, local[3] = {16, 1, 16};
    CL_CHECK(clEnqueueNDRangeKernel(cl.queue, kernel, 3, nullptr, global, local, 0, nullptr, nullptr));
    if (split) {
      cl_kernel combine = cl.kernel(split_program, "qsa_fa_combine");
      set_args(combine, 0, parts, nh, ns, bo);
      size_t cg[3] = {256, nh, (size_t)T}, cz[3] = {256, 1, 1};
      CL_CHECK(clEnqueueNDRangeKernel(cl.queue, combine, 3, nullptr, cg, cz, 0, nullptr, nullptr));
      clReleaseKernel(combine);
    }
    CL_CHECK(clEnqueueReadBuffer(cl.queue, bo, CL_TRUE, 0, out.size() * 4, out.data(), 0, nullptr, nullptr));
    double error = 0, norm = 0;
    for (int t = 0; t < T; ++t) for (int h = 0; h < nh; ++h) {
      std::vector<double> scores;
      double maximum = -1e300, sum = 0;
      for (int key : keys[t]) {
        double s = 0;
        for (int d = 0; d < hd; ++d) s += q[((size_t)t * nh + h) * hd + d] * (double)fp16_to_f32(k[key * hd + d]);
        scores.push_back(s); maximum = std::max(maximum, s);
      }
      for (double& s : scores) { s = std::exp(s - maximum); sum += s; }
      for (int d = 0; d < hd; ++d) {
        double want = 0;
        for (size_t j = 0; j < keys[t].size(); ++j) want += scores[j] / sum * fp16_to_f32(v[keys[t][j] * hd + d]);
        const double got = out[((size_t)t * nh + h) * hd + d];
        if (!std::isfinite(got)) { ++failures; continue; }
        error += (got - want) * (got - want); norm += want * want;
      }
    }
    const double relative = std::sqrt(error / norm);
    const bool ok = relative < 2e-6;
    failures += !ok; ++checks;
    printf("split=%d cap=%d QR=%d T=%d mode=%d relative_L2=%.3g %s\n", split, cap, qr, T, mode, relative, ok ? "PASS" : "FAIL");
    for (cl_mem b : {bq, bk, bv, bm, bo, parts}) clReleaseMemObject(b);
    clReleaseKernel(kernel);
  }
  printf("masked float attention: %d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}

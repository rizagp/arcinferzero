// Gemma 4's float prompt attention (zero/kernels/ops_attn_g.cl attn_fa_gh) at a given depth, and variants of it:
// random f32 queries and f16 K / V rows, one chunk of T queries ending at position N - 1 over every key before them
// (causal), timed; each variant's output compared bit for bit with the first build's.
// usage: gattn_bench [N=32768] [T=512] [-- options of build 1] [-- options of build 2] ...
//   each build: the engine's options for the layer (-DFA_GH -DHD=.. -DHPG=.. -DFS=.. -DGR=.. -DKB=.. -DK_F16) plus extra
// a build's -DFS=n: its own sub-group count (GR stays the run's, from ATT_FS / ATT_GR)
// env: ATT_NH (q heads, 16), ATT_NKV (kv heads, 1), ATT_HD (512), ATT_FS (8), ATT_GR (as the engine), ATT_KB (1),
//      ATT_REPS (timed runs, 3; the best is reported)
// e.g. Gemma 4 12B global layers (the defaults); 26B global: ATT_NH=16 ATT_NKV=2; E4B global: ATT_NH=8 ATT_NKV=2
#include "../src/clrt.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

static int env(const char* k, int def) { return getenv(k) ? atoi(getenv(k)) : def; }
static uint16_t f2h(float f) {  // (round to nearest even, normal range only: the test data stays within it)
  uint32_t u;
  memcpy(&u, &f, 4);
  const uint32_t s = (u >> 16) & 0x8000u;
  int e = (int)((u >> 23) & 0xFF) - 127 + 15;
  uint32_t m = u & 0x7FFFFFu;
  if (e <= 0) return (uint16_t)s;
  uint32_t h = s | (uint32_t)e << 10 | m >> 13;
  const uint32_t rest = m & 0x1FFFu;
  if (rest > 0x1000u || (rest == 0x1000u && (h & 1))) ++h;
  return (uint16_t)h;
}

int main(int argc, char** argv) {
  int N = 32768, T = 512, a = 1;
  if (a < argc && strcmp(argv[a], "--")) N = atoi(argv[a++]);
  if (a < argc && strcmp(argv[a], "--")) T = atoi(argv[a++]);
  std::vector<std::string> builds;
  for (; a < argc; ++a) {
    if (!strcmp(argv[a], "--")) { builds.emplace_back(); continue; }
    if (builds.empty()) builds.emplace_back();
    builds.back() += std::string(" ") + argv[a];
  }
  if (builds.empty()) builds.emplace_back();
  const int nh = env("ATT_NH", 16), nkv = env("ATT_NKV", 1), hd = env("ATT_HD", 512), hpg = nh / nkv;
  const int fs = env("ATT_FS", hd >= 512 ? 8 : 4), gdim = hd / fs / 16;
  const int gr = env("ATT_GR", std::clamp(64 / (hpg * gdim), 1, 4)), kb = env("ATT_KB", 1), reps = env("ATT_REPS", 3);
  if (T < 1 || T > N) { fprintf(stderr, "T must be 1..N\n"); return 1; }
  const int cap = (N + 15) / 16 * 16, p0 = N - T;
  ClRt cl;
  if (!cl.init()) return 1;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> q((size_t)T * nh * hd);
  for (auto& v : q) v = nd(rng) * 0.06f;
  std::vector<uint16_t> k((size_t)nkv * cap * hd), v((size_t)nkv * cap * hd);
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = f2h(nd(rng) * (i % 97 == 5 ? 4.0f : 1.0f));
    v[i] = f2h(nd(rng));
  }
  cl_mem bq = cl.buffer(CL_MEM_READ_ONLY, q.size() * 4, q.data()), bk = cl.buffer(CL_MEM_READ_ONLY, k.size() * 2, k.data());
  cl_mem bv = cl.buffer(CL_MEM_READ_ONLY, v.size() * 2, v.data());
  cl_mem bo = cl.buffer(CL_MEM_READ_WRITE, (size_t)T * nh * hd * 4);
  const std::string src = read_file("zero/kernels/ops.cl") + read_file("zero/kernels/ops_attn.cl") + read_file("zero/kernels/ops_attn_g.cl");
  std::vector<float> ref, out((size_t)T * nh * hd);
  const double pairs = (double)T * (p0 + (T + 1) / 2.0), flop = 4.0 * hd * nh * pairs;  // (QK and PV, 2 per FMA)
  printf("nh %d nkv %d hd %d | N %d, %d queries | FS %d GR %d KB %d\n", nh, nkv, hd, N, T, fs, gr, kb);
  for (size_t bi = 0; bi < builds.size(); ++bi) {
    // (-DFS=n in a build: its own sub-group count)
    const size_t fpos = builds[bi].find("-DFS=");
    const int fsb = fpos != std::string::npos ? atoi(builds[bi].c_str() + fpos + 5) : fs;
    const std::string opts = "-cl-std=CL3.0 -cl-mad-enable -cl-intel-256-GRF-per-thread -DFA_GH -DK_F16 -DHD=" +
                             std::to_string(hd) + " -DHPG=" + std::to_string(hpg) + (fpos != std::string::npos ? std::string() : " -DFS=" + std::to_string(fs)) +
                             " -DKB=" + std::to_string(kb) + builds[bi];
    const int grk = gr;
    cl_kernel kr = cl.kernel(cl.build(src, opts + " -DGR=" + std::to_string(grk)), "attn_fa_gh");
    int cp = cap, nn = nh, nk = nkv, pp = p0, MM = T, w = 0, nlo = -1, nhi = -1;
    set_args(kr, 0, bq, bk, bv, cp, nn, nk, pp, MM, bo, w, nlo, nhi);
    size_t gws[3] = {(size_t)16 * fsb, (size_t)nkv, (size_t)(T + grk - 1) / grk}, lws[3] = {(size_t)16 * fsb, 1, 1};
    double best = 1e30;
    {  // (the output as NaNs first: a build that skips rows cannot pass as the one before it)
      const uint32_t nan = 0xFFFFFFFFu;
      CL_CHECK(clEnqueueFillBuffer(cl.queue, bo, &nan, 4, 0, out.size() * 4, 0, nullptr, nullptr));
    }
    for (int r = 0; r <= reps; ++r) {  // (run 0: warm-up)
      cl_event e;
      CL_CHECK(clEnqueueNDRangeKernel(cl.queue, kr, 3, nullptr, gws, lws, 0, nullptr, &e));
      const double ms = ClRt::event_ms(e);
      clReleaseEvent(e);
      if (r) best = std::min(best, ms);
    }
    clEnqueueReadBuffer(cl.queue, bo, CL_TRUE, 0, out.size() * 4, out.data(), 0, nullptr, nullptr);
    size_t nd2 = 0;
    double md = 0;
    if (bi == 0) ref = out;
    else
      for (size_t i = 0; i < out.size(); ++i) {
        nd2 += memcmp(&out[i], &ref[i], 4) != 0;
        md = std::max(md, (double)std::fabs(out[i] - ref[i]));
      }
    printf("build %zu%s: %.2f ms, %.2f TFLOPS, spill %zu B | %s\n", bi, builds[bi].c_str(), best, flop / (best * 1e-3) / 1e12,
           cl.spill_bytes(kr), bi == 0 ? "reference" : nd2 == 0 ? "bit-identical" : ("DIFFERS: " + std::to_string(nd2) + " values, max " + std::to_string(md)).c_str());
  }
  return 0;
}

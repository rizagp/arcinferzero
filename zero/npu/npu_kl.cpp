// arcinferzero NPU experiment: KL divergence between two files of next-token log-probabilities (T x V floats, as
// tools/ref_forward writes them with REF_ALL=1): mean and largest KL(a || b) over the positions, and how often the
// most likely token is the same. usage: npu_kl a.bin b.bin vocab
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<float> load(const char* p) {
  FILE* f = fopen(p, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
  _fseeki64(f, 0, SEEK_END);
  std::vector<float> v((size_t)_ftelli64(f) / 4);
  _fseeki64(f, 0, SEEK_SET);
  if (fread(v.data(), 4, v.size(), f) != v.size()) { fprintf(stderr, "cannot read %s\n", p); exit(1); }
  fclose(f);
  return v;
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: npu_kl a.bin b.bin vocab\n"); return 1; }
  const std::vector<float> a = load(argv[1]), b = load(argv[2]);
  const size_t V = (size_t)atoll(argv[3]);
  if (!V || a.size() != b.size() || a.size() % V) { fprintf(stderr, "sizes differ or are not a multiple of the vocabulary\n"); return 1; }
  const size_t T = a.size() / V;
  double sum = 0, mx = 0;
  size_t same = 0;
  for (size_t t = 0; t < T; ++t) {
    const float* p = &a[t * V];
    const float* q = &b[t * V];
    double kl = 0;
    for (size_t i = 0; i < V; ++i) kl += std::exp((double)p[i]) * ((double)p[i] - q[i]);
    sum += kl;
    mx = std::max(mx, kl);
    same += std::max_element(p, p + V) - p == std::max_element(q, q + V) - q;
  }
  printf("KL mean %.5f max %.4f over %zu positions, top-1 %zu / %zu\n", sum / T, mx, T, same, T);
  return 0;
}

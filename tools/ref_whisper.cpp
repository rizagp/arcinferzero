// The f32 CPU reference of Whisper (src/whisper.h's Backend): every matrix product, norm and attention in f32 from
// the f16 weights, written plainly; slow (an encoder window takes seconds) but independent of the GPU / NPU engine,
// which is checked against it (docs/whisper-native.md). Also a transcriber on its own:
//
//   ref_whisper <model folder> <audio / video file> [--lang auto|id|en|...] [--beam N] [--words] [--from s] [--to s]
//               [--prompt text] [--json out.json] [--dump-encoder out.bin] [--dump-logits out.bin]
//
// --dump-encoder writes the first window's encoder output (f32 [1500][1280]); --dump-logits every decoder step's
// logits of the first window's greedy decoding (f32 [steps][vocab]): the engine's runs compare against them.
#include "../src/quants.h"
#include "../src/whisper.h"
#include <chrono>
#include <cstdio>

using namespace whisper;

namespace {

// Y[M][N] (+)= X[M][K] . W[N][K]^T + b (W, b f16; b may be null)
void linear(const float* X, int M, int K, const uint16_t* W, const uint16_t* b, int N, float* Y) {
#pragma omp parallel
  {
    std::vector<float> wt((size_t)16 * K);
#pragma omp for schedule(dynamic, 4)
    for (int n0 = 0; n0 < N; n0 += 16) {
      const int nb = std::min(16, N - n0);
      for (int j = 0; j < nb; ++j)
        for (int k = 0; k < K; ++k) wt[(size_t)j * K + k] = h2f(W[(size_t)(n0 + j) * K + k]);
      // (checks: WH_Q8W=1 rounds the weights as the engine's Q8_0 form does: int8 with one scale per 32 values)
      static const bool q8w = getenv("WH_Q8W") && atoi(getenv("WH_Q8W")) != 0;
      if (q8w)
        for (int j = 0; j < nb; ++j)
          for (int bk = 0; bk + 32 <= K; bk += 32) {
            float* w = &wt[(size_t)j * K + bk];
            float amax = 0;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(w[i]));
            const float d = amax / 127.0f, inv = d > 0 ? 1.0f / d : 0.0f, dh = h2f(f32_to_fp16(d));
            for (int i = 0; i < 32; ++i) w[i] = (float)std::lrintf(w[i] * inv) * dh;
          }
      for (int m = 0; m < M; ++m) {
        const float* x = X + (size_t)m * K;
        for (int j = 0; j < nb; ++j) {
          const float* w = &wt[(size_t)j * K];
          float s = 0;
          for (int k = 0; k < K; ++k) s += x[k] * w[k];
          Y[(size_t)m * N + n0 + j] = s + (b ? h2f(b[n0 + j]) : 0.0f);
        }
      }
    }
  }
}
void layer_norm(const float* x, int M, int D, const uint16_t* g, const uint16_t* b, float* y) {
#pragma omp parallel for
  for (int m = 0; m < M; ++m) {
    const float* r = x + (size_t)m * D;
    double mu = 0, var = 0;
    for (int i = 0; i < D; ++i) mu += r[i];
    mu /= D;
    for (int i = 0; i < D; ++i) var += (r[i] - mu) * (r[i] - mu);
    var /= D;
    const float inv = (float)(1.0 / std::sqrt(var + 1e-5));
    for (int i = 0; i < D; ++i) y[(size_t)m * D + i] = (float)(r[i] - mu) * inv * h2f(g[i]) + h2f(b[i]);
  }
}
inline float gelu(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f)); }

class CpuBackend : public Backend {
 public:
  explicit CpuBackend(const Model& m) : m_(m), f_(*m.file), hp_(m.hp) {}

  const uint16_t* W(const std::string& n) const { return (const uint16_t*)f_.get(n).data; }

  void encode(const float* mel) override {
    const int D = hp_.d, C = hp_.n_mels, T = kWindowFrames, A = hp_.n_audio_ctx;
    // conv1 (k 3, pad 1) + GELU: im2col [T][C * 3]
    std::vector<float> col((size_t)T * C * 3), h1((size_t)T * D);
#pragma omp parallel for
    for (int t = 0; t < T; ++t)
      for (int c = 0; c < C; ++c)
        for (int k = 0; k < 3; ++k) {
          const int s = t + k - 1;
          col[((size_t)t * C + c) * 3 + k] = s >= 0 && s < T ? mel[(size_t)c * T + s] : 0.0f;
        }
    linear(col.data(), T, C * 3, W("encoder/conv1/weight"), W("encoder/conv1/bias"), D, h1.data());
    for (float& v : h1) v = gelu(v);
    // conv2 (k 3, stride 2, pad 1) + GELU
    std::vector<float> col2((size_t)A * D * 3);
#pragma omp parallel for
    for (int t = 0; t < A; ++t)
      for (int c = 0; c < D; ++c)
        for (int k = 0; k < 3; ++k) {
          const int s = 2 * t + k - 1;
          col2[((size_t)t * D + c) * 3 + k] = s >= 0 && s < T ? h1[(size_t)s * D + c] : 0.0f;
        }
    enc_.assign((size_t)A * D, 0.0f);
    linear(col2.data(), A, D * 3, W("encoder/conv2/weight"), W("encoder/conv2/bias"), D, enc_.data());
    const uint16_t* pos = W("encoder/position_encodings/encodings");
    for (size_t i = 0; i < enc_.size(); ++i) enc_[i] = gelu(enc_[i]) + h2f(pos[i]);
    // (checks: WH_DUMP_DIR=<folder> writes the residual stream after each encoder layer, L<n>.bin, L-1: after the
    //  convolutions)
    auto dump = [&](int at) {
      if (const char* d = getenv("WH_DUMP_DIR"))
        std::ofstream(std::string(d) + "/L" + std::to_string(at) + ".bin", std::ios::binary).write((const char*)enc_.data(), enc_.size() * 4);
    };
    dump(-1);
    std::vector<float> h((size_t)A * D), qkv((size_t)A * 3 * D), att((size_t)A * D), o((size_t)A * D),
        f((size_t)A * 4 * D);
    for (int l = 0; l < hp_.enc_layers; ++l) {
      const std::string p = "encoder/layer_" + std::to_string(l) + "/";
      layer_norm(enc_.data(), A, D, W(p + "self_attention/layer_norm/gamma"), W(p + "self_attention/layer_norm/beta"), h.data());
      linear(h.data(), A, D, W(p + "self_attention/linear_0/weight"), W(p + "self_attention/linear_0/bias"), 3 * D, qkv.data());
      attention(qkv.data(), 3 * D, 0, qkv.data(), 3 * D, D, 2 * D, A, A, false, att.data());
      linear(att.data(), A, D, W(p + "self_attention/linear_1/weight"), W(p + "self_attention/linear_1/bias"), D, o.data());
      for (size_t i = 0; i < enc_.size(); ++i) enc_[i] += o[i];
      layer_norm(enc_.data(), A, D, W(p + "ffn/layer_norm/gamma"), W(p + "ffn/layer_norm/beta"), h.data());
      const int F = f_.get(p + "ffn/linear_0/weight").shape[0];
      f.resize((size_t)A * F);
      linear(h.data(), A, D, W(p + "ffn/linear_0/weight"), W(p + "ffn/linear_0/bias"), F, f.data());
      for (float& v : f) v = gelu(v);
      linear(f.data(), A, F, W(p + "ffn/linear_1/weight"), W(p + "ffn/linear_1/bias"), D, o.data());
      for (size_t i = 0; i < enc_.size(); ++i) enc_[i] += o[i];
      dump(l + 1);
    }
    layer_norm(enc_.data(), A, D, W("encoder/layer_norm/gamma"), W("encoder/layer_norm/beta"), enc_.data());
    // the decoder layers' cross-attention keys / values: linear_1, fused [k | v]
    ckv_.assign(hp_.dec_layers, {});
    for (int l = 0; l < hp_.dec_layers; ++l) {
      const std::string p = "decoder/layer_" + std::to_string(l) + "/attention/";
      ckv_[l].resize((size_t)A * 2 * D);
      linear(enc_.data(), A, D, W(p + "linear_1/weight"), W(p + "linear_1/bias"), 2 * D, ckv_[l].data());
    }
  }
  const std::vector<float>& encoder_output() const { return enc_; }

  void reset(int n) override {
    seqs_.assign(n, {});
    for (Seq& s : seqs_) { s.k.assign(hp_.dec_layers, {}); s.v.assign(hp_.dec_layers, {}); s.len = 0; }
  }
  void copy(int dst, int src) override { seqs_[dst] = seqs_[src]; }
  // (the reference: the part on the whole window's context; the window's keys / values kept aside meanwhile)
  bool encode_part(const float* mel, int frames, bool short_context) override {
    (void)frames;
    (void)short_context;
    if (!in_part_) { keep_ = ckv_; in_part_ = true; }
    encode(mel);
    return true;
  }
  void part_done() override {
    if (in_part_) { ckv_ = std::move(keep_); in_part_ = false; }
  }
  void append(int s, const int* tok, int n, int rows, float* logits) override {
    run(s, tok, n, rows, logits, nullptr, nullptr);
  }
  void cross_attention(const int* tok, int n, const std::vector<std::pair<int, int>>& heads, std::vector<float>& scores,
                       std::vector<float>& p) override {
    reset(1);
    std::vector<float> lg((size_t)n * hp_.n_vocab);
    scores.assign(heads.size() * (size_t)n * hp_.n_audio_ctx, 0.0f);
    run(0, tok, n, n, lg.data(), &heads, &scores);
    p.assign(n > 0 ? n - 1 : 0, 0.0f);
    std::vector<float> lp;
    for (int i = 0; i + 1 < n; ++i) {
      log_softmax(&lg[(size_t)i * hp_.n_vocab], hp_.n_vocab, lp);
      p[i] = std::exp(lp[tok[i + 1]]);
    }
  }

 private:
  struct Seq {
    std::vector<std::vector<float>> k, v;  // per layer [len][D]
    int len = 0;
  };
  const Model& m_;
  const Ct2File& f_;
  Hparams hp_;
  std::vector<float> enc_;
  std::vector<std::vector<float>> ckv_, keep_;
  std::vector<Seq> seqs_;
  bool in_part_ = false;

  // out[i] = softmax(q_i . k_j / sqrt(64)) . v_j over the heads; q rows at q + i * ldq + qoff, k / v rows likewise;
  // causal: row i sees keys j <= kv_base + i (decoder self-attention), else all nk keys
  void attention(const float* q, int ldq, int qoff, const float* kv, int ldkv, int koff, int voff, int nq, int nk,
                 bool causal, float* out, int kv_base = 0, const float* kv_v = nullptr, int ldv = 0,
                 std::vector<float>* save = nullptr, int save_head = -1, int save_n = 0, int save_row0 = 0) {
    const int D = hp_.d, H = hp_.n_head, hd = D / H;
    const float scale = 1.0f / std::sqrt((float)hd);
    const float* vv = kv_v ? kv_v : kv;
    const int ldvv = kv_v ? ldv : ldkv;
#pragma omp parallel
    {
      std::vector<float> sc(nk);
#pragma omp for collapse(2) schedule(dynamic, 8)
      for (int h = 0; h < H; ++h)
        for (int i = 0; i < nq; ++i) {
          const float* qi = q + (size_t)i * ldq + qoff + h * hd;
          const int lim = causal ? std::min(nk, kv_base + i + 1) : nk;
          float mx = kMasked;
          for (int j = 0; j < lim; ++j) {
            const float* kj = kv + (size_t)j * ldkv + koff + h * hd;
            float s = 0;
            for (int d = 0; d < hd; ++d) s += qi[d] * kj[d];
            sc[j] = s * scale;
            mx = std::max(mx, sc[j]);
          }
          if (save && h == save_head)
            for (int j = 0; j < lim; ++j) (*save)[((size_t)save_row0 + i) * save_n + j] = sc[j];
          double z = 0;
          for (int j = 0; j < lim; ++j) { sc[j] = std::exp(sc[j] - mx); z += sc[j]; }
          float* o = out + (size_t)i * D + h * hd;
          for (int d = 0; d < hd; ++d) o[d] = 0;
          for (int j = 0; j < lim; ++j) {
            const float w = (float)(sc[j] / z);
            const float* vj = vv + (size_t)j * ldvv + voff + h * hd;
            for (int d = 0; d < hd; ++d) o[d] += w * vj[d];
          }
        }
    }
  }

  void run(int s, const int* tok, int n, int rows, float* logits, const std::vector<std::pair<int, int>>* heads,
           std::vector<float>* scores) {
    const int D = hp_.d, A = hp_.n_audio_ctx, V = hp_.n_vocab;
    Seq& q = seqs_[s];
    const int p0 = q.len;
    if (p0 + n > hp_.n_text_ctx) throw std::runtime_error("whisper: more than " + std::to_string(hp_.n_text_ctx) + " text positions");
    const uint16_t* emb = W("decoder/embeddings/weight");
    const uint16_t* pos = W("decoder/position_encodings/encodings");
    std::vector<float> x((size_t)n * D), h((size_t)n * D), qkv((size_t)n * 3 * D), att((size_t)n * D), o((size_t)n * D);
    for (int i = 0; i < n; ++i)
      for (int d = 0; d < D; ++d)
        x[(size_t)i * D + d] = h2f(emb[(size_t)tok[i] * D + d]) + h2f(pos[(size_t)(p0 + i) * D + d]);
    for (int l = 0; l < hp_.dec_layers; ++l) {
      const std::string p = "decoder/layer_" + std::to_string(l) + "/";
      layer_norm(x.data(), n, D, W(p + "self_attention/layer_norm/gamma"), W(p + "self_attention/layer_norm/beta"), h.data());
      linear(h.data(), n, D, W(p + "self_attention/linear_0/weight"), W(p + "self_attention/linear_0/bias"), 3 * D, qkv.data());
      std::vector<float>& K = q.k[l];
      std::vector<float>& Vv = q.v[l];
      K.resize((size_t)(p0 + n) * D);
      Vv.resize((size_t)(p0 + n) * D);
      for (int i = 0; i < n; ++i) {
        memcpy(&K[(size_t)(p0 + i) * D], &qkv[(size_t)i * 3 * D + D], D * 4);
        memcpy(&Vv[(size_t)(p0 + i) * D], &qkv[(size_t)i * 3 * D + 2 * D], D * 4);
      }
      attention(qkv.data(), 3 * D, 0, K.data(), D, 0, 0, n, p0 + n, true, att.data(), p0, Vv.data(), D);
      linear(att.data(), n, D, W(p + "self_attention/linear_1/weight"), W(p + "self_attention/linear_1/bias"), D, o.data());
      for (size_t i = 0; i < x.size(); ++i) x[i] += o[i];
      // cross-attention on the window's encoder output
      layer_norm(x.data(), n, D, W(p + "attention/layer_norm/gamma"), W(p + "attention/layer_norm/beta"), h.data());
      std::vector<float> cq((size_t)n * D);
      linear(h.data(), n, D, W(p + "attention/linear_0/weight"), W(p + "attention/linear_0/bias"), D, cq.data());
      int save_head = -1;
      size_t save_slot = 0;
      std::vector<float> one;
      if (heads)
        for (size_t hi = 0; hi < heads->size(); ++hi)
          if ((*heads)[hi].first == l) {  // (one head saved per call: the layer may hold several)
            one.assign((size_t)n * A, 0.0f);
            attention(cq.data(), D, 0, ckv_[l].data(), 2 * D, 0, D, n, A, false, att.data(), 0, nullptr, 0, &one,
                      (*heads)[hi].second, A, 0);
            save_head = (*heads)[hi].second;
            save_slot = hi;
            memcpy(&(*scores)[save_slot * (size_t)n * A], one.data(), one.size() * 4);
          }
      (void)save_head;
      attention(cq.data(), D, 0, ckv_[l].data(), 2 * D, 0, D, n, A, false, att.data());
      linear(att.data(), n, D, W(p + "attention/linear_2/weight"), W(p + "attention/linear_2/bias"), D, o.data());
      for (size_t i = 0; i < x.size(); ++i) x[i] += o[i];
      layer_norm(x.data(), n, D, W(p + "ffn/layer_norm/gamma"), W(p + "ffn/layer_norm/beta"), h.data());
      const int F = f_.get(p + "ffn/linear_0/weight").shape[0];
      std::vector<float> f((size_t)n * F);
      linear(h.data(), n, D, W(p + "ffn/linear_0/weight"), W(p + "ffn/linear_0/bias"), F, f.data());
      for (float& v : f) v = gelu(v);
      linear(f.data(), n, F, W(p + "ffn/linear_1/weight"), W(p + "ffn/linear_1/bias"), D, o.data());
      for (size_t i = 0; i < x.size(); ++i) x[i] += o[i];
    }
    q.len = p0 + n;
    layer_norm(x.data() + (size_t)(n - rows) * D, rows, D, W("decoder/layer_norm/gamma"), W("decoder/layer_norm/beta"), h.data());
    linear(h.data(), rows, D, emb, nullptr, V, logits);
  }
};

std::string json_escape(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
    else if (c == '\n') o += "\\n";
    else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += (char)c;
  }
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: ref_whisper <model folder> <audio> [--lang auto|id|...] [--beam N] [--words] [--from s] [--to s]\n"
                    "                   [--prompt text] [--json out.json] [--dump-encoder f] [--dump-logits f]\n");
    return 1;
  }
  try {
    Options opt;
    opt.beam = 1;
    std::string json_out, dump_enc, dump_lg;
    for (int i = 3; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(a + ": missing value");
        return argv[++i];
      };
      if (a == "--lang") opt.language = next();
      else if (a == "--beam") opt.beam = std::max(1, atoi(next().c_str()));
      else if (a == "--words") opt.word_timestamps = true;
      else if (a == "--from") opt.from_s = atof(next().c_str());
      else if (a == "--to") opt.to_s = atof(next().c_str());
      else if (a == "--prompt") opt.initial_prompt = next();
      else if (a == "--json") json_out = next();
      else if (a == "--dump-encoder") dump_enc = next();
      else if (a == "--dump-logits") dump_lg = next();
      else if (a == "--no-fallback") opt.temperatures = {0.0f};
      else throw std::runtime_error("unknown option " + a);
    }
    const auto t0 = std::chrono::steady_clock::now();
    Model m(argv[1]);
    fprintf(stderr, "whisper: %d mels, d %d, %d heads, encoder %d layers, decoder %d, vocab %d, %zu languages\n", m.hp.n_mels,
            m.hp.d, m.hp.n_head, m.hp.enc_layers, m.hp.dec_layers, m.hp.n_vocab, m.sp.languages.size());
    std::ifstream fin(std::filesystem::path(argv[2]), std::ios::binary);
    if (!fin) throw std::runtime_error(std::string("cannot read ") + argv[2]);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    const std::vector<float> audio = audio::decode_16k(bytes, 0);
    fprintf(stderr, "audio: %.1f s\n", audio.size() / 16000.0);
    CpuBackend be(m);
    if (!dump_enc.empty() || !dump_lg.empty()) {  // the first window alone, greedy, no fallback
      int nf = 0, content = 0;
      const std::vector<float> mel = log_mel(audio, m.hp.n_mels, nf, content);
      std::vector<float> win((size_t)m.hp.n_mels * kWindowFrames);
      for (int c = 0; c < m.hp.n_mels; ++c)
        for (int t = 0; t < kWindowFrames; ++t) win[(size_t)c * kWindowFrames + t] = mel[(size_t)c * nf + std::min(t, nf - 1)];
      be.encode(win.data());
      if (!dump_enc.empty()) {
        std::ofstream(dump_enc, std::ios::binary).write((const char*)be.encoder_output().data(), be.encoder_output().size() * 4);
      }
      if (!dump_lg.empty()) {
        Transcriber tr(m, be);
        const std::string lang = opt.language == "auto" ? tr.detect_language() : opt.language;
        std::vector<int> toks = {m.sp.sot, m.sp.lang_token(lang), m.sp.transcribe};
        const int V = m.hp.n_vocab;
        std::vector<float> lg(V);
        be.reset(1);
        be.append(0, toks.data(), (int)toks.size(), 1, lg.data());
        std::ofstream o(dump_lg, std::ios::binary);
        for (int step = 0; step < 64; ++step) {
          o.write((const char*)lg.data(), (size_t)V * 4);
          int t = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
          if (t == m.sp.eot) break;
          be.append(0, &t, 1, 1, lg.data());
        }
      }
      fprintf(stderr, "dumped in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      return 0;
    }
    Transcriber tr(m, be);
    const Result r = tr.run(audio, opt, [&](double done, double total, const std::vector<Segment>& segs) {
      static size_t shown = 0;
      for (; shown < segs.size(); ++shown)
        printf("[%7.2f -> %7.2f] %s %s\n", segs[shown].start, segs[shown].end, segs[shown].lang.c_str(), segs[shown].text.c_str());
      fflush(stdout);
      fprintf(stderr, "  %.0f / %.0f s\n", done, total);
      return true;
    });
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "transcribed %.1f s of audio in %.1f s (%.2fx realtime)\n", r.duration, secs, r.duration / secs);
    if (!json_out.empty()) {
      std::ofstream o(json_out, std::ios::binary);
      o << "{\"duration\": " << r.duration << ", \"segments\": [";
      for (size_t i = 0; i < r.segments.size(); ++i) {
        const Segment& g = r.segments[i];
        o << (i ? ",\n" : "\n") << " {\"start\": " << g.start << ", \"end\": " << g.end << ", \"lang\": \"" << g.lang
          << "\", \"text\": \"" << json_escape(g.text) << "\", \"avg_logprob\": " << g.avg_logprob
          << ", \"no_speech_prob\": " << g.no_speech_prob << ", \"temperature\": " << g.temperature << ", \"words\": [";
        for (size_t w = 0; w < g.words.size(); ++w)
          o << (w ? ", " : "") << "{\"start\": " << g.words[w].start << ", \"end\": " << g.words[w].end << ", \"word\": \""
            << json_escape(g.words[w].text) << "\", \"p\": " << g.words[w].p << "}";
        o << "]}";
      }
      o << "\n]}\n";
    }
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}

// arcinferzero: Qwen3-ASR on the Arc iGPU from the command line (zero/asr_qwen3.h): the audio encoder against the f32
// reference (tools/ref_qwen3a) and transcription with timings.
// usage: asr_testz model.gguf mmproj.gguf audio [options]
//   --ref F        compare the encoder output with ref_qwen3a's F (relative error per token, cosine)
//   --enc-out F    write the encoder output (n_tok x 2048 f32)
//   --start S      from S seconds on                --seconds S   only S seconds of it
//   --lang NAME    force the language (Indonesian, en, ...)
//   --context T    the system turn                  --no-text     the encoder only
//   --parts S      parts cut at pauses once S seconds long (default 0: one part up to 20 minutes, as Qwen's code)
//   --ctx N        context (default: what the clip needs)
#include "asr_qwen3.h"
#include "../src/audio.h"
#include <cstdio>

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: asr_testz model.gguf mmproj.gguf audio [--ref F] [--enc-out F] [--seconds S] [--lang L] [--context T] [--no-text]\n"); return 1; }
  std::string ref, enc_out, lang, context;
  double seconds = 0, start = 0, parts = 0;
  bool text = true;
  int ctx = 0;
  for (int i = 4; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(1); } return argv[++i]; };
    if (a == "--ref") ref = next();
    else if (a == "--enc-out") enc_out = next();
    else if (a == "--seconds") seconds = atof(next().c_str());
    else if (a == "--start") start = atof(next().c_str());
    else if (a == "--parts") parts = atof(next().c_str());
    else if (a == "--lang") lang = next();
    else if (a == "--context") context = next();
    else if (a == "--ctx") ctx = atoi(next().c_str());
    else if (a == "--no-text") text = false;
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
  }
  try {
    std::vector<float> s = audio::decode_16k_file(argv[3], 0);
    if (start > 0) s.erase(s.begin(), s.begin() + std::min(s.size(), (size_t)(start * 16000)));
    if (seconds > 0) s.resize(std::min(s.size(), (size_t)(seconds * 16000)));
    printf("audio: %.1f s\n", s.size() / 16000.0);
    if (getenv("ASR_ENERGY")) {  // (loudness per second: RMS and the loudest 100 ms, dBFS)
      for (size_t a = 0; a < s.size(); a += 16000) {
        double sum = 0, best = 0;
        for (size_t b = a; b < std::min(s.size(), a + 16000); b += 1600) {
          double e = 0;
          for (size_t i = b; i < std::min(s.size(), b + 1600); ++i) e += (double)s[i] * s[i];
          sum += e;
          best = std::max(best, e / 1600);
        }
        printf("%4zu s  rms %6.1f  max100ms %6.1f dB\n", a / 16000, 10 * std::log10(sum / 16000 + 1e-12), 10 * std::log10(best + 1e-12));
      }
    }
    EngineOptions eo;
    eo.ctx = ctx > 0 ? ctx : std::max(1024, AudioQwen3A::tokens(s.size()) + 1024);
    Engine eng(argv[1], eo);
    AudioQwen3A enc(argv[2], eng);
    printf("model loaded in %.1f s, ctx %d\n", eng.load_ms / 1000, eng.ctx);
    int n_tok = 0;
    std::vector<float> e = enc.encode(s, n_tok);  // (the first run builds the kernels)
    eng.profile = getenv("ARC_PROF") != nullptr;
    e = enc.encode(s, n_tok);
    printf("encoder: %d tokens, %.1f ms (%.1fx real time)\n", n_tok, enc.encode_ms, s.size() / 16.0 / enc.encode_ms);
    if (eng.profile) {
      std::vector<std::pair<double, std::string>> rows;
      double total = 0;
      for (auto& [n, ms] : eng.prof_ms) { total += ms; rows.push_back({ms, n}); }
      std::sort(rows.rbegin(), rows.rend());
      for (auto& [ms, n] : rows) printf("  %-14s %9.1f ms  %5.1f%%  (%d launches)\n", n.c_str(), ms, 100 * ms / total, eng.prof_n[n]);
      eng.profile = false;
    }
    if (!enc_out.empty()) {
      FILE* f = fopen(enc_out.c_str(), "wb");
      fwrite(e.data(), 4, e.size(), f);
      fclose(f);
    }
    if (!ref.empty()) {
      FILE* f = fopen(ref.c_str(), "rb");
      if (!f) throw std::runtime_error("cannot open " + ref);
      std::vector<float> r(e.size());
      const size_t got = fread(r.data(), 4, r.size(), f);
      fclose(f);
      if (got != r.size()) throw std::runtime_error("the reference has another token count");
      const int D = enc.out_dim();
      double se = 0, sr = 0, worst = 0, cmin = 1;
      for (int t = 0; t < n_tok; ++t) {
        double a = 0, b = 0, ab = 0, d = 0;
        for (int i = 0; i < D; ++i) {
          const double x = e[(size_t)t * D + i], y = r[(size_t)t * D + i];
          a += x * x; b += y * y; ab += x * y; d += (x - y) * (x - y);
        }
        se += d;
        sr += b;
        worst = std::max(worst, std::sqrt(d / std::max(b, 1e-30)));
        cmin = std::min(cmin, ab / std::sqrt(std::max(a * b, 1e-30)));
      }
      printf("encoder vs f32 reference: relative error %.4f %% (worst token %.4f %%), lowest cosine %.6f\n",
             100 * std::sqrt(se / sr), 100 * worst, cmin);
    }
    if (!text) return 0;
    Tokenizer tok(eng.gguf());
    q3asr::Options o;
    o.language = lang.empty() ? "" : q3asr::language_name(lang);
    if (!lang.empty() && o.language.empty()) throw std::runtime_error("not a Qwen3-ASR language: " + lang);
    o.context = context;
    o.part_seconds = parts;
    const q3asr::Result res = q3asr::transcribe(eng, enc, tok, s, o);
    for (const q3asr::Part& p : res.parts) printf("[%.1f-%.1f s] raw: %s\n", p.start, p.end, p.raw.c_str());
    printf("language: %s\ntext: %s\n", res.language().c_str(), res.text().c_str());
    printf("times: encoder %.0f ms, prompt %.0f ms (%d audio tokens), %d tokens in %.0f ms (%.1f tok/s); total %.1f s for %.1f s of audio\n",
           res.encode_ms, res.prefill_ms, res.audio_tokens, res.tokens, res.decode_ms,
           res.tokens * 1000.0 / std::max(1.0, res.decode_ms), (res.encode_ms + res.prefill_ms + res.decode_ms) / 1000,
           s.size() / 16000.0);
  } catch (const std::exception& ex) {
    fprintf(stderr, "error: %s\n", ex.what());
    return 1;
  }
  return 0;
}

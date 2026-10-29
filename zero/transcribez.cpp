// arc-transcribe: Whisper on the Arc iGPU (zero/whisper_engine.h) or the f32 CPU reference, from the command line.
//
//   transcribez <file | folder | wildcard> ... [options]       (docs/whisper-native.md, "Command line")
//
// The model: --model <folder> (a faster-whisper folder: model.bin, tokenizer.json, config.json), else
// faster-whisper-large-v3 (else large-v3-turbo) found in the Hugging Face cache. Outputs go to %USERPROFILE%\.arcinferzero\transcripts
// (--out), one file per format: <name>.json, .srt, .txt by default.
#include "../src/app_dir.h"
#include "../src/whisper.h"
#include "../src/whisper_formats.h"
#include "whisper_engine.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <regex>

namespace fs = std::filesystem;
using namespace whisper;

namespace {

// files from the arguments: a file, every audio / video file of a folder, or a wildcard (* ?) in the file name
std::vector<fs::path> expand(const std::string& a) {
  std::vector<fs::path> out;
  std::error_code ec;
  static const std::vector<std::string> exts = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".wma", ".ogg", ".opus", ".mp4",
                                                ".mov", ".mkv", ".webm", ".wmv", ".avi", ".m4v", ".3gp"};
  auto media = [&](const fs::path& p) {
    std::string e = p.extension().string();
    for (char& c : e) c = (char)tolower((unsigned char)c);
    return std::find(exts.begin(), exts.end(), e) != exts.end();
  };
  const fs::path p(a);
  if (fs::is_directory(p, ec)) {
    for (const auto& d : fs::directory_iterator(p, ec))
      if (d.is_regular_file(ec) && media(d.path())) out.push_back(d.path());
    std::sort(out.begin(), out.end());
    return out;
  }
  const std::string fn = p.filename().string();
  if (fn.find_first_of("*?") != std::string::npos) {
    std::string re;
    for (char c : fn) {
      if (c == '*') re += ".*";
      else if (c == '?') re += '.';
      else if (std::string("\\^$.|+()[]{}").find(c) != std::string::npos) { re += '\\'; re += c; }
      else re += c;
    }
    const std::regex rx(re, std::regex::icase);
    const fs::path dir = p.has_parent_path() ? p.parent_path() : fs::path(".");
    for (const auto& d : fs::directory_iterator(dir, ec))
      if (d.is_regular_file(ec) && std::regex_match(d.path().filename().string(), rx)) out.push_back(d.path());
    std::sort(out.begin(), out.end());
    return out;
  }
  out.push_back(p);
  return out;
}

std::string mmss(double s) {
  char b[32];
  const int t = (int)std::lround(std::max(0.0, s));
  if (t >= 3600) snprintf(b, sizeof b, "%d:%02d:%02d", t / 3600, t / 60 % 60, t % 60);
  else snprintf(b, sizeof b, "%02d:%02d", t / 60, t % 60);
  return b;
}

void usage() {
  fprintf(stderr,
          "usage: transcribez <file | folder | wildcard> ... [options]\n"
          "  --model <folder>            a faster-whisper model folder (default: large-v3 from the HF cache, else turbo)\n"
          "  --device gpu|npu            the Arc iGPU (default), or the encoder on the NPU and the decoder on the iGPU\n"
          "  --lang auto|id|en|id,en     language: detected per 30-s window (auto), fixed, or a list for speech that\n"
          "                              mixes them (each utterance's language; ~3x slower)\n"
          "  --beam 1|5                  greedy or beam search (default 5)\n"
          "  --words / --no-words        word timestamps (default on)\n"
          "  --terms \"<list>\"            a glossary kept before every window (\"GPU, kernel, deploy\"; ~150 tokens\n"
          "                              at most); --terms-file <file>\n"
          "  --prompt \"<text>\"           the first window's previous text (a sample sentence); --prompt-file <f>\n"
          "  --dual-language P           auto: a window whose likeliest language is below P (default 0.8) is split by\n"
          "                              language (as id,en does for every window); 0: off, also for a list\n"
          "  --translate                 to English instead of transcribing\n"
          "  --previous-text             each window with the text before it as its prompt (openai's default; off: it\n"
          "                              carried errors over and dropped whole windows in the tests)\n"
          "  --from <s> --to <s>         only part of the file\n"
          "  --format <list>             json, fw, captions, verbose_json, srt, vtt, sbv, srv3, ass, ttml, txt, md, csv,\n"
          "                              tsv (default json,srt,txt)\n"
          "  --max-chars N --max-lines N --max-cue-s S   subtitle cues (42, 2, 7); --word-cues: one word per cue\n"
          "  --out <folder | file>       default %%USERPROFILE%%\\.arcinferzero\\transcripts\n"
          "  --progress                  progress and segments as JSON lines on stdout (default: a bar on stderr)\n"
          "  --dump-encoder <f> --dump-logits <f>   the first window's encoder output / greedy logits (checks)\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { usage(); return 1; }
  try {
    Options opt;
    std::string model_dir, device = "gpu", out, dump_enc, dump_lg;
    std::vector<std::string> fmts = {"json", "srt", "txt"};
    std::vector<std::string> inputs;
    CueOptions co;
    bool json_progress = false, words = true;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(a + ": missing value");
        return argv[++i];
      };
      if (a == "--help" || a == "-h") { usage(); return 0; }
      else if (a == "--model") model_dir = next();
      else if (a == "--device") device = next();
      else if (a == "--lang" || a == "--language") opt.language = next();
      else if (a == "--beam") opt.beam = std::clamp(atoi(next().c_str()), 1, 5);
      else if (a == "--words") words = true;
      else if (a == "--no-words") words = false;
      else if (a == "--prompt") opt.initial_prompt = next();
      else if (a == "--prompt-file") opt.initial_prompt = read_text(next());
      else if (a == "--terms") opt.terms = next();
      else if (a == "--terms-file") opt.terms = read_text(next());
      else if (a == "--dual-language") opt.dual_language = std::clamp((float)atof(next().c_str()), 0.0f, 1.0f);
      else if (a == "--translate") opt.task = "translate";
      else if (a == "--from") opt.from_s = atof(next().c_str());
      else if (a == "--to") opt.to_s = atof(next().c_str());
      else if (a == "--format") {
        fmts.clear();
        std::string cur;
        for (char c : next() + ",") {
          if (c == ',') { if (!cur.empty()) fmts.push_back(cur); cur.clear(); }
          else if (!isspace((unsigned char)c)) cur += (char)tolower((unsigned char)c);
        }
        for (const std::string& f : fmts)
          if (std::none_of(formats().begin(), formats().end(), [&](const Format& x) { return f == x.name; }))
            throw std::runtime_error("--format: unknown format '" + f + "'");
      }
      else if (a == "--max-chars") co.max_chars = std::max(8, atoi(next().c_str()));
      else if (a == "--max-lines") co.max_lines = std::max(1, atoi(next().c_str()));
      else if (a == "--max-cue-s") co.max_s = std::max(0.5, atof(next().c_str()));
      else if (a == "--word-cues") co.one_word = true;
      else if (a == "--out") out = next();
      else if (a == "--progress") json_progress = true;
      else if (a == "--dump-encoder") dump_enc = next();
      else if (a == "--dump-logits") dump_lg = next();
      else if (a == "--no-fallback") opt.temperatures = {0.0f};
      else if (a == "--previous-text") opt.condition_on_previous_text = true;
      else if (a == "--no-previous-text") opt.condition_on_previous_text = false;  // (the default; kept for scripts)
      else if (!a.empty() && a[0] == '-') throw std::runtime_error("unknown option " + a);
      else inputs.push_back(a);
    }
    opt.word_timestamps = words;
    if (model_dir.empty()) model_dir = find_model();
    if (model_dir.empty())
      throw std::runtime_error("no Whisper model: download one (hf download Systran/faster-whisper-large-v3) or pass --model");
    std::vector<fs::path> files;
    for (const std::string& a : inputs)
      for (const fs::path& p : expand(a)) files.push_back(p);
    if (files.empty()) throw std::runtime_error("no input files");
    const auto t0 = std::chrono::steady_clock::now();
    Model m(model_dir);
    std::unique_ptr<Backend> be;
    WhisperEngine* gpu = nullptr;
    if (device == "gpu" || device == "npu") {  // (npu: the encoder on the NPU, the decoder on the iGPU)
      auto g = std::make_unique<WhisperEngine>(m, "zero/kernels", 5, device == "npu");
      gpu = g.get();
      be = std::move(g);
      if (gpu->npu())
        fprintf(stderr, "whisper: NPU encoder: %.2f GiB of host memory, graphs %s\n", gpu->npu()->host_bytes / 1073741824.0,
                gpu->npu()->all_cached ? "from the cache" : "compiled");
    } else {
      throw std::runtime_error("--device " + device + ": gpu or npu (the CPU reference is tools\\ref_whisper)");
    }
    fprintf(stderr, "whisper: %s loaded in %.1f s (%s, %.2f GiB GPU)\n", fs::path(model_dir).filename().string().c_str(),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), device.c_str(),
            gpu ? gpu->gpu_bytes / 1073741824.0 : 0.0);
    if (!dump_enc.empty() || !dump_lg.empty()) {  // checks: the first window, as tools/ref_whisper dumps it
      const std::vector<float> audio = audio::decode_16k_file(files[0], 0);
      int nf = 0, content = 0;
      const std::vector<float> mel = log_mel(audio, m.hp.n_mels, nf, content);
      std::vector<float> win((size_t)m.hp.n_mels * kWindowFrames);
      for (int c = 0; c < m.hp.n_mels; ++c)
        for (int t = 0; t < kWindowFrames; ++t) win[(size_t)c * kWindowFrames + t] = mel[(size_t)c * nf + std::min(t, nf - 1)];
      be->encode(win.data());
      if (!dump_enc.empty() && gpu) {
        const std::vector<float> e = gpu->encoder_output();
        std::ofstream(dump_enc, std::ios::binary).write((const char*)e.data(), e.size() * 4);
      }
      if (!dump_lg.empty()) {
        Transcriber tr(m, *be);
        const std::string lang = opt.language == "auto" ? tr.detect_language() : opt.language;
        if (m.sp.lang_token(lang) < 0) throw std::runtime_error("--dump-logits: one language code or auto, not '" + lang + "'");
        std::vector<int> toks = {m.sp.sot, m.sp.lang_token(lang), m.sp.transcribe};
        const int V = m.hp.n_vocab;
        std::vector<float> lg(V);
        be->reset(1);
        be->append(0, toks.data(), (int)toks.size(), 1, lg.data());
        std::ofstream o(dump_lg, std::ios::binary);
        for (int step = 0; step < 64; ++step) {
          o.write((const char*)lg.data(), (size_t)V * 4);
          int t = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
          if (t == m.sp.eot) break;
          be->append(0, &t, 1, 1, lg.data());
        }
      }
      return 0;
    }
    int failed = 0;
    for (const fs::path& f : files) {
      try {
        const auto tf = std::chrono::steady_clock::now();
        const std::vector<float> audio = audio::decode_16k_file(f, 0);
        Transcriber tr(m, *be);
        size_t shown = 0;
        const std::string name = f.filename().string();
        const Result r = tr.run(audio, opt, [&](double done, double total, const std::vector<Segment>& segs) {
          const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - tf).count();
          const double speed = el > 0 ? done / el : 0;
          if (json_progress) {
            for (; shown < segs.size(); ++shown)
              printf("{\"type\": \"segment\", \"file\": %s, \"start\": %.3f, \"end\": %.3f, \"lang\": %s, \"text\": %s}\n",
                     json_str(name).c_str(), segs[shown].start, segs[shown].end, json_str(segs[shown].lang).c_str(),
                     json_str(trim_copy(segs[shown].text)).c_str());
            printf("{\"type\": \"progress\", \"file\": %s, \"done\": %.2f, \"total\": %.2f, \"x_realtime\": %.1f}\n",
                   json_str(name).c_str(), done, total, speed);
            fflush(stdout);
          } else {
            const int W = 24, fill = total > 0 ? (int)std::lround(W * done / total) : 0;
            std::string bar(fill, '#');
            bar += std::string(W - fill, '.');
            const double left = speed > 0 ? (total - done) / speed : 0;
            fprintf(stderr, "\r%s  [%s] %3d %%  %s / %s  %.0fx  ~%s left   ", name.c_str(), bar.c_str(),
                    total > 0 ? (int)(100 * done / total) : 0, mmss(done).c_str(), mmss(total).c_str(), speed, mmss(left).c_str());
          }
          return true;
        });
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - tf).count();
        if (!json_progress) fprintf(stderr, "\n");
        // outputs: --out a folder (or nothing: the transcripts folder) -> <stem><ext>; a file name (one input, one format)
        fs::path dir = out.empty() ? fs::path(app_dir()) / "transcripts" : fs::path(out);
        const bool to_file = !out.empty() && dir.has_extension() && files.size() == 1 && fmts.size() == 1;
        if (!to_file) fs::create_directories(dir);
        // (two inputs of one name, talk.mp4 and talk.wav: named with their extension, talk.mp4.srt, not one over the other)
        auto low = [](std::string s) { for (char& c : s) c = (char)tolower((unsigned char)c); return s; };
        const bool twin = std::count_if(files.begin(), files.end(), [&](const fs::path& g) {
                            return low(g.stem().string()) == low(f.stem().string());
                          }) > 1;
        const std::string base = twin ? f.filename().string() : f.stem().string();
        for (const std::string& fm : fmts) {
          const auto it = std::find_if(formats().begin(), formats().end(), [&](const Format& x) { return fm == x.name; });
          const fs::path dst = to_file ? dir : dir / (base + it->ext);
          std::ofstream o(dst, std::ios::binary);
          o << render(fm, r, co, f.stem().string(), opt.task);
          if (!o) throw std::runtime_error("cannot write " + dst.string());
          if (!json_progress) fprintf(stderr, "  %s\n", dst.string().c_str());
        }
        fprintf(stderr, "%s: %.1f s of audio in %.1f s (%.1fx realtime), %zu segments\n", name.c_str(), r.duration, el,
                r.duration / std::max(el, 1e-6), r.segments.size());
      } catch (const std::exception& e) {
        ++failed;
        fprintf(stderr, "\n%s: error: %s\n", f.string().c_str(), e.what());
      }
    }
    if (gpu && !gpu->prof_ms.empty()) {
      std::vector<std::pair<double, std::string>> v;
      double tot = 0;
      for (auto& [k, ms] : gpu->prof_ms) { v.push_back({ms, k}); tot += ms; }
      std::sort(v.rbegin(), v.rend());
      fprintf(stderr, "GPU time %.0f ms%s:\n", tot,
              gpu->npu() ? (" (NPU encoder " + std::to_string((int)gpu->npu_ms) + " ms)").c_str() : "");
      for (auto& [ms, k] : v)
        fprintf(stderr, "  %-14s %8.0f ms  %4.1f %%  (%lld launches)\n", k.c_str(), ms, 100 * ms / tot, (long long)gpu->prof_n[k]);
    }
    return failed ? 1 : 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}

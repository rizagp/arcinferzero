// Qwen3-ASR transcription on the engine: the audio encoder (zero/audio_qwen3a.h) and the Qwen3 1.7B / 0.6B text model
// (zero/engine.h, arch qwen3vl), as Qwen's own qwen_asr package runs them (qwen_asr/inference: qwen3_asr.py, utils.py):
//  - prompt: <|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>{audio rows}<|audio_end|>
//    <|im_end|>\n<|im_start|>assistant\n, and with a forced language "language {Name}<asr_text>" after it;
//  - greedy decoding (temperature 0) up to max_new tokens or <|im_end|>; the answer is "language {Name}<asr_text>{text}"
//    ("language None" for no speech), its repetitions collapsed as detect_and_fix_repetitions does (a character run
//    of more than 20, a pattern of up to 20 characters repeated 20 times or more);
//  - audio over max_seconds (Qwen: 1200 s) cut near every max_seconds at the quietest 100 ms within +-5 s, each part
//    transcribed alone.
// arcinferzero's additions: decoding stops when the text ends in a pattern repeated 40 times (looping); optional
// parts cut at pauses (Options::part_seconds, off by default).
#pragma once
#include "audio_qwen3a.h"
#include "../src/tokenizer.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace q3asr {

// The 30 languages of Qwen3-ASR: ISO 639-1 code -> the name the model writes and takes.
inline const std::vector<std::pair<std::string, std::string>>& languages() {
  static const std::vector<std::pair<std::string, std::string>> l = {
      {"zh", "Chinese"},    {"en", "English"},  {"yue", "Cantonese"}, {"ar", "Arabic"},     {"de", "German"},
      {"fr", "French"},     {"es", "Spanish"},  {"pt", "Portuguese"}, {"id", "Indonesian"}, {"it", "Italian"},
      {"ko", "Korean"},     {"ru", "Russian"},  {"th", "Thai"},       {"vi", "Vietnamese"}, {"ja", "Japanese"},
      {"tr", "Turkish"},    {"hi", "Hindi"},    {"ms", "Malay"},      {"nl", "Dutch"},      {"sv", "Swedish"},
      {"da", "Danish"},     {"fi", "Finnish"},  {"pl", "Polish"},     {"cs", "Czech"},      {"fil", "Filipino"},
      {"fa", "Persian"},    {"el", "Greek"},    {"ro", "Romanian"},   {"hu", "Hungarian"},  {"mk", "Macedonian"}};
  return l;
}
// a code or a name (any case) -> the model's name; "" when it is not one of them
inline std::string language_name(std::string s) {
  for (char& c : s) c = (char)tolower((unsigned char)c);
  if (s == "tl") s = "fil";
  for (const auto& [code, name] : languages()) {
    std::string n = name;
    for (char& c : n) c = (char)tolower((unsigned char)c);
    if (s == code || s == n) return name;
  }
  return "";
}
inline std::string language_code(const std::string& name) {
  for (const auto& [code, n] : languages())
    if (n == name) return code;
  return "";
}

// ---- detect_and_fix_repetitions (on code points)
inline std::u32string to32(const std::string& s) {
  std::u32string o;
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = (unsigned char)s[i];
    const int n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
    char32_t v = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < n && i + k < s.size(); ++k) v = (v << 6) | ((unsigned char)s[i + k] & 0x3F);
    o.push_back(v);
    i += n;
  }
  return o;
}
inline std::string to8(const std::u32string& s) {
  std::string o;
  for (char32_t c : s) {
    if (c < 0x80) o += (char)c;
    else if (c < 0x800) { o += (char)(0xC0 | (c >> 6)); o += (char)(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { o += (char)(0xE0 | (c >> 12)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
    else { o += (char)(0xF0 | (c >> 18)); o += (char)(0x80 | ((c >> 12) & 0x3F)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
  }
  return o;
}
inline std::u32string fix_pattern_repeats(const std::u32string& s, int thresh, int max_len) {
  const int n = (int)s.size();
  if (n < thresh * 2) return s;
  std::u32string out;
  int i = 0;
  while (i <= n - thresh * 2) {
    for (int k = 1; k <= max_len && i + k * thresh <= n; ++k) {
      bool rep = true;
      for (int r = 1; r < thresh && rep; ++r) rep = s.compare(i + r * k, k, s, i, k) == 0;
      if (!rep) continue;
      int end = i + thresh * k;
      while (end + k <= n && s.compare(end, k, s, i, k) == 0) end += k;
      out += s.substr(i, k);
      out += fix_pattern_repeats(s.substr(end), thresh, max_len);
      return out;
    }
    out += s[i++];
  }
  out += s.substr(i);
  return out;
}
inline std::string fix_repetitions(const std::string& text, int thresh = 20) {
  const std::u32string s = to32(text);
  std::u32string a;
  for (size_t i = 0; i < s.size();) {  // character runs longer than thresh -> one
    size_t j = i;
    while (j < s.size() && s[j] == s[i]) ++j;
    a.append(j - i > (size_t)thresh ? 1 : j - i, s[i]);
    i = j;
  }
  return to8(fix_pattern_repeats(a, thresh, 20));
}

struct Part {
  double start = 0, end = 0;  // seconds in the clip
  std::string language, text, raw;
  int audio_tokens = 0, tokens = 0;
  bool cut = false;  // (decoding stopped for want of room, max_new or the context: the text's end lost)
};
struct Result {
  std::vector<Part> parts;
  bool cancelled = false;  // (Options::cancelled stopped it: the parts so far, the last one cut)
  double encode_ms = 0, prefill_ms = 0, decode_ms = 0;
  int audio_tokens = 0, tokens = 0;
  std::string text() const {
    std::string t;
    for (const Part& p : parts)
      if (!p.text.empty()) t += (t.empty() ? "" : " ") + p.text;
    return t;
  }
  std::string language() const {  // the parts' languages, consecutive repeats once, comma separated
    std::string l, last;
    for (const Part& p : parts)
      if (!p.language.empty() && p.language != last) {
        l += (l.empty() ? "" : ",") + p.language;
        last = p.language;
      }
    return l;
  }
};
struct Options {
  std::string language;  // a model language name ("Indonesian"); "" = the model says
  std::string context;   // the system turn (names, terms)
  int max_new = 4096;
  double max_seconds = 1200;
  // parts cut at pauses once they are this long (0, the default: Qwen's way, one part up to max_seconds); parts
  // without a frame above the pause level are not transcribed. (The user's 2:34 recording, 20 s parts: the pauses
  // after 68 s came too early to cut, an 85 s last part repeated a sentence 4 times and lost three; one part kept
  // everything said and invented 3 words over the last 40 s of background sound)
  double part_seconds = 0;
  std::function<bool()> cancelled;                    // (between tokens: the parts so far are returned)
  std::function<void(const std::string&)> on_text;    // the part's text so far, after each token
  std::function<void(const Part&)> on_part;           // each part when it is done
  // every engine call runs inside it (the server: its turn on the GPU); none: called directly
  std::function<void(const std::function<void()>&)> gpu;
};

// Cut points of a clip: parts of at most max_s seconds, each cut at the quietest 100 ms within +-5 s of its target.
inline std::vector<std::pair<size_t, size_t>> split(const std::vector<float>& s, double max_s) {
  const size_t rate = 16000, max_len = (size_t)(max_s * rate), expand = 5 * rate, win = rate / 10;
  std::vector<std::pair<size_t, size_t>> parts;
  size_t start = 0;
  while (s.size() - start > max_len) {
    const size_t cut = start + max_len, lo = std::max(start, cut - expand), hi = std::min(s.size(), cut + expand);
    size_t best = cut;
    if (hi - lo > win) {
      double sum = 0, best_sum = 1e300;
      for (size_t i = lo; i < lo + win; ++i) sum += std::fabs(s[i]);
      for (size_t w = lo;; ++w) {
        if (sum < best_sum) { best_sum = sum; best = w; }
        if (w + win >= hi) break;
        sum += std::fabs(s[w + win]) - std::fabs(s[w]);
      }
      size_t q = best;  // (the quietest sample inside that window)
      for (size_t i = best; i < best + win; ++i)
        if (std::fabs(s[i]) < std::fabs(s[q])) q = i;
      best = q;
    }
    best = std::clamp(best, start + 1, s.size());
    parts.push_back({start, best});
    start = best;
  }
  parts.push_back({start, s.size()});
  return parts;
}

// Loudness of 100 ms frames (dBFS) and the pause level: 27 dB under the loud speech (the 90th percentile of the
// frames), between -60 and -35 dBFS.
inline std::vector<float> frame_db(const std::vector<float>& s, float& pause_db) {
  std::vector<float> db;
  for (size_t a = 0; a < s.size(); a += 1600) {
    double e = 0;
    const size_t b = std::min(s.size(), a + 1600);
    for (size_t i = a; i < b; ++i) e += (double)s[i] * s[i];
    db.push_back((float)(10 * std::log10(e / std::max<size_t>(1, b - a) + 1e-12)));
  }
  std::vector<float> sorted = db;
  float p90 = -120;
  if (!sorted.empty()) {
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() * 9 / 10, sorted.end());
    p90 = sorted[sorted.size() * 9 / 10];
  }
  pause_db = std::clamp(p90 - 27.0f, -60.0f, -35.0f);
  return db;
}
// Parts of a clip: cut in the middle of each pause (8+ frames under the pause level) once the part since the last cut
// is part_s long, every part at most max_s (split); parts with no frame over the pause level left out.
inline std::vector<std::pair<size_t, size_t>> parts_at_pauses(const std::vector<float>& s, double part_s, double max_s) {
  if (part_s <= 0) return split(s, max_s);
  float th = 0;
  const std::vector<float> db = frame_db(s, th);
  std::vector<std::pair<size_t, size_t>> cuts;
  size_t start = 0;
  for (size_t f = 0; f < db.size();) {
    if (db[f] >= th) { ++f; continue; }
    size_t g = f;
    while (g < db.size() && db[g] < th) ++g;
    const size_t mid = (f + g) / 2 * 1600;
    if (g - f >= 8 && mid > start && (mid - start) >= part_s * 16000 && g < db.size()) {
      cuts.push_back({start, mid});
      start = mid;
    }
    f = g;
  }
  cuts.push_back({start, s.size()});
  std::vector<std::pair<size_t, size_t>> out;
  for (const auto& [a, b] : cuts) {
    bool voiced = false;
    for (size_t f = a / 1600; f < db.size() && f * 1600 < b && !voiced; ++f) voiced = db[f] >= th;
    if (!voiced) continue;
    const std::vector<float> part(s.begin() + a, s.begin() + b);
    for (const auto& [x, y] : split(part, max_s)) out.push_back({a + x, a + y});
  }
  return out;
}
// The generated text ends in a pattern of up to 20 characters repeated 40 times: the model loops (over background
// sound after speech; detect_and_fix_repetitions keeps one copy of 20 or more), decoding can stop. (At 20 repeats it
// cut the user's recording, whose first seconds really are "tes" about 20 times.)
inline bool looping(const std::string& text) {
  const std::u32string s = to32(text);
  const size_t n = s.size(), reps = 40;
  for (size_t k = 1; k <= 20 && k * reps <= n; ++k) {
    bool rep = true;
    for (size_t r = 1; r < reps && rep; ++r) rep = s.compare(n - k * (r + 1), k, s, n - k, k) == 0;
    if (rep) return true;
  }
  return false;
}

// How many parts transcribe_ranges decodes together: the engine's slots (Engine::add_slots), as far as a batched pass
// takes them; 1 without slots.
inline int batch_parts(const Engine& eng) {
  return eng.batch_ok() ? std::max(1, std::min(eng.slot_count(), eng.batch_rows())) : 1;
}

// Transcribes the parts [a, b) of 16 kHz samples (times in the clip), in groups of batch_parts(eng): each part of a
// group encoded, then its system turn, audio and prompt prefilled in a slot of its own (reset first), then greedy steps
// of the group's parts still decoding in one batched pass (Engine::forward_batch: per row the same logits, bit for bit,
// as one part alone; a single part: forward). Each part stops at its own <|im_end|>, loop or room; on_part in the order
// of the parts (each once it and those before it are done); a cancel ends the parts decoding (their text so far) and
// the parts after them. on_text: the first part still decoding. Slot 0 is bound at the end.
inline Result transcribe_ranges(Engine& eng, AudioQwen3A& enc, const Tokenizer& tok, const std::vector<float>& s16k,
                                const std::vector<std::pair<size_t, size_t>>& ranges, const Options& o) {
  using clk = std::chrono::steady_clock;
  auto ms = [](clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); };
  Result r;
  auto gpu = [&](const std::function<void()>& f) { o.gpu ? o.gpu(f) : f(); };
  const int im_end = tok.special.count("<|im_end|>") ? tok.special.at("<|im_end|>") : -1;
  const int eot = tok.special.count("<|endoftext|>") ? tok.special.at("<|endoftext|>") : -1;
  const int E = enc.out_dim(), V = eng.hp.n_vocab, B = batch_parts(eng);
  if (E != eng.hp.n_embd)
    throw std::runtime_error("the audio encoder gives rows of " + std::to_string(E) + ", the text model takes " +
                             std::to_string(eng.hp.n_embd) + " (an mmproj of another Qwen3-ASR size)");
  const std::vector<int> pre = tok.encode("<|im_start|>system\n" + o.context + "<|im_end|>\n<|im_start|>user\n<|audio_start|>", true);
  std::string post = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n";
  if (!o.language.empty()) post += "language " + o.language + "<asr_text>";
  const std::vector<int> post_ids = tok.encode(post, true);
  struct Run {
    Part p;
    int slot = 0, room = 0;
    std::vector<int> out;
    std::vector<float> l;   // (the logits of its next token)
    bool done = false, ended = false;  // (ended: <|im_end|>, the loop stop or a cancel; done without it: no room left)
  };
  // parse_asr_output
  auto finish = [&](Run& u) {
    Part& p = u.p;
    p.cut = !u.ended;
    p.tokens = (int)u.out.size();
    p.raw = tok.decode(u.out);
    const std::string s = fix_repetitions(p.raw);
    if (!o.language.empty()) {
      p.language = o.language;
      p.text = s;
    } else {
      const size_t tag = s.find("<asr_text>");
      if (tag == std::string::npos) {
        p.text = s;
      } else {
        std::string meta = s.substr(0, tag), lower = meta;
        for (char& c : lower) c = (char)tolower((unsigned char)c);
        p.text = s.substr(tag + 10);
        if (lower.find("language none") == std::string::npos) {
          const size_t k = lower.find("language ");
          if (k != std::string::npos) {
            std::string name = meta.substr(k + 9);
            name = name.substr(0, name.find_first_of("\r\n"));
            p.language = language_name(name).empty() ? name : language_name(name);
          }
        }
      }
    }
    while (!p.text.empty() && isspace((unsigned char)p.text.front())) p.text.erase(p.text.begin());
    while (!p.text.empty() && isspace((unsigned char)p.text.back())) p.text.pop_back();
    r.audio_tokens += p.audio_tokens;
    r.tokens += p.tokens;
    if (o.on_part) o.on_part(p);
    r.parts.push_back(std::move(p));
  };
  for (size_t g0 = 0; g0 < ranges.size() && !r.cancelled; g0 += B) {
    std::vector<Run> runs(std::min(ranges.size(), g0 + B) - g0);
    for (size_t k = 0; k < runs.size(); ++k) {  // (each part: encoded, then prefilled in slot k)
      Run& u = runs[k];
      const auto [a, b] = ranges[g0 + k];
      u.slot = (int)k;
      u.p.start = a / 16000.0;
      u.p.end = b / 16000.0;
      const std::vector<float> clip(s16k.begin() + a, s16k.begin() + b);
      auto t0 = clk::now();
      int n_tok = 0;
      std::vector<float> emb;
      gpu([&] { emb = enc.encode(clip, n_tok); });
      r.encode_ms += ms(t0);
      if ((size_t)n_tok + pre.size() + post_ids.size() + 16 > (size_t)eng.ctx)
        throw std::runtime_error("the audio needs " + std::to_string(n_tok + pre.size() + post_ids.size()) +
                                 " tokens of context, the model has " + std::to_string(eng.ctx));
      u.p.audio_tokens = n_tok;
      t0 = clk::now();
      gpu([&] {
        eng.bind(u.slot);
        eng.reset();
        eng.forward(pre.data(), (int)pre.size());
        for (int r0 = 0; r0 < n_tok; r0 += eng.max_chunk) {
          const int n = std::min(eng.max_chunk, n_tok - r0);
          eng.forward_embd(emb.data() + (size_t)r0 * E, n, false, false);
        }
        u.l = eng.forward(post_ids.data(), (int)post_ids.size());
        u.room = std::min(o.max_new, eng.ctx - eng.pos - 1);
      });
      r.prefill_ms += ms(t0);
    }
    const auto t0 = clk::now();
    size_t emitted = 0;
    auto emit = [&] {  // (the parts done, in order)
      while (emitted < runs.size() && runs[emitted].done) finish(runs[emitted++]);
    };
    for (;;) {
      if (o.cancelled && o.cancelled()) {
        for (Run& u : runs)
          if (!u.done) u.done = u.ended = r.cancelled = true;
        break;
      }
      std::vector<int> sl, tk;
      std::vector<Run*> rows;
      bool first = true;
      for (Run& u : runs) {
        if (u.done) continue;
        if ((int)u.out.size() >= u.room) {  // (no room left: its text's end lost)
          u.done = true;
          continue;
        }
        const int t = (int)(std::max_element(u.l.begin(), u.l.end()) - u.l.begin());
        if (t == im_end || t == eot) {
          u.done = u.ended = true;
          continue;
        }
        u.out.push_back(t);
        const std::string so_far = tok.decode(u.out);
        if (o.on_text && first) o.on_text(so_far);
        first = false;
        if (looping(so_far)) {
          u.done = u.ended = true;
          continue;
        }
        sl.push_back(u.slot);
        tk.push_back(t);
        rows.push_back(&u);
      }
      emit();
      if (rows.empty()) break;
      gpu([&] {
        if (rows.size() == 1) {
          eng.bind(sl[0]);
          rows[0]->l = eng.forward(&tk[0], 1);
        } else {
          const std::vector<float>& lv = eng.forward_batch(sl, tk);
          for (size_t b = 0; b < rows.size(); ++b) rows[b]->l.assign(lv.begin() + b * V, lv.begin() + (b + 1) * V);
        }
      });
    }
    r.decode_ms += ms(t0);
    emit();
    gpu([&] { eng.bind(0); });
  }
  return r;
}

// Transcribes 16 kHz samples (parts_at_pauses, then transcribe_ranges).
inline Result transcribe(Engine& eng, AudioQwen3A& enc, const Tokenizer& tok, const std::vector<float>& s16k,
                         const Options& o) {
  return transcribe_ranges(eng, enc, tok, s16k, parts_at_pauses(s16k, o.part_seconds, std::max(30.0, o.max_seconds)), o);
}

// A Qwen3-ASR GGUF in the Hugging Face cache (unsloth's or ggml-org's conversion: the model and an mmproj-*.gguf next
// to it), the largest first, its Q8_0 file first (the one measured: docs/qwen3-asr.md); "" when there is none.
inline std::string find_model() {
  namespace fs = std::filesystem;
  std::vector<fs::path> roots;
  if (const char* h = getenv("HF_HUB_CACHE")) roots.push_back(h);
  if (const char* h = getenv("HF_HOME")) roots.push_back(fs::path(h) / "hub");
  if (const char* u = getenv("USERPROFILE")) roots.push_back(fs::path(u) / ".cache" / "huggingface" / "hub");
  for (const char* size : {"1.7B", "0.6B"}) {
    std::string first;
    for (const fs::path& r : roots) {
      std::error_code ec;
      for (const auto& repo : fs::directory_iterator(r, ec)) {
        const std::string n = repo.path().filename().string();
        if (n.find("Qwen3-ASR-" + std::string(size)) == std::string::npos || n.find("GGUF") == std::string::npos) continue;
        for (const auto& snap : fs::directory_iterator(repo.path() / "snapshots", ec))
          for (const auto& f : fs::directory_iterator(snap.path(), ec)) {
            std::string fn = f.path().filename().string();
            if (f.path().extension() != ".gguf" || fn.rfind("mmproj", 0) == 0) continue;
            for (char& c : fn) c = (char)tolower((unsigned char)c);
            if (fn.find("q8_0") != std::string::npos) return f.path().string();
            if (first.empty()) first = f.path().string();
          }
      }
    }
    if (!first.empty()) return first;
  }
  return "";
}
// The mmproj next to a model file (an mmproj*.gguf whose projector is qwen3a; given the text model's width n_embd,
// one whose audio rows are that wide first); "" when there is none.
inline std::string find_mmproj(const std::string& model, int n_embd = 0) {
  namespace fs = std::filesystem;
  std::error_code ec;
  std::string first;
  for (const auto& f : fs::directory_iterator(fs::path(model).parent_path(), ec)) {
    const std::string fn = f.path().filename().string();
    if (fn.rfind("mmproj", 0) != 0 || f.path().extension() != ".gguf") continue;
    try {
      const GgufFile g(f.path().string());
      if (g.get_str("clip.audio.projector_type", "") != "qwen3a") continue;
      if (n_embd <= 0 || (int)g.get("mm.a.mlp.2.weight").dims[1] == n_embd) return f.path().string();
      if (first.empty()) first = f.path().string();
    } catch (const std::exception&) {
    }
  }
  return first;
}

}  // namespace q3asr

// Whisper (openai large-v3 / large-v3-turbo) for arcinferzero, the parts every backend shares: the model files as
// faster-whisper ships them (CTranslate2 model.bin, f16; tokenizer.json; config.json), the log-mel features, the
// tokenizer, and the transcription itself (30-s windows, the previous text as prompt, language detection per window,
// greedy / beam search / sampling with temperature fallback, the timestamp rules, segments), after openai/whisper's
// transcribe.py and decoding.py. A Backend computes the encoder and the decoder's logits: the f32 CPU reference
// (tools/ref_whisper.cpp) and the Level Zero engine (zero/whisper_engine.h, iGPU or NPU) give it the same calls, so
// their results compare token by token. docs/whisper-native.md has the plan and the checks.
#pragma once
#include "audio.h"
#include "json.h"
#include "tokenizer.h"
#include <windows.h>
#include <compressapi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#pragma comment(lib, "cabinet.lib")

namespace whisper {

// masked logits and "no value yet" use large finite numbers, not infinities: the engine builds with /fp:fast, where
// infinities and isfinite() are not reliable
constexpr float kMasked = -1e30f, kMaskedHalf = -5e29f, kHuge = 1e30f;

// ---------------------------------------------------------------- f16
inline float h2f(uint16_t h) {
  const uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  uint32_t f;
  if (e == 0) {
    if (m == 0) f = s;
    else {  // subnormal
      int k = -1;
      uint32_t mm = m;
      do { ++k; mm <<= 1; } while (!(mm & 0x400));
      f = s | ((uint32_t)(127 - 15 - k) << 23) | ((mm & 0x3ff) << 13);
    }
  } else if (e == 31) {
    f = s | 0x7f800000u | (m << 13);
  } else {
    f = s | ((e + 112) << 23) | (m << 13);
  }
  float r;
  memcpy(&r, &f, 4);
  return r;
}

// ---------------------------------------------------------------- CTranslate2 model.bin (format version 6)
// header: u32 version, string name, u32 revision, u32 count; per variable: string name, u8 rank, u32 dims[rank],
// u8 dtype (0 f32, 1 i8, 2 i16, 3 i32, 4 f16, 5 bf16), u32 bytes, the bytes; then u32 count of aliases, (alias,
// target) strings. Strings: u16 length (with the trailing 0), the bytes. Mapped read-only, nothing copied.
struct Ct2Var {
  std::vector<int> shape;
  int dtype = 0;
  const uint8_t* data = nullptr;
  size_t bytes = 0;
  size_t count() const { size_t n = 1; for (int d : shape) n *= (size_t)d; return n; }
};

class Ct2File {
 public:
  explicit Ct2File(const std::string& path) {
    file_ = CreateFileW(std::filesystem::path(path).wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("whisper: cannot open " + path);
    try {  // (a constructor that throws runs no destructor: the handles and the view closed here)
      parse(path);
    } catch (...) {
      close();
      throw;
    }
  }
  ~Ct2File() { close(); }
  Ct2File(const Ct2File&) = delete;
  Ct2File& operator=(const Ct2File&) = delete;

 private:
  void close() {
    if (base_) UnmapViewOfFile(base_);
    if (map_) CloseHandle(map_);
    if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
    base_ = nullptr;
    map_ = nullptr;
    file_ = INVALID_HANDLE_VALUE;
  }
  void parse(const std::string& path) {
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(file_, &sz)) throw std::runtime_error("whisper: cannot read the size of " + path);
    size_ = (size_t)sz.QuadPart;
    map_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!map_) throw std::runtime_error("whisper: cannot map " + path);
    base_ = (const uint8_t*)MapViewOfFile(map_, FILE_MAP_READ, 0, 0, 0);
    if (!base_) throw std::runtime_error("whisper: MapViewOfFile failed for " + path);
    size_t p = 0;
    auto need = [&](size_t n) { if (p + n > size_) throw std::runtime_error("whisper: model.bin is truncated"); };
    auto u8 = [&]() { need(1); return base_[p++]; };
    auto u16 = [&]() { need(2); uint16_t v; memcpy(&v, base_ + p, 2); p += 2; return v; };
    auto u32 = [&]() { need(4); uint32_t v; memcpy(&v, base_ + p, 4); p += 4; return v; };
    auto str = [&]() {
      const uint16_t n = u16();
      need(n);
      std::string s((const char*)base_ + p, n);
      p += n;
      while (!s.empty() && s.back() == 0) s.pop_back();
      return s;
    };
    version = (int)u32();
    if (version < 5) throw std::runtime_error("whisper: model.bin format version " + std::to_string(version) + " (5+ needed)");
    spec = str();
    revision = (int)u32();
    const uint32_t nv = u32();
    for (uint32_t i = 0; i < nv; ++i) {
      const std::string name = str();
      Ct2Var v;
      const int rank = u8();
      for (int r = 0; r < rank; ++r) v.shape.push_back((int)u32());
      v.dtype = u8();
      v.bytes = u32();
      need(v.bytes);
      v.data = base_ + p;
      p += v.bytes;
      vars_[name] = v;
    }
    if (p + 4 <= size_) {
      const uint32_t na = u32();
      for (uint32_t i = 0; i < na; ++i) {
        const std::string a = str(), t = str();
        alias_[a] = t;
      }
    }
  }

 public:
  bool has(const std::string& name) const { return vars_.count(resolve(name)) != 0; }
  const Ct2Var& get(const std::string& name) const {
    auto it = vars_.find(resolve(name));
    if (it == vars_.end()) throw std::runtime_error("whisper: model.bin has no " + name);
    return it->second;
  }
  int scalar(const std::string& name, int def) const {
    if (!has(name)) return def;
    const Ct2Var& v = get(name);
    switch (v.dtype) {
      case 1: return (int8_t)v.data[0];
      case 2: { int16_t x; memcpy(&x, v.data, 2); return x; }
      case 3: { int32_t x; memcpy(&x, v.data, 4); return x; }
      default: return def;
    }
  }
  // a tensor as f32 (f16 / f32 stored)
  std::vector<float> f32(const std::string& name) const {
    const Ct2Var& v = get(name);
    std::vector<float> out(v.count());
    if (v.dtype == 4) {
      const uint16_t* h = (const uint16_t*)v.data;
      for (size_t i = 0; i < out.size(); ++i) out[i] = h2f(h[i]);
    } else if (v.dtype == 0) {
      memcpy(out.data(), v.data, out.size() * 4);
    } else {
      throw std::runtime_error("whisper: " + name + " has an unsupported dtype " + std::to_string(v.dtype));
    }
    return out;
  }
  int version = 0, revision = 0;
  std::string spec;

 private:
  std::string resolve(const std::string& n) const {
    auto it = alias_.find(n);
    return it == alias_.end() ? n : it->second;
  }
  HANDLE file_ = INVALID_HANDLE_VALUE, map_ = nullptr;
  const uint8_t* base_ = nullptr;
  size_t size_ = 0;
  std::map<std::string, Ct2Var> vars_;
  std::map<std::string, std::string> alias_;
};

// ---------------------------------------------------------------- the model folder
constexpr int kSampleRate = 16000, kHop = 160, kFft = 400, kWindowFrames = 3000, kWindowSamples = 480000;

struct Hparams {
  int n_mels = 128, d = 1280, n_head = 20, enc_layers = 32, dec_layers = 4, n_audio_ctx = 1500, n_text_ctx = 448;
  int n_vocab = 51866;
};

// special tokens and the per-model lists (config.json)
struct Specials {
  int eot = -1, sot = -1, translate = -1, transcribe = -1, sot_lm = -1, sot_prev = -1, no_speech = -1,
      no_timestamps = -1, timestamp_begin = -1;
  std::vector<std::pair<std::string, int>> languages;  // (code, token) in the model's order
  std::vector<int> suppress;                            // config.json suppress_ids (+ the openai special tokens)
  std::vector<int> suppress_begin;                      // config.json suppress_ids_begin (" " and <|endoftext|>)
  std::vector<std::pair<int, int>> alignment_heads;     // (decoder layer, head)
  int lang_token(const std::string& code) const {
    for (auto& [c, t] : languages)
      if (c == code) return t;
    return -1;
  }
  std::string lang_code(int tok) const {
    for (auto& [c, t] : languages)
      if (t == tok) return c;
    return "";
  }
};

inline std::string read_text(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("whisper: cannot read " + p.string());
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

class Model {
 public:
  // dir: a faster-whisper model folder (model.bin, tokenizer.json, config.json), or model.bin itself
  explicit Model(const std::string& path) {
    namespace fs = std::filesystem;
    fs::path dir = fs::is_directory(path) ? fs::path(path) : fs::path(path).parent_path();
    dir_ = dir.string();
    file = std::make_unique<Ct2File>((dir / "model.bin").string());
    if (file->spec != "WhisperSpec") throw std::runtime_error("whisper: model.bin is a " + file->spec + ", not a WhisperSpec");
    const Ct2Var& c1 = file->get("encoder/conv1/weight");
    hp.d = c1.shape[0];
    hp.n_mels = c1.shape[1];
    hp.n_head = file->scalar("encoder/num_heads", 20);
    hp.n_audio_ctx = file->get("encoder/position_encodings/encodings").shape[0];
    hp.n_text_ctx = file->get("decoder/position_encodings/encodings").shape[0];
    hp.n_vocab = file->get("decoder/embeddings/weight").shape[0];
    hp.enc_layers = hp.dec_layers = 0;
    while (file->has("encoder/layer_" + std::to_string(hp.enc_layers) + "/ffn/linear_0/weight")) ++hp.enc_layers;
    while (file->has("decoder/layer_" + std::to_string(hp.dec_layers) + "/ffn/linear_0/weight")) ++hp.dec_layers;
    if (!hp.enc_layers || !hp.dec_layers) throw std::runtime_error("whisper: no encoder / decoder layers in model.bin");
    load_tokenizer(dir / "tokenizer.json");
    load_config(dir / "config.json");
  }
  Hparams hp;
  Specials sp;
  std::unique_ptr<Ct2File> file;
  std::unique_ptr<Tokenizer> tok;
  const std::string& dir() const { return dir_; }

  std::string decode_text(const std::vector<int>& ids) const {
    std::vector<int> t;
    for (int id : ids)
      if (id < sp.eot) t.push_back(id);
    return tok->decode(t);
  }

 private:
  std::string dir_;
  void load_tokenizer(const std::filesystem::path& p) {
    const Json j = Json::parse(read_text(p));
    const Json& vocab = j["model"]["vocab"];
    std::vector<std::string> toks;
    std::vector<int64_t> types;
    for (const auto& [s, id] : vocab.entries()) {
      const size_t i = (size_t)id.as_int();
      if (i >= toks.size()) { toks.resize(i + 1); types.resize(i + 1, 1); }
      toks[i] = s;
    }
    for (const Json& a : j["added_tokens"].items()) {
      const size_t i = (size_t)a["id"].as_int();
      if (i >= toks.size()) { toks.resize(i + 1); types.resize(i + 1, 1); }
      toks[i] = a["content"].str();
      types[i] = 3;
    }
    if ((int)toks.size() < hp.n_vocab) { toks.resize(hp.n_vocab); types.resize(hp.n_vocab, 3); }
    std::vector<std::string> merges;
    for (const Json& m : j["model"]["merges"].items()) {
      if (m.is_string()) merges.push_back(m.str());
      else if (m.items().size() == 2) merges.push_back(m[0].str() + " " + m[1].str());  // (the newer [a, b] form)
    }
    tok = std::make_unique<Tokenizer>(toks, types, merges, true);
    auto id = [&](const char* s) { return tok->token(s); };
    sp.eot = id("<|endoftext|>");
    sp.sot = id("<|startoftranscript|>");
    sp.translate = id("<|translate|>");
    sp.transcribe = id("<|transcribe|>");
    sp.sot_lm = id("<|startoflm|>");
    sp.sot_prev = id("<|startofprev|>");
    sp.no_speech = id("<|nospeech|>");
    if (sp.no_speech < 0) sp.no_speech = id("<|nocaptions|>");
    sp.no_timestamps = id("<|notimestamps|>");
    sp.timestamp_begin = id("<|0.00|>");
    if (sp.eot < 0 || sp.sot < 0 || sp.transcribe < 0 || sp.timestamp_begin < 0 || sp.no_timestamps < 0)
      throw std::runtime_error("whisper: tokenizer.json lacks Whisper's special tokens");
  }
  void load_config(const std::filesystem::path& p) {
    Json c = Json::object();
    if (std::filesystem::exists(p)) c = Json::parse(read_text(p));
    for (const Json& t : c["lang_ids"].items()) {
      const int id = (int)t.as_int();
      std::string s = tok->piece(id);  // <|xx|>
      if (s.size() > 4) sp.languages.push_back({s.substr(2, s.size() - 4), id});
    }
    if (sp.languages.empty())  // (no config: the tokens between <|startoftranscript|> and <|translate|>)
      for (int id = sp.sot + 1; id < sp.translate; ++id) {
        const std::string s = tok->piece(id);
        if (s.size() > 4) sp.languages.push_back({s.substr(2, s.size() - 4), id});
      }
    for (const Json& t : c["suppress_ids"].items()) sp.suppress.push_back((int)t.as_int());
    for (int t : {sp.transcribe, sp.translate, sp.sot, sp.sot_prev, sp.sot_lm, sp.no_speech})
      if (t >= 0) sp.suppress.push_back(t);
    std::sort(sp.suppress.begin(), sp.suppress.end());
    sp.suppress.erase(std::unique(sp.suppress.begin(), sp.suppress.end()), sp.suppress.end());
    for (const Json& t : c["suppress_ids_begin"].items()) sp.suppress_begin.push_back((int)t.as_int());
    if (sp.suppress_begin.empty()) {
      for (int t : tok->encode(" ", false)) sp.suppress_begin.push_back(t);
      sp.suppress_begin.push_back(sp.eot);
    }
    for (const Json& h : c["alignment_heads"].items())
      sp.alignment_heads.push_back({(int)h[0].as_int(), (int)h[1].as_int()});
    if (sp.alignment_heads.empty())  // (openai's default: every head of the second half of the decoder layers)
      for (int l = hp.dec_layers / 2; l < hp.dec_layers; ++l)
        for (int h = 0; h < hp.n_head; ++h) sp.alignment_heads.push_back({l, h});
  }
};

// ---------------------------------------------------------------- log-mel features (openai/whisper audio.py)
// 400-sample periodic Hann window, hop 160, reflect-padded STFT (center), power, 128 Slaney mel filters (librosa,
// 0-8000 Hz), log10(max(x, 1e-10)), clamped to the maximum - 8, (x + 4) / 4. The audio is padded with 30 s of zeros
// (as transcribe.py does), so every window, the last one too, has real frames.
inline std::vector<float> mel_filters(int n_mels) {
  const int nb = kFft / 2 + 1;
  auto hz2mel = [](double f) {
    const double f_sp = 200.0 / 3, min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp, logstep = std::log(6.4) / 27.0;
    return f >= min_log_hz ? min_log_mel + std::log(f / min_log_hz) / logstep : f / f_sp;
  };
  auto mel2hz = [](double m) {
    const double f_sp = 200.0 / 3, min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp, logstep = std::log(6.4) / 27.0;
    return m >= min_log_mel ? min_log_hz * std::exp(logstep * (m - min_log_mel)) : f_sp * m;
  };
  std::vector<double> mf(n_mels + 2);
  const double lo = hz2mel(0.0), hi = hz2mel(8000.0);
  for (int i = 0; i < n_mels + 2; ++i) mf[i] = mel2hz(lo + (hi - lo) * i / (n_mels + 1));
  std::vector<float> w((size_t)n_mels * nb, 0.0f);
  for (int i = 0; i < n_mels; ++i) {
    const double enorm = 2.0 / (mf[i + 2] - mf[i]);
    for (int k = 0; k < nb; ++k) {
      const double f = (double)k * kSampleRate / kFft;
      const double lower = (f - mf[i]) / (mf[i + 1] - mf[i]), upper = (mf[i + 2] - f) / (mf[i + 2] - mf[i + 1]);
      w[(size_t)i * nb + k] = (float)(std::max(0.0, std::min(lower, upper)) * enorm);
    }
  }
  return w;
}

// mel[n_mels][n_frames] of samples + 30 s of zeros; content_frames: the frames of the samples themselves
inline std::vector<float> log_mel(const std::vector<float>& x, int n_mels, int& n_frames, int& content_frames) {
  const int nb = kFft / 2 + 1;
  const int64_t L = (int64_t)x.size() + kWindowSamples;  // (the padded length)
  n_frames = (int)(L / kHop);
  content_frames = (int)((int64_t)x.size() / kHop);
  auto sample = [&](int64_t i) -> float {  // the padded signal with reflect padding of kFft / 2 at both ends
    if (i < 0) i = -i;
    if (i >= L) i = 2 * (L - 1) - i;
    return i < (int64_t)x.size() ? x[(size_t)i] : 0.0f;
  };
  std::vector<float> win(kFft), cs((size_t)nb * kFft), sn((size_t)nb * kFft);
  for (int i = 0; i < kFft; ++i) win[i] = (float)(0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * i / kFft));
  for (int k = 0; k < nb; ++k)
    for (int i = 0; i < kFft; ++i) {
      const double a = 2.0 * 3.14159265358979323846 * (double)((int64_t)k * i % kFft) / kFft;
      cs[(size_t)k * kFft + i] = (float)std::cos(a);
      sn[(size_t)k * kFft + i] = (float)std::sin(a);
    }
  const std::vector<float> filt = mel_filters(n_mels);
  std::vector<float> mel((size_t)n_mels * n_frames);
#pragma omp parallel
  {
    std::vector<float> fr(kFft), pw(nb);
#pragma omp for schedule(static)
    for (int t = 0; t < n_frames; ++t) {
      const int64_t c = (int64_t)t * kHop - kFft / 2;
      for (int i = 0; i < kFft; ++i) fr[i] = win[i] * sample(c + i);
      for (int k = 0; k < nb; ++k) {
        const float* ck = &cs[(size_t)k * kFft];
        const float* sk = &sn[(size_t)k * kFft];
        float re = 0, im = 0;
        for (int i = 0; i < kFft; ++i) { re += ck[i] * fr[i]; im += sk[i] * fr[i]; }
        pw[k] = re * re + im * im;
      }
      for (int m = 0; m < n_mels; ++m) {
        const float* f = &filt[(size_t)m * nb];
        float s = 0;
        for (int k = 0; k < nb; ++k) s += f[k] * pw[k];
        mel[(size_t)m * n_frames + t] = std::log10(std::max(s, 1e-10f));
      }
    }
  }
  float mx = -1e30f;
  for (float v : mel) mx = std::max(mx, v);
  for (float& v : mel) v = (std::max(v, mx - 8.0f) + 4.0f) / 4.0f;
  return mel;
}

// ---------------------------------------------------------------- the backend
class Backend {
 public:
  virtual ~Backend() = default;
  // the encoder on one window: mel [n_mels][3000]; its output stays for the decoder calls that follow
  virtual void encode(const float* mel) = 0;
  // n empty decoder sequences on the current encoder output (beams / samples)
  virtual void reset(int n_seq) = 0;
  // appends n tokens to sequence s; logits [rows][n_vocab] of its last `rows` positions
  virtual void append(int s, const int* tok, int n, int rows, float* logits) = 0;
  // one token for each of k sequences (one pass where the backend can): logits [k][n_vocab]
  virtual void append_batch(const int* seqs, const int* tok, int k, float* logits, int n_vocab) {
    for (int i = 0; i < k; ++i) append(seqs[i], &tok[i], 1, 1, logits + (size_t)i * n_vocab);
  }
  // sequence dst becomes a copy of sequence src (its cache)
  virtual void copy(int dst, int src) = 0;
  // one pass of the tokens tok from an empty sequence 0: the alignment heads' cross-attention scores (q.k scaled,
  // before the softmax) over the window's positions, scores [heads][n][n_audio_ctx], and p[i] = the probability of
  // tok[i + 1] after tok[0..i] (i < n - 1)
  virtual void cross_attention(const int* tok, int n, const std::vector<std::pair<int, int>>& heads,
                               std::vector<float>& scores, std::vector<float>& p) = 0;
};

// ---------------------------------------------------------------- transcription options and results
struct Options {
  std::string language = "auto";  // "auto": detected per window; a code ("id") fixes it; "id,en": detected among them
  bool detect_once = false;       // with auto: detect in the first window only (openai's default)
  std::string task = "transcribe";  // or "translate" (to English)
  int beam = 5;                     // 1: greedy
  int best_of = 5;                  // samples at temperature > 0
  std::vector<float> temperatures = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};
  float compression_ratio_threshold = 2.4f, logprob_threshold = -1.0f, no_speech_threshold = 0.6f;
  bool condition_on_previous_text = true;
  float prompt_reset_on_temperature = 0.5f;
  std::string initial_prompt;  // glossary / sample sentence
  bool timestamps = true;
  float max_initial_timestamp = 1.0f;
  bool suppress_blank = true;
  bool word_timestamps = false;
  double from_s = 0, to_s = 0;  // only part of the audio (to_s 0: to the end)
  uint64_t seed = 0;            // sampling (deterministic per window)
};

struct Word {
  double start = 0, end = 0;
  std::string text;
  float p = 0;
};
struct Segment {
  double start = 0, end = 0;
  std::string text, lang;
  std::vector<int> tokens;
  float avg_logprob = 0, no_speech_prob = 0, compression_ratio = 0, temperature = 0;
  std::vector<Word> words;
};
struct Result {
  double duration = 0;
  std::vector<Segment> segments;
  std::string text() const {
    std::string s;
    for (const Segment& g : segments) s += g.text;
    return s;
  }
};
// progress: audio seconds done, total, the segments finished so far (return false to stop)
using Progress = std::function<bool(double done, double total, const std::vector<Segment>& segs)>;

// gzip-style compression ratio of a text (openai: len(text) / len(zlib.compress(text))), with Windows' MSZIP (deflate)
inline float compression_ratio(const std::string& text) {
  if (text.empty()) return 0.0f;
  COMPRESSOR_HANDLE h = nullptr;
  if (!CreateCompressor(COMPRESS_ALGORITHM_MSZIP, nullptr, &h)) return 1.0f;
  std::vector<uint8_t> out(text.size() * 2 + 1024);
  SIZE_T n = 0;
  const BOOL ok = Compress(h, text.data(), text.size(), out.data(), out.size(), &n);
  CloseCompressor(h);
  if (!ok || n == 0) return 1.0f;
  // (MSZIP frames the deflate stream with a 2-byte "CK" signature per block where zlib has a 2-byte header and a
  //  4-byte checksum: within a few bytes of zlib's size)
  return (float)text.size() / (float)(n + 4);
}

inline void log_softmax(const float* x, int n, std::vector<float>& out) {
  out.resize(n);
  float mx = kMasked;
  for (int i = 0; i < n; ++i) mx = std::max(mx, x[i]);
  double s = 0;
  for (int i = 0; i < n; ++i) s += std::exp((double)x[i] - mx);
  const float lse = mx + (float)std::log(s);
  for (int i = 0; i < n; ++i) out[i] = x[i] - lse;
}

// ---------------------------------------------------------------- the transcription
class Transcriber {
 public:
  Transcriber(const Model& m, Backend& b) : m_(m), b_(b) {}

  Result run(const std::vector<float>& audio16k, const Options& opt, const Progress& progress = nullptr) {
    Result res;
    std::vector<float> x = audio16k;
    const double total = (double)x.size() / kSampleRate;
    res.duration = total;
    int n_frames = 0, content = 0;
    // (WH_TIMES=1: where the wall time goes, on stderr at the end: the log-mel, the windows (encoder and decoding;
    //  the GPU's work is asynchronous, so a window's time is what the host waited for it), the word timestamps)
    using clk = std::chrono::steady_clock;
    static const bool times = getenv("WH_TIMES") != nullptr;
    double t_mel = 0, t_win = 0, t_words = 0;
    auto since = [](clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); };
    struct Report {
      const bool& on; double &a, &b, &c;
      ~Report() { if (on) fprintf(stderr, "whisper times: mel %.0f ms, windows %.0f ms, word timestamps %.0f ms\n", a, b, c); }
    } report{times, t_mel, t_win, t_words};
    auto t0 = clk::now();
    const std::vector<float> mel = log_mel(x, m_.hp.n_mels, n_frames, content);
    t_mel = since(t0);
    int seek = std::max(0, (int)std::lround(opt.from_s * kSampleRate / kHop));
    if (opt.to_s > 0) content = std::min(content, (int)std::lround(opt.to_s * kSampleRate / kHop));
    std::vector<int> all_tokens;
    int prompt_reset_since = 0;
    if (!opt.initial_prompt.empty()) {
      all_tokens = m_.tok->encode(" " + trim(opt.initial_prompt), false);
    }
    const int n_initial = (int)all_tokens.size();
    // languages: "auto" (any), one code (fixed), or a list "id,en" (detected per window among them only)
    std::vector<std::string> allowed;
    {
      std::string cur;
      for (char ch : opt.language + ",") {
        if (ch == ',' || ch == ' ' || ch == '+' || ch == ';') {
          if (!cur.empty() && cur != "auto") {
            if (m_.sp.lang_token(cur) < 0) throw std::runtime_error("whisper: unknown language '" + cur + "'");
            allowed.push_back(cur);
          }
          cur.clear();
        } else {
          cur += (char)tolower((unsigned char)ch);
        }
      }
    }
    const bool fixed = allowed.size() == 1;
    std::string lang = fixed ? allowed[0] : "";
    std::vector<float> win((size_t)m_.hp.n_mels * kWindowFrames);
    int window_no = 0;
    while (seek < content) {
      const int seg_size = std::min(kWindowFrames, content - seek);
      const double time_offset = (double)seek * kHop / kSampleRate, seg_duration = (double)seg_size * kHop / kSampleRate;
      for (int c = 0; c < m_.hp.n_mels; ++c)
        for (int t = 0; t < kWindowFrames; ++t) {
          const int f = seek + t;
          win[(size_t)c * kWindowFrames + t] = f < n_frames ? mel[(size_t)c * n_frames + f] : mel[(size_t)c * n_frames + n_frames - 1];
        }
      const auto tw = clk::now();
      b_.encode(win.data());
      std::string wlang = lang;
      if (!fixed && (!opt.detect_once || lang.empty())) {
        wlang = detect_language(allowed);
        if (opt.detect_once) lang = wlang;
      }
      const std::vector<int> prompt(all_tokens.begin() + std::min((int)all_tokens.size(), std::max(prompt_reset_since, 0)),
                                    all_tokens.end());
      Decoded r = decode_with_fallback(prompt, wlang, opt, window_no);
      t_win += since(tw);
      ++window_no;
      if (opt.no_speech_threshold > 0 && r.no_speech_prob > opt.no_speech_threshold &&
          !(opt.logprob_threshold != 0 && r.avg_logprob > opt.logprob_threshold)) {
        seek += seg_size;  // (silence)
        if (progress && !progress(std::min(total, (double)seek * kHop / kSampleRate), total, res.segments)) break;
        continue;
      }
      const std::vector<int>& tokens = r.tokens;
      const int tb = m_.sp.timestamp_begin;
      std::vector<Segment> segs;
      auto add = [&](double s, double e, const std::vector<int>& toks) {
        Segment g;
        g.start = s;
        g.end = e;
        g.tokens = toks;
        g.text = m_.decode_text(toks);
        g.lang = wlang;
        g.avg_logprob = r.avg_logprob;
        g.no_speech_prob = r.no_speech_prob;
        g.compression_ratio = r.compression_ratio;
        g.temperature = r.temperature;
        if (!trim(g.text).empty()) segs.push_back(g);
      };
      const bool single_ts_ending = tokens.size() >= 2 && tokens[tokens.size() - 2] < tb && tokens.back() >= tb;
      std::vector<size_t> consecutive;
      for (size_t i = 1; i < tokens.size(); ++i)
        if (tokens[i] >= tb && tokens[i - 1] >= tb) consecutive.push_back(i);
      if (!consecutive.empty()) {
        std::vector<size_t> slices = consecutive;
        if (single_ts_ending) slices.push_back(tokens.size());
        size_t last = 0;
        for (size_t cur : slices) {
          std::vector<int> sl(tokens.begin() + last, tokens.begin() + cur);
          if (!sl.empty()) {
            const double s = time_offset + (sl.front() - tb) * 0.02, e = time_offset + (sl.back() - tb) * 0.02;
            add(s, e, sl);
          }
          last = cur;
        }
        if (single_ts_ending) seek += seg_size;
        else seek += (tokens[last - 1] - tb) * 2;  // (to the last complete segment's end; 2 frames per 0.02 s)
      } else {
        double duration = seg_duration;
        for (auto it = tokens.rbegin(); it != tokens.rend(); ++it)
          if (*it >= tb) {
            if (*it != tb) duration = (*it - tb) * 0.02;
            break;
          }
        add(time_offset, time_offset + duration, tokens);
        seek += seg_size;
      }
      if (opt.word_timestamps && !segs.empty()) {
        const auto tt = clk::now();
        add_word_timestamps(segs, time_offset, std::min(seg_duration, 30.0));
        t_words += since(tt);
        // (openai: unless the window ended with a lone timestamp, the next one starts at the last word's end)
        if (!single_ts_ending) {
          double last_end = -1;
          for (const Segment& g : segs)
            if (!g.words.empty()) last_end = g.words.back().end;
          if (last_end > time_offset) seek = (int)std::lround(last_end * kSampleRate / kHop);
        }
      }
      for (Segment& g : segs) {
        for (int t : g.tokens) all_tokens.push_back(t);
        res.segments.push_back(g);
      }
      if (!opt.condition_on_previous_text || r.temperature > opt.prompt_reset_on_temperature)
        prompt_reset_since = (int)all_tokens.size();
      (void)n_initial;
      if (progress && !progress(std::min(total, (double)seek * kHop / kSampleRate), total, res.segments)) break;
    }
    return res;
  }

  // the language of the current window: <|startoftranscript|> alone, the most likely language token (among
  // `allowed` when given: "id,en" for speech that mixes them)
  std::string detect_language(const std::vector<std::string>& allowed = {}) {
    std::vector<float> lg(m_.hp.n_vocab);
    b_.reset(1);
    b_.append(0, &m_.sp.sot, 1, 1, lg.data());
    int best = -1;
    for (auto& [c, t] : m_.sp.languages) {
      if (!allowed.empty() && std::find(allowed.begin(), allowed.end(), c) == allowed.end()) continue;
      if (best < 0 || lg[t] > lg[best]) best = t;
    }
    return m_.sp.lang_code(best);
  }

 private:
  const Model& m_;
  Backend& b_;

  struct Decoded {
    std::vector<int> tokens;  // the sampled tokens, without <|endoftext|>
    float avg_logprob = 0, no_speech_prob = 0, compression_ratio = 0, temperature = 0;
    double sum_logprob = 0;
  };

  static std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && isspace((unsigned char)s[a])) ++a;
    while (b > a && isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
  }

  std::vector<int> sot_sequence(const std::string& lang, const Options& opt) const {
    std::vector<int> s = {m_.sp.sot};
    const int lt = m_.sp.lang_token(lang.empty() ? "en" : lang);
    if (lt >= 0) s.push_back(lt);
    s.push_back(opt.task == "translate" ? m_.sp.translate : m_.sp.transcribe);
    if (!opt.timestamps) s.push_back(m_.sp.no_timestamps);
    return s;
  }

  Decoded decode_with_fallback(const std::vector<int>& prompt, const std::string& lang, const Options& opt, int window_no) {
    Decoded best;
    bool have = false;
    for (size_t ti = 0; ti < opt.temperatures.size(); ++ti) {
      const float t = opt.temperatures[ti];
      Decoded r = decode(prompt, lang, opt, t, window_no * 31 + (int)ti);
      bool fallback = false;
      if (opt.compression_ratio_threshold > 0 && r.compression_ratio > opt.compression_ratio_threshold) fallback = true;
      if (opt.logprob_threshold != 0 && r.avg_logprob < opt.logprob_threshold) fallback = true;
      if (opt.no_speech_threshold > 0 && r.no_speech_prob > opt.no_speech_threshold && opt.logprob_threshold != 0 &&
          r.avg_logprob < opt.logprob_threshold)
        fallback = false;  // (silence: no fallback, the window is skipped)
      if (!have || !fallback) { best = r; have = true; }
      if (!fallback) break;
    }
    return best;
  }

  // the logit filters of one step for one sequence (sampled: the tokens after the sot sequence)
  void filter(std::vector<float>& lg, const std::vector<int>& sampled, const Options& opt) const {
    const int V = m_.hp.n_vocab, tb = m_.sp.timestamp_begin, eot = m_.sp.eot;
    for (int t : m_.sp.suppress)
      if (t >= 0 && t < V) lg[t] = kMasked;
    if (sampled.empty() && opt.suppress_blank)
      for (int t : m_.sp.suppress_begin)
        if (t >= 0 && t < V) lg[t] = kMasked;
    if (!opt.timestamps) return;
    // the timestamp rules (openai ApplyTimestampRules)
    lg[m_.sp.no_timestamps] = kMasked;
    const bool last_ts = !sampled.empty() && sampled.back() >= tb;
    const bool penult_ts = sampled.size() < 2 || sampled[sampled.size() - 2] >= tb;
    if (last_ts) {
      if (penult_ts) { for (int i = tb; i < V; ++i) lg[i] = kMasked; }
      else { for (int i = 0; i < eot; ++i) lg[i] = kMasked; }
    }
    int ts_last = -1;
    for (auto it = sampled.rbegin(); it != sampled.rend(); ++it)
      if (*it >= tb) { ts_last = *it; break; }
    if (ts_last >= 0) {
      const int upto = (last_ts && !penult_ts) ? ts_last : ts_last + 1;
      for (int i = tb; i < std::min(upto, V); ++i) lg[i] = kMasked;
    }
    if (sampled.empty()) {
      for (int i = 0; i < tb; ++i) lg[i] = kMasked;
      if (opt.max_initial_timestamp > 0) {
        const int last_allowed = tb + (int)std::lround(opt.max_initial_timestamp / 0.02f);
        for (int i = last_allowed + 1; i < V; ++i) lg[i] = kMasked;
      }
    }
    // a timestamp when their total probability beats every text token
    std::vector<float> lp;
    log_softmax(lg.data(), V, lp);
    float mx_text = kMasked;
    for (int i = 0; i < tb; ++i) mx_text = std::max(mx_text, lp[i]);
    float mx_ts = kMasked;
    for (int i = tb; i < V; ++i) mx_ts = std::max(mx_ts, lp[i]);
    double s = 0;
    if (mx_ts > kMasked)
      for (int i = tb; i < V; ++i) s += std::exp((double)lp[i] - mx_ts);
    const float ts_lp = mx_ts > kMasked ? mx_ts + (float)std::log(s) : kMasked;
    if (ts_lp > mx_text)
      for (int i = 0; i < tb; ++i) lg[i] = kMasked;
  }

  Decoded decode(const std::vector<int>& prompt, const std::string& lang, const Options& opt, float temperature, int seed) {
    const int V = m_.hp.n_vocab, eot = m_.sp.eot;
    const int sample_len = m_.hp.n_text_ctx / 2;
    std::vector<int> init;
    if (!prompt.empty()) {
      init.push_back(m_.sp.sot_prev);
      const int keep = m_.hp.n_text_ctx / 2 - 1;
      init.insert(init.end(), prompt.end() - std::min((int)prompt.size(), keep), prompt.end());
    }
    const int sot_index = (int)init.size();
    const std::vector<int> sot = sot_sequence(lang, opt);
    init.insert(init.end(), sot.begin(), sot.end());
    const int n_init = (int)init.size();
    const bool sampling = temperature > 0;
    const int n_seq = sampling ? std::max(1, opt.best_of) : std::max(1, opt.beam);
    std::vector<float> first((size_t)(n_init - sot_index) * V);
    b_.reset(n_seq);
    b_.append(0, init.data(), n_init, n_init - sot_index, first.data());
    for (int s = 1; s < n_seq; ++s) b_.copy(s, 0);
    Decoded out;
    out.temperature = temperature;
    {  // no-speech probability at <|startoftranscript|>
      std::vector<float> lp;
      log_softmax(first.data(), V, lp);
      out.no_speech_prob = m_.sp.no_speech >= 0 ? std::exp(lp[m_.sp.no_speech]) : 0.0f;
    }
    const float* last = first.data() + (size_t)(n_init - sot_index - 1) * V;
    if (!sampling && opt.beam > 1) return beam_search(last, n_init, opt, out);
    // greedy (one sequence) or best_of samples (each its own sequence)
    std::mt19937_64 rng(opt.seed * 1000003ull + (uint64_t)seed);
    struct Seq { std::vector<int> toks; double lp = 0; bool done = false; };
    std::vector<Seq> seqs(n_seq);
    std::vector<std::vector<float>> logits(n_seq, std::vector<float>(last, last + V));
    std::vector<float> lg, lp, batch((size_t)n_seq * V);
    for (int step = 0; step < sample_len && n_init + step < m_.hp.n_text_ctx; ++step) {
      std::vector<int> ids, next;
      for (int s = 0; s < n_seq; ++s) {
        if (seqs[s].done) continue;
        lg = logits[s];
        filter(lg, seqs[s].toks, opt);
        log_softmax(lg.data(), V, lp);
        int t = 0;
        if (!sampling) {
          t = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
        } else {
          std::vector<double> p(V);
          for (int i = 0; i < V; ++i) p[i] = (lg[i] > kMaskedHalf) ? std::exp((double)lp[i] / temperature) : 0.0;
          std::discrete_distribution<int> d(p.begin(), p.end());
          t = d(rng);
        }
        seqs[s].lp += lp[t];
        if (t == eot) { seqs[s].done = true; continue; }
        seqs[s].toks.push_back(t);
        ids.push_back(s);
        next.push_back(t);
      }
      if (ids.empty()) break;
      b_.append_batch(ids.data(), next.data(), (int)ids.size(), batch.data(), V);
      for (size_t i = 0; i < ids.size(); ++i)
        logits[ids[i]].assign(batch.begin() + i * V, batch.begin() + (i + 1) * V);
    }
    // the best sequence: the highest mean log-probability (openai MaximumLikelihoodRanker, no length penalty)
    int bi = 0;
    double bs = kMasked;
    for (int s = 0; s < n_seq; ++s) {
      const double sc = seqs[s].lp / (double)std::max<size_t>(1, seqs[s].toks.size());
      if (sc > bs) { bs = sc; bi = s; }
    }
    finish(out, seqs[bi].toks, seqs[bi].lp);
    return out;
  }

  // openai BeamSearchDecoder (patience 1): each step, every live beam's (beam + 1) best continuations; the best
  // `beam` that are not finished go on; finished ones are kept until `beam` of them exist
  Decoded beam_search(const float* first_logits, int n_init, const Options& opt, Decoded& out) {
    const int V = m_.hp.n_vocab, eot = m_.sp.eot, B = opt.beam;
    const int sample_len = m_.hp.n_text_ctx / 2;
    struct Beam { std::vector<int> toks; double lp = 0; };
    std::vector<Beam> beams(1);
    std::vector<std::vector<float>> logits(1, std::vector<float>(first_logits, first_logits + V));
    std::vector<int> slot = {0};  // the backend sequence of each beam
    std::vector<Beam> finished;
    std::vector<float> lg, lp, batch((size_t)B * V);
    for (int step = 0; step < sample_len && n_init + step < m_.hp.n_text_ctx; ++step) {
      struct Cand { double lp; int beam, tok; };
      std::vector<Cand> cands;
      for (size_t b = 0; b < beams.size(); ++b) {
        lg = logits[b];
        filter(lg, beams[b].toks, opt);
        log_softmax(lg.data(), V, lp);
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        const int k = std::min(V, B + 1);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int c) { return lp[a] > lp[c]; });
        for (int i = 0; i < k; ++i)
          if ((lp[idx[i]] > kMaskedHalf)) cands.push_back({beams[b].lp + lp[idx[i]], (int)b, idx[i]});
      }
      std::stable_sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& c) { return a.lp > c.lp; });
      std::vector<Beam> nb;
      std::vector<int> from;
      for (const Cand& c : cands) {
        Beam x = beams[c.beam];
        x.lp = c.lp;
        if (c.tok == eot) {
          if ((int)finished.size() < B) finished.push_back(x);
        } else if ((int)nb.size() < B) {
          x.toks.push_back(c.tok);
          nb.push_back(x);
          from.push_back(c.beam);
        }
        if ((int)nb.size() >= B && (int)finished.size() >= B) break;
      }
      if ((int)finished.size() >= B || nb.empty()) break;
      // the backend sequences follow the beams: copies where a beam continues another one
      std::vector<int> nslot(nb.size());
      std::vector<bool> used(B, false);
      for (size_t i = 0; i < nb.size(); ++i) {  // (a beam keeps its parent's sequence when it is the first child)
        const int ps = slot[from[i]];
        if (!used[ps]) { nslot[i] = ps; used[ps] = true; }
        else nslot[i] = -1;
      }
      for (size_t i = 0; i < nb.size(); ++i)
        if (nslot[i] < 0) {
          int fs = 0;
          while (used[fs]) ++fs;
          used[fs] = true;
          nslot[i] = fs;
          b_.copy(fs, slot[from[i]]);
        }
      // (copies before the parent's own continuation: a parent's sequence is copied from before it changes)
      std::vector<int> ids(nb.size()), toks(nb.size());
      for (size_t i = 0; i < nb.size(); ++i) { ids[i] = nslot[i]; toks[i] = nb[i].toks.back(); }
      b_.append_batch(ids.data(), toks.data(), (int)nb.size(), batch.data(), V);
      logits.assign(nb.size(), {});
      for (size_t i = 0; i < nb.size(); ++i) logits[i].assign(batch.begin() + i * V, batch.begin() + (i + 1) * V);
      beams = nb;
      slot = nslot;
    }
    for (const Beam& b : beams)
      if ((int)finished.size() < B) finished.push_back(b);
    int bi = 0;
    double bs = kMasked;
    for (size_t i = 0; i < finished.size(); ++i) {
      const double sc = finished[i].lp / (double)std::max<size_t>(1, finished[i].toks.size());
      if (sc > bs) { bs = sc; bi = (int)i; }
    }
    finish(out, finished[bi].toks, finished[bi].lp);
    return out;
  }

  void finish(Decoded& out, const std::vector<int>& toks, double lp) const {
    out.tokens = toks;
    out.sum_logprob = lp;
    out.avg_logprob = (float)(lp / (double)(toks.size() + 1));
    out.compression_ratio = compression_ratio(m_.decode_text(toks));
  }

  // ---- word timestamps (openai timing.py): the alignment heads' cross-attention over the window, normalized per
  // head, median filter of width 7, averaged, DTW against the frames; words from the tokens
  void add_word_timestamps(std::vector<Segment>& segs, double time_offset, double window_s) {
    const int eot = m_.sp.eot;
    std::vector<int> text_tokens;
    std::vector<size_t> seg_of;  // (the segment of each text token)
    for (size_t g = 0; g < segs.size(); ++g)
      for (int t : segs[g].tokens)
        if (t < eot) { text_tokens.push_back(t); seg_of.push_back(g); }
    if (text_tokens.empty()) return;
    // tokens: the sot sequence, <|notimestamps|>, the text, <|endoftext|>; the rows used are the inputs from
    // <|notimestamps|> on (row r predicts text token r; the last row predicts <|endoftext|>)
    std::vector<int> toks;
    for (int t : {m_.sp.sot, m_.sp.lang_token(segs[0].lang.empty() ? "en" : segs[0].lang), m_.sp.transcribe})
      if (t >= 0) toks.push_back(t);
    const int r0 = (int)toks.size();
    toks.push_back(m_.sp.no_timestamps);
    toks.insert(toks.end(), text_tokens.begin(), text_tokens.end());
    toks.push_back(eot);
    const int n = (int)toks.size(), H = (int)m_.sp.alignment_heads.size(), A = m_.hp.n_audio_ctx;
    std::vector<float> w, tp;
    b_.cross_attention(toks.data(), n, m_.sp.alignment_heads, w, tp);
    const int frames = std::max(1, std::min(A, (int)std::lround(window_s / 0.02)));
    const int R = n - 1 - r0;  // (= text tokens + 1)
    // per head: softmax over the window's frames, normalized over the tokens (mean / std per frame), median filter
    // of width 7 along the frames; averaged over the heads
    // (rows, frames and rows again in parallel: each element computed alone, in the same order, so the same values)
    std::vector<float> mat((size_t)R * frames, 0.0f);
    for (int h = 0; h < H; ++h) {
      std::vector<float> m((size_t)R * frames);
#pragma omp parallel for schedule(static)
      for (int r = 0; r < R; ++r) {
        const float* s = &w[((size_t)h * n + r0 + r) * A];
        float mx = kMasked;
        for (int f = 0; f < frames; ++f) mx = std::max(mx, s[f]);
        double z = 0;
        for (int f = 0; f < frames; ++f) z += std::exp((double)s[f] - mx);
        for (int f = 0; f < frames; ++f) m[(size_t)r * frames + f] = (float)(std::exp((double)s[f] - mx) / z);
      }
#pragma omp parallel for schedule(static)
      for (int f = 0; f < frames; ++f) {
        double mu = 0, sq = 0;
        for (int i = 0; i < R; ++i) mu += m[(size_t)i * frames + f];
        mu /= R;
        for (int i = 0; i < R; ++i) { const double d = m[(size_t)i * frames + f] - mu; sq += d * d; }
        const double sd = std::sqrt(sq / R);
        for (int i = 0; i < R; ++i) m[(size_t)i * frames + f] = sd > 0 ? (float)((m[(size_t)i * frames + f] - mu) / sd) : 0.0f;
      }
#pragma omp parallel for schedule(static)
      for (int i = 0; i < R; ++i) {  // median filter of width 7 along the frames (reflect at the edges)
        std::vector<float> row(m.begin() + (size_t)i * frames, m.begin() + (size_t)(i + 1) * frames), win(7);
        for (int f = 0; f < frames; ++f) {
          for (int k = -3; k <= 3; ++k) {
            int j = f + k;
            if (j < 0) j = -j;
            if (j >= frames) j = 2 * (frames - 1) - j;
            win[k + 3] = row[std::clamp(j, 0, frames - 1)];
          }
          std::nth_element(win.begin(), win.begin() + 3, win.end());
          mat[(size_t)i * frames + f] += win[3] / H;
        }
      }
    }
    // DTW over -mat: the path's row jumps give each row's (token's) start frame
    std::vector<double> cost((size_t)(R + 1) * (frames + 1), kHuge);
    std::vector<uint8_t> trace((size_t)(R + 1) * (frames + 1), 0);
    cost[0] = 0;
    for (int j = 1; j <= frames; ++j)
      for (int i = 1; i <= R; ++i) {
        const double c0 = cost[(size_t)(i - 1) * (frames + 1) + j - 1], c1 = cost[(size_t)(i - 1) * (frames + 1) + j],
                     c2 = cost[(size_t)i * (frames + 1) + j - 1];
        double c;
        uint8_t t;
        if (c0 < c1 && c0 < c2) { c = c0; t = 0; }
        else if (c1 < c0 && c1 < c2) { c = c1; t = 1; }
        else { c = c2; t = 2; }
        cost[(size_t)i * (frames + 1) + j] = -mat[(size_t)(i - 1) * frames + j - 1] + c;
        trace[(size_t)i * (frames + 1) + j] = t;
      }
    std::vector<int> path_i, path_j;
    {
      int i = R, j = frames;
      for (int k = 0; k <= R; ++k) trace[(size_t)k * (frames + 1)] = 1;
      for (int k = 0; k <= frames; ++k) trace[k] = 2;
      while (i > 0 || j > 0) {
        path_i.push_back(i - 1);
        path_j.push_back(j - 1);
        const uint8_t t = trace[(size_t)i * (frames + 1) + j];
        if (t == 0) { --i; --j; }
        else if (t == 1) --i;
        else --j;
      }
      std::reverse(path_i.begin(), path_i.end());
      std::reverse(path_j.begin(), path_j.end());
    }
    // jumps: where the row changes; tok_start[r]: the time row r (text token r; r = R - 1: <|endoftext|>) starts
    std::vector<double> tok_start(R, window_s);
    {
      int prev = -2, k = 0;
      for (size_t p = 0; p < path_i.size() && k < R; ++p)
        if (path_i[p] != prev) {
          tok_start[k++] = path_j[p] * 0.02;
          prev = path_i[p];
        }
    }
    // words: a token starting with a space (or the first one) begins a new word; punctuation joins the word before
    struct W { std::string text; int t0, t1; size_t seg; };
    std::vector<W> words;
    for (int i = 0; i < (int)text_tokens.size(); ++i) {
      const std::string piece = m_.tok->decode({text_tokens[i]});
      const bool space = !piece.empty() && piece[0] == ' ';
      const bool punct = !piece.empty() && !space && std::all_of(piece.begin(), piece.end(), [](char ch) {
        return ispunct((unsigned char)ch) != 0;
      });
      if (words.empty() || (space && !punct) || seg_of[i] != words.back().seg) words.push_back({piece, i, i + 1, seg_of[i]});
      else { words.back().text += piece; words.back().t1 = i + 1; }
    }
    for (Segment& g : segs) g.words.clear();
    for (const W& wd : words) {
      Word x;
      x.text = trim(wd.text);
      if (x.text.empty()) continue;
      x.start = time_offset + tok_start[std::min(wd.t0, R - 1)];
      x.end = time_offset + tok_start[std::min(wd.t1, R - 1)];
      if (x.end < x.start) x.end = x.start;
      double ps = 0;  // (the mean probability of the word's tokens: text token j comes after input r0 + j)
      for (int j = wd.t0; j < wd.t1; ++j) ps += r0 + j < (int)tp.size() ? tp[r0 + j] : 0.0;
      x.p = (float)(ps / std::max(1, wd.t1 - wd.t0));
      segs[wd.seg].words.push_back(x);
    }
    // a segment spans its words
    for (Segment& g : segs)
      if (!g.words.empty()) {
        g.start = g.words.front().start;
        g.end = std::max(g.words.back().end, g.start);
      }
  }
};

// a faster-whisper model folder (model.bin, tokenizer.json) in the Hugging Face cache: large-v3-turbo first
inline std::string find_model() {
  namespace fs = std::filesystem;
  std::vector<fs::path> roots;
  if (const char* h = getenv("HF_HUB_CACHE")) roots.push_back(h);
  if (const char* h = getenv("HF_HOME")) roots.push_back(fs::path(h) / "hub");
  if (const char* u = getenv("USERPROFILE")) roots.push_back(fs::path(u) / ".cache" / "huggingface" / "hub");
  for (const char* name : {"models--mobiuslabsgmbh--faster-whisper-large-v3-turbo", "models--deepdml--faster-whisper-large-v3-turbo-ct2",
                           "models--Systran--faster-whisper-large-v3"}) {
    for (const fs::path& r : roots) {
      std::error_code ec;
      const fs::path snaps = r / name / "snapshots";
      if (!fs::is_directory(snaps, ec)) continue;
      for (const auto& d : fs::directory_iterator(snaps, ec))
        if (fs::exists(d.path() / "model.bin", ec) && fs::exists(d.path() / "tokenizer.json", ec)) return d.path().string();
    }
  }
  return "";
}

}  // namespace whisper

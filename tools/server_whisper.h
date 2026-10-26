// Whisper in the server (--whisper): POST /v1/audio/transcriptions and /v1/audio/translations as OpenAI's API has
// them (multipart: file, model, language, prompt, response_format json | text | srt | verbose_json | vtt, temperature,
// timestamp_granularities[], stream), answered by zero/whisper_engine.h (the iGPU, or the encoder on the NPU).
//
// Extensions: response_format also fw, captions (Remotion), full_json (arcinferzero's), sbv, srv3, ass, ttml, md,
// csv, tsv; language "id,en" (detected among the listed ones per window) or "auto"; beam_size 1..5 (default 5);
// condition_on_previous_text; stream=true also sends {"type": "transcript.progress"} events.
//
// Safety: the upload stays in memory (the HTTP body limit, 256 MB) and is decoded by Media Foundation from memory, no
// client path or file name is used; the decoding stops past --whisper-max-minutes; one transcription runs at a
// time and at most kMaxWaiting wait (429 past that); a client that closes its connection stops its transcription.
#pragma once
#include "../src/http.h"
#include "../src/json.h"
#include "../src/whisper.h"
#include "../src/whisper_formats.h"
#include "../zero/whisper_engine.h"
#include "api_compat.h"
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>

namespace srvwh {

// the engine's calls, each one inside gpu_turn (the LLM engine's lock: both engines share the GPU command list)
class LockedBackend : public whisper::Backend {
 public:
  using Turn = std::function<void(const std::function<void()>&)>;
  LockedBackend(whisper::Backend& b, Turn t) : b_(b), t_(std::move(t)) {}
  void encode(const float* mel) override { t_([&] { b_.encode(mel); }); }
  void reset(int n) override { t_([&] { b_.reset(n); }); }
  void append(int s, const int* tok, int n, int rows, float* logits) override { t_([&] { b_.append(s, tok, n, rows, logits); }); }
  void append_batch(const int* seqs, const int* tok, int k, float* logits, int n_vocab) override {
    t_([&] { b_.append_batch(seqs, tok, k, logits, n_vocab); });
  }
  void copy(int dst, int src) override { t_([&] { b_.copy(dst, src); }); }
  void cross_attention(const int* tok, int n, const std::vector<std::pair<int, int>>& heads, std::vector<float>& scores,
                       std::vector<float>& p) override {
    t_([&] { b_.cross_attention(tok, n, heads, scores, p); });
  }

 private:
  whisper::Backend& b_;
  Turn t_;
};

class Service {
 public:
  static constexpr int kMaxWaiting = 4;
  std::string id;  // "whisper-large-v3-turbo"
  bool npu = false;
  double max_seconds = 180 * 60;
  LockedBackend::Turn gpu_turn = [](const std::function<void()>& f) { f(); };

  Service(const std::string& dir, bool on_npu, const std::string& kernel_dir) : npu(on_npu) {
    m_ = std::make_unique<whisper::Model>(dir);
    eng_ = std::make_unique<WhisperEngine>(*m_, kernel_dir, 5, on_npu);
    std::string name = std::filesystem::path(dir).filename().string();
    if (name.size() == 40 && name.find_first_not_of("0123456789abcdef") == std::string::npos)  // (a snapshot hash)
      name = std::filesystem::path(dir).parent_path().parent_path().filename().string();
    const size_t r = name.rfind("--");
    if (r != std::string::npos) name = name.substr(r + 2);
    if (name.rfind("faster-", 0) == 0) name = name.substr(7);
    if (name.size() > 4 && name.compare(name.size() - 4, 4, "-ct2") == 0) name.resize(name.size() - 4);
    id = name;
  }
  size_t gpu_bytes() const { return eng_->gpu_bytes; }
  Json props() const {
    Json p = Json::object();
    p.set("model", id);
    p.set("device", npu ? "npu" : "gpu");
    p.set("max_minutes", max_seconds / 60);
    return p;
  }

  void handle(HttpResponse& rs, const std::map<std::string, apic::FormField>& form, bool translate) {
    auto field = [&](const char* k) { auto it = form.find(k); return it == form.end() ? std::string() : it->second.data; };
    auto fail = [&](int st, const std::string& msg, const char* type = "invalid_request_error") {
      Json e = Json::object(), r = Json::object();
      e.set("message", msg);
      e.set("type", type);
      r.set("error", e);
      rs.send(st, "application/json", r.dump());
    };
    const auto fit = form.find("file");
    if (fit == form.end() || fit->second.data.empty()) return fail(400, "file required (multipart/form-data)");
    std::string fmt = field("response_format");
    if (fmt.empty()) fmt = "json";
    static const std::map<std::string, std::string> kTypes = {
        {"json", "application/json"}, {"text", "text/plain; charset=utf-8"}, {"srt", "text/plain; charset=utf-8"},
        {"verbose_json", "application/json"}, {"vtt", "text/plain; charset=utf-8"}, {"fw", "application/json"},
        {"captions", "application/json"}, {"full_json", "application/json"}, {"sbv", "text/plain; charset=utf-8"},
        {"srv3", "text/xml; charset=utf-8"}, {"ass", "text/plain; charset=utf-8"}, {"ttml", "application/ttml+xml; charset=utf-8"},
        {"md", "text/markdown; charset=utf-8"}, {"csv", "text/csv; charset=utf-8"}, {"tsv", "text/tab-separated-values; charset=utf-8"}};
    const auto ty = kTypes.find(fmt);
    if (ty == kTypes.end()) return fail(400, "response_format: json, text, srt, verbose_json or vtt (also fw, captions, full_json, sbv, srv3, ass, ttml, md, csv, tsv)");
    whisper::Options opt;
    opt.task = translate ? "translate" : "transcribe";
    std::string lang = field("language");
    for (char& c : lang) c = (char)tolower((unsigned char)c);
    if (!lang.empty()) {
      if (lang.size() > 64 || lang.find_first_not_of("abcdefghijklmnopqrstuvwxyz,") != std::string::npos)
        return fail(400, "language: a code (id), a list (id,en) or auto");
      for (size_t a = 0; lang != "auto";) {  // (every code one the model knows)
        const size_t b = std::min(lang.find(',', a), lang.size());
        if (m_->sp.lang_token(lang.substr(a, b - a)) < 0) return fail(400, "language: unknown code '" + lang.substr(a, b - a) + "'");
        if (b >= lang.size()) break;
        a = b + 1;
      }
      opt.language = lang;
    }
    opt.initial_prompt = field("prompt");
    if (opt.initial_prompt.size() > 4000) opt.initial_prompt.resize(4000);  // (the model reads its last 223 tokens)
    if (!field("beam_size").empty()) opt.beam = std::clamp(atoi(field("beam_size").c_str()), 1, 5);
    if (field("condition_on_previous_text") == "false") opt.condition_on_previous_text = false;
    if (!field("temperature").empty()) {  // (0: the fallback through higher temperatures, as OpenAI's API does)
      const float t = (float)atof(field("temperature").c_str());
      if (!(t >= 0 && t <= 1)) return fail(400, "temperature: 0..1");
      if (t > 0) {
        std::vector<float> ts = {t};
        for (float x : opt.temperatures) if (x > t + 1e-4f) ts.push_back(x);
        opt.temperatures = ts;
      }
    }
    const std::string gran = field("timestamp_granularities[]") + "," + field("timestamp_granularities");
    // word timing: asked for, or the formats that place words (subtitle cues are split on them)
    opt.word_timestamps = gran.find("word") != std::string::npos ||
                          !(fmt == "json" || fmt == "text" || fmt == "verbose_json");
    const bool stream = field("stream") == "true";

    struct Count { std::atomic<int>& n; int before; Count(std::atomic<int>& x) : n(x), before(n++) {} ~Count() { --n; } } waiting{waiting_};
    if (waiting.before >= kMaxWaiting) return fail(429, "too many transcriptions waiting; try again later", "rate_limit_error");
    std::vector<float> audio;
    try {
      audio = audio::decode_16k(std::vector<uint8_t>(fit->second.data.begin(), fit->second.data.end()), max_seconds);
    } catch (const std::exception& e) {
      return fail(400, std::string(e.what()) + " (the server takes audio and video files Windows can decode)");
    }
    std::unique_lock<std::mutex> one(run_mu_);  // (one transcription at a time: one engine)
    if (rs.client_gone()) return;
    size_t sent = 0;
    whisper::Result r;
    if (stream) rs.sse_begin();
    auto event = [&](const Json& e) { return rs.sse_event(e.dump()); };
    try {
      LockedBackend be(*eng_, gpu_turn);
      whisper::Transcriber tr(*m_, be);
      r = tr.run(audio, opt, [&](double done, double total, const std::vector<whisper::Segment>& segs) {
        if (!stream) return !rs.client_gone();
        for (; sent < segs.size(); ++sent) {
          Json e = Json::object();
          e.set("type", "transcript.text.delta");
          e.set("delta", segs[sent].text);
          if (!event(e)) return false;
        }
        Json e = Json::object();
        e.set("type", "transcript.progress");
        e.set("done", done);
        e.set("total", total);
        return event(e);
      });
    } catch (const std::exception& e) {
      if (!stream) return fail(500, std::string("transcription failed: ") + e.what(), "server_error");
      Json er = Json::object(), ev = Json::object();
      er.set("message", std::string("transcription failed: ") + e.what());
      ev.set("type", "error");
      ev.set("error", er);
      event(ev);
      return rs.sse_end();
    }
    const std::string text = whisper::trim_copy(r.text());
    if (stream) {
      for (; sent < r.segments.size(); ++sent) {
        Json e = Json::object();
        e.set("type", "transcript.text.delta");
        e.set("delta", r.segments[sent].text);
        event(e);
      }
      Json e = Json::object(), u = Json::object();
      e.set("type", "transcript.text.done");
      e.set("text", text);
      u.set("type", "duration");
      u.set("seconds", std::ceil(r.duration));
      e.set("usage", u);
      event(e);
      return rs.sse_end();
    }
    whisper::CueOptions co;
    std::string body;
    if (fmt == "json") {
      Json o = Json::object(), u = Json::object();
      o.set("text", text);
      u.set("type", "duration");
      u.set("seconds", std::ceil(r.duration));
      o.set("usage", u);
      body = o.dump();
    } else if (fmt == "text") {
      body = text + "\n";
    } else {
      body = whisper::render(fmt == "full_json" ? "json" : fmt, r, co, "transcript", opt.task);
    }
    rs.send(200, ty->second, body);
  }

 private:
  std::unique_ptr<whisper::Model> m_;
  std::unique_ptr<WhisperEngine> eng_;
  std::mutex run_mu_;
  std::atomic<int> waiting_{0};
};

}  // namespace srvwh

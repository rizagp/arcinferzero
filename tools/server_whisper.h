// Whisper in the server (--whisper): POST /v1/audio/transcriptions and /v1/audio/translations as OpenAI's API has
// them (multipart: file, model, language, prompt, response_format json | text | srt | verbose_json | vtt, temperature,
// timestamp_granularities[], stream), answered by zero/whisper_engine.h (the iGPU, or the encoder on the NPU).
//
// Extensions: response_format also fw, captions (Remotion), full_json (arcinferzero's), sbv, srv3, ass, ttml, md,
// csv, tsv; language "id,en" (detected among the listed ones per window) or "auto"; beam_size 1..5 (default 5);
// condition_on_previous_text; terms (a glossary kept before every window); dual_language (0..1, default 0.8);
// stream=true also sends {"type": "transcript.progress"} events.
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
#include <chrono>
#include <condition_variable>
#include <deque>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

namespace srvwh {

// a query parameter, percent-decoded ("" when absent)
inline std::string query_param(const std::string& query, const std::string& key) {
  for (size_t a = 0; a <= query.size();) {
    size_t b = query.find('&', a);
    if (b == std::string::npos) b = query.size();
    const std::string kv = query.substr(a, b - a);
    const size_t eq = kv.find('=');
    if (kv.substr(0, eq) == key) {
      std::string v = eq == std::string::npos ? "" : kv.substr(eq + 1), o;
      for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] == '+') o += ' ';
        else if (v[i] == '%' && i + 2 < v.size() && isxdigit((unsigned char)v[i + 1]) && isxdigit((unsigned char)v[i + 2])) {
          o += (char)std::stoi(v.substr(i + 1, 2), nullptr, 16);
          i += 2;
        } else {
          o += v[i];
        }
      }
      return o;
    }
    a = b + 1;
  }
  return "";
}

// a transcript's full JSON (whisper::to_json) read back; segment texts get their leading space again (the text of
// the whole is the segments' texts joined)
inline whisper::Result result_from_json(const Json& j) {
  whisper::Result r;
  r.duration = j["duration"].as_number(0);
  for (const Json& s : j["segments"].items()) {
    whisper::Segment g;
    g.start = s["start"].as_number(0);
    g.end = std::max(g.start, s["end"].as_number(g.start));
    g.text = s["text"].str();
    if (!g.text.empty() && g.text[0] != ' ') g.text = " " + g.text;
    g.lang = s["lang"].str();
    g.speaker = s["speaker"].str();
    g.avg_logprob = (float)s["avg_logprob"].as_number(0);
    g.no_speech_prob = (float)s["no_speech_prob"].as_number(0);
    g.compression_ratio = (float)s["compression_ratio"].as_number(0);
    g.temperature = (float)s["temperature"].as_number(0);
    for (const Json& w : s["words"].items()) {
      whisper::Word x;
      x.start = w["start"].as_number(g.start);
      x.end = std::max(x.start, w["end"].as_number(x.start));
      x.text = whisper::trim_copy(w["word"].str());  // (as the engine writes them: no leading space)
      x.p = (float)w["p"].as_number(0);
      g.words.push_back(x);
    }
    r.segments.push_back(std::move(g));
  }
  return r;
}

// the engine's calls, each one inside gpu_turn (the LLM engine's lock: both engines share the GPU command list)
class LockedBackend : public whisper::Backend {
 public:
  using Turn = std::function<void(const std::function<void()>&)>;
  LockedBackend(whisper::Backend& b, Turn t) : b_(b), t_(std::move(t)) {}
  void encode(const float* mel) override { t_([&] { b_.encode(mel); }); }
  bool encode_part(const float* mel, int frames, bool short_context) override {
    bool whole = false;
    t_([&] { whole = b_.encode_part(mel, frames, short_context); });
    return whole;
  }
  void part_done() override { b_.part_done(); }  // (a flag: no GPU work)
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

  // a request's transcription options from its fields (multipart or JSON, as text); false and the message when one
  // is wrong
  bool read_options(const std::function<std::string(const char*)>& field, bool translate, whisper::Options& opt,
                    std::string& bad) const {
    opt.task = translate ? "translate" : "transcribe";
    std::string lang = field("language");
    for (char& c : lang) c = (char)tolower((unsigned char)c);
    if (!lang.empty()) {
      if (lang.size() > 64 || lang.find_first_not_of("abcdefghijklmnopqrstuvwxyz,") != std::string::npos) {
        bad = "language: a code (id), a list (id,en) or auto";
        return false;
      }
      for (size_t a = 0; lang != "auto";) {  // (every code one the model knows)
        const size_t b = std::min(lang.find(',', a), lang.size());
        if (m_->sp.lang_token(lang.substr(a, b - a)) < 0) {
          bad = "language: unknown code '" + lang.substr(a, b - a) + "'";
          return false;
        }
        if (b >= lang.size()) break;
        a = b + 1;
      }
      opt.language = lang;
    }
    opt.initial_prompt = field("prompt");
    if (opt.initial_prompt.size() > 4000) opt.initial_prompt.resize(4000);  // (the model reads its last 223 tokens)
    opt.terms = field("terms");
    if (opt.terms.size() > 4000) opt.terms.resize(4000);  // (its first 150 tokens are used)
    auto unit = [&](const char* k, float& out) {  // (a number 0..1, the whole field)
      const std::string v = field(k);
      char* e = nullptr;
      const float x = strtof(v.c_str(), &e);
      if (e == v.c_str() || *e || !(x >= 0 && x <= 1)) return false;
      out = x;
      return true;
    };
    if (!field("dual_language").empty() && !unit("dual_language", opt.dual_language)) {
      bad = "dual_language: 0..1";
      return false;
    }
    if (!field("beam_size").empty()) opt.beam = std::clamp(atoi(field("beam_size").c_str()), 1, 5);
    if (field("condition_on_previous_text") == "true") opt.condition_on_previous_text = true;  // (default off)
    if (!field("temperature").empty()) {  // (0: the fallback through higher temperatures, as OpenAI's API does)
      float t = 0;
      if (!unit("temperature", t)) {
        bad = "temperature: 0..1";
        return false;
      }
      if (t > 0) {
        std::vector<float> ts = {t};
        for (float x : opt.temperatures) if (x > t + 1e-4f) ts.push_back(x);
        opt.temperatures = ts;
      }
    }
    return true;
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
    std::string bad;
    if (!read_options(field, translate, opt, bad)) return fail(400, bad);
    const std::string gran = field("timestamp_granularities[]") + "," + field("timestamp_granularities");
    // word timing: asked for, or the formats that place words (subtitle cues are split on them)
    opt.word_timestamps = gran.find("word") != std::string::npos ||
                          !(fmt == "json" || fmt == "text" || fmt == "verbose_json");
    const bool stream = field("stream") == "true";

    struct Count { std::atomic<int>& n; int before; Count(std::atomic<int>& x) : n(x), before(n++) {} ~Count() { --n; } } waiting{waiting_};
    if (waiting.before >= kMaxWaiting) return fail(429, "too many transcriptions waiting; try again later", "rate_limit_error");
    std::unique_lock<std::mutex> one(req_mu_);  // (one request at a time; the engine itself a window at a time)
    if (rs.client_gone()) return;
    opt.window_turn = [this](const std::function<void()>& f) { turn_.run(false, f); };
    // (decoded in its turn: one file's samples in memory at a time, instead of every waiting upload's at once)
    std::vector<float> audio;
    try {
      audio = audio::decode_16k(std::vector<uint8_t>(fit->second.data.begin(), fit->second.data.end()), max_seconds);
    } catch (const std::exception& e) {
      return fail(400, std::string(e.what()) + " (the server takes audio and video files Windows can decode)");
    }
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

  // ---------------------------------------------------------------- saved transcripts (the Transcript tab)
  // Each in its own folder <dir>\<id>\: meta.json (name, status, progress, options), source.<ext> (the uploaded file,
  // for playback), transcript.json (the full JSON; edits are saved over it). The file comes in pieces; once complete
  // it is queued, and a worker thread transcribes the queue one file at a time (with the OpenAI endpoint's lock: one
  // engine). A restart queues again what was queued or running; an upload the restart cut is marked failed.
  //   GET    /v1/transcripts                       the list, newest first (status, progress, place in the queue)
  //   POST   /v1/transcripts                       {name, size, language, beam_size, terms, prompt, dual_language,
  //                                                 condition_on_previous_text} -> the record: a new transcript
  //   PUT    /v1/transcripts/<id>/source?offset=N  the file's next piece (at most 256 MB a request)
  //   GET    /v1/transcripts/<id>                  its record, and its transcript once there is one
  //   PUT    /v1/transcripts/<id>                  {name}: renamed; {summary}: a meeting's summary (Markdown)
  //   PUT    /v1/transcripts/<id>/transcript       the transcript edited (its full JSON), saved over it
  //   GET    /v1/transcripts/<id>/export?format=srt[&max_chars=42&max_lines=2&max_cue_s=7&word_cues=1]
  //   GET    /v1/transcripts/<id>/media            the uploaded file (byte ranges: a <video> / <audio> seeks)
  //   POST   /v1/transcripts/<id>/cancel           stops it, queued or running (what was done is kept)
  //   DELETE /v1/transcripts/<id>                  stops it and deletes its folder
  void start_library(const std::filesystem::path& dir) {
    namespace fs = std::filesystem;
    lib_ = dir;
    std::error_code ec;
    fs::create_directories(lib_, ec);
    for (const auto& e : fs::directory_iterator(lib_, ec)) {
      const std::string tid = e.path().filename().string();
      Json m;
      if (!valid_id(tid) || !e.is_directory() || !Json::try_parse(slurp(e.path() / "meta.json"), m) || !m.is_object())
        continue;
      const std::string st = m["status"].str();
      if (st == "uploading") {  // (the server stopped during the upload: the browser has let the file go)
        m.set("status", "failed");
        m.set("error", "the upload did not finish before the server stopped");
        fs::remove(e.path() / ("source." + m["ext"].str()), ec);
      } else if (st == "running") {
        m.set("status", "queued");
        m.set("progress", 0.0);
      } else if (st == "live" || st == "deleting") {  // (a recording the stop cut: its WAV sizes put right)
        const fs::path w = e.path() / "source.wav";
        const uint64_t n = fs::file_size(w, ec);
        std::fstream f(w, std::ios::binary | std::ios::in | std::ios::out);
        if (!ec && n >= 44 && f) {
          const uint32_t data = (uint32_t)std::min<uint64_t>(n - 44, 0xFFFFFFF0u), riff = data + 36;
          f.seekp(4);
          f.write((const char*)&riff, 4);
          f.seekp(40);
          f.write((const char*)&data, 4);
        }
        m.set("status", st == "deleting" ? "cancelled" : "queued");
        m.set("size", (double)n);
      }
      metas_[tid] = m;
      save_meta(tid);
    }
    worker_ = std::thread([this] { work(); });
    live_worker_ = std::thread([this] { live_work(); });
  }
  void stop_library() {
    {
      std::lock_guard<std::mutex> lk(live_mu_);
      stop_live_ = true;
    }
    live_cv_.notify_all();
    if (live_worker_.joinable()) live_worker_.join();
    {
      std::lock_guard<std::mutex> lk(lib_mu_);
      stop_ = true;
    }
    lib_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  }
  ~Service() { stop_library(); }

  void transcripts(const HttpRequest& rq, HttpResponse& rs) {
    namespace fs = std::filesystem;
    auto fail = [&](int st, const std::string& msg) {
      Json e = Json::object(), r = Json::object();
      e.set("message", msg);
      e.set("type", st == 404 ? "not_found_error" : st >= 500 ? "server_error" : "invalid_request_error");
      r.set("error", e);
      rs.send(st, "application/json", r.dump());
    };
    if (lib_.empty()) return fail(503, "the transcript library is not open");
    const std::string rest = rq.path.size() > 15 ? rq.path.substr(16) : "";  // (after "/v1/transcripts/")
    const size_t slash = rest.find('/');
    const std::string tid = rest.substr(0, slash), what = slash == std::string::npos ? "" : rest.substr(slash + 1);
    if (rq.path == "/v1/transcripts" && rq.method == "GET") {
      std::lock_guard<std::mutex> lk(lib_mu_);
      std::vector<const Json*> all;
      for (const auto& [k, m] : metas_)
        if (m["status"].str() != "deleting") all.push_back(&m);
      std::stable_sort(all.begin(), all.end(), [](const Json* a, const Json* b) {
        return (*a)["created"].as_number() > (*b)["created"].as_number();
      });
      Json d = Json::array();
      for (const Json* m : all) {
        Json x = *m;
        if (x["status"].str() == "queued") x.set("position", (double)queue_position(x["id"].str()));
        d.push(x);
      }
      Json r = Json::object();
      r.set("data", d);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path == "/v1/transcripts" && rq.method == "POST") {
      Json in;
      if (!Json::try_parse(rq.body, in) || !in.is_object()) return fail(400, "a JSON object expected");
      auto field = [&](const char* k) -> std::string {
        const Json& v = in[k];
        if (v.is_string()) return v.str();
        if (v.is_bool()) return v.as_bool() ? "true" : "false";
        if (v.is_number()) {
          std::ostringstream o;
          o << v.as_number();
          return o.str();
        }
        return "";
      };
      whisper::Options opt;
      std::string bad;
      if (!read_options(field, false, opt, bad)) return fail(400, bad);
      const double size = in["size"].as_number(0);
      if (!(size >= 1 && size <= 1e12)) return fail(400, "size: the file's size in bytes");
      std::error_code ec;
      const fs::space_info sp = fs::space(lib_, ec);
      if (!ec && (double)sp.available < size + (512.0 * (1 << 20)))
        return fail(507, "not enough free disk space for the file (" + std::to_string((uint64_t)(size / (1 << 20))) + " MB)");
      std::string name = in["name"].str("recording");
      std::string clean;
      for (unsigned char c : name)
        if (c >= 0x20 && c != '"' && c != '\\' && c != '/' && c != 0x7F) clean += (char)c;
      if (clean.size() > 200) clean.resize(200);
      if (clean.empty()) clean = "recording";
      std::string ext;
      const size_t dot = clean.rfind('.');
      if (dot != std::string::npos)
        for (char c : clean.substr(dot + 1)) ext += (char)tolower((unsigned char)c);
      if (ext.empty() || ext.size() > 5 || ext.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789") != std::string::npos)
        ext = "bin";
      Json o = Json::object();
      for (const char* k : {"language", "beam_size", "terms", "prompt", "dual_language", "condition_on_previous_text"})
        if (!field(k).empty()) o.set(k, field(k));
      Json m = Json::object();
      const std::string nid = new_id();
      m.set("id", nid);
      m.set("name", clean);
      m.set("ext", ext);
      m.set("size", size);
      m.set("received", 0.0);
      m.set("status", "uploading");
      m.set("progress", 0.0);
      m.set("options", o);
      if (in["kind"].str() == "meeting") m.set("kind", "meeting");
      fs::create_directories(lib_ / nid, ec);
      if (ec) return fail(500, "cannot create the transcript's folder: " + ec.message());
      std::ofstream(lib_ / nid / ("source." + ext), std::ios::binary);  // (an empty file to append to)
      {
        std::lock_guard<std::mutex> lk(lib_mu_);
        // (seconds with milliseconds, always later than the last one: the queue's order, files added in one second)
        double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        for (const auto& [k, x] : metas_) now = std::max(now, x["created"].as_number() + 0.001);
        m.set("created", (double)std::llround(now * 1000) / 1000);
        metas_[nid] = m;
        save_meta(nid);
      }
      return rs.send(200, "application/json", m.dump());
    }
    if (!valid_id(tid)) return fail(404, "no transcript " + tid);
    Json m;
    {
      std::lock_guard<std::mutex> lk(lib_mu_);
      auto it = metas_.find(tid);
      if (it == metas_.end() || it->second["status"].str() == "deleting") return fail(404, "no transcript " + tid);
      m = it->second;
    }
    const fs::path dir = lib_ / tid, src = dir / ("source." + m["ext"].str());
    if (what == "source" && rq.method == "PUT") {
      const double off = atof(query_param(rq.query, "offset").c_str());
      std::lock_guard<std::mutex> lk(lib_mu_);
      Json& cur = metas_[tid];
      if (cur["status"].str() != "uploading") return fail(409, "the file is complete");
      if (off != cur["received"].as_number()) return fail(409, "offset " + std::to_string((uint64_t)off) +
                                                               ": the next piece starts at " +
                                                               std::to_string((uint64_t)cur["received"].as_number()));
      const double got = cur["received"].as_number() + (double)rq.body.size();
      if (got > cur["size"].as_number()) return fail(400, "more bytes than the size given");
      {
        std::ofstream f(src, std::ios::binary | std::ios::app);
        f.write(rq.body.data(), (std::streamsize)rq.body.size());
        if (!f) return fail(500, "cannot write the file (disk full?)");
      }
      cur.set("received", got);
      if (got == cur["size"].as_number()) {
        cur.set("status", "queued");
        lib_cv_.notify_all();
      }
      save_meta(tid);
      return rs.send(200, "application/json", cur.dump());
    }
    if (what.empty() && rq.method == "GET") {
      Json t;
      if (Json::try_parse(slurp(dir / "transcript.json"), t)) m.set("transcript", t);
      if (fs::exists(dir / "summary.md")) m.set("summary", slurp(dir / "summary.md"));
      if (m["status"].str() == "queued") m.set("position", (double)queue_position(tid));
      return rs.send(200, "application/json", m.dump());
    }
    if (what.empty() && rq.method == "PUT") {
      Json in;
      if (!Json::try_parse(rq.body, in) || !in.is_object() || (!in["name"].is_string() && !in["summary"].is_string()))
        return fail(400, "{\"name\": ...} or {\"summary\": ...} expected");
      if (in["summary"].is_string()) {  // (a meeting's summary, Markdown)
        if (in["summary"].str().size() > (1u << 20)) return fail(400, "summary: 1 MB at most");
        if (!write_atomic(dir / "summary.md", in["summary"].str())) return fail(500, "cannot write the summary");
      }
      std::lock_guard<std::mutex> lk(lib_mu_);
      if (in["name"].is_string() && !in["name"].str().empty()) {
        std::string clean;
        for (unsigned char c : in["name"].str())
          if (c >= 0x20 && c != '"' && c != '\\' && c != '/' && c != 0x7F) clean += (char)c;
        if (clean.size() > 200) clean.resize(200);
        metas_[tid].set("name", clean.empty() ? std::string("recording") : clean);
      }
      if (in["summary"].is_string()) metas_[tid].set("summarized", (double)std::time(nullptr));
      save_meta(tid);
      return rs.send(200, "application/json", metas_[tid].dump());
    }
    if (what == "transcript" && rq.method == "PUT") {
      Json in;
      if (!Json::try_parse(rq.body, in) || !in.is_object() || !in["segments"].is_array())
        return fail(400, "a transcript's full JSON ({\"duration\", \"segments\": [...]}) expected");
      if (m["status"].str() == "running" || m["status"].str() == "queued" || m["status"].str() == "uploading")
        return fail(409, "the transcript is not finished yet");
      const whisper::Result r = result_from_json(in);
      if (!write_atomic(dir / "transcript.json", whisper::to_json(r))) return fail(500, "cannot write the transcript");
      std::lock_guard<std::mutex> lk(lib_mu_);
      metas_[tid].set("edited", (double)std::time(nullptr));
      metas_[tid].set("segments", (double)r.segments.size());
      save_meta(tid);
      return rs.send(200, "application/json", "{\"saved\":true}");
    }
    if (what == "export" && rq.method == "GET") {
      Json t;
      if (!Json::try_parse(slurp(dir / "transcript.json"), t)) return fail(404, "no transcript yet");
      const std::string fmt = query_param(rq.query, "format").empty() ? "srt" : query_param(rq.query, "format");
      const auto& fs_ = whisper::formats();
      const auto f = std::find_if(fs_.begin(), fs_.end(), [&](const whisper::Format& x) { return fmt == x.name; });
      if (f == fs_.end()) return fail(400, "format: json, fw, captions, verbose_json, srt, vtt, sbv, srv3, ass, ttml, txt, md, csv or tsv");
      whisper::CueOptions co;
      if (!query_param(rq.query, "max_chars").empty()) co.max_chars = std::clamp(atoi(query_param(rq.query, "max_chars").c_str()), 8, 200);
      if (!query_param(rq.query, "max_lines").empty()) co.max_lines = std::clamp(atoi(query_param(rq.query, "max_lines").c_str()), 1, 4);
      if (!query_param(rq.query, "max_cue_s").empty()) co.max_s = std::clamp(atof(query_param(rq.query, "max_cue_s").c_str()), 0.5, 30.0);
      co.one_word = query_param(rq.query, "word_cues") == "1";
      std::string stem = m["name"].str("transcript");
      const size_t dot = stem.rfind('.');
      if (dot != std::string::npos && dot > 0) stem.resize(dot);
      std::string body;
      try {
        body = whisper::render(fmt, result_from_json(t), co, stem);
      } catch (const std::exception& e) {
        return fail(500, e.what());
      }
      // (the name as typed, UTF-8, for the browsers; an ASCII one for the rest)
      std::string ascii, enc;
      for (unsigned char c : stem) {
        ascii += (c < 0x80 && c != ';') ? (char)c : '_';
        if (isalnum(c) || c == '-' || c == '_' || c == '.') enc += (char)c;
        else { char b[4]; snprintf(b, sizeof b, "%%%02X", c); enc += b; }
      }
      rs.extra = "Content-Disposition: attachment; filename=\"" + ascii + f->ext + "\"; filename*=UTF-8''" + enc + f->ext + "\r\n";
      const bool json = std::string(f->ext).find(".json") != std::string::npos;
      return rs.send(200, json ? "application/json" : fmt == "srv3" || fmt == "ttml" ? "text/xml; charset=utf-8" : "text/plain; charset=utf-8", body);
    }
    if (what == "media" && rq.method == "GET") {
      static const std::map<std::string, std::string> kMime = {
          {"mp4", "video/mp4"}, {"m4v", "video/mp4"}, {"mov", "video/quicktime"}, {"webm", "video/webm"},
          {"mkv", "video/x-matroska"}, {"avi", "video/x-msvideo"}, {"wmv", "video/x-ms-wmv"}, {"mp3", "audio/mpeg"},
          {"m4a", "audio/mp4"}, {"aac", "audio/aac"}, {"wav", "audio/wav"}, {"ogg", "audio/ogg"}, {"opus", "audio/ogg"},
          {"oga", "audio/ogg"}, {"flac", "audio/flac"}, {"wma", "audio/x-ms-wma"}};
      const auto mt = kMime.find(m["ext"].str());
      rs.extra = "X-Content-Type-Options: nosniff\r\n";
      return rs.send_file(src, mt == kMime.end() ? "application/octet-stream" : mt->second, rq.header("range"));
    }
    if (what == "cancel" && rq.method == "POST") {
      std::lock_guard<std::mutex> lk(lib_mu_);
      Json& cur = metas_[tid];
      const std::string st = cur["status"].str();
      if (tid == running_) cancel_running_ = true;
      else if (st == "live") {  // (the recording stops; its draft is kept, no accurate pass)
        cur.set("status", "cancelled");
        save_meta(tid);
        if (auto s = live_of(tid)) live_stop(s);
      } else if (st == "queued" || st == "uploading") {
        cur.set("status", "cancelled");
        save_meta(tid);
      }
      return rs.send(200, "application/json", cur.dump());
    }
    if (what.empty() && rq.method == "DELETE") {
      std::unique_lock<std::mutex> lk(lib_mu_);
      if (tid == running_) {  // (the worker deletes it when it lets go of it)
        cancel_running_ = delete_running_ = true;
        metas_[tid].set("status", "deleting");
      } else if (metas_[tid]["status"].str() == "live") {  // (and so does the live recording's end)
        metas_[tid].set("status", "deleting");
        if (auto s = live_of(tid)) live_stop(s);
      } else {
        metas_.erase(tid);
        lk.unlock();
        std::error_code ec;
        fs::remove_all(dir, ec);
      }
      return rs.send(200, "application/json", "{\"deleted\":true}");
    }
    return fail(404, "unknown transcripts request " + rq.method + " " + rq.path);
  }

  // ---------------------------------------------------------------- live transcription (Transcript tab's Live, the
  // Meeting tab): the browser sends the audio as it comes, 16-bit PCM at its own rate, one or two tracks interleaved
  // (the microphone; the other side: a tab's or the screen's sound). It is written to the recording (the
  // transcript's source.wav) and cut into utterances per track at its pauses; each utterance is transcribed at once
  // (greedy, word timestamps; before the queue's windows) into the draft. Stopping queues the accurate pass over the
  // recording (its options: beam 5 by default; each track alone with its speaker, merged by time), which replaces the
  // draft; the recording is then kept as MP3 (192 kbps unless mp3_kbps says otherwise).
  //   POST /v1/live                    {name, language, terms, beam_size, tracks: 1 | 2, rate, speakers: [..],
  //                                     mp3_kbps} -> the record (status "live")
  //   POST /v1/live/<id>/audio?since=N the next piece -> {segments: the draft's from the N-th on, count, heard_s,
  //                                     pending: utterances not transcribed yet}
  //   POST /v1/live/<id>/mark          {t}: a mark at t seconds
  //   POST /v1/live/<id>/stop          the recording closed, the rest of the draft and the accurate pass to come
  void live(const HttpRequest& rq, HttpResponse& rs) {
    namespace fs = std::filesystem;
    auto fail = [&](int st, const std::string& msg) {
      Json e = Json::object(), r = Json::object();
      e.set("message", msg);
      e.set("type", st == 404 ? "not_found_error" : st >= 500 ? "server_error" : "invalid_request_error");
      r.set("error", e);
      rs.send(st, "application/json", r.dump());
    };
    if (lib_.empty()) return fail(503, "the transcript library is not open");
    if (rq.path == "/v1/live" && rq.method == "POST") {
      Json in;
      if (!Json::try_parse(rq.body, in) || !in.is_object()) return fail(400, "a JSON object expected");
      auto field = [&](const char* k) -> std::string {
        const Json& v = in[k];
        if (v.is_string()) return v.str();
        if (v.is_bool()) return v.as_bool() ? "true" : "false";
        if (v.is_number()) {
          std::ostringstream o;
          o << v.as_number();
          return o.str();
        }
        return "";
      };
      whisper::Options opt;
      std::string bad;
      if (!read_options(field, false, opt, bad)) return fail(400, bad);
      const int rate = (int)in["rate"].as_number(48000), tracks = (int)in["tracks"].as_number(1);
      if (rate < 8000 || rate > 96000) return fail(400, "rate: 8000..96000");
      if (tracks < 1 || tracks > 2) return fail(400, "tracks: 1 or 2");
      auto s = std::make_shared<Live>();
      s->rate = rate;
      s->tracks = tracks;
      s->t.resize(tracks);
      for (LiveTrack& t : s->t) t.rs = std::make_unique<audio::Resampler>(rate, 16000);
      Json sp = Json::array();
      for (int k = 0; k < tracks; ++k) {
        std::string w = in["speakers"][k].str(tracks == 1 ? "" : k == 0 ? "Me" : "Others");
        if (w.size() > 40) w.resize(40);
        s->speakers.push_back(w);
        sp.push(w);
      }
      s->opt = opt;
      s->opt.beam = 1;  // (the draft: greedy; the accurate pass has the options as given)
      s->opt.word_timestamps = true;
      s->opt.window_turn = [this](const std::function<void()>& f) { turn_.run(true, f); };
      Json o = Json::object();
      for (const char* k : {"language", "beam_size", "terms", "prompt", "dual_language"})
        if (!field(k).empty()) o.set(k, field(k));
      std::string name = in["name"].str("recording");
      std::string clean;
      for (unsigned char c : name)
        if (c >= 0x20 && c != '"' && c != '\\' && c != '/' && c != 0x7F) clean += (char)c;
      if (clean.size() > 200) clean.resize(200);
      const std::string nid = new_id();
      s->id = nid;
      std::error_code ec;
      fs::create_directories(lib_ / nid, ec);
      s->wav.open(lib_ / nid / "source.wav", std::ios::binary | std::ios::trunc);
      if (ec || !s->wav) return fail(500, "cannot create the recording");
      write_wav_header(*s);
      Json m = Json::object();
      m.set("id", nid);
      m.set("name", clean.empty() ? std::string("recording") : clean);
      m.set("ext", "wav");
      m.set("size", 0.0);
      m.set("status", "live");
      m.set("progress", 0.0);
      m.set("options", o);
      m.set("tracks", (double)tracks);
      m.set("speakers", sp);
      m.set("rate", (double)rate);
      m.set("recording", true);
      if (in["kind"].str() == "meeting") m.set("kind", "meeting");
      m.set("mp3_kbps", (double)std::clamp((int)in["mp3_kbps"].as_number(192), 96, 320));
      m.set("marks", Json::array());
      {
        std::lock_guard<std::mutex> lk(lib_mu_);
        double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        for (const auto& [k, x] : metas_) now = std::max(now, x["created"].as_number() + 0.001);
        m.set("created", (double)std::llround(now * 1000) / 1000);
        metas_[nid] = m;
        save_meta(nid);
      }
      {
        std::lock_guard<std::mutex> lk(live_mu_);
        lives_[nid] = s;
      }
      return rs.send(200, "application/json", m.dump());
    }
    const std::string rest = rq.path.size() > 9 ? rq.path.substr(9) : "";  // (after "/v1/live/")
    const size_t slash = rest.find('/');
    const std::string tid = rest.substr(0, slash), what = slash == std::string::npos ? "" : rest.substr(slash + 1);
    std::shared_ptr<Live> s;
    {
      std::lock_guard<std::mutex> lk(live_mu_);
      auto it = lives_.find(tid);
      if (it != lives_.end()) s = it->second;
    }
    if (!s || rq.method != "POST") return fail(404, "no live recording " + tid);
    if (what == "audio") {
      const size_t frame = (size_t)2 * s->tracks;
      if (rq.body.size() % frame) return fail(400, "16-bit samples, the tracks interleaved");
      std::vector<LiveTask> todo;
      Json out = Json::object();
      {
        std::lock_guard<std::mutex> lk(live_mu_);
        if (s->stopped) return fail(409, "the recording has stopped");
        s->wav.write(rq.body.data(), (std::streamsize)rq.body.size());
        if (!s->wav) return fail(500, "cannot write the recording (disk full?)");
        const size_t n = rq.body.size() / frame;
        s->frames += n;
        write_wav_header(*s);  // (sizes kept right as it grows: a crash loses nothing written)
        std::vector<float> x(n);
        for (int k = 0; k < s->tracks; ++k) {
          for (size_t i = 0; i < n; ++i) {
            int16_t v;
            memcpy(&v, rq.body.data() + i * frame + 2 * k, 2);
            x[i] = v / 32768.0f;
          }
          LiveTrack& t = s->t[k];
          t.rs->push(x.data(), n, t.x);
          t.n = t.x0 + (int64_t)t.x.size();
          live_vad(*s, k, todo);
          // (only with nothing else waiting: the final texts first)
          if (t.from >= 0 && t.n - t.from >= 24000 && !t.partial_busy && t.n - t.partial_n >= 16000 &&
              live_q_.empty() && todo.empty()) {
            LiveTask p;
            p.track = k;
            p.from = t.from;
            p.partial = true;
            p.x.assign(t.x.begin() + (t.from - t.x0), t.x.end());
            t.partial_busy = true;
            t.partial_n = t.n;
            todo.push_back(std::move(p));
          }
        }
        for (const LiveTask& u : todo) s->pending += u.partial ? 0 : 1;
        const size_t since = (size_t)std::max(0, atoi(query_param(rq.query, "since").c_str()));
        Json segs = Json::array();
        for (size_t i = since; i < s->draft.size(); ++i) segs.push(segment_json(s->draft[i]));
        out.set("segments", segs);
        Json part = Json::array();  // (the open utterances' text so far, per track)
        for (int k = 0; k < s->tracks; ++k)
          if (!s->t[k].partial.empty()) {
            Json p = Json::object();
            p.set("track", k);
            if (s->tracks > 1) p.set("speaker", s->speakers[k]);
            p.set("start", (double)s->t[k].partial_from / 16000);
            p.set("upto", (double)s->t[k].partial_to / 16000);
            p.set("text", s->t[k].partial);
            part.push(p);
          }
        out.set("partial", part);
        out.set("count", (double)s->draft.size());
        out.set("heard_s", (double)s->frames / s->rate);
        out.set("pending", (double)s->pending);
      }
      if (!todo.empty()) {
        std::lock_guard<std::mutex> lk(live_mu_);
        for (LiveTask& t : todo) {
          t.s = s;
          live_q_.push_back(std::move(t));
        }
        turn_.demand((int)todo.size());
        live_cv_.notify_all();
      }
      return rs.send(200, "application/json", out.dump());
    }
    if (what == "mark") {
      Json in;
      Json::try_parse(rq.body, in);
      std::lock_guard<std::mutex> lk(lib_mu_);
      Json marks = metas_[tid]["marks"];
      marks.push(in["t"].as_number((double)s->frames / s->rate));
      metas_[tid].set("marks", marks);
      save_meta(tid);
      return rs.send(200, "application/json", metas_[tid].dump());
    }
    if (what == "stop") {
      live_stop(s);
      std::lock_guard<std::mutex> lk(live_mu_);
      Json r = Json::object();
      r.set("id", tid);
      r.set("status", "finishing");
      r.set("pending", (double)s->pending);
      return rs.send(200, "application/json", r.dump());
    }
    return fail(404, "unknown live request " + rq.path);
  }

 private:
  struct LiveTrack {
    std::unique_ptr<audio::Resampler> rs;  // (to 16 kHz)
    std::vector<float> x;                  // 16-kHz samples from x0 on
    int64_t x0 = 0, n = 0, at = 0;         // (n: all the samples so far; at: the next frame to look at)
    float floor = -50;                     // the noise floor, dB
    int64_t from = -1, voice = -1;         // the open utterance: its start, its last voiced frame's end
    // the open utterance's text so far (transcribed again each second while it lasts; the final one replaces it)
    std::string partial;
    int64_t partial_from = -1, partial_to = 0, partial_n = 0;
    bool partial_busy = false;
  };
  struct Live {
    std::string id;
    int rate = 48000, tracks = 1;
    std::vector<std::string> speakers;
    whisper::Options opt;
    std::ofstream wav;
    uint64_t frames = 0;
    std::vector<LiveTrack> t;
    std::vector<whisper::Segment> draft;  // (in the order they came)
    int pending = 0;                      // (utterances not transcribed yet)
    bool stopped = false;
  };
  struct LiveTask {
    std::shared_ptr<Live> s;
    int track = 0;
    int64_t from = 0;
    std::vector<float> x;
    bool finish = false, partial = false;
  };
  // the recording closed: the open utterances and what the resamplers hold go to the queue, then its end (the draft
  // finished, the accurate pass queued)
  void live_stop(const std::shared_ptr<Live>& s) {
    std::lock_guard<std::mutex> lk(live_mu_);
    if (s->stopped) return;
    s->stopped = true;
    for (int k = 0; k < s->tracks; ++k) {
      LiveTrack& t = s->t[k];
      t.rs->finish(t.x);
      t.n = t.x0 + (int64_t)t.x.size();
      std::vector<LiveTask> todo;
      live_vad(*s, k, todo, true);
      s->pending += (int)todo.size();
      for (LiveTask& x : todo) {
        x.s = s;
        live_q_.push_back(std::move(x));
      }
      turn_.demand((int)todo.size());
    }
    LiveTask fin;
    fin.s = s;
    fin.finish = true;
    live_q_.push_back(std::move(fin));
    live_cv_.notify_all();
  }
  std::shared_ptr<Live> live_of(const std::string& tid) {
    std::lock_guard<std::mutex> lk(live_mu_);
    auto it = lives_.find(tid);
    return it == lives_.end() ? nullptr : it->second;
  }
  static void write_wav_header(Live& s) {  // (live_mu_ held)
    const uint32_t data = (uint32_t)std::min<uint64_t>(s.frames * 2 * s.tracks, 0xFFFFFFF0u);
    uint8_t h[44];
    auto u32 = [&](int o, uint32_t v) { for (int i = 0; i < 4; ++i) h[o + i] = (uint8_t)(v >> (8 * i)); };
    auto u16 = [&](int o, uint32_t v) { h[o] = (uint8_t)v; h[o + 1] = (uint8_t)(v >> 8); };
    memcpy(h, "RIFF", 4);
    u32(4, 36 + data);
    memcpy(h + 8, "WAVEfmt ", 8);
    u32(16, 16);
    u16(20, 1);
    u16(22, (uint32_t)s.tracks);
    u32(24, (uint32_t)s.rate);
    u32(28, (uint32_t)(s.rate * 2 * s.tracks));
    u16(32, (uint32_t)(2 * s.tracks));
    u16(34, 16);
    memcpy(h + 36, "data", 4);
    u32(40, data);
    const auto end = s.wav.tellp();
    s.wav.seekp(0);
    s.wav.write((const char*)h, 44);
    s.wav.seekp(end);
    s.wav.flush();
  }
  // a track's new 16-kHz samples in 10-ms frames against its noise floor (it follows the quietest frames down at
  // once and rises 2 dB a second): voiced 12 dB above it. An utterance runs from 0.2 s before its first voiced frame
  // to 0.2 s after its last, cut at a pause of 0.6 s (0.3 s once it is 6 s long) or at 15 s; one with less than
  // 0.3 s voiced is left out. (Cut at 0.6 s or 25 s only, continuous speech came out up to 24 s late; the partial
  // text, below, shows it as it goes.)
  // flush: the open one ends here (the recording stopped)
  static void live_vad(Live& s, int k, std::vector<LiveTask>& out, bool flush = false) {
    LiveTrack& t = s.t[k];
    auto cut = [&](int64_t b) {
      if (t.voice - t.from >= 3200 + 4800) {
        LiveTask x;
        x.track = k;
        x.from = t.from;
        x.x.assign(t.x.begin() + (t.from - t.x0), t.x.begin() + (std::min(b, t.n) - t.x0));
        out.push_back(std::move(x));
      }
      t.from = -1;
      t.partial.clear();
      t.partial_from = -1;
    };
    for (; t.at + 160 <= t.n; t.at += 160) {
      const float* p = &t.x[t.at - t.x0];
      double e = 0;
      for (int i = 0; i < 160; ++i) e += (double)p[i] * p[i];
      const float db = (float)(10 * std::log10(e / 160 + 1e-10));
      t.floor = db < t.floor ? db : t.floor + 0.02f;
      const int64_t end = t.at + 160;
      if (db > t.floor + 12 && db > -50) {
        if (t.from < 0) t.from = std::max(t.x0, t.at - 3200);
        t.voice = end;
      }
      if (t.from >= 0 && (end - t.voice >= 9600 || (end - t.from >= 6 * 16000 && end - t.voice >= 4800) ||
                          end - t.from >= 15 * 16000))
        cut(std::min(end, t.voice + 3200));
    }
    if (flush && t.from >= 0) cut(t.n);
    const int64_t keep = t.from >= 0 ? t.from : std::max(t.x0, t.at - 3200);
    if (keep - t.x0 > 16000 * 10) {
      t.x.erase(t.x.begin(), t.x.begin() + (keep - t.x0));
      t.x0 = keep;
    }
  }
  static Json segment_json(const whisper::Segment& g) {
    Json j = Json::object(), ws = Json::array();
    j.set("start", g.start);
    j.set("end", g.end);
    j.set("lang", g.lang);
    if (!g.speaker.empty()) j.set("speaker", g.speaker);
    j.set("text", whisper::trim_copy(g.text));
    for (const whisper::Word& w : g.words) {
      Json x = Json::object();
      x.set("start", w.start);
      x.set("end", w.end);
      x.set("word", w.text);
      x.set("p", (double)w.p);
      ws.push(x);
    }
    j.set("words", ws);
    return j;
  }
  void live_work() {
    for (;;) {
      LiveTask t;
      {
        std::unique_lock<std::mutex> lk(live_mu_);
        live_cv_.wait(lk, [&] { return stop_live_ || !live_q_.empty(); });
        if (live_q_.empty()) return;
        t = std::move(live_q_.front());
        live_q_.pop_front();
      }
      Live& s = *t.s;
      if (t.finish) {
        live_finish(t.s);
        continue;
      }
      struct Done {
        EngineTurn& e;
        ~Done() { e.demand(-1); }
      } done{turn_};
      std::vector<whisper::Segment> segs;
      try {
        LockedBackend be(*eng_, gpu_turn);
        whisper::Transcriber tr(*m_, be);
        whisper::Options o = s.opt;
        if (t.partial) {  // (text only, one pass: it is replaced)
          o.word_timestamps = false;
          o.temperatures = {0.0f};
        }
        segs = tr.run(t.x, o).segments;
      } catch (const std::exception& e) {
        fprintf(stderr, "whisper live: %s\n", e.what());
      }
      const double off = (double)t.from / 16000;
      std::lock_guard<std::mutex> lk(live_mu_);
      if (t.partial) {
        LiveTrack& tk = s.t[t.track];
        tk.partial_busy = false;
        if (tk.from == t.from) {  // (still open: its final text has not come)
          std::string text;
          for (const whisper::Segment& g : segs) text += g.text;
          tk.partial = whisper::trim_copy(text);
          tk.partial_from = t.from;
          tk.partial_to = t.from + (int64_t)t.x.size();
        }
        continue;
      }
      for (whisper::Segment& g : segs) {
        g.start += off;
        g.end += off;
        for (whisper::Word& w : g.words) { w.start += off; w.end += off; }
        if (s.tracks > 1) g.speaker = s.speakers[t.track];
        s.draft.push_back(std::move(g));
      }
      --s.pending;
      whisper::Result r;  // (the draft on disk, in time order: GET /v1/transcripts/<id> shows it)
      r.segments = s.draft;
      std::stable_sort(r.segments.begin(), r.segments.end(),
                       [](const whisper::Segment& a, const whisper::Segment& b) { return a.start < b.start; });
      r.duration = (double)s.frames / s.rate;
      write_atomic(lib_ / s.id / "transcript.json", whisper::to_json(r));
    }
  }
  void live_finish(const std::shared_ptr<Live>& s) {  // (the recording closed; the accurate pass queued)
    {
      std::lock_guard<std::mutex> lk(live_mu_);
      write_wav_header(*s);
      s->wav.close();
      lives_.erase(s->id);
    }
    std::lock_guard<std::mutex> lk(lib_mu_);
    auto it = metas_.find(s->id);
    if (it == metas_.end()) return;
    std::error_code ec;
    if (it->second["status"].str() == "deleting") {
      metas_.erase(it);
      std::filesystem::remove_all(lib_ / s->id, ec);
      return;
    }
    it->second.set("size", (double)std::filesystem::file_size(lib_ / s->id / "source.wav", ec));
    it->second.set("duration", (double)s->frames / s->rate);
    it->second.set("status", it->second["status"].str() == "cancelled" ? "cancelled" : "queued");
    save_meta(s->id);
    lib_cv_.notify_all();
  }

  static bool valid_id(const std::string& tid) {
    return tid.size() >= 8 && tid.size() <= 40 && tid.find_first_not_of("0123456789abcdefghijklmnopqrstuvwxyz-") == std::string::npos;
  }
  static std::string slurp(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
  }
  // written whole or not at all (a crash in the middle leaves the old file)
  static bool write_atomic(const std::filesystem::path& p, const std::string& s) {
    const std::filesystem::path tmp = p.string() + ".tmp";
    {
      std::ofstream f(tmp, std::ios::binary);
      f << s;
      if (!f) return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, p, ec);
    return !ec;
  }
  std::string new_id() const {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &t);
    char b[40];
    std::strftime(b, sizeof b, "%Y%m%d-%H%M%S", &tm);
    static std::mt19937 rng(std::random_device{}());
    char r[8];
    snprintf(r, sizeof r, "-%04x", (unsigned)(rng() & 0xFFFF));
    return std::string(b) + r;
  }
  void save_meta(const std::string& tid) {  // (lib_mu_ held)
    write_atomic(lib_ / tid / "meta.json", metas_[tid].dump());
  }
  int queue_position(const std::string& tid) const {  // (lib_mu_ held)
    const double c = metas_.at(tid)["created"].as_number();
    int n = 1;
    for (const auto& [k, m] : metas_)
      if (k != tid && m["status"].str() == "queued" && (m["created"].as_number() < c || (m["created"].as_number() == c && k < tid))) ++n;
    return n;
  }
  std::string next_queued() const {  // (lib_mu_ held) the oldest
    std::string best;
    double bc = 0;
    for (const auto& [k, m] : metas_)
      if (m["status"].str() == "queued" && (best.empty() || m["created"].as_number() < bc)) {
        best = k;
        bc = m["created"].as_number();
      }
    return best;
  }
  void work() {
    namespace fs = std::filesystem;
    for (;;) {
      std::string tid;
      Json meta;
      {
        std::unique_lock<std::mutex> lk(lib_mu_);
        lib_cv_.wait(lk, [&] { return stop_ || !next_queued().empty(); });
        if (stop_) return;
        tid = next_queued();
        running_ = tid;
        cancel_running_ = delete_running_ = false;
        metas_[tid].set("status", "running");
        metas_[tid].set("started", (double)std::time(nullptr));
        metas_[tid].set("error", nullptr);  // (a re-run's)
        save_meta(tid);
        meta = metas_[tid];
      }
      transcribe_job(tid, meta);
      std::lock_guard<std::mutex> lk(lib_mu_);
      running_.clear();
      if (delete_running_) {
        metas_.erase(tid);
        std::error_code ec;
        fs::remove_all(lib_ / tid, ec);
      }
    }
  }
  void transcribe_job(const std::string& tid, const Json& meta) {
    namespace fs = std::filesystem;
    using clk = std::chrono::steady_clock;
    const fs::path dir = lib_ / tid;
    auto finish = [&](const std::string& status, const std::string& error) {
      std::lock_guard<std::mutex> lk(lib_mu_);
      Json& m = metas_[tid];
      m.set("status", status);
      if (!error.empty()) m.set("error", error);
      save_meta(tid);
    };
    whisper::Options opt;
    std::string bad;
    const Json& o = meta["options"];
    if (!read_options([&](const char* k) { return o[k].str(); }, false, opt, bad)) return finish("failed", bad);
    opt.word_timestamps = true;
    opt.window_turn = [this](const std::function<void()>& f) { turn_.run(false, f); };
    // a live recording of several tracks (the microphone, the other side): each track transcribed alone, its segments
    // marked with its speaker, all of them merged by time
    const int tracks = std::max(1, (int)meta["tracks"].as_number(1));
    const fs::path src = dir / ("source." + meta["ext"].str());
    std::vector<std::vector<float>> audio;
    try {
      if (tracks > 1) audio = audio::decode_16k_wav_channels(src);
      else audio.push_back(audio::decode_16k_file(src, max_seconds));
    } catch (const std::exception& e) {
      return finish("failed", std::string(e.what()) + " (audio and video files Windows can decode)");
    }
    double all_s = 0;
    for (const auto& a : audio) all_s += (double)a.size() / 16000;
    const auto t0 = clk::now();
    auto last_save = t0;
    whisper::Result r;
    bool stopped = false;
    try {
      LockedBackend be(*eng_, gpu_turn);
      whisper::Transcriber tr(*m_, be);
      double before = 0;  // (the tracks done)
      for (size_t k = 0; k < audio.size() && !stopped; ++k) {
        whisper::Result rk = tr.run(audio[k], opt, [&](double done, double, const std::vector<whisper::Segment>& segs) {
          std::lock_guard<std::mutex> lk(lib_mu_);
          const double el = std::chrono::duration<double>(clk::now() - t0).count();
          Json& m = metas_[tid];
          m.set("progress", all_s > 0 ? (before + done) / all_s : 0.0);
          m.set("done_s", before + done);
          m.set("duration", all_s);
          m.set("x_realtime", el > 0 ? (before + done) / el : 0.0);
          m.set("segments", (double)(r.segments.size() + segs.size()));
          if (std::chrono::duration<double>(clk::now() - last_save).count() > 5) {
            save_meta(tid);
            last_save = clk::now();
          }
          stopped = cancel_running_ || stop_;
          return !stopped;
        });
        const std::string who = audio.size() > 1 ? meta["speakers"][(int)k].str("Track " + std::to_string(k + 1)) : "";
        for (whisper::Segment& g : rk.segments) g.speaker = who;
        r.segments.insert(r.segments.end(), rk.segments.begin(), rk.segments.end());
        r.duration = std::max(r.duration, rk.duration);
        before += rk.duration;
      }
      if (audio.size() > 1)
        std::stable_sort(r.segments.begin(), r.segments.end(),
                         [](const whisper::Segment& a, const whisper::Segment& b) { return a.start < b.start; });
    } catch (const std::exception& e) {
      return finish("failed", std::string("transcription failed: ") + e.what());
    }
    std::unique_lock<std::mutex> lk(lib_mu_);
    Json& m = metas_[tid];
    if (stop_ && !cancel_running_) {  // (the server is stopping: transcribed again after the restart)
      m.set("status", "queued");
      m.set("progress", 0.0);
      save_meta(tid);
      return;
    }
    write_atomic(dir / "transcript.json", whisper::to_json(r));
    Json langs = Json::array();
    std::vector<std::string> seen;
    size_t words = 0;
    for (const whisper::Segment& g : r.segments) {
      words += g.words.size();
      if (!g.lang.empty() && std::find(seen.begin(), seen.end(), g.lang) == seen.end()) {
        seen.push_back(g.lang);
        langs.push(g.lang);
      }
    }
    const double el = std::chrono::duration<double>(clk::now() - t0).count();
    m.set("status", stopped ? "cancelled" : "done");
    if (!stopped) m.set("progress", 1.0);
    m.set("duration", r.duration);
    m.set("languages", langs);
    m.set("segments", (double)r.segments.size());
    m.set("words", (double)words);
    m.set("elapsed_s", el);
    m.set("x_realtime", el > 0 ? (stopped ? m["done_s"].as_number() : all_s) / el : 0.0);
    save_meta(tid);
    // a live recording kept as MP3 once transcribed (the WAV it was written as is ~10x larger)
    if (stopped || m["ext"].str() != "wav" || !m["recording"].as_bool()) return;
    const int kbps = (int)m["mp3_kbps"].as_number(192);
    lk.unlock();
    std::string err;
    const bool ok = audio::encode_mp3(src, dir / "source.mp3", kbps, err);
    lk.lock();
    auto it = metas_.find(tid);
    if (it == metas_.end()) return;
    if (ok) {
      std::error_code ec;
      fs::remove(src, ec);
      it->second.set("ext", "mp3");
      std::error_code ec2;
      it->second.set("size", (double)fs::file_size(dir / "source.mp3", ec2));
    } else {
      it->second.set("mp3_error", err);
    }
    save_meta(tid);
  }

  // the engine, a window (or a live utterance) at a time; a live turn goes before every other one, and the others
  // wait while live utterances are queued (with only the waiting ones first, the queue's window took the engine
  // between every two live ones: the live text fell 7 s behind instead of 2 while a file was transcribed)
  class EngineTurn {
   public:
    void demand(int d) {  // (live utterances queued: + when queued, - when done)
      std::lock_guard<std::mutex> lk(m_);
      demand_ += d;
      if (demand_ <= 0) cv_.notify_all();
    }
    void run(bool live, const std::function<void()>& f) {
      std::unique_lock<std::mutex> lk(m_);
      if (live) ++live_waiting_;
      cv_.wait(lk, [&] { return !busy_ && (live || (live_waiting_ == 0 && demand_ <= 0)); });
      if (live) --live_waiting_;
      busy_ = true;
      lk.unlock();
      struct Done {
        EngineTurn& t;
        ~Done() {
          std::lock_guard<std::mutex> g(t.m_);
          t.busy_ = false;
          t.cv_.notify_all();
        }
      } done{*this};
      f();
    }

   private:
    std::mutex m_;
    std::condition_variable cv_;
    bool busy_ = false;
    int live_waiting_ = 0, demand_ = 0;
  };

  std::unique_ptr<whisper::Model> m_;
  std::unique_ptr<WhisperEngine> eng_;
  EngineTurn turn_;
  std::mutex req_mu_;  // (the OpenAI endpoint: one request at a time)
  std::atomic<int> waiting_{0};
  // the transcript library
  std::filesystem::path lib_;
  std::mutex lib_mu_;
  std::condition_variable lib_cv_;
  std::map<std::string, Json> metas_;
  std::string running_;
  bool cancel_running_ = false, delete_running_ = false, stop_ = false;
  std::thread worker_;
  // live recordings
  std::mutex live_mu_;
  std::condition_variable live_cv_;
  std::map<std::string, std::shared_ptr<Live>> lives_;
  std::deque<LiveTask> live_q_;
  bool stop_live_ = false;
  std::thread live_worker_;
};

}  // namespace srvwh

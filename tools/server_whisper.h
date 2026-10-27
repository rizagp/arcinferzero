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
#include "server_files.h"
#include "../src/app_dir.h"
#include "../src/image.h"
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
#include <set>
#include <sstream>
#include <thread>
#include <tuple>

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

  // ---------------------------------------------------------------- audio in a chat
  // A chat request's audio parts as text, for a model that does not hear audio (or when the request says "speech":
  // "whisper"): a file part of an audio file (the web UI's attachments, voice messages) or an input_audio part
  // (base64, a data URL, "/files/<id>"), each transcribed (beam 5, the engine a window at a time) and cached by its
  // bytes, since every turn of the chat sends it again. A transcript of more than kInlineChars characters becomes a
  // text file in the chat's folder, given as that file: the chat reads it as a document (with the file tools, its
  // outline and the parts it needs). False and the reason when a clip cannot be read.
  static constexpr size_t kInlineChars = 12000;
  static bool has_audio_parts(const Json& body) {
    for (const Json& m : body["messages"].items())
      for (const Json& p : m["content"].items()) {
        const std::string ty = p["type"].str();
        if (ty == "input_audio" || ty == "audio") return true;
        if (ty == "file" || ty == "input_file") {
          std::string id = p[ty == "file" ? "file" : "input_file"]["file_id"].str(p["file_id"].str());
          if (id.rfind("/files/", 0) == 0) id = id.substr(7);
          if (id.rfind("file-", 0) == 0) id = id.substr(5);
          files::Info i;
          if (files::valid_id(id, 16) && files::store().get(id, i) && i.kind == "audio") return true;
        }
      }
    return false;
  }
  // (status: 400, or 429 when too many transcriptions wait; gone: the client left, the clip's transcription stops)
  bool audio_to_text(Json& body, std::string& bad, int& status, const std::function<bool()>& gone = nullptr) {
    status = 400;
    Json msgs = Json::array();
    for (const Json& m : body["messages"].items()) {
      if (!m["content"].is_array()) { msgs.push(m); continue; }
      Json mm = Json::object(), parts = Json::array();
      for (const auto& [k, v] : m.entries())
        if (k != "content") mm.set(k, v);
      for (const Json& p : m["content"].items()) {
        const std::string ty = p["type"].str();
        std::string data, name = "audio", chat;
        bool audio = false;
        try {
          if (ty == "file" || ty == "input_file") {
            std::string fid = p[ty == "file" ? "file" : "input_file"]["file_id"].str(p["file_id"].str());
            if (fid.rfind("/files/", 0) == 0) fid = fid.substr(7);
            if (fid.rfind("file-", 0) == 0) fid = fid.substr(5);
            files::Info i;
            if (files::valid_id(fid, 16) && files::store().get(fid, i) && i.kind == "audio") {
              // (unreadable: an error, not the part left as audio: dispatch would find it again and again)
              if (!files::Store::read(i.path, data)) {
                bad = "audio " + i.name + ": the file cannot be read";
                return false;
              }
              audio = true;
              name = i.name;
              chat = i.chat;
            }
          } else if (ty == "input_audio" || ty == "audio") {
            const Json& a = p[ty == "audio" ? "audio" : "input_audio"];
            const std::string d = a.is_string() ? a.str() : a["data"].str();
            if (!files::store().resolve(d, data)) {
              const std::vector<uint8_t> b = d.rfind("data:", 0) == 0 ? img::from_data_url(d) : img::base64_decode(d);
              data.assign(b.begin(), b.end());
            }
            audio = true;
          }
        } catch (const std::exception& e) {
          bad = std::string("audio: ") + e.what();
          return false;
        }
        if (!audio) { parts.push(p); continue; }
        double secs = 0;
        std::string text;
        try {
          text = clip_text(data, secs, status, gone);
        } catch (const std::exception& e) {
          bad = "audio " + name + ": " + e.what();
          return false;
        }
        const int s = (int)std::lround(secs);
        char len[32];
        snprintf(len, sizeof len, "%d:%02d", s / 60, s % 60);
        Json t = Json::object();
        if (text.size() > kInlineChars) {  // (a long one: as a document)
          std::string stem = name;
          const size_t dot = stem.rfind('.');
          if (dot != std::string::npos && dot > 0) stem.resize(dot);
          const files::Info f = files::store().put(stem + " (transcript).txt", "Transcript of " + name + " (" + len + ", Whisper):\n\n" + text, chat);
          Json fr = Json::object();
          fr.set("file_id", f.id);
          fr.set("filename", f.name);
          t.set("type", "file");
          t.set("file", fr);
        } else {
          t.set("type", "text");
          // (ended by a blank line: the template joins a message's text parts as they are, and the question after it
          //  ran into the transcript's last word)
          t.set("text", "[audio: " + name + ", " + len + ", written by Whisper]\n" +
                            (text.empty() ? std::string("(no speech)") : text) + "\n[end of the audio]\n\n");
        }
        parts.push(t);
      }
      mm.set("content", parts);
      msgs.push(mm);
    }
    body.set("messages", msgs);
    return true;
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
  //   POST   /v1/transcripts/import                {path, name, kind, tracks, speakers}: a finished recording of
  //                                                 arc-recorder (<data>\recordings\), from this computer only
  void start_library(const std::filesystem::path& dir) {
    namespace fs = std::filesystem;
    lib_ = dir;
    std::error_code ec;
    fs::create_directories(lib_, ec);
    std::vector<std::string> sweep;
    for (const auto& e : fs::directory_iterator(lib_, ec)) {
      const std::string tid = e.path().filename().string();
      if (!valid_id(tid) || !e.is_directory()) continue;
      Json m;
      if (!Json::try_parse(slurp(e.path() / "meta.json"), m) || !m.is_object()) {
        // (no record: only an empty folder of the server's own naming goes, one whose delete got as far as its
        //  meta.json; any other folder here is the user's (transcribez writes into this folder too) and stays)
        if (own_id(tid) && fs::is_empty(e.path(), ec)) sweep.push_back(tid);
        continue;
      }
      const std::string st = m["status"].str();
      if (st == "deleting") {  // (deleted, but a file of it was held open, or the server stopped first)
        sweep.push_back(tid);
        continue;
      }
      if (m["ext"].str() == "mp3" && m["recording"].as_bool()) fs::remove(e.path() / "source.wav", ec);  // (a WAV a player held)
      if (st == "uploading") {  // (the server stopped during the upload: the browser has let the file go)
        m.set("status", "failed");
        m.set("error", "the upload did not finish before the server stopped");
        fs::remove(e.path() / ("source." + m["ext"].str()), ec);
      } else if (st == "running") {
        m.set("status", "queued");
        m.set("progress", 0.0);
      } else if (st == "live") {  // (a recording the stop cut: its WAV sizes put right)
        const fs::path w = e.path() / "source.wav";
        fix_wav_sizes(w);
        m.set("status", "queued");
        m.set("size", (double)fs::file_size(w, ec));
      }
      metas_[tid] = m;
      save_meta(tid);
    }
    for (const std::string& tid : sweep) remove_folder(tid);
    // arc-recorder's finished recordings the server did not take yet (it was not running when they ended), and those
    // whose recorder ended without finishing them (killed, a shutdown, the power cut: its note says complete false;
    // taken once no recorder runs, the WAV's sizes put right)
    for (const auto& e : fs::directory_iterator(fs::path(app_dir()) / "recordings", ec)) {
      if (e.path().extension() != ".json") continue;
      Json side;
      std::string bad;
      if (!Json::try_parse(slurp(e.path()), side) || !side.is_object()) continue;
      fs::path w = e.path();
      w.replace_extension(".wav");
      if (!side["complete"].as_bool()) {
        const DWORD pid = (DWORD)side["pid"].as_number(0);
        HANDLE ev = OpenEventW(SYNCHRONIZE, FALSE, L"Local\\arcinferzero-recorder-stop");
        HANDLE h = pid ? OpenProcess(SYNCHRONIZE, FALSE, pid) : nullptr;
        const bool running = ev && h && WaitForSingleObject(h, 0) == WAIT_TIMEOUT;
        if (h) CloseHandle(h);
        if (ev) CloseHandle(ev);
        if (running) continue;
        fix_wav_sizes(w);
      }
      side.set("path", w.string());
      if (import_recording(side, bad).empty()) fprintf(stderr, "whisper: recording %s: %s\n", w.string().c_str(), bad.c_str());
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
      const std::string clean = clean_name(in["name"].str("recording"), 200, "recording");
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
      const std::string nid = new_dir();
      if (nid.empty()) return fail(500, "cannot create the transcript's folder");
      m.set("id", nid);
      m.set("name", clean);
      m.set("ext", ext);
      m.set("size", size);
      m.set("received", 0.0);
      m.set("status", "uploading");
      m.set("progress", 0.0);
      m.set("options", o);
      if (in["kind"].str() == "meeting") m.set("kind", "meeting");
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
    if (rq.path == "/v1/transcripts/import" && rq.method == "POST") {  // (arc-recorder's recordings)
      if (!rq.local_peer) return fail(403, "only from this computer");
      Json in;
      if (!Json::try_parse(rq.body, in) || !in.is_object()) return fail(400, "a JSON object expected");
      std::string bad;
      const std::string nid = import_recording(in, bad);
      if (nid.empty()) return fail(400, bad);
      std::lock_guard<std::mutex> lk(lib_mu_);
      return rs.send(200, "application/json", metas_[nid].dump());
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
      double got = 0;
      {
        std::lock_guard<std::mutex> lk(lib_mu_);
        auto it = metas_.find(tid);
        if (it == metas_.end()) return fail(404, "no transcript " + tid);
        const Json& cur = it->second;
        if (cur["status"].str() != "uploading") return fail(409, "the file is complete");
        if (off != cur["received"].as_number()) return fail(409, "offset " + std::to_string((uint64_t)off) +
                                                                 ": the next piece starts at " +
                                                                 std::to_string((uint64_t)cur["received"].as_number()));
        got = cur["received"].as_number() + (double)rq.body.size();
        if (got > cur["size"].as_number()) return fail(400, "more bytes than the size given");
        if (!writing_.insert(tid).second) return fail(409, "a piece of this file is being written");
      }
      // (written outside the lock: a piece of up to 256 MB, while the list, the marks and the live recordings go on;
      //  the file cut back to the bytes received first, so a piece that failed halfway leaves nothing behind)
      bool ok = false;
      {
        std::error_code ec;
        fs::resize_file(src, (uint64_t)off, ec);
        if (!ec) {
          std::ofstream f(src, std::ios::binary | std::ios::app);
          f.write(rq.body.data(), (std::streamsize)rq.body.size());
          f.close();
          ok = (bool)f;
          if (!ok) fs::resize_file(src, (uint64_t)off, ec);
        }
      }
      std::lock_guard<std::mutex> lk(lib_mu_);
      writing_.erase(tid);
      auto it = metas_.find(tid);
      if (it == metas_.end()) return fail(404, "no transcript " + tid);
      if (it->second["status"].str() == "deleting") {  // (deleted while the piece was written: its folder goes now)
        metas_.erase(it);
        remove_folder(tid);
        return fail(404, "no transcript " + tid);
      }
      if (!ok) return fail(500, "cannot write the file (disk full?)");
      Json& cur = it->second;
      cur.set("received", got);
      // (cancelled while the piece was written: it stays cancelled, not queued)
      if (got == cur["size"].as_number() && cur["status"].str() == "uploading") {
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
      // (found again, not metas_[tid]: a delete meanwhile left nothing to rename, and [] would have made an empty one)
      auto it = metas_.find(tid);
      if (it == metas_.end()) return fail(404, "no transcript " + tid);
      if (in["name"].is_string() && !in["name"].str().empty()) it->second.set("name", clean_name(in["name"].str(), 200, "recording"));
      if (in["summary"].is_string()) it->second.set("summarized", (double)std::time(nullptr));
      save_meta(tid);
      return rs.send(200, "application/json", it->second.dump());
    }
    if (what == "transcript" && rq.method == "PUT") {
      Json in;
      if (!Json::try_parse(rq.body, in) || !in.is_object() || !in["segments"].is_array())
        return fail(400, "a transcript's full JSON ({\"duration\", \"segments\": [...]}) expected");
      if (m["status"].str() == "running" || m["status"].str() == "queued" || m["status"].str() == "uploading")
        return fail(409, "the transcript is not finished yet");
      const whisper::Result r = result_from_json(in);
      std::lock_guard<std::mutex> lk(lib_mu_);  // (the check and the write together: a delete or a pass in between)
      auto it = metas_.find(tid);
      if (it == metas_.end() || it->second["status"].str() == "deleting") return fail(404, "no transcript " + tid);
      const std::string st = it->second["status"].str();
      if (st == "running" || st == "queued" || st == "uploading" || st == "live") return fail(409, "the transcript is not finished yet");
      if (!write_atomic(dir / "transcript.json", whisper::to_json(r))) return fail(500, "cannot write the transcript");
      it->second.set("edited", (double)std::time(nullptr));
      it->second.set("segments", (double)r.segments.size());
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
        ascii += (c >= 0x20 && c < 0x7F && c != ';' && c != '"' && c != '\\') ? (char)c : '_';
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
      auto it = metas_.find(tid);
      if (it == metas_.end()) return fail(404, "no transcript " + tid);
      Json& cur = it->second;
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
      auto it = metas_.find(tid);
      if (it == metas_.end()) return fail(404, "no transcript " + tid);
      const std::string st = it->second["status"].str();
      // (marked first, on disk too: a folder that cannot go now, a file of it held open, goes at the next start)
      it->second.set("status", "deleting");
      save_meta(tid);
      if (tid == running_) {  // (the worker deletes it when it lets go of it)
        cancel_running_ = delete_running_ = true;
      } else if (st == "live") {  // (and so does the live recording's end)
        if (auto s = live_of(tid)) live_stop(s);
      } else if (!writing_.count(tid)) {  // (a piece being written: its upload deletes it after)
        metas_.erase(it);
        lk.unlock();
        remove_folder(tid);
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
  //   POST /v1/live/<id>/audio?since=N&offset=F  the next piece (at most 120 s; F: the frames sent before it, so a
  //                                     piece sent again is not written twice) -> {segments: the draft's from the
  //                                     N-th on, count, heard_s, pending: utterances not transcribed yet, behind: the
  //                                     draft left some out (the accurate pass has them)}. No audio for 5 minutes: the
  //                                     recording is stopped (the page is gone) and transcribed
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
        const std::string w = clean_name(in["speakers"][k].str(), 40, tracks == 1 ? "" : k == 0 ? "Me" : "Others");
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
      const std::string clean = clean_name(in["name"].str("recording"), 200, "recording");
      const std::string nid = new_dir();
      if (nid.empty()) return fail(500, "cannot create the recording's folder");
      s->id = nid;
      s->wav.open(lib_ / nid / "source.wav", std::ios::binary | std::ios::trunc);
      if (!s->wav) {
        std::error_code ec;
        fs::remove_all(lib_ / nid, ec);
        return fail(500, "cannot create the recording");
      }
      write_wav_header(*s);
      Json m = Json::object();
      m.set("id", nid);
      m.set("name", clean);
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
      // (a page sends half a second at a time; a background tab's timers, a minute or so; more is not a recording)
      if (rq.body.size() > (size_t)kLiveMaxPiece * s->rate * frame)
        return fail(413, "at most " + std::to_string(kLiveMaxPiece) + " s of audio a request");
      std::vector<LiveTask> todo;
      Json out = Json::object();
      {
        std::lock_guard<std::mutex> lk(live_mu_);
        if (s->stopped) return fail(409, "the recording has stopped");
        s->last = std::chrono::steady_clock::now();
        // (offset: the frames the page had sent before this piece. A piece sent again, whose answer was lost after it
        //  was written, starts before the end of the recording: what it has already is left out)
        size_t skip = 0;
        const std::string offs = query_param(rq.query, "offset");
        if (!offs.empty()) {
          const uint64_t o = (uint64_t)std::max(0.0, atof(offs.c_str()));
          if (o < s->frames) skip = (size_t)std::min<uint64_t>(s->frames - o, rq.body.size() / frame) * frame;
        }
        s->wav.write(rq.body.data() + skip, (std::streamsize)(rq.body.size() - skip));
        if (!s->wav) return fail(500, "cannot write the recording (disk full?)");
        const size_t n = (rq.body.size() - skip) / frame;
        s->frames += n;
        write_wav_header(*s);  // (sizes kept right as it grows: a crash loses nothing written)
        std::vector<float> x(n);
        for (int k = 0; k < s->tracks; ++k) {
          for (size_t i = 0; i < n; ++i) {
            int16_t v;
            memcpy(&v, rq.body.data() + skip + i * frame + 2 * k, 2);
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
        // (the draft behind by more than kLiveMaxPending utterances: the new ones left out of it, and no partial text;
        //  the recording keeps every sample and the accurate pass after the stop writes them all. Without the cap a
        //  client could queue hours of utterances, and live turns go first: every file job would wait for them)
        if (s->pending >= kLiveMaxPending) {
          for (const LiveTask& u : todo)
            if (u.partial) s->t[u.track].partial_busy = false;
          todo.clear();
          s->skipped = true;
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
        if (s->skipped) out.set("behind", true);
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
      auto it = metas_.find(tid);
      if (it == metas_.end()) return fail(404, "no transcript " + tid);
      Json marks = it->second["marks"];
      if (marks.size() >= 10000) return fail(400, "at most 10,000 marks");
      marks.push(in["t"].as_number((double)s->frames / s->rate));
      it->second.set("marks", marks);
      save_meta(tid);
      return rs.send(200, "application/json", it->second.dump());
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
    bool stopped = false, skipped = false;  // (skipped: utterances left out of the draft, it fell too far behind)
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();  // (the last audio)
  };
  static constexpr int kLiveMaxPiece = 120, kLiveMaxPending = 64;
  // the library's files: at most 6 hours (a recording runs a little past the length it was given). The whole file is
  // held while it is transcribed: 6 hours of 16-kHz samples are 1.4 GB a track, plus a copy and the log-mel (~1 GB)
  // of the track being transcribed: a 2-track recording of 6 hours needs ~5 GB of RAM, and fails without it
  static constexpr double kLibraryMaxSeconds = 6 * 3600;
  static constexpr int kLiveIdle = 300;  // (seconds without audio: the page is gone; the recording is stopped)
  struct LiveTask {
    std::shared_ptr<Live> s;
    int track = 0;
    int64_t from = 0;
    std::vector<float> x;
    bool finish = false, partial = false;
  };
  // a clip's text (beam 5, word timestamps off), from the cache when the same bytes came before (the 64 clips used
  // last kept). One clip decoded and transcribed at a time, as the OpenAI endpoint's requests (its waiting cap too:
  // status 429 past it); a client that left stops its clip
  using ClipKey = std::tuple<uint64_t, uint64_t, uint64_t>;  // (length, two FNV-1a hashes of the bytes)
  bool clip_cached(const ClipKey& key, std::string& text, double& secs) {
    std::lock_guard<std::mutex> lk(clip_mu_);
    auto it = clips_.find(key);
    if (it == clips_.end()) return false;
    it->second.used = ++clip_tick_;
    text = it->second.text;
    secs = it->second.secs;
    return true;
  }
  std::string clip_text(const std::string& data, double& secs, int& status, const std::function<bool()>& gone) {
    uint64_t h1 = 1469598103934665603ull, h2 = 0x9E3779B97F4A7C15ull;
    for (unsigned char b : data) {
      h1 = (h1 ^ b) * 1099511628211ull;
      h2 = (h2 ^ (b + 0x5Bu)) * 0x100000001B3ull + 0x7F4A7C15u;
    }
    const ClipKey key{data.size(), h1, h2};
    std::string text;
    if (clip_cached(key, text, secs)) return text;
    struct Count { std::atomic<int>& n; int before; Count(std::atomic<int>& x) : n(x), before(n++) {} ~Count() { --n; } } waiting{waiting_};
    if (waiting.before >= kMaxWaiting) {
      status = 429;
      throw std::runtime_error("too many transcriptions waiting; try again later");
    }
    std::unique_lock<std::mutex> one(req_mu_);
    if (clip_cached(key, text, secs)) return text;  // (the same clip, transcribed for another request meanwhile)
    if (gone && gone()) throw std::runtime_error("the client left");
    const std::vector<float> a = audio::decode_16k(std::vector<uint8_t>(data.begin(), data.end()), max_seconds);
    whisper::Options o;
    o.window_turn = [this](const std::function<void()>& f) { turn_.run(false, f); };
    LockedBackend be(*eng_, gpu_turn);
    whisper::Transcriber tr(*m_, be);
    bool left = false;
    const whisper::Result r = tr.run(a, o, [&](double, double, const std::vector<whisper::Segment>&) {
      left = gone && gone();
      return !left;
    });
    if (left) throw std::runtime_error("the client left");
    for (const whisper::Segment& g : r.segments) text += g.text;
    text = whisper::trim_copy(text);
    secs = r.duration;
    std::lock_guard<std::mutex> lk(clip_mu_);
    if (clips_.size() >= 64)
      clips_.erase(std::min_element(clips_.begin(), clips_.end(), [](const auto& x, const auto& y) { return x.second.used < y.second.used; }));
    clips_[key] = {text, secs, ++clip_tick_};
    return text;
  }
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
  // the sizes of a WAV this project wrote (a 44-byte header, the data to its end) put right from its length, after a
  // stop that cut it (past 4 GB they stay at their fields' end: wav_format then reads the data to the end of the file)
  static void fix_wav_sizes(const std::filesystem::path& w) {
    std::error_code ec;
    const uint64_t n = std::filesystem::file_size(w, ec);
    std::fstream f(w, std::ios::binary | std::ios::in | std::ios::out);
    char tag[4] = {};
    if (ec || n < 44 || !f || !f.seekg(36).read(tag, 4) || memcmp(tag, "data", 4)) return;  // (not that layout: left)
    const uint32_t data = (uint32_t)std::min<uint64_t>(n - 44, 0xFFFFFFF0u), riff = (uint32_t)std::min<uint64_t>(36ull + data, 0xFFFFFFFFull);
    f.seekp(4);
    f.write((const char*)&riff, 4);
    f.seekp(40);
    f.write((const char*)&data, 4);
  }
  static void write_wav_header(Live& s) {  // (live_mu_ held)
    const uint32_t data = (uint32_t)std::min<uint64_t>(s.frames * 2 * s.tracks, 0xFFFFFFF0u);
    uint8_t h[44];
    auto u32 = [&](int o, uint32_t v) { for (int i = 0; i < 4; ++i) h[o + i] = (uint8_t)(v >> (8 * i)); };
    auto u16 = [&](int o, uint32_t v) { h[o] = (uint8_t)v; h[o + 1] = (uint8_t)(v >> 8); };
    memcpy(h, "RIFF", 4);
    u32(4, (uint32_t)std::min<uint64_t>(36ull + data, 0xFFFFFFFFull));  // (past 4 GB: held at its end, as data)
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
    // (the samples go after the header: the first call, at the file's start, went back to 0, so the audio wrote over
    //  the header and each later call wrote it back over the first 44 bytes of audio)
    const std::streamoff end = std::max<std::streamoff>(44, (std::streamoff)s.wav.tellp());
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
        // (every 15 s with nothing to do: recordings no page sends to any more (closed, crashed, offline) stopped)
        if (!live_cv_.wait_for(lk, std::chrono::seconds(15), [&] { return stop_live_ || !live_q_.empty(); })) {
          std::vector<std::shared_ptr<Live>> idle;
          const auto now = std::chrono::steady_clock::now();
          for (const auto& [k, s] : lives_)
            if (!s->stopped && now - s->last > std::chrono::seconds(kLiveIdle)) idle.push_back(s);
          lk.unlock();
          for (const auto& s : idle) {
            fprintf(stderr, "whisper: live recording %s: no audio for %d s, stopped\n", s->id.c_str(), kLiveIdle);
            live_stop(s);
          }
          continue;
        }
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
      remove_folder(s->id);
      return;
    }
    it->second.set("size", (double)std::filesystem::file_size(lib_ / s->id / "source.wav", ec));
    it->second.set("duration", (double)s->frames / s->rate);
    it->second.set("status", it->second["status"].str() == "cancelled" ? "cancelled" : "queued");
    save_meta(s->id);
    lib_cv_.notify_all();
  }

  // a finished recording of arc-recorder (a WAV in <data>\recordings\, its sidecar <name>.json next to it) moved
  // into the library and queued for the accurate pass (its tracks with their speakers); the new id, or "" and why
  std::string import_recording(const Json& in, std::string& bad) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path rec = fs::weakly_canonical(fs::path(app_dir()) / "recordings", ec);
    const fs::path w = fs::weakly_canonical(fs::path(in["path"].str()), ec);
    if (ec || w.parent_path() != rec || w.extension() != ".wav") {
      bad = "path: a recording in " + rec.string();
      return "";
    }
    const uint64_t size = fs::file_size(w, ec);
    if (ec || size < 44) {
      bad = "no recording at " + w.string();
      return "";
    }
    const int tracks = std::clamp((int)in["tracks"].as_number(1), 1, 2);
    Json sp = Json::array();
    for (int k = 0; k < tracks; ++k) sp.push(clean_name(in["speakers"][k].str(), 40, tracks == 1 ? "" : k == 0 ? "Me" : "Others"));
    const std::string nid = new_dir();
    if (nid.empty()) {
      bad = "cannot create a folder in the library";
      return "";
    }
    fs::rename(w, lib_ / nid / "source.wav", ec);  // (the same drive: no copy)
    if (ec) {
      ec.clear();
      fs::copy_file(w, lib_ / nid / "source.wav", ec);
      if (!ec) fs::remove(w, ec);
    }
    if (!fs::exists(lib_ / nid / "source.wav")) {
      bad = "cannot move the recording into the library";
      fs::remove_all(lib_ / nid, ec);
      return "";
    }
    fs::path side = w;
    side.replace_extension(".json");
    fs::remove(side, ec);
    const std::string name = clean_name(in["name"].str(w.filename().string()), 200, "recording");
    Json m = Json::object();
    m.set("id", nid);
    m.set("name", name);
    m.set("ext", "wav");
    m.set("size", (double)size);
    m.set("received", (double)size);
    m.set("status", "queued");
    m.set("progress", 0.0);
    m.set("options", Json::object());
    m.set("tracks", (double)tracks);
    m.set("speakers", sp);
    m.set("recording", true);
    m.set("mp3_kbps", 192.0);
    if (in["kind"].str() == "meeting") m.set("kind", "meeting");
    std::lock_guard<std::mutex> lk(lib_mu_);
    double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    for (const auto& [k, x] : metas_) now = std::max(now, x["created"].as_number() + 0.001);
    m.set("created", (double)std::llround(now * 1000) / 1000);
    metas_[nid] = m;
    save_meta(nid);
    lib_cv_.notify_all();
    return nid;
  }
  // a name or speaker as typed, without control characters, quotes, back / forward slashes; at most max bytes, cut
  // at a character's start
  static std::string clean_name(const std::string& name, size_t max, const std::string& empty) {
    std::string c;
    for (unsigned char ch : name)
      if (ch >= 0x20 && ch != '"' && ch != '\\' && ch != '/' && ch != 0x7F) c += (char)ch;
    if (c.size() > max) {
      c.resize(max);
      size_t k = c.size();  // (the last character's lead byte at k - 1: dropped when its other bytes were cut off)
      while (k > 0 && ((unsigned char)c[k - 1] & 0xC0) == 0x80) --k;
      if (k > 0) {
        const unsigned char lead = (unsigned char)c[k - 1];
        const size_t need = lead < 0xC0 ? 1 : lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : 2;
        if (c.size() - (k - 1) < need) c.resize(k - 1);
      }
    }
    return c.empty() ? empty : c;
  }
  // an id as new_dir makes them: yyyymmdd-hhmmss-xxxx (before 2026-10-28) or -xxxxxx, x lowercase hex
  static bool own_id(const std::string& t) {
    if (t.size() != 20 && t.size() != 22) return false;
    for (size_t i = 0; i < t.size(); ++i) {
      const char c = t[i];
      const bool ok = i == 8 || i == 15 ? c == '-' : i < 15 ? (c >= '0' && c <= '9') : ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
      if (!ok) return false;
    }
    return true;
  }
  static bool valid_id(const std::string& tid) {
    return tid.size() >= 8 && tid.size() <= 40 && tid.find_first_not_of("0123456789abcdefghijklmnopqrstuvwxyz-") == std::string::npos;
  }
  static std::string slurp(const std::filesystem::path& p) {  // (shared: a file being read can be replaced or deleted)
    std::string s;
    files::read_shared(p, s);
    return s;
  }
  // written whole or not at all (a crash in the middle leaves the old file); a failure is logged
  static bool write_atomic(const std::filesystem::path& p, const std::string& s) {
    if (files::write_shared(p, s)) return true;
    fprintf(stderr, "whisper: cannot write %s\n", p.string().c_str());
    return false;
  }
  // a new transcript's id with its folder, made here: a folder that is there already is never taken (several ids in
  // one second, the recordings imported at a start, could meet); "" when none can be made
  std::string new_dir() {
    static std::mutex mu;
    static std::mt19937 rng(std::random_device{}());
    std::lock_guard<std::mutex> g(mu);
    for (int i = 0; i < 100; ++i) {
      const std::time_t t = std::time(nullptr);
      std::tm tm{};
      localtime_s(&tm, &t);
      char b[40];
      std::strftime(b, sizeof b, "%Y%m%d-%H%M%S", &tm);
      char r[12];
      snprintf(r, sizeof r, "-%06x", (unsigned)(rng() & 0xFFFFFF));
      const std::string nid = std::string(b) + r;
      std::error_code ec;
      if (std::filesystem::create_directory(lib_ / nid, ec)) return nid;
      if (ec) return "";
    }
    return "";
  }
  void save_meta(const std::string& tid) {  // (lib_mu_ held)
    auto it = metas_.find(tid);
    if (it != metas_.end()) write_atomic(lib_ / tid / "meta.json", it->second.dump());
  }
  // a transcript's folder deleted: its files first and meta.json last (a file held open keeps its folder known, by a
  // meta.json that says "deleting", until the next start sweeps it), then the folder; false when something stayed
  bool remove_folder(const std::string& tid) {
    namespace fs = std::filesystem;
    const fs::path dir = lib_ / tid;
    std::error_code ec;
    bool all = true;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
      if (e.path().filename() == "meta.json") continue;
      std::error_code e2;
      fs::remove_all(e.path(), e2);
      if (e2) all = false;
    }
    if (all) fs::remove_all(dir, ec);
    if (!all || ec) fprintf(stderr, "whisper: %s not deleted whole (a file of it is open); the next start deletes it\n", tid.c_str());
    return all && !ec;
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
        remove_folder(tid);
      }
    }
  }
  void transcribe_job(const std::string& tid, const Json& meta) {
    namespace fs = std::filesystem;
    using clk = std::chrono::steady_clock;
    const fs::path dir = lib_ / tid;
    auto finish = [&](const std::string& status, const std::string& error) {
      std::lock_guard<std::mutex> lk(lib_mu_);
      if (delete_running_) return;  // (deleted meanwhile: its "deleting" kept on disk until the folder goes)
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
      const double lim = std::max(max_seconds, kLibraryMaxSeconds);  // (--whisper-max-minutes when it is more)
      if (tracks > 1) audio = audio::decode_16k_wav_channels(src, lim);
      else audio.push_back(audio::decode_16k_file(src, lim));
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
    if (delete_running_) return;  // (deleted meanwhile: nothing written over its "deleting")
    Json& m = metas_[tid];
    if (stop_ && !cancel_running_) {  // (the server is stopping: transcribed again after the restart)
      m.set("status", "queued");
      m.set("progress", 0.0);
      save_meta(tid);
      return;
    }
    if (!write_atomic(dir / "transcript.json", whisper::to_json(r))) {
      m.set("status", "failed");
      m.set("error", "cannot write the transcript (disk full?)");
      save_meta(tid);
      return;
    }
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
    bool ok = audio::encode_mp3(src, dir / "source.mp3", kbps, err);
    if (ok) {  // (the WAV goes only for an MP3 as long as it: a short one would lose the rest for good)
      const double mp3_s = audio::duration_seconds(dir / "source.mp3");
      if (std::fabs(mp3_s - r.duration) > std::max(2.0, 0.002 * r.duration)) {
        ok = false;
        char b[96];
        snprintf(b, sizeof b, "the MP3 is %.1f s long, the recording %.1f s: the WAV is kept", mp3_s, r.duration);
        err = b;
        std::error_code ec;
        fs::remove(dir / "source.mp3", ec);
      }
    }
    lk.lock();
    auto it = metas_.find(tid);
    if (it == metas_.end()) return;
    if (ok) {
      // (deleted even while a player reads it: the media is read shared; a WAV that stays is deleted at the next start)
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
  std::set<std::string> writing_;  // (uploads whose piece is being written, outside lib_mu_)
  bool cancel_running_ = false, delete_running_ = false, stop_ = false;
  std::thread worker_;
  // chat audio's texts, by the clip's bytes
  struct Clip {
    std::string text;
    double secs = 0;
    uint64_t used = 0;  // (the clock of its last use: the oldest goes first)
  };
  std::mutex clip_mu_;
  std::map<ClipKey, Clip> clips_;
  uint64_t clip_tick_ = 0;
  // live recordings
  std::mutex live_mu_;
  std::condition_variable live_cv_;
  std::map<std::string, std::shared_ptr<Live>> lives_;
  std::deque<LiveTask> live_q_;
  bool stop_live_ = false;
  std::thread live_worker_;
};

}  // namespace srvwh

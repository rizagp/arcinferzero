// inferzero-server: OpenAI-compatible HTTP server (chat completions with streaming, reasoning and tool calls),
// prefix reuse across requests, and a built-in web UI.
// usage: inferzero-server -m model.gguf [--host 127.0.0.1] [--port 8080] [--ctx 40960] [--alias name] [--webui dir]
//                       [--chunk N] [--budget GB] [--mtp] [--think-budget N] [--attn-f32] [--kernels dir]
//                       [--session-cache-gb N] [--session-dir dir]   (saved conversations; default 0 = off)
// Built as bin\serverz.exe (zero/serverz.cpp includes this file): the inferzero engine, zero/engine.h (Level Zero,
// MoE experts streamed from the SSD), with its data folder %USERPROFILE%\.inferzero\zero (context cache, saved
// prompts and conversations).
#include "../zero/engine.h"
#include "../zero/gguf_pack.h"
#define ARC_DATA "/zero/"  // (under app_dir(), src/app_dir.h)
#define ARC_DATA_HELP "%%USERPROFILE%%\\.inferzero\\zero\\"  // (the same folder, as --help prints it)
#include "../src/tokenizer.h"
#include "../src/chat.h"
#include "../src/chat_gemma4.h"
#include "../src/chat_jinja.h"
#include "../src/image.h"
#include "../src/audio.h"
#include "../src/audio_gemma4.h"
#include "../src/vision_gemma4.h"
#include "../zero/vision_gemma4v.h"
#include "../zero/audio_gemma4a.h"
#include "../zero/vision_qwen3vl.h"
#include "../src/http.h"
#include "../src/full_speed.h"
#include "api_compat.h"
#include "json_constraint.h"
#include "tool_constraint.h"
#include "sampler.h"
#include "web_tools.h"
#include "web_github.h"
#include "web_view.h"
#include "server_files.h"
#include "server_memory.h"
#include "server_projects.h"
#include "pdf_render.h"  // (PDF pages as images, for OCR models: server_ocr.h)
#include "doc_writer.h"
#include "doc_template.h"
#include "xlsx.h"
#include "server_terminal.h"
#include "server_remote.h"
#include "server_metrics.h"  // (GET /metrics: Prometheus, vLLM's names)
#include "server_startup.h"  // (the log's first lines)
#include "../src/crash_log.h"  // (a crash names its stack in the log)
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <random>
#include <sstream>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

// What a model takes as input, in the fields clients look for: llama-server ("modalities", and "capabilities" with
// "multimodal" as in its Ollama-style list), Ollama ("vision"), OpenRouter ("architecture.input_modalities"),
// LM Studio ("type": "vlm"). Only models whose image encoder inferzero runs are marked (Gemma 4 12B, gemma4uv).
// llama-server's own model list next to "data": a "models" array (Ollama style) whose "capabilities" hold
// "multimodal" for a model with an image encoder. Clients that treat inferzero as a llama.cpp server read it there.
// bytes: model id -> its file size ("size"; 0 when not known).
// The served context window in the fields clients read: llama-server's "meta" (n_ctx: the window served, n_ctx_train:
// the trained one; Unsloth Studio takes meta.n_ctx) and vLLM's "max_model_len". served 0: not loaded, not known yet.
static void set_window(Json& m, int64_t served, int64_t trained, int64_t n_vocab = 0, int64_t n_embd = 0) {
  Json meta = Json::object();
  if (served > 0) meta.set("n_ctx", (double)served);
  if (trained > 0) meta.set("n_ctx_train", (double)trained);
  if (n_vocab > 0) meta.set("n_vocab", (double)n_vocab);
  if (n_embd > 0) meta.set("n_embd", (double)n_embd);
  m.set("meta", meta);
  if (served > 0) m.set("max_model_len", (double)served);
}

static Json llama_models(const Json& data, const std::map<std::string, uint64_t>& bytes) {
  Json out = Json::array();
  for (const Json& m : data.items()) {
    Json e = Json::object();
    const std::string id = m["id"].str();
    e.set("name", id);
    e.set("model", id);
    e.set("modified_at", "");
    const auto b = bytes.find(id);
    e.set("size", (double)(b != bytes.end() ? b->second : 0));
    e.set("digest", "");
    e.set("type", "model");
    e.set("description", "");
    Json tags = Json::array();
    tags.push("");
    e.set("tags", tags);
    Json caps = Json::array();
    caps.push("completion");
    if (m["modalities"]["vision"].is_bool() && m["modalities"]["vision"].as_bool()) caps.push("multimodal");
    e.set("capabilities", caps);
    e.set("parameters", "");
    Json det = Json::object();
    det.set("parent_model", "");
    det.set("format", "gguf");
    det.set("family", "");
    Json fam = Json::array();
    fam.push("");
    det.set("families", fam);
    det.set("parameter_size", "");
    det.set("quantization_level", "");
    e.set("details", det);
    out.push(e);
  }
  return out;
}

// The bytes of a model file; a split GGUF (name-00001-of-0000N.gguf) counts every part that is present.
static uint64_t gguf_bytes(const std::string& path) {
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path p(path);
  const std::string f = p.filename().string();
  const size_t k = f.find("-00001-of-");
  uint64_t n = fs::file_size(p, ec);
  if (ec) return 0;
  if (k == std::string::npos) return n;
  const int parts = atoi(f.c_str() + k + 10);
  for (int i = 2; i <= parts; ++i) {
    char num[16];
    snprintf(num, sizeof num, "%05d", i);
    const uint64_t b = fs::file_size(p.parent_path() / (f.substr(0, k + 1) + num + f.substr(k + 6)), ec);
    if (!ec) n += b;
  }
  return n;
}

// ---- web_fetch: what a fetched URL is (a page, a text, or a file to save: a document or an image)
static constexpr size_t kWebFileMax = 20u << 20;    // (a document or an image larger than this is refused)
static constexpr size_t kWebAudioMax = 60u << 20;   // (an audio file: ~1 hour of MP3 at 128 kbit/s, 30 min transcribed)
static constexpr double kWebAudioSeconds = 30 * 60;  // (the longest audio from a URL that is transcribed)
static constexpr size_t kWebTextMax = 4u << 20;     // (a page or a text: its first 4 MB, as before files were fetched)
static std::string url_decode(const std::string& s) {
  std::string o;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size() && isxdigit((unsigned char)s[i + 1]) && isxdigit((unsigned char)s[i + 2])) {
      o += (char)strtol(s.substr(i + 1, 2).c_str(), nullptr, 16);
      i += 2;
    } else o += s[i] == '+' ? ' ' : s[i];
  }
  return o;
}
// the file's name: Content-Disposition's (filename*=UTF-8''..., else filename=), else the URL path's last part
static std::string web_file_name(const webt::Page& pg) {
  const std::string d = pg.disposition, dl = webt::lower(d);
  auto value = [&](size_t at) {
    std::string v = d.substr(at);
    if (!v.empty() && v[0] == '"') v = v.substr(1, v.find('"', 1) - 1);
    else v = v.substr(0, v.find(';'));
    return v;
  };
  size_t k = dl.find("filename*=");
  if (k != std::string::npos) {
    std::string v = value(k + 10);
    const size_t q = v.find("''");
    if (q != std::string::npos) v = v.substr(q + 2);
    if (!url_decode(v).empty()) return url_decode(v);
  }
  k = dl.find("filename=");
  if (k != std::string::npos && !value(k + 9).empty()) return value(k + 9);
  std::string p = pg.url.substr(0, pg.url.find_first_of("?#"));
  const size_t se = p.find("://");
  p = se == std::string::npos ? p : p.substr(se + 3);
  const size_t sl = p.find('/');
  p = sl == std::string::npos ? "" : p.substr(sl);
  while (!p.empty() && p.back() == '/') p.pop_back();
  return url_decode(p.substr(p.rfind('/') + 1));
}
// "html", "text", a file's extension to save it under (pdf, docx, xlsx, xlsm, pptx, png, jpg, gif, webp, bmp), or ""
// (refused): by the content type, else (a generic one: application/octet-stream...) by the name's extension; a file's
// first bytes must be of its kind (%PDF, a ZIP for Office, an image header), so no other file is saved under a name
// (the kind from the headers alone: "html", "text", a file's extension or "" refused; web_file_type checks the body,
//  web_fetch's get() takes the size each may have from it before the body is read. One list of kinds for both, the same as
//  kind_of's audio and web_tools.h file_link's: self audit G4)
static std::string web_kind(const std::string& ct_in, const std::string& name) {
  const std::string ct = webt::lower(ct_in.substr(0, ct_in.find(';')));
  static const std::map<std::string, std::string> by_ct = {
      {"application/pdf", "pdf"}, {"application/x-pdf", "pdf"},
      {"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"},
      {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"},
      {"application/vnd.ms-excel.sheet.macroenabled.12", "xlsm"},
      {"application/vnd.openxmlformats-officedocument.presentationml.presentation", "pptx"},
      {"image/png", "png"}, {"image/jpeg", "jpg"}, {"image/jpg", "jpg"}, {"image/gif", "gif"}, {"image/webp", "webp"},
      {"image/bmp", "bmp"}, {"audio/mpeg", "mp3"}, {"audio/mp3", "mp3"}, {"audio/mp4", "m4a"}, {"audio/x-m4a", "m4a"},
      {"audio/m4a", "m4a"}, {"audio/aac", "aac"}, {"audio/x-aac", "aac"}, {"audio/wav", "wav"}, {"audio/x-wav", "wav"},
      {"audio/wave", "wav"}, {"audio/vnd.wave", "wav"}, {"audio/flac", "flac"}, {"audio/x-flac", "flac"},
      {"audio/ogg", "ogg"}, {"application/ogg", "ogg"}, {"audio/opus", "opus"}, {"audio/webm", "webm"},
      {"audio/x-ms-wma", "wma"}};
  static const std::set<std::string> files_ext = {"pdf", "docx", "xlsx", "xlsm", "pptx", "png", "jpg", "jpeg", "gif", "webp", "bmp",
                                                  "mp3", "m4a", "aac", "wav", "flac", "ogg", "opus", "wma", "webm"},
                                     text_ext = {"txt", "md", "log", "csv", "tsv", "json", "jsonl", "xml", "yaml", "yml",
                                                 "toml", "ini", "cfg", "srt", "vtt", "py", "c", "cpp", "h", "hpp", "js",
                                                 "ts", "ps1", "sh", "bat", "cmd", "rs", "go", "java", "cs", "sql", "diff", "patch"};
  const std::string ext = files::ext_of(name);
  if (auto it = by_ct.find(ct); it != by_ct.end()) return it->second;
  if (ct.find("html") != std::string::npos) return "html";
  if (ct.rfind("text/", 0) == 0 || ct.find("json") != std::string::npos || ct.find("xml") != std::string::npos) return "text";
  if (ct.empty() || ct == "application/octet-stream" || ct == "binary/octet-stream" || ct == "application/force-download" ||
      ct == "application/download" || ct == "application/x-download") {
    if (files_ext.count(ext)) return ext == "jpeg" ? "jpg" : ext;
    if (text_ext.count(ext)) return "text";
    if (ct.empty()) return "html";  // (no type, no known extension: read as a page, as before)
  }
  return "";
}
static bool web_audio(const std::string& t) { return files::kind_of("x." + t) == "audio" || t == "webm"; }
static std::string web_file_type(const std::string& ct, const std::string& name, const std::string& body) {
  const std::string t = web_kind(ct, name);
  if (t.empty() || t == "html") return t;
  if (t == "text") return body.find('\0') == std::string::npos || webt::lower(ct).rfind("text/", 0) == 0 ? t : "";
  // its first bytes
  const bool pdf = body.compare(0, 5, "%PDF-") == 0 || body.substr(0, 1024).find("%PDF-") != std::string::npos,
             zip = body.compare(0, 4, "PK\x03\x04", 4) == 0;
  int w = 0, h = 0;
  if (t == "pdf") return pdf ? t : "";
  if (t == "docx" || t == "xlsx" || t == "xlsm" || t == "pptx") return zip ? t : "";
  if (web_audio(t)) {  // (an MP3 / AAC frame or ID3 tag, MP4 ftyp, RIFF WAVE, fLaC, Ogg, WebM, WMA's ASF header)
    const auto b = [&](size_t k) { return k < body.size() ? (unsigned char)body[k] : 0u; };
    // (bytes at k, false past the end: std::string::compare at a position past a 3-byte body throws out_of_range)
    const auto at = [&](size_t k, std::string_view m) { return body.size() >= k + m.size() && body.compare(k, m.size(), m) == 0; };
    const bool ok = at(0, "ID3") || (b(0) == 0xFF && (b(1) & 0xE0) == 0xE0) || at(4, "ftyp") || (at(0, "RIFF") && at(8, "WAVE")) ||
                    at(0, "fLaC") || at(0, "OggS") || at(0, std::string_view("\x1A\x45\xDF\xA3", 4)) ||
                    at(0, std::string_view("\x30\x26\xB2\x75\x8E\x66\xCF\x11", 8));
    return ok ? t : "";
  }
  return files::image_size(body, w, h) ? t : "";
}

// audio: a Gemma 4 audio encoder (mmproj audio projector gemma4ua or gemma4a)
static void set_modalities(Json& j, bool vision, bool audio) {
  Json caps = Json::array();
  caps.push("completion");
  caps.push("tools");
  if (vision) {
    caps.push("vision");
    caps.push("multimodal");
  }
  j.set("capabilities", caps);
  Json in = Json::array();
  in.push("text");
  if (vision) in.push("image");
  if (audio) in.push("audio");
  Json out = Json::array();
  out.push("text");
  Json arch = Json::object();
  arch.set("input_modalities", in);
  arch.set("output_modalities", out);
  j.set("architecture", arch);
  Json mod = Json::object();
  mod.set("vision", vision);
  mod.set("audio", audio);
  j.set("modalities", mod);
  j.set("type", vision ? "vlm" : "llm");
}

class Server {
 public:
  ~Server() {  // background writes (saved conversations, system prompt) finish first
    for (auto& s : saved_) written(s);
    prefix_written();
  }
  Server(Engine& e, Tokenizer& t, std::string id, std::string webui) : eng_(e), tok_(t), model_id_(std::move(id)), webui_(std::move(webui)) {
    const auto& g = e.gguf();
    q38_ = g.get_str("tokenizer.chat_template").find("reasoning_effort") != std::string::npos;
    can_preserve_ = !q38_ && g.get_str("tokenizer.chat_template").find("preserve_thinking") != std::string::npos;
    defaults_.temperature = (float)g.get_float("general.sampling.temp", 1.0);
    defaults_.top_p = (float)g.get_float("general.sampling.top_p", 0.95);
    defaults_.top_k = (int)g.get_int("general.sampling.top_k", 20);
    defaults_.min_p = (float)g.get_float("general.sampling.min_p", 0.0);
    ocr_ = ocr_model(g, model_id_);  // (tools/server_ocr.h)
    if (ocr_) zlog::eprintf("OCR mode: requests with an image get this model's own task prompts (tools/server_ocr.h)\n");
    gemma_ = e.hp.gemma;
    rings_ = e.hp.swa_rings;
    if (gemma_) {
      // Gemma 4: the GGUF template is rendered with minja; turns are <|turn>role ... <turn|>
      jinja_ = std::make_unique<JinjaChat>(g.get_str("tokenizer.chat_template"), "<bos>", "<eos>");
      im_start_ = tok_.token("<|turn>");
      for (const char* s : {"<turn|>", "<eos>", "<|tool_response>"}) {
        const int tid = tok_.token(s);
        if (tid >= 0) stop_ids_.push_back(tid);
      }
      think_close_ = "<channel|>";
    } else {
      im_start_ = tok_.token("<|im_start|>");
      for (const char* s : {"<|im_end|>", "<|endoftext|>"}) {
        const int tid = tok_.token(s);
        if (tid >= 0) stop_ids_.push_back(tid);
      }
    }
    if (tok_.eos >= 0) stop_ids_.push_back(tok_.eos);
    slots_.resize(e.slot_count());
  }

  bool has_vision() const { return vision_ != nullptr || qvision_ != nullptr; }
  bool has_audio() const { return audio_ != nullptr && aud_tok_ >= 0; }  // (what a chat request with audio needs)
  uint64_t model_bytes_ = 0;  // the model file's size (/v1/models "models": "size"), set by load_model
  Json whisper_props_;        // /props "whisper" (the speech model next to this one), set by load_model
  size_t speech_reserve_ = 0; // /props "speech_reserve_bytes": GPU memory --ctx auto keeps for speech to text now (so.whisper_bytes;
                              // zero-server.ps1 passes it to --list-models for Pi's other models)
  // f with the engine lock, as a request's step holds it: another engine of the process (Whisper) shares the GPU
  // runtime's command list with this one
  void with_engine(const std::function<void()>& f) {
    std::unique_lock<std::mutex> lk(mu_, std::defer_lock);
    acquire(lk);
    f();
  }
  // The messages with the special tokens written in the contents of user and tool messages made plain text
  // (Tokenizer::defuse_specials): the prompt is tokenized with special tokens recognized, for the template's own
  // markup, so a user's text, a fetched web page or a document read by a tool could otherwise write
  // "<|im_start|>system", end a tool response, or add an image placeholder. System and assistant messages are the
  // client's and the model's own, as they are. Messages without such text are returned unchanged (the same tokens).
  Json defuse_contents(const Json& msgs) const {
    if (!msgs.is_array()) return msgs;
    Json out = Json::array();
    for (const Json& m : msgs.items()) {
      const std::string role = m["role"].str();
      if (role != "user" && role != "tool" && role != "function") {
        out.push(m);
        continue;
      }
      Json m2 = m;
      const Json& c = m["content"];
      if (c.is_string()) {
        m2.set("content", tok_.defuse_specials(c.as_string()));
      } else if (c.is_array()) {
        Json parts = Json::array();
        for (const Json& p : c.items()) {
          if (p["text"].is_string()) {
            Json p2 = p;
            p2.set("text", tok_.defuse_specials(p["text"].as_string()));
            parts.push(p2);
          } else {
            parts.push(p);
          }
        }
        m2.set("content", parts);
      }
      out.push(m2);
    }
    return out;
  }
  // The prompt tokens of a chat request (Anthropic's count_tokens): its rendered prompt, each image or audio part
  // counted as its placeholder token. No engine lock: rendering and tokenizing use no engine state (chat() does both
  // before taking it). -1: the template could not render the messages. A prompt that cannot fit the context: the
  // count it has at least (more than the context), not tokenized (as chat() refuses it).
  int count_tokens(const Json& body) {
    ChatOptions co;
    if (body["enable_thinking"].is_bool()) co.enable_thinking = body["enable_thinking"].as_bool();
    std::string prompt;
    try {
      const Json msgs = defuse_contents(body["messages"]);
      prompt = jinja_ ? jinja_->render(gemma_ ? gemma_history(msgs) : msgs, body["tools"], true, co.enable_thinking)
                      : chat::render(msgs, body["tools"], co, q38_);
    } catch (const std::exception&) {
      return -1;
    }
    if (const size_t L = tok_.max_token_bytes(); L && prompt.size() / L > (size_t)eng_.ctx)
      return (int)std::min<size_t>(prompt.size() / L, INT_MAX);
    return (int)tok_.encode(prompt, true).size();
  }
  void handle(const HttpRequest& rq, HttpResponse& rs) {
    if (rq.method == "GET" && (rq.path == "/" || rq.path == "/index.html")) return serve_ui(rs);
    if (rq.path == "/health") return rs.send(200, "application/json", "{\"status\":\"ok\"}");
    if (rq.path == "/v1/models" || rq.path == "/models") {
      Json m = Json::object();
      m.set("id", model_id_);
      m.set("object", "model");
      m.set("owned_by", "inferzero");
      m.set("context_length", eng_.ctx);
      set_window(m, eng_.ctx, eng_.gguf().get_int(eng_.gguf().get_str("general.architecture") + ".context_length", 0),
                 eng_.hp.n_vocab, eng_.hp.n_embd);
      set_modalities(m, has_vision(), has_audio());
      Json d = Json::array();
      d.push(m);
      Json r = Json::object();
      r.set("object", "list");
      r.set("models", llama_models(d, {{model_id_, model_bytes_}}));
      r.set("data", d);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path == "/sessions" || (rq.path == "/sessions/save" && rq.method == "POST"))
      return sessions(rq, rs);  // GET: size of the session cache; DELETE: clear it; POST /sessions/save: keep the held one
    if (rq.path == "/props") {
      Json r = Json::object();
      r.set("model", model_id_);
      r.set("n_ctx", eng_.ctx);
      r.set("engine", "inferzero");
      r.set("chunk", eng_.max_chunk);
      r.set("weights", eng_.w8_resident ? "w8-resident" : "w8-just-in-time");
      r.set("vision", has_vision());
      r.set("ocr", ocr_);  // (OCR mode: the web UI shows its tasks)
      Json mod = Json::object();  // (llama-server's form)
      mod.set("vision", has_vision());
      mod.set("audio", has_audio());
      r.set("modalities", mod);
      r.set("audio", has_audio());
      if (!whisper_props_.is_null()) r.set("whisper", whisper_props_);
      r.set("speech_reserve_bytes", (double)speech_reserve_);
      r.set("mtp", eng_.has_mtp);
      r.set("mtp_drafts", eng_.opt.mtp_drafts);
      {  // (the web UI's controls follow the model: its thinking switch, Qwen3.8's effort levels, the GGUF's sampling)
        const std::string& tpl = eng_.gguf().get_str("tokenizer.chat_template");
        r.set("thinking", tpl.empty() || tpl.find("enable_thinking") != std::string::npos);
        Json eff = Json::array();
        if (q38_)
          for (const char* e : {"low", "medium", "xhigh"}) eff.push(e);
        r.set("reasoning_effort", eff);
        Json smp = Json::object();
        smp.set("temperature", (double)defaults_.temperature);
        smp.set("top_p", (double)defaults_.top_p);
        smp.set("top_k", (double)defaults_.top_k);
        smp.set("min_p", (double)defaults_.min_p);
        r.set("sampling", smp);
        r.set("think_budget", default_think_budget_);
      }
      {  // GPU memory now (bytes): parts of the loaded model, encoders loaded since (on first use)
        const Engine::MemParts mp = eng_.mem_parts(eng_.ctx);
        Json mem = Json::object();
        mem.set("device", (double)eng_.device_mem());
        mem.set("total", (double)eng_.gpu_bytes);
        mem.set("weights", mp.weights);
        mem.set("mtp", mp.mtp);
        mem.set("buffers", mp.buffers);
        mem.set("context", mp.context);
        mem.set("encoders", (double)(eng_.gpu_bytes - eng_.loaded_bytes()));
        r.set("memory", mem);
      }
      {
        std::lock_guard<std::mutex> lk(stats_mu_);
        Json st = Json::object();
        st.set("requests", n_req_);
        st.set("busy", busy_.load() > 0);
        st.set("position", eng_.pos);
        st.set("slots", (int)slots_.size());
        if (!last_timings_.is_null()) st.set("last", last_timings_);
        r.set("stats", st);
      }
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path == "/tokenize" && rq.method == "POST") {
      Json b;
      if (!Json::try_parse(rq.body, b)) return rs.send(400, "application/json", "{\"error\":{\"message\":\"invalid JSON body\"}}");
      // (as chat(): text longer than the context's tokens could hold is refused before the tokenizer allocates for it)
      const std::string text = b["content"].str();
      if (const size_t L = tok_.max_token_bytes(); L && text.size() / L > (size_t)eng_.ctx)
        return rs.send(400, "application/json", err("content longer than the context (" + std::to_string(eng_.ctx) + " tokens)").dump());
      Json ids = Json::array();
      for (int id : tok_.encode(text, true)) ids.push(id);
      Json r = Json::object();
      r.set("tokens", ids);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path == "/v1/chat/completions" && rq.method == "POST") return chat(rq, rs);
  }

 private:
  Engine& eng_;
  Tokenizer& tok_;
  std::string model_id_, webui_;
  bool q38_ = false, gemma_ = false;
  bool rings_ = false;  // (ring caches, no recurrent state: a prefix is reused by rewind(), not snapshots)

 public:
  void set_vision(std::unique_ptr<GemmaImageEncoder> v) {
    vision_ = std::move(v);
    img_tok_ = tok_.token("<|image|>");
    img_beg_ = tok_.token("<|image>");
    img_end_ = tok_.token("<image|>");
  }
  // Qwen3-VL (Qwen3.5 / 3.6 / 3.8 mmproj): the template already writes <|vision_start|> ... <|vision_end|> around
  // each <|image_pad|>, which becomes the image's embedding rows (causal, M-RoPE positions)
  void set_qvision(std::unique_ptr<VisionQwen3VL> v) {
    qvision_ = std::move(v);
    img_tok_ = tok_.token("<|image_pad|>");
    img_beg_ = img_end_ = -1;
  }
  void set_audio(std::unique_ptr<GemmaAudioEncoder> a) {
    audio_ = std::move(a);
    aud_tok_ = tok_.token("<|audio|>");
    aud_beg_ = tok_.token("<|audio>");
    aud_end_ = tok_.token("<audio|>");
  }

 private:
  std::unique_ptr<JinjaChat> jinja_;
  // vision (Gemma 4 mmproj): image token ids and an embedding cache keyed by the image bytes
  std::unique_ptr<GemmaImageEncoder> vision_;  // Gemma 4: 12B (gemma4uv, CPU) or E2B / E4B (gemma4v, GPU)
  std::unique_ptr<VisionQwen3VL> qvision_;
  int img_tok_ = -1, img_beg_ = -1, img_end_ = -1;
  // audio (Gemma 4 mmproj with the unified audio projector): token ids; embeddings share the cache below
  std::unique_ptr<GemmaAudioEncoder> audio_;  // Gemma 4: 12B (gemma4ua, CPU) or E2B / E4B (gemma4a conformer, GPU)
  int aud_tok_ = -1, aud_beg_ = -1, aud_end_ = -1;
  // monitoring (/props "stats"): requests served, one in progress, the last one's timings
  std::mutex stats_mu_;
  std::atomic<int> busy_{0};  // requests running
  int n_req_ = 0;
  Json last_timings_;
  struct MediaEmb {
    std::shared_ptr<std::vector<float>> e;
    int gw = 0, gh = 0;  // merged-token grid (Qwen M-RoPE); 0 for Gemma and audio
  };
  std::map<uint64_t, MediaEmb> img_cache_;  // media embeddings keyed by the file bytes
  size_t img_cache_bytes_ = 0;               // (their floats: the cache is emptied past 512 MB or 64 entries)
  std::string think_close_ = "</think>";
  SamplingParams defaults_;
  int im_start_ = -1;
  std::vector<int> stop_ids_;
  std::mutex mu_;
  std::vector<int> cached_;  // tokens whose state the engine currently holds (positions 0..pos)
  // ---- conversation slots (--slots N, inferzero): the engine holds N conversations (Engine::bind). A request
  // takes a slot for its whole run and the engine lock for each step (a piece of its prompt, a generated token or
  // verification pass). Between two steps it hands the engine to a waiting request whose engine time is behind its
  // own (take_turns), so a request's tokens do not wait behind each 8K-token piece of another's long prompt: the
  // short steps run until they have used as much time. cached_ / cur_saved_ / last_gen_cut_ are the bound slot's.
  struct SlotSrv {
    std::vector<int> cached;
    bool cur_saved = false, busy = false;
    bool stale = false;  // a step of its conversation threw: its engine state is reset when it is next bound
    int last_gen_cut = 0;
    uint64_t used = 0;
    double vtime = 0;  // engine time of its request (ms; a new request starts kNewCredit behind the others)
    clk::time_point since;
  };
  static constexpr double kNewCredit = 10000;  // ms
  std::vector<SlotSrv> slots_;
  int cur_slot_ = 0;
  uint64_t slot_clock_ = 0, turns_ = 0;
  std::condition_variable slot_cv_, turn_cv_;
  std::atomic<int> waiting_{0};  // requests waiting for the engine lock (to start, or for their next step)
  void use_slot(int s) {
    if (s != cur_slot_ || eng_.bound_slot() != s) {  // (the engine can be on another slot after a failed step)
      eng_.bind(s);  // (first: if it throws, the server's fields still match the slot the engine holds)
      if (s != cur_slot_) {
        SlotSrv& o = slots_[cur_slot_];
        o.cached = std::move(cached_);
        o.cur_saved = cur_saved_;
        o.last_gen_cut = last_gen_cut_;
        SlotSrv& n = slots_[s];
        cached_ = std::move(n.cached);
        cur_saved_ = n.cur_saved;
        last_gen_cut_ = n.last_gen_cut;
        cur_slot_ = s;
      }
    }
    if (slots_[s].stale) {
      slots_[s].stale = false;
      wipe_bound();
    }
  }
  // the bound slot's engine state from scratch: no position, no recurrent-state snapshots
  void wipe_bound() {
    try {
      eng_.reset();
      for (int k = 0; k < kSnaps; ++k) eng_.drop_snapshot(k);
    } catch (...) {
    }
  }
  // the engine lock for a request (counted: the request holding it lets this one start)
  std::atomic<int> starting_{0};
  void acquire(std::unique_lock<std::mutex>& lk) {
    ++waiting_;
    ++starting_;
    lk.lock();
    --starting_;
    --waiting_;
    ++turns_;
    turn_cv_.notify_all();
  }
  // between two steps of a request in `slot`: a request starting, or one whose engine time is behind, goes first
  void take_turns(std::unique_lock<std::mutex>& lk, int slot) {
    if (slots_.size() < 2 || waiting_ == 0) return;
    SlotSrv& me = slots_[slot];
    const clk::time_point now = clk::now();
    me.vtime += std::chrono::duration<double, std::milli>(now - me.since).count();
    me.since = now;
    bool behind = starting_ > 0;
    for (int k = 0; k < (int)slots_.size(); ++k)
      behind = behind || (k != slot && slots_[k].busy && slots_[k].vtime < me.vtime);
    if (!behind) return;
    const uint64_t t = turns_;
    ++waiting_;
    turn_cv_.wait(lk, [&] { return turns_ != t; });
    --waiting_;
    ++turns_;
    turn_cv_.notify_all();
    me.since = clk::now();
    use_slot(slot);
  }
  // end of the system prompt (the start of the second turn; Gemma's first turn follows <bos>): every conversation of
  // a client (Pi: its system prompt and tool list, ~4K tokens) shares it
  size_t system_end(const std::vector<int>& P) const {
    size_t i0 = 0;
    while (i0 < P.size() && i0 < 2 && P[i0] != im_start_) ++i0;
    if (i0 + 1 < P.size() && P[i0] == im_start_ && tok_.piece(P[i0 + 1]) == "system")
      for (size_t i = i0 + 1; i < P.size(); ++i)
        if (P[i] == im_start_) return i;
    return 0;
  }
  // A free slot for the request (waits while every slot is in use): the one whose conversation it continues (shares
  // more than the system prompt), else an empty one, else the one used longest ago (its conversation goes to the
  // session cache when the request does not continue it).
  int claim_slot(std::unique_lock<std::mutex>& lk, const std::vector<int>& P) {
    if (slots_.size() < 2) return 0;
    slot_cv_.wait(lk, [&] {
      for (const auto& x : slots_)
        if (!x.busy) return true;
      return false;
    });
    const size_t sys = system_end(P);
    int cont = -1, empty = -1, lru = -1;
    size_t cl = 0;
    for (int k = 0; k < (int)slots_.size(); ++k) {
      if (slots_[k].busy) continue;
      const std::vector<int>& c = k == cur_slot_ ? cached_ : slots_[k].cached;
      const size_t l = common_prefix(P, c);
      if (l > sys + 8 && l > cl) { cont = k; cl = l; }
      if (c.empty() && empty < 0) empty = k;
      if (lru < 0 || slots_[k].used < slots_[lru].used) lru = k;
    }
    const int k = cont >= 0 ? cont : empty >= 0 ? empty : lru;  // (see below)
    double vmin = -1;
    for (const auto& x : slots_)
      if (x.busy && (vmin < 0 || x.vtime < vmin)) vmin = x.vtime;
    use_slot(k);  // (before busy: a slot left busy by a bind that threw would never be free again)
    slots_[k].vtime = std::max(0.0, vmin - kNewCredit);
    slots_[k].since = clk::now();
    slots_[k].busy = true;
    slots_[k].used = ++slot_clock_;
    if (cont < 0) eng_.reset_drafter_choice();  // (another conversation: its own DFlash2 / MTP choice)
    return k;
  }
  // ---- batched generation steps (--slots, Qwen models on inferzero: QUEUE 20.4). A request at a generation step
  // (its next token, with drafts or without) while other requests are generating waits for one of them to reach its
  // own step: that one runs both in one engine pass (Engine::forward_batch / verify_batch: every slot drafts in its
  // own state, one pass verifies all rows) and hands back the rows; each request then samples and settles its drafts
  // as before. The weights are read once for both: 27B two slots 1.45x (drafts) / 1.8x (none) the tokens per second.
  struct GenStep {
    int slot = 0, tok = 0, max_drafts = 0;  // max_drafts 0: one token without drafts
    bool done = false, alone = false;       // done: run by another request; alone: run it yourself
    bool dflash = false, batched = false;
    std::vector<int> drafts;
    std::vector<float> rows;  // logits of the token and of each draft
    std::string failed;       // the pass that ran it threw (its request fails too)
  };
  std::vector<GenStep*> gen_wait_;  // steps waiting for another request to run them
  int gen_active_ = 0;              // requests in their generation loop
  std::condition_variable gen_cv_;
  // one step of the bound slot's request: alone, or with waiting steps of other slots
  void run_steps(std::vector<GenStep*> st) {
    try {
      run_batch(st);
    } catch (const std::exception& e) {
      // every request of the step fails (the waiting ones would otherwise wait for good), and their slots start
      // over: the engine left their positions as they were, but part of their device state may be written
      try { eng_.bind(cur_slot_); } catch (...) {}
      for (GenStep* g : st) {
        g->failed = e.what();
        g->done = true;
        drop_conversation(g->slot);
      }
      gen_cv_.notify_all();
      throw;
    }
    for (GenStep* g : st) {
      g->done = true;
      g->batched = st.size() > 1;
    }
  }
  // the conversation of slot k is gone (a step that threw): it starts over. Its engine state is wiped now when the
  // engine holds k, else when k is next bound (use_slot)
  void drop_conversation(int k) {
    if (k == cur_slot_) {
      cached_.clear();
      cur_saved_ = false;
      last_gen_cut_ = 0;
    } else {
      slots_[k].cached.clear();
      slots_[k].cur_saved = false;
      slots_[k].last_gen_cut = 0;
    }
    if (eng_.bound_slot() == k) wipe_bound();
    else slots_[k].stale = true;
  }
  void run_batch(const std::vector<GenStep*>& st) {
    const int V = eng_.hp.n_vocab;
    if (st.size() == 1) {
      GenStep& g = *st[0];
      if (g.max_drafts > 0) {
        g.drafts = eng_.mtp_draft(g.tok, g.max_drafts);
        g.dflash = eng_.last_draft_dflash();
        const auto& lv = eng_.verify(g.tok, g.drafts);
        g.rows.assign(lv.begin(), lv.begin() + (1 + g.drafts.size()) * V);
      } else {
        g.rows = eng_.forward(&g.tok, 1);
      }
      return;
    }
    std::vector<int> sl;
    for (GenStep* g : st) sl.push_back(g->slot);
    if (st[0]->max_drafts > 0) {
      const int nd = eng_.batch_rows() / (int)st.size() - 1;  // (the rows of every slot within one pass)
      std::vector<std::vector<int>> rows;
      for (GenStep* g : st) {
        eng_.bind(g->slot);
        g->drafts = eng_.mtp_draft(g->tok, std::min(g->max_drafts, nd));
        g->dflash = eng_.last_draft_dflash();
        rows.push_back({g->tok});
        rows.back().insert(rows.back().end(), g->drafts.begin(), g->drafts.end());
      }
      eng_.bind(cur_slot_);
      const std::vector<float>& lv = eng_.verify_batch(sl, rows);
      size_t r0 = 0;
      for (size_t b = 0; b < st.size(); ++b) {
        st[b]->rows.assign(lv.begin() + r0 * V, lv.begin() + (r0 + rows[b].size()) * V);
        r0 += rows[b].size();
      }
    } else {
      std::vector<int> tk;
      for (GenStep* g : st) tk.push_back(g->tok);
      const std::vector<float>& lv = eng_.forward_batch(sl, tk);
      for (size_t b = 0; b < st.size(); ++b) st[b]->rows.assign(lv.begin() + b * V, lv.begin() + (b + 1) * V);
    }
  }
  // the step `me` of the request in slot `slot` (the engine lock held, the slot bound; bound again on return)
  void gen_step(std::unique_lock<std::mutex>& lk, GenStep& me, int slot) {
    bool can = slots_.size() > 1;
    static const bool off = getenv("ARC_NO_BATCH") && atoi(getenv("ARC_NO_BATCH")) != 0;  // (A/B: turns only)
    can = can && !off && (me.max_drafts == 0 ? eng_.batch_ok() : eng_.batch_verify_ok() && eng_.batch_rows() >= 4);
    if (!can && !gen_wait_.empty()) {
      // (this step cannot lead a pass, e.g. Gemma's MTP drafts with --slots: the waiting steps run alone now, else
      //  a step without drafts waited for the other request's whole answer)
      for (GenStep* g : gen_wait_) g->alone = true;
      gen_wait_.clear();
      gen_cv_.notify_all();
    }
    for (;;) {
      if (!can) break;
      if (!gen_wait_.empty()) {  // lead: this step and the waiting ones of the same kind that fit one pass
        std::vector<GenStep*> st{&me};
        const size_t most = me.max_drafts > 0 ? (size_t)eng_.batch_rows() / 2 : (size_t)eng_.batch_rows();
        for (GenStep* g : gen_wait_) {
          if ((g->max_drafts > 0) == (me.max_drafts > 0) && st.size() < most) st.push_back(g);
          else g->alone = true;
        }
        gen_wait_.clear();
        run_steps(st);
        gen_cv_.notify_all();
        return;
      }
      if (gen_active_ - (int)gen_wait_.size() <= 1) break;  // nobody else will reach a step: alone
      gen_wait_.push_back(&me);
      gen_cv_.wait(lk, [&] { return me.done || me.alone || gen_active_ - (int)gen_wait_.size() <= 0; });
      gen_wait_.erase(std::remove(gen_wait_.begin(), gen_wait_.end(), &me), gen_wait_.end());
      use_slot(slot);
      eng_.cancel = nullptr;  // (a prompt step of another request may have set its own)
      if (!me.failed.empty()) {  // (this slot starts over, as the leader did with its own)
        drop_conversation(slot);
        throw std::runtime_error("batched step: " + me.failed);
      }
      if (me.done) return;
      if (me.alone) break;
      // (every generating request waits: the next one leads)
    }
    run_steps({&me});
  }
  std::atomic<uint64_t> counter_{0};
 public:
  int default_think_budget_ = -1;  // -1 = unlimited
  bool can_preserve_ = false;      // the template has Qwen3.6's preserve_thinking option
  bool default_preserve_ = false;  // --preserve-thinking (requests can override it: chat_template_kwargs)
 private:

  void serve_ui(HttpResponse& rs) {
    std::ifstream f(webui_ + "/index.html", std::ios::binary);
    if (!f) return rs.send(404, "text/plain", "web UI not found: " + webui_ + "/index.html");
    std::stringstream ss;
    ss << f.rdbuf();
    // (asked again each time: an updated page shows at once; a cached one kept the old code, a new error box was not
    //  there until the user reloaded, 2026-11-04)
    rs.extra = "Cache-Control: no-cache\r\n";
    // not inside another site's frame (a click there would act on this page: delete chats, approve a command); no
    // plugins, no <base> that moves the page's links (inline scripts stay allowed: the page is one file)
    rs.extra += "X-Frame-Options: DENY\r\n"
                "Content-Security-Policy: frame-ancestors 'none'; object-src 'none'; base-uri 'none'\r\n";
    rs.send(200, "text/html; charset=utf-8", ss.str());
  }

  // s with every byte that is not part of a valid UTF-8 sequence replaced by U+FFFD (JSON text must be valid UTF-8)
  static std::string utf8_safe(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size();) {
      const unsigned char c = (unsigned char)s[i];
      const size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
      bool ok = n > 0 && i + n <= s.size();
      for (size_t k = 1; ok && k < n; ++k) ok = ((unsigned char)s[i + k] >> 6) == 2;
      if (ok) { o.append(s, i, n); i += n; }
      else { o += "\xEF\xBF\xBD"; ++i; }
    }
    return o;
  }
  static Json err(const std::string& msg) {
    Json e = Json::object();
    e.set("message", msg);
    e.set("type", "invalid_request_error");
    Json r = Json::object();
    r.set("error", e);
    return r;
  }

  static constexpr int kSnaps = 3;  // recurrent-state snapshot slots (see prepare_prefix)
  // Brings the engine to the state after `P[0..start)` reusing cached work; returns start.
  size_t prepare_prefix(const std::vector<int>& P) {
    if (P.empty()) return 0;  // (chat() refuses empty prompts; P.size() - 1 below would wrap)
    size_t L = 0;
    while (L < P.size() && L < cached_.size() && P[L] == cached_[L]) ++L;
    if (L >= P.size()) L = P.size() - 1;  // at least one token must be evaluated to get logits
    while (L > 0 && L < P.size() && P[L] < 0 && P[L - 1] == P[L]) --L;  // back to the start of an image span
    if (L == cached_.size() && (int)cached_.size() == eng_.pos) return L;  // pure extension
    if (rings_) {  // no recurrent state: move back to the common prefix if the sliding-window rings allow it
      if (eng_.rewind((int)L)) {
        for (int s = 0; s < kSnaps; ++s) if (eng_.snapshot_pos(s) > (int)L) eng_.drop_snapshot(s);  // (overwritten next)
        cached_.resize(L);
        return L;
      }
      // the rings moved on (a long answer or tool result since): the latest snapshot at or before L has their copy
      // (Engine::snapshot), e.g. the turn start after the user's last message, where a new message diverges
      int best = -1, bp = -1;
      for (int s = 0; s < kSnaps; ++s) {  // (only those with the rings' copy: a loaded conversation's have none)
        const int p = eng_.snapshot_pos(s);
        if (eng_.snapshot_usable(s) && (size_t)p <= L && p > bp) { best = s; bp = p; }
      }
      if (best >= 0 && eng_.restore(best)) {
        for (int s = 0; s < kSnaps; ++s) if (eng_.snapshot_pos(s) > bp) eng_.drop_snapshot(s);
        cached_.resize(bp);
        return bp;
      }
      eng_.reset();
      for (int s = 0; s < kSnaps; ++s) eng_.drop_snapshot(s);
      cached_.clear();
      return 0;
    }
    // the DeltaNet states can only be restored where a snapshot was taken: the end of the last prompt (slot 1), its
    // last assistant turn start (slot 0), the turn start after the user's last message (slot 2: Qwen's template
    // drops the reasoning of earlier turns once the user writes again, so a new message diverges there), and (in
    // host memory) the end of the system prompt, which conversations of a client share
    int best = -1, bp = -1;
    for (int s = 0; s < kSnaps; ++s) {
      const int p = eng_.snapshot_pos(s);
      if (p >= 0 && (size_t)p <= L && p > bp) { best = s; bp = p; }
    }
    const int hpos = eng_.host_snapshot_pos();
    if (hpos > (int)L) eng_.drop_host_snapshot();  // the request diverges inside the system prompt
    if (hpos >= 0 && hpos <= (int)L && hpos > bp) { best = kSnaps; bp = hpos; }
    // (a restore the engine refuses: the conversation from the start, not the new tokens on the old position)
    if (best >= 0 && (best == kSnaps ? eng_.restore_host() : eng_.restore(best))) {
      for (int s = 0; s < kSnaps; ++s) if (eng_.snapshot_pos(s) > bp) eng_.drop_snapshot(s);
      cached_.resize(bp);
      return bp;
    }
    eng_.reset();
    for (int s = 0; s < kSnaps; ++s) eng_.drop_snapshot(s);
    cached_.clear();
    return 0;
  }

  // ---- session cache: saved conversations on disk
  // When a request belongs to another conversation than the one the engine holds (it does not continue that one's
  // last prompt), the held conversation is written to disk first: its KV cache, DeltaNet states and snapshots
  // (Engine::save_state). A saved conversation that shares more with the request than the held one is loaded instead
  // of processing its prompt again. One file per conversation: [magic, version, n, tokens[n], gen cut, resume
  // floor, n resume points, points..., engine state]; the least recently used go beyond the size cap.
  struct SavedConv {
    std::string file;
    std::vector<int> tokens, resume;  // resume: positions with a recurrent-state snapshot (Qwen)
    int gen_cut = 0, floor = 0;       // floor (Gemma): the first position the sliding-window rings still allow
    uint64_t bytes = 0, used = 0;
    std::shared_ptr<std::thread> writer;          // the file is being written in the background
    std::shared_ptr<std::atomic<bool>> failed;    // ... and that write failed
  };
  // waits for a background write of s; false if it failed
  static bool written(SavedConv& s) {
    if (s.writer && s.writer->joinable()) s.writer->join();
    return !(s.failed && *s.failed);
  }
  void forget(size_t i) {
    written(saved_[i]);
    std::error_code ec;
    std::filesystem::remove(saved_[i].file, ec);
    saved_.erase(saved_.begin() + i);
  }
  std::string sess_dir_;
  uint64_t sess_cap_ = 0, use_clock_ = 0;
  std::vector<SavedConv> saved_;
  bool cur_saved_ = false;  // the engine holds exactly a saved conversation
  int last_gen_cut_ = 0;    // assistant turn start of the last prompt processed

  static size_t common_prefix(const std::vector<int>& a, const std::vector<int>& b) {
    size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
    return n;
  }
  // how much of P a state could keep: the largest resume point within the common prefix
  int usable(const std::vector<int>& P, const std::vector<int>& tokens, const std::vector<int>& resume, int floor) const {
    const int L = (int)std::min(common_prefix(P, tokens), P.size() - 1);
    if (rings_) return L >= floor ? L : 0;
    int u = 0;
    for (int r : resume)
      if (r <= L) u = std::max(u, r);
    return u;
  }
  std::vector<int> current_resume() const {
    std::vector<int> r = {eng_.host_snapshot_pos()};
    for (int s = 0; s < kSnaps; ++s) r.push_back(eng_.snapshot_pos(s));
    if ((int)cached_.size() == eng_.pos) r.push_back(eng_.pos);
    return r;
  }
  int current_floor() const { return rings_ ? eng_.rewind_floor() : 0; }

  // Saves the held conversation to the session cache and waits until it is on disk (zero-server stop / restart and a
  // model switch call it first, so that going back to the conversation afterwards loads it). False when there is
  // nothing worth saving (cache off, already saved, under 1,024 tokens).
#include "server_cache.h"  // (saved state: prefix cache, sessions)

#include "server_chat.h"  // (the chat endpoint)
#include "server_ocr.h"  // (OCR models: asked with their own task prompts)
};

#include "server_catalog.h"  // (model catalog, --ctx auto)

#include "server_load.h"  // (loading a model)
#include "server_whisper.h"  // (the speech service: Whisper, Qwen3-ASR, the transcript library)
#include "server_speech.h"  // (starting it, Gemma 4 E4B for transcripts, /v1/audio/*)

#include "server_file_tools.h"  // (the model's file tools, /v1/tools/*, and the files API)
#include "server_mcp.h"         // (MCP servers' tools for the model, /v1/mcp)

int main(int argc, char** argv) {
  crashlog::install();
  crashlog::extra = [] { zrt::print_last_launch(); };  // (a crash inside a launch: which kernel, with what)
  full_speed_process();
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--list-models") {
      size_t speech = 0;  // (--speech-reserve N: bytes a running server keeps for speech to text, its /props)
      for (int j = 1; j + 1 < argc; ++j)
        if (std::string(argv[j]) == "--speech-reserve") speech = (size_t)std::max(0.0, atof(argv[j + 1]));
      return list_models(speech);
    }
  zlog::start();  // (the log's prefixes and whole lines: src/log.h; --log-* below)
  std::string model, host = "127.0.0.1", alias;
  int port = 8080;
  bool swap = true, help = false, web_tools = true, terminal = true, precise = false;
  std::string whisper_opt = "off", whisper_dev = "gpu", asr_opt = "default";  // (default: auto when Whisper is on)
  bool whisper_given = false;  // (--whisper on the command line; without it the cache's model is offered, not loaded)
  double whisper_max_min = 180;
  ServeOpts so;
  so.eo.ctx = 40960;
  // A bad option value is a message, not an uncaught std::stoi exception (which aborted the process).
  auto to_int = [](const std::string& opt, const std::string& s) {
    try {
      size_t used = 0;
      const int v = std::stoi(s, &used);
      if (used == s.size()) return v;
    } catch (...) {}
    throw std::runtime_error(opt + ": expected an integer (got \"" + s + "\")");
  };
  auto to_double = [](const std::string& opt, const std::string& s) {
    try {
      size_t used = 0;
      const double v = std::stod(s, &used);
      if (used == s.size() && std::isfinite(v)) return v;  // (nan / inf refused: they become sizes and limits)
    } catch (...) {}
    throw std::runtime_error(opt + ": expected a number (got \"" + s + "\")");
  };
  // the options and what each takes ("": a switch), for the messages of a missing value and of an unknown option
  static const std::vector<std::pair<std::string, std::string>> kOpts = {
      {"--model", "a .gguf file, e.g. -m \"C:\\models\\Qwen3.5-4B-UD-Q4_K_XL.gguf\", or none"}, {"--host", "an address, e.g. 0.0.0.0"},
      {"--port", "a port, 1..65535"}, {"--ctx", "a number of tokens, or auto"}, {"--chunk", "a number of tokens, 0 = auto"},
      {"--expert-cache-gb", "GiB, e.g. 6"}, {"--expert-cache-pct", "a percentage, e.g. 40"}, {"--expert-ram-gb", "GiB, e.g. 200"}, {"--driver-reserve-gb", "GiB, 0 to 8, e.g. 0.5"},{"--no-stream", ""},
      {"--pack", "a MoE .gguf file (its -00001-of- part when split)"}, {"--alias", "a model id"},
      {"--webui", "a folder with index.html"}, {"--budget", "GiB"}, {"--attn-f32", ""}, {"--act256", ""}, {"--precise", ""}, {"--npu", "on, off or force"},
      {"--mtp", ""}, {"--dflash", "a DFlash2 drafter folder (with model.safetensors)"}, {"--no-dflash", ""},
      {"--draft", "auto, mtp or dflash"}, {"--mmproj", "an mmproj .gguf file"}, {"--no-mmproj", ""},
      {"--image-min-tokens", "a number of tokens"}, {"--image-max-tokens", "a number of tokens"},
      {"--mtp-drafts", "a number, e.g. 3"}, {"--think-budget", "a number of tokens"}, {"--preserve-thinking", ""},
      {"--kernels", "a folder"}, {"--session-cache-gb", "GiB, 0 = off"}, {"--session-dir", "a folder"},
      {"--slots", "1..8"}, {"--gpus", "2..64"}, {"--no-swap", ""}, {"--no-web-tools", ""},
      {"--whisper", "auto (or on), off, or a faster-whisper folder with model.bin"},
      {"--asr", "auto (or on), off, or a Qwen3-ASR .gguf file"}, {"--asr-slots", "1..4"},
      {"--whisper-device", "gpu or npu"}, {"--whisper-max-minutes", "minutes, 1..600"}, {"--no-terminal", ""},
      {"--no-prefix-cache", ""}, {"--prefix-cache-gb", "GiB"}, {"--list-models", ""}, {"--help", ""},
      {"--log-verbosity", "0 errors, 1 warnings, 2 info, 3 debug (or error / warn / info / debug)"},
      {"--log-timestamps", "on or off"}, {"--log-colors", "auto, on or off"}, {"--log-file", "a file"}, {"--log-disable", ""}};
  auto takes = [&](std::string o) {  // (what option o takes, as " (...)")
    if (o == "-m") o = "--model";
    if (o == "-c") o = "--ctx";
    for (const auto& [n, v] : kOpts)
      if (n == o && !v.empty()) return " (" + v + ")";
    return std::string();
  };
  try {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    // (a missing value is a message, not an empty one: "--whisper" at the end gave "cannot open model.bin")
    auto next = [&]() {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value" + takes(a));
      const std::string v = argv[i + 1];
      if (v.rfind("--", 0) == 0) throw std::runtime_error(a + " needs a value" + takes(a) + ", not the next option " + v);
      ++i;
      return v;
    };
    if (a == "-m" || a == "--model") model = next();
    else if (a == "--log-verbosity" || a == "-lv") {  // (the log: src/log.h)
      const std::string v = next();
      int l = 2;
      if (!zlog::level_name(v, l)) throw std::runtime_error(a + ": expected 0-3 or error / warn / info / debug (got " + v + ")");
      zlog::cfg().level = l;
    } else if (a == "--log-timestamps" || a == "--log-colors") {
      const std::string v = next();
      if (v != "on" && v != "off" && !(v == "auto" && a == "--log-colors")) throw std::runtime_error(a + ": expected on or off" + takes(a));
      if (a == "--log-timestamps") zlog::cfg().stamps = v == "on";
      else if (v != "auto") zlog::cfg().colors = v == "on";
    } else if (a == "--log-file") {
      const std::string f = next();
      FILE* lf = fopen(f.c_str(), "ab");
      if (!lf) throw std::runtime_error("--log-file: cannot open " + f);
      // (the file stderr already goes to, e.g. zero-server's server.log: two writers on one file would overwrite each
      //  other's lines; refused, audit 2026-11-12b #6)
      BY_HANDLE_FILE_INFORMATION a1{}, a2{};
      const HANDLE he = (HANDLE)_get_osfhandle(_fileno(stderr)), hf = (HANDLE)_get_osfhandle(_fileno(lf));
      if (he != INVALID_HANDLE_VALUE && GetFileInformationByHandle(he, &a1) && GetFileInformationByHandle(hf, &a2) &&
          a1.dwVolumeSerialNumber == a2.dwVolumeSerialNumber && a1.nFileIndexHigh == a2.nFileIndexHigh &&
          a1.nFileIndexLow == a2.nFileIndexLow) {
        fclose(lf);
        throw std::runtime_error("--log-file: " + f + " is the file stderr already goes to (every line is there "
                                 "already); name another file, or leave the option out");
      }
      zlog::cfg().file = lf;
    } else if (a == "--log-disable") zlog::cfg().disabled = true;
    else if (a == "--host") host = next();
    else if (a == "--port") port = to_int(a, next());
    else if (a == "--ctx" || a == "-c") {
      const std::string v = next();
      if (v == "auto") so.ctx_auto = true;
      else so.eo.ctx = std::clamp(to_int(a, v), 0, 1 << 22);  // (4M tokens: past it the sizes overflow int)
    }
    else if (a == "--chunk") so.eo.chunk = std::clamp(to_int(a, next()), 0, 65536);
    else if (a == "--expert-cache-gb") so.eo.expert_cache_gb = to_double(a, next());
    else if (a == "--expert-cache-pct") so.eo.expert_cache_pct = to_double(a, next());
    else if (a == "--expert-ram-gb") so.eo.expert_ram_gb = to_double(a, next());
    else if (a == "--driver-reserve-gb") {  // (Engine::driver_slack reads it, once, before any engine exists)
      const double g = to_double(a, next());
      if (!(g >= 0.0 && g <= 8.0)) { LOG_ERR("inferzero-server: --driver-reserve-gb: 0 to 8 GiB\n"); return 1; }
      _putenv_s("ARC_DRIVER_RESERVE_GB", std::to_string(g).c_str());
    }
    else if (a == "--no-stream") so.eo.stream_experts = false;
    else if (a == "--pack") {  // make an inferzero package of a MoE GGUF, then exit
      const std::string src = next();
      try {
        zlog::eprintf("inferzero: making a package of %s (reads the whole model once)\n", src.c_str());
        const zpack::Result r = zpack::make_package(src);
        zlog::eprintf("inferzero: package ready in %.0f s\n  %s (%.2f GB)\n  %s (%.2f GB)\n"
                        "The original GGUF is no longer needed to run this model with inferzero: the model list now\n"
                        "picks the package. The engine's own expert file for the GGUF moved into the package, so loading\n"
                        "the GGUF itself (by its path) would write that file again; keep the GGUF only as a backup.\n",
                r.seconds, r.gguf.c_str(), r.rest_bytes / 1e9, r.experts.c_str(), r.expert_bytes / 1e9);
        return 0;
      } catch (const std::exception& e) {
        zlog::eprintf("inferzero: --pack failed: %s\n", e.what());
        return 1;
      }
    }
    else if (a == "--alias") alias = next();
    else if (a == "--webui") so.webui = next();
    else if (a == "--budget") { so.eo.w8_budget_gb = to_double(a, next()); so.budget_given = true; }
    else if (a == "--attn-f32") so.eo.attn_int8 = false;
    else if (a == "--act256") so.eo.act32_all = false;
    else if (a == "--precise") precise = true;
    else if (a == "--npu") {  // (kept by every model switch: so.eo)
      std::string v = next();
      for (char& c : v) c = (char)tolower((unsigned char)c);
      so.eo.npu = v == "on" || v == "yes" || v == "true" || v == "1" ? 1 : v == "off" || v == "no" || v == "false" || v == "0" ? 0
                : v == "force" ? 2 : -2;
      if (so.eo.npu == -2) throw std::runtime_error("--npu: expected on, off or force (got \"" + v + "\")");
    }
    else if (a == "--mtp") so.eo.mtp = true;
    else if (a == "--dflash") { so.eo.dflash = next(); so.eo.mtp = true; }
    else if (a == "--no-dflash") { so.eo.dflash.clear(); so.dflash_auto = false; }
    else if (a == "--draft") {  // (the drafter of the -nodf / -df variants)
      const std::string d = next();
      if (d == "mtp") so.eo.dflash.clear(), so.dflash_auto = false, so.eo.mtp_block = true;
      else if (d == "dflash") so.dflash_auto = true, so.eo.mtp_block = false;
      else if (d == "auto") so.dflash_auto = true, so.eo.mtp_block = true;
      else throw std::runtime_error("--draft: expected auto, mtp or dflash (got \"" + d + "\")");
    }
    else if (a == "--mmproj") so.mmproj = next();
    else if (a == "--no-mmproj") so.mmproj = "-";
    else if (a == "--image-min-tokens") so.img_min = atoi(next().c_str());
    else if (a == "--image-max-tokens") so.img_max = atoi(next().c_str());
    else if (a == "--mtp-drafts") so.eo.mtp_drafts = to_int(a, next());
    else if (a == "--think-budget") so.think_budget = to_int(a, next());
    else if (a == "--preserve-thinking") so.preserve_thinking = true;
    else if (a == "--kernels") so.eo.kernel_dir = next();
    else if (a == "--session-cache-gb") so.sess_gb = std::clamp(to_double(a, next()), 0.0, 4096.0);
    else if (a == "--session-dir") so.sess_dir = next();
    else if (a == "--slots") so.eo.slots = std::clamp(to_int(a, next()), 1, 8);
    else if (a == "--gpus") so.gpus = std::clamp(to_int(a, next()), 1, 64);
    else if (a == "--no-swap") swap = false;
    else if (a == "--no-web-tools") web_tools = false;
    else if (a == "--whisper") whisper_opt = next(), whisper_given = true;
    else if (a == "--asr") asr_opt = next();
    else if (a == "--asr-slots") so.asr_slots = std::clamp(to_int(a, next()), 1, 4);
    else if (a == "--whisper-device") {
      whisper_dev = next();
      if (whisper_dev != "gpu" && whisper_dev != "npu")
        throw std::runtime_error("--whisper-device: expected gpu or npu (got \"" + whisper_dev + "\")");
    }
    else if (a == "--whisper-max-minutes") whisper_max_min = std::clamp(to_double(a, next()), 1.0, 600.0);
    else if (a == "--no-terminal") terminal = false;
    else if (a == "--no-prefix-cache") so.prefix_cache = false;
    else if (a == "--prefix-cache-gb") so.prefix_gb = std::clamp(to_double(a, next()), 0.0, 4096.0);
    else if (a == "--help" || a == "-h") help = true;
    else {
      // (what was meant: a value without its option, one dash where two belong, or a typo)
      if (a.empty() || a[0] != '-')
        throw std::runtime_error("'" + a + "' is a value without an option before it (options start with --; a path with spaces "
                                 "goes in quotes: \"C:\\My models\\model.gguf\")");
      std::string b = a;
      for (char& c : b) c = (char)tolower((unsigned char)c);
      if (b.size() > 2 && b[1] != '-') b = "-" + b;
      auto dist = [](const std::string& x, const std::string& y) {  // (edit distance)
        std::vector<size_t> d(y.size() + 1);
        for (size_t j = 0; j <= y.size(); ++j) d[j] = j;
        for (size_t p = 1; p <= x.size(); ++p) {
          size_t prev = d[0];
          d[0] = p;
          for (size_t j = 1; j <= y.size(); ++j) {
            const size_t t = d[j];
            d[j] = std::min({d[j] + 1, d[j - 1] + 1, prev + (x[p - 1] == y[j - 1] ? 0 : 1)});
            prev = t;
          }
        }
        return d[y.size()];
      };
      std::string best;
      size_t bd = 3;
      for (const auto& [n, v] : kOpts)
        if (const size_t d = dist(b, n); d < bd) bd = d, best = n;
      throw std::runtime_error("unknown option " + a +
                               (best.empty() ? std::string(" (--help lists them)")
                                : bd == 0    ? " (this server's options take two dashes: " + best + takes(best) + ")"
                                             : " (did you mean " + best + takes(best) + "?)"));
    }
  }
  if (!help) {
    // (--precise: Flash Next's precise mode, as ARC_FN_PRECISE=1, read by the engine when it first loads such a model;
    //  for the whole process, so a model switch keeps it)
    if (precise) {
      _putenv_s("ARC_FN_PRECISE", "1");
      zlog::eprintf("precise mode: on (Qwen3.8 Flash Next: BF16 hyper-connection rows and router, two int8 passes)\n");
    }
    if (port < 1 || port > 65535) throw std::runtime_error("--port: expected 1..65535 (got " + std::to_string(port) + ")");
    if (host.empty()) throw std::runtime_error("--host: an address or name is required");
    if (!so.ctx_auto && so.eo.ctx < 1) throw std::runtime_error("--ctx: expected 1 or more tokens (or auto)");
    if (so.eo.chunk < 0) throw std::runtime_error("--chunk: expected 0 (auto) or more tokens");
    if (so.eo.mtp_drafts < 0) throw std::runtime_error("--mtp-drafts: expected 0 or more");
    // the files and folders given, checked now: what each must be, said here instead of a later, vaguer error
    namespace fs = std::filesystem;
    auto check = [](const std::string& opt, const std::string& p, bool folder, const std::string& holds) {
      std::error_code ec;
      if (p.find('"') != std::string::npos)
        throw std::runtime_error(opt + ": '" + p + "' has a quote in it. A path written as \"C:\\folder\\\" (a backslash just "
                                 "before the closing quote) reads as a quote: leave the last backslash out, \"C:\\folder\"");
      if (!fs::exists(p, ec))
        throw std::runtime_error(opt + ": no " + (folder ? "folder" : "file") + " '" + p + "'" +
                                 (fs::path(p).is_relative() ? " (a relative path is from the folder the server starts in; "
                                                              "give the full path, C:\\...)" : ""));
      if (folder && !fs::is_directory(p, ec))
        throw std::runtime_error(opt + ": '" + p + "' is a file: give the folder" +
                                 (holds.empty() ? std::string() : " that holds " + holds) + ", " + fs::path(p).parent_path().string());
      if (!folder && fs::is_directory(p, ec)) {
        std::string inside;  // (the .gguf files there, as the hint)
        int n = 0;
        for (const auto& d : fs::directory_iterator(p, ec))
          if (const std::string f = d.path().filename().string();  // (not the projectors and drafters beside a model)
              d.path().extension() == ".gguf" && f.rfind("mmproj", 0) != 0 && f.rfind("mtp-", 0) != 0 && n++ < 3)
            inside += "\n  " + d.path().string();
        throw std::runtime_error(opt + ": '" + p + "' is a folder: give the " + holds + " file in it" +
                                 (inside.empty() ? std::string(" (it has no .gguf file)") : ", e.g." + inside));
      }
      if (folder && !holds.empty() && !fs::exists(fs::path(p) / holds, ec))
        throw std::runtime_error(opt + ": '" + p + "' holds no " + holds + ": give the folder that has it");
      if (!folder && holds == ".gguf" && fs::path(p).extension() != ".gguf")
        throw std::runtime_error(opt + ": '" + p + "' is no .gguf file");
    };
    auto word = [](std::string v) {  // (auto / on / off and the like)
      for (char& c : v) c = (char)tolower((unsigned char)c);
      return v == "auto" || v == "auto-quiet" || v == "default" || v == "on" || v == "off" || v == "yes" || v == "no" ||
             v == "true" || v == "false" || v == "1" || v == "0";
    };
    if (!model.empty() && model != "none") check("-m", model, false, ".gguf");
    if (model == "none" && !swap) throw std::runtime_error("-m none: a chat model loads later on request, which --no-swap turns off");
    if (!so.mmproj.empty() && so.mmproj != "-") check("--mmproj", so.mmproj, false, ".gguf");
    if (!so.eo.dflash.empty()) check("--dflash", so.eo.dflash, true, "model.safetensors");
    if (!so.webui.empty()) check("--webui", so.webui, true, "index.html");
    if (!word(whisper_opt)) check("--whisper", whisper_opt, true, "model.bin");
    if (!word(asr_opt)) check("--asr", asr_opt, false, ".gguf");
  }
  } catch (const std::exception& e) {
    LOG_ERR("inferzero-server: %s\n", e.what());
    return 2;
  }
  if (model.empty() || help) {
    zlog::raw(  // (the usage text as it is, without the log's prefix; also to --log-file)
            "usage: inferzero-server -m model.gguf [options]   OpenAI-compatible server (API at /v1, web UI at /)\n"
            "       inferzero-server -m none [options]         speech to text only (Qwen3-ASR, --whisper auto for\n"
            "                                                 Whisper too); a chat request naming a model loads it later\n"
            "       inferzero-server --list-models             the models it can load, as JSON (Hugging Face cache,\n"
            "                                                 %%USERPROFILE%%\\.inferzero\\models, model-dirs.txt) with each one's auto context\n"
            "                        (--speech-reserve BYTES: that much GPU memory kept for speech to text, a running server's\n"
            "                         /props speech_reserve_bytes, as its model switches keep it)\n"
            "  -m, --model FILE      model GGUF; an mmproj-* image projector and an mtp-* drafter next to it are found\n"
            "  --host H, --port P    listen address (default 127.0.0.1:8080). A browser page of another origin (another\n"
            "                        site, or another port of this computer) reaches only the model endpoints: not the\n"
            "                        tools, files, chats, transcripts, live recording, slots or sessions\n"
            "  --alias NAME          model id in the API (default: the file name)\n"
            "  -c, --ctx N|auto      context in tokens (default 40960); auto: the largest that fits the GPU memory with\n"
            "                        0.5 GiB left for the driver, at most the trained context (remembered in\n"
            "                        " ARC_DATA_HELP "ctx-auto.txt)\n"
            "  --chunk N             prompt tokens per step (default 4096; MoE with experts on the SSD 8192, Flash\n"
            "                        Next 2048; MoE with --no-stream 2048; 512 when decoded weights stay resident;\n"
            "                        Gemma with images at least 2048; Gemma MoE 2048)\n"
            "  --mtp                 speculative decoding with the model's MTP block or Gemma assistant drafter\n"
            "  --mtp-drafts N        drafts per step: Qwen 1..7, Gemma 1..3 (default: 7 for Qwen 27B-class and with\n"
            "                        DFlash2, 5 for MoE, 3 for 4B and Gemma 4 26B, 2 otherwise)\n"
            "  --npu on|off|force    the NPU (Intel AI Boost) computes part of each prompt chunk next to the GPU\n"
            "                        (Qwen3.8 27B +15 %%, Gemma 4 12B +10 %%; others off, force: on anyway);\n"
            "                        kept when a request switches models; default off (or ARC_NPU=1)\n"
            "  --gpus N              the model split over N GPUs (Qwen3.5 / 3.6 / 3.8 dense and the 35B-A3B; docs/multi-gpu.md):\n"
            "                        text chat with --slots and batched steps; no MTP / DFlash2, images, saved prompts\n"
            "                        or sessions yet\n"
            "  --slots N             hold N conversations at once (default 1), each with its own KV cache and states\n"
            "                        of --ctx tokens (auto: the context that fits N of them; MoE with experts on the\n"
            "                        SSD: 128K in all, at least 32K each); requests in different slots take turns step\n"
            "                        by step (a prompt piece, a generated token), and requests generating at the same\n"
            "                        time run their next tokens and drafts in one pass (Qwen, Gemma 4; Flash Next\n"
            "                        without a drafter; ARC_NO_BATCH=1: turns only)\n"
            "  --dflash DIR          speculative decoding with a DFlash2 block drafter (z-lab Qwen3.8 27B, incoai\n"
            "                        Qwen3.6 35B-A3B; 7 drafts per step); default with --mtp: a *DFlash2* drafter\n"
            "                        for the model in the Hugging Face cache or the extra model folders, if any\n"
            "                        (incoai Qwen3.8 Flash Next: with --dflash DIR only; no faster on an SSD-bound laptop)\n"
            "  --no-dflash           no DFlash2 drafter (--mtp: the model's MTP block only)\n"
            "  --draft auto|mtp|dflash  with --mtp and a DFlash2 drafter: both, picked per step (auto, default), the MTP\n"
            "                        block only (= --no-dflash), or DFlash2 only (the MTP block is not loaded: more\n"
            "                        expert cache); the catalog's -nodf / -df variants\n"
            "  --think-budget N      default reasoning budget in tokens (requests can override it)\n"
            "  --preserve-thinking   earlier turns keep their reasoning (Qwen3.6 template option; requests can override it\n"
            "                        with chat_template_kwargs.preserve_thinking; Qwen3.8 always keeps it)\n"
            "  --mmproj FILE         image projector (default: found next to the model); --no-mmproj: none\n"
            "  --image-min-tokens N  an image becomes at least N tokens (smaller images are scaled up; Qwen: 32x32 pixels\n"
            "                        per token, default 256; Gemma 4 12B: 48x48 pixels, default 1120)\n"
            "  --image-max-tokens N  at most N tokens per image (larger images are scaled down; Qwen default 2048, Gemma\n"
            "                        4 12B 2048, above the 1120 it was trained with)\n"
            "                        A Qwen image goes through the layers with its prompt's text in one pass (models\n"
            "                        without an MTP block or drafter, e.g. Flash Next; ARC_MEDIA_PASSES=1: a pass each)\n"
            "  --budget GB           keep decoded weights resident up to GB\n"
            "MoE models (experts read from the SSD into a GPU cache; expert file next to a GGUF inside\n"
            "%%USERPROFILE%%\\.inferzero, else in its own %%USERPROFILE%%\\.inferzero\\zero\\models\\<name>,\n"
            "made on the first start, safe to delete):\n"
            "  --expert-cache-gb N   expert cache size in GiB (default: automatic: the GPU share Windows reports, your\n"
            "                        setting, less 0.5 GiB for the driver, less the rest of the model)\n"
            "  --expert-cache-pct P  expert cache as a percentage of all experts\n"
            "  --driver-reserve-gb N GPU memory left to the driver, GiB (default 0.5; ARC_DRIVER_RESERVE_GB does the same)\n"
            "  --expert-ram-gb N     experts read from the SSD also kept in RAM, up to N GiB, and read from there again\n"
            "                        (default: a GPU with memory of its own: the free RAM less 16 GiB; an iGPU: 0)\n"
            "  --pack FILE.gguf      make a package of a MoE model (its first part when split): the non-expert\n"
            "                        weights + an expert file under %%USERPROFILE%%\\.inferzero\\zero\\models; afterwards\n"
            "                        the original GGUF can be moved away (e.g. to a USB disk)\n"
            "  --no-stream           every expert on the GPU (no SSD streaming; MoE models then get less context)\n"
            "  --attn-f32            f32 attention (tests)\n"
            "  --act256              one activation scale per 256 values in the prompt GEMMs (not 32; faster on XMX\n"
            "                        GPUs perhaps, less accurate: check the KL first)\n"
            "  --precise             Qwen3.8 Flash Next's precise mode (as ARC_FN_PRECISE=1): BF16 hyper-connection rows and\n"
            "                        router, two int8 passes everywhere: KL 0.0186 -> 0.0127 at 3.5K tokens, prompts about\n"
            "                        half as fast; other models are unchanged\n"
            "  GPU: picked at start: Xe2 / Xe3 GPUs (no 8-lane sub-groups) get 16-lane kernels, and XMX for the prompt\n"
            "  GEMMs when their results match the DP4a kernels' on a test (ARC_NO_XMX=1: DP4a; ARC_NO_SG8=1: 16 lanes)\n"
            "  --kernels DIR         kernel folder (default zero\\kernels);  --webui DIR  web UI folder\n"
            "  --log-verbosity N, -lv N  the log: 0 errors, 1 + warnings, 2 + info (default), 3 + debug (every\n"
            "                        request's access line, llama.cpp's timing lines); also ARC_LOG_LEVEL\n"
            "  --log-timestamps on|off  the date and time in front of each line (default on);  --log-colors auto|on|off\n"
            "  --log-file F          every line to F as well;  --log-disable  no log at all (docs/logging.md; GET /metrics:\n"
            "                        Prometheus, vLLM's names with inferzero:)\n"
            "  --session-cache-gb N  disk space for saved conversations, all models together (default 0 = off; zero-server\n"
            "                        and the launcher: 8): switching between conversations loads the other one's state\n"
            "                        instead of processing its prompt again (other models' go first, oldest first)\n"
            "  --session-dir DIR     where they are kept (default " ARC_DATA_HELP "sessions)\n"
            "  --no-prefix-cache     do not keep system prompts on disk (default: kept in " ARC_DATA_HELP "prefix,\n"
            "                        per project folder for Pi (its 2 latest versions), the 8 latest for other clients;\n"
            "                        a request that starts with one loads it instead of processing it)\n"
            "  --prefix-cache-gb N   disk space for them, all models together (default 8, 0 = no limit; least recently\n"
            "                        used go first)\n"
            "  --no-swap             keep the loaded model (default: a request whose \"model\" is another model of\n"
            "                        --list-models, by id, name or file, makes the server load that one, context auto)\n"
            "  --no-web-tools        no POST /v1/tools/web_search and /v1/tools/web_fetch (the web UI's search and page\n"
            "                        tools; default: on, public http(s) addresses only)\n"
            "  --whisper auto|DIR|off  speech to text for POST /v1/audio/transcriptions and /translations (OpenAI's\n"
            "                        API; docs/whisper-native.md): auto (or on) = faster-whisper large-v3 from the Hugging\n"
            "                        Face cache (large-v3-turbo when it is not there), DIR = a faster-whisper folder;\n"
            "                        also the web UI's Transcripts / Meetings tabs and voice messages; default off (the\n"
            "                        endpoint then asks the loaded model, if it hears audio); --ctx auto keeps its GPU\n"
            "                        memory free\n"
            "  --whisper-device gpu|npu  Whisper's encoder on the iGPU (default) or the NPU (the decoder: the iGPU)\n"
            "  --whisper-max-minutes N  longest upload or chat clip (default 180); the transcript library takes files\n"
            "                        up to 6 hours, or N when it is more\n"
            "  --asr auto|FILE|off   Qwen3-ASR, the default speech model: requests, transcripts and chat audio go to it\n"
            "                        unless they name Whisper (docs/qwen3-asr.md); auto = Qwen3-ASR-1.7B GGUF from the\n"
            "                        Hugging Face cache (its mmproj next to it); default: auto with --whisper when the\n"
            "                        cache has it, else Whisper alone. Without --whisper: Qwen3-ASR alone (~3.7 GB of\n"
            "                        GPU memory instead of ~9.8 with Whisper; no translations, no text while recording\n"
            "                        live, no word times)\n"
            "  --asr-slots N         Qwen3-ASR parts of a clip longer than 4 minutes decoded together, 1-4 (default 1;\n"
            "                        each more ~0.6 GB of GPU memory; 4: a 14-minute file 129 -> 80 s)\n"
            "  --no-terminal       no POST /v1/tools/run_command (the web UI's terminal tool: PowerShell in the\n"
            "                        chat's folder after the user allows it; default: on, this computer only)\n"
            "  A client that closes its connection (Pi: Esc) cancels its request: prompt processing stops within a\n"
            "  layer, generation before the next token.\n");
    return help ? 0 : 1;
  }
  // locate kernels/ and webui/ next to the executable's parent (bin/..) unless given
  std::filesystem::path exe_dir = std::filesystem::path(argv[0]).parent_path();
  auto find_dir = [&](const std::string& name) {
    for (auto p : {std::filesystem::current_path() / name, exe_dir / name, exe_dir.parent_path() / name})
      if (std::filesystem::exists(p)) return p.string();
    return name;
  };
  if (so.eo.kernel_dir == "kernels" || so.eo.kernel_dir == "zero/kernels") so.eo.kernel_dir = find_dir(so.eo.kernel_dir);
  if (so.webui.empty()) so.webui = find_dir("webui");
  srvstart::print(exe_dir, false);  // (the log's first lines: build, system, switches; tools/server_startup.h)
  if (alias.empty()) alias = std::filesystem::path(model).stem().string();
  // -m none: speech to text only (transcripts), a chat model loaded later by the first request naming one; with no
  // speech option given, Qwen3-ASR alone (the default transcript model, ~3.7 GB of GPU memory)
  const bool no_chat = model == "none";
  const bool asr_given = asr_opt != "default";  // (--asr on the command line, before the default below)
  if (no_chat && whisper_opt == "off" && asr_opt == "default") asr_opt = "auto";

  // the NPU (--npu on / force, ARC_NPU): Level Zero initialized for every driver type before anything starts the GPU
  // runtime (it initializes for GPUs only, which hides the NPU driver for the rest of the process). It was done only
  // in the speech service's start, after its early return without Whisper / ASR, and --ctx auto starts the GPU
  // runtime before the engine's own npu_preinit: "--npu on" with Gemma 4 12B said "no NPU device" (2026-10-31)
  if ((so.eo.npu > 0 || (so.eo.npu < 0 && getenv("ARC_NPU") && atoi(getenv("ARC_NPU")) != 0)) && ze_load()) zeInit(0);
  // the speech service before the model: --ctx auto then leaves its GPU memory free (tools/server_speech.h)
  std::unique_ptr<srvwh::Service> wh;
  loadp::st().on = true;  // ("progress: P % ..." lines while loading: zero-server's and the launcher's progress bar)
  try {
    wh = srvsp::start(whisper_opt, whisper_dev, asr_opt, whisper_max_min, so);
  } catch (const std::exception& e) {
    LOG_ERR("inferzero-server: --whisper: %s\n", e.what());
    return 1;
  }
  if (no_chat && !wh) {
    LOG_ERR("inferzero-server: -m none with --whisper off --asr off: nothing to run\n");
    return 1;
  }
  // (the GPU line: its probe starts the GPU runtime, which hides the NPU driver from a later zeInit(0), so only after
  //  the NPU's start above and the speech service's (--whisper-device npu), audit 2026-11-12)
  srvstart::print(exe_dir, true);
  // Settings can load and unload the speech models whatever the start said, unless it said off: without --whisper the
  // cache's Whisper is offered (unloaded), without --asr its Qwen3-ASR, and the service exists with nothing loaded when
  // a chat model started alone. Before, Settings said "not on this server" with the models on the disk.
  if (!whisper_given || (wh && !asr_given)) {
    if (!wh) {
      wh = std::make_unique<srvwh::Service>(std::string(), false, so.eo.kernel_dir);
      wh->max_seconds = whisper_max_min * 60;
    }
    if (!whisper_given) wh->offer_whisper(whisper::find_model());
    if (!asr_given) wh->offer_asr(q3asr::find_model(), so.asr_slots);
    so.whisper_props = wh->props();
  }
  std::unique_ptr<Loaded> cur;
  std::string unloaded_path;  // (the chat model unloaded in Settings: first in /v1/models while none is loaded)
  if (no_chat)
    zlog::eprintf("no chat model (-m none): speech to text only; a chat request naming a model loads it\n");
  else
    try {  // (a model that cannot load: its reason, not a silent abort)
      cur = load_model(model, alias, so);
    } catch (const std::exception& e) {
      LOG_ERR("inferzero-server: cannot load %s: %s\n", model.c_str(), e.what());
      // (one that does not fit next to speech to text, models switchable: the server starts without it, as -m none,
      //  so the web UI can free memory and load it; else the reason and the end, as before)
      const bool fit = std::string(e.what()).find("does not fit in the GPU memory") != std::string::npos;
      if (!(fit && swap && wh && wh->gpu_bytes())) return 1;  // (speech to text loaded: it holds the memory)
      zlog::eprintf("starting without a chat model (unload speech to text in the web UI's Settings, GPU memory, then a "
                      "request naming the model loads it)\n");
    }
  ServeOpts cur_so = so;  // (the loaded model's options: a change of --slots loads it again with them)
  std::vector<ModelInfo> catalog = swap ? scan_models() : std::vector<ModelInfo>();
  std::shared_mutex swap_mu;  // requests share it; switching models takes it exclusively
  if (wh)  // (each Whisper step: with the loaded model's engine lock; a model switch can come between two)
    wh->gpu_turn = [&](const std::function<void()>& f) {
      std::shared_lock<std::shared_mutex> lk(swap_mu);
      if (cur && cur->srv) cur->srv->with_engine(f);
      else f();
    };
  // the catalog entry a request's "model" names (alias, org/repo name or file), rescanning once for new downloads
  // names no catalog entry has (clients send their own: "gpt-4o", "claude-..."): the request stays on the loaded model,
  // and for a minute it takes neither the exclusive lock (it would wait for every running request) nor a rescan
  std::mutex unknown_mu;
  std::map<std::string, clk::time_point> unknown_names;
  auto known_unknown = [&](const std::string& want) {
    std::lock_guard<std::mutex> g(unknown_mu);
    auto it = unknown_names.find(want);
    return it != unknown_names.end() && clk::now() - it->second < std::chrono::seconds(60);
  };
  auto find_model = [&](const std::string& want0) -> const ModelInfo* {
    // (the data folder's models were "arcinferzero/<name>" before the rename to inferzero, 2026-11-10)
    const std::string want = want0.rfind("arcinferzero/", 0) == 0 ? want0.substr(3) : want0;
    // (an entry made stale since the scan: its file deleted, or a GGUF packed since (zero-server pack while this server
    //  ran another model): loaded from the catalog as it was, the original's expert file would be written again next
    //  to the package, or the load fails once the original is gone. The scan again lists the package under its id)
    auto stale = [](const ModelInfo& m) {
      namespace fs = std::filesystem;
      std::error_code ec;
      if (!fs::exists(m.path, ec)) return true;
      const std::string pk = lower_ascii(fs::weakly_canonical(zpack::packages_dir(), ec).string()) + "\\";
      if (lower_ascii(fs::weakly_canonical(m.path, ec).string()).rfind(pk, 0) == 0) return false;  // (a package)
      std::string stem = fs::path(m.path).stem().string();
      if (const size_t sp = stem.find("-00001-of-"); sp != std::string::npos) stem.resize(sp);
      const fs::path d = fs::path(zpack::packages_dir()) / stem;
      return fs::exists(d / (stem + ".gguf"), ec) && fs::exists(d / (stem + ".experts"), ec);
    };
    for (int pass = 0; pass < 2; ++pass) {
      for (const ModelInfo& m : catalog)
        if (m.supported && (m.alias == want || m.name == want || m.file == want)) {
          if (pass == 0 && stale(m)) break;
          return &m;
        }
      if (pass == 0) catalog = scan_models();
    }
    return nullptr;
  };

  HttpServer http;
  std::function<void(const HttpRequest&, HttpResponse&)> dispatch;
  if (wh) srvsp::attach_e4b(*wh, catalog, cur.get(), dispatch, so, cur_so);  // (Gemma 4 E4B as a transcript model)
  // (no model loaded: a model switch failed and loading the previous model again did too; caller holds swap_mu)
  auto no_model = [&](HttpResponse& rs) {
    Json err = Json::object(), r = Json::object();
    err.set("message", "no chat model is loaded (started with -m none, or the last model switch failed): name one in the request (\"model\": an id of /v1/models)");
    err.set("type", "server_error");
    r.set("error", err);
    return rs.send(503, "application/json", r.dump());
  };
  // remote access through a tunnel, with a password (tools/server_remote.h, docs/remote-access.md)
  remote::access().load();
  http.gate = [](HttpRequest& rq, HttpResponse& rs) { return remote::access().gate(rq, rs); };
  http.head_gate = [](const HttpRequest& rq) { return remote::access().head_gate(rq); };
  // the log's access line for every request the handler answered (docs/logging.md) and /metrics' HTTP counts:
  // polling (GETs that succeed: the web UI's and the launcher's /props, /health, /v1/models) only at -lv 3; an error
  // reply a warning (5xx an error) with its message
  http.on_done = [](const HttpRequest& rq, const HttpResponse& rs, double ms) {
    const int st = rs.status();
    // (the method as one of HTTP's, else "other": a client's own word must not make series or lines of its own)
    static const char* kMethods[] = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"};
    std::string method = "other";
    for (const char* m : kMethods)
      if (rq.method == m) method = m;
    zmet::http_done(method, rq.path, st);
    // (the path in one line: a control character in it (a bare line feed in the request line) would write lines
    //  of its own into the log, which zero-server reads)
    std::string path = rq.path.substr(0, 300);
    for (char& c : path)
      if ((unsigned char)c < 0x20 || c == 0x7f) c = '?';
    const bool poll = rq.method == "GET" || rq.method == "HEAD" || rq.method == "OPTIONS" || rq.path == "/tokenize" ||
                      rq.path == "/detokenize";
    if (poll) {  // (a poll that keeps failing, e.g. /health without a model: logged once per path and status, again
                 //  after it succeeded)
      static std::mutex mu;
      static std::map<std::string, int> last;
      std::lock_guard<std::mutex> lk(mu);
      if (st < 400) {
        last.erase(path);
      } else {
        if (last.size() > 256) last.clear();
        int& l = last[path];
        if (l == st) return;
        l = st;
      }
    }
    const bool quiet = st < 400 && poll;
    const int lvl = st >= 500 && !poll ? zlog::ERR : st >= 400 ? zlog::WRN : quiet ? zlog::DBG : zlog::INF;
    char sz[32];
    const double b = (double)rs.bytes();
    if (b < 1024) snprintf(sz, sizeof sz, "%.0f B", b);
    else if (b < 1048576) snprintf(sz, sizeof sz, "%.1f KB", b / 1024);
    else snprintf(sz, sizeof sz, "%.1f MB", b / 1048576);
    zlog::log(lvl, "%s %s %d %s %.0f ms %s%s%s%s\n", method.c_str(), path.c_str(), st, sz, ms,
              rq.peer.empty() ? "-" : rq.peer.c_str(), rq.remote ? " (remote)" : "", rs.error().empty() ? "" : " | ",
              rs.error().c_str());
  };
  http.handler = [&](const HttpRequest& rq, HttpResponse& rs) {
    // (Prometheus: numbers only, vLLM's names with inferzero:, tools/server_metrics.h)
    if (rq.path == "/metrics" && rq.method == "GET")
      return rs.send(200, "text/plain; version=0.0.4; charset=utf-8", zmet::render());
    // (remote access settings: this computer only; the gate refuses them for remote sessions)
    if (rq.path == "/v1/remote/config" && !rq.remote) {
      // (this computer: with --host, a peer on the network is not "remote" but must not set the password either)
      if (!rq.local_peer) return rs.send(403, "application/json", "{\"error\":{\"message\":\"remote access settings: from this computer only\"}}");
      return remote::access().config(rq, rs);
    }
    if (rq.path == "/v1/remote/status") {
      Json j = Json::object();
      j.set("remote", rq.remote);
      j.set("tools", !rq.remote || remote::access().tools_allowed());  // (run_command, folders, MCP)
      return rs.send(200, "application/json", j.dump());
    }
    // (the web UI installed as an app: a phone's "Add to home screen", a browser's "Install")
    if (rq.method == "GET" && rq.path == "/manifest.webmanifest")
      return rs.send(200, "application/manifest+json",
                     R"({"name":"inferzero","short_name":"inferzero","start_url":"/","scope":"/","display":"standalone",)"
                     R"("background_color":"#111b21","theme_color":"#00a884","icons":[{"src":"/icon.svg","sizes":"any","type":"image/svg+xml","purpose":"any"}]})");
    // the inferzero mark (site/icon.svg; the launcher draws the same in its splash, window and tray icon):
    // 4 x 4 rounded tiles, the lit ones drawing a Z; Windows' default accent, 35 % lighter on a dark theme
    if (rq.method == "GET" && rq.path == "/icon.svg")
      return rs.send(200, "image/svg+xml",
                     R"(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><style>.l{fill:#0078d4}.d{fill:#c9d3de}@)"
                     R"(media (prefers-color-scheme:dark){.l{fill:#59a7e3}.d{fill:#2b3440}}</style><rect class="l" x="1.5" y)"
                     R"(="1.5" width="13" height="13" rx="3"/><rect class="l" x="17.5" y="1.5" width="13" height="13" rx="3")"
                     R"(/><rect class="l" x="33.5" y="1.5" width="13" height="13" rx="3"/><rect class="l" x="49.5" y="1.5" w)"
                     R"(idth="13" height="13" rx="3"/><rect class="d" x="1.5" y="17.5" width="13" height="13" rx="3"/><rect )"
                     R"(class="d" x="17.5" y="17.5" width="13" height="13" rx="3"/><rect class="l" x="33.5" y="17.5" width=")"
                     R"(13" height="13" rx="3"/><rect class="d" x="49.5" y="17.5" width="13" height="13" rx="3"/><rect class)"
                     R"(="d" x="1.5" y="33.5" width="13" height="13" rx="3"/><rect class="l" x="17.5" y="33.5" width="13" he)"
                     R"(ight="13" rx="3"/><rect class="d" x="33.5" y="33.5" width="13" height="13" rx="3"/><rect class="d" x)"
                     R"(="49.5" y="33.5" width="13" height="13" rx="3"/><rect class="l" x="1.5" y="49.5" width="13" height=")"
                     R"(13" rx="3"/><rect class="l" x="17.5" y="49.5" width="13" height="13" rx="3"/><rect class="l" x="33.5)"
                     R"(" y="49.5" width="13" height="13" rx="3"/><rect class="l" x="49.5" y="49.5" width="13" height="13" r)"
                     R"(x="3"/></svg>)");
    // the web UI's files and chats on disk (tools/server_files.h): no model involved
    if (rq.path == "/v1/files" || rq.path.rfind("/v1/files/", 0) == 0 || rq.path.rfind("/files/", 0) == 0 ||
        rq.path == "/v1/chats" || rq.path.rfind("/v1/chats/", 0) == 0)
      return files_api(rq, rs);
    // the web UI's memory (tools/server_memory.h): the Settings panel, and the model's remember / forget
    if (rq.path == "/v1/memory") {
      // (a project's memory only for a project in the list: no folder made for any 16 hex digits)
      // (the query read as memo::api reads it; this computer only first, there)
      const std::string q = "&" + rq.query + "&";
      const size_t p = q.find("&project=");
      const std::string pid = p == std::string::npos ? "" : q.substr(p + 9, q.find('&', p + 9) - p - 9);
      proj::Project known;
      if (rq.local_peer && rq.method == "PUT" && memo::valid_project(pid) && !proj::find(pid, known)) {
        Json e = Json::object(), r = Json::object();
        e.set("message", "no project " + pid);
        r.set("error", e);
        return rs.send(404, "application/json", r.dump());
      }
      return memo::api(rq, rs);
    }
    // the web UI's projects (tools/server_projects.h)
    if (rq.path == "/v1/projects" || rq.path.rfind("/v1/projects/", 0) == 0) return proj::api(rq, rs);
    // MCP servers for the model (tools/server_mcp.h): programs on this computer, so for this computer only
    if (rq.path == "/v1/mcp" || rq.path.rfind("/v1/mcp/", 0) == 0) {
      auto fail = [&](int code, const std::string& m) {
        Json e = Json::object(), r = Json::object();
        e.set("message", m);
        r.set("error", e);
        return rs.send(code, "application/json", r.dump());
      };
      if (!rq.local_peer) return fail(403, "MCP servers are only for this computer (the web UI on 127.0.0.1)");
      std::string pid;
      const std::string q = "&" + rq.query + "&";
      if (const size_t p = q.find("&project="); p != std::string::npos) pid = q.substr(p + 9, q.find('&', p + 9) - p - 9);
      if (pid.empty() && !rq.body.empty()) try { pid = Json::parse(rq.body)["project"].str(); } catch (...) {}
      if (!pid.empty()) {
        proj::Project known;
        if (!memo::valid_project(pid)) return fail(400, "project: 16 hex digits");
        if (!proj::find(pid, known)) return fail(404, "no project " + pid);
      }
      return mcp::api(rq, rs, pid, [&] { return rs.client_gone(); });
    }
    if (rq.method == "POST" && (rq.path == "/v1/tools/remember" || rq.path == "/v1/tools/forget")) {
      auto fail = [&](int code, const std::string& m) {
        Json e = Json::object(), r = Json::object();
        e.set("message", m);
        r.set("error", e);
        return rs.send(code, "application/json", r.dump());
      };
      if (!rq.local_peer) return fail(403, "the memory is only for this computer (the web UI on 127.0.0.1)");
      try {
        const Json j = Json::parse(rq.body);
        const std::string project = j["project"].as_string();
        if (!project.empty() && !memo::valid_project(project)) return fail(400, "project: 16 hex digits");
        proj::Project known;
        if (!project.empty() && !proj::find(project, known)) return fail(404, "no project " + project);
        if (rq.path == "/v1/tools/remember") return rs.send(200, "application/json", memo::remember(project, j["fact"].as_string()).dump());
        // (forget in a project's chat: the project's notes, else the global ones)
        Json r;
        try {
          r = memo::forget(project, j["match"].as_string());
        } catch (const std::exception& e) {
          if (project.empty() || std::string(e.what()).rfind("no note contains", 0) != 0) throw;
          r = memo::forget("", j["match"].as_string());
          r.set("scope", "global");
        }
        return rs.send(200, "application/json", r.dump());
      } catch (const std::exception& e) {
        return fail(400, e.what());
      }
    }
    if (rq.method == "POST" && (rq.path == "/v1/tools/create_file" || rq.path == "/v1/tools/list_files" ||
                                rq.path == "/v1/tools/minutes_templates" || rq.path == "/v1/tools/minutes_docx" ||
                                rq.path == "/v1/tools/read_file" || rq.path == "/v1/tools/search_file" ||
                                rq.path == "/v1/tools/search_chats" || rq.path == "/v1/tools/digest_file" ||
                                rq.path == "/v1/tools/run_command")) {
      int ctx = 0;  // (the loaded model's context; not waited for during a model switch: then the usual limits)
      {
        std::shared_lock<std::shared_mutex> lk(swap_mu, std::try_to_lock);
        if (lk.owns_lock() && cur) ctx = cur->eng->ctx;
      }
      // (read_file's scanned pages: a chat request of the loaded vision model on this connection, so the web UI's
      //  stop, which closes it, cancels the transcription too)
      FileVision vis;
      vis.model = [&]() -> std::string {
        std::shared_lock<std::shared_mutex> lk(swap_mu);
        return cur && cur->srv && cur->srv->has_vision() ? cur->alias : "";
      };
      vis.see = [&](const std::string& bytes, const std::string& mime) {
        Json url = Json::object(), img = Json::object(), txt = Json::object(), content = Json::array(), msg = Json::object(),
             msgs = Json::array(), kw = Json::object(), body = Json::object();
        url.set("url", "data:" + mime + ";base64," + apic::base64_encode(bytes));
        img.set("type", "image_url");
        img.set("image_url", url);
        txt.set("type", "text");
        txt.set("text", kTranscribe);
        content.push(img);
        content.push(txt);
        msg.set("role", "user");
        msg.set("content", content);
        msgs.push(msg);
        kw.set("enable_thinking", false);
        body.set("model", vis.model());
        body.set("messages", msgs);
        body.set("temperature", 0.0);
        body.set("max_tokens", 4096);
        body.set("stream", false);
        body.set("chat_template_kwargs", kw);
        HttpRequest rq2 = rq;
        rq2.method = "POST";
        rq2.path = "/v1/chat/completions";
        rq2.query.clear();
        rq2.body = body.dump();
        int status = 0;
        std::string out;
        rs.on_send = [&](int st, const std::string& b) { status = st; out = b; };
        {
          struct Unhook { HttpResponse& r; ~Unhook() { r.on_send = nullptr; } } unhook{rs};  // (also on a throw)
          dispatch(rq2, rs);
        }
        Json j;
        Json::try_parse(out, j);
        if (status != 200) throw std::runtime_error(j["error"]["message"].str("HTTP " + std::to_string(status)));
        return j["choices"][0]["message"]["content"].str();
      };
      vis.ask = [&](const std::string& prompt, int max_tokens) {
        std::string alias;
        {
          std::shared_lock<std::shared_mutex> lk(swap_mu);
          if (cur) alias = cur->alias;
        }
        Json msg = Json::object(), msgs = Json::array(), kw = Json::object(), body = Json::object();
        msg.set("role", "user");
        msg.set("content", prompt);
        msgs.push(msg);
        kw.set("enable_thinking", false);
        body.set("model", alias);
        body.set("messages", msgs);
        body.set("temperature", 0.2);
        body.set("max_tokens", max_tokens);
        body.set("stream", false);
        body.set("chat_template_kwargs", kw);
        HttpRequest rq2 = rq;
        rq2.method = "POST";
        rq2.path = "/v1/chat/completions";
        rq2.query.clear();
        rq2.body = body.dump();
        int status = 0;
        std::string out;
        rs.on_send = [&](int st, const std::string& b) { status = st; out = b; };
        {
          struct Unhook { HttpResponse& r; ~Unhook() { r.on_send = nullptr; } } unhook{rs};
          dispatch(rq2, rs);
        }
        Json j;
        Json::try_parse(out, j);
        if (status != 200) throw std::runtime_error(j["error"]["message"].str("HTTP " + std::to_string(status)));
        return j["choices"][0]["message"]["content"].str();
      };
      return file_tools(rq, rs, terminal, ctx, vis);
    }
    // the web UI's tools (tools/web_tools.h): no model involved, so they answer while a model loads or generates
    if (rq.method == "POST" && (rq.path == "/v1/tools/web_search" || rq.path == "/v1/tools/web_fetch" || rq.path == "/v1/tools/web_view")) {
      auto fail = [&](int st, const std::string& m) {
        Json er = Json::object(), r = Json::object();
        er.set("message", m);
        r.set("error", er);
        return rs.send(st, "application/json", r.dump());
      };
      if (!web_tools) return fail(404, "web tools are off (--no-web-tools)");
      Json in;
      if (!Json::try_parse(rq.body, in) || !in.is_object()) return fail(400, "invalid JSON body");
      Json r = Json::object();
      if (rq.path == "/v1/tools/web_search") {
        const std::string q = in["query"].str();
        if (q.empty()) return fail(400, "query required");
        std::string used, err;
        const auto res = webt::search(q, (size_t)std::clamp<int64_t>(in["max_results"].as_int(8), 1, 20), in["engine"].str("auto"), &used, &err);
        if (res.empty()) return fail(502, "search failed: " + err);
        Json a = Json::array();
        for (const webt::Result& x : res) {
          Json o = Json::object();
          o.set("title", x.title);
          o.set("url", x.url);
          size_t n = std::min<size_t>(x.snippet.size(), 360);  // (some snippets run to whole paragraphs)
          while (n < x.snippet.size() && ((unsigned char)x.snippet[n] & 0xc0) == 0x80) --n;
          o.set("snippet", x.snippet.substr(0, n) + (n < x.snippet.size() ? "..." : ""));
          a.push(o);
        }
        r.set("query", q);
        r.set("engine", used);
        r.set("results", a);
        return rs.send(200, "application/json", r.dump());
      }
      const std::string url = in["url"].str();
      if (url.empty()) return fail(400, "url required");
      // at most 4 fetches or views at once (each may hold a 60 MB file, or two Edge processes): the others wait up to
      // 2 minutes, then 429; 64 connections at once could otherwise hold 6 GB of bodies
      struct Gate {
        static std::mutex& mu() { static std::mutex m; return m; }
        static std::condition_variable& cv() { static std::condition_variable c; return c; }
        static int& n() { static int k = 0; return k; }
        bool in = false;
        bool enter() {
          std::unique_lock<std::mutex> lk(mu());
          in = cv().wait_for(lk, std::chrono::minutes(2), [] { return n() < 4; });
          if (in) ++n();
          return in;
        }
        ~Gate() {
          if (!in) return;
          { std::lock_guard<std::mutex> lk(mu()); --n(); }
          cv().notify_one();
        }
      } gate;
      if (!gate.enter()) return fail(429, "too many pages being fetched at once; try again later");
      std::string title, text, final_url = url, gh_err;
      // (an issue or pull request on GitHub: the thread and its comments from GitHub's API, tools/web_github.h; when
      //  the API fails, the page itself)
      if (rq.path == "/v1/tools/web_view") {
        // the page as Edge shows it (tools/web_view.h): a screenshot of one screen (saved in the chat's files, shown to
        // a model that sees images by the web UI) and the text of the page after its scripts ran
        // (the first URL's host public and its port a web port, parsed as get() parses a URL; what the page loads goes
        //  through the proxy's own check)
        std::string vhost;
        int vport = 0;
        if (!webt::url_host_port(url, vhost, vport)) return fail(400, "an http(s) URL");
        if (!webt::web_port_ok(vport)) return fail(400, "port " + std::to_string(vport) + " refused (a port of another service, not the web)");
        if (const std::string why = webt::check_host(vhost); !why.empty()) return fail(400, why);
        const int screen = (int)std::clamp<int64_t>(in["screen"].as_int(0), 0, 5);
        const webv::View v = webv::render(url, screen);
        if (v.png.empty() && v.html.empty()) return fail(502, v.error.empty() ? "the page could not be shown" : v.error);
        int hidden = 0;
        const bool all_links = in["links"].is_bool() && in["links"].as_bool();
        // (its first 4 MB, as web_fetch reads a page: the conversion makes ~4 copies of it, self audit O3)
        text = v.html.empty() ? std::string() : webt::html_to_text(v.html.substr(0, kWebTextMax), &title, url, all_links, &hidden);
        if (v.html.size() > kWebTextMax || v.html_cut) text = "[the page's HTML was longer than 4 MB: its text from the first 4 MB]\n" + text;
        if (!v.note.empty()) text = "[" + v.note + "]\n" + text;
        if (hidden > 0)
          text = "[" + std::to_string(hidden) + (hidden == 1 ? " link to another page is" : " links to other pages are") +
                 " shown as text only: to explore, web_view or web_fetch this URL again with \"links\": true for the URLs, "
                 "then open the ones you need]\n" + text;
        if (!v.png.empty()) {
          const std::string chat = files::valid_id(in["chat_id"].str()) ? in["chat_id"].str() : "";
          files::Info fi;
          try {
            fi = files::store().put(files::safe_name("screen " + std::to_string(screen + 1) + " of " + vhost + ".png"), v.png, chat);
          } catch (const std::exception& e) {
            return fail(500, std::string("cannot save the screenshot: ") + e.what());
          }
          r.set("file_id", fi.id);
          r.set("filename", fi.name);
          r.set("kind", "image");
          r.set("bytes", (double)fi.bytes);
          text = "[screen " + std::to_string(screen + 1) + " of the page (1280 x 900, from the top" +
                 (screen ? ", " + std::to_string(screen * 900) + " pixels down" : std::string()) + "): saved in this chat's files as file_id \"" +
                 fi.id + "\"" + (screen < 5 ? "; web_view with \"screen\": " + std::to_string(screen + 1) +
                 " shows the next one (blank where the page has ended)" : std::string("; the last screen web_view shows")) + "]\n" + text;
        } else {
          text = "[no screenshot: " + v.error + "]\n" + text;
        }
        r.set("screen", (double)screen);
        r.set("source", "Microsoft Edge (the page as a browser shows it)");
      } else if (webt::github_thread(url, title, text, gh_err)) {
        r.set("source", "GitHub API (the issue and every comment)");
      } else {
        if (!gh_err.empty()) r.set("note", gh_err + "; the page itself was read instead (it may lack the comments)");
        // (up to 60 MB for an audio file, 20 MB for a document or an image; a page or a text is read from its first
        //  4 MB, as before; the whole request within 2 minutes: a large file)
        // (the size each kind may have, decided from the headers before the body: a refused type reads nothing, a
        //  document over 20 MB stops there, self audit O2 / G3)
        // (a file is whole or useless: one that announces more than its limit is not read at all; a page or a text is
        //  read up to its 4 MB as before, self audit R1)
        const webt::Page pg = webt::get(url, kWebAudioMax, 120000, "en-US,en;q=0.9",
                                        "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
                                        [](const webt::Page& h, bool& whole) {  // (each looked up once, R20)
                                          const std::string k = web_kind(h.content_type, web_file_name(h));
                                          whole = !k.empty() && k != "html" && k != "text";
                                          return k.empty() ? (size_t)0 : !whole ? kWebTextMax : web_audio(k) ? kWebAudioMax : kWebFileMax;
                                        });
        if (!pg.error.empty()) return fail(502, pg.error);
        if (pg.status != 200) return fail(502, "HTTP " + std::to_string(pg.status) + " from " + pg.url);
        final_url = pg.url;
        const std::string name = web_file_name(pg);
        // what it is from its headers first: a refused kind (nothing was read), then a file too large or not arrived
        // whole (413 / 504, self audit R2 / R3), and only then its bytes (web_file_type: a file's first bytes)
        const std::string kind = web_kind(pg.content_type, name);
        if (kind.empty())
          return fail(415, "unsupported content type " + (pg.content_type.empty() ? std::string("(none)") : pg.content_type) +
                               (name.empty() ? "" : " (" + name + ")") +
                               ": web pages, text, PDF, Word, Excel, PowerPoint, images and audio only");
        const bool file = kind != "html" && kind != "text";
        if (file) {
          const size_t cap = web_audio(kind) ? kWebAudioMax : kWebFileMax;
          if (pg.too_big || pg.cut || pg.body.size() > cap)
            return fail(413, "the file is larger than " + std::to_string(cap >> 20) + " MB: not fetched");
          if (pg.timed_out) return fail(504, "the file did not arrive whole (stopped, or not within 2 minutes): not fetched");
        }
        const std::string ty = web_file_type(pg.content_type, name, pg.body);
        if (ty.empty())
          return fail(415, "the content is not what its type (" + kind + ") says" + (name.empty() ? "" : " (" + name + ")"));
        if (ty == "html") {
          // (links to files with their URLs; the other links' URLs only when asked: "links": true)
          const bool all_links = in["links"].is_bool() && in["links"].as_bool();
          int hidden = 0;
          text = webt::html_to_text(pg.body.substr(0, kWebTextMax), &title, pg.url, all_links, &hidden);
          if (hidden > 0)
            text = "[" + std::to_string(hidden) + (hidden == 1 ? " link to another page is" : " links to other pages are") +
                   " shown as text only: to explore, web_fetch this URL again with \"links\": true for the URLs, then open "
                   "the ones you need]\n" + text;
        }
        else if (ty == "text") {
          text = pg.body.substr(0, kWebTextMax);
          title = name;  // (a text file has no title of its own: its name, an attachment's "log.txt")
        }
        else {
          // a document or an image: saved into this chat's files (a file card for the user), its text read from there
          // with the reader read_file uses (PDF, Word, Excel, PowerPoint); an image is shown to a model that sees
          // images by the web UI, after this result
          const bool is_audio = web_audio(ty);
          const std::string chat = files::valid_id(in["chat_id"].str()) ? in["chat_id"].str() : "";
          std::string fname = name.empty() ? "download" : name;  // (its extension the one of its kind)
          const std::string fe = files::ext_of(fname);
          if (fe != ty && !(ty == "jpg" && fe == "jpeg")) fname += "." + ty;
          files::Info fi;
          try {
            fi = files::store().put(fname, pg.body, chat);
          } catch (const std::exception& e) {
            return fail(500, std::string("cannot save the file: ") + e.what());
          }
          r.set("file_id", fi.id);
          r.set("filename", fi.name);
          r.set("kind", fi.kind);
          r.set("bytes", (double)fi.bytes);
          title = fi.name;
          if (is_audio) {
            // (an audio file: written out by the speech service, Qwen3-ASR or Whisper, as a chat's voice clip is; the
            //  same clip again comes from its cache)
            const std::string head = "[an audio file: " + fi.name + ", " + std::to_string((fi.bytes + 1023) / 1024) +
                                     " KB; saved in this chat's files as file_id \"" + fi.id + "\"";
            if (!wh) {
              text = head + "; not transcribed: the server runs no speech to text (--whisper / --asr)]";
            } else {
              double secs = 0;
              int st = 0;
              try {
                // (30 minutes at most: the whole clip is decoded in RAM, ~115 MB of samples then; the service's own
                //  3 hours would be ~700 MB, self audit O1)
                const std::string t = wh->transcribe_clip(pg.body, secs, st, [&] { return rs.client_gone(); }, kWebAudioSeconds);
                char len[48];
                snprintf(len, sizeof len, "%d:%02d", (int)secs / 60, (int)secs % 60);
                text = head + "; " + len + " long; its speech as text, by " + wh->clip_model() + "]\n" +
                       (t.empty() ? std::string("(no speech heard)") : t);
              } catch (const std::exception& e) {
                if (st == 429) return fail(429, e.what());
                text = head + "; it could not be transcribed: " + e.what() + "]";
              }
            }
          } else if (fi.kind == "image") {
            int w = 0, h = 0;
            files::image_size(pg.body, w, h);
            text = "[an image: " + fi.name + (w > 0 ? ", " + std::to_string(w) + "x" + std::to_string(h) : std::string()) +
                   ", " + std::to_string((fi.bytes + 1023) / 1024) + " KB; saved in this chat's files as file_id \"" + fi.id + "\"]";
          } else {
            const auto ix = files::doc_index(fi);
            const std::string head = "[a " + ty + " file: " + fi.name + ", " + std::to_string((fi.bytes + 1023) / 1024) +
                                     " KB; saved in this chat's files as file_id \"" + fi.id + "\" (read_file and search_file read it too)" +
                                     (ix && ix->doc.pages > 0 ? "; " + std::to_string(ix->doc.pages) + " pages" : std::string()) + "]\n";
            if (!ix || ix->doc.paras.empty())
              text = head + "[" + (ix && !ix->doc.note.empty() ? ix->doc.note : std::string("no text could be read from it")) + "]";
            else text = head + (ix->doc.note.empty() ? "" : "[" + ix->doc.note + "]\n") + ix->text(0, ix->doc.paras.size());
          }
        }
        // (a page or text cut at its 4 MB or at the 2-minute deadline says so: the model is not told it read all, G5)
        if (!file && (pg.cut || pg.timed_out))
          text = "[only the first " + std::to_string(pg.body.size() >> 10) + " KB of it could be read (" +
                 (pg.timed_out ? "the rest did not arrive: stopped, or not within 2 minutes" : "4 MB at most") + ")]\n" + text;
      }
      // a window of the text: offset, max_chars, never more than the model's context takes in one read (as read_file;
      // before, 12,000 characters a fetch whatever the context) (cut on a UTF-8 character boundary)
      int ctx = 0;
      {
        std::shared_lock<std::shared_mutex> lk(swap_mu, std::try_to_lock);
        if (lk.owns_lock() && cur) ctx = cur->eng->ctx;
      }
      const size_t total = text.size(),
                   max_chars = std::min((size_t)std::clamp<int64_t>(in["max_chars"].as_int(12000), 500, 100000), files::read_budget(ctx));
      size_t off = (size_t)std::clamp<int64_t>(in["offset"].as_int(0), 0, (int64_t)total), end = std::min(total, off + max_chars);
      while (off < total && ((unsigned char)text[off] & 0xc0) == 0x80) ++off;
      while (end < total && ((unsigned char)text[end] & 0xc0) == 0x80) --end;
      r.set("url", final_url);
      r.set("title", title);
      r.set("content", text.substr(off, end - off));
      r.set("offset", (double)off);
      r.set("total_chars", (double)total);
      if (end < total) r.set("next_offset", (double)end);
      return rs.send(200, "application/json", r.dump());
    }
    // conversations at once (the web UI's Settings): the loaded model again with N slots, its context shared out
    // (--ctx auto: each slot 128K / N for streamed MoE models, what fits N for the others). Waits for the running
    // requests; this computer only. Later model switches keep N.
    if (rq.method == "POST" && rq.path == "/v1/server/slots") {
      auto fail = [&](int st, const std::string& m) {
        Json er = Json::object(), r = Json::object();
        er.set("message", m);
        r.set("error", er);
        return rs.send(st, "application/json", r.dump());
      };
      if (!rq.local_peer) return fail(403, "changing the slots is only for this computer");
      Json in;
      if (!Json::try_parse(rq.body, in) || !in["slots"].is_number()) return fail(400, "expected {\"slots\": 1-8}");
      const int n = (int)in["slots"].as_clamped(1, 1, 8);  // (clamped before the cast: 1e308 is valid JSON)
      std::unique_lock<std::shared_mutex> lk(swap_mu);
      if (!cur) return no_model(rs);
      auto reply = [&] {
        Json r = Json::object();
        r.set("slots", cur->eng->slot_count());
        r.set("n_ctx", cur->eng->ctx);
        r.set("model", cur->alias);
        return rs.send(200, "application/json", r.dump());
      };
      if (cur->eng->slot_count() == n) return reply();
      const std::string path = cur->path, al = cur->alias;
      const ServeOpts prev = cur_so;
      const auto t0 = clk::now();
      zlog::eprintf("slots: %d -> %d, loading %s again\n", cur->eng->slot_count(), n, al.c_str());
      if (cur->srv) cur->srv->save_held();
      cur.reset(), zmet::unloaded();
      ServeOpts s2 = prev;
      s2.eo.slots = n;
      s2.ctx_auto = true;
      s2.warm_prompt = false;
      try {
        cur = load_model(path, al, s2);
        cur_so = s2;
        so.eo.slots = n;  // (the next model switches too)
        zlog::eprintf("slots: %d, loaded in %.1f s\n", n, ms_since(t0) / 1000);
        return reply();
      } catch (const std::exception& e) {
        zlog::eprintf("slots: could not load %s with %d slots: %s; loading it as before\n", al.c_str(), n, e.what());
        std::string also;
        try {
          cur = load_model(path, al, prev);
        } catch (const std::exception& e2) {
          cur.reset(), zmet::unloaded();
          also = std::string("; loading it as before failed too: ") + e2.what();
        }
        return fail(500, std::string("could not load the model with ") + std::to_string(n) + " slots: " + e.what() + also);
      }
    }
    // GPU memory (the web UI's Settings): what is loaded and its GPU bytes, the context the chat model got, Windows'
    // budget for this process and what is in use. POST /v1/server/unload {"which": "model" | "whisper" | "asr"} frees
    // one; POST /v1/server/load {"which": "whisper" | "asr"} loads it again (a chat model loads when a request names
    // it). This computer only. The next chat model load counts the speech models' memory as it is then (--ctx auto).
    if (rq.path == "/v1/server/memory" || rq.path == "/v1/server/unload" || rq.path == "/v1/server/load") {
      auto fail = [&](int st, const std::string& m) {
        Json er = Json::object(), r = Json::object();
        er.set("message", m);
        r.set("error", er);
        return rs.send(st, "application/json", r.dump());
      };
      // (an unloaded model comes back when a request names it: only with model switching and a model of the catalog)
      auto can_unload = [&] {
        if (!swap || !cur) return false;
        for (const ModelInfo& m : catalog)
          if (m.alias == cur->alias || same_file(m.path, cur->path)) return true;
        return false;
      };
      auto state = [&] {  // (swap_mu held, shared or alone)
        Json r = Json::object(), c = Json::object();
        if (cur) {
          c.set("can_unload", can_unload());
          c.set("model", cur->alias);
          c.set("n_ctx", cur->eng->ctx);
          c.set("gpu_bytes", (double)cur->eng->gpu_bytes);
        }
        r.set("chat", cur ? c : Json());
        Json s = Json::object();
        if (wh) {
          s.set("whisper", wh->has_whisper());
          s.set("whisper_can_load", wh->can_load("whisper"));
          s.set("asr", wh->asr_loaded());
          s.set("asr_can_load", wh->can_load("asr"));
          // (Gemma 4 E4B as a transcript model: there with Whisper and Qwen3-ASR unloaded too, when /props has no
          //  speech to text and so no chat_asr)
          if (!wh->chat_asr_id.empty() && wh->chat_asr) s.set("chat_asr", wh->chat_asr_id);
          s.set("gpu_bytes", (double)wh->gpu_bytes());
        }
        r.set("speech", wh ? s : Json());
        uint64_t b = 0, u = 0;
        if (zrt::video_budget(b, u)) {
          r.set("budget_bytes", (double)b);
          r.set("used_bytes", (double)u);
        }
        return r;
      };
      if (rq.method == "GET" && rq.path == "/v1/server/memory") {
        std::shared_lock<std::shared_mutex> lk(swap_mu);
        return rs.send(200, "application/json", state().dump());
      }
      if (rq.method != "POST") return fail(405, "GET /v1/server/memory, POST /v1/server/unload or /load");
      if (!rq.local_peer) return fail(403, "only for this computer");
      Json in;
      Json::try_parse(rq.body, in);
      const std::string which = in["which"].str();
      const bool load = rq.path == "/v1/server/load";
      if (which == "model") {
        if (load) return fail(400, "a chat model loads when a request names it");
        std::unique_lock<std::shared_mutex> lk(swap_mu);
        if (cur && !can_unload()) return fail(409, "this model could not be loaded again by a request (--no-swap, or a file "
                                                    "outside the model folders): restart the server instead");
        if (cur) {
          zlog::eprintf("unloading %s (asked)\n", cur->alias.c_str());
          if (cur->srv) cur->srv->save_held();
          unloaded_path = cur->path;
          cur.reset(), zmet::unloaded();
        }
        return rs.send(200, "application/json", state().dump());
      }
      if (which != "whisper" && which != "asr") return fail(400, "which: \"model\", \"whisper\" or \"asr\"");
      if (!wh) return fail(400, "this server runs no speech to text (--whisper off --asr off)");
      std::string bad;
      zlog::eprintf("%s %s (asked)\n", load ? "loading" : "unloading", which == "asr" ? "Qwen3-ASR" : "Whisper");
      if (!wh->set_loaded(which, load, wh->gpu_turn, bad)) return fail(409, bad);
      std::unique_lock<std::shared_mutex> lk(swap_mu);  // (the memory --ctx auto keeps for speech, /props: as it is now)
      so.whisper_bytes = wh->gpu_bytes() ? wh->gpu_bytes() + (128u << 20) : 0;
      so.whisper_props = wh->props();
      cur_so.whisper_bytes = so.whisper_bytes;
      cur_so.whisper_props = so.whisper_props;
      if (cur && cur->srv) cur->srv->whisper_props_ = so.whisper_props, cur->srv->speech_reserve_ = so.whisper_bytes;
      return rs.send(200, "application/json", state().dump());
    }
    // other API formats (tools/api_compat.h): translated into a chat request; the answer is translated back
    static const std::map<std::string, std::string> kinds = {
        {"/v1/completions", "completions"}, {"/completions", "completions"}, {"/api/chat", "ollama_chat"},
        {"/api/generate", "ollama_generate"}, {"/v1/messages", "anthropic"}, {"/messages", "anthropic"},
        {"/v1/responses", "responses"}, {"/responses", "responses"}};
    const auto kit = kinds.find(rq.path);
    if (rq.method == "POST" && kit != kinds.end()) {
      Json in, chat_body;
      std::string e;
      if (!Json::try_parse(rq.body, in)) return rs.send(400, "application/json", "{\"error\":{\"message\":\"invalid JSON body\"}}");
      if (!apic::to_chat(kit->second, in, chat_body, e)) {
        Json er = Json::object(), r = Json::object();
        er.set("message", e);
        er.set("type", "invalid_request_error");
        r.set("error", er);
        return rs.send(400, "application/json", r.dump());
      }
      if (chat_body["model"].str().empty()) {
        std::shared_lock<std::shared_mutex> lk(swap_mu);
        if (!cur) return no_model(rs);
        chat_body.set("model", cur->alias);
      }
      HttpRequest rq2 = rq;
      rq2.path = "/v1/chat/completions";
      rq2.body = chat_body.dump();
      apic::Adapter ad(kit->second, in, rs);
      return dispatch(rq2, rs);
    }
    // Anthropic token count
    if (rq.method == "POST" && (rq.path == "/v1/messages/count_tokens" || rq.path == "/messages/count_tokens")) {
      Json in, chat_body;
      std::string e;
      if (!Json::try_parse(rq.body, in) || !apic::to_chat("anthropic", in, chat_body, e))
        return rs.send(400, "application/json", "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"bad request\"}}");
      std::shared_lock<std::shared_mutex> lk(swap_mu);
      if (!cur) return no_model(rs);
      const int n = cur->srv->count_tokens(chat_body);
      if (n < 0)  // (the template could not render these messages)
        return rs.send(400, "application/json", "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"the chat template cannot render these messages\"}}");
      Json r = Json::object();
      r.set("input_tokens", n);
      return rs.send(200, "application/json", r.dump());
    }
    // the saved transcripts and their queue (the Transcript tab; tools/server_whisper.h)
    if (rq.path == "/v1/transcripts" || rq.path.rfind("/v1/transcripts/", 0) == 0 || rq.path == "/v1/live" ||
        rq.path.rfind("/v1/live/", 0) == 0) {
      if (!wh)
        return rs.send(400, "application/json", "{\"error\":{\"message\":\"Whisper is off: start the server with --whisper auto\",\"type\":\"invalid_request_error\"}}");
      return rq.path.rfind("/v1/live", 0) == 0 ? wh->live(rq, rs) : wh->transcripts(rq, rs);
    }
    // OpenAI audio transcription / translation (tools/server_speech.h)
    {
      srvsp::AudioRoute ar;
      ar.wh = wh.get();
      ar.is_chat_model = [&](const std::string& want) {
        std::shared_lock<std::shared_mutex> lk(swap_mu);
        bool r = cur && want == cur->alias;
        for (const ModelInfo& m : catalog) r = r || (m.supported && (m.alias == want || m.name == want || m.file == want));
        return r;
      };
      ar.loaded_alias = [&]() -> std::string {
        std::shared_lock<std::shared_mutex> lk(swap_mu);
        return cur ? cur->alias : std::string();
      };
      ar.no_model = [&](HttpResponse& r) { no_model(r); };
      ar.dispatch = &dispatch;
      if (srvsp::audio_route(rq, rs, ar)) return;
    }
    // Ollama: the model list, one model's details, the version, the loaded model
    if (rq.path == "/api/version") return rs.send(200, "application/json", "{\"version\":\"0.9.0\"}");
    if (rq.path == "/api/tags" || rq.path == "/api/ps" || (rq.path == "/api/show" && rq.method == "POST")) {
      std::shared_lock<std::shared_mutex> lk(swap_mu);
      if (!cur) return no_model(rs);
      auto entry =[&](const ModelInfo* m, const std::string& alias, bool vision, bool mtp) {
        Json d = Json::object();
        d.set("format", "gguf");
        d.set("family", m ? m->arch : std::string());
        Json fam = Json::array();
        fam.push(m ? m->arch : std::string());
        d.set("families", fam);
        d.set("parameter_size", "");
        d.set("quantization_level", "");
        Json j = Json::object();
        j.set("name", alias + ":latest");
        j.set("model", alias + ":latest");
        j.set("modified_at", apic::iso_now());
        j.set("size", (double)(m ? m->bytes : 0));
        j.set("digest", "");
        j.set("details", d);
        (void)vision;
        (void)mtp;
        return j;
      };
      if (rq.path == "/api/show") {
        Json in;
        Json::try_parse(rq.body, in);
        const std::string want = apic::strip_tag(in["model"].str(in["name"].str()));
        const ModelInfo* m = nullptr;
        for (const ModelInfo& c : catalog)
          if (c.supported && (c.alias == want || c.name == want)) m = &c;
        if (!m && want != cur->alias) return rs.send(404, "application/json", "{\"error\":\"model not found\"}");
        const bool loaded = want == cur->alias;
        Json r = Json::object();
        r.set("modelfile", "");
        r.set("parameters", "");
        r.set("template", "");
        Json e = entry(m, want, m ? m->vision : cur->srv->has_vision(), m ? m->mtp : false);
        r.set("details", e["details"]);
        Json mi = Json::object();
        const std::string arch = m ? m->arch : std::string("unknown");
        mi.set("general.architecture", arch);
        mi.set(arch + ".context_length", loaded ? (double)cur->eng->ctx : (double)(m ? m->ctx_train : 0));
        r.set("model_info", mi);
        Json caps = Json::array();
        caps.push("completion");
        caps.push("tools");
        caps.push("thinking");
        if (m ? m->vision : cur->srv->has_vision()) caps.push("vision");
        r.set("capabilities", caps);
        r.set("modified_at", apic::iso_now());
        return rs.send(200, "application/json", r.dump());
      }
      Json ms = Json::array();
      for (const ModelInfo& m : catalog)
        if (m.supported && (rq.path == "/api/tags" || m.alias == cur->alias)) ms.push(entry(&m, m.alias, m.vision, m.mtp));
      if ((rq.path == "/api/ps" || rq.path == "/api/tags") && !ms.size())  // (--no-swap: no catalog, the loaded model)
        ms.push(entry(nullptr, cur->alias, false, false));
      Json r = Json::object();
      r.set("models", ms);
      return rs.send(200, "application/json", r.dump());
    }
    dispatch(rq, rs);
  };
  dispatch = [&](const HttpRequest& rq, HttpResponse& rs) {
    const bool chat_post = rq.method == "POST" && rq.path == "/v1/chat/completions";
    // (answered with no model too, below: the speech service's state and the models a chat can load)
    const bool info = rq.method == "GET" && (rq.path == "/" || rq.path == "/index.html" || rq.path == "/health" || rq.path == "/props" ||
                                             (swap && (rq.path == "/v1/models" || rq.path == "/models")));
    {  // (no model: -m none, or a switch failed and so did loading the previous model again; a chat request naming
       //  one loads it)
      std::shared_lock<std::shared_mutex> lk(swap_mu);
      if (!cur && !(swap && chat_post) && !info) {
        Json err = Json::object(), r = Json::object();
        err.set("message", "no chat model is loaded (started with -m none, or the last model switch failed): name one in the request (\"model\": an id of /v1/models)");
        err.set("type", "server_error");
        r.set("error", err);
        return rs.send(503, "application/json", r.dump());
      }
    }
    // audio the model does not hear itself (or "speech": "whisper" in the request): Whisper writes it as text first,
    // here, before the model's lock (Whisper's GPU turns take it), for every API the adapters bring through here;
    // "speech": "model" keeps it as audio for a model that hears it (Gemma 4 E4B / 12B)
    if (chat_post && wh) {
      Json in;
      if (Json::try_parse(rq.body, in) && srvwh::Service::has_audio_parts(in)) {
        bool hears = false;
        {
          std::shared_lock<std::shared_mutex> lk(swap_mu);
          hears = cur && cur->srv->has_audio();
        }
        const std::string speech = in["speech"].str();
        // (nothing of the service loaded and nothing asked of it: as without the service, the request goes to the model
        //  it names, which may hear it; every server has the service now, to load speech models from Settings)
        const bool none = speech.empty() && !wh->has_whisper() && !wh->asr_loaded();
        if (!none && (speech == "whisper" || (speech != "model" && !hears))) {
          std::string bad;
          int st = 400;
          if (!wh->audio_to_text(in, bad, st, [&] { return rs.client_gone(); }))
            return rs.send(st, "application/json", "{\"error\":{\"message\":" + Json(bad).dump() + ",\"type\":\"" +
                                                       (st == 429 ? "rate_limit_error" : "invalid_request_error") + "\"}}");
          HttpRequest rq2 = rq;
          rq2.body = in.dump();
          return dispatch(rq2, rs);
        }
      }
    }
    // n > 1 (several answers to one chat request; here in dispatch, which the API adapters call too, so that
    // /v1/completions takes n): the request runs n times (the prompt is cached after the first;
    // a given seed changes per answer); not with streaming
    if (rq.method == "POST" && rq.path == "/v1/chat/completions") {
      Json in;
      if (Json::try_parse(rq.body, in) && in["n"].as_int(1) > 1) {
        if (in["stream"].as_bool(false))
          return rs.send(400, "application/json", "{\"error\":{\"message\":\"n > 1 is supported without streaming\",\"type\":\"invalid_request_error\"}}");
        const int n = (int)std::min<int64_t>(in["n"].as_int(1), 16);
        // (a request through an API adapter, /v1/completions: its hook is kept around each run and takes the merged
        //  answer, so the reply is in that API's format)
        const auto outer = rs.on_send;
        Json merged, choices = Json::array();
        int64_t prompt_tok = 0, compl_tok = 0;
        for (int k = 0; k < n; ++k) {
          Json one = in;
          one.set("n", 1);
          if (in["seed"].is_number()) one.set("seed", in["seed"].as_int() + k);
          HttpRequest rq2 = rq;
          rq2.body = one.dump();
          int status = 0;
          std::string body;
          rs.on_send = [&](int st, const std::string& b) { status = st; body = b; };
          {
            struct Unhook {  // (also on a throw)
              HttpResponse& r;
              const std::function<void(int, const std::string&)>& o;
              ~Unhook() { r.on_send = o; }
            } unhook{rs, outer};
            dispatch(rq2, rs);
          }
          if (status != 200) return rs.send(status ? status : 500, "application/json", body);
          Json r;
          Json::try_parse(body, r);
          if (merged.is_null()) merged = r;
          Json c = r["choices"][0];
          c.set("index", k);
          choices.push(c);
          prompt_tok = r["usage"]["prompt_tokens"].as_int();
          compl_tok += r["usage"]["completion_tokens"].as_int();
        }
        merged.set("choices", choices);
        Json u = Json::object();
        u.set("prompt_tokens", prompt_tok);
        u.set("completion_tokens", compl_tok);
        u.set("total_tokens", prompt_tok + compl_tok);
        merged.set("usage", u);
        return rs.send(200, "application/json", merged.dump());
      }
    }
    // (no chat model: the web UI, and the speech service's state)
    if (rq.method == "GET" && (rq.path == "/" || rq.path == "/index.html" || rq.path == "/health" || rq.path == "/props")) {
      std::shared_lock<std::shared_mutex> lk(swap_mu);
      if (!cur) {
        if (rq.path == "/health") return rs.send(200, "application/json", "{\"status\":\"ok\"}");
        if (rq.path != "/props") {
          std::ifstream f(so.webui + "/index.html", std::ios::binary);
          if (!f) return rs.send(404, "text/plain", "web UI not found: " + so.webui + "/index.html");
          std::stringstream ss;
          ss << f.rdbuf();
          return rs.send(200, "text/html; charset=utf-8", ss.str());
        }
        Json r = Json::object(), mod = Json::object();
        r.set("model", "");
        r.set("n_ctx", 0);
        r.set("engine", "inferzero");
        mod.set("vision", false);
        mod.set("audio", false);
        r.set("modalities", mod);
        r.set("audio", false);
        if (!so.whisper_props.is_null()) r.set("whisper", so.whisper_props);
        return rs.send(200, "application/json", r.dump());
      }
    }
    if (swap && (rq.path == "/v1/models" || rq.path == "/models")) {  // every model a request can switch to
      std::shared_lock<std::shared_mutex> lk(swap_mu);
      // (no model loaded, -m none or a failed switch: none is "loaded", the one started last comes first, as the
      //  model a client taking the first entry then loads)
      std::string last;
      if (!cur) {  // (written by zero-server: a UTF-8 BOM, CRLF; the catalog's Hugging Face paths mix / and \)
        std::ifstream f(app_dir() + "/last-model.txt");
        std::getline(f, last);
        if (last.rfind("\xEF\xBB\xBF", 0) == 0) last.erase(0, 3);
        while (!last.empty() && (last.back() == '\r' || last.back() == ' ')) last.pop_back();
        if (!last.empty()) last = std::filesystem::path(last).lexically_normal().string();
      }
      Json d = Json::array();
      std::map<std::string, uint64_t> sizes;  // (llama_models "size")
      bool listed = !cur;
      // (unloaded in Settings: that model's file first, not last-model.txt's)
      if (!cur && !unloaded_path.empty()) last = std::filesystem::path(unloaded_path).lexically_normal().string();
      for (const ModelInfo& m : catalog) {
        if (!m.supported) continue;
        const bool loaded = !listed && m.alias == cur->alias;
        const bool front = loaded || (!last.empty() && std::filesystem::path(m.path).lexically_normal().string() == last);
        if (front) last.clear();  // (its first variant only)
        listed = listed || loaded;
        Json j = Json::object();
        j.set("id", m.alias);
        j.set("object", "model");
        j.set("owned_by", "inferzero");
        j.set("name", m.name);
        j.set("loaded", loaded);
        if (loaded) j.set("context_length", cur->eng->ctx);
        set_window(j, loaded ? cur->eng->ctx : 0, m.ctx_train, loaded ? cur->eng->hp.n_vocab : 0, loaded ? cur->eng->hp.n_embd : 0);
        // (the loaded model: the encoders that did load)
        set_modalities(j, loaded ? cur->srv->has_vision() : m.vision, loaded ? cur->srv->has_audio() : m.audio);
        sizes[m.alias] = m.bytes;
        // the loaded model first (clients that take the first entry get the running one)
        if (front) {
          Json d2 = Json::array();
          d2.push(j);
          for (const Json& e : d.items()) d2.push(e);
          d = d2;
        } else {
          d.push(j);
        }
      }
      if (!listed) {  // a model file outside the scanned folders: first, under the alias it was started with
        Json j = Json::object(), d2 = Json::array();
        j.set("id", cur->alias);
        j.set("object", "model");
        j.set("owned_by", "inferzero");
        j.set("name", cur->alias);
        j.set("loaded", true);
        j.set("context_length", cur->eng->ctx);
        set_window(j, cur->eng->ctx, 0, cur->eng->hp.n_vocab, cur->eng->hp.n_embd);
        set_modalities(j, cur->srv->has_vision(), cur->srv->has_audio());
        sizes[cur->alias] = cur->srv->model_bytes_;
        d2.push(j);
        for (const Json& e : d.items()) d2.push(e);
        d = d2;
      }
      Json r = Json::object();
      r.set("object", "list");
      r.set("models", llama_models(d, sizes));
      r.set("data", d);
      return rs.send(200, "application/json", r.dump());
    }
    if (swap && rq.method == "POST" && rq.path == "/v1/chat/completions") {
      Json body;
      const std::string want = Json::try_parse(rq.body, body) ? body["model"].str() : "";
      bool other = false;
      {
        std::shared_lock<std::shared_mutex> lk(swap_mu);
        other = !want.empty() && (!cur || want != cur->alias) && !known_unknown(want);
      }
      if (other) {
        std::unique_lock<std::shared_mutex> lk(swap_mu);
        const ModelInfo* m = !cur || want != cur->alias ? find_model(want) : nullptr;
        if (!m && (!cur || want != cur->alias)) {
          std::lock_guard<std::mutex> g(unknown_mu);
          if (unknown_names.size() > 256) unknown_names.clear();
          unknown_names[want] = clk::now();
        }
        if (m && (!cur || m->alias != cur->alias)) {
          const std::string prev_path = cur ? cur->path : std::string(), prev_alias = cur ? cur->alias : std::string();
          const auto t0 = clk::now();
          zlog::eprintf("switching model: %s -> %s\n", prev_alias.c_str(), m->alias.c_str());
          if (cur && cur->srv) cur->srv->save_held();  // (switching back later loads the conversation instead of redoing it)
          cur.reset(), zmet::unloaded();  // frees the GPU memory first
          ServeOpts s2 = so;
          s2.ctx_auto = true;  // the --ctx given at start was for the first model
          s2.eo.mtp = m->mtp;  // the variant's settings: MTP block / drafter, image projector
          s2.dflash_auto = m->draft != 1;  // (-nodf: the MTP block only; -df: DFlash2 only)
          s2.eo.mtp_block = m->draft != 2;
          if (m->draft == 1) s2.eo.dflash.clear();
          s2.mmproj = m->vision ? "" : "-";
          // (a --dflash drafter is for the first model, and only with MTP; other models find their own)
          if (!m->mtp || !same_file(m->path, model)) s2.eo.dflash.clear();
          try {
            s2.warm_prompt = false;  // (the waiting request warms it)
            cur = load_model(m->path, m->alias, s2);
            cur_so = s2;
            zlog::eprintf("switched to %s in %.1f s\n", m->alias.c_str(), ms_since(t0) / 1000);
            // (zero-pi starts with this model next time; not for the transcript service's Gemma 4 E4B requests, which
            //  carry "x-arc-speech: 1" (tools/server_speech.h attach_e4b): the user did not choose that model)
            if (rq.header("x-arc-speech").empty()) std::ofstream(app_dir() + "/last-model.txt") << m->path << "\n";
          } catch (const std::exception& e) {
            zlog::eprintf("could not load %s: %s; %s\n", m->path.c_str(), e.what(),
                    prev_path.empty() ? "no model was loaded before it" : ("loading " + prev_alias + " again").c_str());
            ServeOpts s3 = so;
            s3.ctx_auto = true;
            for (const ModelInfo& pm : catalog)  // (the previous variant's settings)
              if (pm.alias == prev_alias) {
                s3.eo.mtp = pm.mtp;
                s3.dflash_auto = pm.draft != 1;
                s3.eo.mtp_block = pm.draft != 2;
                if (pm.draft == 1) s3.eo.dflash.clear();
                s3.mmproj = pm.vision ? "" : "-";
              }
            if (!s3.eo.mtp || !same_file(prev_path, model)) s3.eo.dflash.clear();
            std::string also;
            try {  // (that failing too leaves no model: requests get 503 until one names a model that loads)
              if (!prev_path.empty()) cur = load_model(prev_path, prev_alias, s3), cur_so = s3;
            } catch (const std::exception& e2) {
              cur.reset(), zmet::unloaded();
              also = std::string("; loading ") + prev_alias + " again failed too: " + e2.what();
              zlog::eprintf("could not load %s again: %s; no model loaded\n", prev_alias.c_str(), e2.what());
            }
            Json err = Json::object(), r = Json::object();
            err.set("message", std::string("could not load model ") + want + ": " + e.what() + also);
            err.set("type", "server_error");
            r.set("error", err);
            return rs.send(500, "application/json", r.dump());
          }
        }
      }
    }
    // (--gpus: a split model that a GPU failure broke, a GPU that hung or failed: /health says so (503), requests get
    //  the error. Not loaded again here: a GPU that hung keeps its memory and usually its fault until the driver
    //  resets it, so the way back is a restart of the server)
    std::shared_lock<std::shared_mutex> lk(swap_mu);
    if (cur && cur->tp && cur->tp->broken() && rq.path == "/health")
      return rs.send(503, "application/json",
                     "{\"status\":\"error\",\"message\":\"a GPU failed or hangs: restart the server (zero-server restart)\"}");
    if (!cur) {  // (a chat request naming no model, or an unknown one, while none is loaded)
      Json err = Json::object(), r = Json::object();
      err.set("message", "no chat model is loaded (started with -m none, or the last model switch failed): name one in the request (\"model\": an id of /v1/models)");
      err.set("type", "server_error");
      r.set("error", err);
      return rs.send(503, "application/json", r.dump());
    }
    cur->srv->handle(rq, rs);
  };
  if (!http.listen(host, port)) {
    LOG_ERR("cannot listen on %s:%d: another program (often a server started earlier) may use the port; stop it, or start this one with --port N\n", host.c_str(), port);
    return 1;
  }
  zlog::eprintf("listening on http://%s:%d  (OpenAI API at /v1, web UI at /)\n", host.c_str(), port);
  if (host != "127.0.0.1" && host != "localhost" && host != "::1")  // (review 6.2: there is no authentication)
    zlog::eprintf("warning: listening on %s: anyone who can reach this address can use the model (no API key)\n",
            host.c_str());
  // (the saved transcripts' queue: started once the server answers; stopped before the locals its GPU turns use go)
  if (wh) wh->start_library(std::filesystem::path(app_dir()) / "transcripts");
  http.run();
  if (wh) wh->stop_library();
  return 0;
}

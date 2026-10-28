// arcinferzero-server: OpenAI-compatible HTTP server (chat completions with streaming, reasoning and tool calls),
// prefix reuse across requests, and a built-in web UI.
// usage: arcinferzero-server -m model.gguf [--host 127.0.0.1] [--port 8080] [--ctx 40960] [--alias name] [--webui dir]
//                       [--chunk N] [--budget GB] [--mtp] [--think-budget N] [--attn-f32] [--kernels dir]
//                       [--session-cache-gb N] [--session-dir dir]   (saved conversations; default 0 = off)
// Built as bin\serverz.exe (zero/serverz.cpp includes this file): the arcinferzero engine, zero/engine.h (Level Zero,
// MoE experts streamed from the SSD), with its data folder %USERPROFILE%\arcinferzero\zero (context cache, saved
// prompts and conversations).
#include "../zero/engine.h"
#include "../zero/gguf_pack.h"
#define ARC_DATA "/zero/"  // (under app_dir(), src/app_dir.h)
#define ARC_DATA_HELP "%%USERPROFILE%%\\arcinferzero\\zero\\"  // (the same folder, as --help prints it)
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
#include "server_files.h"
#include "doc_writer.h"
#include "xlsx.h"
#include "server_terminal.h"
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
// LM Studio ("type": "vlm"). Only models whose image encoder arcinferzero runs are marked (Gemma 4 12B, gemma4uv).
// llama-server's own model list next to "data": a "models" array (Ollama style) whose "capabilities" hold
// "multimodal" for a model with an image encoder. Clients that treat arcinferzero as a llama.cpp server read it there.
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
      prompt = jinja_ ? jinja_->render(msgs, body["tools"], true, co.enable_thinking) : chat::render(msgs, body["tools"], co, q38_);
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
      m.set("owned_by", "arcinferzero");
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
      r.set("engine", "arcinferzero");
      r.set("chunk", eng_.max_chunk);
      r.set("weights", eng_.w8_resident ? "w8-resident" : "w8-just-in-time");
      r.set("vision", has_vision());
      Json mod = Json::object();  // (llama-server's form)
      mod.set("vision", has_vision());
      mod.set("audio", has_audio());
      r.set("modalities", mod);
      r.set("audio", has_audio());
      if (!whisper_props_.is_null()) r.set("whisper", whisper_props_);
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
  // ---- conversation slots (--slots N, arcinferzero): the engine holds N conversations (Engine::bind). A request
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
  // ---- batched generation steps (--slots, Qwen models on arcinferzero: QUEUE 20.4). A request at a generation step
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
        cached_.resize(L);
        return L;
      }
      eng_.reset();
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
    if (best >= 0) {
      if (best == kSnaps) eng_.restore_host();
      else eng_.restore(best);
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

  // Saves the held conversation to the session cache and waits until it is on disk (arc-server stop / restart and a
  // model switch call it first, so that going back to the conversation afterwards loads it). False when there is
  // nothing worth saving (cache off, already saved, under 1,024 tokens).
#include "server_cache.h"  // (saved state: prefix cache, sessions)

#include "server_chat.h"  // (the chat endpoint)
};

#include "server_catalog.h"  // (model catalog, --ctx auto)

#include "server_load.h"  // (loading a model)
#include "server_whisper.h"  // (the speech service: Whisper, Qwen3-ASR, the transcript library)
#include "server_speech.h"  // (starting it, Gemma 4 E4B for transcripts, /v1/audio/*)

// The web UI's file tools for the model (it calls them as functions, the page runs them here):
//   create_file {filename, format: docx | pdf | doc | rtf | xlsx,
//                    made from Markdown content (tools/doc_writer.h; xlsx: each table a sheet, tools/xlsx.h;
//                    number_format auto | id | us), or any text extension (md, html, py...: saved as written), title,
//                    subtitle, author, date, page, orientation, margins_mm, font, font_size, line_spacing,
//                    heading_color, accent_color, text_color, header, footer, page_numbers, chat_id}: writes the file
//                    into the chat's folder (and its source next to it), returns it as a file
//   list_files {kind, name_contains, within_hours, chat_id, limit}: the saved files, newest first, with their metadata
//   read_file {file_id (or its name), section, pages, offset, max_chars, chat_id}: a document's text (DOCX, PDF, XLSX
//                    and text files, tools/doc_reader.h) in parts (a fifth of the loaded model's context at most, the rest by
//                    offset: files::read_budget); a long one without section / pages / offset: its outline;
//                    a document the model wrote: its source (to make a corrected version); images, audio: metadata.
//                    A PDF's scanned pages (images without a text layer) read by pages: the loaded model transcribes
//                    them from their images first, up to 4 a call, when it has an image encoder (FileVision)
//   search_file {query, file_id, chat_id, top_k}: the passages of a document (or of every document of the chat)
//                    that fit the query best (BM25 over chunks of ~2,400 characters), with their section and pages
//   read_file / search_file also take "chat:<id>" (another chat: a section per exchange) and "transcript:<id>"
//                    (a recording or meeting of the Whisper library: its summary, then 5-minute sections) as file_id
//   search_chats {query, limit, chat_id}: the other chats; no query: newest first (limit 1-100, 20); a query: their
//                    exchanges that fit it best (BM25, limit 1-10, 5)
//   run_command {command, chat_id, timeout_s}: one PowerShell command in the chat's folder (tools/server_terminal.h),
//                    sent by the page only after the user allowed it; refused (403) to any computer but this one.
//                    session: true: in the chat's terminal (folder and environment kept for the next command; the
//                    reply says cwd and venv); reset: true: that terminal closed (with no command: only that)
// ctx: the loaded model's context in tokens per conversation (0: not known): a read or a search gives at most what
// files::read_budget allows, the rest by offset
// Scanned pages are transcribed here, by the server, not attached to the model's next turn as images: the text is
// kept per file and page (tools/server_files.h), so search_file finds it and a later chat reads it without the
// vision model; in the conversation a page costs its text (~300-800 tokens) instead of an image's ~1,000 tokens on
// every later turn; and the web UI needs no image tool results, which the chat templates do not all take. The cost:
// the transcription is a conversation of its own, so with one slot the chat's prompt is processed again at its next
// turn (--slots 2 or the session cache keep it).
struct FileVision {
  std::function<std::string()> model;  // the loaded model's name when it has an image encoder, else ""
  std::function<std::string(const std::string& bytes, const std::string& mime)> see;  // an image -> its text
};
static constexpr const char* kTranscribe =
    "Transcribe all the text on this scanned page exactly as written, in reading order and in its own language. Use "
    "Markdown: # for headings, - for list items, | cells | for table rows. Write only the page's text, without comments; "
    "a page without text: (no text)";
static void file_tools(const HttpRequest& rq, HttpResponse& rs, bool terminal, int ctx, const FileVision& vis) {
  auto& st = files::store();
  auto reply = [&](int code, const Json& j) { rs.send(code, "application/json", j.dump()); };
  auto fail = [&](int code, const std::string& m) {
    Json e = Json::object(), r = Json::object();
    e.set("message", m);
    r.set("error", e);
    reply(code, r);
  };
  Json a;
  if (!Json::try_parse(rq.body, a) || !a.is_object()) return fail(400, "expected a JSON object");
  auto when = [](int64_t t) {
    char b[40] = "";
    const std::time_t ct = (std::time_t)t;
    std::tm tm{};
    if (localtime_s(&tm, &ct) == 0) strftime(b, sizeof b, "%Y-%m-%d %H:%M", &tm);
    return std::string(b);
  };
  try {
    if (rq.path == "/v1/tools/create_file") {
      const std::string content = a["content"].str();
      if (content.empty()) return fail(400, "content required");
      // docx, pdf, doc, rtf, xlsx are made from Markdown; md and every other text format (html, svg, css, js, py,
      // csv, json, txt...) are saved as written. No format: the file name's extension, else docx
      std::string fmt = files::lower(a["format"].str(files::ext_of(a["filename"].str()).empty() ? "docx" : files::ext_of(a["filename"].str())));
      if (!fmt.empty() && fmt[0] == '.') fmt.erase(0, 1);
      bool fmt_ok = !fmt.empty() && fmt.size() <= 10;
      for (unsigned char c : fmt) fmt_ok = fmt_ok && c < 128 && isalnum(c);
      if (!fmt_ok) return fail(400, "format: an extension such as docx, pdf, xlsx, md, html, svg, csv, json, py, txt");
      // (binary formats cannot be written from text: images, audio, video, archives, programs, other Office files)
      static const std::set<std::string> binary = {"pptx", "ppt", "xls", "xlsm", "odt", "ods", "odp", "zip", "7z", "rar",
          "gz", "tar", "exe", "dll", "msi", "bin", "gguf", "ico", "tif", "tiff", "heic", "woff", "woff2", "ttf", "otf"};
      if (files::kind_of("x." + fmt) != "document" || binary.count(fmt))
        return fail(400, "format " + fmt + " is binary and cannot be written from text; for an image write svg, for a "
                         "presentation docx or pdf, for anything else a script with run_command if the terminal is offered");
      doc::Options o;
      o.title = a["title"].str();
      o.subtitle = a["subtitle"].str();
      o.author = a["author"].str();
      o.date = a["date"].str();
      o.page = a["page"].str("A4");
      o.landscape = files::lower(a["orientation"].str()) == "landscape";
      if (a["font"].is_string() && !a["font"].str().empty()) o.font = a["font"].str();
      if (a["font_size"].is_number()) o.font_size = std::clamp(a["font_size"].as_number(), 6.0, 36.0);
      if (a["line_spacing"].is_number()) o.line_spacing = std::clamp(a["line_spacing"].as_number(), 0.8, 3.0);
      if (a["margins_mm"].is_number()) for (double& m : o.margin_mm) m = std::clamp(a["margins_mm"].as_number(), 5.0, 60.0);
      else if (a["margins_mm"].is_array())
        for (size_t k = 0; k < 4 && k < a["margins_mm"].items().size(); ++k) o.margin_mm[k] = std::clamp(a["margins_mm"].items()[k].as_number(20), 5.0, 60.0);
      for (auto [key, dst] : {std::pair<const char*, std::string*>{"heading_color", &o.heading_color}, {"accent_color", &o.accent_color}, {"text_color", &o.text_color}})
        if (const std::string c = doc::hex_color(a[key].str()); !c.empty()) *dst = c;
      o.header = a["header"].str();
      o.footer = a["footer"].str();
      if (a["page_numbers"].is_bool()) o.page_numbers = a["page_numbers"].as_bool();
      // (an empty filename too: the title, else "document"; before, "" made "file.docx")
      const std::string fname = doc::trim(a["filename"].str());
      std::string name = files::safe_name(!fname.empty() ? fname : o.title.empty() ? "document" : o.title);
      if (files::ext_of(name) != fmt) name += "." + fmt;
      std::string data;
      if (fmt == "docx") data = doc::docx(content, o);
      else if (fmt == "pdf") data = doc::pdf(content, o);
      else if (fmt == "doc" || fmt == "rtf") data = doc::rtf(content, o);
      else if (fmt == "xlsx") data = xlsx::write(content, o, a["number_format"].str("auto"));
      // (md: the title as its heading, unless the content starts with a heading of its own: else it came twice)
      else if (fmt == "md") data = (o.title.empty() || doc::trim(content).rfind("# ", 0) == 0 ? "" : "# " + o.title + "\n\n") + content;
      else data = content;
      const std::string chat = files::valid_id(a["chat_id"].str()) ? a["chat_id"].str() : "";
      const int64_t before = (int64_t)std::time(nullptr);
      const files::Info i = st.put(name, data, chat);
      // (the same bytes saved before: put gives that file back, under its own name; its source stays the one that
      //  made it, and the card says so, instead of this request's source and name over it)
      const bool existed = i.created < before;
      if (!existed) st.put_source(i.id, chat, rq.body);
      Json r = files::info_json(i);
      r.set("file_id", i.id);
      r.set("note", existed ? "these exact bytes were already saved as " + i.name + ": that file is shown to the user (the "
                              "same file card); to make a different file, change its content"
                            : "created; the user sees it as a file card in your answer (download / open). To change it, "
                              "read its source with read_file and call create_file again: that makes a new version.");
      return reply(200, r);
    }
    if (rq.path == "/v1/tools/list_files") {
      const std::string kind = files::lower(a["kind"].str()), q = files::lower(a["name_contains"].str()), chat = a["chat_id"].str();
      const double hours = a["within_hours"].as_number(0);
      const int limit = (int)std::clamp(a["limit"].as_number(50), 1.0, 200.0);
      const auto uses = st.uses();
      const int64_t now = (int64_t)std::time(nullptr);
      Json d = Json::array();
      for (const files::Info& i : st.list()) {
        if (!kind.empty() && i.kind != kind) continue;
        if (!q.empty() && files::lower(i.name).find(q) == std::string::npos) continue;
        if (hours > 0 && now - i.created > hours * 3600) continue;
        auto u = uses.find(i.id);
        const auto held = st.holders(i.id);
        if (!chat.empty() && std::find(held.begin(), held.end(), chat) == held.end() &&
            (u == uses.end() || std::find(u->second.begin(), u->second.end(), chat) == u->second.end()))
          continue;
        Json j = Json::object();
        j.set("file_id", i.id);
        j.set("filename", i.name);
        j.set("kind", i.kind);
        j.set("bytes", (double)i.bytes);
        j.set("uploaded", when(i.created));
        if (i.kind == "image" || i.kind == "audio" || i.kind == "video") {  // (its size or length: an image's header)
          std::string head(i.kind == "image" ? 65536 : 0, '\0');
          std::ifstream hf(std::filesystem::path(i.path), std::ios::binary);
          hf.read(head.data(), (std::streamsize)head.size());
          head.resize((size_t)hf.gcount());
          const std::string x = files::extent_of(i, head);
          if (!x.empty()) j.set(i.kind == "image" ? "size" : "length", x.substr(2));
        }
        std::string src;
        j.set("written_by_model", st.source(i.id, src));
        j.set("chats", (double)std::max(held.size(), u == uses.end() ? size_t(0) : u->second.size()));
        d.push(j);
        if ((int)d.size() >= limit) break;
      }
      Json r = Json::object();
      r.set("files", d);
      r.set("count", (double)d.size());
      return reply(200, r);
    }
    // the call's file: by its id, or by its name (models often pass the name they see: the newest file of that
    // name, this chat's first)
    auto find_file = [&](files::Info& i) {
      std::string id = a["file_id"].str();
      if (id.rfind("file-", 0) == 0) id = id.substr(5);
      if (id.rfind("/files/", 0) == 0) id = id.substr(7);
      if (files::valid_id(id, 16) && st.get(id, i)) return true;
      const std::string want = files::lower(a["file_id"].str()), chat = a["chat_id"].str();
      bool found = false;
      for (const files::Info& x : st.list()) {
        if (files::lower(x.name) != want) continue;
        const auto held = st.holders(x.id);
        const bool here = std::find(held.begin(), held.end(), chat) != held.end();
        if (!found || here) { i = x; found = true; }
        if (here) break;
      }
      return found;
    };
    if (rq.path == "/v1/tools/search_file") {
      const std::string query = a["query"].str();
      if (query.empty()) return fail(400, "query required");
      const size_t top = (size_t)std::clamp(a["top_k"].as_number(5), 1.0, 10.0);
      std::vector<files::Info> targets;
      std::vector<std::shared_ptr<const rdoc::Index>> ixs;
      if (files::is_virtual(a["file_id"].str())) {  // (another chat, or a transcript)
        files::VirtualDoc v;
        if (!files::virtual_doc(a["file_id"].str(), v)) return fail(404, "no " + a["file_id"].str() + " (search_chats without a query lists the chats)");
        files::Info i;
        i.id = a["file_id"].str();
        i.name = v.title;
        targets.push_back(i);
        ixs.push_back(v.ix);
      } else if (!a["file_id"].str().empty()) {
        files::Info i;
        if (!find_file(i)) return fail(404, "no file " + a["file_id"].str() + " (list_files shows the saved files)");
        targets.push_back(i);
      } else if (files::valid_id(a["chat_id"].str())) {  // (every document of this chat)
        const std::string chat = a["chat_id"].str();
        const auto uses = st.uses();
        for (const files::Info& i : st.list()) {
          if (i.kind != "document") continue;
          const auto held = st.holders(i.id);
          auto u = uses.find(i.id);
          if (std::find(held.begin(), held.end(), chat) != held.end() ||
              (u != uses.end() && std::find(u->second.begin(), u->second.end(), chat) != u->second.end()))
            targets.push_back(i);
        }
      } else return fail(400, "file_id (or chat_id: every document of the chat) required");
      struct Found { double score; size_t file, chunk; };
      std::vector<Found> all;
      for (size_t f = 0; f < targets.size(); ++f) {
        if (ixs.size() <= f) ixs.push_back(files::doc_index(targets[f]));
        if (!ixs[f]) continue;
        for (const auto& h : ixs[f]->search(query, top)) all.push_back({h.score, f, h.chunk});
      }
      std::sort(all.begin(), all.end(), [](const Found& x, const Found& y) { return x.score > y.score; });
      if (all.size() > top) all.resize(top);
      Json res = Json::array();
      size_t given = 0, left_out = 0;  // (the passages' text within the read budget; the best one always)
      for (const Found& h : all) {
        const rdoc::Index& ix = *ixs[h.file];
        const rdoc::Chunk& c = ix.chunks[h.chunk];
        if (res.size() && given + c.text.size() > files::read_budget(ctx)) { ++left_out; continue; }
        given += c.text.size();
        Json j = Json::object();
        j.set("file_id", targets[h.file].id);
        j.set("filename", targets[h.file].name);
        j.set("section", (double)(c.section + 1));
        j.set("section_title", ix.outline[(size_t)c.section].title);
        if (ix.doc.pages > 1) j.set("pages", c.page_from == c.page_to ? std::to_string(c.page_from) : std::to_string(c.page_from) + "-" + std::to_string(c.page_to));
        char sc[16];
        snprintf(sc, sizeof sc, "%.1f", h.score);
        j.set("score", std::string(sc));
        j.set("text", c.text);
        res.push(j);
      }
      Json r = Json::object();
      r.set("results", res);
      r.set("files_searched", (double)targets.size());
      std::string unread;  // (scanned pages no vision model has read yet: their text is not searched)
      for (size_t f = 0; f < targets.size(); ++f)
        if (ixs[f] && !ixs[f]->doc.unread.empty())
          unread += (unread.empty() ? "" : "; ") + targets[f].name + ": pages " + rdoc::pdfimg::ranges(ixs[f]->doc.unread);
      if (!unread.empty())
        r.set("not_searched", "scanned pages not transcribed yet (" + unread + "): read_file with pages has a vision model read them");
      // (nothing to search is not "no passage has these words": that sent the model off retrying with other words)
      const size_t searchable = (size_t)std::count_if(ixs.begin(), ixs.end(), [](const auto& x) { return x != nullptr; });
      if (targets.empty())
        r.set("note", "this chat has no documents to search: list_files shows the saved files (then search_file with file_id)");
      else if (searchable == 0)
        r.set("note", targets.size() == 1 ? targets[0].name + " has no text to search (" + targets[0].kind + "): read_file gives what it holds"
                                          : "none of these files has text to search");
      else if (res.size() == 0)
        r.set("note", "no passage has these words: try other words (the document's own terms, synonyms), or read the outline (read_file)");
      else if (left_out)
        r.set("note", std::to_string(left_out) + " more passages left out to keep within the model's context: search with more "
                      "precise words, or read a result's section (read_file)");
      return reply(200, r);
    }
    // the other chats (the web UI's, one reads another): listed newest first (a query: in their text), or searched
    // by their exchanges (BM25); each is read with read_file {"file_id": "chat:<id>"} (an outline per exchange)
    // (one tool for both since 2026-10-28, list_chats before: no query lists them, a query searches their exchanges)
    if (rq.path == "/v1/tools/search_chats") {
      const std::string q = a["query"].str(), self = a["chat_id"].str();
      struct C { std::string id, title, model, snippet; int64_t at; };
      std::vector<C> cs;
      for (const auto& [cid, text] : st.chats_all()) {
        if (cid == self) continue;
        Json c;
        if (!Json::try_parse(text, c)) continue;
        cs.push_back({cid, c["title"].str("chat"), c["model"].str(), c["snippet"].str(), (int64_t)(c["at"].as_number(0) / 1000)});
      }
      std::sort(cs.begin(), cs.end(), [](const C& x, const C& y) { return x.at > y.at; });
      if (doc::trim(q).empty()) {
        const size_t limit = (size_t)std::clamp(a["limit"].as_number(20), 1.0, 100.0);
        Json d = Json::array();
        for (const C& c : cs) {
          if (d.size() >= limit) break;
          files::VirtualDoc v;
          Json j = Json::object();
          j.set("file_id", "chat:" + c.id);
          j.set("title", c.title);
          j.set("updated", when(c.at));
          if (files::virtual_doc("chat:" + c.id, v)) j.set("exchanges", (double)v.parts);
          if (!c.model.empty()) j.set("model", c.model);
          if (!c.snippet.empty()) j.set("snippet", c.snippet);
          d.push(j);
        }
        Json r = Json::object();
        r.set("chats", d);
        r.set("count", (double)d.size());
        r.set("note", "read one with read_file {\"file_id\": \"chat:<id>\"} (long ones: their outline, one section per exchange), "
                      "or find passages in all of them with search_chats and a query");
        return reply(200, r);
      }
      const size_t top = (size_t)std::clamp(a["limit"].as_number(5), 1.0, 10.0);
      struct Hit { double score; size_t chat, chunk; };
      std::vector<Hit> hits;
      std::vector<std::shared_ptr<const rdoc::Index>> ixs(cs.size());
      for (size_t k = 0; k < cs.size(); ++k) {
        files::VirtualDoc v;
        if (!files::virtual_doc("chat:" + cs[k].id, v)) continue;
        ixs[k] = v.ix;
        for (const auto& h : v.ix->search(q, top)) hits.push_back({h.score, k, h.chunk});
      }
      std::sort(hits.begin(), hits.end(), [](const Hit& x, const Hit& y) { return x.score > y.score; });
      if (hits.size() > top) hits.resize(top);
      Json res = Json::array();
      size_t given = 0;
      for (const Hit& h : hits) {
        const rdoc::Chunk& c = ixs[h.chat]->chunks[h.chunk];
        if (res.size() && given + c.text.size() > files::read_budget(ctx)) continue;
        given += c.text.size();
        Json j = Json::object();
        j.set("file_id", "chat:" + cs[h.chat].id);
        j.set("title", cs[h.chat].title);
        j.set("section", (double)(c.section + 1));
        j.set("exchange", ixs[h.chat]->outline[(size_t)c.section].title);
        char sc[16];
        snprintf(sc, sizeof sc, "%.1f", h.score);
        j.set("score", std::string(sc));
        j.set("text", c.text);
        res.push(j);
      }
      Json r = Json::object();
      r.set("results", res);
      r.set("chats_searched", (double)cs.size());
      if (res.size() == 0) r.set("note", "no exchange has these words: try other words, or search_chats without a query (the chats, newest first)");
      return reply(200, r);
    }
    if (rq.path == "/v1/tools/read_file") {
      files::Info i;
      Json r = Json::object();
      std::shared_ptr<const rdoc::Index> ix;
      if (files::is_virtual(a["file_id"].str())) {  // (another chat, or a transcript: read as a document)
        files::VirtualDoc v;
        if (!files::virtual_doc(a["file_id"].str(), v)) return fail(404, "no " + a["file_id"].str() + " (search_chats without a query lists the chats)");
        i.id = a["file_id"].str();
        i.name = v.title;
        i.kind = "document";
        r.set("file_id", i.id);
        r.set("title", v.title);
        r.set("kind", v.kind);
        r.set(v.kind == "chat" ? "exchanges" : "minutes", (double)v.parts);
        r.set("updated", when(v.at));
        ix = v.ix;
      } else {
        if (!find_file(i)) return fail(404, "no file " + a["file_id"].str() + " (list_files shows the saved files)");
        r.set("file_id", i.id);
        r.set("filename", i.name);
        r.set("kind", i.kind);
        r.set("bytes", (double)i.bytes);
        r.set("uploaded", when(i.created));
        std::string src;
        if (st.source(i.id, src)) {  // (the model's own document: its source)
          Json sj;
          std::string note = "a file written with create_file: 'source' holds its arguments; to correct it, change them and call create_file again";
          if (Json::try_parse(src, sj)) {
            // (its content within the model's read budget too, in parts by offset: a 60 KB page read whole overflowed
            //  a small context)
            const std::string body = sj["content"].str();
            const size_t mx = files::read_budget(ctx), off = std::min(body.size(), (size_t)std::max(0.0, a["offset"].as_number(0)));
            if (off > 0 || body.size() > mx) {
              size_t b = off, end = std::min(body.size(), off + mx);
              while (b < body.size() && ((unsigned char)body[b] & 0xC0) == 0x80) ++b;
              while (end < body.size() && end > b && ((unsigned char)body[end] & 0xC0) == 0x80) --end;
              sj.set("content", body.substr(b, end - b));
              r.set("offset", (double)b);
              r.set("total_chars", (double)body.size());
              if (end < body.size()) {
                r.set("next_offset", (double)end);
                note += "; content is a part (as much as the model's context takes at once): read_file with offset: " +
                        std::to_string(end) + " for the rest; a corrected version needs the whole content again";
              }
            }
            r.set("source", sj);
          }
          r.set("note", note);
          return reply(200, r);
        }
        if (i.kind != "document") {
          r.set("note", i.kind == "image" || i.kind == "audio"
                            ? "an " + i.kind + ": its content is not text; the user can attach it to a message (@) for a model that " +
                                  (i.kind == "image" ? "sees images" : "hears audio")
                            : "a " + i.kind + " file: its content is not text");
          return reply(200, r);
        }
        ix = files::doc_index(i);
        if (!ix) return fail(500, "cannot read the file");
      }
      // the pages this call asks for ("12" or "12-15"; none: from 1, to 0)
      const bool by_pages = !a["pages"].str().empty() || a["pages"].is_number();
      const std::string pg = a["pages"].is_number() ? std::to_string((int)a["pages"].as_number()) : a["pages"].str();
      const int from = std::max(1, atoi(pg.c_str())), to = !by_pages ? 0 : pg.find('-') != std::string::npos ? std::max(from, atoi(pg.c_str() + pg.find('-') + 1)) : from;
      // scanned pages among them, not transcribed yet: the loaded vision model reads them now (kScanPagesPerCall)
      const std::string seer = vis.model ? vis.model() : "";
      std::vector<int> want;
      for (int p : ix->doc.unread)
        if (p >= from && p <= to) want.push_back(p);
      if (!want.empty() && !seer.empty() && vis.see) {
        const std::vector<int> now(want.begin(), want.begin() + std::min<size_t>(want.size(), files::kScanPagesPerCall));
        r.set("transcribed", files::transcribe_pages(i, now, seer, vis.see));
        r.set("transcribed_by", seer);
        ix = files::doc_index(i);
        if (!ix) return fail(500, "cannot read the file");
      }
      std::string scan_note;  // (scanned pages left: how to read them)
      if (!ix->doc.unread.empty()) {
        const std::string left = rdoc::pdfimg::ranges(ix->doc.unread);
        scan_note = !seer.empty() ? "scanned pages not transcribed yet: " + left + ". read_file with pages (e.g. \"" +
                                        std::to_string(ix->doc.unread[0]) + "\"; up to " + std::to_string(files::kScanPagesPerCall) +
                                        " pages a call, 10-60 s a page) has the vision model transcribe them; afterwards search_file finds their text too"
                                  : "scanned pages " + left + " are images: the loaded model has no image encoder, so they cannot be read. "
                                    "Tell the user to load a vision model (Qwen 3.5 4B / 9B, Gemma 4 E4B / 12B / 26B with its mmproj) and ask again";
      }
      if (ix->doc.paras.empty()) {
        r.set("note", (ix->doc.note.empty() ? "no text in the file" : ix->doc.note) + (scan_note.empty() ? "" : "; " + scan_note));
        if (ix->doc.pages > 0) r.set("pages", (double)ix->doc.pages);
        return reply(200, r);
      }
      if (!ix->doc.note.empty() || !scan_note.empty())
        r.set("warning", ix->doc.note + (ix->doc.note.empty() || scan_note.empty() ? "" : "; ") + scan_note);
      if (ix->doc.pages > 0) r.set("pages", (double)ix->doc.pages);
      std::string data;  // (what this call reads: a section, pages, or the whole text)
      const size_t n = ix->doc.paras.size();
      if (a["section"].is_number()) {
        const size_t k = (size_t)std::max(1.0, a["section"].as_number()) - 1;
        if (k >= ix->outline.size()) return fail(400, "no section " + std::to_string(k + 1) + " (the outline has " + std::to_string(ix->outline.size()) + ")");
        const rdoc::Section& s = ix->outline[k];
        data = ix->text(s.first, s.span);  // (with its subsections)
        r.set("section", (double)(k + 1));
        r.set("title", s.title);
        if (ix->doc.pages > 1) r.set("pages_read", files::pages_of(s));
        Json subs = Json::array();  // (its subsections' numbers, to read one of them alone)
        for (size_t j = k + 1; j < ix->outline.size() && ix->outline[j].first < s.span && subs.size() < 40; ++j) {
          Json x = Json::object();
          x.set("section", (double)(j + 1));
          x.set("title", ix->outline[j].title);
          subs.push(x);
        }
        if (subs.size()) r.set("subsections", subs);
        size_t next = k + 1;
        while (next < ix->outline.size() && ix->outline[next].first < s.span) ++next;
        if (next < ix->outline.size()) r.set("next_section", (double)(next + 1));
      } else if (by_pages) {
        size_t first = n, last = 0;
        for (size_t k = 0; k < n; ++k)
          if (ix->doc.paras[k].page >= from && ix->doc.paras[k].page <= to) { first = std::min(first, k); last = k + 1; }
        if (first >= last && !scan_note.empty() &&
            std::any_of(ix->doc.unread.begin(), ix->doc.unread.end(), [&](int p) { return p >= from && p <= to; })) {
          r.set("pages_read", pg);
          r.set("note", "no text read on page" + std::string(from == to ? " " : "s ") + pg + ": " + scan_note);
          return reply(200, r);
        }
        if (first >= last && r.has("transcribed")) {  // (the vision model saw no text there)
          r.set("pages_read", pg);
          r.set("note", "the vision model found no text on page" + std::string(from == to ? " " : "s ") + pg);
          return reply(200, r);
        }
        if (first >= last) return fail(400, "no text on page" + std::string(from == to ? " " : "s ") + pg + (ix->doc.pages > 0 ? " (the document has " + std::to_string(ix->doc.pages) + ")" : ""));
        data = ix->text(first, last);
        r.set("pages_read", pg);
      } else {
        data = ix->text(0, n);
        if (data.size() > files::kInlineChars && !a["offset"].is_number()) {  // (a long document: its outline first)
          r.set("total_tokens", (double)files::tokens_of(data.size()));
          r.set("outline", files::outline_json(*ix));
          size_t cut = std::min(data.size(), (size_t)2000);
          while (cut < data.size() && cut > 0 && ((unsigned char)data[cut] & 0xC0) == 0x80) --cut;
          r.set("beginning", data.substr(0, cut));
          r.set("note", "a long document: read the sections the question needs (read_file with section: N, or pages: \"A-B\"), "
                        "or find passages with search_file; offset: 0 reads it from the start in parts");
          return reply(200, r);
        }
      }
      // (a part at a time: max_chars, and never more than the model's context takes in one read)
      const size_t asked = (size_t)std::clamp(a["max_chars"].as_number(12000), 500.0, 60000.0);
      const size_t off = (size_t)std::max(0.0, a["offset"].as_number(0)), mx = std::min(asked, files::read_budget(ctx));
      size_t end = std::min(data.size(), off + mx);
      if (end < data.size()) {  // (cut at a paragraph's end, else a line's or a space, when one is in the last third)
        size_t at = data.rfind("\n\n", end);
        if (at == std::string::npos || at <= off + mx * 2 / 3) at = data.rfind(' ', end);
        if (at != std::string::npos && at > off + mx * 2 / 3) end = at;
      }
      while (end < data.size() && end > off && ((unsigned char)data[end] & 0xC0) == 0x80) --end;  // (not inside a UTF-8 character)
      r.set("content", off < data.size() ? data.substr(off, end - off) : std::string());
      r.set("offset", (double)off);
      r.set("total_chars", (double)data.size());
      if (end < data.size()) {
        r.set("next_offset", (double)end);
        r.set("note", "a part of " + std::to_string(files::tokens_of(data.size())) + " tokens" +
                          (mx < asked ? " (as much as the model's context takes at once)" : "") +
                          ": read on with offset: " + std::to_string(end) + " only when the question needs the rest");
      }
      return reply(200, r);
    }
    if (rq.path == "/v1/tools/run_command") {
      if (!terminal) return fail(404, "the terminal tool is off (--no-terminal)");
      if (!rq.local_peer) return fail(403, "run_command is only for this computer (the web UI on 127.0.0.1)");
      const std::string cmd = a["command"].str();
      const bool session = a["session"].is_bool() && a["session"].as_bool(), reset = a["reset"].is_bool() && a["reset"].as_bool();
      if (cmd.empty() && !reset && !session) return fail(400, "command required");
      // (a command line holds 32767 characters: the script goes in as base64 of UTF-16, 2.7 characters a byte)
      if (cmd.size() > 8000) return fail(400, "command too long (8000 characters at most): split it, e.g. write a script with Set-Content in parts, then run it");
      const std::string chat = a["chat_id"].str();
      const std::string cwd = files::valid_id(chat) ? st.chat_dir_make(chat) : app_dir();
      // session: the chat's terminal (its folder and environment carried from one command to the next, kept in the
      // chat's folder); reset: that terminal closed, the next one starts fresh in the chat's folder
      const std::filesystem::path state = std::filesystem::path(cwd) / ".terminal.json";
      std::error_code ec;
      if (reset) std::filesystem::remove(state, ec);
      auto where = [&](Json& r) {  // (the session's folder and venv now)
        std::string s, dir = cwd, venv;
        Json j;
        if (session && files::Store::read(state.string(), s) && Json::try_parse(s.rfind("\xEF\xBB\xBF", 0) == 0 ? s.substr(3) : s, j)) {
          if (!j["cwd"].str().empty()) dir = j["cwd"].str();
          venv = j["env"]["VIRTUAL_ENV"].str();
        }
        r.set("cwd", dir);
        if (!venv.empty()) r.set("venv", venv);
      };
      if (cmd.empty()) {  // (only the reset, or where the session is)
        Json r = Json::object();
        where(r);
        return reply(200, r);
      }
      const int tmo = (int)std::clamp(a["timeout_s"].as_number(120), 1.0, 600.0);
      std::string path = term::exe_dir();
      if (std::filesystem::is_directory(std::filesystem::path(app_dir()) / "bin", ec)) path = (std::filesystem::path(app_dir()) / "bin").string() + ";" + path;
      // (the page closed the request, Stop or Esc: the command ends with it, not at its timeout)
      term::Result t = term::run(cmd, cwd, path, tmo, session ? state.string() : "", [&] { return rs.client_gone(); });
      if (t.cancelled) return;  // (nobody reads the reply)
      // the output as the model can take it: within the same share of its context as a read_file part (64 KB is
      // ~16K tokens, more than a small model's whole context): the start and the end of a longer one
      const size_t budget = files::read_budget(ctx);
      if (t.output.size() > budget) {
        size_t head = budget * 3 / 4, tail = t.output.size() - (budget - head);
        while (head > 0 && ((unsigned char)t.output[head] & 0xC0) == 0x80) --head;
        while (tail < t.output.size() && ((unsigned char)t.output[tail] & 0xC0) == 0x80) ++tail;
        t.output = t.output.substr(0, head) + "\n[... " + std::to_string(tail - head) + " characters left out: a smaller output "
                   "(Select-Object -First N, Select-String, a file) shows the part you need ...]\n" + t.output.substr(tail);
        t.truncated = true;
      }
      Json r = Json::object();
      r.set("exit_code", (double)t.exit_code);
      r.set("output", t.output);
      r.set("timed_out", t.timed_out);
      r.set("truncated", t.truncated);
      r.set("seconds", std::round(t.seconds * 10) / 10);
      where(r);
      return reply(200, r);
    }
    return fail(404, "no such tool");
  } catch (const std::exception& e) {
    return fail(500, e.what());
  }
}

// /v1/files (POST multipart upload, GET list with the chats using each file), /v1/files/<id> (DELETE),
// /v1/files/delete_unused (POST), /files/<id> (GET the bytes; ?download=1 as an attachment), /v1/chats (GET list),
// /v1/chats/<id> (GET, PUT the web UI's JSON, DELETE: its folder with its files, unless ?keep_files=1)
static void files_api(const HttpRequest& rq, HttpResponse& rs) {
  auto& st = files::store();
  auto fail = [&](int code, const std::string& m) {
    Json e = Json::object(), r = Json::object();
    e.set("message", m);
    r.set("error", e);
    rs.send(code, "application/json", r.dump());
  };
  auto q = [&](const std::string& k) { return ("&" + rq.query + "&").find("&" + k + "=1&") != std::string::npos; };
  try {
    if (rq.path == "/v1/files" && rq.method == "POST") {
      const auto form = apic::parse_multipart(rq.header("content-type"), rq.body);
      const auto f = form.find("file");
      if (f == form.end() || f->second.data.empty()) return fail(400, "file required (multipart/form-data, field \"file\")");
      std::string chat;  // (the chat's folder: a form field chat_id, or ?chat_id=)
      if (auto c = form.find("chat_id"); c != form.end()) chat = c->second.data;
      else if (const size_t k = ("&" + rq.query).find("&chat_id="); k != std::string::npos) chat = rq.query.substr(k + 8, rq.query.find('&', k + 8) - k - 8);
      if (!chat.empty() && !files::valid_id(chat)) return fail(400, "bad chat_id");
      return rs.send(200, "application/json", files::info_json(st.put(f->second.filename, f->second.data, chat)).dump());
    }
    if (rq.path == "/v1/files" && rq.method == "GET") {
      const auto uses = st.uses();
      std::map<std::string, std::string> titles;  // (chat id -> title, for "used by")
      for (const auto& [cid, text] : st.chats_all()) {
        try { titles[cid] = Json::parse(text)["title"].str(); } catch (const std::exception&) { titles[cid] = cid; }
      }
      Json d = Json::array();
      double total = 0;
      for (const files::Info& i : st.list()) {
        Json j = files::info_json(i), by = Json::array();
        std::vector<std::string> cids = st.holders(i.id);
        if (auto it = uses.find(i.id); it != uses.end())
          for (const std::string& cid : it->second)
            if (std::find(cids.begin(), cids.end(), cid) == cids.end()) cids.push_back(cid);
        for (const std::string& cid : cids) {
          if (!titles.count(cid)) continue;  // (a folder whose chat was deleted: its files are unused)
            Json c = Json::object();
            c.set("id", cid);
            c.set("title", titles.count(cid) ? titles[cid] : cid);
            by.push(c);
          }
        j.set("used_by", by);
        d.push(j);
        total += (double)i.bytes;
      }
      Json r = Json::object();
      r.set("object", "list");
      r.set("data", d);
      r.set("total_bytes", total);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path == "/v1/files/delete_unused" && rq.method == "POST") {
      // (a body {"ids": [...]}: only those files, e.g. the ones of a deleted message; no body: every file)
      std::set<std::string> only;
      Json b;
      const bool limited = !rq.body.empty() && Json::try_parse(rq.body, b) && b["ids"].is_array();  // ([] : none)
      if (!rq.body.empty() && !limited) return fail(400, "expected no body (every unused file) or {\"ids\": [...]}");
      if (limited)
        for (const Json& x : b["ids"].items()) only.insert(x.str());
      // (a file out of each folder whose chat does not refer to it any more; gone when no chat holds it)
      const auto uses = st.uses();
      int n = 0;
      double freed = 0;
      for (const files::Info& i : st.list()) {
        if (limited && !only.count(i.id)) continue;
        const auto u = uses.find(i.id);
        auto used_by = [&](const std::string& c) { return u != uses.end() && std::find(u->second.begin(), u->second.end(), c) != u->second.end(); };
        for (const std::string& c : st.holders(i.id))
          if (!used_by(c) && !st.draft(c)) st.remove_from(i.id, c);  // (a chat not saved yet: its files stay)
        if (u == uses.end()) st.remove_from(i.id, "");  // (the old uploads folder's copy)
        files::Info left;
        if (!st.get(i.id, left)) { ++n; freed += (double)i.bytes; }
      }
      Json r = Json::object();
      r.set("deleted", n);
      r.set("freed_bytes", freed);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path.rfind("/v1/files/", 0) == 0 && rq.method == "DELETE") {
      const std::string id = rq.path.substr(10);
      if (!files::valid_id(id, 16) || !st.remove(id)) return fail(404, "no file " + id);
      return rs.send(200, "application/json", "{\"deleted\":true}");
    }
    if (rq.path.rfind("/files/", 0) == 0 && rq.method == "GET") {
      files::Info i;
      const std::string id = rq.path.substr(7);
      if (!files::valid_id(id, 16) || !st.get(id, i)) return fail(404, "no file " + id);
      // (content-addressed: the same id is always the same bytes)
      rs.extra = "Cache-Control: max-age=31536000, immutable\r\nX-Content-Type-Options: nosniff\r\n";
      if (q("download")) rs.extra += "Content-Disposition: attachment; filename=\"" + i.name + "\"\r\n";
      // (read from disk 1 MB at a time, with Range: before, the whole file in memory, and 404 above 2 GB)
      return rs.send_file(std::filesystem::path(i.path), files::mime_of(i.name), rq.header("range"));
    }
    if (rq.path == "/v1/chats" && rq.method == "DELETE") {  // every chat; ?files=1: every uploaded file too
      int nc = 0, nf = 0;
      double freed = 0;
      const auto before = st.list();
      for (const auto& [cid, text] : st.chats_all())
        if (st.chat_remove(cid, !q("files"))) ++nc;
      if (q("files"))
        for (const files::Info& i : st.list()) st.remove(i.id);
      for (const files::Info& i : before) {
        files::Info left;
        if (!st.get(i.id, left)) { ++nf; freed += (double)i.bytes; }
      }
      Json r = Json::object();
      r.set("chats_deleted", nc);
      r.set("files_deleted", nf);
      r.set("freed_bytes", freed);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path == "/v1/chats" && rq.method == "GET") {
      Json d = Json::array();
      for (const auto& [cid, text] : st.chats_all()) {
        Json c;
        try { c = Json::parse(text); } catch (const std::exception&) { continue; }  // (an unreadable file: left out)
        Json j = Json::object();
        j.set("id", cid);
        j.set("title", c["title"].str("New chat"));
        j.set("at", c["at"].as_number(0));
        j.set("model", c["model"].str(""));
        j.set("snippet", c["snippet"].str(""));
        d.push(j);
      }
      Json r = Json::object();
      r.set("data", d);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path.rfind("/v1/chats/", 0) == 0) {
      const std::string id = rq.path.substr(10);
      if (!files::valid_id(id)) return fail(400, "bad chat id");
      if (rq.method == "GET") {
        std::string t;
        if (!st.chat_get(id, t)) return fail(404, "no chat " + id);
        return rs.send(200, "application/json", t);
      }
      if (rq.method == "PUT" || rq.method == "POST") {
        Json c = Json::parse(rq.body);  // (checked: only JSON objects are kept)
        if (!c.is_object()) return fail(400, "a chat is a JSON object");
        st.chat_put(id, rq.body);
        return rs.send(200, "application/json", "{\"saved\":true}");
      }
      if (rq.method == "DELETE") {
        std::string t;
        if (!st.chat_get(id, t)) return fail(404, "no chat " + id);
        const auto before = st.list();
        if (!st.chat_remove(id, q("keep_files"))) return fail(404, "no chat " + id);  // (its folder, with its files)
        int n = 0;
        if (!q("keep_files")) {
          const auto uses = st.uses();
          for (const std::string& fid : files::Store::refs(t))  // (old-layout files of it that no other chat uses)
            if (!uses.count(fid) && st.holders(fid).empty()) st.remove(fid);
          for (const files::Info& i : before) {
            files::Info left;
            if (!st.get(i.id, left)) ++n;
          }
        }
        Json r = Json::object();
        r.set("deleted", true);
        r.set("files_deleted", n);
        return rs.send(200, "application/json", r.dump());
      }
    }
    return fail(405, "method not allowed");
  } catch (const std::exception& e) {
    return fail(500, e.what());
  }
}

int main(int argc, char** argv) {
  full_speed_process();
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--list-models") return list_models();
  std::string model, host = "127.0.0.1", alias;
  int port = 8080;
  bool swap = true, help = false, web_tools = true, terminal = true;
  std::string whisper_opt = "off", whisper_dev = "gpu", asr_opt = "default";  // (default: auto when Whisper is on)
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
  try {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "-m" || a == "--model") model = next();
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
    else if (a == "--no-stream") so.eo.stream_experts = false;
    else if (a == "--pack") {  // make an arcinferzero package of a MoE GGUF, then exit
      const std::string src = next();
      try {
        fprintf(stderr, "arcinferzero: making a package of %s (reads the whole model once)\n", src.c_str());
        const zpack::Result r = zpack::make_package(src);
        fprintf(stderr, "arcinferzero: package ready in %.0f s\n  %s (%.2f GB)\n  %s (%.2f GB)\n"
                        "The original GGUF is no longer needed to run this model with arcinferzero: the model list now\n"
                        "picks the package. The engine's own expert file for the GGUF moved into the package, so loading\n"
                        "the GGUF itself (by its path) would write that file again; keep the GGUF only as a backup.\n",
                r.seconds, r.gguf.c_str(), r.rest_bytes / 1e9, r.experts.c_str(), r.expert_bytes / 1e9);
        return 0;
      } catch (const std::exception& e) {
        fprintf(stderr, "arcinferzero: --pack failed: %s\n", e.what());
        return 1;
      }
    }
    else if (a == "--alias") alias = next();
    else if (a == "--webui") so.webui = next();
    else if (a == "--budget") { so.eo.w8_budget_gb = to_double(a, next()); so.budget_given = true; }
    else if (a == "--attn-f32") so.eo.attn_int8 = false;
    else if (a == "--npu") {  // (kept by every model switch: so.eo)
      const std::string v = next();
      so.eo.npu = v == "on" ? 1 : v == "off" ? 0 : v == "force" ? 2 : -2;
      if (so.eo.npu == -2) throw std::runtime_error("--npu: expected on, off or force");
    }
    else if (a == "--mtp") so.eo.mtp = true;
    else if (a == "--dflash") { so.eo.dflash = next(); so.eo.mtp = true; }
    else if (a == "--no-dflash") { so.eo.dflash.clear(); so.dflash_auto = false; }
    else if (a == "--draft") {  // (the drafter of the -nodf / -df variants)
      const std::string d = next();
      if (d == "mtp") so.eo.dflash.clear(), so.dflash_auto = false, so.eo.mtp_block = true;
      else if (d == "dflash") so.dflash_auto = true, so.eo.mtp_block = false;
      else if (d == "auto") so.dflash_auto = true, so.eo.mtp_block = true;
      else throw std::runtime_error("--draft: auto, mtp or dflash");
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
    else if (a == "--no-swap") swap = false;
    else if (a == "--no-web-tools") web_tools = false;
    else if (a == "--whisper") whisper_opt = next();
    else if (a == "--asr") asr_opt = next();
    else if (a == "--whisper-device") {
      whisper_dev = next();
      if (whisper_dev != "gpu" && whisper_dev != "npu") throw std::runtime_error("--whisper-device: gpu or npu");
    }
    else if (a == "--whisper-max-minutes") whisper_max_min = std::clamp(to_double(a, next()), 1.0, 600.0);
    else if (a == "--no-terminal") terminal = false;
    else if (a == "--no-prefix-cache") so.prefix_cache = false;
    else if (a == "--prefix-cache-gb") so.prefix_gb = std::clamp(to_double(a, next()), 0.0, 4096.0);
    else if (a == "--help" || a == "-h") help = true;
    else throw std::runtime_error("unknown option " + a + " (--help lists them; --q2 and --v4 were removed 2026-10-22)");
  }
  if (!help) {
    if (port < 1 || port > 65535) throw std::runtime_error("--port: expected 1..65535 (got " + std::to_string(port) + ")");
    if (host.empty()) throw std::runtime_error("--host: an address or name is required");
    if (!so.ctx_auto && so.eo.ctx < 1) throw std::runtime_error("--ctx: expected 1 or more tokens (or auto)");
    if (so.eo.chunk < 0) throw std::runtime_error("--chunk: expected 0 (auto) or more tokens");
    if (so.eo.mtp_drafts < 0) throw std::runtime_error("--mtp-drafts: expected 0 or more");
  }
  } catch (const std::exception& e) {
    fprintf(stderr, "arcinferzero-server: %s\n", e.what());
    return 2;
  }
  if (model.empty() || help) {
    fprintf(stderr,
            "usage: arcinferzero-server -m model.gguf [options]   OpenAI-compatible server (API at /v1, web UI at /)\n"
            "       arcinferzero-server --list-models             the models it can load, as JSON (Hugging Face cache,\n"
            "                                                 %%USERPROFILE%%\\arcinferzero\\models, model-dirs.txt) with each one's auto context\n"
            "  -m, --model FILE      model GGUF; an mmproj-* image projector and an mtp-* drafter next to it are found\n"
            "  --host H, --port P    listen address (default 127.0.0.1:8080). A browser page of another origin (another\n"
            "                        site, or another port of this computer) reaches only the model endpoints: not the\n"
            "                        tools, files, chats, transcripts, live recording, slots or sessions\n"
            "  --alias NAME          model id in the API (default: the file name)\n"
            "  -c, --ctx N|auto      context in tokens (default 40960); auto: the largest that fits the GPU memory with\n"
            "                        1.2 GiB left for the driver, at most the trained context (remembered in\n"
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
            "  --budget GB           keep decoded weights resident up to GB\n"
            "MoE models (experts read from the SSD into a GPU cache; expert file in\n"
            "%%USERPROFILE%%\\arcinferzero\\zero, made on the first start, safe to delete):\n"
            "  --expert-cache-gb N   expert cache size in GiB (default: automatic: the GPU share Windows reports, your\n"
            "                        setting, less 1.2 GB for the driver, less the rest of the model)\n"
            "  --expert-cache-pct P  expert cache as a percentage of all experts\n"
            "  --pack FILE.gguf      make a package of a MoE model (its first part when split): the non-expert\n"
            "                        weights + an expert file under %%USERPROFILE%%\\arcinferzero\\zero\\models; afterwards\n"
            "                        the original GGUF can be moved away (e.g. to a USB disk)\n"
            "  --no-stream           every expert on the GPU (no SSD streaming; MoE models then get less context)\n"
            "  --attn-f32            f32 attention (tests)\n"
            "  GPU: picked at start: Xe2 / Xe3 GPUs (no 8-lane sub-groups) get 16-lane kernels, and XMX for the prompt\n"
            "  GEMMs when their results match the DP4a kernels' on a test (ARC_NO_XMX=1: DP4a; ARC_NO_SG8=1: 16 lanes)\n"
            "  --kernels DIR         kernel folder (default zero\\kernels);  --webui DIR  web UI folder\n"
            "  --session-cache-gb N  disk space for saved conversations (default 0 = off): switching between\n"
            "                        conversations loads the other one's state instead of processing its prompt again\n"
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
            "                        API; docs/whisper-native.md): auto = faster-whisper large-v3 from the Hugging Face\n"
            "                        cache (large-v3-turbo when it is not there), DIR = a faster-whisper model folder;\n"
            "                        also the web UI's Transcripts / Meetings tabs and voice messages; default off (the\n"
            "                        endpoint then asks the loaded model, if it hears audio); --ctx auto keeps its GPU\n"
            "                        memory free\n"
            "  --whisper-device gpu|npu  Whisper's encoder on the iGPU (default) or the NPU (the decoder: the iGPU)\n"
            "  --whisper-max-minutes N  longest upload or chat clip (default 180); the transcript library takes files\n"
            "                        up to 6 hours, or N when it is more\n"
            "  --asr auto|FILE|off   Qwen3-ASR next to Whisper (needs --whisper), the default speech model: requests,\n"
            "                        transcripts and chat audio go to it unless they name Whisper (docs/qwen3-asr.md);\n"
            "                        auto = Qwen3-ASR-1.7B GGUF from the Hugging Face cache (its mmproj next to it);\n"
            "                        default: auto with --whisper when the cache has it, else Whisper alone\n"
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
  if (alias.empty()) alias = std::filesystem::path(model).stem().string();

  // the speech service before the model: --ctx auto then leaves its GPU memory free (tools/server_speech.h)
  std::unique_ptr<srvwh::Service> wh;
  try {
    wh = srvsp::start(whisper_opt, whisper_dev, asr_opt, whisper_max_min, so);
  } catch (const std::exception& e) {
    fprintf(stderr, "arcinferzero-server: --whisper: %s\n", e.what());
    return 1;
  }
  std::unique_ptr<Loaded> cur;
  try {  // (a model that cannot load: its reason, not a silent abort)
    cur = load_model(model, alias, so);
  } catch (const std::exception& e) {
    fprintf(stderr, "arcinferzero-server: cannot load %s: %s\n", model.c_str(), e.what());
    return 1;
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
  auto find_model = [&](const std::string& want) -> const ModelInfo* {
    for (int pass = 0; pass < 2; ++pass) {
      for (const ModelInfo& m : catalog)
        if (m.supported && (m.alias == want || m.name == want || m.file == want)) return &m;
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
    err.set("message", "no model is loaded (the last model switch failed); ask for a model or restart the server");
    err.set("type", "server_error");
    r.set("error", err);
    return rs.send(503, "application/json", r.dump());
  };
  http.handler = [&](const HttpRequest& rq, HttpResponse& rs) {
    // the web UI's files and chats on disk (tools/server_files.h): no model involved
    if (rq.path == "/v1/files" || rq.path.rfind("/v1/files/", 0) == 0 || rq.path.rfind("/files/", 0) == 0 ||
        rq.path == "/v1/chats" || rq.path.rfind("/v1/chats/", 0) == 0)
      return files_api(rq, rs);
    if (rq.method == "POST" && (rq.path == "/v1/tools/create_file" || rq.path == "/v1/tools/list_files" ||
                                rq.path == "/v1/tools/read_file" || rq.path == "/v1/tools/search_file" ||
                                rq.path == "/v1/tools/search_chats" ||
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
      return file_tools(rq, rs, terminal, ctx, vis);
    }
    // the web UI's tools (tools/web_tools.h): no model involved, so they answer while a model loads or generates
    if (rq.method == "POST" && (rq.path == "/v1/tools/web_search" || rq.path == "/v1/tools/web_fetch")) {
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
      const webt::Page pg = webt::get(url);
      if (!pg.error.empty()) return fail(502, pg.error);
      if (pg.status != 200) return fail(502, "HTTP " + std::to_string(pg.status) + " from " + pg.url);
      const std::string ct = webt::lower(pg.content_type);
      std::string title, text;
      if (ct.find("html") != std::string::npos || ct.empty()) text = webt::html_to_text(pg.body, &title);
      else if (ct.rfind("text/", 0) == 0 || ct.find("json") != std::string::npos || ct.find("xml") != std::string::npos) text = pg.body;
      else return fail(415, "unsupported content type " + pg.content_type + " (text and HTML pages only)");
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
      r.set("url", pg.url);
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
      const int n = std::clamp((int)in["slots"].as_number(), 1, 8);
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
      fprintf(stderr, "slots: %d -> %d, loading %s again\n", cur->eng->slot_count(), n, al.c_str());
      if (cur->srv) cur->srv->save_held();
      cur.reset();
      ServeOpts s2 = prev;
      s2.eo.slots = n;
      s2.ctx_auto = true;
      s2.warm_prompt = false;
      try {
        cur = load_model(path, al, s2);
        cur_so = s2;
        so.eo.slots = n;  // (the next model switches too)
        fprintf(stderr, "slots: %d, loaded in %.1f s\n", n, ms_since(t0) / 1000);
        return reply();
      } catch (const std::exception& e) {
        fprintf(stderr, "slots: could not load %s with %d slots: %s; loading it as before\n", al.c_str(), n, e.what());
        std::string also;
        try {
          cur = load_model(path, al, prev);
        } catch (const std::exception& e2) {
          cur.reset();
          also = std::string("; loading it as before failed too: ") + e2.what();
        }
        return fail(500, std::string("could not load the model with ") + std::to_string(n) + " slots: " + e.what() + also);
      }
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
    {  // (no model: a switch failed and so did loading the previous model again; a chat request naming one loads it)
      std::shared_lock<std::shared_mutex> lk(swap_mu);
      if (!cur && !(swap && chat_post)) {
        Json err = Json::object(), r = Json::object();
        err.set("message", "no model is loaded (the last model switch failed); ask for a model or restart the server");
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
        if (speech == "whisper" || (speech != "model" && !hears)) {
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
    if (swap && (rq.path == "/v1/models" || rq.path == "/models")) {  // every model a request can switch to
      std::shared_lock<std::shared_mutex> lk(swap_mu);
      if (!cur) return no_model(rs);  // (a failed switch: before, cur->alias of no model)
      Json d = Json::array();
      std::map<std::string, uint64_t> sizes;  // (llama_models "size")
      bool listed = false;
      for (const ModelInfo& m : catalog) {
        if (!m.supported) continue;
        const bool loaded = !listed && m.alias == cur->alias;
        listed = listed || loaded;
        Json j = Json::object();
        j.set("id", m.alias);
        j.set("object", "model");
        j.set("owned_by", "arcinferzero");
        j.set("name", m.name);
        j.set("loaded", loaded);
        if (loaded) j.set("context_length", cur->eng->ctx);
        set_window(j, loaded ? cur->eng->ctx : 0, m.ctx_train, loaded ? cur->eng->hp.n_vocab : 0, loaded ? cur->eng->hp.n_embd : 0);
        // (the loaded model: the encoders that did load)
        set_modalities(j, loaded ? cur->srv->has_vision() : m.vision, loaded ? cur->srv->has_audio() : m.audio);
        sizes[m.alias] = m.bytes;
        // the loaded model first (clients that take the first entry get the running one)
        if (loaded) {
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
        j.set("owned_by", "arcinferzero");
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
          fprintf(stderr, "switching model: %s -> %s\n", prev_alias.c_str(), m->alias.c_str());
          if (cur && cur->srv) cur->srv->save_held();  // (switching back later loads the conversation instead of redoing it)
          cur.reset();  // frees the GPU memory first
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
            fprintf(stderr, "switched to %s in %.1f s\n", m->alias.c_str(), ms_since(t0) / 1000);
            // (pi-arc starts with this model next time; not for the transcript service's Gemma 4 E4B requests, which
            //  carry "x-arc-speech: 1" (tools/server_speech.h attach_e4b): the user did not choose that model)
            if (rq.header("x-arc-speech").empty()) std::ofstream(app_dir() + "/last-model.txt") << m->path << "\n";
          } catch (const std::exception& e) {
            fprintf(stderr, "could not load %s: %s; loading %s again\n", m->path.c_str(), e.what(), prev_alias.c_str());
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
              cur.reset();
              also = std::string("; loading ") + prev_alias + " again failed too: " + e2.what();
              fprintf(stderr, "could not load %s again: %s; no model loaded\n", prev_alias.c_str(), e2.what());
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
    std::shared_lock<std::shared_mutex> lk(swap_mu);
    if (!cur) {  // (a chat request naming no model, or an unknown one, while none is loaded)
      Json err = Json::object(), r = Json::object();
      err.set("message", "no model is loaded (the last model switch failed); ask for a model or restart the server");
      err.set("type", "server_error");
      r.set("error", err);
      return rs.send(503, "application/json", r.dump());
    }
    cur->srv->handle(rq, rs);
  };
  if (!http.listen(host, port)) {
    fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port);
    return 1;
  }
  fprintf(stderr, "listening on http://%s:%d  (OpenAI API at /v1, web UI at /)\n", host.c_str(), port);
  if (host != "127.0.0.1" && host != "localhost" && host != "::1")  // (review 6.2: there is no authentication)
    fprintf(stderr, "warning: listening on %s: anyone who can reach this address can use the model (no API key)\n",
            host.c_str());
  // (the saved transcripts' queue: started once the server answers; stopped before the locals its GPU turns use go)
  if (wh) wh->start_library(std::filesystem::path(app_dir()) / "transcripts");
  http.run();
  if (wh) wh->stop_library();
  return 0;
}

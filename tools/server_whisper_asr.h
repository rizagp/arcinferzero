// Qwen3-ASR and Gemma 4 E4B in the speech service: loading Qwen3-ASR (zero/asr_qwen3.h), which model a request names,
// and running either into a whisper::Result. A part of class srvwh::Service (tools/server_whisper.h), included inside
// the class body (its private part): not a header of its own. docs/qwen3-asr.md.

 public:
  // ---------------------------------------------------------------- Qwen3-ASR next to Whisper (--asr)
  // zero/asr_qwen3.h on its own engine: a request (or a transcript's options) with "model" naming it ("qwen3-asr",
  // asr_id) is transcribed by it instead of Whisper. Its context holds parts of up to kAsrPartSeconds (longer audio is
  // cut at the quietest point near each; a meeting's tracks at pauses, kAsrTrackPartSeconds); one segment per part
  // (the model gives no timestamps).
  static constexpr int kAsrCtx = 6144;  // (f16 K and V: ~0.7 GB; shorter parts below instead of more GPU memory)
  // (a part's context: its audio (13 tokens a second), the system turn (glossary and prompt, cut to
  //  kAsrContextTokens), ~50 tokens of template and its text (speech 4.5-6 tokens a second, 7 kept free); split may
  //  cut 5 s past kAsrPartSeconds. 360-s parts left ~1,400 tokens for 6 minutes of text: their ends were lost)
  static constexpr int kAsrContextTokens = 512;
  static constexpr int kAsrTextPerSecond = 7;
  static constexpr double kAsrPartSeconds = 240;
  static_assert((kAsrPartSeconds + 5) * (AudioQwen3A::kChunkTok + kAsrTextPerSecond) + kAsrContextTokens + 64 <= kAsrCtx);
  // (a meeting's tracks: each cut at pauses into parts of about this long, so their segments interleave by time
  //  instead of 4-minute monologues one track after the other)
  static constexpr double kAsrTrackPartSeconds = 25;
  std::string asr_id;  // "qwen3-asr-1.7b"; empty: not loaded
  void load_asr(const std::string& gguf, const std::string& kernel_dir) {
    if (q3asr::find_mmproj(gguf).empty()) throw std::runtime_error("no Qwen3-ASR mmproj (projector qwen3a) next to " + gguf);
    EngineOptions eo;
    eo.ctx = kAsrCtx;
    eo.kernel_dir = kernel_dir;
    // (the GEMM form decoded just in time per 512-token chunk, not kept: 1.37 GiB less, the prompt as fast (2048
    //  tokens 927 vs 938 tok/s), KL unchanged; the 2:34 recording 24.0 s vs 29.3 s, the same text)
    eo.w8_budget_gb = 0;
    eo.chunk = 512;
    // (kept only once all of it loaded: a throw frees what was made, the service goes on with Whisper alone)
    auto eng = std::make_unique<Engine>(gguf, eo);
    if (!eng->hp.qwen3) throw std::runtime_error(gguf + ": not a Qwen3-ASR text model");
    const std::string mm = q3asr::find_mmproj(gguf, eng->hp.n_embd);  // (several: the one as wide as the model first)
    auto enc = std::make_unique<AudioQwen3A>(mm, *eng);
    if (enc->out_dim() != eng->hp.n_embd)
      throw std::runtime_error(mm + ": audio rows of " + std::to_string(enc->out_dim()) + ", " + gguf + " takes " +
                               std::to_string(eng->hp.n_embd) + " (the mmproj of another Qwen3-ASR size)");
    auto tok = std::make_unique<Tokenizer>(eng->gguf());
    asr_bytes_ = eng->gpu_bytes + AudioQwen3A::gpu_bytes(GgufFile(mm));
    asr_eng_ = std::move(eng);
    asr_enc_ = std::move(enc);
    asr_tok_ = std::move(tok);
    std::string n = std::filesystem::path(gguf).stem().string();
    for (char& c : n) c = (char)tolower((unsigned char)c);
    const size_t q = n.find("-q");  // (the quantization suffix: -q8_0, -q4_k_m ...)
    if (q != std::string::npos && q > 0) n.resize(q);
    asr_id = n;
  }
  // a "model" field naming Qwen3-ASR
  bool names_asr(std::string m) const {
    for (char& c : m) c = (char)tolower((unsigned char)c);
    return m.find("qwen3-asr") != std::string::npos || (!asr_id.empty() && m == asr_id);
  }
  // a "model" field naming Whisper itself ("whisper", its id); OpenAI's generic "whisper-1" is not: it takes the default
  bool names_whisper(std::string m) const {
    for (char& c : m) c = (char)tolower((unsigned char)c);
    return m == "whisper" || m == id || m.rfind("whisper-large", 0) == 0 || m.rfind("faster-whisper", 0) == 0;
  }

  // ---------------------------------------------------------------- Gemma 4 E4B as a transcript model
  // The server's chat model that hears audio (tools/server_speech.h attach_e4b sets chat_asr: a chat request to it
  // with the clip through the server's dispatch, marked as the speech service's own; the server loads that model if
  // another one is loaded). The clip goes in parts of up to kChatPartSeconds, cut at the quietest point near each; one
  // segment per part.
  static constexpr double kChatPartSeconds = 120;
  std::string chat_asr_id;  // its model id ("gemma-4-e4b..."); empty: none
  std::function<std::string(const std::string& wav, double secs, const std::string& language, const std::string& context)>
      chat_asr;
  bool names_chat_asr(std::string m) const {
    for (char& c : m) c = (char)tolower((unsigned char)c);
    return !m.empty() && (m == chat_asr_id || m == "e4b" || m == "gemma-4-e4b" || m == "gemma4-e4b");
  }

 private:
  // Qwen3-ASR on 16 kHz samples as a whisper::Result: a segment per part (its time span; its language's code, Malay
  // as Indonesian when the request's languages have id and not ms: the model often names Indonesian speech Malay);
  // the progress callback after each part and about once a second while it decodes (false stops it: the text so far
  // is kept). The engine's calls take the Whisper engine's turn and the GPU lock, as Whisper's windows do; the engine
  // itself is held a part at a time (asr_turn: a queued job lets a waiting clip or /v1/audio request in between its
  // parts). pause_parts > 0: parts cut at pauses once about that long (a meeting's track). A part that fails is tried
  // once more; then, with error, the segments so far are returned and error says what failed (else it throws).
  whisper::Result asr_run(const std::vector<float>& audio, const whisper::Options& opt, const whisper::Progress& cb,
                          bool queued = false, std::string* error = nullptr, double pause_parts = 0) {
    using clk = std::chrono::steady_clock;
    whisper::Result r;
    r.duration = (double)audio.size() / 16000;
    auto has = [&](const char* code) {
      const std::string l = "," + opt.language + ",";
      return l.find("," + std::string(code) + ",") != std::string::npos;
    };
    q3asr::Options o;
    o.max_seconds = kAsrPartSeconds + 10;  // (each call below gets one part of the cuts here)
    std::string context = opt.terms;
    if (!opt.initial_prompt.empty()) context += (context.empty() ? "" : "\n") + opt.initial_prompt;
    if (std::vector<int> ids = asr_tok_->encode(context, false); (int)ids.size() > kAsrContextTokens) {
      ids.resize(kAsrContextTokens);  // (the glossary first, then the prompt: its end goes)
      context = asr_tok_->decode(ids);
      size_t i = context.size();  // (no UTF-8 character cut in two)
      while (i > 0 && ((unsigned char)context[i - 1] & 0xC0) == 0x80) --i;
      if (i > 0) {
        const unsigned char c = (unsigned char)context[i - 1];
        const size_t n = c < 0x80 ? 1 : c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : 2;
        if (i - 1 + n > context.size()) context.resize(i - 1);
      }
      fprintf(stderr, "asr: the glossary and prompt cut to their first %d tokens\n", kAsrContextTokens);
    }
    o.context = context;
    o.gpu = [this](const std::function<void()>& f) { turn_.run(false, [&] { gpu_turn(f); }); };
    double done = 0, off = 0;  // (off: the part's start in the clip)
    bool stop = false;
    auto last = clk::now();
    o.on_part = [&](const q3asr::Part& p) {
      done = off + p.end;
      if (p.cut)
        fprintf(stderr, "asr: %.1f-%.1f s: no room left to decode (%d audio tokens, %d tokens): its text's end is lost\n",
                off + p.start, off + p.end, p.audio_tokens, p.tokens);
      if (!p.text.empty()) {
        whisper::Segment g;
        g.start = off + p.start;
        g.end = off + p.end;
        g.text = " " + p.text;
        g.lang = q3asr::language_code(p.language);
        if (g.lang == "ms" && has("id") && !has("ms")) g.lang = "id";
        r.segments.push_back(std::move(g));
      }
      if (cb && !stop) stop = !cb(done, r.duration, r.segments);
      last = clk::now();
    };
    o.cancelled = [&] {
      if (!stop && cb && clk::now() - last > std::chrono::seconds(1)) {
        stop = !cb(done, r.duration, r.segments);
        last = clk::now();
      }
      return stop;
    };
    const auto parts = q3asr::parts_at_pauses(audio, pause_parts, kAsrPartSeconds);
    double enc_ms = 0, pre_ms = 0, dec_ms = 0;
    int audio_tok = 0, tokens = 0, cut = 0;
    auto report = [&] {
      fprintf(stderr, "asr: %.1f s of audio, %d parts%s: encoder %.0f ms, prompt %.0f ms (%d audio tokens), %d tokens in %.0f ms\n",
              r.duration, (int)parts.size(), cut ? " (some cut: see above)" : "", enc_ms, pre_ms, audio_tok, tokens, dec_ms);
    };
    for (const auto& [a, b] : parts) {
      if (stop) break;
      off = a / 16000.0;
      const std::vector<float> clip(audio.begin() + a, audio.begin() + b);
      q3asr::Result q;
      for (int attempt = 0;; ++attempt) {
        try {
          asr_turn(queued, [&] {
            last = clk::now();
            q = q3asr::transcribe(*asr_eng_, *asr_enc_, *asr_tok_, clip, o);
          });
          break;
        } catch (const std::exception& e) {
          fprintf(stderr, "asr: %.1f-%.1f s failed%s: %s\n", a / 16000.0, b / 16000.0, attempt ? "" : " (trying once more)", e.what());
          if (attempt == 0) continue;
          if (!error) throw;
          char at[64];
          snprintf(at, sizeof at, "at %.0f-%.0f s: ", a / 16000.0, b / 16000.0);
          *error = at + std::string(e.what());
          report();
          return r;
        }
      }
      enc_ms += q.encode_ms;
      pre_ms += q.prefill_ms;
      dec_ms += q.decode_ms;
      audio_tok += q.audio_tokens;
      tokens += q.tokens;
      for (const q3asr::Part& p : q.parts) cut += p.cut;
    }
    report();
    return r;
  }
  // The Qwen3-ASR engine for f (a part): one caller at a time, a queued job only when no other caller waits.
  template <class F>
  void asr_turn(bool queued, F&& f) {
    std::unique_lock<std::mutex> lk(asr_mu_);
    if (!queued) ++asr_waiting_;
    asr_cv_.wait(lk, [&] { return !asr_busy_ && (!queued || asr_waiting_ == 0); });
    if (!queued) --asr_waiting_;
    asr_busy_ = true;
    lk.unlock();
    struct Done {
      Service& s;
      ~Done() {
        std::lock_guard<std::mutex> g(s.asr_mu_);
        s.asr_busy_ = false;
        s.asr_cv_.notify_all();
      }
    } done{*this};
    f();
  }

  // Gemma 4 E4B on 16 kHz samples (chat_asr) as a whisper::Result: parts of up to kChatPartSeconds, a segment each
  // (language: the request's when it names one); the progress callback after each part (false stops before the next).
  // A part that fails is tried once more; then, with error, the segments so far are returned and error says what
  // failed (else it throws).
  whisper::Result chat_run(const std::vector<float>& audio, const whisper::Options& opt, const whisper::Progress& cb,
                           std::string* error = nullptr) {
    whisper::Result r;
    r.duration = (double)audio.size() / 16000;
    const std::string lang = opt.language == "auto" || opt.language.find(',') != std::string::npos ? "" : opt.language;
    std::string context = opt.terms;
    if (!opt.initial_prompt.empty()) context += (context.empty() ? "" : "\n") + opt.initial_prompt;
    for (const auto& [a, b] : q3asr::split(audio, kChatPartSeconds)) {
      std::string wav = "RIFF0000WAVEfmt ";  // (16 kHz 16-bit mono PCM)
      auto u32 = [&](uint32_t v) { wav.append((const char*)&v, 4); };
      auto u16 = [&](uint16_t v) { wav.append((const char*)&v, 2); };
      u32(16); u16(1); u16(1); u32(16000); u32(32000); u16(2); u16(16);
      wav += "data";
      u32((uint32_t)((b - a) * 2));
      for (size_t i = a; i < b; ++i) u16((uint16_t)(int16_t)std::lround(std::clamp(audio[i], -1.0f, 1.0f) * 32767));
      const uint32_t riff = (uint32_t)wav.size() - 8;
      memcpy(&wav[4], &riff, 4);
      whisper::Segment g;
      g.start = a / 16000.0;
      g.end = b / 16000.0;
      for (int attempt = 0;; ++attempt) {
        try {
          g.text = whisper::trim_copy(chat_asr(wav, g.end - g.start, lang, context));
          break;
        } catch (const std::exception& e) {
          fprintf(stderr, "e4b: %.1f-%.1f s failed%s: %s\n", g.start, g.end, attempt ? "" : " (trying once more)", e.what());
          if (attempt == 0) continue;
          if (!error) throw;
          char at[64];
          snprintf(at, sizeof at, "at %.0f-%.0f s: ", g.start, g.end);
          *error = at + std::string(e.what());
          return r;
        }
      }
      g.lang = lang;
      if (!g.text.empty()) {
        g.text = " " + g.text;
        r.segments.push_back(std::move(g));
      }
      if (cb && !cb(b / 16000.0, r.duration, r.segments)) break;
    }
    return r;
  }

  std::unique_ptr<Engine> asr_eng_;
  std::unique_ptr<AudioQwen3A> asr_enc_;
  std::unique_ptr<Tokenizer> asr_tok_;
  size_t asr_bytes_ = 0;
  std::mutex asr_mu_;
  std::condition_variable asr_cv_;
  bool asr_busy_ = false;
  int asr_waiting_ = 0;  // (callers other than the queue waiting for the engine)

// Qwen3-ASR and Gemma 4 E4B in the speech service: loading Qwen3-ASR (zero/asr_qwen3.h), which model a request names,
// and running either into a whisper::Result. A part of class srvwh::Service (tools/server_whisper.h), included inside
// the class body (its private part): not a header of its own. docs/qwen3-asr.md.

 public:
  // ---------------------------------------------------------------- Qwen3-ASR next to Whisper (--asr)
  // zero/asr_qwen3.h on its own engine: a request (or a transcript's options) with "model" naming it ("qwen3-asr",
  // asr_id) is transcribed by it instead of Whisper. Its context holds parts of up to kAsrPartSeconds (longer audio is
  // cut at the quietest point near each); one segment per part (the model gives no timestamps).
  static constexpr int kAsrCtx = 6144;                // (float K / V: ~1 GB)
  static constexpr double kAsrPartSeconds = 360;      // (13 tokens a second + the prompt and the text within kAsrCtx)
  std::string asr_id;  // "qwen3-asr-1.7b"; empty: not loaded
  void load_asr(const std::string& gguf, const std::string& kernel_dir) {
    const std::string mm = q3asr::find_mmproj(gguf);
    if (mm.empty()) throw std::runtime_error("no Qwen3-ASR mmproj (projector qwen3a) next to " + gguf);
    EngineOptions eo;
    eo.ctx = kAsrCtx;
    eo.kernel_dir = kernel_dir;
    // (the GEMM form decoded just in time per 512-token chunk, not kept: 1.37 GiB less, the prompt as fast (2048
    //  tokens 927 vs 938 tok/s), KL unchanged; the 2:34 recording 24.0 s vs 29.3 s, the same text)
    eo.w8_budget_gb = 0;
    eo.chunk = 512;
    asr_eng_ = std::make_unique<Engine>(gguf, eo);
    if (!asr_eng_->hp.qwen3) throw std::runtime_error(gguf + ": not a Qwen3-ASR text model");
    asr_enc_ = std::make_unique<AudioQwen3A>(mm, *asr_eng_);
    asr_tok_ = std::make_unique<Tokenizer>(asr_eng_->gguf());
    asr_bytes_ = asr_eng_->gpu_bytes + AudioQwen3A::gpu_bytes(GgufFile(mm));
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
  // The server's chat model that hears audio (tools/server.cpp sets chat_asr: a chat request to it with the clip, as
  // /v1/audio/transcriptions with a chat model does; the server loads that model if another one is loaded). The clip
  // goes in parts of up to kChatPartSeconds, cut at the quietest point near each; one segment per part.
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
  // is kept). The engine's calls take the Whisper engine's turn and the GPU lock, as Whisper's windows do.
  whisper::Result asr_run(const std::vector<float>& audio, const whisper::Options& opt, const whisper::Progress& cb) {
    using clk = std::chrono::steady_clock;
    std::lock_guard<std::mutex> one(asr_mu_);
    whisper::Result r;
    r.duration = (double)audio.size() / 16000;
    auto has = [&](const char* code) {
      const std::string l = "," + opt.language + ",";
      return l.find("," + std::string(code) + ",") != std::string::npos;
    };
    q3asr::Options o;
    o.max_seconds = kAsrPartSeconds;
    o.context = opt.terms;
    if (!opt.initial_prompt.empty()) o.context += (o.context.empty() ? "" : "\n") + opt.initial_prompt;
    o.gpu = [this](const std::function<void()>& f) { turn_.run(false, [&] { gpu_turn(f); }); };
    double done = 0;
    bool stop = false;
    auto last = clk::now();
    o.on_part = [&](const q3asr::Part& p) {
      done = p.end;
      if (!p.text.empty()) {
        whisper::Segment g;
        g.start = p.start;
        g.end = p.end;
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
    const q3asr::Result q = q3asr::transcribe(*asr_eng_, *asr_enc_, *asr_tok_, audio, o);
    fprintf(stderr, "asr: %.1f s of audio, %d parts: encoder %.0f ms, prompt %.0f ms (%d audio tokens), %d tokens in %.0f ms\n",
            r.duration, (int)q.parts.size(), q.encode_ms, q.prefill_ms, q.audio_tokens, q.tokens, q.decode_ms);
    return r;
  }

  // Gemma 4 E4B on 16 kHz samples (chat_asr) as a whisper::Result: parts of up to kChatPartSeconds, a segment each
  // (language: the request's when it names one); the progress callback after each part (false stops before the next)
  whisper::Result chat_run(const std::vector<float>& audio, const whisper::Options& opt, const whisper::Progress& cb) {
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
      g.text = whisper::trim_copy(chat_asr(wav, g.end - g.start, lang, context));
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

// Loading a model: engine options, tokenizer, encoders, the Server around them.
// Included by tools/server.cpp only, at the place it was cut from (after class Server); not a header of its
// own. Split out of tools/server.cpp unchanged on 2026-10-24 (QUEUE item 11).

// ---------------------------------------------------------------- loading a model
// the files next to a model ("-m model.gguf" without a folder: the current one; an unreadable folder: none)
static std::vector<std::filesystem::directory_entry> model_dir_files(const std::string& model) {
  std::filesystem::path dir = std::filesystem::path(model).parent_path();
  if (dir.empty()) dir = ".";
  std::vector<std::filesystem::directory_entry> out;
  std::error_code ec;
  for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) out.push_back(*it);
  return out;
}
// the model's own projector next to it, or "" (and the log says when the folder has others: --mmproj takes one)
static std::string own_mmproj(const std::string& model) {
  const auto own = model_mmprojs(std::filesystem::path(model));
  if (!own.empty()) return own[0].string();
  int others = 0;
  for (const auto& de : model_dir_files(model))
    others += de.path().extension() == ".gguf" && lower_ascii(de.path().filename().string()).find("mmproj") != std::string::npos;
  if (others)
    zlog::eprintf("vision: %d projector file(s) next to the model, none named after it (a folder of several models): "
                    "--mmproj FILE uses one\n", others);
  return "";
}
struct ServeOpts {
  EngineOptions eo;
  bool ctx_auto = false;
  bool warm_prompt = true;  // (the server's start: also a long warm-up prompt; a switch for a request skips it)
  std::string webui, mmproj, sess_dir;
  int think_budget = -1;
  bool preserve_thinking = false;
  double sess_gb = 0.0;
  bool prefix_cache = true;
  double prefix_gb = 8.0;  // all models' saved system prompts together (0 = no limit)
  int img_min = 0, img_max = 0;  // image tokens per image (0 = the encoder's default)
  bool budget_given = false;     // --budget: no automatic choice of resident weights
  bool dflash_auto = true;       // inferzero, with MTP: a DFlash2 drafter found for the model drafts too (--no-dflash: not)
  size_t whisper_bytes = 0;      // GPU memory of the Whisper engine loaded before the model (--whisper): kept free
  Json whisper_props;            // its /props entry
  // --asr-slots: Qwen3-ASR parts decoded together (1-4; each slot past the first ~0.55 GiB of GPU memory, faster only
  // for clips of more than 4 minutes: the 14-minute file 129 -> 80 s with 4; 1 by default since 2026-10-30)
  int asr_slots = 1;
  // --gpus N: the model split over N GPUs (zero/mgpu): the server drives shard 0, its calls mirrored on the others.
  // Qwen3.5 / 3.6 / 3.8 text only for now: no MTP / DFlash2, images, saved prompts or sessions, warm-up
  int gpus = 1;
};
// the engine a Loaded refers to: its own, or shard 0 of its TpEngine (which owns and frees it)
struct EngineHold {
  bool own = true;
  void operator()(Engine* e) const {
    if (own) delete e;
  }
};
// a split model: deleted, unless a GPU fault left a thread running on it (TpEngine::faulted: then left for good)
struct TpHold {
  void operator()(TpEngine* t) const {
    if (t && t->faulted()) {
      t->disarm(true);
      zlog::eprintf("multi-GPU: the split model's memory is left after a GPU fault: restart the server to free it\n");
      return;
    }
    delete t;
  }
};
struct Loaded {  // destroyed in reverse order: the server before the tokenizer and engine it refers to
  std::unique_ptr<TpEngine, TpHold> tp;  // (--gpus: the split model, freed after everything that uses its shard 0)
  std::unique_ptr<Engine, EngineHold> eng;
  std::unique_ptr<Tokenizer> tok;
  std::unique_ptr<Server> srv;
  std::string alias, path;
};
static std::unique_ptr<Loaded> load_model(const std::string& model, const std::string& alias, ServeOpts so) {
  auto L = std::make_unique<Loaded>();
  L->alias = alias;
  L->path = model;
  zlog::eprintf("inferzero-server: loading %s\n", model.c_str());
  const ServeOpts so_given = so;  // (failsafe: a split that cannot load loads on one GPU with these)
  if (so.gpus > 1) {  // (what a split model does not do yet: see ServeOpts::gpus)
    so.eo.mtp = false;
    so.eo.dflash.clear();
    so.dflash_auto = false;
    so.mmproj = "-";
    so.prefix_cache = false;
    so.sess_gb = 0;
  }
  EngineOptions eo = so.eo;
  bool dfl_auto = false;
  if (eo.mtp && eo.dflash.empty() && so.dflash_auto) {  // (before --ctx auto: the drafter takes GPU memory)
    eo.dflash = find_dflash(model);
    dfl_auto = !eo.dflash.empty();
    if (!eo.dflash.empty()) zlog::eprintf("DFlash2 drafter found: %s (--no-dflash: MTP block only)\n", eo.dflash.c_str());
  }
  if (eo.mtp && !eo.mtp_block && eo.dflash.empty()) {
    zlog::eprintf("--draft dflash: no DFlash2 drafter for this model, the MTP block drafts\n");
    eo.mtp_block = true;
  }
  if (eo.mtp && !eo.dflash.empty())
    zlog::eprintf("drafting: %s\n", eo.mtp_block ? "DFlash2 and the MTP block, picked per step" : "DFlash2 only (no MTP block)");
  // Gemma 4 with an image projector: an image's tokens attend to each other both ways, so one image is one forward
  // step; the steps must hold the largest image (resident E4B weights otherwise step through prompts in 512s)
  if (so.mmproj != "-") {
    bool gemma = false, proj = false;
    try {  // (E2B / E4B, with per-layer embeddings, process images causally in any steps)
      const GgufFile g(model);
      gemma = g.get_str("general.architecture", "") == "gemma4" && g.get_int("gemma4.embedding_length_per_layer_input", 0) == 0;
    } catch (const std::exception&) {
    }
    if (gemma && so.mmproj.empty()) proj = !model_mmprojs(model).empty();
    if (gemma && (proj || !so.mmproj.empty())) eo.min_chunk = std::max(so.img_max, 2048);
  }
  AutoCtx used_ctx;  // (--ctx auto's count, for the log's memory line)
  bool have_ctx = false;
  if (so.ctx_auto) {
    const size_t enc = encoder_reserve(model, so.mmproj, so.img_max) + so.whisper_bytes;
    if (enc) zlog::eprintf("--ctx auto: %.2f GiB kept for the image / audio encoders%s\n", enc / 1073741824.0, so.whisper_bytes ? " and speech to text" : "");
    // (a split model, --gpus N: the probe is shard 0 of N, the largest (it takes the extra unit of a count that does
    //  not divide, the shared expert, ...): its weights, buffers and its share of the KV against GPU 0's memory. Before,
    //  it sized the whole model on one GPU, where a model bigger than one GPU got the 4096 floor. A model that cannot be
    //  split: the one-GPU probe, as the load then falls back to one GPU)
    auto probe = [&](EngineOptions o) {
      if (so.gpus > 1) {
        EngineOptions s = o;
        s.tp_size = so.gpus, s.tp_rank = 0;
        try {
          return auto_ctx(model, s, so.budget_given, enc);
        } catch (const std::exception& e) {
          LOG_WRN("--ctx auto: the split over %d GPUs could not be sized (%s): sized as on one GPU\n", so.gpus, e.what());
        }
      }
      return auto_ctx(model, o, so.budget_given, enc);
    };
    AutoCtx a = probe(eo);
    if (dfl_auto && eo.mtp_block && a.ctx < kDflashMinCtx) {  // (a drafter that leaves little context: the MTP block only)
      zlog::eprintf("DFlash2 drafter not used: it would leave %d tokens of context (--dflash DIR: use it anyway)\n", a.ctx);
      eo.dflash.clear();
      a = probe(eo);
    }
    // (a split model whose shard does not fit either: the floor, as before; the split's own load checks each GPU and
    //  falls back to one GPU, where the refusal below then speaks)
    if (a.ctx <= 0 && so.gpus > 1) a.ctx = 4096;
    if (a.ctx <= 0) {  // (refused here, before any GPU memory is taken: a switch goes back to the model before it)
      char b[400];
      snprintf(b, sizeof b, "it does not fit in the GPU memory%s: it needs %.1f GiB without any context, %.1f GiB are kept "
                            "for the encoders and speech to text, and Windows gives the GPU %.1f GiB now. Unloading speech "
                            "to text (Settings, GPU memory) or a smaller model leaves it room",
               enc ? " next to the image / audio encoders and speech to text" : "",
               (a.weights + a.mtp + a.buffers) / 1073741824.0, enc / 1073741824.0, a.dev / 1073741824.0);
      throw std::runtime_error(b);
    }
    eo.ctx = a.ctx;
    used_ctx = a, have_ctx = true;
    if (!so.budget_given) eo.w8_budget_gb = a.resident ? 1e9 : eo.w8_budget_gb;
    // (the weights the probe put in host memory: the load puts the same, not fewer when RAM got scarcer since, so the
    //  context chosen still fits on the GPU)
    if (a.host_gb > 0 && eo.host_weights_gb < 0) eo.host_weights_gb = a.host_gb;
  }
  eo.stream_reserve = encoder_reserve(model, so.mmproj, so.img_max) + so.whisper_bytes;  // (streamed experts' cache)
  if (so.gpus > 1) {
    // (failsafe: a model that cannot be split (Gemma 4 for transcripts, Flash Next) or a split that fails to load
    //  (memory, the GPUs' layouts) loads on one GPU instead, with every feature, rather than failing the server)
    try {
      L->tp.reset(new TpEngine(model, eo, so.gpus));
    } catch (const std::exception& e) {
      zlog::eprintf("multi-GPU: %s; loading %s on one GPU instead\n", e.what(), alias.c_str());
      ServeOpts one = so_given;
      one.gpus = 1;
      return load_model(model, alias, one);
    }
    L->tp->serve();
    L->eng = std::unique_ptr<Engine, EngineHold>(&L->tp->shard(0), EngineHold{false});
  } else {
    L->eng = std::unique_ptr<Engine, EngineHold>(new Engine(model, eo));
  }
  Engine& eng = *L->eng;
  L->tok = std::make_unique<Tokenizer>(eng.gguf());
  zlog::eprintf("loaded in %.1f s | %d layers | ctx %d%s | chunk %d | GPU %.2f GiB%s | weights %s\n", eng.load_ms / 1000,
          eng.hp.n_layer, eng.ctx, eng.slot_count() > 1 ? (" in each of " + std::to_string(eng.slot_count()) + " slots").c_str() : "",
          eng.max_chunk, eng.gpu_bytes / 1073741824.0, so.gpus > 1 ? (" on each of " + std::to_string(so.gpus) + " GPUs").c_str() : "",
          eng.w8_resident ? "W8 resident" : "W8 just-in-time");
  try {  // (the log's model and memory lines, docs/logging.md; /metrics' model_info: never a reason the load fails)
    const auto& h = eng.hp;
    // (general.file_type as llama.cpp names it: the main quant; a mixed file such as UD-Q4_K_XL says its base)
    static const std::map<int, const char*> kFt = {
        {0, "F32"}, {1, "F16"}, {2, "Q4_0"}, {3, "Q4_1"}, {7, "Q8_0"}, {8, "Q5_0"}, {9, "Q5_1"}, {10, "Q2_K"},
        {11, "Q3_K_S"}, {12, "Q3_K_M"}, {13, "Q3_K_L"}, {14, "Q4_K_S"}, {15, "Q4_K_M"}, {16, "Q5_K_S"}, {17, "Q5_K_M"},
        {18, "Q6_K"}, {19, "IQ2_XXS"}, {20, "IQ2_XS"}, {21, "Q2_K_S"}, {22, "IQ3_XS"}, {23, "IQ3_XXS"}, {24, "IQ1_S"},
        {25, "IQ4_NL"}, {26, "IQ3_S"}, {27, "IQ3_M"}, {28, "IQ2_S"}, {29, "IQ2_M"}, {30, "IQ4_XS"}, {31, "IQ1_M"},
        {32, "BF16"}, {36, "TQ1_0"}, {37, "TQ2_0"}};
    std::string ftype = "unknown";
    try {
      const int ft = (int)eng.gguf().get_int("general.file_type", -1);
      const auto it = kFt.find(ft);
      ftype = it != kFt.end() ? it->second : ft >= 0 ? "type " + std::to_string(ft) : "unknown";
    } catch (const std::exception&) {}
    zlog::eprintf("model: %s, %d layers, embedding %d, heads %d / kv %d, head %d, ffn %d, vocab %d, trained ctx %lld%s, "
                  "file %.2f GiB, %s\n", h.arch.c_str(), h.n_layer, h.n_embd, h.n_head, h.n_head_kv, h.head_dim,
                  h.n_ff, h.n_vocab, (long long)eng.gguf().get_int(h.arch + ".context_length", 0),
                  h.n_expert ? (", experts " + std::to_string(h.n_expert) + " (" + std::to_string(h.n_expert_used) + " used)").c_str() : "",
                  gguf_bytes(model) / 1073741824.0, ftype.c_str());
    // (only when the count is known: an old ctx-auto.txt row has no parts; "left" only for one slot on one GPU: with
    //  --slots the context is one slot's, with --gpus the count is one GPU's (the split's own load checks each))
    if (have_ctx && used_ctx.dev > 0 && used_ctx.weights > 0) {
      const double g = 1073741824.0;
      const bool one = eng.slot_count() <= 1 && so.gpus <= 1;
      const double left = used_ctx.dev - used_ctx.keep - used_ctx.snap - used_ctx.reserve - used_ctx.weights -
                          used_ctx.mtp - used_ctx.buffers - used_ctx.context;
      char tail[64] = "";
      if (one) snprintf(tail, sizeof tail, " + left %.2f", left / g);
      LOG_INF("memory (--ctx auto): device %.2f GiB = kept free %.2f + snapshots %.2f + encoders / speech %.2f + "
              "weights %.2f + MTP / drafter %.2f + buffers %.2f + context %.2f%s (%.1f MiB per 1K tokens)%s\n",
              used_ctx.dev / g, used_ctx.keep / g, used_ctx.snap / g, used_ctx.reserve / g, used_ctx.weights / g,
              used_ctx.mtp / g, used_ctx.buffers / g, used_ctx.context / g, one ? "" : " (one slot, one GPU)",
              used_ctx.per_1k / 1048576.0, tail);
    }
    zmet::loaded(alias, h.arch, eng.ctx, so.gpus, (double)eng.gpu_bytes);
  } catch (const std::exception& e) {
    LOG_WRN("model information incomplete: %s\n", e.what());
  }
  bool warmed = false;  // (the kernels compiled now, first start after an update: the audio encoder's too, below)
  if (so.gpus == 1) {  // compile kernels before serving (all of the generation's after an update; --gpus: not yet)
    const auto tw = std::chrono::steady_clock::now();
    loadp::busy("warming up the kernels (first start after an update: up to a minute)");
    if ((warmed = eng.warm_up_once(L->tok->encode("Hi", false)[0], so.warm_prompt)))
      zlog::eprintf("kernels warmed up for this model in %.1f s (once after an update)\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count());
  }
  if (const char* f = getenv("ARC_FAIL_PASS")) eng.fail_after = atol(f);  // (tests: a pass that throws, see Engine)
  L->srv = std::make_unique<Server>(eng, *L->tok, alias, so.webui);
  Server& srv = *L->srv;
  srv.model_bytes_ = gguf_bytes(model);
  srv.whisper_props_ = so.whisper_props;
  srv.speech_reserve_ = so.whisper_bytes;
  srv.gpus_ = L->tp ? so.gpus : 1;  // (a split that fell back to one GPU says 1)
  std::string mmproj = so.mmproj;
  if (!eng.hp.gemma && mmproj != "-") {
    // Qwen vision (dense and MoE): --mmproj, or the model's own mmproj-*.gguf next to it (model_mmprojs) with a
    // Qwen3-VL projector inferzero runs
    if (mmproj.empty()) mmproj = own_mmproj(model);
    if (!mmproj.empty()) {
      try {
        auto v = std::make_unique<VisionQwen3VL>(mmproj, eng);
        if (v->n_out != eng.hp.n_embd)  // (a projector of another model size)
          throw std::runtime_error("the projector outputs " + std::to_string(v->n_out) + " values per token, the model takes " +
                                   std::to_string(eng.hp.n_embd));
        if (so.img_min > 0) v->min_tokens = so.img_min;
        if (so.img_max > 0) v->max_tokens = so.img_max;
        v->min_tokens = std::min(v->min_tokens, v->max_tokens);
        zlog::eprintf("vision: %s (Qwen3-VL, %d layers, %d..%d tokens per image, encoded on the GPU)\n", mmproj.c_str(),
                v->n_layer, v->min_tokens, v->max_tokens);
        srv.set_qvision(std::move(v));
      } catch (const std::exception& e) {
        zlog::eprintf("vision disabled: %s\n", e.what());
      }
    }
  }
  if (eng.hp.gemma && mmproj != "-") {
    // vision: --mmproj, or the model's own mmproj-*.gguf next to it (model_mmprojs)
    if (mmproj.empty()) mmproj = own_mmproj(model);
    if (!mmproj.empty()) {
      try {
        std::unique_ptr<GemmaImageEncoder> v;
        std::string what;
        if (GgufFile(mmproj).get_str("clip.vision.projector_type", "") == "gemma4v") {
          auto g = std::make_unique<VisionGemma4V>(mmproj, eng);
          what = std::to_string(g->n_layer) + "-layer ViT, 48-pixel tokens, encoded on the GPU";
          v = std::move(g);
        } else {
          auto g = std::make_unique<VisionGemma4>(mmproj);
          what = std::to_string(g->patch) + "-pixel patches";
          v = std::move(g);
        }
        if (v->out_dim() != eng.hp.n_embd)  // (a projector of another model size; checked before its lazy load)
          throw std::runtime_error("the projector outputs " + std::to_string(v->out_dim()) + " values per token, the model takes " +
                                   std::to_string(eng.hp.n_embd));
        if (so.img_min > 0) v->min_tokens = so.img_min;
        if (so.img_max > 0) v->max_tokens = so.img_max;
        v->min_tokens = std::min(v->min_tokens, v->max_tokens);
        zlog::eprintf("vision: %s (%s, %d..%d tokens per image)\n", mmproj.c_str(), what.c_str(), v->min_tokens, v->max_tokens);
        srv.set_vision(std::move(v));
      } catch (const std::exception& e) {
        zlog::eprintf("vision disabled: %s\n", e.what());
      }
      try {
        auto width_ok = [&](const GemmaAudioEncoder& a) {  // (a projector of another model size)
          if (a.out_dim() != eng.hp.n_embd)
            throw std::runtime_error("the projector outputs " + std::to_string(a.out_dim()) + " values per token, the model takes " +
                                     std::to_string(eng.hp.n_embd));
        };
        if (GgufFile(mmproj).get_str("clip.audio.projector_type", "") == "gemma4a") {
          auto a = std::make_unique<AudioGemma4A>(mmproj, eng);
          width_ok(*a);
          zlog::eprintf("audio: %d-layer conformer on the GPU (25 tokens per second of audio)\n", a->n_layer);
          if (warmed) {
            // (first start after an update: one second of silence through the encoder, so that its kernels compile now
            //  and not in a first meeting or voice message (E4B: 89 s of audio encoded in 21.6 s, then 0.5 s for 29 s);
            //  its weights load then too, into the GPU memory kept for it)
            const auto tw = std::chrono::steady_clock::now();
            loadp::busy("warming up the audio encoder (first start after an update)");
            try {  // (a warm-up that fails leaves audio on: the first clip compiles the kernels then, as before)
              int n = 0;
              a->encode(std::vector<float>(16000, 0.0f), n);
              zlog::eprintf("audio encoder warmed up in %.1f s\n",
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count());
            } catch (const std::exception& e) {
              zlog::eprintf("audio encoder warm-up skipped: %s\n", e.what());
            }
          }
          srv.set_audio(std::move(a));
        } else {
          auto a = std::make_unique<AudioGemma4>(mmproj);
          width_ok(*a);
          zlog::eprintf("audio: %d-sample frames at %d Hz (%.0f tokens per second of audio)\n", a->frame, a->rate, (double)a->rate / a->frame);
          srv.set_audio(std::move(a));
        }
      } catch (const std::exception& e) {
        zlog::eprintf("audio disabled: %s\n", e.what());
      }
    }
  }
  srv.default_think_budget_ = so.think_budget;
  srv.default_preserve_ = so.preserve_thinking;
  if (so.preserve_thinking && !srv.can_preserve_)
    zlog::eprintf("--preserve-thinking: this model's chat template has no such option (Qwen3.6 has it; Qwen3.8 always keeps the reasoning)\n");
  // (saved states only load into the same layout: no MTP block (-std) and DFlash2 without the MTP block (-df) each
  // keep their own folders next to the plain one; auto and -nodf share one: DFlash2 saves no state of its own, its
  // context rings start empty after a load — give it a suffix once that changes)
  std::string layout;
  if (!eo.mtp) layout += "-std";
  if (eo.mtp && !eo.dflash.empty() && !eo.mtp_block) layout += "-nomtp";
  if (L->eng->hp.gemma) layout += "-q2";  // (Gemma 4: two-pass states; the one-pass ones of before 2026-10-22 stay unused)
  if (so.prefix_cache && eo.attn_int8) {  // %USERPROFILE%\.inferzero\prefix\<model>-<size>[<layout>]
    std::error_code ec;
    const std::filesystem::path mp(model);
    srv.open_prefix_cache(app_dir() + ARC_DATA "prefix/" + mp.stem().string() + "-" +
                          std::to_string(std::filesystem::file_size(mp, ec)) + layout, so.prefix_gb);
  }
  if (so.sess_gb > 0 && eo.attn_int8) {
    // saved conversations of this model file: %USERPROFILE%\.inferzero\sessions\<model>-<size>
    std::error_code ec;
    const std::filesystem::path mp(model);
    std::string dir = so.sess_dir;
    if (dir.empty()) dir = app_dir() + ARC_DATA "sessions";
    srv.open_sessions(dir + "/" + mp.stem().string() + "-" + std::to_string(std::filesystem::file_size(mp, ec)) + layout,
                      so.sess_gb);
  }
  zlog::eprintf("serving %s as \"%s\"\n", model.c_str(), alias.c_str());
  return L;
}

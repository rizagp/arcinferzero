// The server's speech side around its service (tools/server_whisper.h: Whisper, Qwen3-ASR, the transcript library):
// starting it (--whisper, --asr), Gemma 4 E4B as a transcript model through the chat models, and the OpenAI audio
// endpoints (POST /v1/audio/transcriptions and /translations). Part of tools/server.cpp (included after
// server_whisper.h, server_load.h and server_catalog.h); docs/qwen3-asr.md, docs/whisper-native.md.
#pragma once
#include "server_whisper.h"
#include <chrono>
#include <functional>
#include <memory>
#include <string>

namespace srvsp {

// The chat request that has a model with an audio encoder (Gemma 4 E4B) transcribe or translate a clip (its bytes,
// a file extension): the clip and an instruction in one user turn, no thinking, temperature 0 unless given. A cap of 8
// tokens a second of audio (speech is ~4; max_tokens <= 0) and a mild repeat penalty unless given: greedy, Gemma 4 E4B
// fell into a loop on a 29-s clip of mixed Indonesian and English and wrote 16,687 tokens in 489 s.
inline Json audio_chat_body(const std::string& data, const std::string& ext, const std::string& model,
                            const std::string& language, const std::string& prompt, bool translate, double secs,
                            double temperature = 0.0, double max_tokens = 0, double repeat_penalty = 1.1) {
  std::string instr = translate ? "Translate the speech in this audio into English. Reply with the translation only."
                                : "Transcribe the speech in this audio exactly as spoken" +
                                      (language.empty() ? std::string() : " (language: " + language + ")") +
                                      ". Reply with the transcription only.";
  if (!prompt.empty()) instr += " Context: " + prompt;
  Json audio = Json::object();
  audio.set("data", apic::base64_encode(data));
  audio.set("format", ext);
  Json pa = Json::object();
  pa.set("type", "input_audio");
  pa.set("input_audio", audio);
  Json parts = Json::array();
  parts.push(pa);
  parts.push(apic::text_part(instr));
  Json msgs = Json::array();
  msgs.push(apic::msg("user", parts));
  Json body = Json::object();
  body.set("model", model);
  body.set("messages", msgs);
  body.set("enable_thinking", false);
  body.set("temperature", temperature);
  body.set("max_tokens", max_tokens > 0 ? max_tokens : (double)std::clamp((int)(secs * 8) + 64, 96, 16384));
  body.set("repeat_penalty", repeat_penalty);
  // (the model named hears the clip, whatever model is loaded now: without it the clip went to the speech service,
  //  which the transcript request calling here holds: "resource deadlock would occur")
  body.set("speech", "model");
  return body;
}

// The speech service before the chat model (--ctx auto then leaves its GPU memory free: so.whisper_bytes): Whisper
// (whisper_opt: auto = large-v3, else turbo, from the Hugging Face cache; or a folder) and Qwen3-ASR next to it
// (asr_opt: "default" = auto when the cache has it, auto, a GGUF, off). Null with whisper_opt off; throws what is
// wrong. The NPU's driver is initialized before the GPU runtime (which asks for GPUs only) when Whisper or the model
// uses the NPU.
inline std::unique_ptr<srvwh::Service> start(const std::string& whisper_opt_in, const std::string& whisper_dev,
                                             std::string asr_opt, double max_minutes, ServeOpts& so) {
  // (on / yes / true / 1: auto, as people type it; "--whisper on" was taken for a folder named "on")
  auto onoff = [](std::string v) {
    std::string l = v;
    for (char& c : l) c = (char)tolower((unsigned char)c);
    if (l == "on" || l == "yes" || l == "true" || l == "1" || l == "auto") return std::string("auto");
    if (l == "no" || l == "false" || l == "0" || l == "off") return std::string("off");
    return v;
  };
  const std::string whisper_opt = onoff(whisper_opt_in);
  if (asr_opt != "default" && asr_opt != "auto-quiet") asr_opt = onoff(asr_opt);
  if (asr_opt == "default") asr_opt = whisper_opt == "off" ? "off" : "auto-quiet";  // (not in the cache: Whisper alone)
  // (--whisper off --asr auto / a GGUF: Qwen3-ASR alone, ~3.7 GiB of GPU instead of ~9.8 with Whisper: the service
  //  without Whisper; translations, the live text while recording and word times need Whisper)
  const bool asr_alone = whisper_opt == "off" && asr_opt != "off";
  if (whisper_opt == "off" && !asr_alone) return nullptr;
  std::string dir;
  if (!asr_alone) {
    dir = whisper_opt == "auto" ? whisper::find_model() : whisper_opt;
    if (dir.empty()) throw std::runtime_error("no faster-whisper model in the Hugging Face cache (hf download Systran/faster-whisper-large-v3)");
    if (std::error_code ec; !std::filesystem::exists(std::filesystem::path(dir) / "model.bin", ec))
      throw std::runtime_error("'" + dir + "' is no faster-whisper folder (no model.bin in it): --whisper auto | off | a folder");
  }
  const bool npu_wanted = (!asr_alone && whisper_dev == "npu") || so.eo.npu > 0 || (getenv("ARC_NPU") && atoi(getenv("ARC_NPU")) != 0);
  if (npu_wanted && ze_load()) zeInit(0);
  const auto t0 = std::chrono::steady_clock::now();
  auto wh = std::make_unique<srvwh::Service>(dir, !asr_alone && whisper_dev == "npu", so.eo.kernel_dir);
  wh->max_seconds = max_minutes * 60;
  if (asr_opt != "off") {  // (Qwen3-ASR: its own engine, in the service; before the model as Whisper)
    const bool quiet = asr_opt == "auto-quiet";
    const std::string f = asr_opt == "auto" || quiet ? q3asr::find_model() : asr_opt;
    if (f.empty() && !quiet)
      throw std::runtime_error("--asr auto: no Qwen3-ASR GGUF in the Hugging Face cache (hf download unsloth/Qwen3-ASR-1.7B-GGUF)");
    if (f.empty()) {
      fprintf(stderr, "asr: no Qwen3-ASR GGUF in the Hugging Face cache: Whisper transcribes (docs/qwen3-asr.md)\n");
    } else {
      try {
        wh->load_asr(f, so.eo.kernel_dir, so.asr_slots);
        fprintf(stderr, "asr: %s, %d parts at a time (%s %.2f GiB GPU)\n", wh->asr_id.c_str(), wh->asr_parts(),
                asr_alone ? "alone, no Whisper:" : "Whisper and Qwen3-ASR", wh->gpu_bytes() / 1073741824.0);
      } catch (const std::exception& e) {
        if (!quiet) throw;  // (asked for: the server says why and stops)
        fprintf(stderr, "asr: %s could not be loaded (%s): Whisper transcribes (docs/qwen3-asr.md)\n", f.c_str(), e.what());
      }
    }
  }
  so.whisper_bytes = wh->gpu_bytes() + (128u << 20);  // (+ the word-timing buffers it grows later)
  so.whisper_props = wh->props();
  if (!asr_alone)
    fprintf(stderr, "whisper: %s on the %s in %.1f s (%.2f GiB GPU)\n", wh->id.c_str(), whisper_dev == "npu" ? "NPU + iGPU" : "iGPU",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), wh->gpu_bytes() / 1073741824.0);
  return wh;
}

// Gemma 4 E4B as a transcript model of the service (srvwh::Service::chat_run): a model of the catalog (or the loaded
// one) with "E4B" in its name and its audio encoder; each part of a clip a chat request to it through dispatch, made
// here without a connection (the server loads it when another model is loaded). The service's /props entry is set
// again (so, cur_so, the loaded server's). Nothing when there is no such model.
inline void attach_e4b(srvwh::Service& wh, const std::vector<ModelInfo>& catalog, Loaded* cur,
                       std::function<void(const HttpRequest&, HttpResponse&)>& dispatch, ServeOpts& so, ServeOpts& cur_so) {
  std::string e4b;
  auto is_e4b = [](const std::string& n) { return lower_ascii(n).find("e4b") != std::string::npos; };
  for (const ModelInfo& m : catalog)
    if (e4b.empty() && m.supported && m.audio && m.arch == "gemma4" && (is_e4b(m.name) || is_e4b(m.file))) e4b = m.alias;
  if (e4b.empty() && cur && cur->srv && cur->srv->has_audio() && is_e4b(cur->alias)) e4b = cur->alias;
  if (e4b.empty()) return;
  wh.chat_asr_id = e4b;
  wh.chat_asr_slots = [&so] { return so.eo.slots; };  // (--slots, also for the model loaded later; POST /slots changes it)
  wh.chat_asr =[&dispatch, e4b](const std::string& wav, double secs, const std::string& lang, const std::string& context) {
    HttpRequest rq;
    rq.method = "POST";
    rq.path = "/v1/chat/completions";
    rq.local_peer = true;
    rq.headers["content-type"] = "application/json";
    // (the speech service's own request: a model switch for it is not the user's choice, server.cpp leaves
    //  last-model.txt as it is)
    rq.headers["x-arc-speech"] = "1";
    rq.body = audio_chat_body(wav, "wav", e4b, lang, context, false, secs).dump();
    HttpResponse rs(INVALID_SOCKET);
    int status = 0;
    std::string out;
    rs.on_send = [&](int st, const std::string& b) { status = st; out = b; };
    dispatch(rq, rs);
    Json j;
    Json::try_parse(out, j);
    if (status != 200) throw std::runtime_error("Gemma 4 E4B: " + j["error"]["message"].str("HTTP " + std::to_string(status)));
    return j["choices"][0]["message"]["content"].str();
  };
  so.whisper_props = wh.props();
  cur_so.whisper_props = so.whisper_props;
  if (cur && cur->srv) cur->srv->whisper_props_ = so.whisper_props;
}

// What the audio endpoints need of the server around them.
struct AudioRoute {
  srvwh::Service* wh = nullptr;                                     // the speech service (null: --whisper off)
  std::function<bool(const std::string&)> is_chat_model;            // a model name of the catalog or the loaded one
  std::function<std::string()> loaded_alias;                        // "" when no model is loaded
  std::function<void(HttpResponse&)> no_model;                      // the answer then
  std::function<void(const HttpRequest&, HttpResponse&)>* dispatch = nullptr;
};
// POST /v1/audio/transcriptions and /translations (multipart upload): the speech service (Qwen3-ASR, Gemma 4 E4B,
// Whisper itself; translations Whisper's) unless the request names another chat model (400 then); without the
// service, the clip goes to a chat model with an audio encoder (the speech service's models named: 400). False when
// the request is not one of them.
inline bool audio_route(const HttpRequest& rq, HttpResponse& rs, const AudioRoute& a) {
  if (!(rq.method == "POST" && (rq.path == "/v1/audio/transcriptions" || rq.path == "/v1/audio/translations"))) return false;
  const auto form = apic::parse_multipart(rq.header("content-type"), rq.body);
  const auto fit = form.find("file");
  if (fit == form.end() || fit->second.data.empty()) {
    rs.send(400, "application/json", "{\"error\":{\"message\":\"file required (multipart/form-data)\"}}");
    return true;
  }
  auto field = [&](const char* k) { auto it = form.find(k); return it == form.end() ? std::string() : it->second.data; };
  const bool translate = rq.path == "/v1/audio/translations";
  const std::string want = apic::strip_tag(field("model"));
  const bool chat_model = !want.empty() && a.is_chat_model(want);
  if (a.wh && (!chat_model || a.wh->names_chat_asr(want))) {
    a.wh->handle(rs, form, translate);
    return true;
  }
  // (with the speech service only its models transcribe, and Whisper translates: Gemma 4 12B, which hears audio
  //  without an encoder, wrote a translated summary of the user's recording and garbled a clear one; the user's
  //  choice, 2026-10-28)
  if (a.wh) {
    rs.send(400, "application/json",
            std::string("{\"error\":{\"message\":\"model: ") +
                (translate ? "translations take Whisper (no model, whisper-1 or whisper)"
                           : "transcription takes qwen3-asr (the default), whisper or Gemma 4 E4B (" + a.wh->chat_asr_id + ")") +
                "\",\"type\":\"invalid_request_error\"}}");
    return true;
  }
  std::string lw = want;
  for (char& c : lw) c = (char)tolower((unsigned char)c);
  if (lw.find("whisper") != std::string::npos || lw.find("qwen3-asr") != std::string::npos) {
    rs.send(400, "application/json", "{\"error\":{\"message\":\"" + std::string(lw.find("whisper") != std::string::npos ? "Whisper" : "Qwen3-ASR") +
                                         " is off: start the server with --whisper auto\",\"type\":\"invalid_request_error\"}}");
    return true;
  }
  Json req = Json::object();
  req.set("model", want);
  req.set("response_format", field("response_format").empty() ? "json" : field("response_format"));
  req.set("language", field("language"));
  req.set("task", translate ? "translate" : "transcribe");
  std::string model = want;
  if (model.empty()) model = a.loaded_alias();
  if (model.empty()) {
    a.no_model(rs);
    return true;
  }
  double secs = 30;
  try {
    secs = (double)audio::decode_16k(std::vector<uint8_t>(fit->second.data.begin(), fit->second.data.end()), 3600).size() / 16000;
  } catch (const std::exception&) {}  // (the chat request says what is wrong with the audio)
  const Json chat_body = audio_chat_body(
      fit->second.data, fit->second.filename.substr(fit->second.filename.rfind('.') + 1), model, field("language"),
      field("prompt"), translate, secs, field("temperature").empty() ? 0.0 : atof(field("temperature").c_str()),
      field("max_tokens").empty() ? 0.0 : atof(field("max_tokens").c_str()),
      field("repeat_penalty").empty() ? 1.1 : atof(field("repeat_penalty").c_str()));
  HttpRequest rq2 = rq;
  rq2.path = "/v1/chat/completions";
  rq2.body = chat_body.dump();
  apic::Adapter ad("transcription", req, rs);
  (*a.dispatch)(rq2, rs);
  return true;
}

}  // namespace srvsp

// arcinferzero: the Gemma 4 vision encoder (gemma4v) on one image, for tools/gemma4v_ref.py (the f32 reference):
// writes <image>.g4v.bin (n_tok x n_out f32 embeddings) and <image>.g4v.rgb (the resized pixels).
// ARC_VIS_LM=file: then the model itself on "<image> Describe this image in detail." (Gemma 4 chat turns): 32 greedy
// steps, their log-probs (f32, vocab per step) to file and the chosen tokens to file.tok; ARC_VIS_LM_FORCE=file.tok
// takes those tokens instead of its own (teacher forcing): the KL between two runs' files (an encoder variant against
// the f32 one) is the encoder's effect on what the model says. ARC_VIS_EMB=file: the embeddings from that file (as
// written here or by tools/gemma4v_ref REF_VOUT) instead of the encoder's, for the model run.
// A Qwen3-VL mmproj (Qwen3.5 / 3.6 / 3.8): the same timing (ARC_PROF) and ARC_VIS_LM / ARC_VIS_LM_FORCE model run, the
// image placed as the server places it (causal rows, M-RoPE positions, the rotary shift after it); no .g4v files.
// An audio clip (.wav / .mp3 ...) with a Gemma 4 mmproj that hears (gemma4ua: 12B, gemma4a: E2B / E4B): the clip
// encoded as the server encodes it, then always the model run of ARC_VIS_LM (default <clip>.g4a.lm) on
// "<audio> Transcribe ..." with the rows in causal pieces of max_chunk, as the server feeds them: 48 greedy steps, or
// with ARC_AUDIO_REPLY=text that reply's tokens forced (the text printed; log-probs per step to the file). For
// tools/ref_forward (REF_EMB, REF_ALL): <clip>.g4a.bin (header n_tok, n_out, the first row's position; f32 rows) and
// <clip>.g4a.ids (the prompt with 0 at the audio rows, then the steps' tokens but the last). ARC_AUDIO_PREP=1 (gemma4ua,
// whose encoder runs on the CPU): only those two files, with ARC_AUDIO_REPLY, no GPU (the reference can run first).
// usage: vision_testz model.gguf mmproj.gguf image.(png|jpg) [max_tokens]
//        vision_testz model.gguf mmproj.gguf clip.(wav|mp3|...)
#include "engine.h"
#include "audio_gemma4a.h"
#include "vision_gemma4v.h"
#include "vision_qwen3vl.h"
#include "../src/audio.h"
#include "../src/tokenizer.h"
#include <cmath>
#include <cstdio>

// 32 greedy (or forced) steps from logits l, their log-probs to lm and the tokens to lm.tok; the text printed
static void lm_steps(Engine& eng, const Tokenizer& tok, std::vector<float> l, const char* lm) {
  std::vector<int> forced;
  if (const char* ff = getenv("ARC_VIS_LM_FORCE")) {
    FILE* f = fopen(ff, "rb");
    if (!f) throw std::runtime_error("cannot open ARC_VIS_LM_FORCE");
    for (int t; fread(&t, 4, 1, f) == 1;) forced.push_back(t);
    fclose(f);
  }
  FILE* fl = fopen(lm, "wb");
  FILE* ft = fopen((std::string(lm) + ".tok").c_str(), "wb");
  std::string text;
  for (int step = 0; step < 32; ++step) {
    const double mx = *std::max_element(l.begin(), l.end());
    double z = 0;
    for (float v : l) z += std::exp((double)v - mx);
    std::vector<float> lp(l.size());
    for (size_t i = 0; i < l.size(); ++i) lp[i] = (float)(l[i] - mx - std::log(z));
    fwrite(lp.data(), 4, lp.size(), fl);
    const int t = step < (int)forced.size() ? forced[step] : (int)(std::max_element(l.begin(), l.end()) - l.begin());
    fwrite(&t, 4, 1, ft);
    text += tok.piece(t);
    l = eng.forward(&t, 1);
  }
  fclose(fl);
  fclose(ft);
  printf("model: %s\n", text.c_str());
}

static int qwen_main(Engine& eng, const char* mmproj, const ImageRGB& im) {
  VisionQwen3VL vis(mmproj, eng);
  int n_tok = 0, gw = 0, gh = 0;
  std::vector<float> emb = vis.encode(im, n_tok, gw, gh);
  eng.profile = getenv("ARC_PROF") != nullptr;
  emb = vis.encode(im, n_tok, gw, gh);  // (the second run: kernels built)
  printf("image %dx%d -> %d tokens (%d x %d), GPU encoder %.1f ms\n", im.w, im.h, n_tok, gw, gh, vis.encode_ms);
  if (eng.profile) {
    std::vector<std::pair<double, std::string>> rows;
    double total = 0;
    for (auto& [n, ms] : eng.prof_ms) { total += ms; rows.push_back({ms, n}); }
    std::sort(rows.rbegin(), rows.rend());
    for (auto& [ms, n] : rows) printf("  %-14s %9.1f ms  %5.1f%%  (%d launches)\n", n.c_str(), ms, 100 * ms / total, eng.prof_n[n]);
  }
  eng.profile = false;
  if (const char* lm = getenv("ARC_VIS_LM")) {
    Tokenizer tok(eng.gguf());
    eng.reset();
    eng.rope_shift = 0;
    const std::vector<int> pre = tok.encode("<|im_start|>user\n<|vision_start|>", true);
    const std::vector<int> post =
        tok.encode("<|vision_end|>Describe this image in detail.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", true);
    eng.forward(pre.data(), (int)pre.size());
    const int t = (int)pre.size();  // (t, t + row, t + column), as the server writes them
    std::vector<int> rp((size_t)n_tok * 3);
    for (int i = 0; i < n_tok; ++i) {
      rp[3 * i] = t;
      rp[3 * i + 1] = t + i / gw;
      rp[3 * i + 2] = t + i % gw;
    }
    eng.forward_embd(emb.data(), n_tok, false, false, rp.data());
    eng.rope_shift = std::max(gw, gh) - n_tok;
    lm_steps(eng, tok, eng.forward(post.data(), (int)post.size()), lm);
  }
  return 0;
}

// The clip's rows and the prompt around them; writes <clip>.g4a.bin and, once the steps' tokens are known, .g4a.ids.
struct AudioCase {
  std::vector<float> emb;
  int n_tok = 0, n_out = 0;
  std::vector<int> pre, post, forced;
  AudioCase(GemmaAudioEncoder& enc, const Tokenizer& tok, const std::string& clip) {
    const std::vector<float> s16 = audio::decode_16k_file(clip);
    if (const char* wv = getenv("ARC_AUDIO_WAV")) {  // (the decoded clip as 16-bit 16 kHz WAV: for llama.cpp, which reads no AAC)
      FILE* fw = fopen(wv, "wb");
      const uint32_t n = (uint32_t)s16.size() * 2, rate = 16000, br = 32000, chunk = 36 + n, fmt_len = 16;
      const uint16_t pcm = 1, ch = 1, align = 2, bits = 16;
      fwrite("RIFF", 1, 4, fw); fwrite(&chunk, 4, 1, fw); fwrite("WAVEfmt ", 1, 8, fw); fwrite(&fmt_len, 4, 1, fw);
      fwrite(&pcm, 2, 1, fw); fwrite(&ch, 2, 1, fw); fwrite(&rate, 4, 1, fw); fwrite(&br, 4, 1, fw);
      fwrite(&align, 2, 1, fw); fwrite(&bits, 2, 1, fw); fwrite("data", 1, 4, fw); fwrite(&n, 4, 1, fw);
      for (float v : s16) {
        const int16_t q = (int16_t)std::lround(std::clamp(v, -1.0f, 1.0f) * 32767.0f);
        fwrite(&q, 2, 1, fw);
      }
      fclose(fw);
    }
    emb = enc.encode(s16, n_tok);
    n_out = enc.out_dim();
    printf("audio %.1f s -> %d tokens\n", s16.size() / 16000.0, n_tok);
    pre = tok.encode("<bos><|turn>user\n<|audio>", true);
    post = tok.encode(
        "<audio|>Transcribe the speech in this audio exactly as spoken. Reply with the transcription only.<turn|>\n"
        "<|turn>model\n", true);
    if (const char* r = getenv("ARC_AUDIO_REPLY")) forced = tok.encode(r, false);
    FILE* fo = fopen((clip + ".g4a.bin").c_str(), "wb");
    int hdr[3] = {n_tok, n_out, (int)pre.size()};
    fwrite(hdr, 4, 3, fo);
    fwrite(emb.data(), 4, emb.size(), fo);
    fclose(fo);
  }
  void write_ids(const std::string& clip, const std::vector<int>& steps) const {
    std::vector<int> ids = pre;
    ids.insert(ids.end(), n_tok, 0);
    ids.insert(ids.end(), post.begin(), post.end());
    printf("prompt %zu tokens (REF_ALL_FROM=%zu), %zu steps\n", ids.size(), ids.size() - 1, steps.size());
    if (!steps.empty()) ids.insert(ids.end(), steps.begin(), steps.end() - 1);
    FILE* fo = fopen((clip + ".g4a.ids").c_str(), "wb");
    for (size_t i = 0; i < ids.size(); ++i) fprintf(fo, i ? ",%d" : "%d", ids[i]);
    fclose(fo);
  }
};

static int audio_main(Engine& eng, const char* mmproj, const std::string& clip) {
  std::unique_ptr<GemmaAudioEncoder> enc;
  if (GgufFile(mmproj).get_str("clip.audio.projector_type", "") == "gemma4a") enc = std::make_unique<AudioGemma4A>(mmproj, eng);
  else enc = std::make_unique<AudioGemma4>(mmproj);
  if (enc->out_dim() != eng.hp.n_embd) throw std::runtime_error("the projector does not fit this model");
  Tokenizer tok(eng.gguf());
  const AudioCase c(*enc, tok, clip);
  eng.reset();
  eng.forward(c.pre.data(), (int)c.pre.size());
  for (int o = 0; o < c.n_tok;) {  // (causal pieces of max_chunk rows, as the server feeds audio)
    const int k = std::min(eng.max_chunk, c.n_tok - o);
    eng.forward_embd(c.emb.data() + (size_t)o * c.n_out, k, false, false);
    o += k;
  }
  std::vector<float> l = eng.forward(c.post.data(), (int)c.post.size());
  const char* lm = getenv("ARC_VIS_LM");
  FILE* fl = fopen(lm ? lm : (clip + ".g4a.lm").c_str(), "wb");
  std::vector<int> steps;
  std::string text;
  const int n_steps = c.forced.empty() ? 48 : (int)c.forced.size();
  for (int step = 0; step < n_steps; ++step) {
    const double mx = *std::max_element(l.begin(), l.end());
    double z = 0;
    for (float v : l) z += std::exp((double)v - mx);
    std::vector<float> lp(l.size());
    for (size_t i = 0; i < l.size(); ++i) lp[i] = (float)(l[i] - mx - std::log(z));
    fwrite(lp.data(), 4, lp.size(), fl);
    const int t = c.forced.empty() ? (int)(std::max_element(l.begin(), l.end()) - l.begin()) : c.forced[step];
    steps.push_back(t);
    text += tok.piece((int)(std::max_element(l.begin(), l.end()) - l.begin()));  // (its own choice, also when forced)
    l = eng.forward(&t, 1);
  }
  fclose(fl);
  printf("model: %s\n", text.c_str());
  c.write_ids(clip, steps);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: vision_testz model.gguf mmproj.gguf image.(png|jpg) [max_tokens]\n");
    return 1;
  }
  try {
    EngineOptions eo;
    eo.ctx = 4096;
    eo.min_chunk = 2048;  // (an image in one chunk, as the server sets for Gemma with an image projector)
    std::string ext = std::filesystem::path(argv[3]).extension().string();
    for (char& c : ext) c = (char)tolower((unsigned char)c);
    const bool audio = ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".webp" && ext != ".bmp" && ext != ".gif";
    if (audio && getenv("ARC_AUDIO_PREP")) {  // (gemma4ua on the CPU: the reference's files without the GPU)
      AudioGemma4 enc(argv[2]);
      const GgufFile mf(argv[1]);
      const Tokenizer tok(mf);
      const AudioCase c(enc, tok, argv[3]);
      c.write_ids(argv[3], c.forced);
      return 0;
    }
    Engine eng(argv[1], eo);
    if (audio) return audio_main(eng, argv[2], argv[3]);
    if (VisionQwen3VL::usable(GgufFile(argv[2]))) {
      std::vector<uint8_t> b;
      FILE* f = fopen(argv[3], "rb");
      if (!f) throw std::runtime_error("cannot open image");
      fseek(f, 0, SEEK_END);
      b.resize(ftell(f));
      fseek(f, 0, SEEK_SET);
      if (fread(b.data(), 1, b.size(), f) != b.size()) throw std::runtime_error("cannot read image");
      fclose(f);
      return qwen_main(eng, argv[2], img::decode(b));
    }
    VisionGemma4V vis(argv[2], eng);
    if (argc > 4) vis.min_tokens = vis.max_tokens = atoi(argv[4]);
    std::vector<uint8_t> bytes;
    {
      FILE* f = fopen(argv[3], "rb");
      if (!f) throw std::runtime_error("cannot open image");
      fseek(f, 0, SEEK_END);
      bytes.resize(ftell(f));
      fseek(f, 0, SEEK_SET);
      if (fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) throw std::runtime_error("cannot read image");
      fclose(f);
    }
    const ImageRGB im = img::decode(bytes);
    int n_tok, W, H;
    vis.plan(im.w, im.h, W, H);
    std::vector<float> gpu = vis.encode(im, n_tok);
    eng.profile = getenv("ARC_PROF") != nullptr;  // (ARC_PROF=1: GPU time per kernel of the second run)
    gpu = vis.encode(im, n_tok);  // (the second run: kernels built)
    printf("image %dx%d -> %dx%d, %d tokens, GPU encoder %.1f ms\n", im.w, im.h, W, H, n_tok, vis.encode_ms);
    if (eng.profile) {
      double total = 0;
      std::vector<std::pair<double, std::string>> rows;
      for (auto& [n, ms] : eng.prof_ms) { total += ms; rows.push_back({ms, n}); }
      std::sort(rows.rbegin(), rows.rend());
      for (auto& [ms, n] : rows) printf("  %-14s %9.1f ms  %5.1f%%  (%d launches)\n", n.c_str(), ms, 100 * ms / total, eng.prof_n[n]);
    }
    const ImageRGB r = img::resize_bicubic(im, W, H);
    FILE* fo = fopen((std::string(argv[3]) + ".g4v.bin").c_str(), "wb");
    int hdr[3] = {n_tok, vis.n_out, 0};
    fwrite(hdr, 4, 3, fo);
    fwrite(gpu.data(), 4, gpu.size(), fo);
    fclose(fo);
    fo = fopen((std::string(argv[3]) + ".g4v.rgb").c_str(), "wb");
    int dim[2] = {W, H};
    fwrite(dim, 4, 2, fo);
    fwrite(r.px.data(), 1, r.px.size(), fo);
    fclose(fo);
    if (const char* lm = getenv("ARC_VIS_LM")) {
      Tokenizer tok(eng.gguf());
      std::vector<int> forced;
      if (const char* ff = getenv("ARC_VIS_LM_FORCE")) {
        FILE* f = fopen(ff, "rb");
        if (!f) throw std::runtime_error("cannot open ARC_VIS_LM_FORCE");
        for (int t; fread(&t, 4, 1, f) == 1;) forced.push_back(t);
        fclose(f);
      }
      // (ARC_VIS_EMB: embeddings from a file for the model run, e.g. tools/gemma4v_ref REF_VOUT; the .g4v.bin above
      //  keeps the encoder's own)
      if (const char* ef = getenv("ARC_VIS_EMB")) {
        FILE* f = fopen(ef, "rb");
        int hd[3] = {0, 0, 0};
        if (!f || fread(hd, 4, 3, f) != 3 || hd[0] != n_tok || hd[1] != vis.n_out ||
            fread(gpu.data(), 4, gpu.size(), f) != gpu.size())
          throw std::runtime_error("ARC_VIS_EMB: not this image's embeddings");
        fclose(f);
      }
      eng.reset();
      const std::vector<int> pre = tok.encode("<bos><|turn>user\n<|image>", true);
      const std::vector<int> post = tok.encode("<image|>Describe this image in detail.<turn|>\n<|turn>model\n", true);
      eng.forward(pre.data(), (int)pre.size());
      eng.forward_embd(gpu.data(), n_tok, true, false);
      std::vector<float> l = eng.forward(post.data(), (int)post.size());
      FILE* fl = fopen(lm, "wb");
      FILE* ft = fopen((std::string(lm) + ".tok").c_str(), "wb");
      std::string text;
      for (int step = 0; step < 32; ++step) {
        const double mx = *std::max_element(l.begin(), l.end());
        double z = 0;
        for (float v : l) z += std::exp((double)v - mx);
        std::vector<float> lp(l.size());
        for (size_t i = 0; i < l.size(); ++i) lp[i] = (float)(l[i] - mx - std::log(z));
        fwrite(lp.data(), 4, lp.size(), fl);
        const int t = step < (int)forced.size() ? forced[step] : (int)(std::max_element(l.begin(), l.end()) - l.begin());
        fwrite(&t, 4, 1, ft);
        text += tok.piece(t);
        l = eng.forward(&t, 1);
      }
      fclose(fl);
      fclose(ft);
      printf("model: %s\n", text.c_str());
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}

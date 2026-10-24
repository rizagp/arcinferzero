// arcinferzero: the Gemma 4 vision encoder (gemma4v) on one image, for tools/gemma4v_ref.py (the f32 reference):
// writes <image>.g4v.bin (n_tok x n_out f32 embeddings) and <image>.g4v.rgb (the resized pixels).
// ARC_VIS_LM=file: then the model itself on "<image> Describe this image in detail." (Gemma 4 chat turns): 32 greedy
// steps, their log-probs (f32, vocab per step) to file and the chosen tokens to file.tok; ARC_VIS_LM_FORCE=file.tok
// takes those tokens instead of its own (teacher forcing): the KL between two runs' files (an encoder variant against
// the f32 one) is the encoder's effect on what the model says. ARC_VIS_EMB=file: the embeddings from that file (as
// written here or by tools/gemma4v_ref REF_VOUT) instead of the encoder's, for the model run.
// usage: vision_testz model.gguf mmproj.gguf image.(png|jpg) [max_tokens]
#include "engine.h"
#include "vision_gemma4v.h"
#include "../src/tokenizer.h"
#include <cmath>
#include <cstdio>

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: vision_testz model.gguf mmproj.gguf image.(png|jpg) [max_tokens]\n");
    return 1;
  }
  try {
    EngineOptions eo;
    eo.ctx = 4096;
    eo.min_chunk = 2048;  // (an image in one chunk, as the server sets for Gemma with an image projector)
    Engine eng(argv[1], eo);
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

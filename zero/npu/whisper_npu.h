// Whisper's encoder on the NPU (Intel AI Boost) through Level Zero's graph extension (zero/npu/npu_rt.h; no
// OpenVINO): `transcribez --device npu`. Three graphs, written as OpenVINO IR by zero/npu/npu_ir.h, compiled once
// (blob cache) and run per 30-s window:
//   front  mel [1][128][3000] -> conv1 + GELU -> conv2 (stride 2) + GELU -> + positions -> x [1500][1280]
//          (its weights constants of the graph);
//   layer  one encoder layer (LayerNorm, q / k / v, attention over the 1500 positions, output, LayerNorm, MLP); all
//          32 layers have its shapes, so one graph serves every layer: the weights are its inputs, int8 with one f16
//          scale per 32 values (dequantized in the graph, the "w8g" form of zero/npu/engine_npu.h), q pre-scaled by
//          1/8;
//   tail   the final LayerNorm.
// Activations are f16 (the NPU's arithmetic); LayerNorm is the compiler's MVN (its own precision path: Whisper's
// residual stream reaches a few hundred, whose squares f16 cannot sum). The decoder stays on the iGPU
// (zero/whisper_engine.h takes the encoder output and computes the cross-attention keys / values).
#pragma once
#include "../../src/whisper.h"
#include "npu_ir.h"
#include "npu_rt.h"
#include <chrono>
#include <memory>

class WhisperNpuEncoder {
 public:
  // (Level Zero must have been initialized for every driver type first: zeInit(0) before the GPU runtime)
  explicit WhisperNpuEncoder(const whisper::Model& m) : m_(m), hp_(m.hp) {
    rt_ = std::make_unique<npu::Rt>();
    if (!rt_->init()) throw std::runtime_error("whisper: no NPU (Level Zero graph extension) found: " + rt_->why);
    const auto t0 = std::chrono::steady_clock::now();
    const int D = hp_.d, A = hp_.n_audio_ctx, C = hp_.n_mels, T = whisper::kWindowFrames;
    F_ = m.file->get("encoder/layer_0/ffn/linear_0/weight").shape[0];
    mel_ = (uint16_t*)host((size_t)C * T * 2);
    xa_ = (uint16_t*)host((size_t)A * D * 2);
    xb_ = (uint16_t*)host((size_t)A * D * 2);
    out_ = (uint16_t*)host((size_t)A * D * 2);
    // the layers' weights (int8 + scales) and small parameters in NPU-visible host memory
    for (int l = 0; l < hp_.enc_layers; ++l) {
      const std::string p = "encoder/layer_" + std::to_string(l) + "/";
      Layer L;
      L.ln1g = copy_f16(p + "self_attention/layer_norm/gamma");
      L.ln1b = copy_f16(p + "self_attention/layer_norm/beta");
      const whisper::Ct2Var& qkv = m.file->get(p + "self_attention/linear_0/weight");
      const whisper::Ct2Var& qkvb = m.file->get(p + "self_attention/linear_0/bias");
      for (int t = 0; t < 3; ++t) {  // q (scaled by 1/8), k, v: rows t*D .. of the fused matrix
        L.w[t] = w8g((const uint16_t*)qkv.data + (size_t)t * D * D, D, D, t == 0 ? 0.125f : 1.0f);
        L.b[t] = slice_f16((const uint16_t*)qkvb.data + (size_t)t * D, D, t == 0 ? 0.125f : 1.0f);
      }
      L.w[3] = w8g_named(p + "self_attention/linear_1/weight");
      L.b[3] = copy_f16(p + "self_attention/linear_1/bias");
      L.ln2g = copy_f16(p + "ffn/layer_norm/gamma");
      L.ln2b = copy_f16(p + "ffn/layer_norm/beta");
      L.w[4] = w8g_named(p + "ffn/linear_0/weight");
      L.b[4] = copy_f16(p + "ffn/linear_0/bias");
      L.w[5] = w8g_named(p + "ffn/linear_1/weight");
      L.b[5] = copy_f16(p + "ffn/linear_1/bias");
      layers_.push_back(L);
    }
    lng_ = copy_f16("encoder/layer_norm/gamma");
    lnb_ = copy_f16("encoder/layer_norm/beta");
    front_ = build_front(C, T, D, A);
    layer_ = build_layer(D, A);
    tail_ = build_tail(D, A);
    load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  double load_ms = 0, compile_ms = 0;
  size_t host_bytes = 0;
  bool all_cached = true;

  // the encoder on one window: mel [n_mels][3000] f32 -> the output after the final LayerNorm, f16 [1500][1280]
  const uint16_t* encode(const float* mel) {
    const size_t n = (size_t)hp_.n_mels * whisper::kWindowFrames;
    for (size_t i = 0; i < n; ++i) mel_[i] = npu::f2h(mel[i]);
    run(front_);
    uint16_t* x = xa_;
    uint16_t* y = xb_;
    for (const Layer& L : layers_) {
      bind_layer(layer_, L, x, y);
      run(layer_);
      std::swap(x, y);
    }
    if (!rt_->set(tail_, "x", x)) throw std::runtime_error("whisper: NPU tail input");
    run(tail_);
    return out_;
  }

 private:
  struct W8 {
    int8_t* q = nullptr;
    uint16_t* s = nullptr;
    int N = 0, K = 0;
  };
  struct Layer {
    uint16_t *ln1g, *ln1b, *ln2g, *ln2b;
    W8 w[6];         // q, k, v, out, fc1, fc2
    uint16_t* b[6];  // their biases
  };
  const whisper::Model& m_;
  whisper::Hparams hp_;
  int F_ = 0;
  std::unique_ptr<npu::Rt> rt_;
  std::vector<Layer> layers_;
  uint16_t *mel_, *xa_, *xb_, *out_, *lng_, *lnb_;
  ze_graph_handle_t front_ = nullptr, layer_ = nullptr, tail_ = nullptr;
  std::vector<int> layer_args_;  // (argument indices of the layer graph, in bind_layer's order)

  void* host(size_t bytes) {
    void* p = rt_->host(bytes);
    if (!p) throw std::runtime_error("whisper: out of NPU host memory");
    host_bytes += bytes;
    return p;
  }
  uint16_t* copy_f16(const std::string& name) {
    const whisper::Ct2Var& v = m_.file->get(name);
    if (v.dtype != 4) throw std::runtime_error("whisper: " + name + " is not f16");
    uint16_t* p = (uint16_t*)host(v.bytes);
    memcpy(p, v.data, v.bytes);
    return p;
  }
  uint16_t* slice_f16(const uint16_t* src, int n, float scale) {
    uint16_t* p = (uint16_t*)host((size_t)n * 2);
    for (int i = 0; i < n; ++i) p[i] = scale == 1.0f ? src[i] : npu::f2h(whisper::h2f(src[i]) * scale);
    return p;
  }
  // an f16 matrix [N][K] as int8 rows with one f16 scale per 32 values (the scale rounded to f16 first, then the
  // values quantized with it: what the graph multiplies back is exactly that scale)
  W8 w8g(const uint16_t* src, int N, int K, float scale) {
    W8 w;
    w.N = N;
    w.K = K;
    w.q = (int8_t*)host((size_t)N * K);
    w.s = (uint16_t*)host((size_t)N * (K / 32) * 2);
#pragma omp parallel for
    for (int n = 0; n < N; ++n)
      for (int b = 0; b < K / 32; ++b) {
        float v[32], amax = 0;
        for (int i = 0; i < 32; ++i) {
          v[i] = whisper::h2f(src[(size_t)n * K + b * 32 + i]) * scale;
          amax = std::max(amax, std::fabs(v[i]));
        }
        const uint16_t sh = npu::f2h(std::max(amax / 127.0f, amax > 0 ? 6.103515625e-05f : 0.0f));
        const float sf = npu::h2f(sh), inv = sf > 0 ? 1.0f / sf : 0.0f;
        w.s[(size_t)n * (K / 32) + b] = sh;
        for (int i = 0; i < 32; ++i) w.q[(size_t)n * K + b * 32 + i] = (int8_t)std::clamp((int)std::lrint(v[i] * inv), -127, 127);
      }
    return w;
  }
  W8 w8g_named(const std::string& name) {
    const whisper::Ct2Var& v = m_.file->get(name);
    if (v.dtype != 4) throw std::runtime_error("whisper: " + name + " is not f16");
    return w8g((const uint16_t*)v.data, v.shape[0], (int)(v.count() / v.shape[0]), 1.0f);
  }

  ze_graph_handle_t compile(const npu::Ir& ir, const char* what) {
    double ms = 0;
    bool cached = false;
    ze_graph_handle_t gr = rt_->compile(ir, ms, cached);
    if (!gr) throw std::runtime_error(std::string("whisper: the NPU did not compile the ") + what + " graph");
    compile_ms += ms;
    all_cached = all_cached && cached;
    fprintf(stderr, "whisper: NPU %s graph %s in %.0f ms\n", what, cached ? "loaded" : "compiled", ms);
    return gr;
  }
  // x [M][K] . dq(B)^T + b: B int8 [N][K] with scales [N][K/32][1], b f16 [N]
  static npu::Ir::Out linear(npu::Ir& ir, const std::string& nm, const npu::Ir::Out& x, int N, int K) {
    const auto qb = ir.param(nm + "_q", {"i8", {N, K}});
    const auto sb = ir.param(nm + "_s", {"f16", {N, K / 32, 1}});
    const auto bias = ir.param(nm + "_b", {"f16", {N}});
    const auto b3 = ir.convert(nm + "_qf", ir.reshape(nm + "_q3", qb, {N, K / 32, 32}), "f16");
    const auto W = ir.reshape(nm + "_dq", ir.multiply(nm + "_qs", b3, sb), {N, K});
    return ir.add_op(nm + "_plus_b", ir.matmul_tb(nm + "_mm", x, W), bias);
  }
  static npu::Ir::Out layer_norm(npu::Ir& ir, const std::string& nm, const npu::Ir::Out& x, int D) {
    const auto g = ir.param(nm + "_g", {"f16", {D}});
    const auto b = ir.param(nm + "_bt", {"f16", {D}});
    return ir.add_op(nm + "_plus", ir.multiply(nm + "_mul", ir.mvn_last(nm + "_mvn", x, 1e-5f), g), b);
  }
  ze_graph_handle_t build_front(int C, int T, int D, int A) {
    npu::Ir ir;
    const auto mel = ir.param("mel", {"f16", {1, C, T}});
    const whisper::Ct2Var& w1 = m_.file->get("encoder/conv1/weight");
    const whisper::Ct2Var& w2 = m_.file->get("encoder/conv2/weight");
    const auto c1w = ir.const_f16_bits("c1w", {D, C, 3}, (const uint16_t*)w1.data, w1.count());
    const auto c1b = ir.const_f16_bits("c1b", {1, D, 1}, (const uint16_t*)m_.file->get("encoder/conv1/bias").data, D);
    const auto c2w = ir.const_f16_bits("c2w", {D, D, 3}, (const uint16_t*)w2.data, w2.count());
    const auto c2b = ir.const_f16_bits("c2b", {1, D, 1}, (const uint16_t*)m_.file->get("encoder/conv2/bias").data, D);
    const auto pos = ir.const_f16_bits("pos", {A, D}, (const uint16_t*)m_.file->get("encoder/position_encodings/encodings").data,
                                       (size_t)A * D);
    auto h = ir.gelu_erf("g1", ir.add_op("c1", ir.conv1d("conv1", mel, c1w, 1, 1), c1b));
    h = ir.gelu_erf("g2", ir.add_op("c2", ir.conv1d("conv2", h, c2w, 2, 1), c2b));
    const auto t = ir.reshape("x2d", ir.transpose("xt", h, {0, 2, 1}), {A, D});
    ir.result(ir.add_op("x_pos", t, pos), "x_out");
    ze_graph_handle_t gr = compile(ir, "front");
    const int o = rt_->output_named(gr, "x");
    if (o < 0 || rt_->g->pfnSetArgumentValue(gr, (uint32_t)o, xa_) != 0 || !rt_->set(gr, "mel", mel_) || !rt_->initialize(gr))
      throw std::runtime_error("whisper: NPU front graph arguments");
    return gr;
  }
  ze_graph_handle_t build_layer(int D, int A) {
    npu::Ir ir;
    const int H = hp_.n_head, hd = D / H, F = F_;
    const auto x = ir.param("x", {"f16", {A, D}});
    const auto h = layer_norm(ir, "ln1", x, D);
    auto heads = [&](const npu::Ir::Out& a, const std::string& nm) {  // [A][D] -> [H][A][hd]
      return ir.transpose(nm + "_t", ir.reshape(nm + "_r", a, {A, H, hd}), {1, 0, 2});
    };
    const auto q = heads(linear(ir, "wq", h, D, D), "q");
    const auto k = heads(linear(ir, "wk", h, D, D), "k");
    const auto v = heads(linear(ir, "wv", h, D, D), "v");
    const auto p = ir.softmax("sm", ir.matmul("qk", q, k, true), 2);  // [H][A][A] (q pre-scaled by 1/8)
    const auto o = ir.reshape("o_r", ir.transpose("o_t", ir.matmul("pv", p, v, false), {1, 0, 2}), {A, D});
    const auto x1 = ir.add_op("res1", x, linear(ir, "wo", o, D, D));
    const auto h2 = layer_norm(ir, "ln2", x1, D);
    const auto f = ir.gelu_erf("gelu", linear(ir, "w1", h2, F, D));
    ir.result(ir.add_op("y_res", x1, linear(ir, "w2", f, D, F)), "y_out");
    ze_graph_handle_t gr = compile(ir, "layer");
    // the arguments' indices, in bind_layer's order (looked up once)
    const char* names[] = {"x", "ln1_g", "ln1_bt", "wq_q", "wq_s", "wq_b", "wk_q", "wk_s", "wk_b", "wv_q", "wv_s", "wv_b",
                           "wo_q", "wo_s", "wo_b", "ln2_g", "ln2_bt", "w1_q", "w1_s", "w1_b", "w2_q", "w2_s", "w2_b"};
    for (const char* n : names) {
      const int i = rt_->arg(gr, n);
      if (i < 0) throw std::runtime_error(std::string("whisper: NPU layer graph has no input ") + n);
      layer_args_.push_back(i);
    }
    const int out = rt_->output_named(gr, "y");
    if (out < 0) throw std::runtime_error("whisper: NPU layer graph output");
    layer_args_.push_back(out);
    // (initialize needs every argument: the first layer's)
    bind_layer(gr, layers_[0], xa_, xb_);
    if (!rt_->initialize(gr)) throw std::runtime_error("whisper: NPU layer graph initialize");
    return gr;
  }
  ze_graph_handle_t build_tail(int D, int A) {
    npu::Ir ir;
    const auto x = ir.param("x", {"f16", {A, D}});
    const auto g = ir.param("lng", {"f16", {D}});
    const auto b = ir.param("lnb", {"f16", {D}});
    ir.result(ir.add_op("h_ln", ir.multiply("h_mul", ir.mvn_last("h_mvn", x, 1e-5f), g), b), "h_out");
    ze_graph_handle_t gr = compile(ir, "tail");
    const int o = rt_->output_named(gr, "h");
    if (o < 0 || rt_->g->pfnSetArgumentValue(gr, (uint32_t)o, out_) != 0 || !rt_->set(gr, "x", xa_) || !rt_->set(gr, "lng", lng_) ||
        !rt_->set(gr, "lnb", lnb_) || !rt_->initialize(gr))
      throw std::runtime_error("whisper: NPU tail graph arguments");
    return gr;
  }
  void bind_layer(ze_graph_handle_t gr, const Layer& L, uint16_t* x, uint16_t* y) {
    void* v[] = {x, L.ln1g, L.ln1b, L.w[0].q, L.w[0].s, L.b[0], L.w[1].q, L.w[1].s, L.b[1], L.w[2].q, L.w[2].s, L.b[2],
                 L.w[3].q, L.w[3].s, L.b[3], L.ln2g, L.ln2b, L.w[4].q, L.w[4].s, L.b[4], L.w[5].q, L.w[5].s, L.b[5], y};
    for (size_t i = 0; i < layer_args_.size(); ++i)
      if (rt_->g->pfnSetArgumentValue(gr, (uint32_t)layer_args_[i], v[i]) != 0)
        throw std::runtime_error("whisper: NPU layer argument " + std::to_string(i));
  }
  void run(ze_graph_handle_t gr) {
    if (!rt_->start(gr) || !rt_->wait()) throw std::runtime_error("whisper: NPU run failed");
  }
};

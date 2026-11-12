// inferzero: the engine on the Level Zero runtime (zero/clrt.h); it began as a copy of the OpenCL engine's src/engine.h (removed 2026-10-20).
// inferzero engine: Qwen3.5 / Qwen3.6 / Qwen3.8 dense (hybrid Gated DeltaNet + gated attention, GGUF arch
// "qwen35"), their mixture-of-experts variant ("qwen35moe", GSQ 2-bit experts from tools/convert_gsq.py) and
// Gemma 4 on Intel Xe-LPG integrated GPUs via OpenCL.
//
// Weights: the raw GGUF blocks of every matrix live on the GPU (used by the single-token GEMV path). For prompt
// processing each matrix is multiplied in the int8 "W8" form: decoded once at load when it fits the memory
// budget, otherwise decoded just in time before each GEMM (zero/kernels/decode.cl). Q8_0 uses its own repacked form.
#pragma once
#include "clrt.h"
#include "xcache.h"
#include "npu/npu_rt.h"  // (the NPU experiment, ARC_NPU=1: zero/npu/engine_npu.h)
#include "../src/gguf.h"
#include "../src/quants.h"
#include "../src/json.h"
#include "../src/load_progress.h"
#include <algorithm>
#include <chrono>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <set>

struct GpuWeight {
  uint32_t type = 0;  // GGUF type of the raw rows (BF16/F16/F32 matrices are requantized to Q8_0)
  int N = 0, K = 0;
  cl_mem raw = nullptr;               // raw GGUF rows
  cl_mem wq = nullptr, wh = nullptr;  // resident GEMM form (W8, or repacked Q8_0); null = decode just in time
  int gemm_qtype = 0;                 // zero/kernels/gemm.cl QTYPE
  int role = 0;                       // just-in-time W8 staging slot (one per matrix role within a layer)
};

struct Hparams {
  std::string arch = "qwen35";
  bool gemma = false;  // Gemma 4: sliding-window + global attention layers, GeGLU, sandwich norms
  bool qwen3 = false;  // Qwen3 dense (qwen3 / qwen3vl, Qwen3-ASR's text model): full attention in every layer, no
                       // output gate, 128-dim heads with RoPE over all dims, SwiGLU FFN
  bool ds41 = false;  // DeepSeek-V4.1-Flash (deepseek41): zero/engine_ds41.h
  bool swa_rings = false;  // sliding-window layers in ring caches (Gemma 4): no recurrent state, a cached
                          // prefix is reused by rewind() while the rings still hold it, per-layer KV geometry (kv_layers)
  int n_layer = 0, n_embd = 0, n_ff = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0, n_vocab = 0;
  int ssm_d_state = 0, ssm_nk = 0, ssm_nv = 0, ssm_d_inner = 0, conv_k = 0, full_attn_interval = 4;
  float eps = 1e-6f, rope_base = 1e7f;
  int rope_sect[4] = {0, 0, 0, 0};  // M-RoPE sections (Qwen3.5 / 3.6 / 3.8 with vision: 11, 11, 10, 0)
  // Gemma 4 (head_dim above = the largest head size, for buffer sizing)
  int n_swa = 0, hd_swa = 0, hd_full = 0;
  int n_ple = 0, n_layer_kv = 0;  // per-layer embedding size (E2B/E4B), layers that own a KV cache
  float rope_base_swa = 1e4f, softcap = 0.0f;
  std::vector<int> swa, kv_heads;  // per layer
  // mixture of experts (qwen35moe): routed experts (2-bit, zero/kernels/moe.cl) + a sigmoid-gated shared expert
  bool moe = false, moe_std = false;  // moe_std: standard GGUF expert tensors (not GSQ)
  int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;
  int n_ff_exp_pad = 0;  // expert FFN width as the down projection reads it (Flash Next: 640 padded to 768)
  int router_rows() const { return (n_expert + 1 + 15) / 16 * 16; }  // router + shared-expert gate, padded
  // Qwen3.8 Flash Next (qwen4exp): hyper-connections (hc_n residual streams of n_embd, low rank hc_lr) and the PLE
  // n-gram embedding block at layer ple_layer (docs/flash-next-spec.md)
  bool hc = false;
  int hc_n = 1, hc_lr = 0, ple_layer = -1, ple_dim = 0, ple_ng = 0, ple_per = 0, ple_taps = 0;
  int64_t ple_eos = -1;
  int ple_img = -1;  // (ple.image_token_id, <|image_pad|>: the id of an image's rows in the n-gram hash)
  std::vector<int64_t> ple_mult, ple_off, ple_voc;
  // QSA sparse attention (Flash Next): block size per layer (0: dense), indexer heads / key size, keys selected
  std::vector<int> qsa_ratio;
  int qsa_heads = 0, qsa_dim = 0, qsa_topk = 0, n_qsa = 0;
  int hce() const { return hc_n * n_embd; }  // residual floats per token
  int conv_dim() const { return 2 * ssm_nk * ssm_d_state + ssm_d_inner; }
  bool recurrent(int il) const { return !gemma && (il + 1) % full_attn_interval != 0; }
};

struct EngineOptions {
  int ctx = 8192;
  int chunk = 0;              // prompt tokens per forward step; 0 = auto
  int min_chunk = 0;          // at least this many (Gemma 4 images: one non-causal block of up to the image limit)
  double w8_budget_gb = 9.0;  // keep decoded W8 weights resident if raw + W8 fit in this budget
  bool act32_all = true;      // per-32 activation scales for every GEMM (false: only for the sensitive inputs)
  bool attn_int8 = true;      // int8 (DP4a) attention scores; false = f32 scores
  bool mtp = false;           // load the MTP (nextn) block for speculative decoding (mtp_draft / verify)
  int mtp_drafts = 0;         // drafts per step (1..7; later ones chained on the MTP block's own output); 0 = auto:
                              // 3 for wide dense Qwen models (27B: code 7.8 -> 9.5 tok/s, text +2.5 %), else 2
  float mtp_min_p = 0.5f;     // draft a second token only if the MTP block gives the first this probability
  std::string mtp_model;      // Gemma 4: assistant drafter GGUF (empty: an mtp-*.gguf next to the model, if any)
  std::string dflash;         // DFlash2 block drafter (a z-lab DFlash2 folder or its model.safetensors): with mtp, it
                              // drafts instead of the MTP block (Qwen3.8 27B)
  bool mtp_block = true;      // with dflash: false = the MTP block is not loaded, DFlash2 drafts every step (--draft
                              // dflash); true = both, picked per step by how many DFlash2 drafts were taken
  std::string kernel_dir = "zero/kernels";  // (inferzero: its own copy, with the -DXCACHE kernels)
  bool dry = false;  // only add up the GPU memory a load takes (no allocations, uploads or kernels): --ctx auto
  // inferzero expert streaming (standard-type MoE GGUFs): routed experts read from the SSD into an LRU cache on
  // the GPU. Cache size: expert_cache_gb when > 0, else expert_cache_pct (0..100) of all experts when > 0, else
  // automatic: what is left after the rest of the model within the GPU's memory share less driver_slack().
  bool stream_experts = true;
  double expert_cache_gb = 0, expert_cache_pct = 0;
  // RAM tier between the expert cache and the SSD (zero/expert_store.h set_ram), GiB: < 0 automatic (a GPU with
  // memory of its own: the free RAM less 16 GiB, shared by a split model's GPUs; an iGPU: none), 0 none
  double expert_ram_gb = -1;
  size_t stream_reserve = 0;  // bytes the automatic cache size leaves for GPU memory allocated later (image encoders)
  int slots = 1;  // conversations held at once (server --slots): each further one its own KV caches and states
  // the NPU next to the GPU for prompt chunks (zero/npu/engine_npu.h, --npu): 0 off, 1 on where it pays off (the Qwen3.8
  // 27B), 2 forced on every Qwen model; -1: the ARC_NPU environment variable (unset: off)
  int npu = -1;
  // tensor parallel (zero/engine_tp.h, --gpus): this engine is shard tp_rank of a model split over tp_size GPUs
  int tp_rank = 0, tp_size = 1;
  // weights in host memory (iGPU: the same DRAM, read at ~91 % of device memory's speed, not counted in the GPU share),
  // GiB: < 0 automatic (a dense model whose weights fill more than 85 % of the share: Gemma 4 31B), 0 none
  double host_weights_gb = -1;
};

class VisionQwen3VL;
struct TpComm;
class TpEngine;
#include "mgpu/tp_mirror.h"  // (serverz --gpus: shard 0's calls run on every shard)

class Engine {
  friend class VisionQwen3VL;
  friend class VisionGemma4V;
  friend class AudioGemma4A;  // the vision encoders use the engine's OpenCL context, kernels and GEMM
  friend class AudioQwen3A;

 public:
  Hparams hp;
  EngineOptions opt;
  int ctx = 0, max_chunk = 0, pos = 0;
  // rotary position = cache position + rope_shift for text (Qwen M-RoPE: an image of gw x gh merged tokens advances
  // the rotary position by max(gw, gh) instead of gw * gh); kept by the caller with the conversation
  int rope_shift = 0;
  bool profile = false, w8_resident = false, has_mtp = false;
  // keep_rows: forward() takes a prompt in chunks of max_chunk only, never in the 8-token passes of just-in-time
  // weights, so logits_at() can read every row of the call (runz's checks of every position: --kl-ref with every
  // position's reference, --perplexity). x_rows_: the rows of x_ the last forward() left, the ones logits_at reads.
  // (2026-11-10: Gemma 4 12B's --kl-ref read rows 0..19 of a 20-token prompt that had gone in passes of 8, 8 and 4:
  //  KL 10.75 at every position, while its own next token was right)
  bool keep_rows = false;
  int x_rows_ = 0;
  // weight bytes the GEMVs read (token generation; bin/runz.exe reports GB/s against read_gbs())
  uint64_t rd_bytes = 0;
  // this GPU's read bandwidth in GB/s: the best of 5 streams of a 256 MB buffer (block reads, 16 lanes; this laptop's
  // iGPU ~75, a discrete card several times that), so runz's "% of" fits the machine it runs on
  double read_gbs() {
    static const char* src = R"CLC(
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void rd_block(__global const uint* p, __global uint* out, int per_item) {
  __global const uint* q = p + get_global_id(0) / 16 * (size_t)per_item * 16;
  uint acc = 0;
  for (int i = 0; i < per_item; i += 8) {
    const uint8 v = intel_sub_group_block_read8(q + (size_t)i * 16);
    acc ^= v.s0 ^ v.s1 ^ v.s2 ^ v.s3 ^ v.s4 ^ v.s5 ^ v.s6 ^ v.s7;
  }
  if (acc == 0x12345678u) out[0] = acc;
})CLC";
    const size_t bytes = (size_t)256 << 20;
    const int per_item = 1024;  // uints each sub-group lane strip covers: 64 KB per sub-group
    cl_program p = cl_.build(src, "-cl-std=CL3.0");
    progs_.push_back(p);
    cl_kernel k = cl_.kernel(p, "rd_block");
    // (an allocation that fails throws here instead of ending the process: a GPU share nearly full of a model left
    //  runz exiting instead of printing without the "% of" figure; its caller catches it. Audit 2026-11-12)
    cl_mem b = nullptr, o = nullptr;
    {
      const bool was = zrt::oom_throws;
      zrt::oom_throws = true;
      try {
        b = buf(bytes);
        o = buf(64);
      } catch (...) {
        zrt::oom_throws = was;
        clReleaseKernel(k);  // (before its module goes, in release_all, as below)
        if (b) {
          free_buf(b);
          gpu_bytes -= 64;  // (buf counted o before it failed)
        } else {
          gpu_bytes -= bytes;
        }
        throw;
      }
      zrt::oom_throws = was;
    }
    double best = 1e30;
    for (int r = 0; r < 6; ++r) {
      set_args(k, 0, b, o, per_item);
      size_t gws[1] = {bytes / 4 / per_item}, lws[1] = {256};
      clFinish(cl_.queue);
      const auto t0 = std::chrono::steady_clock::now();
      run("read_gbs", k, 1, gws, lws);
      clFinish(cl_.queue);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (r > 0) best = std::min(best, ms);  // (the first: warm-up)
    }
    free_buf(b);
    free_buf(o);
    clReleaseKernel(k);  // (before its module goes, in release_all)
    return bytes / 1e6 / best;
  }
  static uint64_t raw_bytes(const GpuWeight& w) { return (uint64_t)w.N * (w.K / 256) * gguf_sb_bytes(w.type); }
  std::map<std::string, double> prof_ms;
  std::map<std::string, int> prof_n;
  size_t gpu_bytes = 0;
  double load_ms = 0;

  Engine(const std::string& path, const EngineOptions& o) : opt(o), f_(path), path_(path) {
    // (a constructor that throws runs no destructor: what init() made on the GPU so far is freed here, or a failed
    //  model switch in the server would keep it until the process ends)
    try {
      init(o);
    } catch (...) {
      try {
        release_all();
      } catch (...) {  // (a split model's GPU wait that gave up while freeing: the first error is the one to report)
      }
      throw;
    }
  }

  // Frees everything on the GPU (the server switches models in one process).
  ~Engine() { release_all(); }
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

 private:
  void release_all() {
    if (zrt::D().cl) clFinish(cl_.queue);  // (init may have failed before the runtime existed: nothing to wait for)
    // (a split model's pipelined sums, ARC_TP_PIPE: copies still queued after a failure read tp_h_ and host memory;
    //  they drain once the compute list has: every wait of that list comes after its signal)
    if (zrt::D().cl_tp) zrt::list_wait(zrt::D().cl_tp);
    if (npu_runs_) zlog::eprintf("ARC_NPU: %d NPU runs, %.0f ms waited for the NPU after the GPU's share\n", npu_runs_, npu_wait_ms_);
    npu_teardown();  // (ARC_NPU=1 only)
    for (auto& [m, p] : orig_ptrs_) m->ptr = p;  // (slots: each buffer object frees its own memory)
    for (auto& kv : kcache_) clReleaseKernel(kv.second);
    for (cl_kernel k : {k_tp_add_, k_tp_pack_, k_tp_moe_mask_, k_tp_moe_scatter_, k_tp_moe_tiles1_, k_tp_moe_combine_,
                        k_tp_half_, k_tp_rs_sum_, k_tp_rs_add_, k_tp_rs_add_r_})
      if (k) clReleaseKernel(k);
    k_tp_add_ = k_tp_pack_ = k_tp_moe_mask_ = k_tp_moe_scatter_ = k_tp_moe_tiles1_ = k_tp_moe_combine_ = nullptr;
    k_tp_half_ = k_tp_rs_sum_ = k_tp_rs_add_ = k_tp_rs_add_r_ = nullptr;
    for (cl_program p : progs_) clReleaseProgram(p);
    for (auto it = mems_.rbegin(); it != mems_.rend(); ++it)  // sub-buffers (created later) before their parents
      if (*it) clReleaseMemObject(*it);
    for (void* p : host_ptrs_) zrt::mem_free(zrt::ctx, p);  // (weights in host memory: plan_tight)
    host_ptrs_.clear();
    if (mtp_xbuf_) {  // (the MTP head's pinned experts; the expert cache's table row went with xc_ or goes after)
      zrt::mem_free(zrt::ctx, mtp_xbuf_);
      zrt::D().device_bytes -= mtp_xbytes_;
      mtp_xbuf_ = nullptr;
    }
    if (cl_.queue) clReleaseCommandQueue(cl_.queue);
    if (cl_.ctx) clReleaseContext(cl_.ctx);
    dfl_close();  // (the DFlash2 drafter's mapped safetensors file)
    kcache_.clear();
    progs_.clear();
    mems_.clear();
    orig_ptrs_.clear();
  }
  void init(const EngineOptions& o) {
    npu_preinit();  // (ARC_NPU=1 only)
    if (!cl_.init(true)) throw std::runtime_error("OpenCL init failed");
    ctx = (o.ctx + 15) / 16 * 16;
    adopt_mtp_file();  // (Flash Next's MTP head in a file of its own: before anything keeps a pointer into f_)
    load_hparams();
    if (mtp_ext_) {  // (its layer, il = n_layer: full attention, QSA with the file's compress ratio)
      hp.qsa_ratio.resize(hp.n_layer + 1, 0);
      const auto* cr = mtp_file_->get_ints("qwen4exp.attention.compress_ratios");
      if (cr && (int)cr->size() > hp.n_layer && f_.find("blk." + std::to_string(hp.n_layer) + ".indexer.k_proj.weight")) {
        const int r = (int)(*cr)[hp.n_layer];
        if (r > 0 && hp.n_qsa && r != hp.qsa_ratio[hp.full_attn_interval - 1])
          throw std::runtime_error("MTP head: its QSA block size differs from the model's");
        hp.qsa_ratio[hp.n_layer] = r;
        hp.n_qsa += r > 0;
      }
    }
    if (opt.tp_size > 1) tp_setup();  // (this shard's heads and columns: engine_tp_shard.h)
    // (MoE, Qwen3.6 35B: 1 and 2 drafts equally fast, 20.2 vs 20.3 tok/s; 1 keeps one DeltaNet state copy less)
    // (measured on inferzero, server answers at temperature 0, 2026-10-16: 35B 1 -> 3 drafts 23.9 -> 26.1 tok/s,
    //  4B 2 -> 3 22.9 -> 24.8; 9B 2 and 3 alike (15.8 / 15.6); 27B 3 = 5 = 7; Gemma's drafter unchanged at 2)
    if (!opt.dflash.empty()) opt.mtp = true;
    if (!opt.mtp_block && opt.dflash.empty()) {  // (no drafter to draft instead)
      if (opt.mtp) zlog::eprintf("inferzero: --draft dflash without a DFlash2 drafter: the MTP block drafts\n");
      opt.mtp_block = true;
    }
    if (opt.mtp_drafts <= 0) opt.mtp_drafts = !opt.dflash.empty() ? 7 : mtp_block_drafts();
    auto t0 = std::chrono::steady_clock::now();
    plan_weights();
    stream_ = hp.moe_std && opt.stream_experts && !getenv("ARC_ZERO_NO_STREAM");
    if (f_.has("inferzero.package") && !stream_)  // (a package keeps its experts only in the expert file)
      throw std::runtime_error("this is an inferzero package: it runs with expert streaming only (no --no-stream)");
    if (stream_ && !opt.dry) open_experts();
    // (streamed experts: 8192-token chunks, each chunk reads a layer's missing experts once and decodes them once;
    // Qwen3.6 35B at ctx 262K, chunk 2048 / 4096 / 8192: expert cache 83 / 81 / 77 %, 32K prompt 485 / 533 / 604
    // tok/s, 128K prompt 284 / 306 / 324, generation at 128K 13.0 / 12.8 / 12.6 tok/s)
    // (Flash Next: 2048, its residual is 4 streams wide)
    // (Gemma 4's two-pass mode: 2048 for streamed experts too. Its float K / V rings of the sliding layers hold the
    //  window + a chunk + 4096 rows, 4.1 GB at 8192 on the 26B, which left the expert cache at 39 %. 26B, ctx 16K,
    //  chunk 8192 / 4096 / 2048: expert cache 39 / 64 / 77 %, 4K prompt 216 / 184 / 217 tok/s, generation after a
    //  14K prompt 9.9 / 12.3 / 13.0 tok/s)
    plan_tight();  // (a dense model that nearly fills the GPU share: weights in host memory, smaller chunks and rings)
    // (DeepSeek-V4.1: 1024, its queries are 64 x 512 floats a row)
    max_chunk = o.chunk > 0 ? o.chunk : hp.ds41 ? 1024 : stream_ ? (hp.hc || hp.gemma ? 2048 : 8192) : hp.moe ? 2048 : tight_ ? 1024 : (w8_resident ? 512 : 4096);
    max_chunk = (std::max(max_chunk, o.min_chunk) + 127) / 128 * 128;
    q2_init();
    // (also in a dry run, --ctx auto's memory plan: H8's buffers and the float copy in aq_ are counted then; self audit
    //  2026-11-08, ~0.35 GB on the 27B)
    p140_ = p140_wanted();  // (the Arc 140T preset, zero/xe2/engine_xe2.h: it needs H8)
    h8_auto_ = !p140_ && h8_auto_wanted();  // (H8 by default on discrete cards: zero/xe2/engine_xe2.h)
    h8_init();  // (H8 prompt GEMMs, ARC_H8=1: zero/engine_h8.h; after q2_init, before alloc_buffers)
    p140_ = p140_ && h8_;
    if (!opt.dry) build_kernels();
    // (8-lane GPUs, Arrow Lake H / Arc 140T: XMX only with ARC_XMX_FORCE=1. A 140T, issue #5, 2026-11-07: gemm_x8 right
    //  bit for bit on real DPAS, but the 27B's 2K prompt 114-120 tok/s against 180 on DP4a, GEMMs 1.6x slower, while
    //  the start-time speed check on its small shape had picked XMX. Off, the GPU runs as with ARC_NO_XMX)
    //  (discrete cards with 8-lane DPAS, the Alchemist A-series: XMX checked, then timed on a model-sized GEMM in
    //  xmx_check as an iGPU is; no one has measured gemm_x8 on one. Self audit 2026-11-08: the gate had caught them too)
    const bool xmx_force = getenv("ARC_XMX_FORCE") && atoi(getenv("ARC_XMX_FORCE")) != 0;
    const bool sg8_igpu = cl_.sg8 && zrt::integrated();
    if (!opt.dry && cl_.xmx && sg8_igpu && !xmx_force && !p140_ && !getenv("ARC_XMX_TRY")) {
      static bool said = false;
      if (!said) zlog::eprintf("inferzero: XMX off on this 8-lane GPU (slower than DP4a on a 140T); ARC_XMX_FORCE=1 to try it\n");
      said = true;
    }
    if (!opt.dry && ((cl_.xmx && (!sg8_igpu || xmx_force || p140_)) || getenv("ARC_XMX_TRY"))) {  // (ARC_XMX_TRY=1: the check on any GPU, tests)
      // (the switches of zero/xe2/engine_xe2.h, before the check so it covers them: every GPU that offers XMX, discrete
      // or integrated; a GPU without it, as the 125H, never comes here)
      xe2_ = true;
      // (once per process: a split model's GPUs are all the same device (ClRt::open_devices), the check's settings
      //  come from the environment; 4 B60s loaded in 8.7-14.9 s, each shard checking again)
      static std::mutex xmx_mu;
      static int xmx_known = -1, x2d_known = 0, slow_known = 0;
      {
        std::lock_guard<std::mutex> g(xmx_mu);
        if (xmx_known < 0) {
          // (zero/xe2/engine_xe2.h: the check with the 2D block loads first, where they are wanted; a build that fails
          //  or differs leaves them out and checks again, so XMX itself is not lost to them)
          x2d_ = x2d_wanted();
          xmx_known = xmx_check() ? 1 : 0;
          if (x2d_ && !xmx_known && !xmx_slow_) {
            zlog::eprintf("inferzero: 2D block loads (ARC_GEMM_2D) do not build or differ: not used\n");
            x2d_ = false;
            xmx_known = xmx_check() ? 1 : 0;
          }
          x2d_known = x2d_ && xmx_known ? 1 : 0;
          slow_known = xmx_slow_ ? 1 : 0;
          if (xmx_known) zlog::eprintf("inferzero: XMX on for the prompt GEMMs (results checked against DP4a)%s\n",
                                 x2d_known ? ", with 2D block loads" : "");
        }
      }
      xmx_ = xmx_known == 1;
      x2d_ = x2d_known == 1;
      // (XMX slower than DP4a in the speed check of an iGPU: the GPU's own defaults, as on one without XMX)
      if (slow_known) xe2_ = false;
    }
    // (the H8 kernels this engine runs, checked against DP4a apart from the rest of XMX and kept per kind of check, not
    //  once per process: a server that loads the 35B (no H8) and then the 27B (H8 + 4 column groups) checks those then;
    //  a failing H8 kernel turns H8 off, not XMX. Self audit 2026-11-08)
    if (!opt.dry && xmx_ && h8_) {
      static std::mutex h8_mu;
      static std::map<std::string, bool> h8_known;
      const std::string key = std::to_string(h8_qtype()) + "|cg" + std::to_string(h8_cg_for(64)) + (p140_ ? "|140t" : "") +
                              (h8_env() ? "|env" : "") + (x2d_ ? "|2d" : "");
      std::lock_guard<std::mutex> g(h8_mu);
      auto it = h8_known.find(key);
      if (it == h8_known.end()) it = h8_known.emplace(key, xmx_check(1)).first;
      if (!it->second) {
        zlog::eprintf("inferzero: the H8 XMX kernels differ from DP4a or do not build: H8 off, XMX kept (please report)\n");
        p140_ = false;
        h8_auto_ = false;
        h8_ = false;
      }
    }
    if (p140_ && !xmx_) {  // (its kernels did not build or differ: the plain DP4a path, and H8 only if asked for)
      p140_ = false;
      if (!h8_env()) h8_ = false;
    }
    if (h8_auto_ && !xmx_) {  // (the discrete default needs XMX: H8 on DP4a is not what was measured)
      h8_auto_ = false;
      if (!h8_env()) h8_ = false;
    }
    if (p140_) {
      static std::once_flag said;
      std::call_once(said, [] {
        zlog::eprintf("inferzero: Arc 140T preset: XMX + H8 for prompt chunks of %d+ tokens, DP4a below (ARC_140T=0: off)\n", kP140Min);
      });
    }
    if (!opt.dry) x16_init();  // (the f16 XMX prompt GEMM: zero/xe2/engine_xe2.h)
    if (!opt.dry) xw_init();   // (the H8R GEMM with wider XMX tiles: the 140T preset, or ARC_GEMM_XW=1: zero/xe2/engine_xw.h)
    if (!opt.dry && hp.ssm_nv > 0) gdnx_init();  // (the chunked DeltaNet on XMX, ARC_GDN_CHUNK=3: the same file)
    if (!opt.dry && k_attn_q8_) attnx_init();      // (the prompt attention on XMX, ARC_ATTN_XMX=1: the same file)
    {  // (the scripts' progress bar: the bytes of the tensors uploaded here, not the experts streamed from the SSD)
      // (kept on the host and read row by row: the token embedding when the model has its own output head, Gemma 4
      //  E2B / E4B's per-layer embeddings; E4B's bar stalled at ~60 % and jumped with them counted)
      const bool tied = !f_.find("output.weight");
      double total = 0;
      for (const GgufTensor& t : f_.tensors)
        if (!(stream_ && t.name.find("_exps.") != std::string::npos) && !(t.name == "token_embd.weight" && !tied) &&
            t.name != "per_layer_token_embd.weight")
          total += (double)t.nbytes;
      if (!opt.dry) loadp::begin(total);
      load_weights();
      tp_loaded();  // (a split model: the host copies of its cut tensors freed)
      if (host_bytes_ && !opt.dry)
        zlog::eprintf("inferzero: %.2f GiB of the weights in host memory (outside the GPU share); prompt chunks of %d, "
                        "sliding-window rings of %d rows\n", host_bytes_ / 1073741824.0, max_chunk,
                std::min(ctx, (hp.n_swa + max_chunk + swa_margin() + 255) / 256 * 256));
      if (!opt.dry) loadp::step(loadp::st().what, total, total);
    }
    {
      const size_t g0 = gpu_bytes;
      alloc_buffers();
      mem_bufs_ = gpu_bytes - g0;
    }
    npu_setup();  // (ARC_NPU=1 only)
    if (opt.slots > 1) add_slots(opt.slots);  // (before the expert cache takes what is left)
    if (xc_) size_expert_cache();
    // (--ctx auto: room for an expert cache of 90 % of the experts, as far as what the weights and buffers leave
    //  allows with 128K tokens of context (32K until 18 Oct), or at least 3/4 of it. The context took the memory
    //  before (35B: 254K tokens and a 60 % cache: generation 33 / 19 instead of 43 / 23 tok/s, with pauses for the
    //  SSD); Flash Next, whose experts are 3x the GPU share, was left at the minimum context of 4096. Context sweep
    //  18 Oct (docs/speed-progress.md): 35B UD-IQ4_XS at 131K keeps 81 % of the experts and answers as fast as at
    //  64K; Flash Next loses 2-6 % from 32K to 131K.)
    if (stream_ && opt.dry) {
      const double left = (double)device_mem() - driver_slack() - (double)gpu_bytes + (double)ctx_bytes(ctx) -
                          (double)snapshot_bytes() - (double)opt.stream_reserve;
      // (--slots N: 128K of context in all, each slot 128K / N; the floor of 3/4 only as far as it leaves every slot
      //  32K: with a plain floor Gemma 4 26B-A4B kept 8,192 tokens per slot, its sliding-window rings ~1.4 GB each.
      //  One slot: as before)
      const int ns = std::max(opt.slots, 1), per = (131072 / ns + 4095) / 4096 * 4096;
      auto ctx_all = [&](int c) { return (double)ctx_bytes(c) + (double)(ns - 1) * (double)slot_bytes(c); };
      // (Gemma 4 26B-A4B: a floor of 0.6, 128K instead of 49K: generation at 36K of depth 11.4 tok/s with 89 % of the
      //  experts cached (49K), 10.5 with 70 % (131K), 8.8 with 39 % (262K); the user ran 131K by hand. 2026-10-23)
      const double fl = hp.gemma ? 0.6 : 0.75;
      const double room = std::max(left - ctx_all(per), std::min(fl * left, left - ctx_all(kSlotCtxFloor)));
      gpu_bytes += (size_t)std::max(0.0, std::min(0.9 * (double)stream_bytes_, room));
    }
    mem_load_ = gpu_bytes;
    // the GGUF pages read while loading leave the working set (standby: Windows reuses them when it needs RAM)
    SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
    clFinish(cl_.queue);
    load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    reset();
  }

 public:
  // Clears the KV cache and recurrent states.
  void reset() {
    TP_MIRROR(reset(), TpMirror::kNone, 0);
    pos = 0;
    ds_hist_.clear();
    written_hi_ = 0;
    swa_lo_ = 0;
    pend_toks_.clear();
    last_drafts_.clear();  // (a verification pass never settled, a batch that threw: nothing left of them)
    vrows_ = dfl_used_ = spec_ = false;
    batch_.clear();
    dfl_cap_lo_ = dfl_cap_base_ = dfl_cap_end_ = 0;
    nc_lo_ = nc_hi_ = -1;
    if (hlast_) fill(hlast_, mtp_hw() * 4);
    for (auto& L : layers_)
      if (hp.recurrent(L.il)) {
        fill(L.conv_state, conv_bytes());
        fill(L.S, (size_t)hp.ssm_nv * hp.ssm_d_state * hp.ssm_d_state * 4);
      }
    clFinish(cl_.queue);
  }

  // Feeds tokens at the current position; returns the logits for the last token.
  const std::vector<float>& forward(const int* tokens, int n) {
    TP_MIRROR(forward(tokens, n), TpMirror::kLogits, 0);
    if (pos + n > ctx) throw std::runtime_error("context overflow");
    if (has_mtp) flush_mtp_pending();
    if (hp.ple_layer >= 0) ple_sync();
    struct DflCommit { Engine& e; int p0; ~DflCommit() { if (e.dfl_) e.dfl_commit_prompt(p0); } } dfl_commit{*this, pos};
    if (dfl_) dfl_window(pos, n);
    if ((!w8_resident || stream_) && n > max_chunk && !hp.n_ple && !hp.ds41) {
      // layer-major in segments of up to lm_seg_ tokens (each matrix is decoded once per segment)
      prog_total_ = n;
      for (int s0 = 0; s0 < n; s0 += lm_seg_) {
        prog_base_ = s0;
        const auto& l = forward_layer_major(tokens + s0, std::min(lm_seg_, n - s0));
        if (s0 + lm_seg_ >= n) {
          prog_base_ = prog_total_ = 0;
          return l;
        }
      }
    }
    const bool by_layer = cancel && !w8_resident;  // a chunk takes seconds: stop after any layer
    struct NoRpos { int& u; ~NoRpos() { u = 0; } } no_rpos{use_rpos_};  // (set_media's positions: this call only)
    // just-in-time weights: a chunk decodes every matrix first (27B: ~2 s whatever its size), a pass of up to 8 rows
    // reads them directly (~0.35 s), so short prompts (a tool result's tail, a short message) go in passes of 8:
    // up to 40 tokens (27B, 10 tokens: 2.0 -> 0.65 s; 40: 2.1 -> 1.7 s; 49: 2.15 -> 2.5 s), up to 32 with two-pass
    // matmuls, whose passes cost more (Gemma 4 12B, ~11 / 22 / 27 tokens 1.1 / 1.2 / 1.25 -> 0.6 / 0.85 / 1.1 s;
    // ~40: 1.25 -> 1.4)
    const int few = q2mm_ ? 32 : 40;
    const int step = !keep_rows && !w8_resident && !stream_ && n <= few && max_chunk >= 64 ? 8 : max_chunk;
    for (int c0 = 0; c0 < n; c0 += step) {
      if (c0 > 0 && cancelled()) throw Cancelled();  // between chunks the state is complete
      const int T = std::min(step, n - c0);
      // pad to a multiple of 128 tokens so GEMM work-groups always get 8 token tiles (rows beyond T are zero)
      const int Mp = T >= 128 ? std::min((T + 127) / 128 * 128, max_chunk) : (T + 15) / 16 * 16;
      if (by_layer) keep_block();
      ring_guard(T);  // (Gemma: a snapshot's rings' window copied before this chunk overwrites it)
      embed(tokens + c0, T, Mp);
      media_pos(tokens + c0, T);
      for (auto& L : layers_) {
        run_layer(L, T, Mp);
        if (by_layer && &L != &layers_.back() && cancelled()) cancel_block(pos);
      }
      pos += T;
      x_rows_ = T;
      if (mtp_block_) mtp_prefill(tokens + c0, T, Mp, pos - T);
      if (c0 + T >= n) head(T);
      if (profile) collect_profile();
    }
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, (size_t)n_logits() * 4, logits_host_.data(), 0,
                                 nullptr, nullptr));
    collect_profile();
    return logits_host_;
  }

  // Prompt processing with just-in-time weights: every chunk goes through a layer before the next layer starts,
  // so each matrix is decoded once per prompt instead of once per chunk. The residual stream of all chunks lives
  // in x_all_ (x_ points at the current chunk's view).
  const std::vector<float>& forward_layer_major(const int* tokens, int n) {
    ring_guard(n);  // (each layer writes the whole span: see forward)
    const int pos0 = pos, nch = (n + max_chunk - 1) / max_chunk;
    struct SegN { int& v; ~SegN() { v = 0; } } seg_n{seg_n_};  // (the experts' deep prefetch, see stream_layer)
    seg_n_ = n;
    cl_mem x_saved = x_;
    // (any exception, not only a cancel: x_ is the engine's own buffer again, not a chunk's view)
    struct XBack { cl_mem& p; cl_mem s; ~XBack() { p = s; } } x_back{x_, x_saved};
    auto chunk_T = [&](int c) { return std::min(max_chunk, n - c * max_chunk); };
    auto chunk_Mp = [&](int c) {
      const int T = chunk_T(c);
      return T >= 128 ? std::min((T + 127) / 128 * 128, max_chunk) : (T + 15) / 16 * 16;
    };
    if (cancel) {
      if (cancelled()) throw Cancelled();  // (between segments)
      keep_block();
    }
    cl_mem ple_saved = ple_emb_;
    struct PleBack { cl_mem& p; cl_mem s; ~PleBack() { p = s; } } ple_back{ple_emb_, ple_saved};
    for (int c = 0; c < nch; ++c) {
      x_ = x_views_[c];
      if (!ple_views_.empty()) ple_emb_ = ple_views_[c];
      embed(tokens + (size_t)c * max_chunk, chunk_T(c), chunk_Mp(c));
    }
    // (set_media's rotary positions: the segment's rows written once, each chunk reads its part; every write syncs)
    struct NoRpos { int& u; cl_mem& p; cl_mem s; ~NoRpos() { u = 0; p = s; } } no_rpos{use_rpos_, rpos_, rpos_};
    const bool mpos = media_rpos_ && tokens >= media_base_ && tokens + n <= media_base_ + media_n_;
    cl_mem rp_seg = nullptr;
    if (mpos) {
      if (!rpos_all_) rpos_all_ = buf((size_t)std::max(lm_seg_, max_chunk) * 3 * 4);
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, rpos_all_, CL_TRUE, 0, (size_t)n * 3 * 4,
                                    media_rpos_->data() + (size_t)(tokens - media_base_) * 3, 0, nullptr, nullptr));
      rp_seg = rpos_all_;
    }
    for (size_t li = 0; li < layers_.size(); ++li) {
      Layer& L = layers_[li];
      for (int c = 0; c < nch; ++c) {
        x_ = x_views_[c];
        if (!ple_views_.empty()) ple_emb_ = ple_views_[c];
        cur_T_ = chunk_T(c);
        pos = pos0 + c * max_chunk;
        if (rp_seg) {
          rpos_ = from(rp_seg, (size_t)c * max_chunk * 3 * 4);
          use_rpos_ = 1;
        }
        run_layer(L, cur_T_, chunk_Mp(c));
      }
      if (progress) {
        const int total = prog_total_ ? prog_total_ : n;
        progress(prog_base_ + (int)((double)n * (li + 1) / layers_.size()), total);
      }
      if (profile) collect_profile();
      if (cancel && li + 1 < layers_.size() && cancelled()) {
        x_ = x_saved;
        cancel_block(pos0);
      }
    }
    pos = pos0 + n;
    if (mtp_block_)
      for (int c = 0; c < nch; ++c) {
        x_ = x_views_[c];
        mtp_prefill(tokens + (size_t)c * max_chunk, chunk_T(c), chunk_Mp(c), pos0 + c * max_chunk);
      }
    x_ = x_views_[nch - 1];
    cur_T_ = chunk_T(nch - 1);
    x_rows_ = cur_T_;
    head(cur_T_);
    // leave the last chunk in x_ as the chunk-by-chunk path does (logits_at() reads its rows)
    CL_CHECK(clEnqueueCopyBuffer(cl_.queue, x_, x_saved, 0, 0, (size_t)chunk_Mp(nch - 1) * hp.hce() * 4, 0, nullptr,
                                 nullptr));
    x_ = x_saved;
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, (size_t)n_logits() * 4, logits_host_.data(), 0,
                                 nullptr, nullptr));
    collect_profile();
    return logits_host_;
  }

  // Image rows inside a token prompt (causal Qwen M-RoPE images): while set, forward() over tokens in
  // base[0 .. n) takes row i from rows[i] when that is set (n_embd floats, as forward_embd takes them; the token id
  // there is a placeholder; Flash Next: <|image_pad|> in its n-gram hash) and gives every row its rotary position
  // rpos[3 i .. 3 i + 2] (t, h, w; a text row's three are its cache position + rope_shift). Text and images then go
  // through the layers in one pass: a streamed MoE model reads its experts once, not once per piece of the prompt.
  // Not with per-layer embeddings (Gemma E2B / E4B), an MTP block or a DFlash2 drafter (media_inline_ok).
  bool media_inline_ok() const { return !hp.n_ple && !mtp_block_ && !dfl_ && !hp.ds41; }
  void set_media(const int* base, size_t n, const std::vector<const float*>* rows, const std::vector<int>* rpos) {
    TP_MIRROR(set_media(base, n, rows, rpos), TpMirror::kNone, 0);
    media_base_ = base;
    media_n_ = n;
    media_ = rows;
    media_rpos_ = rpos;
  }
  void clear_media() { set_media(nullptr, 0, nullptr, nullptr); }

  // Feeds n rows of precomputed input embeddings (an image) at the current position as one chunk; with
  // non_causal the rows attend to each other in both directions in the sliding-window layers. Returns the logits
  // after the last row when `logits` is set.
  // With rpos (3 ints per row: t, h, w) the rows get those rotary positions (Qwen M-RoPE image tokens).
  const std::vector<float>& forward_embd(const float* embd, int n, bool non_causal, bool logits,
                                         const int* rpos = nullptr) {
    TP_MIRROR(forward_embd(embd, n, non_causal, logits, rpos), logits ? TpMirror::kLogits : TpMirror::kNone, 0);
    if (n <= 0 || n > max_chunk) throw std::runtime_error("forward_embd: 1..max_chunk rows");
    if (hp.ds41) throw std::runtime_error("DeepSeek-V4.1: image / audio rows are not supported yet");
    if (rpos) {
      if (!rpos_) rpos_ = buf((size_t)max_chunk * 3 * 4);
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, rpos_, CL_TRUE, 0, (size_t)n * 3 * 4, rpos, 0, nullptr, nullptr));
      use_rpos_ = 1;
    }
    struct NoRpos { int& u; ~NoRpos() { u = 0; } } no_rpos{use_rpos_};
    if (pos + n > ctx) throw std::runtime_error("context overflow");
    const int Mp = n >= 128 ? std::min((n + 127) / 128 * 128, max_chunk) : (n + 15) / 16 * 16;
    cur_T_ = n;
    if (hp.hc) {
      // Flash Next: each row as hc_n copies, its residual streams (as embed lays out a token's), and the PLE n-gram
      // rows of <|image_pad|> (ple.image_token_id) for its id; a row of n_embd laid out as one stream put four image
      // rows into one row's streams, and the PLE block read the last text chunk's rows
      const int W = hp.hce(), E = hp.n_embd;
      attn_in_ready_ = ffn_in_ready_ = false;  // (as embed)
      std::fill(x_host_.begin(), x_host_.begin() + (size_t)Mp * W, 0.0f);
      for (int t = 0; t < n; ++t)
        for (int c = 0; c < hp.hc_n; ++c)
          std::copy(embd + (size_t)t * E, embd + (size_t)(t + 1) * E, x_host_.begin() + (size_t)t * W + (size_t)c * E);
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, x_, CL_TRUE, 0, (size_t)Mp * W * 4, x_host_.data(), 0, nullptr, nullptr));
      if (hp.ple_layer >= 0) {
        const std::vector<int> ids(n, hp.ple_img >= 0 ? hp.ple_img : (int)hp.ple_eos);
        ple_embed(ids.data(), n);
      }
    } else {
      std::fill(x_host_.begin(), x_host_.begin() + (size_t)Mp * hp.n_embd, 0.0f);
      std::copy(embd, embd + (size_t)n * hp.n_embd, x_host_.begin());
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, x_, CL_TRUE, 0, (size_t)Mp * hp.n_embd * 4, x_host_.data(), 0, nullptr,
                                    nullptr));
    }
    if (hp.n_ple) {
      // E2B / E4B: every embedding row takes the per-layer token embedding of token 0 (padding), combined as usual
      // with the projection of the row itself (llama.cpp models/gemma4.cpp, TAG_GEMMA4_IMG_PADDING)
      const size_t np = (size_t)hp.n_ple * hp.n_layer;
      std::vector<float> pad(np);
      dequant_row(ple_tok_embd_->type, ple_tok_embd_->data, (int)np, pad.data());
      const float sc = std::sqrt((float)hp.n_ple);
      for (size_t t = 0; t < (size_t)n; ++t)
        for (size_t i = 0; i < np; ++i) ple_host_[t * np + i] = pad[i] * sc;
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, ple_tok_, CL_TRUE, 0, (size_t)n * np * 4, ple_host_.data(), 0, nullptr,
                                    nullptr));
      ple_inputs(n, Mp);
    }
    if (non_causal) { nc_lo_ = pos; nc_hi_ = pos + n - 1; }
    ring_guard(n);
    {
      struct Causal { int& lo; int& hi; ~Causal() { lo = hi = -1; } } causal{nc_lo_, nc_hi_};  // (also on a throw)
      for (auto& L : layers_) run_layer(L, n, Mp);
    }
    pos += n;
    if (logits) {
      head(n);
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, (size_t)n_logits() * 4, logits_host_.data(), 0,
                                   nullptr, nullptr));
    }
    collect_profile();
    return logits_host_;
  }

  // Optional progress callback for long prompts: (tokens-equivalent done, total).
  std::function<void(int, int)> progress;
  // Optional cancellation of prompt processing (the server: its client went away), polled between chunks and, with
  // just-in-time weights (where a block of the prompt takes seconds to minutes), after every layer. forward() then
  // throws Cancelled with the engine at the start of the block it was in: complete blocks stay processed. The
  // DeltaNet states come back from a host copy taken at the block start (27B: 157 MB, ~20 ms per block); Gemma moves
  // back while its sliding-window rings allow it (blocks up to max_chunk + 4096 tokens), else it is reset. `pos`
  // tells where it stands. Not for generation (the copy would cost every token).
  std::function<bool()> cancel;
  struct Cancelled {};

  const GgufFile& gguf() const { return f_; }
  // Runs the generation kernels once so that a first request does not wait for their just-in-time compilation: a
  // short prompt and a token, then with MTP / DFlash2 drafts and a verification pass of every row count (DFlash2 and
  // the MTP block each); leaves the engine reset. Only when this model, these options and these kernel sources have
  // not been warmed before (a marker in the kernel cache, "w2": since the Gemma drafter is warmed too): after an update the first requests took seconds per new
  // kernel (27B with DFlash2: a 27-token prompt 37 s, then 1-3 tok/s), later starts skip it. Returns true when it ran.
  bool warm_up_once(int tok, bool prompt = true) {
    const std::string& dir = cl_.cache_dir;
    char hex[17];
    snprintf(hex, sizeof hex, "%016llx",
             (unsigned long long)ClRt::fnv1a(gemm_src_ + "\x01" + ops_.all() + "\x01" + dec_.all() + "\x01" + moe_src_ + "\x01" + f_.path +
                                             "|" + std::to_string(opt.mtp) + "|" + opt.dflash + (opt.mtp_block ? "" : "|nomtp") +
                                             "|" + std::to_string(opt.mtp_drafts) +
                                             "|" + std::to_string(max_chunk) + "|" + cl_.device_id + "|w2"));
    const std::string marker = dir.empty() ? std::string() : dir + "\\warm-" + hex;
    // (every start (prompt = true): a prompt long enough for the large-chunk kernels (moe_rg from 320 rows), then a token: their first
    //  use in a process cost seconds even from the kernel cache; 35B with drafters, a first 1,749-token request 8.6 s
    //  (203 tok/s), the next ones 3.5 s)
    const int nwarm = std::max(1, std::min({prompt ? 512 : 32, max_chunk, ctx - 12}));
    if (!marker.empty() && std::filesystem::exists(marker)) {
      if (prompt) {
        std::vector<int> w(nwarm, tok);
        forward(w.data(), nwarm);
      }
      forward(&tok, 1);
      reset();
      return false;
    }
    std::vector<int> t(nwarm, tok);
    forward(t.data(), (int)t.size());
    forward(&tok, 1);
    // (Gemma 4's drafter too, through the same calls, its code unchanged: an E4B first request after an update
    //  generated at 3.7 tok/s, its drafter and verification kernels compiling, the next ones at 26)
    if (has_mtp) {
      for (int round = 0; round < (dfl_ ? 2 : 1); ++round) {
        for (int k = 1; k <= max_drafts() && pos + k + 2 < ctx; ++k) {
          dfl_ema_ = round == 0 ? 4.0 : 0.0;  // (round 0 with DFlash2: its drafts; round 1: the MTP block's)
          dfl_wait_ = 0;
          dfl_gap_ = 1 << 16;
          mtp_draft(tok, k);
          verify(tok, std::vector<int>(k, tok));
          verify_done(0);
        }
      }
    }
    reset();
    dfl_ema_ = 4.0;
    dfl_wait_ = 0;
    dfl_gap_ = 8;
    dfl_probe_ = false;
    if (!marker.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(dir, ec);
      if (FILE* f = fopen(marker.c_str(), "w")) fclose(f);
    }
    return true;
  }

  // Moves back to position P to reuse a cached prefix (models without recurrent state: swa_rings): possible
  // while the sliding-window rings still hold the n_swa positions before P. Qwen3.5 uses snapshot()/restore().
  bool rewind(int P) {
    TP_MIRROR(rewind(P), TpMirror::kNone, 0);
    if (!hp.swa_rings || P < 0 || P > pos) return false;
    if (written_hi_ - 1 - swa_cap_ >= P - hp.n_swa) return false;  // those slots were overwritten since
    if (swa_lo_ > 0 && P - hp.n_swa < swa_lo_) return false;  // a loaded conversation keeps K/V from there
    pos = P;
    return true;
  }

  // Logits predicting the token after row `row` of the last processed chunk (for perplexity evaluation).
  const std::vector<float>& logits_at(int row) {
    if (row < 0 || row >= x_rows_)  // (a row the last pass did not leave: another chunk's, or nothing)
      throw std::runtime_error("logits_at: row " + std::to_string(row) + " is not one of the " + std::to_string(x_rows_) +
                               " the last pass left (keep_rows: the prompt in one chunk)");
    head_rows(row, 1, out_norm_, nullptr);
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, (size_t)n_logits() * 4, logits_host_.data(), 0,
                                 nullptr, nullptr));
    return logits_host_;
  }
  // Recurrent-state snapshots for prefix reuse: the attention KV cache is position-indexed, so restoring
  // the DeltaNet states (and the position) is enough to resume from an earlier point of the sequence.
  void snapshot(int slot) {
    TP_MIRROR(snapshot(slot), TpMirror::kNone, 0);
    if (slot >= (int)snaps_.size()) snaps_.resize(slot + 1);
    Snap& s = snaps_[slot];
    // (DeepSeek-V4.1: its compressed rows are position-indexed like a KV cache; the window rings, the pending
    //  compressor rows and the Engram history are copied, ~10 MB on the real model)
    if (hp.ds41) {
      ds_save(s.ds, s.dsh);
      s.pos = pos;
      return;
    }
    const size_t sb = (size_t)hp.ssm_nv * hp.ssm_d_state * hp.ssm_d_state * 4, cb = conv_bytes();
    if (s.S.empty())
      for (auto& L : layers_)
        if (hp.recurrent(L.il)) { s.S.push_back(buf(sb)); s.conv.push_back(buf(cb)); }
    size_t k = 0;
    for (auto& L : layers_)
      if (hp.recurrent(L.il)) {
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, L.S, s.S[k], 0, 0, sb, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, L.conv_state, s.conv[k], 0, 0, cb, 0, nullptr, nullptr));
        ++k;
      }
    if (has_mtp) {
      if (!s.h) s.h = buf(mtp_hw() * 4);
      flush_mtp_pending();
      CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hlast_, s.h, 0, 0, mtp_hw() * 4, 0, nullptr, nullptr));
    }
    s.pos = pos;
    // Gemma (ring caches): the sliding-window rings' last n_swa positions, in host memory. The global layers' caches
    // keep every position, so with these the cache goes back here after the rings moved on: rewind() reaches only
    // ~4K positions back, and a turn after a long answer or a long tool result (Gemma's template drops the earlier
    // reasoning, so the next request diverges at the turn start) processed the whole conversation again, 42 s for 22K
    // tokens on E4B (2026-10-31)
    // The copy is taken only when the rings are about to overwrite those positions (ring_guard, before every pass):
    // most turns never get that far, and a copy costs (12B: 472 MB and 233 ms; E4B 31 MB, 14 ms). Not for slot 1, the
    // prompt's end (the next request diverges at the model's turn start, slot 0, or within the ~4K rewind reaches).
    s.ring.clear();
    s.ring_ok = false;
    s.ring_want = hp.swa_rings && slot != 1 && pos > 0 && pos >= rewind_floor();
  }
  // Gemma: before a pass writes positions [pos, pos + T) of the bound slot: a snapshot whose rings' window those
  // writes would overwrite gets its copy now (still intact, else none)
  void ring_guard(int T) {
    if (!hp.swa_rings) return;
    for (Snap& s : snaps_) {
      if (s.pos < 0 || !s.ring_want || s.ring_ok) continue;
      if (pos + T - 1 - swa_cap_ < s.pos - hp.n_swa) continue;  // (not reached yet)
      s.ring_want = false;
      if (written_hi_ - 1 - swa_cap_ >= s.pos - hp.n_swa || (swa_lo_ > 0 && s.pos - hp.n_swa < swa_lo_)) continue;  // (gone)
      const auto segs = ring_segs(s.pos);
      s.ring.resize(segs.size());
      for (size_t i = 0; i < segs.size(); ++i) {
        s.ring[i].resize(segs[i].n);
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, segs[i].b, CL_FALSE, segs[i].off, segs[i].n, s.ring[i].data(), 0, nullptr, nullptr));
      }
      clFinish(cl_.queue);
      s.ring_ok = true;
    }
  }
  int ring_cap() const { return swa_cap_; }  // (Gemma: the sliding-window rings' positions)
  int ds_window() const { return hp.ds41 ? ds_.W : 0; }  // (DeepSeek-V4.1: its window ring's rows)
  int snapshot_pos(int slot) const { return slot < (int)snaps_.size() ? snaps_[slot].pos : -1; }
  // a snapshot restore() can use (Gemma: with its rings' copy; one of a loaded conversation has none)
  bool snapshot_usable(int slot) const {
    return snapshot_pos(slot) >= 0 && (!hp.swa_rings || snaps_[slot].ring_ok || snaps_[slot].ring_want);
  }
  size_t snapshot_host_bytes(int slot) const {  // (Gemma: the rings' copy in host memory)
    size_t n = 0;
    if (slot < (int)snaps_.size())
      for (const auto& v : snaps_[slot].ring) n += v.size();
    return n;
  }
  void drop_snapshot(int slot) {
    TP_MIRROR(drop_snapshot(slot), TpMirror::kNone, 0);
    if (slot < (int)snaps_.size()) snaps_[slot].pos = -1;
  }
  bool restore(int slot) {
    TP_MIRROR(restore(slot), TpMirror::kNone, 0);
    if (snapshot_pos(slot) < 0) return false;
    Snap& s = snaps_[slot];
    if (hp.ds41) {  // (see snapshot())
      ds_load(s.ds, s.dsh);
      pos = s.pos;
      return true;
    }
    if (hp.swa_rings && !s.ring_ok) {  // (no copy taken yet: the rings still hold its window, or it cannot be used)
      if (!s.ring_want || !rewind(s.pos)) return false;
      last_drafts_.clear();
    } else if (hp.swa_rings) {  // (the rings' copy back to their slots, the position there: see snapshot())
      const auto segs = ring_segs(s.pos);
      if (segs.size() != s.ring.size()) return false;
      for (size_t i = 0; i < segs.size(); ++i)  // (every size first: nothing written when one does not match)
        if (segs[i].n != s.ring[i].size()) return false;
      for (size_t i = 0; i < segs.size(); ++i)
        CL_CHECK(clEnqueueWriteBuffer(cl_.queue, segs[i].b, CL_FALSE, segs[i].off, segs[i].n, s.ring[i].data(), 0, nullptr, nullptr));
      clFinish(cl_.queue);
      written_hi_ = s.pos;  // (the ring slots past it hold a later conversation's positions now: never rewound into)
      // the rings hold only [pos - n_swa, pos) of this conversation now: the slots below it still hold the abandoned
      // one, so no rewind, snapshot or save may reach below it (rewind() checked written_hi_ alone, which let a later
      // rewind to just before pos read those stale keys; audit 2026-10-31)
      swa_lo_ = std::max(swa_lo_, std::max(0, s.pos - hp.n_swa) / 16 * 16);
      last_drafts_.clear();
    }
    const size_t sb = (size_t)hp.ssm_nv * hp.ssm_d_state * hp.ssm_d_state * 4, cb = conv_bytes();
    size_t k = 0;
    for (auto& L : layers_)
      if (hp.recurrent(L.il)) {
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, s.S[k], L.S, 0, 0, sb, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, s.conv[k], L.conv_state, 0, 0, cb, 0, nullptr, nullptr));
        ++k;
      }
    if (has_mtp) {
      CL_CHECK(clEnqueueCopyBuffer(cl_.queue, s.h, hlast_, 0, 0, mtp_hw() * 4, 0, nullptr, nullptr));
      pend_toks_.clear();
    }
    pos = s.pos;
    return true;
  }

#include "engine_slots.h"  // (sizing the context)
#include "engine_sessions.h"  // (saved conversations)
#include "engine_setup.h"  // (setup)
#include "engine_load.h"  // (the kernels the forward passes use)
#include "engine_stream.h"  // (expert streaming)
#include "engine_load_gemma.h"  // (the Gemma 4 weights)
#include "engine_alloc.h"  // (the activation buffers and the memory plan)
  // ---------------------------------------------------------------- dispatch helpers
  void run(const char* name, cl_kernel k, int dims, const size_t* gws, const size_t* lws) {
    cl_event e = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(cl_.queue, k, dims, nullptr, gws, lws, 0, nullptr, profile ? &e : nullptr));
    if (profile) events_.push_back({prof_group_ >= 0 ? std::string(name) + "@" + std::to_string(prof_group_) : std::string(name), e});
  }
  void collect_profile() {
    if (!profile) return;
    clFinish(cl_.queue);
    for (auto& [n, e] : events_) {
      prof_ms[n] += ClRt::event_ms(e);
      prof_n[n]++;
      clReleaseEvent(e);
    }
    events_.clear();
  }

  void decode_w8(const GpuWeight& w, cl_mem dq, cl_mem dh) { decode_w8(w.raw, w.type, w.N, w.K, dq, dh); }
  void decode_w8(cl_mem raw, uint32_t type, int N, int K, cl_mem dq, cl_mem dh) {
    // (the W8 layout is in groups of 16 rows: a partial last group would write past w8_bytes_q / w8_bytes_h)
    if (N % 16) throw std::runtime_error("decode_w8: N = " + std::to_string(N) + " is not a multiple of 16");
    // big-codebook types decode faster per superblock with the codebook in local memory; the rest per 32-block
    // one work-item per (row, superblock): the superblock is vector-loaded into registers and the codebook staged in
    // local memory (27B, 11K prompt: IQ3_S / IQ4_XS / IQ3_XXS 5.8 -> 2.2 s than a per-32-block kernel)
    static const bool by_type = getenv("ARC_PROF_W8TYPE") != nullptr;  // (profiling: decode time per weight type)
    const std::string nm = by_type ? std::string("decode_w8_") + ggml_type_name(type) : std::string("decode_w8");
    // Q8_0 / Q4_0 with their vector block decoder (vblock, decode_w8_vb): 3x faster than decode_w8_sb. (For the
    // codebook types it was slower: each lane loads its own superblock, 16 cache lines per load instruction.)
    size_t lx = 256;
    while (N % lx) lx /= 2;
    const bool vb = lx >= 64 && (type == GGML_Q8_0 || type == GGML_Q4_0);
    // whole rows of up to 16 superblocks, staged in local memory with coalesced loads (decode_w8_sl): types without
    // mins (Qwen3.6 35B prompt experts, 2048-token chunk: IQ3_XXS 1038 -> 518 ms, IQ4_XS 433 -> 303 ms)
    const int nsb = K / 256;
    const bool sl = !w8_requant(type) && nsb <= 16 && N % 16 == 0 &&
                    16 * nsb * raw_padded_sb_bytes(type) <= 32768 && type != GGML_IQ1_M && type != GGML_IQ1_S;
    if (sl) {
      cl_kernel k = kernel("decode", "-DQT=" + std::to_string(type) + " -DSL_NSB=" + std::to_string(nsb), "decode_w8_sl");
      set_args(k, 0, raw, N, K, tabs_, dq, dh);
      size_t g1 = (size_t)N * nsb, l1 = (size_t)16 * nsb;
      run(nm.c_str(), k, 1, &g1, &l1);
      return;
    }
    cl_kernel k = kernel("decode", "-DQT=" + std::to_string(type), vb ? "decode_w8_vb" : "decode_w8_sb");
    set_args(k, 0, raw, N, K, tabs_, dq, dh);
    size_t gws[2] = {(size_t)N, (size_t)K / 256}, lws[2] = {lx, 1};
    run(nm.c_str(), k, 2, gws, lws);
  }

  // Gemma's final logit soft-capping happens in the output head's GEMV (heads only ever see <= 8 rows).
  std::string head_defs(const GpuWeight& w) const {
    if (&w != &lm_head_ || hp.softcap <= 0.0f || q2_nocap_) return "";
    char b[48];
    snprintf(b, sizeof b, " -DSOFTCAP=%.9ef", hp.softcap);
    return b;
  }
  // gemv_v / gemv_mt_v (zero/kernels/decode.cl: weights loaded straight into registers, bit-identical to gemv): for IQ4_XS,
  // whose codebook decoding made gemv compute-bound, and Q6_K output heads; other types are as fast or faster in gemv.
  static bool gemv_v_ok(const GpuWeight& w) {
    // (measured per type with bench/gemv_bench on the regrouped GPU forms; IQ3_S stays faster in gemv)
    return (w.type == GGML_IQ4_XS || w.type == GGML_Q4_0 || w.type == GGML_IQ3_XXS || w.type == GGML_IQ2_XXS ||
                    w.type == GGML_IQ2_XS || w.type == GGML_IQ2_S || w.type == GGML_IQ1_M || w.type == GGML_IQ1_S || w.type == GGML_Q6_K);
    // (Q6_K: every matrix, not only output heads: 27B MTP block FFN 63-67 -> 71-77 GB/s, attn_output 0.41 -> 0.34 ms,
    //  eh_proj 0.78 -> 0.61 ms; tiny matrices equal; 16 Oct)
  }
  static int gemv_v_rows(const GpuWeight& w) {
    // (tuning; 1-8 rows a sub-group. Audit 2026-11-12 #8: a negative value or one above the matrix's rows gave a grid
    //  of 0 and a launch skipped without a word, the output left as it was: such values are ignored, with a note)
    static const int force = [] {
      const int v = getenv("ARC_GVROWS") ? atoi(getenv("ARC_GVROWS")) : 0;
      if (v < 0 || v > 8) zlog::eprintf("inferzero: ARC_GVROWS=%d ignored (1-8)\n", v);
      return v >= 1 && v <= 8 ? v : 0;
    }();
    // (a per-type rule tuned on single matrices repeated in bench/gemv_bench, e.g. IQ4_XS one row, IQ3_XXS four,
    // was no faster in the model, where each matrix streams once: 4.68 / 4.67 / 4.79 vs 4.68 / 4.51 / 4.65 tok/s)
    if (force && force <= w.N) return force;
    // codebook types (IQ1 / IQ2 / IQ3_XXS) 4 rows where the others take 2: 27B (mostly these types in the FFN) 181.8 ->
    // 179.2 ms a token, every one of 4 alternated runs (2026-11-04; bench/gemv_bench IQ2_XS gate 62.9 -> 69.1 GB/s);
    // bit-identical (a row's sums are the same, only which sub-group computes it changes). ARC_GV4=0: 2 as before
    static const bool gv4 = !getenv("ARC_GV4") || atoi(getenv("ARC_GV4")) != 0;
    if (gv4 && (w.N > 4096 || w.K >= 8192) && w.type != GGML_IQ4_XS && w.type != GGML_Q6_K && w.type != GGML_Q4_0) return 4;
    return (w.N > 4096 || w.K >= 8192) ? 2 : 1;
  }
  // (A/B: an env var set to 256 builds that kernel with the large register file; same results, other speed)
  static std::string grf_env(const char* var) {
    const char* e = getenv(var);
    return e && std::string(e) == "256" ? std::string("-cl-intel-256-GRF-per-thread ") : std::string();
  }
  // C[:, c_off : c_off+N] (+)= act * W^T for the current chunk. A single token uses the raw-row GEMV.
  // acc: 0 = store, 1 = accumulate into C, 2 = C = silu(C) * result (fused SwiGLU, GEMM path only)
  // ---- two int8 passes (Gemma 4, always; QUEUE F21.4): every activation row is stored twice by quant_store16,
  // the int8 values and, at fixed offsets in the same buffers, their rounding residual in int8; a matrix product
  // runs on both and adds them (the f32 reference simulating this: Gemma 4 26B KL 0.0985 -> 0.0053 together with
  // two-pass attention). The offsets come from alloc_buffers' sizes (max_chunk rows of the widest input).
  bool q2_ = false;
  bool frows_ = false;  // (the float K / V rows (f16 K, f16 V) and f32 queries of attn_fa_g: Gemma 4's two-pass mode)
  // (the K rows in f16 (ops.cl -DK_F16): Qwen3 dense (ARC_Q3_KF32=1: f32) and Gemma 4 since 2026-11-11 (ARC_G4_KF16=0: f32))
  bool kf16_ = false;
  size_t kc_bytes() const { return kf16_ ? 2 : 4; }
  // (two-pass matmul inputs: Gemma 4's mode, and Flash Next's dense matmuls with ARC_FN_Q2=1, an F21.13 experiment:
  //  363-token KL against f32 0.0483 -> 0.0384, 4K prompt 191 -> 160 tok/s; its experts stay one pass)
  bool q2mm_ = false;
  // (Gemma 4 E2B / E4B prompts: the global layers' attention on the int8 K / V copy, attn_ps, instead of the float
  //  attention; f32 reference simulation, 480 tokens: KL 0.00081 -> 0.00145 with q / K / V / P rounded as attn_ps
  //  does; 13,921-token prompt: attn_global 11.8 s of 29.7 s GPU. Generation stays float. ARC_G4_ATT8=0: off)
  bool g4_att8_ = false;
  cl_mem qf_ = nullptr, mh2_ = nullptr;
  size_t q2m_hq_ = 0, q2m_sc_ = 0;  // (the experts' down input: residual offsets in mhq_ / mhsc_, bytes)
  bool q2_off_ = false;
  bool q2_nocap_ = false;  // (the output head's two passes run without their softcap epilogue)
  cl_mem fa_part_ = nullptr;  // (attn_fa_g -DFA_SPLIT partials: 8 rows x heads x 64 splits x (head dim + 2) floats)
  bool q2_fuse_ = false, q2_fused_ = false;  // (matmul -> matmul1: run the fused two-pass GEMV; it ran)
  size_t q2_aq_ = 0, q2_sc_ = 0, q2_ab_ = 0;  // bytes
  std::string q2_defs_;
  void q2_init() {
    q2_ = hp.gemma;  // (Gemma 4's only mode since 2026-10-22: KL < 0.01 against f32, the one-pass mode removed)
    if (hp.hc && fn_precise()) q2_ = true;  // (Flash Next's precise mode: two passes, the experts too)
    // (Qwen3 dense: float attention as well, the int8 one is far off: f32 reference, 72-token prompt, q / K / V rounded
    //  to int8 per row KL 0.159, per 32 values 0.123; f16 K / V 0.0066 with the int8 matmul inputs)
    frows_ = (q2_ && hp.gemma) || hp.qwen3;
    // (Gemma 4 E2B / E4B (per-layer embeddings): one int8 pass for the matmuls, the float attention kept; the
    //  user's limit is KL 0.015 and the one-pass E4B measured 0.0010 with int8 attention as well (2026-10-23).
    //  ARC_G4_Q2MM=1: two passes there too, as before)
    // (ARC_G4_ONE=1: one pass on any Gemma 4, to measure its KL)
    const bool g4_one = hp.gemma && ((hp.n_ple > 0 && !(getenv("ARC_G4_Q2MM") && atoi(getenv("ARC_G4_Q2MM")) != 0)) ||
                                     (getenv("ARC_G4_ONE") && atoi(getenv("ARC_G4_ONE")) != 0));
    // (Qwen3 dense, Qwen3-ASR: two passes too. 72-token prompt against f32: one pass KL 0.0064 prompt / 0.0093 decode,
    //  two passes < 0.00001; 2048-token prompt 1376 -> 901 tok/s, generation 24.5 -> 22.2 tok/s. ARC_Q3_Q2=0: one pass)
    const bool q3_one = getenv("ARC_Q3_Q2") && atoi(getenv("ARC_Q3_Q2")) == 0;
    q2mm_ = (q2_ && !g4_one) || (hp.hc && getenv("ARC_FN_Q2") && atoi(getenv("ARC_FN_Q2")) > 0) || (hp.qwen3 && !q3_one);
    g4_att8_ = q2_ && hp.gemma && hp.n_ple > 0 && !(getenv("ARC_G4_ATT8") && atoi(getenv("ARC_G4_ATT8")) == 0);
    if (!q2mm_) return;
    const size_t M = max_chunk;
    const int maxK = aq_max_k();
    q2_aq_ = M * maxK;
    q2_sc_ = M * (maxK / 32) * 4;
    q2_ab_ = M * (maxK / 16) * 2;
    q2_defs_ = " -DQ2_AQ=" + std::to_string(q2_aq_) + " -DQ2_SC=" + std::to_string(q2_sc_ / 4) + " -DQ2_AB=" +
               std::to_string(q2_ab_ / 2);

  }
  // the longest row written into aq_ (the shared expert's input, and the MTP head's 2 * n_embd, count too)
  int aq_max_k() const {
    return std::max({hp.n_embd, 2 * hp.n_embd, hp.n_ff, hp.ssm_d_inner, hp.n_head * hp.head_dim, hp.hce(), hp.n_ff_exp_pad, hp.n_ff_shexp});
  }
  void matmul(const char* name, const GpuWeight& w, int Mp, cl_mem C, int ldc, int c_off, int acc, bool act32) {
    // (the output head with a logit softcap: the cap applies to the sum of both passes, not to each; the head GEMVs
    //  cap in their epilogue, so both run without it and softcap_rows caps the sum)
    const bool cap = q2mm_ && !q2_off_ && &w == &lm_head_ && hp.softcap > 0.0f;
    q2_nocap_ = cap;
    // (one token: the GEMV runs both passes over one read of the weights, -DQ2F; it says so in q2_fused_)
    q2_fuse_ = q2mm_ && !q2_off_ && cur_T_ <= 4;  // (2-4 tokens: gemv_mt_v only, see matmul1)
    q2_fused_ = false;
    matmul1(name, w, Mp, C, ldc, c_off, acc, act32);
    q2_fuse_ = false;
    if (!q2mm_ || q2_off_) return;
    if (acc > 1) throw std::logic_error(std::string("two-pass activations with a fused epilogue: ") + name);
    if (!q2_fused_) {
      const cl_mem a = aq_, s = asc_, b = absum_;
      aq_ = from(a, q2_aq_);
      asc_ = from(s, q2_sc_);
      absum_ = from(b, q2_ab_);
      matmul1(name, w, Mp, C, ldc, c_off, 1, act32);
      aq_ = a;
      asc_ = s;
      absum_ = b;
    }
    if (cap) {
      q2_nocap_ = false;
      if (cur_T_ > 8) throw std::logic_error("two-pass output head over more than 8 rows (no softcap there)");
      cl_kernel k = kernel("ops", "", "softcap_rows");
      int ld = cur_T_ == 1 ? 0 : ldc, off = c_off, N = w.N;
      float cp = hp.softcap;
      set_args(k, 0, C, ld, off, N, cp);
      size_t gws[2] = {((size_t)N + 255) / 256 * 256, (size_t)cur_T_};
      run(name, k, 2, gws, nullptr);
    }
  }
  // One token's GEMV of w: gemv_v (vk) or gemv, and its rows per sub-group (matmul1 and gemv_gu alike).
  // (ARC_GV_IQ3S=1, zero/xe2: IQ3_S through gemv_v too, its weights straight into registers instead of gemv's local
  //  window with two barriers a step; bit-identical. Slower on the 125H's iGPU (bench/gemv_bench), unmeasured on a
  //  B60, whose 27B GEMV reads ~55 % of its bandwidth)
  // (Qwen3-ASR's Q8_0 matrices, 2026-10-28: 1 / 2 / 4 / 8 rows for the stacked gate + up (12,288 rows), qkv
  //  (4,096), o / down (2,048) and the head all within 1 % in the model: the rule kept)
  static int gemv1_rows(const GpuWeight& w, bool& vk) {
    static const bool gv_iq3s = getenv("ARC_GV_IQ3S") && atoi(getenv("ARC_GV_IQ3S")) != 0;
    vk = gemv_v_ok(w) || (gv_iq3s && w.type == GGML_IQ3_S);
    return vk ? gemv_v_rows(w) : w.N <= 4096 ? 1 : w.N <= 8192 ? 2 : w.N <= 65536 ? 4 : 2;
  }
  // One token: the FFN's gate and up GEMVs in one launch (decode.cl -DGU2, QUEUE 33.5): rows [0, F) of gate and
  // [0, F) of up at C + F, each row computed by the kernel, grows and register file of its own launch, bit for bit;
  // one launch of 2F rows instead of two of F (a B60's 27B: 4,096 FFN launches of ~135 us per 32 steps). Off by
  // default until measured: ARC_GEMV_GU=1. false: not taken (two launches as before)
  bool gemv_gu(const GpuWeight& g, const GpuWeight& u, int Mp, cl_mem C) {
    static const bool on = getenv("ARC_GEMV_GU") && atoi(getenv("ARC_GEMV_GU")) != 0;
    if (!on || cur_T_ != 1 || q2mm_ || !g.raw || !u.raw || g.type != u.type || g.N != u.N || g.K != u.K ||
        &g == &lm_head_ || &u == &lm_head_)
      return false;
    bool vk = false;
    const int grows = gemv1_rows(g, vk);  // (matmul1's kernel and rows for one of the matrices)
    if (g.N % grows) return false;  // (the up rows must start a sub-group of their own)
    int N = g.N, K = g.K, M = Mp, c_off = 0, a = 0;
    rd_bytes += raw_bytes(g) + raw_bytes(u);
    cl_kernel k = kernel("decode", xe2_gemv_grf() + "-DQT=" + std::to_string(g.type) + " -DGROWS=" + std::to_string(grows) +
                                       head_defs(g) + (vk ? std::string() : xe2_gemv_pf()) + " -DGU2",
                         vk ? "gemv_v" : "gemv");
    set_args(k, 0, g.raw, N, K, tabs_, aq_, asc_, absum_, M, C, c_off, a, u.raw);
    const size_t sgs = (size_t)N / grows, l1 = vk ? 256 : xe2_gemv_wg(sgs);  // (the work-group of one matrix's launch)
    size_t g1 = (2 * sgs * 16 + l1 - 1) / l1 * l1;
    run("gemm_gate_up", k, 1, &g1, &l1);
    return true;
  }
  void matmul1(const char* name, const GpuWeight& w, int Mp, cl_mem C, int ldc, int c_off, int acc, bool act32) {
    int a = acc;
    if (cur_T_ == 1) {
      // rows per sub-group: enough sub-groups for small N; very tall matrices (an lm_head of short rows) do better
      // with more sub-groups again (gemv1_rows)
      bool vk = false;
      const int grows = gemv1_rows(w, vk);
      int N = w.N, K = w.K, M = Mp;
      rd_bytes += raw_bytes(w);
      if (q2_fuse_ && a > 1) throw std::logic_error(std::string("fused two-pass GEMV with an epilogue: ") + name);
      cl_kernel k = kernel("decode", xe2_gemv_grf() + "-DQT=" + std::to_string(w.type) + " -DGROWS=" + std::to_string(grows) + head_defs(w) +
                                         (q2_fuse_ ? " -DQ2F" + q2_defs_ : std::string()) + (vk ? std::string() : xe2_gemv_pf()),
                           vk ? "gemv_v" : "gemv");
      q2_fused_ = q2_fuse_;
      set_args(k, 0, w.raw, N, K, tabs_, aq_, asc_, absum_, M, C, c_off, a);
      // (gemv_v stages its tables with the first 64 items of a work-group: it keeps 256)
      const size_t sgs = (size_t)(N + grows - 1) / grows, l1 = vk ? 256 : xe2_gemv_wg(sgs);
      size_t g1 = (sgs * 16 + l1 - 1) / l1 * l1;
      run(name, k, 1, &g1, &l1);
      return;
    }
    if (cur_T_ > 1 && cur_T_ <= 8) {
      // small batch (e.g. speculative verification): read each weight once for all tokens
      // token-outer kernels (each token's activations loaded once for all of a sub-group's rows, no per-token
      // branches; bit-identical to the earlier row-outer ones), built for exactly T tokens with 256 registers.
      // 27B, 8 rows (bench/gemv_bench): IQ3_XXS gate 23.4 -> 42.9 GB/s with 4 rows per sub-group, IQ3_S down (now
      // gemv_mt_v) 22.6 -> 38.9 with 2; Q6_K one row (it spills with more)
      const int tb = cur_T_;
      // (Qwen3-ASR's Q8_0 in two-pass mode, 2-4 rows (batched parts): gemv_mt_v, both passes over one read of the
      //  weights (gemv_mt has no -DQ2F: two launches, every matrix read twice))
      const bool vk = gemv_v_ok(w) || w.type == GGML_IQ3_S || (hp.qwen3 && w.type == GGML_Q8_0 && q2_fuse_ && a <= 1);
      int gr = vk ? gemv_v_rows(w) : w.N <= 4096 ? 1 : w.N <= 8192 ? 2 : 4;
      // Below 6 tokens (MTP verification of prose: 2-3 rows) 4 rows per sub-group for the codebook types and for long
      // rows (K >= 8192, the FFN down projection), Q4_K / IQ4_XS short rows 2: 27B, bench/gemv_bench, 3 tokens: gate
      // IQ3_XXS 57.8 -> 64.7 GB/s, down Q4_K 57.5 -> 70.6; Indonesian prose with MTP 6.95 -> 7.2 tok/s. With 6-8
      // tokens the earlier rule stays: there 4 for the down projection and 2 for gate / up were faster only in
      // bench/gemv_bench (code with 7 drafts or DFlash2 2 % slower in the model). Bit-identical either way.
      {
        const bool cb = vk && w.type != GGML_IQ4_XS;
        if (w.type == GGML_Q6_K) gr = 1;
        else if (cur_T_ < 6 && (vk || w.type == GGML_Q4_K)) gr = w.K >= 8192 || cb ? 4 : 2;
        else gr = w.type == GGML_IQ3_S ? 2 : vk ? (w.N >= 8192 ? 4 : 2) : std::min(gr, 2);
      }
      // (two-pass mode, 2-4 tokens: gemv_mt_v with the residuals as TB further slots, -DQ2F, both passes in one launch)
      const bool fuse = q2_fuse_ && vk && a <= 1;
      cl_kernel k = kernel("decode", xe2_gemv_mt_grf() + "-DQT=" +
                                         std::to_string(w.type) + " -DGROWS=" + std::to_string(gr) + " -DTB=" +
                                         std::to_string(fuse ? 2 * tb : tb) + head_defs(w) +
                                         (fuse ? " -DQ2F" + q2_defs_ : std::string()),
                           vk ? "gemv_mt_v" : "gemv_mt");
      q2_fused_ = fuse;
      int N = w.N, K = w.K, M = Mp, T = cur_T_;
      rd_bytes += raw_bytes(w);
      set_args(k, 0, w.raw, N, K, tabs_, aq_, asc_, absum_, M, T, C, ldc, c_off, a);
      size_t g1 = ((size_t)(N + gr - 1) / gr * 16 + 255) / 256 * 256, l1 = 256;
      run(name, k, 1, &g1, &l1);
      return;
    }
    // (ARC_H8=1, zero/engine_h8.h; the 140T preset: H8 only where it takes the GEMM, p140_take)
    const bool pt = p140_take(w, Mp);
    if (h8_ok(w) && (!p140_ || h8_env() || pt)) return h8_matmul(name, w, Mp, C, ldc, c_off, a, act32);
    cl_mem wq = w.wq, wh = w.wh;
    if (!wq) {
      wq = stage_q_[w.role];
      wh = stage_h_[w.role];
      // (still decoded from the previous chunk otherwise; the 140T preset's H8 chunks leave the rotated form there)
      if (stage_owner_[w.role] != &w || stage_h8_[w.role]) {
        decode_w8(w, wq, wh);
        stage_owner_[w.role] = &w;
        stage_h8_[w.role] = false;
      }
    }
    if (x16_ && x16_matmul(name, w, wq, wh, Mp, C, ldc, c_off, a, act32)) return;  // (zero/xe2/engine_xe2.h)
    // (the 140T preset: XMX with 32 tokens, 2 column groups and 128 registers where it takes the GEMM, else the DP4a
    //  default: 16 tokens, 256 registers, one column group)
    const int cg = pt ? 2 : p140_ ? 1 : xe2_cg_for(w.N);
    cl_kernel k = pt ? gemm_kernel(w.gemm_qtype, act32, 1, 2, 32, 1)
                     : p140_ ? gemm_kernel(w.gemm_qtype, act32, 0, 1, 16, 0) : gemm_kernel(w.gemm_qtype, act32, -1, cg);
    int M = Mp, N = w.N, K = w.K;
    // Long reductions are split into ~17-superblock pieces (the activation slab of a token-row group then stays in
    // L3); later pieces accumulate into C. Token rows are swept together per weight slice (rgroup).
    static const int rg_env = getenv("ARC_GEMM_RG") ? std::max(1, atoi(getenv("ARC_GEMM_RG"))) : 4;
    static const int wgy_max = getenv("ARC_GEMM_WGY") ? atoi(getenv("ARC_GEMM_WGY")) : 8;
    const int tm = pt ? 32 : p140_ ? 16 : gemm_tm();
    // (the grid covers whole 16-column groups and tm-token tiles: other shapes would leave rows or columns out)
    if (Mp % tm || N % 16 || K % 256)
      throw std::runtime_error(std::string("gemm ") + name + ": shape " + std::to_string(Mp) + " x " + std::to_string(N) + " x " +
                               std::to_string(K) + " is not a multiple of " + std::to_string(tm) + " x 16 x 256");
    // (ARC_GEMM_SPLIT: superblocks per split-K piece, default 17; 0: no split-K. Not bit-identical to one pass: the
    //  pieces' float sums are added in another order. 27B, 600-token prompt + 8 tokens, against ARC_GEMM_SPLIT=0:
    //  KL mean 0.00067, max 0.0017, the same greedy tokens (laptop, 2026-11-12, audit "needs a GPU" #1))
    static const int sk_sb = getenv("ARC_GEMM_SPLIT") ? std::max(0, atoi(getenv("ARC_GEMM_SPLIT"))) : 17;
    // (the fused SwiGLU needs the whole sum in one piece: silu(gate) * (u1 + u2), not silu(gate) * u1 + u2; no model
    //  fuses it over rows longer than 8192 yet)
    const int nsb = K / 256, splits = a != 2 && K > 8192 && sk_sb > 0 ? (nsb + sk_sb - 1) / sk_sb : 1, rgroup = rg_env;
    int wgy = 1;
    while (wgy < wgy_max && (Mp / tm) % (wgy * 2) == 0) wgy *= 2;
    size_t gws[2] = {(size_t)N * (Mp / tm / wgy), (size_t)wgy}, lws[2] = {16 * (size_t)cg, (size_t)wgy};
    for (int sp = 0; sp < splits; ++sp) {
      int s0 = nsb * sp / splits, s1 = nsb * (sp + 1) / splits, ac = sp == 0 ? a : 1;
      set_args(k, 0, wq, wh, aq_, asc_, absum_, C, M, N, K, ldc, c_off, ac, rgroup, s0, s1);
      run(name, k, 2, gws, lws);
    }
  }

  // Activation layout for a GEMM input: per-32 scales for sensitive inputs, the single-token GEMV, or on request.
  int act32_for(bool sensitive) const { return (cur_T_ <= 8 || sensitive || opt.act32_all) ? 1 : 0; }

  void rmsnorm_quant(cl_mem x, int row_off, cl_mem w, int K, int Mp, int T, int act32) {
    float eps = hp.eps;
    int wh = 0;
    set_args(k_rmsnorm_, 0, x, row_off, w, K, Mp, T, eps, aq_, asc_, absum_, big_, wh, act32);
    size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
    if (wide_rows(T)) {  // (a work-group per 16 superblocks, each summing the row itself; the valid rows only)
      gws[0] = 256 * (size_t)((K / 256 + 15) / 16);
      gws[1] = (size_t)T;
    }
    run("rmsnorm_quant", k_rmsnorm_, 2, gws, lws);
  }

  // ---------------------------------------------------------------- forward pieces
  void embed(const int* tokens, int T, int Mp) {
    cur_T_ = T;
    attn_in_ready_ = ffn_in_ready_ = false;
    const size_t row_bytes = tok_embd_->nbytes / hp.n_vocab;
    if (hp.hc || hp.ds41) {  // Flash Next, DeepSeek-V4.1: hc_n copies of the embedding per row
      const int W = hp.hce(), E = hp.n_embd;
      std::fill(x_host_.begin(), x_host_.begin() + (size_t)Mp * W, 0.0f);
#pragma omp parallel for if (T > 64)
      for (int t = 0; t < T; ++t) {
        if (tokens[t] < 0 || tokens[t] >= hp.n_vocab) continue;
        float* r = &x_host_[(size_t)t * W];
        dequant_row(tok_embd_->type, tok_embd_->data + (size_t)tokens[t] * row_bytes, E, r);
        for (int c = 1; c < hp.hc_n; ++c) memcpy(r + (size_t)c * E, r, (size_t)E * 4);
      }
      const float* const* med = media_rows(tokens, T);
      if (med)
        for (int t = 0; t < T; ++t)
          if (med[t])
            for (int c = 0; c < hp.hc_n; ++c) std::copy(med[t], med[t] + E, &x_host_[(size_t)t * W + (size_t)c * E]);
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, x_, CL_TRUE, 0, (size_t)Mp * W * 4, x_host_.data(), 0, nullptr, nullptr));
      if (hp.ds41) ds_prepare(tokens, T);
      if (hp.ple_layer >= 0) {
        if (med) {  // (an image's rows: <|image_pad|> in the n-gram hash, as forward_embd)
          std::vector<int> ids(tokens, tokens + T);
          for (int t = 0; t < T; ++t)
            if (med[t]) ids[t] = hp.ple_img >= 0 ? hp.ple_img : (int)hp.ple_eos;
          ple_embed(ids.data(), T);
        } else {
          ple_embed(tokens, T);
        }
      }
      return;
    }
    std::fill(x_host_.begin(), x_host_.begin() + (size_t)Mp * hp.n_embd, 0.0f);
    const float* const* med = media_rows(tokens, T);
#pragma omp parallel for if (T > 64)
    for (int t = 0; t < T; ++t) {
      if (med && med[t]) {  // (an image's row as given, as forward_embd copies it)
        std::copy(med[t], med[t] + hp.n_embd, &x_host_[(size_t)t * hp.n_embd]);
        continue;
      }
      if (tokens[t] < 0 || tokens[t] >= hp.n_vocab) continue;
      dequant_row(tok_embd_->type, tok_embd_->data + (size_t)tokens[t] * row_bytes, hp.n_embd,
                  &x_host_[(size_t)t * hp.n_embd]);
      if (hp.gemma) {  // Gemma scales the embeddings by sqrt(n_embd)
        const float sc = std::sqrt((float)hp.n_embd);
        for (int i = 0; i < hp.n_embd; ++i) x_host_[(size_t)t * hp.n_embd + i] *= sc;
      }
      if (hp.n_ple) {  // per-layer token embeddings, scaled by sqrt(n_ple)
        const size_t np = (size_t)hp.n_ple * hp.n_layer, rb = ple_tok_embd_->nbytes / hp.n_vocab;
        float* dst = &ple_host_[(size_t)t * np];
        dequant_row(ple_tok_embd_->type, ple_tok_embd_->data + (size_t)tokens[t] * rb, (int)np, dst);
        const float sc = std::sqrt((float)hp.n_ple);
        for (size_t i = 0; i < np; ++i) dst[i] *= sc;
      }
    }
    // blocking: x_host_ is refilled by the next chunk while this write may still be queued behind earlier work
    CL_CHECK(clEnqueueWriteBuffer(cl_.queue, x_, CL_TRUE, 0, (size_t)Mp * hp.n_embd * 4, x_host_.data(), 0, nullptr,
                                  nullptr));
    if (hp.n_ple) {
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, ple_tok_, CL_TRUE, 0, (size_t)T * hp.n_ple * hp.n_layer * 4,
                                    ple_host_.data(), 0, nullptr, nullptr));
      ple_inputs(T, Mp);
    }
  }

  int prof_group_ = -1;  // ARC_PROF_LAYERS=n: profile names carry the layer group (il / n)
  void run_layer(Layer& L, int T, int Mp) {
    static const int pg = getenv("ARC_PROF_LAYERS") ? atoi(getenv("ARC_PROF_LAYERS")) : 0;
    prof_group_ = pg > 0 ? L.il / pg : -1;
    struct Reset { int& g; ~Reset() { g = -1; } } reset{prof_group_};
    if (hp.gemma) {
      layer_gemma_attn(L, T, Mp);
      layer_gemma_ffn(L, T, Mp);
      return;
    }
    if (hp.ds41) return ds_layer(L, T, Mp);
    if (hp.hc) {  // Flash Next: each block reads the mixed streams and adds its output to every stream
      if (L.il == hp.ple_layer) ple_block(L, T, Mp);
      hc_mix(L.hc_norm[0], L.hc_down[0], L.hc_up[0], T, Mp, act32_for(false), 0);
      if (hp.recurrent(L.il)) layer_gdn(L, T, Mp);
      else layer_attn(L, T, Mp);
      hc_combine(L.hc_down[0].N, T);
      if (dbg_rows) dbg_row(T - 1);
      hc_mix(L.hc_norm[1], L.hc_down[1], L.hc_up[1], T, Mp, 1, 0);
      static const int dbg_mix = getenv("ARC_DBG_MOE") ? atoi(getenv("ARC_DBG_MOE")) : -1;
      if (L.il == dbg_mix && T > 1) {  // (debug: the FFN mixer's output and inject logits of the last row)
        const int E = hp.n_embd, LD = L.hc_down[1].N;
        std::vector<float> a(E), b(LD);
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, hco_, CL_TRUE, (size_t)(T - 1) * E * 4, E * 4, a.data(), 0, nullptr, nullptr));
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, hc_lo_, CL_TRUE, (size_t)(T - 1) * LD * 4, LD * 4, b.data(), 0, nullptr, nullptr));
        FILE* f = fopen("mix_dbg_gpu.bin", "wb");
        fwrite(a.data(), 4, E, f); fwrite(b.data() + hp.hc_lr, 4, hp.hc_n, f); fwrite(b.data(), 4, hp.hc_lr, f);
        fclose(f);
      }
      layer_moe(L, T, Mp);
      hc_combine(L.hc_down[1].N, T);
      if (dbg_rows) dbg_row(T - 1);
      if (dfl_) dfl_capture(L, T, Mp);
      return;
    }
    if (hp.recurrent(L.il)) layer_gdn(L, T, Mp);
    else layer_attn(L, T, Mp);
    if (dbg_rows) dbg_row(T - 1);
    if (hp.moe) layer_moe(L, T, Mp);
    else layer_ffn(L, T, Mp);
    if (dbg_rows) dbg_row(T - 1);
    if (dfl_) dfl_capture(L, T, Mp);
  }
  // (debug: the residual stream's row r after each half of every layer, to compare two decode paths)
 public:
  bool dbg_rows = false;
  std::vector<std::vector<float>> dbg_;
 private:
  void dbg_row(int r) {
    std::vector<float> v(hp.hce());
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, x_, CL_TRUE, (size_t)r * hp.hce() * 4, v.size() * 4, v.data(), 0, nullptr,
                                 nullptr));
    dbg_.push_back(std::move(v));
  }

#include "engine_gemma.h"  // (Gemma 4 layers)

  // x_ = (x_ + rmsnorm(big_) * w) * s over the first T rows, then the next GEMV input from the new x_ (rmsnorm * w2,
  // or plain with norm2 = false) into aq_ / asc_ / absum_
  void add_norm_quant(cl_mem w, int T, int Mp, float s, cl_mem w2, bool norm2, int act32) {
    // (registers for the row's superblocks of each sub-group only: ANQ_MAX 4 spilled 3.7 KB per thread)
    if (!k_add_nq_) k_add_nq_ = kernel("ops", "-DANQ_MAX=" + std::to_string((hp.n_embd / 256 + 15) / 16), "add_norm_quant");
    int E = hp.n_embd, nrm = norm2 ? 1 : 0, MM = Mp, a32 = act32;
    float eps = hp.eps, sc = s;
    set_args(k_add_nq_, 0, x_, big_, w, E, eps, sc, w2, nrm, MM, aq_, asc_, absum_, a32);
    size_t gws[2] = {256, (size_t)T}, lws[2] = {256, 1};
    run("add_rmsnorm", k_add_nq_, 2, gws, lws);
  }
  // A 1-8 token pass's row kernels that quantize for the next GEMV (swiglu_quant on the Qwen and MoE paths,
  // rmsnorm_quant, gated_norm_quant, attn_gate_quant): their superblocks over many work-groups, the valid rows only
  // (the GEMVs read only those), instead of one work-group per padded row, where one Xe core did a whole row: a Pro
  // B60's 27B spent 15 us on each of 64 swiglu_quant a token and 7 us on each of 129 rmsnorm_quant (issue #4,
  // 2026-11-09). The same arithmetic per superblock: bit-identical. ARC_WIDE_ROWS=0: as before 2026-11-10
  bool wide_rows(int T) const {
    static const bool on = !getenv("ARC_WIDE_ROWS") || atoi(getenv("ARC_WIDE_ROWS")) != 0;
    return on && T <= 8 && cur_T_ <= 8;
  }
  // a row kernel of K columns (arguments set) with one sub-group per superblock of the T valid rows
  void run_rows_wide(const char* name, cl_kernel k, int K, int T) {
    size_t gws[2] = {(size_t)16 * (K / 256), (size_t)T}, lws[2] = {16, 1};
    run(name, k, 2, gws, lws);
  }
  // launches k_swiglu_ (arguments set) over K columns: a work-group per padded row for prompt chunks, one sub-group
  // per superblock of the T valid rows for single-token passes (the GEMVs read only those rows)
  void run_rows_quant(const char* name, int K, int T, int Mp) {
    if (T <= 8) {
      size_t gws[2] = {(size_t)16 * (K / 256), (size_t)T}, lws[2] = {16, 1};
      run(name, k_swiglu_, 2, gws, lws);
    } else {
      size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
      run(name, k_swiglu_, 2, gws, lws);
    }
  }
  // RoPE table of the current rows (pos .. pos + T - 1) for the sliding (0) or global (1) layers, computed once per
  // pass (T <= 8) or per chunk
  cl_mem rope_table(int type, int hd, float base, cl_mem ff, int T) {
    if (rt_pos_[type] == pos && rt_T_[type] == T) return rope_tab_[type];
    if (!k_rope_tab_) k_rope_tab_ = kernel("ops", "", "rope_table");
    int h = hd, p0 = pos, uff = ff ? 1 : 0;
    float b = base;
    set_args(k_rope_tab_, 0, rope_tab_[type], h, p0, b, ff ? ff : rope_tab_[type], uff);
    size_t gws[2] = {(size_t)hd / 2, (size_t)T};
    run("attn_prep", k_rope_tab_, 2, gws, nullptr);
    rt_pos_[type] = pos;
    rt_T_[type] = T;
    return rope_tab_[type];
  }

  // quantizes rows of an f32 matrix (K columns) for a GEMM / GEMV input, without normalization
  void quant_rows(cl_mem src, int K, int Mp, int T, int act32) {
    int KK = K, MM = Mp, TT = T, a32 = act32, fused = 1, gelu = 0;
    set_args(k_swiglu_, 0, src, KK, MM, TT, aq_, asc_, absum_, a32, fused, gelu);
    size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
    run("quant", k_swiglu_, 2, gws, lws);
  }

  // Gemma 4 E2B/E4B: per-layer inputs of the current chunk = combination of a per-layer token embedding and a
  // projection of the (scaled) token embedding. ple_tok_ was uploaded by embed().
  void ple_inputs(int T, int Mp) {
    const int E = hp.n_embd, NP = hp.n_ple * hp.n_layer, a32 = act32_for(false);
    quant_rows(x_, E, Mp, T, a32);
    matmul("gemm_ple_model", ple_model_proj_, Mp, ple_, NP, 0, 0, a32);
    if (!k_ple_comb_) k_ple_comb_ = kernel("ops", "", "ple_combine");
    int nl = hp.n_layer, TT = T;
    float pre = 1.0f / std::sqrt((float)E), eps = hp.eps, post = 1.0f / std::sqrt(2.0f);
    set_args(k_ple_comb_, 0, ple_, ple_tok_, ple_proj_norm_, nl, TT, pre, eps, post);
    size_t gws[2] = {(size_t)16 * hp.n_layer, (size_t)T}, lws[2] = {16, 1};
    run("ple", k_ple_comb_, 2, gws, lws);
  }

  // x_ = (x_ + rmsnorm(big_) * w) * s over the first T rows (Gemma sandwich norm + residual + layer scale)
  void add_rmsnorm(cl_mem w, int T, float s) {
    if (!k_add_norm_) k_add_norm_ = kernel("ops", "", "add_rmsnorm");
    int E = hp.n_embd;
    float eps = hp.eps, sc = s;
    set_args(k_add_norm_, 0, x_, big_, w, E, eps, sc);
    size_t gws[2] = {256, (size_t)T}, lws[2] = {256, 1};
    run("add_rmsnorm", k_add_norm_, 2, gws, lws);
  }

  // Flash-decoding of query rows 0..T-1 (positions pos0..pos0+T-1, keys up to each row's position, from
  // pos - win + 1 with a window) of one layer into o_. Keys are split so that at least ~512 sub-groups run. When a
  // work-group of 128+ items can cover the q heads of up to 4 rows (attn_split), each K/V tile goes through SLM once
  // for all of them; otherwise (few q heads per kv head) rows go one by one through attn_decode, whose head slices
  // share the reads through L1.
  void attn_decode_rows(const char* name, Layer& L, int cap, int nkv, int hpg, int hd, int pos0, int T, int win,
                        int n_q_heads = 0) {
    // the Gemma 4 assistant drafter's layers (n_q_heads given) keep the original kernels and split choices
    const bool drafter = n_q_heads != 0;
    // heads per sub-group: one at HD 512 (registers); else as the prompt kernels. (Gemma 4's own layers run their
    // float attention in layer_gemma_attn; only its assistant drafter comes here)
    const int hs = hd >= 512 ? 1 : attn_hs_for(hpg), nsp = hpg / hs;
    // (experiments: ARC_ATTN_DEC_GRF=128 builds the decode attention kernels with the default register file, twice
    //  the threads per EU; ARC_ATTN_DEC_SG sets the sub-groups the key split aims for, default 512)
    static const bool dec_small = [] {
      const char* e = getenv("ARC_ATTN_DEC_GRF");
      return e != nullptr && std::string(e) == "128";
    }();
    const std::string dec_grf = dec_small ? std::string() : std::string("-cl-intel-256-GRF-per-thread ");
    static const int dec_sg = getenv("ARC_ATTN_DEC_SG") ? atoi(getenv("ARC_ATTN_DEC_SG")) : 512;
    static const int dec_smax = getenv("ARC_ATTN_DEC_SMAX") ? (std::max(1, atoi(getenv("ARC_ATTN_DEC_SMAX"))) + 15) / 16 * 16 : 256;  // (keys per split, most)
    auto opts = [&](int tpw) {
      return dec_grf + "-DHD=" + std::to_string(hd) + " -DHPG=" + std::to_string(hpg) +
             " -DHS=" + std::to_string(hs) + " -DTPW=" + std::to_string(tpw) + sb_opt(hd);
    };
    // split length: the longest (fewest partials to combine, up to smax) that still gives `target` sub-groups
    auto pick_split = [&](int lo, int plast, int sg_per_split, int smax, int target, int& S, int& kstart, int& ns) {
      S = smax;
      while (S > 32 && ((plast - lo / S * S) / S + 1) * nkv * sg_per_split < target) S /= 2;
      kstart = lo / S * S;
      ns = (plast - kstart) / S + 1;
    };
    int nn = n_q_heads ? n_q_heads : hp.n_head, hdd = hd, cp = cap, w = win;
    for (int r0 = 0; r0 < T;) {
      const int left = T - r0;
      // Qwen3.6 35B (8 q heads per kv head, 256-dim heads, one scale per row): attn_dsplit, 4 q heads per sub-group
      // and the dims split over 4 sub-groups, one row per launch as attn_decode before: bench/attn_dec_bench at 64K
      // 1.16 -> 0.94 ms per layer. The keys split in fixed pieces of 256 from position 0 and each row's P bound covers
      // only its own keys (-DROWVMT), so a verification row gets exactly the values of decoding its token alone.
      // 4B / 9B (4 per kv head, +7 %) and 27B (6, +4 %) keep attn_decode / attn_split, which read the K/V of up to 4
      // verification rows once. ARC_QWEN_DSPLIT=0: never, =1: every Qwen model (2 heads per sub-group, 2 rows).
      static const int qwen_ds = [] {
        const char* e = getenv("ARC_QWEN_DSPLIT");
        return e ? atoi(e) : -1;
      }();
      if (!drafter && !hp.gemma && hd == 256 && hpg % 2 == 0 && (qwen_ds == 1 || (qwen_ds < 0 && hpg >= 8))) {
        const int hg = hpg % 4 == 0 && qwen_ds != 1 ? 4 : 2, n = hg == 4 ? 1 : std::min(left, 2), tr = n, nsub = 4;
        const int sgs = hpg / hg * nsub, lsz = 16 * sgs;
        const int lo = win ? std::max(0, pos0 + r0 - win + 1) : 0, plast = pos0 + r0 + n - 1;
        static const int qs_len = getenv("ARC_QWEN_DS_S") ? (std::max(1, atoi(getenv("ARC_QWEN_DS_S"))) + 15) / 16 * 16 : 256;  // (experiment)
        const int S = qs_len, kstart = lo / S * S, ns = (plast - kstart) / S + 1;
        int p0 = pos0 + r0, nv = n, q0 = r0, SS = S, ks = kstart, nss = ns;
        cl_kernel k = kernel("ops", opts(1) + " -DHG=" + std::to_string(hg) + " -DNSUB=" + std::to_string(nsub) +
                                        " -DTR=" + std::to_string(tr) + " -DROWVMT", "attn_dsplit");
        set_args(k, 0, qi_, qsc_, L.kq, L.ksc, L.vq, L.vsc, cp, nn, p0, nv, apart_, w, ks, SS, nss, q0);
        size_t gws[2] = {(size_t)ns * lsz, (size_t)nkv}, lws[2] = {(size_t)lsz, 1};
        run(name, k, 2, gws, lws);
        for (int m = 0; m < n; ++m) {
          int orow = r0 + m, prow = m;
          set_args(k_attn_comb_, 0, apart_, nss, nn, o_, orow, hdd, prow);
          size_t g1 = (size_t)nn * hd, l1 = 64;
          run("attn_combine", k_attn_comb_, 1, &g1, &l1);
        }
        r0 += n;
        continue;
      }
      int tpw = 1;
      // (A single token takes attn_decode when 16 * nsp < 128, and attn_split over several rows splits the keys
      // differently (its split length follows the rows' sub-groups), so MTP / DFlash2 output drifted from plain
      // decoding once the context needs more than one split: Qwen3.6 35B after ~40 tokens of a 2K prompt, Qwen3.8
      // 27B from the first verified token after 1,097 (logits 1.39 apart). Verification rows of every Qwen model now
      // go through attn_decode one by one, as single tokens do.)
      if (16 * nsp >= 128 || drafter)
        while (tpw < T - r0 && tpw < 4 && 16 * nsp * tpw * 2 <= 512) tpw *= 2;  // apart_ holds 4 rows of partials
      const int n = std::min(tpw, T - r0);
      if (16 * nsp * tpw >= 128) {
        const int lsz = 16 * nsp * tpw, plast = pos0 + r0 + n - 1, lo = win ? std::max(0, pos0 + r0 - win + 1) : 0;
        int S, kstart, ns, p0 = pos0 + r0, nv = n, q0 = r0;
        pick_split(lo, plast, lsz / 16, dec_smax, dec_sg, S, kstart, ns);
        cl_kernel k = kernel("ops", opts(tpw), "attn_split");
        set_args(k, 0, qi_, qsc_, L.kq, L.ksc, L.vq, L.vsc, cp, nn, p0, nv, apart_, w, kstart, S, ns, q0);
        size_t gws[2] = {(size_t)ns * lsz, (size_t)nkv}, lws[2] = {(size_t)lsz, 1};
        run(name, k, 2, gws, lws);
        for (int m = 0; m < n; ++m) {
          int orow = r0 + m, prow = m;
          set_args(k_attn_comb_, 0, apart_, ns, nn, o_, orow, hdd, prow);
          size_t g1 = (size_t)nn * hd, l1 = 64;
          run("attn_combine", k_attn_comb_, 1, &g1, &l1);
        }
      } else {
        cl_kernel kd = kernel("ops", opts(1), "attn_decode");
        for (int t = r0; t < r0 + n; ++t) {
          const int p = pos0 + t, lo = win ? std::max(0, p - win + 1) : 0;
          int S, kstart, ns, pp = p, row = t, prow = 0;
          pick_split(lo, p, nsp, dec_smax, dec_sg, S, kstart, ns);
          set_args(kd, 0, qi_, qsc_, L.kq, L.ksc, L.vq, L.vsc, cp, nn, pp, apart_, row, w, kstart, S);
          size_t gws[2] = {(size_t)ns * 16, (size_t)(nkv * nsp)}, lws[2] = {16, (size_t)nsp};
          run(name, kd, 2, gws, lws);
          set_args(k_attn_comb_, 0, apart_, ns, nn, o_, row, hdd, prow);
          size_t g1 = (size_t)nn * hd, l1 = 64;
          run("attn_combine", k_attn_comb_, 1, &g1, &l1);
        }
      }
      r0 += n;
    }
  }

#include "engine_flashnext.h"  // (Qwen3.8 Flash Next)
#include "engine_ds41.h"  // (DeepSeek-V4.1-Flash)

#include "engine_qwen.h"  // (Qwen layers)
#include "mgpu/engine_tp_shard.h"  // (multi-GPU, zero/mgpu/: this engine as one shard of a model split over GPUs)
#include "xe2/engine_xe2.h"  // (tuning switches for GPUs with XMX, zero/xe2/; off on the iGPUs)
#include "engine_h8.h"  // (H8 prompt GEMMs, QUEUE 31.12: Hadamard-rotated int8, ARC_H8=1)
#include "xe2/engine_xw.h"  // (the H8R GEMM on XMX with wider tiles, QUEUE 33.4: ARC_GEMM_XW=1)

#include "npu/engine_npu.h"  // (the NPU experiment: a share of the Qwen FFN's gate / up on the NPU, ARC_NPU=1)

#include "engine_moe.h"  // (Mixture-of-experts FFN (qwen35moe, Flash Next, Gemma 4 26B))

  // Final norm of rows [r0, r0 + n) of x_ (per-32 layout, as the GEMVs expect; the f32 rows are also written to
  // `hout` when given), then the lm_head: logits_ row i belongs to x_ row r0 + i.
  void head_rows(int r0, int n, cl_mem normw, cl_mem hout) {
    const int saved = cur_T_;
    cur_T_ = n;
    if (hp.ds41) {  // DeepSeek-V4.1: the streams collapsed with the last FFN site's pre, then the output norm
      ds_collapse(ds_mixF_, 0, normw, r0, n, 16);
      matmul("lm_head", lm_head_, 16, logits_, n_logits(), 0, 0, true);
      cur_T_ = saved;
      return;
    }
    if (hp.hc) {  // Flash Next: the final mixer of the streams is the output norm
      // (the MTP head reads the rows' residual streams before the mixer, hc * E floats each: QUEUE F34)
      if (hout && mtp_block_) {
        const size_t W4 = (size_t)hp.hce() * 4;
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, x_, hout, (size_t)r0 * W4, 0, (size_t)n * W4, 0, nullptr, nullptr));
      }
      hc_mix(out_hc_norm_, out_hc_down_, out_hc_up_, n, 16, 1, r0);
      matmul("lm_head", lm_head_, 16, logits_, n_logits(), 0, 0, true);
      cur_T_ = saved;
      return;
    }
    float eps = hp.eps;
    int E = hp.n_embd, M = 16, nn = n, wh = hout ? 1 : 0, a32 = 1;
    set_args(k_rmsnorm_, 0, x_, r0, normw, E, M, nn, eps, aq_, asc_, absum_, hout ? hout : big_, wh, a32);
    size_t gws[2] = {256, (size_t)n}, lws[2] = {256, 1};
    run("rmsnorm_quant", k_rmsnorm_, 2, gws, lws);
    matmul("lm_head", lm_head_, 16, logits_, n_logits(), 0, 0, true);
    cur_T_ = saved;
  }
  void head(int T) { head_rows(T - 1, 1, out_norm_, has_mtp ? hlast_ : nullptr); }

#include "engine_drafters.h"  // (Speculative decoding)

 private:
};

#include "mgpu/engine_tp.h"  // (multi-GPU: TpEngine)

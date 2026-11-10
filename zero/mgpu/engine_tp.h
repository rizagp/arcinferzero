// inferzero tensor parallel: one model split over several GPUs (--gpus N), each GPU running an Engine shard
// (zero/mgpu/engine_tp_shard.h says what each holds) on a thread of its own, the shards in lockstep.
//
// The output projections' partial sums meet in host memory: GPUs that cannot reach each other's memory (Arc B60s on
// Windows: no peer access, docs/multi-gpu.md) all reach pinned host memory of the one context. Each shard copies its
// part there, waits for its GPU, meets the others at a barrier, then adds every shard's part to its residual stream
// in rank order (the same float sum on every GPU, so their x_ stay identical). Parts alternate between two halves of
// the buffer: a shard can be one projection ahead of another, never two (it would have passed a barrier the other
// had not reached).
//
// ARC_TP_SIM=1: the "GPUs" are all device 0, each shard with its own command list (tests the split on one GPU).
#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <immintrin.h>
#include <mutex>
#include <thread>

// a shard left its barrier because another one failed (its own error is the one to report)
// (a barrier's own timeout is one too: the shards are out of step after either)
struct TpAbort : std::runtime_error {
  TpAbort() : std::runtime_error("another GPU's shard failed") {}
  explicit TpAbort(const std::string& m) : std::runtime_error(m) {}
};

struct TpComm {
  int n = 1;
  size_t slot_floats = 0;  // rows x n_embd of one part
  float* host = nullptr;   // pinned host memory: [2 halves][n shards][slot_floats]
  int* flags = nullptr;    // pinned host memory: per shard, its f16 part went out of range (tp_reduce_rs)
  std::vector<int> half;   // per shard: the half its next part goes to
  std::atomic<int> arrived{0}, gen{0};
  std::atomic<bool> failed{false};  // (a shard threw: the others leave their barrier instead of waiting forever)
  std::atomic<bool> fault{false};   // (a wait gave up on a GPU, barrier or zrt::sync: never in step again, leaked at shutdown)
  std::atomic<int> first_failed{-1};  // (the shard that threw first: its error is the one to report)
  bool served = false;              // (serverz drives shard 0: TpEngine::serve)
  std::atomic<bool> cancel_now{false};  // (served: shard 0's answer to the request's cancel, for every shard)
  std::vector<double> wait_ms;      // per shard: time in reduce from its GPU wait to the barrier's end
  std::vector<long> reduces;        // per shard: how many
  // Failsafe against a GPU that hangs: a shard waiting longer than this at a barrier (ARC_TP_TIMEOUT seconds,
  // default 120: far beyond one layer of a prompt chunk) stops every shard and names the GPUs that did not come.
  // Each GPU's own waits give up after it too (zrt::Dev::sync_timeout_ns), or once another shard failed.
  double timeout_s = 120;
  std::unique_ptr<std::atomic<int>[]> reached;  // (per shard: 1 + the barrier generation it last reached)
  void barrier(int r) {
    const int g = gen.load(std::memory_order_acquire);
    reached[r].store(g + 1, std::memory_order_relaxed);
    if (arrived.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
      arrived.store(0, std::memory_order_relaxed);
      gen.fetch_add(1, std::memory_order_release);
      return;
    }
    std::chrono::steady_clock::time_point t0{};
    for (long long spins = 0; gen.load(std::memory_order_acquire) == g; ++spins) {  // (64-bit: an hour of yields, MG5)
      if (failed.load(std::memory_order_relaxed)) throw TpAbort();
      if (spins < 4096) {
        _mm_pause();  // (a token's sums come microseconds apart: spin)
        continue;
      }
      std::this_thread::yield();  // (a prompt chunk's GEMMs take milliseconds: let the core go)
      if (spins % 1024) continue;
      const auto now = std::chrono::steady_clock::now();
      if (t0 == std::chrono::steady_clock::time_point{}) t0 = now;
      if (std::chrono::duration<double>(now - t0).count() < timeout_s) continue;
      std::string late;
      for (int k = 0; k < n; ++k)
        if (reached[k].load(std::memory_order_relaxed) != g + 1) late += " " + std::to_string(k);
      int none = -1;
      first_failed.compare_exchange_strong(none, r);
      fault = true;
      failed = true;
      throw TpAbort("multi-GPU: GPU" + late + " did not reach the sum within " + std::to_string((int)timeout_s) +
                               " s (a GPU hangs?); restart (or load the model again)");
    }
  }
};

inline bool Engine::tp_served() const { return tp_->served; }

// A served split model's cancel check: every shard reaches it at the same point of a pass (the same chunks and layers),
// shard 0 asks the request's cancel, and all take its answer (one shard stopping alone would leave the others at a
// barrier). Two barriers: no shard reads the answer after shard 0 wrote the next one.
inline bool Engine::tp_cancelled() {
  TpComm& c = *tp_;
  clFinish(cl_.queue);
  if (opt.tp_rank == 0) c.cancel_now = cancel && cancel();
  c.barrier(opt.tp_rank);
  const bool v = c.cancel_now;
  c.barrier(opt.tp_rank);
  return v;
}

inline void Engine::tp_wrap_host() {
  if (tp_host_) return;
  TpComm& c = *tp_;
  tp_host_ = zrt::wrap(c.host, (size_t)2 * c.n * c.slot_floats * 4);
  mems_.push_back(tp_host_);
}

// From this many rows (prompt chunks) the f16 reduce-scatter; fewer (a generated token, drafts) keep the one-step f32
// sum, whose cost is its latency, not its bytes. ARC_TP_SUM=f32: the one-step f32 sum always (the build before).
// ARC_TP_RS_MIN=N: the reduce-scatter from N rows instead of 64 (to measure on B60s; at least the GPU count)
inline bool tp_rs_rows(int rows, int n) {
  static const int min_rows = [] {
    const char* e = getenv("ARC_TP_SUM");
    if (e != nullptr && std::string(e) == "f32") return INT_MAX;
    const char* m = getenv("ARC_TP_RS_MIN");
    return m ? std::max(1, atoi(m)) : 64;
  }();
  return rows >= min_rows && rows >= n;
}

inline void Engine::tp_reduce(int rows) {
  TpComm& c = *tp_;
  const int r = opt.tp_rank, E = hp.n_embd;
  {  // (tests of the failsafe: ARC_TP_HANG=r:k, shard r stops for good at its k-th sum, as a GPU that hangs;
     //  ARC_TP_HANG=r:k:wait, it throws there as a GPU wait that gave up (zrt::sync's timeout); once per process)
    struct Hang { int r = -1; long k = 0; bool wait = false; };
    static const Hang hang = [] {
      Hang h;
      const char* e = getenv("ARC_TP_HANG");
      char mode[8] = {};
      if (e && sscanf(e, "%d:%ld:%7s", &h.r, &h.k, mode) >= 2) h.wait = std::string(mode) == "wait";
      return h;
    }();
    static std::atomic<bool> hung{false};
    if (hang.r == r && c.reduces[r] + 1 == hang.k && !hung.exchange(true)) {
      fprintf(stderr, "inferzero: ARC_TP_HANG: GPU %d %s at sum %ld\n", r, hang.wait ? "gives up a wait" : "stops", hang.k);
      if (hang.wait) {
        c.fault = true;
        throw zrt::GpuWaitError("GPU " + std::to_string(r) + ": its work did not finish (ARC_TP_HANG)");
      }
      for (;;) std::this_thread::sleep_for(std::chrono::seconds(60));
    }
  }
  const size_t n = (size_t)rows * E;
  if (n > c.slot_floats) throw std::runtime_error("tensor parallel: more rows than the shared buffer holds");
  tp_build();
  tp_wrap_host();
  if (tp_rs_rows(rows, c.n)) return tp_reduce_rs(rows);
  const int h = c.half[r];
  const cl_ulong off = (cl_ulong)h * c.n * c.slot_floats, stride = c.slot_floats;
  ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), c.host + off + (size_t)r * stride, tp_part_->ptr, n * 4, nullptr, 0, nullptr));
  const auto t0 = std::chrono::steady_clock::now();
  zrt::sync();
  c.barrier(opt.tp_rank);
  c.wait_ms[r] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  ++c.reduces[r];
  int nn = (int)n, np = c.n;
  set_args(k_tp_add_, 0, x_, tp_host_, off, nn, np, stride);
  size_t g1 = (n + 255) / 256 * 256, l1 = 256;
  run("tp_add", k_tp_add_, 1, &g1, &l1);
  c.half[r] = h ^ 1;
}

// The f16 reduce-scatter (prompt chunks). GPU r owns rows [rows * r / N, rows * (r + 1) / N). In the half of the
// buffer this sum uses (N f32 slots: room for N f16 parts and the f16 sums):
//  1. its part to f16; every row but its own to its slot (put)
//  2. after a barrier: its rows summed over all parts (tp_rs_sum), added to x, the sums to the shared sums (res)
//  3. after a second barrier: the other GPUs' sums added to x (tp_rs_add)
// Per GPU and sum about 3 (N - 1) / N part sizes of f16 cross PCIe, against 1 + N in f32 for the one-step sum: 4 B60s
// 2.25 f16 (1.1 f32) instead of 5 f32 part sizes, 4.4x fewer bytes; 2 GPUs 4x. (Why one buffer half is enough: a GPU writes this half
// again two sums later, after a barrier the others reach only once their GPU finished reading it.)
inline void Engine::tp_reduce_rs(int rows) {
  TpComm& c = *tp_;
  const int r = opt.tp_rank, E = hp.n_embd, N = c.n;
  const size_t n = (size_t)rows * E, i0 = (size_t)rows * r / N * E, i1 = (size_t)rows * (r + 1) / N * E;
  if (!tp_h_) tp_h_ = buf((size_t)max_chunk * E * 2);
  const int h = c.half[r];
  const size_t stride = c.slot_floats;  // (halves per part)
  char* region = (char*)(c.host + (size_t)h * N * c.slot_floats);
  char* put = region + (size_t)r * stride * 2;
  char* res = region + (size_t)N * stride * 2;
  char* dev = (char*)tp_h_->ptr;
  int nn = (int)n, a = (int)i0, b = (int)i1;
  auto barrier = [&] {
    const auto t0 = std::chrono::steady_clock::now();
    zrt::sync();
    c.barrier(opt.tp_rank);
    c.wait_ms[r] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  // (failsafe: a part beyond f16's range, flagged by tp_half on any GPU, makes every GPU sum this one in f32)
  static const float lim = getenv("ARC_TP_F16_LIMIT") ? (float)atof(getenv("ARC_TP_F16_LIMIT")) : 65504.0f;  // (tests)
  if (!tp_flag_) {
    tp_flag_ = zrt::wrap(c.flags, (size_t)N * sizeof(int));
    mems_.push_back(tp_flag_);
  }
  c.flags[r] = 0;
  const cl_ulong fo = (cl_ulong)r * sizeof(int);
  // (less 1/1024: each part's f16 may round up half an ulp, and N parts at the limit summed to inf; kernel audit)
  float lim_n = lim * (1.0f - 1.0f / 1024) / N;
  set_args(k_tp_half_, 0, tp_part_, tp_h_, nn, from(tp_flag_, fo), lim_n, a, b);
  size_t g1 = (n + 255) / 256 * 256, l1 = 256;
  run("tp_sum", k_tp_half_, 1, &g1, &l1);
  if (i0 > 0) ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), put, dev, i0 * 2, nullptr, 0, nullptr));
  if (n > i1) ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), put + i1 * 2, dev + i1 * 2, (n - i1) * 2, nullptr, 0, nullptr));
  barrier();
  bool over = false;  // (every GPU reads all flags after the barrier: they decide alike; a flag is cleared for the
  for (int k = 0; k < N; ++k) over = over || ((volatile int*)c.flags)[k] != 0;  // next sum only after the next barrier)
  if (over) {  // the one-step f32 sum in the same buffer half (no GPU reads its f16 parts any more)
    static std::atomic<long> warned{0};
    if (r == 0 && warned++ < 3) fprintf(stderr, "inferzero: a partial sum beyond f16's range: summed in f32 (failsafe)\n");
    const cl_ulong off = (cl_ulong)h * N * c.slot_floats;
    ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), c.host + off + (size_t)r * stride, tp_part_->ptr, n * 4, nullptr, 0, nullptr));
    barrier();
    ++c.reduces[r];
    int np = N;
    cl_ulong st = stride;
    set_args(k_tp_add_, 0, x_, tp_host_, off, nn, np, st);
    run("tp_add", k_tp_add_, 1, &g1, &l1);
    c.half[r] = h ^ 1;
    return;
  }
  const cl_ulong put_off = (cl_ulong)h * N * c.slot_floats * 2, str = stride, res_off = put_off + (cl_ulong)N * stride;
  set_args(k_tp_rs_sum_, 0, x_, tp_part_, tp_host_, put_off, str, N, r, a, b, tp_h_);
  size_t g2 = (i1 - i0 + 255) / 256 * 256;
  if (i1 > i0) run("tp_sum", k_tp_rs_sum_, 1, &g2, &l1);
  if (i1 > i0) ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), res + i0 * 2, dev + i0 * 2, (i1 - i0) * 2, nullptr, 0, nullptr));
  barrier();
  ++c.reduces[r];
  set_args(k_tp_rs_add_, 0, x_, tp_host_, res_off, nn, a, b);
  run("tp_sum", k_tp_rs_add_, 1, &g1, &l1);
  c.half[r] = h ^ 1;
}

// N shards of one model, each on its own GPU and thread; forward() runs them together and joins their logits.
class TpEngine {
 public:
  TpEngine(const std::string& path, EngineOptions o, int n) : n_(n) {
    try {
      start(path, o);
    } catch (...) {
      shutdown();  // (no destructor runs after a throwing constructor: its threads must stop here)
      throw;
    }
  }
  ~TpEngine() { shutdown(); }
  TpEngine(const TpEngine&) = delete;
  TpEngine& operator=(const TpEngine&) = delete;
  int gpus() const { return n_; }
  Engine& shard(int r) { return *shards_[r]; }
  TpComm& comm() { return comm_; }
  int vocab() const { return (int)logits_.size(); }
  // Feeds tokens on every shard; the full vocabulary's logits of the last one (each shard's rows in rank order).
  const std::vector<float>& forward(const int* tokens, int n) {
    run_all([&](int r) { shards_[r]->forward(tokens, n); });
    size_t at = 0;
    for (int r = 0; r < n_; ++r) {
      const size_t per = (size_t)shards_[r]->n_logits();
      const std::vector<float>& l = shards_[r]->logits_host();
      std::copy(l.begin(), l.begin() + per, logits_.begin() + at);
      at += per;
    }
    return logits_;
  }
  void reset() {  // (the bound slot)
    run_all([&](int r) { shards_[r]->reset(); });
    std::fill(comm_.half.begin(), comm_.half.end(), 0);
  }
  // serverz --gpus: shard 0 becomes the server's engine, its state-changing calls mirrored on the other shards
  // (zero/mgpu/tp_mirror.h); call before the first request
  void serve() {
    comm_.served = true;
    shards_[0]->tp_front_ = this;
  }
  // the mirror's two halves: the other shards start a job and return at once; then they are waited for, giving
  // each one's error (null: none)
  void start_others(std::function<void(int)> job) {
    if (broken_) throw std::runtime_error("a GPU's shard failed earlier: restart (or load the model again)");
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (int r = 1; r < n_; ++r)  // (as run_on: never over a job still running, MG3)
        if (w_[r].has || stuck_[r]) throw std::runtime_error("multi-GPU: GPU " + std::to_string(r) + " still has a job running");
      for (int r = 1; r < n_; ++r) {
        w_[r].job = job;
        w_[r].has = true;
        w_[r].err = nullptr;
      }
      pending_ = n_ - 1;
    }
    cv_.notify_all();
  }
  std::vector<std::exception_ptr> wait_others() {
    std::unique_lock<std::mutex> lk(mu_);
    wait_done(lk);
    std::vector<std::exception_ptr> e;
    for (int r = 1; r < n_; ++r) e.push_back(w_[r].err);
    return e;
  }
  void set_broken() { broken_ = true; }
  bool broken() const { return broken_; }  // (serverz loads the model again)
  // a wait gave up on a GPU, or a thread was given up on: such a TpEngine is never deleted (TpHold), as a thread
  // still running on it would touch freed memory
  bool faulted() const { return comm_.fault; }
  // the GPUs' waits back to no limit and no fault flag, so the next model on these GPUs does not inherit them. From
  // shutdown: no abort either (the shards' destructors wait for their GPUs, and a wait that threw there would end the
  // process). For a faulted split model left unfreed (TpHold, release_stuck): every GPU's abort set instead, so a
  // thread given up on but still polling its wait leaves it rather than polling that GPU's list forever (only split
  // models' threads read it, zrt::tp_guarded; the next split model sets every GPU's again, start)
  // set once a faulted split model was left unfreed (TpHold): no other split model in this process
  static std::atomic<bool>& leaked_fault() {
    static std::atomic<bool> v{false};
    return v;
  }
  void disarm(bool release_stuck = false) {
    static std::atomic<bool> gone{true};
    if (release_stuck) leaked_fault() = true;
    for (int r = 0; r < n_ && r < (int)zrt::devs.size(); ++r) {
      zrt::devs[r]->abort = release_stuck ? &gone : nullptr;
      zrt::devs[r]->sync_timeout_ns = UINT64_MAX;
      zrt::devs[r]->fault = nullptr;
    }
  }
  // Conversation slots (EngineOptions::slots): every shard binds the same slot; forward() feeds the bound one.
  int slot_count() const { return shards_[0]->slot_count(); }
  int batch_rows() const { return shards_[0]->batch_rows(); }
  bool batch_ok() const { return shards_[0]->batch_ok(); }
  void bind(int s) { run_all([&](int r) { shards_[r]->bind(s); }); }
  // One generation step of several slots (row b: the next token of slots[b]), as Engine::forward_batch: a step's sums
  // carry B rows for the latency of one. Returns B rows of the full vocabulary's logits.
  const std::vector<float>& forward_batch(const std::vector<int>& slots, const std::vector<int>& toks) {
    const int B = (int)slots.size();
    run_all([&](int r) { shards_[r]->forward_batch(slots, toks); });
    const size_t V = logits_.size();
    if (batch_logits_.size() < V * B) batch_logits_.resize(V * B);
    size_t at = 0;
    for (int r = 0; r < n_; ++r) {
      const size_t per = (size_t)shards_[r]->n_logits();
      const std::vector<float>& l = shards_[r]->batch_logits_host();
      for (int b = 0; b < B; ++b) std::copy(l.begin() + b * per, l.begin() + (b + 1) * per, batch_logits_.begin() + b * V + at);
      at += per;
    }
    return batch_logits_;
  }

 private:
  void start(const std::string& path, const EngineOptions& o) {
    const int n = n_;
    if (n < 2 || n > 64) throw std::runtime_error("--gpus: 2 to 64");
    const bool sim = getenv("ARC_TP_SIM") && atoi(getenv("ARC_TP_SIM")) != 0;
    {
      ClRt rt;  // (the runtime: context and device 0, made here before any shard thread)
      if (!rt.init()) throw std::runtime_error("cannot start Level Zero");
      std::string err;
      if (ClRt::open_devices(n, sim, err) != n) throw std::runtime_error("--gpus " + std::to_string(n) + ": " + err);
    }
    fprintf(stderr, "inferzero: %d GPUs%s\n", n, sim ? " (ARC_TP_SIM: all on GPU 0)" : "");
    comm_.n = n;
    comm_.half.assign(n, 0);
    comm_.wait_ms.assign(n, 0.0);
    comm_.reduces.assign(n, 0);
    // (5 s to an hour: a huge value made the nanoseconds' cast below undefined, self audit MG5)
    if (const char* t = getenv("ARC_TP_TIMEOUT")) {
      const double v = atof(t);
      comm_.timeout_s = std::isfinite(v) ? std::clamp(v, 5.0, 3600.0) : 120.0;
    }
    // (a split model left after a GPU fault, TpHold: its threads may still run and read every GPU's abort / fault;
    //  a new split model would get its flags set by them, self audit MG2. One GPU then, the server's failsafe)
    if (leaked_fault())
      throw std::runtime_error("--gpus: a GPU fault earlier left a split model's threads running: restart the server for several GPUs");
    comm_.reached.reset(new std::atomic<int>[n]);
    for (int r = 0; r < n; ++r) comm_.reached[r] = 0;
    for (int r = 0; r < n; ++r) {  // (GPU 0's waits keep no abort: the server's other users, Whisper, share it)
      zrt::devs[r]->sync_timeout_ns = (uint64_t)(comm_.timeout_s * 1e9);
      zrt::devs[r]->fault = &comm_.fault;
      zrt::devs[r]->abort = r > 0 ? &comm_.failed : nullptr;  // (GPU 0: an abort a faulted model left, disarm, cleared)
    }
    stuck_.assign(n, 0);
    shards_.resize(n);
    w_.resize(n);
    for (int r = 0; r < n; ++r) w_[r].th = std::thread([this, r] { loop(r); });
    // shard 0 first (it compiles and caches the kernels the others then load), then the others one after another:
    // made together (kernels built, the XMX check, buffers allocated on several GPUs of one context at once), 3 and 4
    // B60s ended at random in the driver's "Abort was called at 252 line" (issue #4, 2026-10-31).
    // ARC_TP_PARALLEL_LOAD=1: the others together again (faster to load; for tests)
    const int oom_rank = getenv("ARC_TP_OOM") ? atoi(getenv("ARC_TP_OOM")) : -1;
    auto make = [&, o](int r) mutable {
      EngineOptions so = o;
      so.tp_rank = r;
      so.tp_size = n;
      zrt::oom_throws = true;  // (failsafe: a shard short of memory throws, the server loads on one GPU instead)
      zrt::oom_test = r == oom_rank;
      try {
        shards_[r] = std::make_unique<Engine>(path, so);
      } catch (...) {
        zrt::oom_throws = zrt::oom_test = false;
        throw;
      }
      zrt::oom_throws = zrt::oom_test = false;
      shards_[r]->tp_ = &comm_;
    };
    run_on({0}, make, false, false);  // (loads: no barrier to show progress, not watched)
    const bool together = getenv("ARC_TP_PARALLEL_LOAD") && atoi(getenv("ARC_TP_PARALLEL_LOAD")) != 0;
    std::vector<int> rest;
    for (int r = 1; r < n; ++r) rest.push_back(r);
    if (together && !rest.empty()) run_on(rest, make, false, false);
    else
      for (int r : rest) run_on({r}, make, false, false);
    Engine& e0 = *shards_[0];
    // (failsafe: the shards run the same passes only with the same weight layout and chunk; GPUs with different free
    //  memory could choose otherwise, and their barriers would then not meet)
    for (int r = 1; r < n; ++r)
      if (shards_[r]->w8_resident != e0.w8_resident || shards_[r]->max_chunk != e0.max_chunk || shards_[r]->ctx != e0.ctx)
        throw std::runtime_error("--gpus: GPU " + std::to_string(r) + " chose another weight layout than GPU 0 (W8 " +
                                 (shards_[r]->w8_resident ? "resident" : "just-in-time") + ", chunk " +
                                 std::to_string(shards_[r]->max_chunk) + ", ctx " + std::to_string(shards_[r]->ctx) +
                                 "): give --budget, --chunk and --ctx");
    comm_.slot_floats = (size_t)e0.max_chunk * e0.hp.n_embd;
    zrt::oom_throws = true;
    try {
      comm_.host = (float*)zrt::host_alloc((size_t)2 * n * comm_.slot_floats * 4);
      comm_.flags = (int*)zrt::host_alloc((size_t)n * sizeof(int));
    } catch (...) {
      zrt::oom_throws = false;
      throw;
    }
    zrt::oom_throws = false;
    for (int r = 0; r < n; ++r) comm_.flags[r] = 0;
    size_t vocab = 0;
    for (int r = 0; r < n; ++r) vocab += (size_t)shards_[r]->n_logits();
    logits_.assign(vocab, 0.0f);
  }
  // the shards freed (each on its own GPU's thread), the threads ended, the shared memory freed; from the
  // destructor or a constructor that threw part way
  void shutdown() {
    // (first the GPUs' waits back to no limit and no abort: a shard's destructor waits for its GPU, and a wait that
    //  threw there would end the process. After a wait gave up on a GPU (fault) its work may still be queued: then
    //  nothing is freed, the server says to restart)
    disarm();
    const bool fault = comm_.fault;
    // (stuck_ read under mu_, as wait_done writes it and a stuck thread reads it there: self audit MG4)
    auto is_stuck = [this](int r) {
      std::lock_guard<std::mutex> g(mu_);
      return r < (int)stuck_.size() && stuck_[r];
    };
    if (!w_.empty() && !fault) {  // (after a failure too: the GPUs that answer get their memory back, for a load again)
      std::vector<int> made;
      for (int r = 0; r < (int)shards_.size(); ++r)
        if (shards_[r] && !is_stuck(r)) made.push_back(r);
      try {
        if (!made.empty()) run_on(made, [this](int r) { shards_[r].reset(); }, true, false);
      } catch (...) {}
    }
    {
      std::lock_guard<std::mutex> g(mu_);
      quit_ = true;
    }
    cv_.notify_all();
    bool any_stuck = false;
    for (int r = 0; r < (int)w_.size(); ++r) {
      if (!w_[r].th.joinable()) continue;
      if (is_stuck(r)) {
        w_[r].th.detach();  // (a thread stuck in the driver: never joined; its shard and the shared memory leak)
        any_stuck = true;
      } else {
        w_[r].th.join();
      }
    }
    if (any_stuck || fault) {  // (the shards and the shared memory a stuck thread or a busy GPU may still touch: left)
      for (auto& sh : shards_) sh.release();
      return;
    }
    shards_.clear();  // (after a failure: whatever is left, freed here)
    if (comm_.host) zrt::mem_free(zrt::ctx, comm_.host);
    if (comm_.flags) zrt::mem_free(zrt::ctx, comm_.flags);
    comm_.flags = nullptr;
    comm_.host = nullptr;
  }
  struct Worker {
    std::thread th;
    std::function<void(int)> job;
    bool has = false;
    std::exception_ptr err;
  };
  int n_;
  TpComm comm_;
  std::vector<std::unique_ptr<Engine>> shards_;
  std::vector<Worker> w_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  bool quit_ = false;
  std::atomic<bool> broken_{false};  // (a shard failed during a call: the others' barrier count and parts are out of step)
  int pending_ = 0;
  std::vector<float> logits_, batch_logits_;
  std::vector<char> stuck_;  // (per shard: its thread was given up on, wait_done)

  // The workers' jobs waited for. Failsafe: once a shard failed, the others leave their barriers and GPU waits within
  // 100 ms; a thread still running 10 s later is stuck in the driver (its GPU hangs): it is given up on (left running,
  // detached at shutdown), the call ends with an error for it and the split model is broken until loaded again.
  // watch (the lockstep calls: passes, slots, the server's mirrored calls): when no barrier completes for timeout_s +
  // 30 s, the same, without any shard having failed: a shard stuck inside a driver call throws nothing and sets
  // nothing, and the request waited forever (self audit MG1). Not for loads and frees, which meet at no barrier.
  void wait_done(std::unique_lock<std::mutex>& lk, bool watch = true) {
    std::chrono::steady_clock::time_point since{}, moved = std::chrono::steady_clock::now();
    int last_gen = comm_.gen.load(std::memory_order_acquire);
    bool warned = false;
    while (!done_cv_.wait_for(lk, std::chrono::seconds(1), [&] { return pending_ == 0; })) {
      const auto now = std::chrono::steady_clock::now();
      const int g = comm_.gen.load(std::memory_order_acquire);
      if (g != last_gen) last_gen = g, moved = now, warned = false;
      // (a line first, 30 s before giving up: a slow path between two sums can be told from a hang in the log, R9)
      if (watch && !warned && now - moved > std::chrono::duration<double>(comm_.timeout_s)) {
        warned = true;
        fprintf(stderr, "inferzero: multi-GPU: no sum completed for %d s; given up on in 30 s more\n", (int)comm_.timeout_s);
      }
      const bool still = watch && now - moved > std::chrono::duration<double>(comm_.timeout_s + 30);
      if (!comm_.failed && !still) continue;
      if (!still) {
        if (since == std::chrono::steady_clock::time_point{}) since = now;
        if (now - since < std::chrono::seconds(10)) continue;
      }
      comm_.failed = true;  // (the others leave their barriers and waits)
      for (int r = 0; r < n_; ++r)
        if (w_[r].has) {
          stuck_[r] = 1;
          w_[r].has = false;
          w_[r].err = std::make_exception_ptr(std::runtime_error(
              "multi-GPU: GPU " + std::to_string(r) + (still ? " made no progress for " + std::to_string((int)comm_.timeout_s + 30) + " s" : " does not answer") +
              " (it hangs?); restart (or load the model again)"));
          fprintf(stderr, "inferzero: GPU %d does not answer: given up on\n", r);
        }
      pending_ = 0;
      comm_.fault = true;
      broken_ = true;
      return;
    }
  }

  void loop(int r) {
    zrt::cur = zrt::devs[r];
    zrt::tp_guarded = true;
    for (;;) {
      std::function<void(int)> job;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return quit_ || w_[r].has; });
        if (quit_ && !w_[r].has) return;
        job = w_[r].job;
      }
      std::exception_ptr err;
      try {
        job(r);
      } catch (const Engine::Cancelled&) {
        err = std::current_exception();  // (served: every shard cancels at the same point, tp_cancelled)
      } catch (...) {
        err = std::current_exception();
        int none = -1;
        comm_.first_failed.compare_exchange_strong(none, r);
        comm_.failed = true;
      }
      std::lock_guard<std::mutex> lk(mu_);
      if (stuck_[r]) return;  // (given up on by wait_done: the call ended without it)
      w_[r].has = false;
      w_[r].err = err;
      if (--pending_ == 0) done_cv_.notify_all();
    }
  }
  void run_on(const std::vector<int>& ranks, std::function<void(int)> job, bool even_broken = false, bool watch = true) {
    if (broken_ && !even_broken) throw std::runtime_error("a GPU's shard failed earlier: restart (or load the model again)");
    {
      std::lock_guard<std::mutex> lk(mu_);
      // (a job still running on a shard would have its count and result overwritten: never published over it, and
      //  never to a shard given up on, MG3)
      for (int r : ranks)
        if (w_[r].has || stuck_[r]) throw std::runtime_error("multi-GPU: GPU " + std::to_string(r) + " still has a job running");
      for (int r : ranks) {
        w_[r].job = job;
        w_[r].has = true;
        w_[r].err = nullptr;
      }
      pending_ = (int)ranks.size();
    }
    cv_.notify_all();
    std::unique_lock<std::mutex> lk(mu_);
    wait_done(lk, watch);
    std::exception_ptr err;
    for (int r : ranks)
      if (w_[r].err && (!err || r == comm_.first_failed)) err = w_[r].err;
    if (err) {
      try {
        std::rethrow_exception(err);
      } catch (const Engine::Cancelled&) {
        throw;  // (every shard at the same point: still in step)
      } catch (...) {
        broken_ = true;  // (shards that threw left their barrier or their sums mid-way)
        throw;
      }
    }
  }
  void run_all(std::function<void(int)> job) {
    std::vector<int> all;
    for (int r = 0; r < n_; ++r) all.push_back(r);
    run_on(all, std::move(job));
  }
};

// ---- TpMirror (zero/mgpu/tp_mirror.h): a served shard 0's call on the other shards, then their logits joined
inline void TpMirror::start(Engine* e, std::function<void(Engine&)> f, Gather g, int rows) {
  e_ = e;
  if (e->tp_depth_++ > 0) {
    nested_ = true;
    return;
  }
  g_ = g;
  rows_ = rows;
  uncaught_ = std::uncaught_exceptions();
  TpEngine& t = *e->tp_front_;
  if (t.broken()) {
    --e->tp_depth_;
    e_ = nullptr;
    throw std::runtime_error("a GPU's shard failed earlier: restart (or load the model again)");
  }
  t.comm().cancel_now = false;
  was_guarded_ = zrt::tp_guarded;
  zrt::tp_guarded = true;  // (shard 0's waits on this thread take GPU 0's limit: zrt::sync)
  for (int r = 1; r < t.gpus(); ++r) {  // (the fields the server sets on its engine before a call)
    Engine& s = t.shard(r);
    s.rope_shift = e->rope_shift;
    s.profile = e->profile;
    // (a cancel only where shard 0 has one, so every shard takes the same path; shard 0 answers: tp_cancelled)
    s.cancel = e->cancel ? std::function<bool()>([] { return false; }) : nullptr;
  }
  try {
    t.start_others([f, &t](int r) { f(t.shard(r)); });
  } catch (...) {
    --e->tp_depth_;
    e_ = nullptr;
    zrt::tp_guarded = was_guarded_;
    throw;
  }
}
// After shard 0's own part (returned or threw): the others waited for. All of them having thrown the same way at
// the same point (a cancel, a context overflow) leaves them in step; anything else (one shard failing on its own,
// shards that left a barrier for it) breaks the split model until it is loaded again.
inline void TpMirror::finish() {
  Engine& e = *e_;
  --e.tp_depth_;
  if (nested_) return;
  zrt::tp_guarded = was_guarded_;
  TpEngine& t = *e.tp_front_;
  TpComm& c = t.comm();
  const bool threw = std::uncaught_exceptions() > uncaught_;
  if (threw && !c.cancel_now) c.failed = true;  // (the others leave any barrier they wait at)
  const std::vector<std::exception_ptr> errs = t.wait_others();
  bool all_threw = true, abort = false;
  std::exception_ptr first;
  for (const std::exception_ptr& x : errs) {
    if (!x) {
      all_threw = false;
      continue;
    }
    if (!first) first = x;
    try {
      std::rethrow_exception(x);
    } catch (const TpAbort&) {
      abort = true;
    } catch (const zrt::GpuWaitError&) {
      abort = true;
    } catch (...) {
    }
  }
  bool same_sums = true;
  for (int r = 1; r < t.gpus(); ++r) same_sums = same_sums && c.reduces[r] == c.reduces[0];
  if (threw || first) {
    const bool in_step = threw && all_threw && !abort && !c.fault && same_sums && c.arrived.load() == 0;
    if (in_step) {  // (every shard stopped on its own at the same point: a cancel, a context overflow)
      c.failed = false;
      c.first_failed = -1;
      return;
    }
    t.set_broken();
    for (size_t k = 0; k < errs.size(); ++k) {  // (the other GPUs' own errors in the log: shard 0's may only say
      if (!errs[k]) continue;                  //  that another failed)
      try {
        std::rethrow_exception(errs[k]);
      } catch (const std::exception& x) {
        fprintf(stderr, "multi-GPU: GPU %zu: %s\n", k + 1, x.what());
      } catch (...) {
      }
    }
    if (threw) return;  // (shard 0's own error goes on)
    std::rethrow_exception(first);
  }
  const size_t V = (size_t)e.hp.n_vocab;
  if (g_ == kLogits) {
    std::vector<float>& l = e.logits_host_;
    size_t at = (size_t)e.n_logits();
    l.resize(V);
    for (int r = 1; r < t.gpus(); ++r) {
      const size_t per = (size_t)t.shard(r).n_logits();
      const std::vector<float>& s = t.shard(r).logits_host();
      std::copy(s.begin(), s.begin() + per, l.begin() + at);
      at += per;
    }
  } else if (g_ == kBatch) {
    std::vector<float>& l = e.batch_logits_host_;
    const size_t n0 = (size_t)e.n_logits(), B = (size_t)rows_;
    if (l.size() < V * B) l.resize(V * B);
    for (size_t b = B; b-- > 0;) memmove(&l[b * V], &l[b * n0], n0 * 4);  // (rows of n0 -> rows of V, last first)
    size_t at = n0;
    for (int r = 1; r < t.gpus(); ++r) {
      const size_t per = (size_t)t.shard(r).n_logits();
      const std::vector<float>& s = t.shard(r).batch_logits_host();
      for (size_t b = 0; b < B; ++b) std::copy(s.begin() + b * per, s.begin() + (b + 1) * per, l.begin() + b * V + at);
      at += per;
    }
  }
}

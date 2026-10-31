// arcinferzero tensor parallel: one model split over several GPUs (--gpus N), each GPU running an Engine shard
// (zero/engine_tp_shard.h says what each holds) on a thread of its own, the shards in lockstep.
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
#include <atomic>
#include <condition_variable>
#include <exception>
#include <immintrin.h>
#include <mutex>
#include <thread>

struct TpComm {
  int n = 1;
  size_t slot_floats = 0;  // rows x n_embd of one part
  float* host = nullptr;   // pinned host memory: [2 halves][n shards][slot_floats]
  std::vector<int> half;   // per shard: the half its next part goes to
  std::atomic<int> arrived{0}, gen{0};
  std::atomic<bool> failed{false};  // (a shard threw: the others leave their barrier instead of waiting forever)
  std::vector<double> wait_ms;      // per shard: time in reduce from its GPU wait to the barrier's end
  std::vector<long> reduces;        // per shard: how many
  void barrier() {
    const int g = gen.load(std::memory_order_acquire);
    if (arrived.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
      arrived.store(0, std::memory_order_relaxed);
      gen.fetch_add(1, std::memory_order_release);
      return;
    }
    while (gen.load(std::memory_order_acquire) == g) {
      if (failed.load(std::memory_order_relaxed)) throw std::runtime_error("another GPU's shard failed");
      _mm_pause();
    }
  }
};

inline void Engine::tp_reduce(int rows) {
  TpComm& c = *tp_;
  const int r = opt.tp_rank, E = hp.n_embd;
  const size_t n = (size_t)rows * E;
  if (n > c.slot_floats) throw std::runtime_error("tensor parallel: more rows than the shared buffer holds");
  if (!k_tp_add_) {
    static const char* src = R"CLC(
__kernel void tp_add(__global float* x, __global const float* p, ulong off, int n, int np, ulong stride) {
  const int i = get_global_id(0);
  if (i >= n) return;
  float s = x[i];
  for (int k = 0; k < np; ++k) s += p[off + k * stride + i];
  x[i] = s;
})CLC";
    cl_program p = cl_.build(src, "-cl-std=CL3.0");
    progs_.push_back(p);
    k_tp_add_ = cl_.kernel(p, "tp_add");
    tp_host_ = zrt::wrap(c.host, (size_t)2 * c.n * c.slot_floats * 4);
    mems_.push_back(tp_host_);
  }
  const int h = c.half[r];
  const cl_ulong off = (cl_ulong)h * c.n * c.slot_floats, stride = c.slot_floats;
  ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::D().cl, c.host + off + (size_t)r * stride, tp_part_->ptr, n * 4, nullptr, 0, nullptr));
  const auto t0 = std::chrono::steady_clock::now();
  zrt::sync();
  c.barrier();
  c.wait_ms[r] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  ++c.reduces[r];
  int nn = (int)n, np = c.n;
  set_args(k_tp_add_, 0, x_, tp_host_, off, nn, np, stride);
  size_t g1 = (n + 255) / 256 * 256, l1 = 256;
  run("tp_add", k_tp_add_, 1, &g1, &l1);
  c.half[r] = h ^ 1;
}

// N shards of one model, each on its own GPU and thread; forward() runs them together and joins their logits.
class TpEngine {
 public:
  TpEngine(const std::string& path, EngineOptions o, int n) : n_(n) {
    const bool sim = getenv("ARC_TP_SIM") && atoi(getenv("ARC_TP_SIM")) != 0;
    {
      ClRt rt;  // (the runtime: context and device 0, made here before any shard thread)
      if (!rt.init()) throw std::runtime_error("cannot start Level Zero");
      std::string err;
      if (ClRt::open_devices(n, sim, err) != n) throw std::runtime_error("--gpus " + std::to_string(n) + ": " + err);
    }
    fprintf(stderr, "arcinferzero: %d GPUs%s\n", n, sim ? " (ARC_TP_SIM: all on GPU 0)" : "");
    comm_.n = n;
    comm_.half.assign(n, 0);
    comm_.wait_ms.assign(n, 0.0);
    comm_.reduces.assign(n, 0);
    shards_.resize(n);
    w_.resize(n);
    for (int r = 0; r < n; ++r) w_[r].th = std::thread([this, r] { loop(r); });
    // shard 0 first (it compiles and caches the kernels the others then load), then the others together
    auto make = [&, o](int r) mutable {
      EngineOptions so = o;
      so.tp_rank = r;
      so.tp_size = n;
      shards_[r] = std::make_unique<Engine>(path, so);
      shards_[r]->tp_ = &comm_;
    };
    run_on({0}, make);
    std::vector<int> rest;
    for (int r = 1; r < n; ++r) rest.push_back(r);
    if (!rest.empty()) run_on(rest, make);
    Engine& e0 = *shards_[0];
    comm_.slot_floats = (size_t)e0.max_chunk * e0.hp.n_embd;
    comm_.host = (float*)zrt::host_alloc((size_t)2 * n * comm_.slot_floats * 4);
    logits_.assign((size_t)e0.n_logits() * n, 0.0f);
  }
  ~TpEngine() {
    try {
      std::vector<int> all;
      for (int r = 0; r < n_; ++r) all.push_back(r);
      run_on(all, [this](int r) { shards_[r].reset(); });  // (each freed on its own GPU's thread)
    } catch (...) {}
    {
      std::lock_guard<std::mutex> g(mu_);
      quit_ = true;
    }
    cv_.notify_all();
    for (auto& w : w_) if (w.th.joinable()) w.th.join();
    if (comm_.host) zeMemFree(zrt::ctx, comm_.host);
  }
  int gpus() const { return n_; }
  Engine& shard(int r) { return *shards_[r]; }
  TpComm& comm() { return comm_; }
  // Feeds tokens on every shard; the full vocabulary's logits of the last one.
  const std::vector<float>& forward(const int* tokens, int n) {
    run_all([&](int r) { shards_[r]->forward(tokens, n); });
    const size_t per = (size_t)shards_[0]->n_logits();
    for (int r = 0; r < n_; ++r) {
      const std::vector<float>& l = shards_[r]->logits_host();
      std::copy(l.begin(), l.begin() + per, logits_.begin() + r * per);
    }
    return logits_;
  }
  void reset() {
    run_all([&](int r) { shards_[r]->reset(); });
    std::fill(comm_.half.begin(), comm_.half.end(), 0);
  }

 private:
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
  int pending_ = 0;
  std::vector<float> logits_;

  void loop(int r) {
    zrt::cur = zrt::devs[r];
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
      } catch (...) {
        err = std::current_exception();
        comm_.failed = true;
      }
      std::lock_guard<std::mutex> lk(mu_);
      w_[r].has = false;
      w_[r].err = err;
      if (--pending_ == 0) done_cv_.notify_all();
    }
  }
  void run_on(const std::vector<int>& ranks, std::function<void(int)> job) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (int r : ranks) {
        w_[r].job = job;
        w_[r].has = true;
        w_[r].err = nullptr;
      }
      pending_ = (int)ranks.size();
    }
    cv_.notify_all();
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [&] { return pending_ == 0; });
    for (int r : ranks)
      if (w_[r].err) {
        comm_.failed = false;
        std::rethrow_exception(w_[r].err);
      }
  }
  void run_all(std::function<void(int)> job) {
    std::vector<int> all;
    for (int r = 0; r < n_; ++r) all.push_back(r);
    run_on(all, std::move(job));
  }
};

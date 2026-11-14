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
// GPUs that do reach each other's memory (peer access, as Linux's driver reports for B60s on one PCIe switch or root
// port, issue #17) can do better: TpEngine::choose_sums times the links at start and, when a direct copy is faster
// than the host link, every GPU gets an inbox in its own memory, laid out as a half of the host buffer, which the
// others' parts and sums are copied straight into (one PCIe crossing instead of two; TpComm::p2p). The kernels then
// read their own inbox. The same values are summed in the same order: the logits are the same bit for bit.
//
// ARC_TP_SIM=1: the "GPUs" are all device 0, each shard with its own command list (tests the split on one GPU).
#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
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
  // Direct GPU-to-GPU sums (TpEngine::choose_sums): per GPU an inbox in its own memory, [2 halves][half_b bytes].
  // In a half: the reduce-scatter's f16 parts at k * slot_floats halves and its sums after the N parts (as in host
  // memory), or the one-step f32 sum's parts packed (k * rows * E floats). The f32 failsafe and the flags stay in host
  // memory. Off (empty) when the sums go through host memory.
  bool p2p = false;
  std::vector<char*> inbox;
  size_t half_b = 0;
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
  // The pipelined reduce-scatter (ARC_TP_PIPE, tp_reduce_rs_pipe): per shard, how many pieces of its parts (put_n)
  // and of its sums (res_n) it has in host memory so far, over all sums. The shards make the same sums in the same
  // pieces, so a shard waits until every other one has as many as it has itself.
  std::unique_ptr<std::atomic<long>[]> put_n, res_n;
  void wait_all(std::atomic<long>* cnt, long target, int r, const char* what) {
    std::chrono::steady_clock::time_point t0{};
    for (long long spins = 0;; ++spins) {
      bool all = true;
      for (int k = 0; k < n && all; ++k) all = cnt[k].load(std::memory_order_acquire) >= target;
      if (all) return;
      if (failed.load(std::memory_order_relaxed)) throw TpAbort();
      if (spins < 4096) {
        _mm_pause();
        continue;
      }
      std::this_thread::yield();
      if (spins % 1024) continue;
      const auto now = std::chrono::steady_clock::now();
      if (t0 == std::chrono::steady_clock::time_point{}) t0 = now;
      if (std::chrono::duration<double>(now - t0).count() < timeout_s) continue;
      std::string late;
      for (int k = 0; k < n; ++k)
        if (cnt[k].load(std::memory_order_relaxed) < target) late += " " + std::to_string(k);
      int none = -1;
      first_failed.compare_exchange_strong(none, r);
      fault = true;
      failed = true;
      throw TpAbort("multi-GPU: GPU" + late + " did not bring " + what + " within " + std::to_string((int)timeout_s) +
                    " s (a GPU hangs?); restart (or load the model again)");
    }
  }
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

// ARC_TP_OVERLAP=1: a layer-major prompt's sums (forward_layer_major, just-in-time weights) on a thread of their own
// per shard, so the GPU computes while a sum crosses PCIe. Chunk c's FFN sum at layer L is needed only when layer L+1
// reaches chunk c; meanwhile layer L computes chunk c+1's attention (which reads chunk c's KV cache and DeltaNet
// state, both written before that sum). So:
//  - the main thread matmuls a projection into one of two partial buffers (parts), its list signals the job's
//    `ready` event and the job goes to the sum thread's queue; an attention / DeltaNet sum is waited for at once
//    (`done`: the FFN that follows reads its rows), an FFN sum only before the chunk's next layer (tpo_chunk) or the
//    pass's end; a partial buffer is written again only after the job that read it is done
//  - the sum thread runs the sums one after another in the same order on every shard (tp_reduce_at: the same
//    barriers, counters, time limits and f16 failsafe), on a Dev of its own (`dev`: an immediate compute list on the
//    compute group's second queue when there is one, its own copy list and events), waiting for `ready` first and
//    its list signalling `done` last
//  - every wait for the other list's event: on the host, then on the GPU for the signalled event (tpo_wait)
//  - the same kernels on the same values in the same order: the logits bit for bit as without it
//  - events: two per job, never reused within a pass; reset at the next pass's start, when both lists are idle
//  - failsafes: a sum's error is kept, the rest of the queue skipped (their `done` signalled from the host so the
//    main list never waits for good) and thrown on the main thread at its next submit or join; a pass that ends early
//    with sums queued marks the split model failed (the others leave their barriers) and waits for its sum thread;
//    a sum thread stuck in a hang keeps its shard's thread waiting, which TpEngine::wait_done gives up on as before
// Not with MoE (tp_moe_combine's sum stays where it is). Chunk-by-chunk passes have nothing to overlap with, so a dense
// split with resident weights runs its long prompts layer by layer when this is on (Engine::lm_resident_tp, 2026-11-14).
inline bool tpo_wanted() {
  static const bool on = getenv("ARC_TP_OVERLAP") && atoi(getenv("ARC_TP_OVERLAP")) != 0;
  return on;
}
struct Engine::TpOverlap {
  struct Job {
    int rows = 0;
    cl_mem x = nullptr, part = nullptr;
    ze_event_handle_t ready = nullptr, done = nullptr;
  };
  zrt::Dev dev;  // (the sum thread's: zrt::cur there)
  TpKs ks{};     // (the sum thread's own kernels: arguments are set per kernel object)
  cl_mem flag_r = nullptr;  // (this GPU's f16 flag as a kernel argument, made on the main thread)
  cl_mem parts[2] = {};
  ze_event_handle_t part_done[2] = {};  // (the job that last read each partial buffer)
  int next_part = 0;
  std::vector<ze_event_pool_handle_t> pools;
  std::vector<ze_event_handle_t> evs;  // (every event made; the first `used` handed out since the pass began)
  size_t used = 0;
  bool live = false;  // (inside a pass: tp_out queues its sums)
  int chunk = 0;      // (the chunk run_layer is on)
  std::vector<ze_event_handle_t> pend;  // (per chunk: the FFN sum its next layer waits for)
  std::thread th;
  std::mutex mu;
  std::condition_variable cv, idle_cv;
  std::deque<Job> q;
  bool busy = false, quit = false;
  long queued = 0, finished = 0;  // (jobs, for the log)
  std::exception_ptr err;  // (the first sum that failed; the queue after it skipped)
  ze_event_handle_t event() {
    if (used == evs.size()) {
      const uint32_t k = 512;
      ze_event_pool_desc_t pd{ZE_STYPE_EVENT_POOL_DESC, nullptr, 0x1 /* host visible */, k};
      ze_event_pool_handle_t pool = nullptr;
      ZE_CHECK(zeEventPoolCreate(zrt::ctx, &pd, 1, &dev.dev, &pool));
      pools.push_back(pool);
      for (uint32_t i = 0; i < k; ++i) {
        ze_event_desc_t ed{ZE_STYPE_EVENT_DESC, nullptr, i, 0x4 /* signal: host scope */, 0x2 /* wait: device scope */};
        ze_event_handle_t e = nullptr;
        ZE_CHECK(zeEventCreate(pool, &ed, &e));
        evs.push_back(e);
      }
    }
    return evs[used++];
  }
  // (the queue empty and no job running; the first error, cleared. limit_s > 0: given up on after that, *gave_up set:
  //  a sum thread stuck in a host or driver wait otherwise held its shard's thread for good, and in serverz shard 0's
  //  part of a call has no watchdog, so the server hung; audit 2026-11-14)
  std::exception_ptr wait_idle(double limit_s = 0, bool* gave_up = nullptr) {
    std::unique_lock<std::mutex> lk(mu);
    auto idle = [&] { return q.empty() && !busy; };
    if (limit_s > 0) {
      if (!idle_cv.wait_for(lk, std::chrono::duration<double>(limit_s), idle)) {
        if (gave_up) *gave_up = true;
        return std::make_exception_ptr(zrt::GpuWaitError("the split model's sum thread did not finish within " +
                                                         std::to_string((int)limit_s) + " s (a GPU hangs?)"));
      }
    } else {
      idle_cv.wait(lk, idle);
    }
    std::exception_ptr e = err;
    err = nullptr;
    return e;
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
  TpComm& c = *tp_;
  if (!tp_host_) {
    tp_host_ = zrt::wrap(c.host, (size_t)2 * c.n * c.slot_floats * 4);
    mems_.push_back(tp_host_);
  }
  if (c.p2p && !tp_in_) {
    tp_in_ = zrt::wrap(c.inbox[opt.tp_rank], 2 * c.half_b);
    mems_.push_back(tp_in_);
  }
}
// (direct sums: before this GPU's kernels read half h of its inbox, which other GPUs' copy engines wrote over PCIe,
//  a memory-ranges barrier on its list so no line of an earlier sum's values is read from its caches; the host learnt
//  the copies were done, the GPU itself did not. Skipped when the loader lacks the call)
inline void tp_inbox_fence(const TpComm& c, int r, int h) {
  if (!zeCommandListAppendMemoryRangesBarrier) return;
  const size_t sz = c.half_b;
  const void* p = c.inbox[r] + (size_t)h * c.half_b;
  ZE_CHECK(zeCommandListAppendMemoryRangesBarrier(zrt::imm(), 1, &sz, &p, nullptr, 0, nullptr));
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

// (the sum's kernels: this engine's own, or the sum thread's (ARC_TP_OVERLAP), which sets their arguments apart)
inline Engine::TpKs Engine::tp_ks(const TpSum& a) const {
  if (a.bg) return tpo_->ks;
  return TpKs{k_tp_add_, k_tp_half_, k_tp_rs_sum_, k_tp_rs_add_, k_tp_rs_add_r_};
}
inline void Engine::tp_run(const TpSum& a, const char* name, cl_kernel k, size_t g, size_t l) {
  if (!a.bg) return run(name, k, 1, &g, &l);
  CL_CHECK(clEnqueueNDRangeKernel(cl_.queue, k, 1, nullptr, &g, &l, 0, nullptr, nullptr));  // (no profiling there)
}

inline void Engine::tp_reduce_at(int rows, const TpSum& a) {
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
      zlog::eprintf("inferzero: ARC_TP_HANG: GPU %d %s at sum %ld\n", r, hang.wait ? "gives up a wait" : "stops", hang.k);
      if (hang.wait) {
        c.fault = true;
        throw zrt::GpuWaitError("GPU " + std::to_string(r) + ": its work did not finish (ARC_TP_HANG)");
      }
      for (;;) std::this_thread::sleep_for(std::chrono::seconds(60));
    }
  }
  const size_t n = (size_t)rows * E;
  if (n > c.slot_floats) throw std::runtime_error("tensor parallel: more rows than the shared buffer holds");
  tp_build();      // (the sum thread: both made before it starts, tpo_begin)
  tp_wrap_host();
  if (tp_rs_rows(rows, c.n)) return tp_reduce_rs(rows, a);
  const TpKs ks = tp_ks(a);
  const int h = c.half[r];
  // (direct: the part to every GPU's inbox, its own included, packed; a part too large for a half, which only
  //  ARC_TP_SUM=f32 makes, goes through host memory as before; every GPU decides alike, from the same rows)
  const bool p2p = c.p2p && !a.bg && (size_t)c.n * n * 4 <= c.half_b;
  cl_ulong off = (cl_ulong)h * c.n * c.slot_floats, stride = c.slot_floats;
  if (p2p) {
    off = (cl_ulong)(h * c.half_b / 4);
    stride = n;
    for (int k = 0; k < c.n; ++k)
      ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), c.inbox[k] + (off + (size_t)r * stride) * 4, a.part->ptr, n * 4, nullptr, 0, nullptr));
  } else {
    ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), c.host + off + (size_t)r * stride, a.part->ptr, n * 4, nullptr, 0, nullptr));
  }
  const auto t0 = std::chrono::steady_clock::now();
  zrt::sync();
  c.barrier(opt.tp_rank);
  c.wait_ms[r] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  ++c.reduces[r];
  int nn = (int)n, np = c.n;
  if (p2p) tp_inbox_fence(c, r, h);
  set_args(ks.add, 0, a.x, p2p ? tp_in_ : tp_host_, off, nn, np, stride);
  tp_run(a, "tp_add", ks.add, (n + 255) / 256 * 256, 256);
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
// ARC_TP_PIPE (on unless =0): the reduce-scatter in pieces, both directions of every GPU's link at once (tp_reduce_rs_pipe; after
// manoreken2's HostCopy, PR #10: a block cut into pieces copied by two command lists, 1.85x on two B60s). On by
// default since 2026-11-12, measured faster everywhere it ran (the same logits bit for bit): 27B 4K prompt over 2 Pro
// B60s 1,298.9 -> 1,368.8 tok/s, over 4 1,629.6 -> 1,734.0 (issue #4, package fffaf1a); a 27B IQ4_XS 32K prompt over
// two Arcs, the second on PCIe x4, 903.9 -> 1,033.7 (Reddit). ARC_TP_PIPE=0: the three steps one after another, as
// before. ARC_TP_PIPE_PIECES=N: pieces per GPU block (default: ~2 MB of f16 each, 2-8)
constexpr int kTpPipeMax = 8;
inline bool tp_pipe_on() {
  static const bool on = !getenv("ARC_TP_PIPE") || atoi(getenv("ARC_TP_PIPE")) != 0;
  return on;
}
inline int tp_pipe_pieces(int rows, int n, int E) {
  static const int forced = getenv("ARC_TP_PIPE_PIECES") ? atoi(getenv("ARC_TP_PIPE_PIECES")) : 0;
  const int min_rows = rows / n;  // (the smallest block: rows * b / n .. rows * (b + 1) / n)
  int p = forced > 0 ? forced : (int)((size_t)min_rows * E * 2 / ((size_t)2 << 20));
  p = std::max(forced > 0 ? 1 : 2, std::min(p, kTpPipeMax));
  return std::max(1, std::min(p, min_rows));
}

inline void Engine::tp_reduce_rs(int rows, const TpSum& a) {
  if (tp_pipe_on()) return tp_reduce_rs_pipe(rows, a);
  const TpKs ks = tp_ks(a);
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
  int nn = (int)n, ia = (int)i0, ib = (int)i1;
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
  set_args(ks.half, 0, a.part, tp_h_, nn, a.bg ? tpo_->flag_r : from(tp_flag_, fo), lim_n, ia, ib);
  size_t g1 = (n + 255) / 256 * 256, l1 = 256;
  tp_run(a, "tp_sum", ks.half, g1, l1);
  if (i0 > 0) ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), put, dev, i0 * 2, nullptr, 0, nullptr));
  if (n > i1) ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), put + i1 * 2, dev + i1 * 2, (n - i1) * 2, nullptr, 0, nullptr));
  barrier();
  bool over = false;  // (every GPU reads all flags after the barrier: they decide alike; a flag is cleared for the
  for (int k = 0; k < N; ++k) over = over || ((volatile int*)c.flags)[k] != 0;  // next sum only after the next barrier)
  if (over) {  // the one-step f32 sum in the same buffer half (no GPU reads its f16 parts any more)
    static std::atomic<long> warned{0};
    if (r == 0 && warned++ < 3) zlog::eprintf("inferzero: a partial sum beyond f16's range: summed in f32 (failsafe)\n");
    const cl_ulong off = (cl_ulong)h * N * c.slot_floats;
    ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), c.host + off + (size_t)r * stride, a.part->ptr, n * 4, nullptr, 0, nullptr));
    barrier();
    ++c.reduces[r];
    int np = N;
    cl_ulong st = stride;
    set_args(ks.add, 0, a.x, tp_host_, off, nn, np, st);
    tp_run(a, "tp_add", ks.add, g1, l1);
    c.half[r] = h ^ 1;
    return;
  }
  const cl_ulong put_off = (cl_ulong)h * N * c.slot_floats * 2, str = stride, res_off = put_off + (cl_ulong)N * stride;
  set_args(ks.rs_sum, 0, a.x, a.part, tp_host_, put_off, str, N, r, ia, ib, tp_h_);
  size_t g2 = (i1 - i0 + 255) / 256 * 256;
  if (i1 > i0) tp_run(a, "tp_sum", ks.rs_sum, g2, l1);
  if (i1 > i0) ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), res + i0 * 2, dev + i0 * 2, (i1 - i0) * 2, nullptr, 0, nullptr));
  barrier();
  ++c.reduces[r];
  set_args(ks.rs_add, 0, a.x, tp_host_, res_off, nn, ia, ib);
  tp_run(a, "tp_sum", ks.rs_add, g1, l1);
  c.half[r] = h ^ 1;
}

// The same reduce-scatter in pieces (ARC_TP_PIPE=1). Every GPU block (rows [T b / N, T (b + 1) / N)) is cut into P
// pieces of rows, the same on every GPU. This GPU's copy list (zrt::tp_copy_list, a copy engine) puts the pieces of
// its part to host memory, while its compute list already sums the pieces of its own block that every GPU has put:
// the puts go GPU -> host while the sums read host -> GPU. Then its sums' pieces go out the same way while the others'
// sums come in. Shards learn about each other's pieces from TpComm::put_n / res_n (host counters), the copy list about
// this GPU's kernels from events. Each value is summed by the same kernels' arithmetic in the same order as in
// tp_reduce_rs: the same results, bit for bit.
//  - the f16 failsafe: every GPU's flag is final once every GPU has put its first piece (its tp_half ran before);
//    then all decide alike, and an overflow finishes every put before the one-step f32 sum overwrites the buffer half
//  - reuse of a buffer half (written again two sums later): a GPU's puts of the next sum wait for its kernels of this
//    one (the copy list waits for tp_half, after them on the in-order compute list), so a GPU that has seen every
//    other GPU's first piece of the next sum knows they all finished reading this half
//  - every wait has the barrier's time limit and leaves when another GPU failed
inline void Engine::tp_reduce_rs_pipe(int rows, const TpSum& a) {
  const TpKs ks = tp_ks(a);
  TpComm& c = *tp_;
  const int r = opt.tp_rank, E = hp.n_embd, N = c.n;
  const size_t n = (size_t)rows * E, i0 = (size_t)rows * r / N * E, i1 = (size_t)rows * (r + 1) / N * E;
  const int P = tp_pipe_pieces(rows, N, E);
  auto piece = [&](int b, int p, size_t& p0, size_t& p1) {  // (values [p0, p1) of block b's piece p)
    const size_t R0 = (size_t)rows * b / N, R1 = (size_t)rows * (b + 1) / N;
    p0 = (R0 + (R1 - R0) * p / P) * E;
    p1 = (R0 + (R1 - R0) * (p + 1) / P) * E;
  };
  if (!tp_h_) tp_h_ = buf((size_t)max_chunk * E * 2);
  if (!tp_flag_) {
    tp_flag_ = zrt::wrap(c.flags, (size_t)N * sizeof(int));
    mems_.push_back(tp_flag_);
  }
  const int h = c.half[r];
  const size_t stride = c.slot_floats;
  char* region = (char*)(c.host + (size_t)h * N * c.slot_floats);
  char* put = region + (size_t)r * stride * 2;
  char* res = region + (size_t)N * stride * 2;
  char* dev = (char*)tp_h_->ptr;
  // (direct, TpComm::p2p: block k's pieces straight to GPU k's inbox, this GPU's sums to every other GPU's; the
  //  kernels read their own inbox. The same offsets within a half as in host memory)
  const bool p2p = c.p2p && !a.bg;
  auto put_to = [&](int k) { return p2p ? c.inbox[k] + (size_t)h * c.half_b + (size_t)r * stride * 2 : put; };
  auto res_to = [&](int k) { return p2p ? c.inbox[k] + (size_t)h * c.half_b + (size_t)N * stride * 2 : res; };
  cl_mem shared = p2p ? tp_in_ : tp_host_;
  ze_command_list_handle_t cp = zrt::tp_copy_list();
  // (events: 0 tp_half done; 1 + p piece p put; 1 + kTpPipeMax + p piece p summed; 1 + 2 kTpPipeMax + p its sums put.
  //  The copy list is idle here (every event of the last sum was waited for), so they can be reset)
  zrt::list_wait(cp);
  ze_event_handle_t evH = zrt::tp_event(0);
  auto ev_put = [](int p) { return zrt::tp_event(1 + p); };
  auto ev_sum = [](int p) { return zrt::tp_event(1 + kTpPipeMax + p); };
  auto ev_res = [](int p) { return zrt::tp_event(1 + 2 * kTpPipeMax + p); };
  ZE_CHECK(zeEventHostReset(evH));
  for (int p = 0; p < P; ++p) {
    ZE_CHECK(zeEventHostReset(ev_put(p)));
    ZE_CHECK(zeEventHostReset(ev_sum(p)));
    ZE_CHECK(zeEventHostReset(ev_res(p)));
  }
  const auto t0 = std::chrono::steady_clock::now();
  auto waited = [&] { c.wait_ms[r] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); };
  // 1. the part to f16 (and range-checked), then its other blocks' pieces put by the copy list, piece by piece
  static const float lim = getenv("ARC_TP_F16_LIMIT") ? (float)atof(getenv("ARC_TP_F16_LIMIT")) : 65504.0f;  // (tests)
  c.flags[r] = 0;
  const cl_ulong fo = (cl_ulong)r * sizeof(int);
  float lim_n = lim * (1.0f - 1.0f / 1024) / N;
  int nn = (int)n, ia = (int)i0, ib = (int)i1;
  set_args(ks.half, 0, a.part, tp_h_, nn, a.bg ? tpo_->flag_r : from(tp_flag_, fo), lim_n, ia, ib);
  size_t g1 = (n + 255) / 256 * 256, l1 = 256;
  tp_run(a, "tp_sum", ks.half, g1, l1);
  ZE_CHECK(zeCommandListAppendBarrier(zrt::imm(), evH, 0, nullptr));
  ZE_CHECK(zeCommandListAppendBarrier(cp, nullptr, 1, &evH));
  for (int p = 0; p < P; ++p) {
    for (int k = 0; k < N; ++k) {
      if (k == r) continue;
      size_t p0, p1;
      piece(k, p, p0, p1);
      if (p1 > p0) ZE_CHECK(zeCommandListAppendMemoryCopy(cp, put_to(k) + p0 * 2, dev + p0 * 2, (p1 - p0) * 2, nullptr, 0, nullptr));
    }
    ZE_CHECK(zeCommandListAppendBarrier(cp, ev_put(p), 0, nullptr));
  }
  // 2. the first pieces of every GPU: the f16 flags are final
  zrt::event_wait(ev_put(0));
  const long base_put = c.put_n[r].fetch_add(1, std::memory_order_acq_rel) + 1;
  c.wait_all(c.put_n.get(), base_put, r, "the first piece of its part");
  bool over = false;
  for (int k = 0; k < N; ++k) over = over || ((volatile int*)c.flags)[k] != 0;
  if (over) {  // (every GPU: its puts finished, then the one-step f32 sum in the same buffer half)
    static std::atomic<long> warned{0};
    if (r == 0 && warned++ < 3) zlog::eprintf("inferzero: a partial sum beyond f16's range: summed in f32 (failsafe)\n");
    zrt::event_wait(ev_put(P - 1));
    c.barrier(r);
    const cl_ulong off = (cl_ulong)h * N * c.slot_floats;
    ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), c.host + off + (size_t)r * stride, a.part->ptr, n * 4, nullptr, 0, nullptr));
    zrt::sync();
    c.barrier(r);
    waited();
    ++c.reduces[r];
    int np = N;
    cl_ulong st = stride;
    set_args(ks.add, 0, a.x, tp_host_, off, nn, np, st);
    tp_run(a, "tp_add", ks.add, g1, l1);
    c.half[r] = h ^ 1;
    return;
  }
  // 3. piece by piece: once every GPU has put it, this GPU's own block summed (tp_rs_sum over the piece's values) and
  //    the piece's sums put by the copy list
  const cl_ulong put_off = p2p ? (cl_ulong)(h * c.half_b / 2) : (cl_ulong)h * N * c.slot_floats * 2, str = stride,
                 res_off = put_off + (cl_ulong)N * stride;
  for (int p = 0; p < P; ++p) {
    if (p > 0) {
      zrt::event_wait(ev_put(p));
      const long t = c.put_n[r].fetch_add(1, std::memory_order_acq_rel) + 1;
      c.wait_all(c.put_n.get(), t, r, "a piece of its part");
    }
    size_t p0, p1;
    piece(r, p, p0, p1);
    if (p2p) tp_inbox_fence(c, r, h);
    if (p1 > p0) {
      int pa = (int)p0, pb = (int)p1;
      set_args(ks.rs_sum, 0, a.x, a.part, shared, put_off, str, N, r, pa, pb, tp_h_);
      tp_run(a, "tp_sum", ks.rs_sum, (p1 - p0 + 255) / 256 * 256, l1);
    }
    ze_event_handle_t es = ev_sum(p);
    ZE_CHECK(zeCommandListAppendBarrier(zrt::imm(), es, 0, nullptr));
    ZE_CHECK(zeCommandListAppendBarrier(cp, nullptr, 1, &es));
    if (p1 > p0) {
      if (!p2p) ZE_CHECK(zeCommandListAppendMemoryCopy(cp, res + p0 * 2, dev + p0 * 2, (p1 - p0) * 2, nullptr, 0, nullptr));
      else
        for (int k = 0; k < N; ++k)
          if (k != r) ZE_CHECK(zeCommandListAppendMemoryCopy(cp, res_to(k) + p0 * 2, dev + p0 * 2, (p1 - p0) * 2, nullptr, 0, nullptr));
    }
    ZE_CHECK(zeCommandListAppendBarrier(cp, ev_res(p), 0, nullptr));
  }
  // 4. piece by piece: once every GPU has put its sums of it, the other blocks' sums added (tp_rs_add_r per block)
  for (int p = 0; p < P; ++p) {
    zrt::event_wait(ev_res(p));
    const long t = c.res_n[r].fetch_add(1, std::memory_order_acq_rel) + 1;
    c.wait_all(c.res_n.get(), t, r, "a piece of its sums");
    if (p2p) tp_inbox_fence(c, r, h);
    for (int k = 0; k < N; ++k) {
      if (k == r) continue;
      size_t p0, p1;
      piece(k, p, p0, p1);
      if (p1 <= p0) continue;
      int ja = (int)p0, jb = (int)p1;
      set_args(ks.rs_add_r, 0, a.x, shared, res_off, ja, jb);
      tp_run(a, "tp_sum", ks.rs_add_r, (p1 - p0 + 255) / 256 * 256, l1);
    }
  }
  waited();
  ++c.reduces[r];
  c.half[r] = h ^ 1;
}

// ---- ARC_TP_OVERLAP (see TpOverlap above)
inline void Engine::tpo_begin(int nch) {
  if (!tp_on() || !tpo_wanted() || tp_moe() || opt.dry) return;
  TpComm& c = *tp_;
  const int r = opt.tp_rank, E = hp.n_embd;
  if (!tpo_) {
    // (everything the sums would make on first use, made here on the shard's own thread: the sum thread allocates
    //  nothing, and finds every buffer it names in its Dev's list of buffers)
    tp_build();
    tp_wrap_host();
    if (!tp_part_) tp_part_ = buf((size_t)max_chunk * E * 4);
    if (!tp_h_) tp_h_ = buf((size_t)max_chunk * E * 2);
    if (!tp_flag_) {
      tp_flag_ = zrt::wrap(c.flags, (size_t)c.n * sizeof(int));
      mems_.push_back(tp_flag_);
    }
    tpo_ = new TpOverlap;
    TpOverlap& o = *tpo_;
    o.parts[0] = tp_part_;
    o.parts[1] = buf((size_t)max_chunk * E * 4);
    o.flag_r = from(tp_flag_, (size_t)r * sizeof(int));
    o.ks = TpKs{cl_.kernel(tp_prog_, "tp_add"), cl_.kernel(tp_prog_, "tp_half"), cl_.kernel(tp_prog_, "tp_rs_sum"),
                cl_.kernel(tp_prog_, "tp_rs_add"), cl_.kernel(tp_prog_, "tp_rs_add_r")};
    zrt::Dev& d = zrt::D();
    o.dev.dev = d.dev;
    o.dev.index = d.index;
    o.dev.guard = &d;
    for (cl_mem m : x_views_) o.dev.mems.insert(m);
    for (cl_mem m : {o.parts[0], o.parts[1], tp_h_, tp_host_, tp_flag_, o.flag_r}) o.dev.mems.insert(m);
    // (an immediate list on the compute group: its second queue when it has one, so the sums' kernels run beside the
    //  main list's GEMMs; else the same queue, still correct)
    uint32_t ng = 0;
    ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(d.dev, &ng, nullptr));
    std::vector<ze_command_queue_group_properties_t> gp(ng);
    for (auto& g : gp) g.stype = ZE_STYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(d.dev, &ng, gp.data()));
    uint32_t ord = 0, idx = 0;
    for (uint32_t i = 0; i < ng; ++i)
      if (gp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
        ord = i;
        idx = gp[i].numQueues > 1 ? 1 : 0;
        break;
      }
    ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, ord, idx, 2 /* in order */, 2 /* asynchronous */, 0};
    ZE_CHECK(zeCommandListCreateImmediate(zrt::ctx, d.dev, &qd, &o.dev.cl));
    zlog::eprintf("inferzero: GPU %d: ARC_TP_OVERLAP: the prompt's sums on a thread of their own, %s (queue group %u, queue %u)\n",
                  d.index, idx ? "the second compute queue" : "the same compute queue", ord, idx);
    o.th = std::thread([this] { tpo_loop(); });
  }
  TpOverlap& o = *tpo_;
  if (o.used) {  // (the last pass's events: both lists idle once the main list is, as every sum was waited for)
    zrt::sync();
    for (size_t i = 0; i < o.used; ++i) ZE_CHECK(zeEventHostReset(o.evs[i]));
    o.used = 0;
  }
  o.pend.assign(nch, nullptr);
  o.part_done[0] = o.part_done[1] = nullptr;
  o.next_part = 0;
  o.chunk = 0;
  o.live = true;
}
inline bool Engine::tpo_out(const char* name, const GpuWeight& w, int Mp, int act32) {
  TpOverlap& o = *tpo_;
  if (!o.live) return false;
  {
    std::lock_guard<std::mutex> lk(o.mu);
    if (o.err) {  // (a sum failed: stop the pass here; the guard waits for the queue, TpoPass)
      std::exception_ptr e = o.err;
      o.err = nullptr;
      std::rethrow_exception(e);
    }
  }
  // (rows the sum thread does not know (not a chunk view of x_all_): this sum on the main thread, as without the
  //  switch, once the queue is empty, so the sums stay in order)
  if (!o.dev.mems.count(x_) || o.chunk < 0 || o.chunk >= (int)o.pend.size()) {
    tpo_join();  // (also the last sums' kernels on the GPU: they use tp_h_)
    return false;
  }
  const int s = o.next_part;
  o.next_part ^= 1;
  if (o.part_done[s]) tpo_wait(o.part_done[s]);
  matmul(name, w, Mp, o.parts[s], hp.n_embd, 0, 0, act32);
  TpOverlap::Job j;
  j.rows = cur_T_;
  j.x = x_;
  j.part = o.parts[s];
  j.ready = o.event();
  j.done = o.event();
  ZE_CHECK(zeCommandListAppendBarrier(zrt::imm(), j.ready, 0, nullptr));
  o.part_done[s] = j.done;
  {
    std::lock_guard<std::mutex> lk(o.mu);
    o.q.push_back(j);
    ++o.queued;
  }
  o.cv.notify_one();
  if (!strcmp(name, "gemm_down")) {
    if (o.pend[o.chunk]) tpo_wait(o.pend[o.chunk]);
    o.pend[o.chunk] = j.done;  // (waited for when this chunk's next layer starts)
  } else {
    tpo_wait(j.done);  // (the FFN reads these rows next)
  }
  return true;
}
inline void Engine::tpo_chunk(int c) {
  if (!tpo_ || !tpo_->live) return;
  TpOverlap& o = *tpo_;
  o.chunk = c;
  if (c >= 0 && c < (int)o.pend.size() && o.pend[c]) {
    tpo_wait(o.pend[c]);
    o.pend[c] = nullptr;
  }
}
// (the sum thread given up on: the model faulted, so it is never freed while that thread may still use it, TpHold)
inline void Engine::tpo_gave_up() {
  tp_->fault = true;
  int none = -1;
  tp_->first_failed.compare_exchange_strong(none, opt.tp_rank);
  tp_->failed = true;
}
inline void Engine::tpo_join() {
  if (!tpo_ || !tpo_->live) return;
  bool gave_up = false;
  std::exception_ptr e = tpo_->wait_idle(tp_->timeout_s + 30, &gave_up);
  if (gave_up) {
    tpo_gave_up();
    std::rethrow_exception(e);
  }
  tpo_gpu_done();
  if (e) std::rethrow_exception(e);
}
// (the main list waits for the last two sums on the GPU: after them the sum list is done with every buffer, so what
//  the main list runs next (a cancel's state restore, a sum of its own, the next generation step) cannot overlap them)
inline void Engine::tpo_gpu_done() {
  for (ze_event_handle_t d : tpo_->part_done)
    if (d) tpo_wait(d);
}
// A wait of one list for the other's event: first on the host (the time limit and abort of zrt::event_wait), then on
// the GPU, where it passes at once and has the waiting side see the other's writes (device scope). Never a GPU wait
// for an event not yet signalled: a GPU with one compute queue (this laptop's iGPU) runs both lists in it, and a wait
// there for the other list's later work never ends (seen: both shards stopped until the 120-s limit)
inline void Engine::tpo_wait(ze_event_handle_t e) {
  zrt::event_wait(e);
  ZE_CHECK(zeCommandListAppendBarrier(zrt::imm(), nullptr, 1, &e));
}
inline void Engine::tpo_end() {
  if (!tpo_ || !tpo_->live) return;
  TpOverlap& o = *tpo_;
  bool gave_up = false;
  std::exception_ptr e = o.wait_idle(tp_->timeout_s + 30, &gave_up);
  if (gave_up) {
    o.live = false;
    tpo_gave_up();
    std::rethrow_exception(e);
  }
  for (ze_event_handle_t& p : o.pend)
    if (p) {
      tpo_wait(p);
      p = nullptr;
    }
  tpo_gpu_done();
  o.live = false;
  if (e) std::rethrow_exception(e);
}
inline void Engine::tpo_abandon() {
  if (!tpo_ || !tpo_->live) return;
  TpOverlap& o = *tpo_;
  o.live = false;
  bool pending;
  {
    std::lock_guard<std::mutex> lk(o.mu);
    pending = !o.q.empty() || o.busy;
  }
  if (pending) {  // (this shard stops with sums queued: the others would wait for its later sums; they leave instead)
    TpComm& c = *tp_;
    int none = -1;
    c.first_failed.compare_exchange_strong(none, opt.tp_rank);
    c.failed = true;
  }
  bool gave_up = false;
  o.wait_idle(tp_->timeout_s + 30, &gave_up);  // (its error, if any, is the second one: the pass's own goes on)
  if (gave_up) {
    tpo_gave_up();
    return;
  }
  try {  // (signalled by now, by its list or from the host after a failure; a GPU that hangs: the wait's limit)
    tpo_gpu_done();
  } catch (...) {  // (a destructor's caller: the pass's own error goes on)
  }
}
inline void Engine::tpo_loop() {
  TpOverlap& o = *tpo_;
  TpComm& c = *tp_;
  zrt::cur = &o.dev;
  zrt::tp_guarded = true;
  for (;;) {
    TpOverlap::Job j;
    bool skip;
    {
      std::unique_lock<std::mutex> lk(o.mu);
      o.cv.wait(lk, [&] { return o.quit || !o.q.empty(); });
      if (o.q.empty()) return;
      j = o.q.front();
      o.q.pop_front();
      o.busy = true;
      skip = o.err || c.failed.load();
    }
    std::exception_ptr e;
    if (!skip) {
      try {
        tpo_wait(j.ready);
        tp_reduce_at(j.rows, TpSum{j.x, j.part, true});
        ZE_CHECK(zeCommandListAppendBarrier(zrt::imm(), j.done, 0, nullptr));
      } catch (...) {  // (as TpEngine::loop: the others leave their barriers)
        e = std::current_exception();
        int none = -1;
        c.first_failed.compare_exchange_strong(none, opt.tp_rank);
        c.failed = true;
        skip = true;
      }
    }
    if (skip) ZE_CHECK(zeEventHostSignal(j.done));  // (the main list's waits for it end; the model is broken now)
    std::lock_guard<std::mutex> lk(o.mu);
    if (e && !o.err) o.err = e;
    o.busy = false;
    ++o.finished;
    if (o.q.empty()) o.idle_cv.notify_all();
  }
}
inline void Engine::tpo_free() {
  if (!tpo_) return;
  TpOverlap* o = tpo_;
  if (o->th.joinable()) {
    {
      std::lock_guard<std::mutex> lk(o->mu);
      o->quit = true;
    }
    o->cv.notify_all();
    o->th.join();
  }
  // (its lists drain: every wait there follows a signal of the main list, idle by now (release_all), or of its own)
  for (ze_command_list_handle_t l : {o->dev.cl_tp, o->dev.cl})
    if (l) {
      zeCommandListHostSynchronize(l, UINT64_MAX);
      zeCommandListDestroy(l);
    }
  for (cl_kernel k : {o->ks.add, o->ks.half, o->ks.rs_sum, o->ks.rs_add, o->ks.rs_add_r})
    if (k) clReleaseKernel(k);
  for (ze_event_handle_t e : o->evs) zeEventDestroy(e);
  for (ze_event_pool_handle_t p : o->pools) zeEventPoolDestroy(p);
  for (ze_event_handle_t e : o->dev.tp_ev)
    if (e) zeEventDestroy(e);
  delete o;
  tpo_ = nullptr;
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
    return joined();
  }
  // the full vocabulary's logits after row `row` of the last pass (every shard's keep_rows set before it: runz
  // --kl-ref's prompt path); each shard's head reads its own copy of the row, no sum
  const std::vector<float>& logits_at(int row) {
    run_all([&](int r) { shards_[r]->logits_at(row); });
    return joined();
  }
  const std::vector<float>& joined() {  // (each shard's vocabulary rows in rank order)
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
      zrt::devs[r]->abort = release_stuck || leaked_fault() ? &gone : nullptr;  // (a model left earlier: its threads still leave)
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
    zlog::eprintf("inferzero: %d GPUs%s\n", n, sim ? " (ARC_TP_SIM: all on GPU 0)" : "");
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
    comm_.put_n.reset(new std::atomic<long>[n]);
    comm_.res_n.reset(new std::atomic<long>[n]);
    for (int r = 0; r < n; ++r) comm_.put_n[r] = comm_.res_n[r] = 0;
    for (int r = 0; r < n; ++r) {  // (GPU 0's waits keep no abort: the server's other users, Whisper, share it)
      zrt::devs[r]->sync_timeout_ns = (uint64_t)(comm_.timeout_s * 1e9);
      zrt::devs[r]->fault = &comm_.fault;
      zrt::devs[r]->abort = r > 0 ? &comm_.failed : nullptr;  // (GPU 0: an abort a faulted model left, disarm, cleared)
    }
    stuck_.assign(n, 0);
    shards_.resize(n);
    w_.resize(n);
    for (int r = 0; r < n; ++r) w_[r].th = std::thread([this, r, life = life_] { loop(r, life); });
    // shard 0 first (it compiles and caches the kernels the others then load), then the others one after another:
    // made together (kernels built, the XMX check, buffers allocated on several GPUs of one context at once), 3 and 4
    // B60s ended at random in the driver's "Abort was called at 252 line" (issue #4, 2026-10-31).
    // ARC_TP_PARALLEL_LOAD=1: the others together again (faster to load; for tests)
    const int oom_rank = getenv("ARC_TP_OOM") ? atoi(getenv("ARC_TP_OOM")) : -1;
    auto make = [&, o, life = life_](int r) mutable {
      EngineOptions so = o;
      so.tp_rank = r;
      so.tp_size = n;
      zrt::oom_throws = true;  // (failsafe: a shard short of memory throws, the server loads on one GPU instead)
      zrt::oom_test = r == oom_rank;
      std::unique_ptr<Engine> made;
      try {
        made = std::make_unique<Engine>(path, so);
      } catch (...) {
        zrt::oom_throws = zrt::oom_test = false;
        throw;
      }
      zrt::oom_throws = zrt::oom_test = false;
      std::lock_guard<std::mutex> alive(life->mu);  // (given up on while it loaded: its TpEngine may be gone)
      if (life->gone) {
        made.release();  // (left, as a stuck shard is)
        return;
      }
      made->tp_ = &comm_;
      shards_[r] = std::move(made);
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
    choose_sums(sim);
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
      // (the shards taken out here, so a free that is given up on touches nothing of this TpEngine when it ends; one
      //  limit for all the frees, not the 10 s after a failure that wait_done gives passes, nor none: a free after a
      //  failure took longer than 10 s and its thread was left on a TpEngine then freed, or a GPU that stopped while
      //  freeing hung the unload for good. Audit 2026-11-14)
      std::vector<int> made;
      std::vector<Engine*> raw(shards_.size(), nullptr);
      for (int r = 0; r < (int)shards_.size(); ++r)
        if (shards_[r] && !is_stuck(r)) {
          made.push_back(r);
          raw[r] = shards_[r].release();
        }
      free_limit_s_ = comm_.timeout_s + 30;
      try {
        if (!made.empty()) {
#ifdef _WIN32
          run_on(made, [raw](int r) { delete raw[r]; }, true, false);
#else
          // (Linux: one shard at a time, each still on its own thread: Intel's Linux runtime faulted while two
          //  shards freed their kernels and modules at once, B70 with ARC_TP_SIM, pull request #24 by chumbleyjl.
          //  One deadline for all of them; after a free given up on, the shards left are left too, as its thread
          //  may still be freeing and a second free at the same time is what faulted)
          const auto until = std::chrono::steady_clock::now() + std::chrono::duration<double>(free_limit_s_);
          for (int r : made) {
            const double left = std::chrono::duration<double>(until - std::chrono::steady_clock::now()).count();
            if (left <= 0) break;
            free_limit_s_ = left;
            try {
              run_on({r}, [raw](int q) { delete raw[q]; }, true, false);
            } catch (...) {
              break;
            }
          }
#endif
        }
      } catch (...) {}
      free_limit_s_ = 0;
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
      if (any_stuck) {
        {
          std::lock_guard<std::mutex> g(life_->mu);
          life_->gone = true;  // (the threads left running touch nothing of this TpEngine when their jobs end)
        }
        disarm(true);  // (no other split model while they run; their GPU waits leave at the next piece)
      }
      return;
    }
    shards_.clear();  // (after a failure: whatever is left, freed here)
    if (comm_.host) zrt::mem_free(zrt::ctx, comm_.host);
    if (comm_.flags) zrt::mem_free(zrt::ctx, comm_.flags);
    for (char* p : comm_.inbox)
      if (p) zrt::mem_free(zrt::ctx, p);
    comm_.inbox.clear();
    comm_.p2p = false;
    comm_.flags = nullptr;
    comm_.host = nullptr;
  }
  // one copy's speed in GB/s on this GPU's copy list (the one the prompt sums use): the best of 4 after 2 warm-ups
  // (a copy the driver refuses: -1, the link not used. One that does not finish within 20 s: an error that says how
  //  to load without the direct path; a peer link the driver reports but the IOMMU blocks may do either)
  struct CopyStuck : std::runtime_error {
    using std::runtime_error::runtime_error;
  };
  static double link_gbps(void* dst, const void* src, size_t b, const char* what) {
    ze_command_list_handle_t l = zrt::tp_copy_list();
    double best = 1e30;
    for (int i = 0; i < 6; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      if (zeCommandListAppendMemoryCopy(l, dst, src, b, nullptr, 0, nullptr) != ZE_RESULT_SUCCESS) return -1;
      const ze_result_t w = zeCommandListHostSynchronize(l, 20000000000ull);
      if (w != ZE_RESULT_SUCCESS)
        throw CopyStuck(std::string("--gpus: a timed copy ") + what + " did not finish (0x" +
                                 [&] { char h[16]; snprintf(h, sizeof h, "%x", (unsigned)w); return std::string(h); }() +
                                 "): restart with ARC_TP_P2P=0 to keep the sums in host memory");
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (i >= 2) best = std::min(best, s);
    }
    return b / std::max(best, 1e-9) / 1e9;
  }
  // How the sums cross between GPUs on this machine, from its links timed once here (16 MB copies, one GPU at a time):
  // through pinned host memory, which every GPU reaches, or straight into the other GPUs' memory (TpComm::p2p) when
  // the driver reports peer access between every pair and the slowest direct copy is at least 0.8x the slowest
  // direction of the host links. Both, and the choice, go to the log. (Why direct wins at equal links: through host
  // memory a GPU's kernels read the other GPUs' parts across PCIe while they compute; direct, they read their own
  // memory. Two Pro B60s at 14.1 GB/s direct and 14.2 through host, issue #17: the 27B's 4K prompt 1,384-1,442 ->
  // 1,578-1,630 tok/s, generation 36.1 -> 37.8, the logits identical; the first rule, 1.1x, kept them on host memory)
  // ARC_TP_P2P=0: host memory always; =1: direct whenever peer access is there (under ARC_TP_SIM too, where every
  // inbox is on GPU 0: a test of the data flow). Not with ARC_TP_OVERLAP (its sum thread's copies stay as they are).
  // Failsafe: no memory for the inboxes (2 x (N + 1) x chunk x n_embd x 2 bytes per GPU), host memory.
  void choose_sums(bool sim) {
    const int n = n_;
    const char* e = getenv("ARC_TP_P2P");
    const int want = e && *e ? atoi(e) : -1;  // (-1: by the links)
    std::vector<int> all;
    for (int r = 0; r < n; ++r) all.push_back(r);
    bool peer = sim || zeDeviceCanAccessPeer != nullptr;
    for (int i = 0; i < n && peer && !sim; ++i)
      for (int j = 0; j < n && peer; ++j) {
        uint8_t v = 0;
        if (i != j && (zeDeviceCanAccessPeer(zrt::devs[i]->dev, zrt::devs[j]->dev, &v) != ZE_RESULT_SUCCESS || !v)) peer = false;
      }
    // the links, timed (not under ARC_TP_SIM: one GPU's copies to itself say nothing)
    const size_t tb = std::min((size_t)16 << 20, (size_t)2 * n * comm_.slot_floats * 4);
    std::vector<char*> tmp(n, nullptr);
    std::vector<double> d2h(n, 0), h2d(n, 0);
    std::vector<std::vector<double>> pp(n, std::vector<double>(n, 0));
    bool timed = false, refused = false;  // (refused: peer access reported, a direct copy refused)
    // (ARC_TP_LINKS_SIM=1: timed under ARC_TP_SIM too, GPU 0 to itself: a test of the timing on one GPU)
    const bool time_sim = getenv("ARC_TP_LINKS_SIM") && atoi(getenv("ARC_TP_LINKS_SIM")) != 0;
    if ((!sim || time_sim) && want != 0) {
      struct Free {  // (also when a timed copy throws; not when one did not finish: it may still write them)
        std::vector<char*>& v;
        bool keep = false;
        ~Free() {
          if (keep) return;
          for (char* p : v)
            if (p) zrt::mem_free(zrt::ctx, p);
        }
      } free_tmp{tmp};
      // (one GPU at a time, as the shards are made: allocations on several GPUs at once ended in the driver's abort
      //  on 3 and 4 B60s, issue #4)
      for (int q = 0; q < n; ++q)
        run_on({q}, [&](int r) {
          zrt::oom_throws = true;
          try { tmp[r] = (char*)zrt::device_alloc(tb); } catch (...) {}
          zrt::oom_throws = false;
        }, false, false);
      timed = std::all_of(tmp.begin(), tmp.end(), [](char* p) { return p != nullptr; });
      if (timed) {
        try {
          for (int q = 0; q < n; ++q)
            run_on({q}, [&](int r) {
              d2h[r] = link_gbps(comm_.host, tmp[r], tb, "to host memory");
              h2d[r] = link_gbps(tmp[r], comm_.host, tb, "from host memory");
              for (int k = 0; k < n && peer; ++k)
                if (k != r) pp[r][k] = link_gbps(tmp[k], tmp[r], tb, "between GPUs");
            }, false, false);
        } catch (const CopyStuck&) {
          // (its copy is still queued on that GPU's list: the buffers and the shards it may touch are left, as after
          //  a GPU fault, and no other split model runs in this process; the server loads on one GPU. Audit 2026-11-14)
          free_tmp.keep = true;
          comm_.fault = true;
          leaked_fault() = true;
          throw;
        }
      }
      for (int r = 0; r < n && timed; ++r) {
        timed = d2h[r] > 0 && h2d[r] > 0;
        for (int k = 0; k < n && peer; ++k)
          if (k != r && pp[r][k] <= 0) peer = false, refused = true;
      }
    }
    double host = 1e30, direct = 1e30;
    for (int r = 0; r < n && timed; ++r) {
      std::string line = "inferzero: links: GPU " + std::to_string(r);
      char b[96];
      snprintf(b, sizeof b, " -> host %.1f GB/s, host -> GPU %d %.1f GB/s", d2h[r], r, h2d[r]);
      line += b;
      host = std::min({host, d2h[r], h2d[r]});
      for (int k = 0; k < n && (peer || refused); ++k) {
        if (k == r) continue;
        snprintf(b, sizeof b, pp[r][k] > 0 ? ", -> GPU %d %.1f GB/s" : ", -> GPU %d refused", k, pp[r][k]);
        line += b;
        direct = std::min(direct, pp[r][k]);
      }
      zlog::eprintf("%s\n", line.c_str());
    }
    bool use = want == 1 ? peer : want == 0 ? false : (peer && timed && direct >= 0.8 * host);
    std::string why;
    char b[160];
    if (want == 0) why = "ARC_TP_P2P=0";
    else if (refused) why = "peer access reported, but a direct copy was refused";
    else if (!peer) why = zeDeviceCanAccessPeer ? "the driver reports no peer access between them" : "this Level Zero has no peer query";
    else if (want == 1) why = sim ? "ARC_TP_P2P=1, ARC_TP_SIM" : "ARC_TP_P2P=1";
    else if (sim && !time_sim) why = "ARC_TP_SIM: one GPU; ARC_TP_P2P=1 tests the direct path";
    else if (!timed) why = "the links could not be timed";
    else snprintf(b, sizeof b, "%.1f GB/s directly against %.1f through host memory", direct, host), why = b;
    if (use && tpo_wanted()) use = false, why = "ARC_TP_OVERLAP keeps them in host memory";
    if (use) {
      comm_.half_b = (size_t)(n + 1) * comm_.slot_floats * 2;
      comm_.inbox.assign(n, nullptr);
      for (int q = 0; q < n; ++q)  // (one GPU at a time, as above)
        run_on({q}, [&](int r) {
          zrt::oom_throws = true;
          // (the first pass's sum buffers first: the memory plan counts them, not the inbox, which takes what is left
          //  or the sums stay in host memory. Before, the inbox was made first and took their room, audit 2026-11-14)
          try {
            shards_[r]->tp_reserve_sums();
            comm_.inbox[r] = (char*)zrt::device_alloc(2 * comm_.half_b);
          } catch (...) {}
          zrt::oom_throws = false;
        }, false, false);
      if (!std::all_of(comm_.inbox.begin(), comm_.inbox.end(), [](char* p) { return p != nullptr; })) {
        for (char* p : comm_.inbox)
          if (p) zrt::mem_free(zrt::ctx, p);
        comm_.inbox.clear();
        snprintf(b, sizeof b, "no GPU memory for the inboxes (%.0f MB each)", 2 * comm_.half_b / 1e6);
        use = false, why = b;
      }
    }
    comm_.p2p = use;
    zlog::eprintf("inferzero: the GPUs' sums go %s (%s)\n", use ? "straight from GPU to GPU" : "through host memory", why.c_str());
    // (a slow link: the prompt's sums wait on it; docs/multi-gpu.md says what helps)
    const double eff = use ? direct : host;
    if (timed && eff < 4.0)
      zlog::eprintf("inferzero: the slowest link measures %.1f GB/s: long prompts will wait on their sums (a PCIe x4 or "
                    "chipset slot?)\n", eff);
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
  // A thread given up on may finish its job long after (a free that took 10 s after a failure, a GPU that answers late):
  // what it touches of this TpEngine afterwards, it touches under life_->mu and only while gone is false. shutdown sets
  // gone when it leaves such a thread, so the TpEngine can be freed (by its destructor, or a constructor that threw)
  // without the thread writing into freed memory. Before the audit of 2026-11-14 it took mu_ and read stuck_ there.
  struct Life {
    std::mutex mu;
    bool gone = false;
  };
  std::shared_ptr<Life> life_ = std::make_shared<Life>();
  double free_limit_s_ = 0;  // (shutdown's frees: one limit for all of them, wait_done)

  // The workers' jobs waited for. Failsafe: once a shard failed, the others leave their barriers and GPU waits within
  // 100 ms; a thread still running 10 s later is stuck in the driver (its GPU hangs): it is given up on (left running,
  // detached at shutdown), the call ends with an error for it and the split model is broken until loaded again.
  // watch (the lockstep calls: passes, slots, the server's mirrored calls): when no barrier completes for timeout_s +
  // 30 s, the same, without any shard having failed: a shard stuck inside a driver call throws nothing and sets
  // nothing, and the request waited forever (self audit MG1). Not for loads and frees, which meet at no barrier.
  void wait_done(std::unique_lock<std::mutex>& lk, bool watch = true) {
    std::chrono::steady_clock::time_point since{}, moved = std::chrono::steady_clock::now();
    const auto started = moved;
    // (progress: barriers completed, and the pipelined sums' pieces, ARC_TP_PIPE, which meet at no barrier: a long
    //  prompt of such sums is not a hang)
    auto progress = [this] {
      long v = comm_.gen.load(std::memory_order_acquire);
      for (int r = 0; r < n_ && comm_.put_n; ++r) v += comm_.put_n[r].load(std::memory_order_relaxed) + comm_.res_n[r].load(std::memory_order_relaxed);
      return v;
    };
    long last_gen = progress();
    bool warned = false;
    while (!done_cv_.wait_for(lk, std::chrono::seconds(1), [&] { return pending_ == 0; })) {
      const auto now = std::chrono::steady_clock::now();
      if (free_limit_s_ > 0) {  // (shutdown's frees: given up on only after free_limit_s_, failure or not)
        if (now - started < std::chrono::duration<double>(free_limit_s_)) continue;
        comm_.failed = true;
        for (int r = 0; r < n_; ++r)
          if (w_[r].has) {
            stuck_[r] = 1;
            w_[r].has = false;
            w_[r].err = std::make_exception_ptr(std::runtime_error("multi-GPU: GPU " + std::to_string(r) + " did not finish freeing within " +
                                                                    std::to_string((int)free_limit_s_) + " s"));
            zlog::eprintf("inferzero: GPU %d did not finish freeing within %d s: given up on (restart to free its memory)\n", r,
                          (int)free_limit_s_);
          }
        pending_ = 0;
        comm_.fault = true;
        broken_ = true;
        return;
      }
      const long g = progress();
      if (g != last_gen) last_gen = g, moved = now, warned = false;
      // (a line first, 30 s before giving up: a slow path between two sums can be told from a hang in the log, R9)
      if (watch && !warned && now - moved > std::chrono::duration<double>(comm_.timeout_s)) {
        warned = true;
        zlog::eprintf("inferzero: multi-GPU: no sum completed for %d s; given up on in 30 s more\n", (int)comm_.timeout_s);
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
          zlog::eprintf("inferzero: GPU %d does not answer: given up on\n", r);
        }
      pending_ = 0;
      comm_.fault = true;
      broken_ = true;
      return;
    }
  }

  void loop(int r, std::shared_ptr<Life> life) {
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
      bool threw = false;
      try {
        job(r);
      } catch (const Engine::Cancelled&) {
        err = std::current_exception();  // (served: every shard cancels at the same point, tp_cancelled)
      } catch (...) {
        err = std::current_exception();
        threw = true;
      }
      std::lock_guard<std::mutex> alive(life->mu);  // (given up on and its TpEngine freed since: touch nothing of it)
      if (life->gone) return;
      if (threw) {
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
        zlog::eprintf("multi-GPU: GPU %zu: %s\n", k + 1, x.what());
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

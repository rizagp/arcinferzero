// inferzero multi-GPU: the via-host copy between two GPUs that cannot reach each other's memory (Arc B60s on
// Windows: no peer access), in pieces and on two threads. Header-only: the probe (zero/mgpu_probe.cpp) and the
// benchmark (bench/hostcopy_bench.cpp) each build it into their own program.
//
// The old path copies the whole block GPU A -> host, waits for A on the host, then host -> GPU B (via_host_us, in
// zero/mgpu_probe.cpp, which measures it): two ~13.6 GB/s links one after the other, 6.44 ms for 40 MB on two B60s
// (6.5 GB/s of payload). HostCopy cuts the block into `chunk` pieces kept in a ring of `slots` staging pieces and
// copies with two threads, each on its own command list (Level Zero: never one command list from two threads at
// once, ze_api.h zeCommandListAppendMemoryCopy), so the two directions -- two different links -- run at the same
// time: 3.47 ms for 40 MB, 12.1 GB/s, 1.85x.
//
// Piece size, measured on two B60s (1 sync per piece, 8 slots, compute queues; copy engines within 2%):
//          256 KB   512 KB   1 MB    2 MB    4 MB    8 MB    16 MB   32 MB
//   40 MB   4.46     3.87    3.56    3.47    3.51    3.77    4.42    (1 piece: no)
//  256 MB  28.49    24.34   22.04   21.16   20.74   20.66   21.18   22.26
// so 2 MB is the default and 256 KB the least init() takes (kMinChunk).
//
// What the whole copy against the pieces measured (same machine):
//   bytes    whole    pieces   piece
//   128 KB   32.8 us    -         -   (the pieces lose below 384 KB)
//   256 KB   53.2       -         -   (exactly even: 53.2 both ways, so copy() keeps it whole)
//   384 KB   73.4      70.8      192 KB
//   512 KB   91.7      81.6      256 KB
//   1 MB    166        135       256 KB
//   2 MB    318        243       256 KB
//   4 MB    619        418       512 KB
//   8 MB     1228      770       1 MB
//   16 MB    2520      1462      2 MB
//   40 MB    6139      3424      2 MB
//   256 MB   ~40000    20660     2 MB
// Hence copy() splits a copy of 384 KB and up (piece_for: about bytes/8, at least kMinChunk, at most the ring's
// chunk, never fewer than two pieces), and copies smaller blocks whole. A 20 KB token sync (17 us, latency-bound)
// is not for this path at all.
//
// The two threads used to cost ~50 us of fixed overhead per copy (a std::barrier to start and to end it), which is
// why nothing below 4 MB was pipelined before; they now wait on a counter they spin for a few microseconds, and
// sleep on a condition variable only when no copy follows soon, which is what made the small copies pay.
//
// The pieces are handed over with a CPU flag per ring slot: a GPU-side wait on the other GPU's event is not
// something Level Zero offers across devices (ze_api.h: the events "must be accessible by the device on which the
// command list was created", and the two lists are on different devices), so each thread waits for the other with
// a spin on the flag between its own host syncs.
#pragma once
#include "../ze_min.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace hostcopy {

inline double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// (the elapsed times are float: 6 significant digits are plenty for microseconds; the timestamps above stay double)
inline float median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// The old path: A -> host on A's list, then host -> B on B's list, both whole (the path without peer access).
// (kept in the probe that measures it: zero/mgpu_probe.cpp, via_host_us)

// The same copy in pieces, still on one thread: A -> host, host -> B per piece, each piece synced. This only shows
// what cutting the block up costs (the two directions still take turns); it is the baseline for HostCopy.
inline float via_host_chunked_us(ze_command_list_handle_t a, ze_command_list_handle_t b, void* dst, void* stage,
                                 const void* src, size_t bytes, size_t chunk, int reps) {
  std::vector<float> t;
  const int warm = std::min(10, reps);
  for (int r = 0; r < reps + warm; ++r) {
    const double t0 = now_us();
    for (size_t off = 0; off < bytes; off += chunk) {
      const size_t len = std::min(chunk, bytes - off);
      if (zeCommandListAppendMemoryCopy(a, stage, (const char*)src + off, len, nullptr, 0, nullptr) != 0) return -1;
      if (zeCommandListHostSynchronize(a, UINT64_MAX) != 0) return -1;
      if (zeCommandListAppendMemoryCopy(b, (char*)dst + off, stage, len, nullptr, 0, nullptr) != 0) return -1;
      if (zeCommandListHostSynchronize(b, UINT64_MAX) != 0) return -1;
    }
    if (r >= warm) t.push_back((float)(now_us() - t0));
  }
  return median(t);
}

// The pieces on two threads, one thread per direction, with the threads started once and kept for every copy.
//
//   HostCopy c;
//   if (c.init(ctx, list_a, list_b)) {                   // 2 MB pieces, 4 ring slots, a sync a piece (defaults)
//     c.copy(dst, src, bytes);                           // one copy: it picks pieces or the whole block by size
//     const float us = c.benchmark_us(dst, src, bytes, 200);  // timed, median of 200 (the probe, pieces path)
//     c.term();
//   }
//
// init() allocates pinned host memory for the ring (slots x piece) and starts the two threads; term() stops them
// and frees it. One HostCopy copies one direction pair at a time, from one thread.
struct HostCopy {
  static constexpr size_t kDefaultChunk = 2 << 20;  // bytes per piece unless init() is given another

  // The smallest piece init() takes. Measured on two B60s, 40 MB: pieces of 256 KB still gain (9.4 GB/s against
  // 6.8 for the whole copy), 128 KB only 7.1, 64 KB less than the whole copy: below 256 KB the ~8 us host sync of
  // each piece costs more than the overlap saves.
  static constexpr size_t kMinChunk = 256 << 10;

  // The smallest copy copy() splits: 384 KB. Below that the pieces are exactly even (256 KB: 53.2 us both ways on
  // two B60s -- two 128 KB pieces overlap so briefly that the hand-over costs what it saves) or worse, and from
  // 448 KB up they gain (1.07x), 512 KB 1.11x, 768 KB 1.19x.
  static constexpr size_t kMinPipelined = 384 << 10;

  // Starts the two threads and the ring. Lists: one per direction, different handles (the same from both threads is
  // what Level Zero forbids); `a` copies GPU A -> host, `b` host -> GPU B. `piece` bytes per piece (at least
  // kMinChunk: smaller pieces lose more in host syncs than the overlap wins, so init refuses them), `ring` slots,
  // `each` pieces per host sync (1 = sync per piece; up to ring / 2).
  bool init(ze_context_handle_t context, ze_command_list_handle_t a, ze_command_list_handle_t b,
            size_t piece = kDefaultChunk, int ring = 4, int each = 1) {
    term();
    if (!context || !a || !b || piece < kMinChunk || ring < 2 || each < 1 || each > ring / 2) return false;
    ctx = context;
    cl_a = a;
    cl_b = b;
    chunk = piece;
    slots = ring;
    sync_every = each;
    ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
    if (zeMemAllocHost(ctx, &hd, (size_t)slots * chunk, 64, &stage) != 0) {
      stage = nullptr;
      return false;
    }
    full = std::vector<std::atomic<int>>((size_t)slots);
    for (auto& f : full) f.store(0);
    rc_.store(0);
    quit_.store(0);
    gen_.store(0);
    done_.store(0);
    waiting_.store(0);
    running_ = true;
    prod_ = std::thread(&HostCopy::work_, this, true);
    cons_ = std::thread(&HostCopy::work_, this, false);
    return true;
  }

  // Stops the threads and frees the ring. Safe to call at any time, twice, or after a failed init().
  void term() {
    if (running_) {
      quit_.store(1, std::memory_order_release);
      gen_.fetch_add(1, std::memory_order_acq_rel);  // (wake them for their last look)
      if (waiting_.load(std::memory_order_acquire) > 0) cv_.notify_all();
      prod_.join();
      cons_.join();
      running_ = false;
    }
    if (stage) zeMemFree(ctx, stage);
    stage = nullptr;
    full.clear();
    rc_.store(0);
    quit_.store(0);
  }

  // The piece size copy() uses for a copy of `bytes` -- 0 when the whole-block copy is the faster one. Measured on
  // two B60s (ring of 4, 1 sync per piece), microseconds:
  //   bytes    128 KB  256 KB  384 KB  512 KB  1 MB  2 MB  4 MB  8 MB  16 MB  40 MB  256 MB
  //   whole      32.8    53.5    70.9    91.7   166   318   619  1228   2520   6139   ~40000
  //   pieces        -       -     66.1    81.6   135   243   418   770   1462   3424   20660
  // so the pieces win from 384 KB up, and the best piece is about bytes/8 (192 KB for 384 KB, 256 KB for
  // 1-2 MB, 512 KB for 4 MB, 1 MB for 8 MB), at most the ring's own piece, and never so big that the copy holds
  // fewer than two pieces.
  size_t piece_for(size_t bytes) const {
    if (bytes < kMinPipelined) return 0;
    size_t piece = std::min(chunk, std::max(kMinChunk, bytes / 8));
    if (piece * 2 > bytes) piece = bytes / 2;
    return piece;
  }

  // Copies `bytes` from GPU A's memory `src` to GPU B's memory `dst`, with the faster of the two ways for that size.
  // False on an error or if not initialised.
  bool copy(void* dst, const void* src, size_t bytes) {
    if (!running_ || bytes == 0) return false;
    const size_t piece = piece_for(bytes);
    if (piece != 0 && copy_pipelined(dst, src, bytes, piece)) return true;
    return copy_whole(dst, src, bytes);
  }


  ze_context_handle_t ctx = nullptr;
  ze_command_list_handle_t cl_a = nullptr, cl_b = nullptr;  // one per thread
  void* stage = nullptr;                                    // pinned host memory: slots x chunk pieces
  size_t chunk = kDefaultChunk;  // 2 MB (measured best for a 40 MB copy: 3.47 ms; 1 MB 3.56, 4 MB 3.51, 8 MB 3.77;
                                 // and for 256 MB within 2% of the best: 8 MB 20.66 ms, 2 MB 21.16)
  int slots = 4;                 // ring depth: with a sync per piece the producer cannot run more than ~1 piece
                                 // ahead (it waits for its own copy), so 2 pieces already keep both links busy:
                                 // measured 40 MB/2 MB pieces 3.478 (ring 2), 3.462 (3), 3.448 (4), 3.453 (8).
                                 // 4 leaves slack and gives copy_whole a staging block of 4 x chunk. `each`
                                 // pieces share one sync, so `each` <= slots / 2: 8 for the 4-per-sync variant.
  int sync_every = 1;            // pieces per host sync per thread

  std::vector<std::atomic<int>> full;  // per slot: 1 = holds its piece (producer done), 0 = free again

  // Would the pipelined path be the faster one for `bytes`? (copy() then uses the piece piece_for() gives)
  bool pays(size_t bytes) const { return piece_for(bytes) != 0; }

  bool ok() const { return running_; }

  ~HostCopy() { term(); }
  HostCopy() = default;
  HostCopy(const HostCopy&) = delete;
  HostCopy& operator=(const HostCopy&) = delete;

  // Straightforward, one thread: the whole block GPU A -> host, the host waiting for it, then host -> GPU B and waiting
  // again (6.8 GB/s at 40 MB on two B60s against 12.1 for the pipelined path). `stage` holds the piece in
  // between, so the copy must fit in the ring.
  bool copy_whole(void* dst, const void* src, size_t bytes) {
    if (!running_ || bytes == 0 || bytes > (size_t)slots * chunk) return false;
    if (zeCommandListAppendMemoryCopy(cl_a, stage, src, bytes, nullptr, 0, nullptr) != 0) return false;
    if (zeCommandListHostSynchronize(cl_a, UINT64_MAX) != 0) return false;
    if (zeCommandListAppendMemoryCopy(cl_b, dst, stage, bytes, nullptr, 0, nullptr) != 0) return false;
    if (zeCommandListHostSynchronize(cl_b, UINT64_MAX) != 0) return false;
    return true;
  }

  // The copy in pieces on two threads (at least two pieces: with fewer there is nothing to overlap). `piece` overrides
  // the piece size for this copy (the ring only has to be big enough: any piece up to `chunk` fits); 0 = `chunk`.
  // False on an error or if not initialised.
  bool copy_pipelined(void* dst, const void* src, size_t bytes, size_t piece = 0) {
    piece = piece != 0 ? piece : chunk;
    if (!running_ || piece > chunk || bytes < piece * 2) return false;
    cur_dst_ = dst;
    cur_src_ = src;
    cur_bytes_ = bytes;
    cur_piece_ = piece;
    return round_();
  }

  // Benchmarking: the pipelined path repeated, median of `reps` copies after `warm` warm-ups (us; < 0 on error).
  float benchmark_us(void* dst, const void* src, size_t bytes, int reps, int warm = 10, size_t piece = 0) {
    piece = piece != 0 ? piece : chunk;
    if (!running_ || piece > chunk || bytes < piece * 2 || reps < 1) return -1;
    cur_dst_ = dst;
    cur_src_ = src;
    cur_bytes_ = bytes;
    cur_piece_ = piece;
    std::vector<float> t;
    t.reserve((size_t)reps);
    for (int r = 0; r < reps + warm && rc_.load() == 0; ++r) {
      const double t0 = now_us();
      const bool ok = round_();
      if (!ok) return -1;
      if (r >= warm) t.push_back((float)(now_us() - t0));
    }
    return t.empty() ? -1 : median(t);
  }

 private:
  // One round: both threads copy cur_bytes_ in cur_piece_ pieces, then come back for the next round. The handshake
  // is one atomic counter they spin on (a copy follows a copy within microseconds in an engine, and a barrier's
  // sleep costs ~50 us to wake from, which is a whole 256 KB copy); after a while of spinning they sleep on the
  // condition variable instead, so an idle HostCopy does not hold two cores.
  bool round_() {
    done_.store(0, std::memory_order_relaxed);
    gen_.fetch_add(1, std::memory_order_acq_rel);
    if (waiting_.load(std::memory_order_acquire) > 0) {
      { std::lock_guard<std::mutex> lock(m_); }  // (they take m_ to sleep: wait for them to be asleep, then wake)
      cv_.notify_all();
    }
    while (done_.load(std::memory_order_acquire) < 2) YieldProcessor();
    return rc_.load() == 0;
  }

  // One thread's half: `producer` copies GPU A -> host into the ring, the other host -> GPU B out of it.
  void work_(bool producer) {
    const ze_command_list_handle_t cl = producer ? cl_a : cl_b;
    const int want = producer ? 0 : 1;  // (a slot this thread waits for: free to fill, or filled to send out)
    unsigned seen = 0;
    for (;;) {
      // wait for the next round
      int spins = 0;
      while (gen_.load(std::memory_order_acquire) == seen) {
        if (quit_.load(std::memory_order_acquire)) return;
        if (++spins < 20000) {
          YieldProcessor();
          continue;
        }
        std::unique_lock<std::mutex> lock(m_);
        ++waiting_;
        cv_.wait(lock, [&] {
          return gen_.load(std::memory_order_acquire) != seen || quit_.load(std::memory_order_acquire);
        });
        --waiting_;
        spins = 0;
      }
      if (quit_.load(std::memory_order_acquire)) return;
      seen = gen_.load(std::memory_order_acquire);
      if (rc_.load(std::memory_order_acquire) == 0) {
        const size_t piece = cur_piece_;
        const size_t n = (cur_bytes_ + piece - 1) / piece;
        for (size_t base = 0; base < n && rc_.load(std::memory_order_acquire) == 0; base += (size_t)sync_every) {
          const size_t g = std::min<size_t>((size_t)sync_every, n - base);
          // every slot of this group: free (producer) / filled (consumer)
          for (size_t k = 0; k < g && rc_.load(std::memory_order_acquire) == 0; ++k) {
            std::atomic<int>& f = full[(base + k) % (size_t)slots];
            while (f.load(std::memory_order_acquire) != want && rc_.load(std::memory_order_acquire) == 0)
              YieldProcessor();
          }
          if (rc_.load(std::memory_order_acquire) != 0) break;
          for (size_t k = 0; k < g; ++k) {
            const size_t i = base + k, off = i * piece, len = std::min(piece, cur_bytes_ - off);
            void* slot = (char*)stage + (i % (size_t)slots) * chunk;  // (slots stay `chunk` apart)
            void* d = producer ? slot : (void*)((char*)cur_dst_ + off);
            const void* s = producer ? (const void*)((const char*)cur_src_ + off) : (const void*)slot;
            if (zeCommandListAppendMemoryCopy(cl, d, s, len, nullptr, 0, nullptr) != 0) rc_.store(1);
          }
          if (zeCommandListHostSynchronize(cl, UINT64_MAX) != 0) rc_.store(1);
          for (size_t k = 0; k < g; ++k)
            full[(base + k) % (size_t)slots].store(producer ? 1 : 0, std::memory_order_release);
        }
      }
      done_.fetch_add(1, std::memory_order_acq_rel);
    }
  }

  bool running_ = false;
  const void* cur_src_ = nullptr;
  void* cur_dst_ = nullptr;
  size_t cur_bytes_ = 0;
  size_t cur_piece_ = kDefaultChunk;
  std::thread prod_, cons_;
  std::atomic<unsigned> gen_{0};    // (main: one per round; the threads wait for it to change)
  std::atomic<int> done_{0};        // (the threads: one each when their half is done)
  std::atomic<int> waiting_{0};     // (the threads asleep on the condition variable)
  std::mutex m_;
  std::condition_variable cv_;
  std::atomic<int> rc_{0}, quit_{0};
};

}  // namespace hostcopy

// Multi-GPU (zero/mgpu/): serverz with --gpus drives shard 0 of a split model as its one Engine. Every Engine call
// that changes a conversation's state (forward, forward_batch, bind, reset, snapshots) starts with a TpMirror: when
// the engine is such a shard 0, the same call runs on the other shards' threads at once (they meet at the sums'
// barriers), and at the end their logits are joined into shard 0's, so the server sees the whole vocabulary. On one
// GPU (or a split model that runz drives through TpEngine) the guard is one test of a pointer.
// Included by zero/engine.h before class Engine; start / finish are in zero/mgpu/engine_tp.h.
#pragma once
#include <exception>
#include <functional>
#include <utility>

class Engine;
struct TpMirror {
  enum Gather { kNone, kLogits, kBatch };  // (what to join: logits_host_, or `rows` rows of batch_logits_host_)
  template <class F>
  TpMirror(bool active, Engine* e, F&& f, Gather g = kNone, int rows = 0) {
    if (active) start(e, std::function<void(Engine&)>(std::forward<F>(f)), g, rows);
  }
  ~TpMirror() noexcept(false) {
    if (e_) finish();
  }
  TpMirror(const TpMirror&) = delete;
  TpMirror& operator=(const TpMirror&) = delete;

 private:
  Engine* e_ = nullptr;
  bool nested_ = false;  // (a call inside a mirrored one: the other shards make it themselves)
  Gather g_ = kNone;
  int rows_ = 0;
  int uncaught_ = 0;
  bool was_guarded_ = false;  // (zrt::tp_guarded of the calling thread before: set during the call)
  void start(Engine* e, std::function<void(Engine&)> f, Gather g, int rows);
  void finish();
};

// the first line of a mirrored member function: TP_MIRROR(forward(tokens, n), TpMirror::kLogits, 0)
#define TP_MIRROR(call, gather, rows) \
  TpMirror tp_mirror_(tp_front_ != nullptr, this, [&](Engine& tp_e_) { tp_e_.call; }, gather, rows)

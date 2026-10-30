// Loading progress for whoever starts the server and waits for it (scripts/arc-server.ps1, the launcher): lines
//   progress: P % <what>        (P 0-100; -1: a step without a measure, shown as busy)
// on stderr (the server's log), each step's first and last line and at most one every 0.3 s between. The engine
// counts the weight bytes it uploads (begin / add), the expert file and the expert cache count their layers / bytes
// (step), the speech models their layers.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>

namespace loadp {

struct State {
  std::string what;  // (the step now)
  double done = 0, total = 0;
  int last_pct = -2;
  std::chrono::steady_clock::time_point last{};
  std::string label = "model";  // (what the engine's next load is: "model", "Qwen3-ASR", ...)
  bool on = false;              // (the server turns it on; the command-line tools print their own)
};
inline State& st() {
  static State s;
  return s;
}
inline void print(bool force) {
  State& s = st();
  if (!s.on) return;
  const auto now = std::chrono::steady_clock::now();
  const int pct = s.total > 0 ? (int)(100.0 * std::min(s.done, s.total) / s.total) : -1;
  // (the same percentage of the same step: never twice; a step's first and last line: without the 0.3 s wait)
  if (pct == s.last_pct || (!force && now - s.last < std::chrono::milliseconds(300))) return;
  s.last_pct = pct;
  s.last = now;
  fprintf(stderr, "progress: %d %% %s\n", pct, s.what.c_str());
  fflush(stderr);
}
// a step measured in units of `total` (bytes: shown in GB); done reported as it grows
inline void step(const std::string& what, double done, double total) {
  State& s = st();
  const bool first = s.what != what;
  if (first) s.last_pct = -2;
  s.what = what;
  s.done = done;
  s.total = total;
  print(first || done >= total);
}
// a step without a measure (kernels compiled, an encoder loaded)
inline void busy(const std::string& what) { step(what, 0, 0); }
// the engine's weights: total bytes, then add() per upload
// (the label is used once: an engine that throws while loading leaves "model" for the next one)
inline void begin(double total_bytes) {
  const std::string l = st().label;
  st().label = "model";
  step("loading the " + l + " weights", 0, total_bytes);
}
inline void add(double bytes) {
  State& s = st();
  if (s.total <= 0) return;
  s.done += bytes;
  print(s.done >= s.total);
}
// the next engine load's name ("Qwen3-ASR"); back to "model" after it
inline void label(const std::string& l) { st().label = l; }

}  // namespace loadp

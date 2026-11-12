// GET /metrics: the server's numbers in Prometheus' text format (docs/logging.md), named as vLLM's are with an
// inferzero: prefix in place of vllm:, so a vLLM dashboard reads them with that prefix changed. Numbers only, never a
// prompt's text. Counters and histograms per model (label model_name, the served alias); gauges for now.
#pragma once
#include <algorithm>
#include <cmath>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace zmet {

// (vLLM's buckets)
inline const std::vector<double> kTtft = {0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0,
                                          2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0};
inline const std::vector<double> kE2e = {0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0,
                                         60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0};
inline const std::vector<double> kTpot = {0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5,
                                          5.0, 7.5, 10.0, 20.0, 40.0, 80.0};
inline const std::vector<double> kTok = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000,
                                         100000, 200000, 500000};

struct Hist {
  const std::vector<double>* b = nullptr;
  std::vector<uint64_t> c;  // (per bucket, not cumulative; the last: above every bound)
  double sum = 0;
  uint64_t n = 0;
  explicit Hist(const std::vector<double>& bounds) : b(&bounds), c(bounds.size() + 1, 0) {}
  void add(double v) {
    size_t i = 0;
    while (i < b->size() && v > (*b)[i]) ++i;
    ++c[i];
    sum += v;
    ++n;
  }
};
struct PerModel {
  uint64_t prompt = 0, gen = 0, pc_queries = 0, pc_hits = 0, sd_drafts = 0, sd_accepted = 0;
  std::map<std::string, uint64_t> success;  // finished_reason -> count
  Hist ttft{kTtft}, e2e{kE2e}, tpot{kTpot}, prefill{kE2e}, decode{kE2e}, ptok{kTok}, gtok{kTok};
};
struct Reg {
  std::mutex mu;
  std::map<std::string, PerModel> m;
  std::map<std::tuple<std::string, std::string, int>, uint64_t> http;  // (method, handler, status)
  std::atomic<int> inflight{0}, running{0};  // (chat requests in the server; those past their turn)
  double kv_usage = 0;                       // (the last request's context use, 0..1)
  std::string model, arch;
  int ctx = 0, gpus = 1;
  double gpu_bytes = 0;
};
inline Reg& reg() {
  static Reg r;
  return r;
}
// a chat request in the server (waiting = inflight - running)
struct Inflight {
  Inflight() { ++reg().inflight; }
  ~Inflight() { --reg().inflight; }
};
struct Running {
  Running() { ++reg().running; }
  ~Running() { --reg().running; }
};
inline void loaded(const std::string& model, const std::string& arch, int ctx, int gpus, double gpu_bytes) {
  Reg& r = reg();
  std::lock_guard<std::mutex> lk(r.mu);
  r.model = model, r.arch = arch, r.ctx = ctx, r.gpus = gpus, r.gpu_bytes = gpu_bytes;
  r.m[model];  // (the model's series appear at 0)
}
// (the chat model freed: the gauges describe no model until the next load; the counters stay, per model)
inline void unloaded() noexcept {
  try {
    Reg& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    r.model.clear(), r.arch.clear();
    r.ctx = 0, r.gpus = 1, r.gpu_bytes = 0, r.kv_usage = 0;
  } catch (...) {}
}
struct Done {
  std::string model, finish;
  size_t prompt = 0, cached = 0, gen = 0;
  double ttft_s = 0, e2e_s = 0, prefill_s = 0, decode_s = 0;
  uint64_t drafts = 0, accepted = 0;
  double kv = 0;
};
inline void request_done(const Done& d) {
  Reg& r = reg();
  std::lock_guard<std::mutex> lk(r.mu);
  PerModel& p = r.m[d.model];
  p.prompt += d.prompt, p.gen += d.gen, p.pc_queries += d.prompt, p.pc_hits += d.cached;
  p.sd_drafts += d.drafts, p.sd_accepted += d.accepted;
  ++p.success[d.finish];
  p.ttft.add(d.ttft_s), p.e2e.add(d.e2e_s), p.prefill.add(d.prefill_s), p.decode.add(d.decode_s);
  if (d.gen > 1) p.tpot.add(d.decode_s / (double)(d.gen - 1));
  p.ptok.add((double)d.prompt), p.gtok.add((double)d.gen);
  r.kv_usage = d.kv;
}
// a request the client cancelled (vLLM's "abort"): counted, its partial times kept out of the histograms
inline void aborted(const std::string& model, size_t prompt, size_t cached, size_t gen) noexcept {
  try {
    Reg& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    PerModel& p = r.m[model];
    p.prompt += prompt, p.gen += gen, p.pc_queries += prompt, p.pc_hits += cached;
    ++p.success["abort"];
  } catch (...) {}
}
// (the handler label: the API's paths as they are, everything else "other", so the series stay few)
inline std::string handler_of(const std::string& path) {
  static const char* known[] = {"/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions",
                                "/v1/messages", "/messages", "/v1/responses", "/api/chat", "/api/generate",
                                "/v1/embeddings", "/embeddings", "/v1/models", "/models", "/api/tags",
                                "/v1/audio/transcriptions", "/health", "/props", "/metrics", "/tokenize",
                                "/detokenize", "/slots", "/sessions", "/"};
  for (const char* k : known)
    if (path == k) return k;
  return "other";
}
inline void http_done(const std::string& method, const std::string& path, int status) {
  Reg& r = reg();
  std::lock_guard<std::mutex> lk(r.mu);
  ++r.http[{method, handler_of(path), status}];
}

inline std::string esc(const std::string& s) {  // (a label value)
  std::string o;
  for (char ch : s) {
    if (ch == '\\' || ch == '"') o += '\\';
    if (ch == '\n') { o += "\\n"; continue; }
    o += ch;
  }
  return o;
}
inline std::string num(double v) {  // (Prometheus' spelling: NaN, +Inf, -Inf; the range checked before the cast)
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "+Inf" : "-Inf";
  char b[40];
  if (v < 1e15 && v > -1e15 && v == (double)(int64_t)v) snprintf(b, sizeof b, "%lld", (long long)v);
  else snprintf(b, sizeof b, "%.9g", v);
  return b;
}
inline std::string render() {
  Reg& r = reg();
  std::lock_guard<std::mutex> lk(r.mu);
  std::string o;
  auto head = [&](const char* name, const char* type, const char* help) {
    o += std::string("# HELP inferzero:") + name + " " + help + "\n# TYPE inferzero:" + name + " " + type + "\n";
  };
  const std::string cur = esc(r.model);
  // gauges
  head("num_requests_running", "gauge", "Requests being processed (prompt or generation).");
  o += "inferzero:num_requests_running{model_name=\"" + cur + "\"} " + std::to_string(r.running.load()) + "\n";
  head("num_requests_waiting", "gauge", "Requests waiting for their turn.");
  o += "inferzero:num_requests_waiting{model_name=\"" + cur + "\"} " +
       std::to_string(std::max(0, r.inflight.load() - r.running.load())) + "\n";
  head("kv_cache_usage_perc", "gauge", "Context use of the last request, 0 to 1.");
  o += "inferzero:kv_cache_usage_perc{model_name=\"" + cur + "\"} " + num(r.kv_usage) + "\n";
  head("context_max_tokens", "gauge", "The context the loaded model serves, tokens.");
  o += "inferzero:context_max_tokens{model_name=\"" + cur + "\"} " + std::to_string(r.ctx) + "\n";
  head("gpu_memory_bytes", "gauge", "GPU memory the loaded model takes.");
  o += "inferzero:gpu_memory_bytes{model_name=\"" + cur + "\"} " + num(r.gpu_bytes) + "\n";
  head("model_info", "gauge", "The loaded model, 1.");
  if (!r.model.empty())
    o += "inferzero:model_info{model_name=\"" + cur + "\",arch=\"" + esc(r.arch) + "\",ctx=\"" + std::to_string(r.ctx) +
         "\",gpus=\"" + std::to_string(r.gpus) + "\"} 1\n";
  // counters
  auto counter = [&](const char* name, const char* help, uint64_t PerModel::*f) {
    head(name, "counter", help);
    for (auto& [mn, p] : r.m) o += std::string("inferzero:") + name + "{model_name=\"" + esc(mn) + "\"} " + std::to_string(p.*f) + "\n";
  };
  counter("prompt_tokens_total", "Prompt tokens processed (cached ones included).", &PerModel::prompt);
  counter("generation_tokens_total", "Tokens generated.", &PerModel::gen);
  counter("prefix_cache_queries_total", "Prompt tokens looked up in the prompt caches.", &PerModel::pc_queries);
  counter("prefix_cache_hits_total", "Prompt tokens found there (not processed again).", &PerModel::pc_hits);
  counter("spec_decode_num_draft_tokens_total", "Draft tokens proposed (MTP, DFlash2).", &PerModel::sd_drafts);
  counter("spec_decode_num_accepted_tokens_total", "Draft tokens accepted.", &PerModel::sd_accepted);
  head("request_success_total", "counter", "Requests finished, by finish reason.");
  for (auto& [mn, p] : r.m)
    for (auto& [fr, n] : p.success)
      o += "inferzero:request_success_total{model_name=\"" + esc(mn) + "\",finished_reason=\"" + esc(fr) + "\"} " + std::to_string(n) + "\n";
  // histograms
  auto hist = [&](const char* name, const char* help, Hist PerModel::*f) {
    head(name, "histogram", help);
    for (auto& [mn, p] : r.m) {
      const Hist& h = p.*f;
      const std::string lab = "model_name=\"" + esc(mn) + "\"";
      uint64_t acc = 0;
      for (size_t i = 0; i < h.b->size(); ++i) {
        acc += h.c[i];
        o += std::string("inferzero:") + name + "_bucket{" + lab + ",le=\"" + num((*h.b)[i]) + "\"} " + std::to_string(acc) + "\n";
      }
      acc += h.c.back();
      o += std::string("inferzero:") + name + "_bucket{" + lab + ",le=\"+Inf\"} " + std::to_string(acc) + "\n";
      o += std::string("inferzero:") + name + "_sum{" + lab + "} " + num(h.sum) + "\n";
      o += std::string("inferzero:") + name + "_count{" + lab + "} " + std::to_string(h.n) + "\n";
    }
  };
  hist("time_to_first_token_seconds", "Time from the request's arrival to its first token.", &PerModel::ttft);
  hist("e2e_request_latency_seconds", "Time from arrival to the last token.", &PerModel::e2e);
  hist("time_per_output_token_seconds", "Time per generated token after the first.", &PerModel::tpot);
  hist("request_prefill_time_seconds", "Time processing the prompt.", &PerModel::prefill);
  hist("request_decode_time_seconds", "Time generating.", &PerModel::decode);
  hist("request_prompt_tokens", "Prompt tokens per request.", &PerModel::ptok);
  hist("request_generation_tokens", "Generated tokens per request.", &PerModel::gtok);
  head("http_requests_total", "counter", "HTTP requests answered, by method, handler and status.");
  for (auto& [k, n] : r.http)
    o += "inferzero:http_requests_total{method=\"" + esc(std::get<0>(k)) + "\",handler=\"" + esc(std::get<1>(k)) +
         "\",status=\"" + std::to_string(std::get<2>(k)) + "\"} " + std::to_string(n) + "\n";
  return o;
}

}  // namespace zmet

# The server's log and metrics

The server (`serverz`) writes its log to stderr. `zero-server` and the launcher send it to
`%USERPROFILE%\.inferzero\server.log`. It is written anew at each start, and earlier starts are kept in `logs\` next to it.
`GET /metrics` gives the same numbers in Prometheus' format. The code is in `src/log.h` (the log),
`tools/server_startup.h` (the first lines) and `tools/server_metrics.h` (the metrics).

## A line

```
2026-11-12 14:58:09.716 I [chatcmpl-arc1] prompt 16 tok (cached 0) 0.2 s (68.6 t/s) | gen 16 tok 1.0 s (16.50 t/s) | length | first token 0.23 s | ctx 32/8192 | slot 0
```

- **Prefix:** the local date and time to the millisecond, then the level: `E` error, `W` warning, `I` info, `D` debug.
- **One write per line, under a lock:** a line printed in parts by one thread is held until its end, so another
  thread's line cannot land inside it.
- **Lines without the prefix:** `progress: P % <step>` while a model loads. `zero-server` and the launcher read these for
  their progress bars, so they stay as they were. One is written per 5 points or 2 s, plus each step's first and last
  (~20 a load, ~100 before).

## The options

| Option | What it does |
|---|---|
| `--log-verbosity N`, `-lv N` | 0 errors only, 1 + warnings, 2 + info (default), 3 + debug. Names work too: `error`, `warn`, `info`, `debug`. `ARC_LOG_LEVEL` sets the same. |
| `--log-timestamps on\|off` | the date and time in front (default on); off leaves the level letter |
| `--log-colors auto\|on\|off` | colors for W / E / D; auto: in Windows Terminal only, never in a redirected log; on: always, a redirected `server.log` too |
| `--log-file F` | every line to F as well (appended), except a crash report: that goes to stderr only (`server.log`). Not rotated. Refused when F is the file stderr already goes to (zero-server's `server.log`) |
| `--log-disable` | no log, not even the `progress:` lines (zero-server's and the launcher's bars stay empty); a crash report is still written |

runz and the other tools print as before, byte for byte: the bench scripts parse their output.

## What the log says

**At start**, before the model:
- **Build:** `inferzero 2026.11.12.12 (built …, commit …)` from `bin\PREBUILT.txt` in a package, else "built from source".
- **System:** the Windows build, the CPU, logical processors, RAM (total and free).
- **GPU:** name, driver, EUs, clock, memory (shared with the CPU or its own), XMX, sub-group sizes.
- **Switches:** every `ARC_*` variable set (any case), with its value; your user folder in a value shows as
  `%USERPROFILE%`. A name with TOKEN, SECRET, PASS, KEY, AUTH, CRED or COOKIE shows `(set)` only.
- **GPU line:** after the NPU and speech to text have started (its probe would hide the NPU from them).

**At each model load:**
- **`loaded in …`:** layers, context, chunk, GPU memory, resident or just-in-time weights.
- **`model:`:** arch, layers, embedding, heads / KV heads, head size, FFN, vocab, trained context, experts, file size and type.
- **`memory (--ctx auto):`:** how the device's memory was shared out: kept free, snapshots, encoders and speech to
  text, weights, MTP / drafter, buffers, the context (and per 1K tokens), what is left. With `--ctx auto` only; with
  `--slots` or `--gpus` the numbers are one slot's on one GPU and "left" is not shown.

**Per request:**
- **The completion line** (above): prompt tokens and how many came from a cache, prompt time and speed, generated
  tokens, time and speed, the finish reason, the drafters' accepted drafts. Then the time to the first token (from the
  request's arrival), the context used / served, and the slot.
- **An access line for every request the server answers** (the handler, the remote sign-in gate, OPTIONS): `POST /v1/chat/completions 200 726 B 1874 ms 127.0.0.1`. A GET that succeeds
  (the web UI's and the launcher's polls), HEAD, OPTIONS and `/tokenize` / `/detokenize` show at debug only. A poll that
  keeps failing (e.g. `/health` while no model is loaded) is logged once per path and status, and again after it
  worked. A request refused before it is read (a malformed line, a DNS-rebinding Host, another site's Origin, a
  body too slow) is a `W HTTP 403, connection closed: ...` line. The method is one of HTTP's or "other", and control
  characters in a path show as `?`, so a request cannot write lines of its own.
- **Errors:** a 4xx reply is a `W` line with its message, a 5xx an `E` line, e.g.
  `W POST /v1/chat/completions 400 353 B 19 ms 127.0.0.1 | the request exceeds the available context size (prompt 20010 tokens, context 8192 tokens), try increasing it`.
- **At debug (`-lv 3`), llama.cpp's three lines:** `prompt eval time = … ms / N tokens (… ms per token, … tokens per second)`, `eval time`, `total time`.

**When the GPU ends the process:** an `E` line says what happened in words where the code is sure: device lost (the
GPU was reset: a driver timeout or a fault; the process must restart), out of memory, the driver not started. A
failed device allocation names its size, the GPU and what the engine had allocated, then the GPU budget and use and
the free RAM. A kernel that fails names itself, its launch size and, while it waited, the last 64 launches.

## Copying the log for an issue

The launcher's Log tab has **Copy for an issue**: the log since the start as a GitHub code block, the user folder as
`%USERPROFILE%` and the user name taken out of paths (`\Users\<user>`), the `progress:` lines left out, at most 2,040
lines (the first 40, which name the build, system, GPU and switches, and the last 2,000). Paste it into the issue.

## Old logs

`zero-server` moves the last log to `logs\server_<date>_<time>.log` at each start, for example
`server_2026-11-12_14-13-48.log`. The names sort by time, and they keep `.log`, so a double-click opens them.
- The newest 10 stay as they are.
- Older ones each go into a zip of the same name (`server_2026-11-12_14-13-48.zip`), about 10x smaller, keeping
  their time. Every Windows opens a zip with a double-click.
- A zip older than 90 days, or past 200 MB together, is removed, oldest first.

## `GET /metrics`

Prometheus' text format. The names are vLLM's, with `inferzero:` in place of `vllm:`, so a vLLM Grafana dashboard
works with that prefix changed. Numbers only, never a prompt's text. Every series has the label `model_name` (the served
alias).

| Kind | Names |
|---|---|
| gauges | `num_requests_running`, `num_requests_waiting`, `kv_cache_usage_perc` (the last request's context use, 0 to 1), `context_max_tokens`, `gpu_memory_bytes`, `model_info{arch,ctx,gpus}` |
| counters | `prompt_tokens_total`, `generation_tokens_total`, `prefix_cache_queries_total`, `prefix_cache_hits_total`, `spec_decode_num_draft_tokens_total`, `spec_decode_num_accepted_tokens_total`, `request_success_total{finished_reason}`, `http_requests_total{method,handler,status}` |
| histograms (vLLM's buckets) | `time_to_first_token_seconds`, `e2e_request_latency_seconds`, `time_per_output_token_seconds`, `request_prefill_time_seconds`, `request_decode_time_seconds`, `request_prompt_tokens`, `request_generation_tokens` |

A request the client cancelled counts as `finished_reason="abort"` (its times stay out of the histograms).
`handler` is the API's path (`/v1/chat/completions`, `/api/chat`, …); any other path counts as `other`, so the series
stay few. Through remote access, `/metrics` needs the sign-in like every other page.

## For code

New messages use `LOG_ERR / LOG_WRN / LOG_INF / LOG_DBG(fmt, …)`. The old `fprintf(stderr, …)` calls became
`zlog::eprintf(…)`: info, or a level from the line itself: E for "error" / "fatal" / "crashed" at its start
(also after a tag such as `inferzero: `), "error 0x" near the start, "out of memory" or "device lost" anywhere; W for
"warning", "failed", "cannot", "could not", "not found" at its start. The Level Zero errors, a missing driver and
the GPU running out of memory are `LOG_ERR` by name. The `LOG_*` macros also hand their arguments to an `fprintf` that
never runs, so the compiler checks the format (C4477) as it did for the old calls. A line that a program reads goes
through `zlog::raw` (no prefix), and so does the server's usage text. A line a thread had not finished (printed without its `\n`) is lost if the process crashes or ends with `_Exit`; every
message ends with its `\n` today. The crash report (`src/crash_log.h`, `zrt::print_last_launch`) writes plainly, so a
crash never waits on the log's lock.

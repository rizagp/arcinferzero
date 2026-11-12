# Working on inferzero (for coding agents)

Read by Pi, Codex and other agents that load `AGENTS.md`; Claude Code reads it through `CLAUDE.md`. What the
project is: [README.md](README.md). Where things stand and what is next: [QUEUE.md](QUEUE.md) ("Open now" at the top).

## The project in one paragraph

A C++ LLM inference engine for Intel Arc GPUs on Windows, built on Level Zero, with kernels in OpenCL C:
- `zero/`: the engine;
- `zero/kernels/*.cl`: the kernels;
- `tools/server.cpp`: an HTTP server with a web UI (`webui/index.html`).

It was tuned on a Core Ultra 5 125H iGPU (Xe-LPG: DP4a, no XMX). Discrete Arc B-series cards run it too:
- `zero/xe2/`: their switches;
- `zero/mgpu/`: one model over several GPUs.

Their speed can only be measured by users who have those cards (GitHub issue #4).

## Rules

- **Accuracy:** every model's next-token KL against an f32 reference stays at or below **0.015**, with two
  exceptions:
  - speech models stay below 0.01;
  - Flash Next's default (fast) mode may reach ~0.02 for text work such as summaries, while its precise mode
    (`--precise`) stays under 0.015 for work that needs exactness. A speed change that moves the logits needs a KL check. A change meant to be bit-identical must be
  checked bit for bit (`runz --dump-logits` before and after).
- **Every speed or accuracy change** gets a row in `docs/before-after.md`, measured the same way before and after,
  and a CHANGELOG entry.
- **Where code goes:**
  - multi-GPU code in `zero/mgpu/` and `zero/kernels/tp.cl`;
  - code only for discrete cards in `zero/xe2/` and kernel files of its own;
  - the iGPU path must stay bit-identical when such code is added.
- **Work-group sizes:** a kernel launched without a local size (`run(..., nullptr)`) gets the driver's suggestion
  shrunk until it divides the grid (`zero/clrt.h`), so its group size depends on the data: such a kernel must not
  depend on `get_local_size()` beyond size-agnostic loops (`for (i = get_local_id(0); i < n; i += get_local_size(0))`).
  A kernel whose local memory is laid out per sub-group (`win_all[16]`) gets an explicit size (audit 2026-11-12 #9).
- **Build:** `.\build.ps1 zero\runz.cpp zero\serverz.cpp` (and the other `.cpp` you touched). After any change to
  the engine or kernels, rebuild **both** runz and serverz before testing: a stale serverz looks like a crash.
- **Before any GPU test:** list running `serverz` / `runz` processes (`Get-Process serverz,runz`); a user's server
  running at the same time caused device-lost errors.
- **Multi-GPU without several GPUs:** `ARC_TP_SIM=1` runs a split model's shards on one GPU (correctness only, not
  speed).
- **Messages on stderr** go through the log (`src/log.h`, [docs/logging.md](docs/logging.md)): `LOG_ERR / LOG_WRN /
  LOG_INF / LOG_DBG`, not `fprintf(stderr)`; a line a program reads (`progress:`) through `zlog::raw`. Off in runz and
  the tools, which print as before.
- **Temporary files** (scripts, logs, test outputs) go in a temp folder, never in the repository. Never commit
  `tests/data` recordings. Never delete model files (`.gguf`).
- **Shell:**
  - no heredocs: write a `.py` or message file and run that;
  - long runs in the background with a log file;
  - Python that writes text files uses `newline=""`.
- **Git:** commit and push only when the user asks. Commits use the GitHub noreply address, with no co-author
  trailer (no `Co-Authored-By:` line for an AI or anyone else; the user, 2026-11-07).
- **Docs** are Markdown in English, in plain words. (The user chats in Indonesian.)

## Map

| | |
|---|---|
| `zero/engine.h` | the Engine class: forward passes, dispatch; parts included in its body: setup `engine_setup.h`, `engine_load.h`, `engine_load_gemma.h`, `engine_alloc.h`, `engine_stream.h`; slots `engine_slots.h`, `engine_sessions.h`; layers `engine_qwen.h`, `engine_gemma.h`, `engine_moe.h`, `engine_drafters.h`, `engine_flashnext.h`, `engine_h8.h` |
| `zero/kernels/ops*.cl` | non-GEMM kernels in parts (`KernelParts`, `zero/clrt.h`): `ops.cl` norms / DeltaNet, `ops_attn.cl`, `ops_attn_g.cl`, `ops_vit.cl`, `ops_flashnext.cl`, `ops_dflash.cl`, `ops_gdn.cl` (chunked DeltaNet); `decode.cl` + `decode_moe.cl` likewise; a kernel builds from its part and the ones before. Discrete-card XMX kernels in files of their own: `gdn_x.cl`, `attn_x.cl` (put in front of the ops parts for `attn_q8`), `gemm_x16.cl`, `gemm_x8.cl`, `h8.cl` |
| `zero/clrt.h` | the OpenCL-style calls on Level Zero (kernels compiled through OpenCL, cached in `%USERPROFILE%\.inferzero\kernels`) |
| `zero/runz.cpp` | CLI: `--bench-pp`, `--dump-logits`, `--perplexity`, `--kl-ref`, `--gpus`; `runz --help` |
| `tools/ref_forward.cpp` | the f32 CPU reference used for KL |
| `docs/before-after.md` | every change with its numbers; `docs/speed-progress.md`, `docs/lessons-intel-arc-igpu.md` for history and what did not work |
| `docs/multi-gpu.md`, `docs/xe2-tuning.md` | split models and discrete cards: design, failsafes, switches, what users measured |
| `bench/` | `gemm_sweep.ps1`, `decode_try.ps1` (discrete cards), micro-benchmarks |

Measurements from users' cards are in the attachments of GitHub issue #4: read them before concluding how a path
behaves on a B60. For example, the logs show "XMX on for the prompt GEMMs".

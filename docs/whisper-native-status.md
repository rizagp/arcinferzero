# Whisper native: work status

The plan is [whisper-native.md](whisper-native.md). This page records where the work stands (updated 2026-10-26).

## Done and checked

| File | What |
|---|---|
| `src/whisper.h` | the CTranslate2 `model.bin` reader (mapped, f16), the tokenizer from `tokenizer.json`, the log-mel (openai recipe), the `Backend` interface, and the `Transcriber`: 30-s windows, previous-text prompt (or none: `--no-previous-text`), timestamp rules, greedy / beam 5 / best-of-5 sampling with temperature fallback, no-speech skip, language `auto` / fixed / a list like `id,en` detected per window, word timestamps from the alignment heads + DTW, the next window from the last word's end |
| `src/whisper_formats.h` | json, fw (the Remotion pipeline's JSON), captions (Remotion `Caption[]`), verbose_json, srt, vtt (word timing), sbv, srv3, ass (karaoke), ttml, txt, md, csv, tsv; subtitle cues from the word timestamps (2 x 42 characters, at most 7 s). fw used on the 14-minute file; the other formats not yet opened in their target programs |
| `tools/ref_whisper.cpp` | the f32 CPU reference (0.2x realtime): `--dump-encoder` / `--dump-logits`, `WH_DUMP_DIR` per-layer dumps, `WH_Q8W=1` (weights rounded as Q8_0) |
| `zero/whisper_engine.h`, `zero/kernels/whisper.cl` | the iGPU engine: Q8_0 int8 GEMMs (gemm.cl), `vit_attn_rows` for the encoder attention, the cross-attention split over the positions (flash decoding) on f16 head-major keys / values, `ARC_WH_Q2=n` (two int8 passes from encoder layer n on; off by default) |
| `zero/transcribez.cpp` | the CLI: files / folders / wildcards, every format, `--out`, a progress bar or `--progress` JSON lines |
| `zero/npu/whisper_npu.h`, `zero/npu/npu_ir.h` | the NPU path (`--device npu`): the encoder as three graphs (front, one layer for all 32 layers with its weights as int8 + per-32 f16 scale inputs, the final LayerNorm), compiled once (blob cache); the decoder on the iGPU |
| `tools/server_whisper.h`, `tools/server.cpp` | the server: `--whisper auto\|DIR`, `--whisper-device gpu\|npu`, `--whisper-max-minutes`; `/v1/audio/transcriptions` and `/translations` as OpenAI's API; Whisper loads before the model and `--ctx auto` keeps its GPU memory free; each Whisper step takes the model engine's lock (one GPU command list) |
| `src/tokenizer.h`, `src/audio.h`, `build.ps1` | GPT-2 pre-tokenizer constructor; `decode_16k(bytes, max_seconds)` (0 = no limit); `ref_whisper` with `/fp:precise` |

## Measured

**60-s clip** (`smoke60.wav`): the f32 reference against faster-whisper f32: greedy 1 real word apart, beam 5 1 word
apart. The iGPU against the reference: decoder logits KL 0.00039, the top token the same on 64 / 64 steps. The
encoder output looks far off on some rows (worst 140 %), but the reference with nothing but its weights rounded to
Q8_0 deviates on the same rows (worst 120 %, KL 0.00008): those rows are chaotic, not wrong; the logits are what
count. Beam 5 on the iGPU dropped one phrase the reference keeps: a near-tie, not a bug (two int8 passes from layer
24 restored it, from layer 16 or 0 did not).

**14-minute file** (the channel's `work/bench/audio16k.wav`, `--lang id --no-previous-text`, as the earlier bench):

| | Time | x realtime | Hard terms | Word differences vs faster-whisper beam 5 (hyphens joined) | Word timestamps (mean / p90) |
|---|---|---|---|---|---|
| **arcinferzero iGPU greedy** (2026-10-28, W1) | **25.1 s** | **33.8x** | 14 / 24 | 9.1 % | 123 / 379 ms |
| **arcinferzero iGPU beam 5** (2026-10-28, W1) | **32.2 s** | **26.4x** | 16 / 24 | 13.3 % | 133 / 460 ms |
| arcinferzero iGPU greedy / beam 5 before W1 | 31.8 / 39.3 s | 26.7 / 21.6x | 14 / 16 | 11.7 / 14.1 % | 124 / 400, 124 / 340 ms |
| OpenVINO GenAI iGPU greedy | 57 s | 14.8x | 15 / 24 | 10.1 % | 131 / 440 ms |
| OpenVINO NPU greedy | 144 s | 5.9x | 16 / 24 | 12.2 % | 140 / 520 ms |

Greedy is 2.27x OpenVINO's speed (1.8x before W1); beam 5 (as accurate on the hard terms as faster-whisper) is 1.77x
faster than OpenVINO's greedy. faster-whisper writes "orang -orang" as two words where arcinferzero writes "orang-orang": joined
before counting (the channel's compare.py counts them as errors: 13.7 / 15.1 % there).

**W1 (2026-10-28)**, the 14-minute file greedy 30.8 -> 25.1 s, beam 5 38.2 -> 32.2 s (the earlier runs of the table
were slower: 31.8 / 39.3); decoder logits KL against the f32 reference 0.00039 -> 0.00040, top tokens 64 / 64 (NPU
mode 0.00053, two int8 passes 0.00035):
- cross-attention rows per work-group as many as the pass has (1 for greedy, 5 for a beam; it computed 4 rows for
  one): 2.7 -> 1.6 s; 8 rows for the word timestamps' 100-row pass moved the timestamps a little and, on this file
  with beam 5, cascaded into 100 fewer words: kept at 5 (`ARC_WH_RB=n`);
- LayerNorm + quantization, the residual before it, and bias + GELU + quantization in one launch each
  (ops.cl wh_ln_quant / wh_gelu_quant; 24 -> 18 launches per decoder layer): 27.3 s, transcript identical;
- the word timestamps' host work (the 100-row pass's log-softmax, the heads' softmax, normalization and median
  filter) on every core: 2.0 -> 1.3 s, identical;
- the decoder's passes of 1-8 rows through the engine's Q8_0 GEMVs (decode.cl gemv / gemv_mt) on a raw copy of its
  weights: 2,885 passes, 2.89 -> 2.33 ms each (58 -> 72 GB/s); +0.16 GiB of GPU memory (1.55 GiB, NPU mode 0.89).
Left as it is: the encoder GEMMs (7.0 s, ~8 TOPS, the DP4a peak) and its attention (4.5 s, vit_attn_rows f32 at
~2.7 TFLOPS, ~68 % of the f32 peak). `WH_TIMES=1` splits the wall time (log-mel 0.26 s, windows, word timestamps);
`ARC_PROF=1` lists each kernel's time and launches.

**NPU mode** (14-minute file): NPU + iGPU greedy 81 s (10.5x realtime; OpenVINO NPU 144 s + 253 s compiling), beam 5
91 s; logits KL 0.00052 against the f32 reference (64 / 64 top tokens). It keeps the iGPU free for the LLM: 0.73 GiB
of GPU memory instead of 1.39 GiB.

**Server** (Qwen3.5 4B loaded, 60-s clip): json / srt / verbose_json with words / text / stream / translations
answered; beam 5 in 5.2 s with the upload (iGPU), 9.6 s in NPU mode; a chat request at the same time answered
normally; 6 uploads at once: 4 answered, 2 got 429; an unknown language, a non-audio file and an unknown format get
400.

## Next steps

1. Speed: done (W1, above).
2. Dual decode on windows whose language is ambiguous; the mixed-language sample prompt.
3. The formats opened in their targets (YouTube, Premiere / Resolve, Remotion), the bench row, docs (README,
   CHANGELOG, before-after).
4. Then the web UI tabs (Transcript, Meeting), `arc-recorder.exe` (separate process, WASAPI process loopback +
   microphone, `recorder.json` + named pipe, reconnect after a server crash), Talk mode with the choice of Gemma
   hearing directly or Whisper -> any model.

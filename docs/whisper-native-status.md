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
| **arcinferzero iGPU greedy** | **31.8 s** | **26.7x** | 14 / 24 | 11.7 % | 124 / 400 ms |
| **arcinferzero iGPU beam 5** | **39.3 s** | **21.6x** | 16 / 24 | 14.1 % | 124 / 340 ms |
| OpenVINO GenAI iGPU greedy | 57 s | 14.8x | 15 / 24 | 10.1 % | 131 / 440 ms |
| OpenVINO NPU greedy | 144 s | 5.9x | 16 / 24 | 12.2 % | 140 / 520 ms |

Greedy is 1.8x OpenVINO's speed; beam 5 (as accurate on the hard terms as faster-whisper) is still 1.45x faster than
OpenVINO's greedy. faster-whisper writes "orang -orang" as two words where arcinferzero writes "orang-orang": joined
before counting (the channel's compare.py counts them as errors: 13.7 / 15.1 % there).

GPU time on the 60-s clip (greedy): encoder GEMMs 27 % (6.6 TOPS), decoder GEMMs 26 %, encoder attention 17 %
(~0.8 TFLOPS: vit_attn_rows is shuffle-bound; FT 8 and more sub-groups per work-group change nothing), cross-attention
10 %.

## Next steps

1. The NPU path (`--device npu`): one encoder-layer graph (all 32 layers share its shapes: weights as inputs, compiled
   once) and the decoder per token, through `zero/npu/npu_ir.h` and the graph extension.
2. The server: `/v1/audio/transcriptions` and `/translations` with Whisper, following OpenAI's API (json, text, srt,
   verbose_json, vtt; `timestamp_granularities[]`), the other formats as extensions; security (size / duration
   limits, uploads streamed to disk, content checks, no client paths, temp files cleaned, a bounded queue).
3. Speed: a faster encoder attention (not shuffle-bound), the decoder through the GEMV kernels, fewer small launches.
4. Dual decode on windows whose language is ambiguous; the mixed-language sample prompt.
5. The formats opened in their targets (YouTube, Premiere / Resolve, Remotion), the bench row, docs (README,
   CHANGELOG, before-after), audit, commit, push.
6. Then the web UI tabs (Transcript, Meeting), `arc-recorder.exe` (separate process, WASAPI process loopback +
   microphone, `recorder.json` + named pipe, reconnect after a server crash), Talk mode with the choice of Gemma
   hearing directly or Whisper -> any model.

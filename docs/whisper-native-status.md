# Whisper native: work status

The plan is [whisper-native.md](whisper-native.md). This page records where the work stands (updated 2026-10-28).

## Done and checked

| File | What |
|---|---|
| `src/whisper.h` | the CTranslate2 `model.bin` reader (mapped, f16), the tokenizer from `tokenizer.json`, the log-mel (openai recipe), the `Backend` interface, and the `Transcriber`: 30-s windows, a glossary before every window (`--terms`), the previous text as prompt (off by default since W2: `--previous-text`), timestamp rules, greedy / beam 5 / best-of-5 sampling with temperature fallback, no-speech skip, language `auto` / fixed / a list like `id,en` (each utterance's language, W2), word timestamps from the alignment heads + DTW, the next window from the last word's end |
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

**W2 (2026-10-28): mixed languages and jargon.**

*Glossary*: `--terms "<list>"` / `--terms-file` (server: `terms`), kept before every window's prompt as
faster-whisper's `hotwords`, at most 150 tokens (the previous text, when on, gets the rest of the 223). On the
14-minute file with a 32-term glossary of the channel's words: hard terms 14 -> 21 / 24 greedy, 16 -> 21 / 24 beam
5; faster-whisper beam 5 with the same glossary as hotwords: 21 / 24. Both engines turn the text formal with a bare
list ("gak" -> "tidak", "pake" -> "pakai") and make longer segments (276 -> 41-53): 17.2 % word differences
against faster-whisper without a glossary (faster-whisper with it: 17.8 %). The same terms inside a casual sentence
("Eh, jadi gini ya, tadi kita pake Qwen, Gemma, ..., gitu, oke.") keep the casual spelling: 19 / 24 terms, 13.2 %.
No default sample prompt: any prompt changes a file's text, and single-language files are to stay as they were.

*Mixed languages*: Whisper decodes a window in one language; where a window holds both, it translates the other
language's sentences or leaves them out. Test file: 8 pieces of the 14-minute recording (Indonesian, ~12 s each)
alternating with 8 English sentences (Windows' TTS voices), 189 s; the truth: the English text, and faster-whisper
beam 5's words for the Indonesian pieces. Word differences:

| | Indonesian pieces | English pieces |
|---|---|---|
| faster-whisper beam 5, language detected once | 57.3 % | 28.2 % |
| faster-whisper beam 5, `multilingual=True` (per window) | 61.6 % | 19.1 % |
| arcinferzero beam 5, `auto`, before W2 | 45.7 % | 43.5 % |
| arcinferzero beam 5, `--lang en` (the English pieces at their best) | 78.0 % | 11.5 % |
| **arcinferzero greedy or beam 5, `--lang id,en` (W2)** | **18.5 %** | **6.9 %** |
| arcinferzero beam 5, `auto` (W2: only ambiguous windows split) | 32.8 % | 31.3 % |

How (`src/whisper.h` `language_runs`): with a list of languages, each window's pauses (>= 0.3 s quiet in the
10-ms energy) cut it into utterances (>= 1.5 s); each utterance's language between the window's two likeliest
comes from the encoder run on it alone with silence after it (`encode_part`, into a second set of cross-attention
keys / values: the window's stay): a first look on a 10-s context (~1/4 of the work), and where that says the other
language at >= 0.8, again on the whole 30-s context, where the other language counts at >= 0.8; runs of the other
language shorter than 4 s go back to the window's. Each run is then decoded alone, in its language. With `auto`,
only windows whose likeliest language is below 0.8 (`--dual-language`, server `dual_language`) are split; one
without pauses is decoded in both languages and the likelier text (mean log-probability) kept. The window's own
probability cannot tell mixing apart: windows with ~7 s of English out of 30 came out 0.995 Indonesian, as sure as
windows of Indonesian full of English terms (>= 0.988).

Tried and dropped: the decoder attending to the utterance's part of the window's encoder output only (no extra
encoder run, but the encoder mixes the positions: ~40 % of the utterances near a change came out wrong); the
encoder on the utterance without the silence after it (most Indonesian utterances came out English); utterances of
>= 2.5 / 3.5 s (1.6x / 2x faster, English pieces 15 / 37 % wrong).

Cost of `id,en`: the 14-minute file greedy 71.9 s (11.8x realtime; `id` 25.2 s), beam 5 83.9 s (32.2 s), and on
that Indonesian file the transcript is the same as `--lang id`'s (greedy and beam 5). NPU mode: its graphs have the
whole window's shape, so every utterance runs a whole window there: the mixed file 145.7 s (1.3x realtime; `id`
22.0 s), same accuracy (19.4 % / 6.9 %); the iGPU is the place for `id,en`.

*The previous text as prompt is off by default now* (`--previous-text`, server `condition_on_previous_text=true`
turn it on): with it, the 14-minute file came out in one-phrase segments, 23.4 % different from faster-whisper
(9.1 % without) with 12 / 24 terms, and a 7-minute narration lost two windows (one became "Terima kasih.").

Unchanged: `--lang id` greedy and beam 5 give W1's transcripts; `auto` on the 14-minute file too (no window below
0.8).

**NPU mode** (14-minute file): NPU + iGPU greedy 81 s (10.5x realtime; OpenVINO NPU 144 s + 253 s compiling), beam 5
91 s; logits KL 0.00052 against the f32 reference (64 / 64 top tokens). It keeps the iGPU free for the LLM: 0.73 GiB
of GPU memory instead of 1.39 GiB.

**Server** (Qwen3.5 4B loaded, 60-s clip): json / srt / verbose_json with words / text / stream / translations
answered; beam 5 in 5.2 s with the upload (iGPU), 9.6 s in NPU mode; a chat request at the same time answered
normally; 6 uploads at once: 4 answered, 2 got 429; an unknown language, a non-audio file and an unknown format get
400.

## Next steps

1. Speed: done (W1, above).
2. Mixed languages and jargon: done (W2, above).
3. The formats opened in their targets (YouTube, Premiere / Resolve, Remotion), the bench row, docs (README,
   CHANGELOG, before-after).
4. Then the web UI tabs (Transcript, Meeting), `arc-recorder.exe` (separate process, WASAPI process loopback +
   microphone, `recorder.json` + named pipe, reconnect after a server crash), Talk mode with the choice of Gemma
   hearing directly or Whisper -> any model.

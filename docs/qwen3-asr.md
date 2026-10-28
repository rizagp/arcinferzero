# Qwen3-ASR on the Arc iGPU

Qwen3-ASR 1.7B (speech recognition, 30 languages) runs on arcinferzero's Level Zero engine next to Whisper: the
server's `/v1/audio/transcriptions` and the Transcripts tab can use it instead of Whisper (added 2026-10-28).

Why: on the user's noisy recording of mixed Indonesian and English, Whisper turbo and large-v3 repeated words, chose
wrong languages and invented text over the quiet end; Gemma 4 E4B read better but added sentences nobody said.
Qwen3-ASR came out the closest to what was said (see [Comparison](#comparison-on-a-real-recording)).

## Files

The GGUF conversion of [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), for example
`unsloth/Qwen3-ASR-1.7B-GGUF` (the user's copy: unslothai's), two files in one folder:

| File | What it is |
|---|---|
| `Qwen3-ASR-1.7B-Q8_0.gguf` (2.0 GB) | the text model: Qwen3 1.7B dense (GGUF arch `qwen3vl`), 28 layers, width 2048, 16 q heads / 8 kv heads of 128, FFN 6144 |
| `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` (0.36 GB) | the audio encoder (projector type `qwen3a`): 3 convolutions, 24 transformer layers of width 1024, projection to 2048 |

## How it runs

- **Audio encoder** ([zero/audio_qwen3a.h](../zero/audio_qwen3a.h), its kernels `q3a_*` in
  [zero/kernels/asr.cl](../zero/kernels/asr.cl), the vision encoders' `vit_*` in [zero/kernels/ops.cl](../zero/kernels/ops.cl)): Whisper's 128-bin log-mel without padding (src/whisper.h
  `log_mel`, `pad` 0), cut into chunks of 100 frames (1 s); the first 3x3 stride-2 convolution on the CPU, the second
  and third as im2col + f16 GEMM on the GPU; 13 tokens per chunk; 24 pre-LayerNorm transformer blocks whose attention
  stays inside windows of 8 chunks (104 tokens), so the encoder runs on groups of 16 windows one after the other;
  LayerNorm and the projector (GELU). f16 weights, f32 activations. About 60-100x real time.
- **Text model** ([zero/engine.h](../zero/engine.h) `load_hparams_qwen3`, [zero/engine_qwen.h](../zero/engine_qwen.h)
  `layer_qwen3_attn`): Qwen3 dense on the engine: q / k RMS norms and NEOX RoPE over the whole 128-dim head (M-RoPE
  with text and audio positions is plain RoPE), **float attention** (f32 queries and K, f16 V; `attn_fa_gh` for prompt
  chunks, the split `attn_fa_g` for single tokens) and **two-pass int8 matmul inputs**, as Gemma 4 12B / 26B.
- **Transcription** ([zero/asr_qwen3.h](../zero/asr_qwen3.h)), as Qwen's `qwen_asr` package does it: the prompt
  `<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>` + the audio rows +
  `<|audio_end|><|im_end|>\n<|im_start|>assistant\n`; greedy decoding; the answer `language Indonesian<asr_text>...`
  (`language None` for no speech); repetitions collapsed as `detect_and_fix_repetitions` does (a pattern of up to 20
  characters repeated 20 times or more is kept once). One part up to 20 minutes in Qwen's code; the server's
  engine holds 6,144 tokens of context, so it cuts audio into parts of up to 6 minutes at the quietest 100 ms near
  each cut. Added here: decoding stops once the text ends in a pattern repeated 40 times (the model looping over
  background sound; Qwen's fix would keep one copy anyway, 40 leaves real repetition such as "tes tes tes ..."
  untouched).

## Accuracy against f32 references

| Check | Result |
|---|---|
| Text model, 72-token chat prompt, KL against [tools/ref_forward.cpp](../tools/ref_forward.cpp) (f32 activations, dequantized weights) over every position | < 0.00001 prompt path and token-by-token path, top token 72 / 72 |
| ... with int8 attention (the Qwen3.5 kernels, first try) | KL 0.236 (int8 q / K / V per row in the reference: 0.159; per 32 values 0.123) |
| ... float attention, one int8 pass for the matmuls (`ARC_Q3_Q2=0`) | KL 0.0064 prompt / 0.0093 token by token |
| Encoder against [tools/ref_qwen3a.cpp](../tools/ref_qwen3a.cpp) (f32, written from Hugging Face's modeling code), 20 s and 23.37 s of the user's recording (a short last chunk, a partial window) | relative error 0.078 % / 0.074 %, worst token 0.41 %, lowest cosine 0.999992 |
| Synthetic: 20 s of digital silence; 20 s of white noise at -50 dBFS | empty text (Whisper turbo: empty too) |

Qwen3's attention does not survive int8 scores: its k_norm weights make a few dims of K large, and one int8 scale
per row (or per 32 values) drowns the rest. The user's limit for this model is KL < 0.01; the one-pass mode was
under it by little, so two passes are the default.

## Speed (Core Ultra 5 125H)

| | |
|---|---|
| Text model, `runz --bench-pp 2048` (best of two) | prompt 901 tok/s, generation 22.2 tok/s (one pass: 1,376 / 24.5) |
| Encoder | 1.45 s for 153.6 s of audio (1,997 tokens) |
| The 2:34 recording end to end, `asr_testz` (best of two) | 23.2 s (encoder 1.4 s, prompt 2.6 s, 507 tokens 19.2 s), ~6.6x real time |
| ... through the server (`/v1/audio/transcriptions`, curl's time) | 25.2 s (the server's log: encoder 1.6 s, prompt 2.6 s, tokens 19.8 s) |
| GPU memory of Qwen3-ASR (6,144-token context, with the encoder) | 3.72 GiB (raw Q8_0 rows 2.1, float KV cache 1.0, encoder 0.6 + buffers); with Whisper turbo in the server 5.31 GiB together |

The GEMM form of the weights is decoded just in time for each 512-token prompt chunk (`w8_budget_gb` 0, `chunk`
512) instead of being kept next to the raw rows: 5.09 -> 3.72 GiB (server with Whisper 6.68 -> 5.31 GiB). The prompt
costs about 0.2 s more (2048 tokens 927 vs 938 tok/s), generation reads the raw rows either way (25.5 vs 25.9
tok/s), the recording took 23.2 s both ways, the text is the same and the KL unchanged (< 0.00001). Chunks of 4,096
(the engine's default without resident weights) need larger buffers (4.06 GiB) and are no faster.

## Comparison on a real recording

The user's 2:34 recording (2026-10-28, a laptop microphone in a noisy room): Indonesian with English sentences,
"tes tes ..." at the start, speech until ~113 s, then 40 s of background sound (-30 to -45 dBFS with bursts) and
maybe a few words near 141 s.

| | Time | Mixed language | Over the last 40 s |
|---|---|---|---|
| Whisper large-v3-turbo (beam 5, `transcribez`) | 10.3 s | English kept, but "Hello, I'm talking in English. Saya nggak ngomong buah, bukan" and the second "How come..." lost; "best-best-best" | "hai hai hai ..." (perhaps said), "Terima kasih.", "ありがとうございました。" |
| Whisper large-v3 | 39.5 s | lowercase fragments, "Indonesia Indonesia How come you Translate...", "Hai tetap kaitan antar ini ini tetap kaitan" | Thai gibberish twice |
| Gemma 4 E4B (`/v1/audio/transcriptions` with the chat model) | 50 s | readable, English kept, but sentences added ("Sudah tahu sudah paham gua", "Ya itu dia masalahnya", "Saya enggak setuju nih") | a made-up paragraph ("Kita ada nama ya ... Itu lima orang jadi satu voice") |
| **Qwen3-ASR 1.7B** (one part) | 23-30 s | every utterance there, English in English, punctuation; a few misheard words ("Ijin prinsipnya supaya tembuh nanti", "secercahitan") | "Keterkaitan dulu. Keterkaitan" (3 words; it looped on that word until the loop stop) |

A second, clearer recording (`tests/data/audio test 2.aac`, 2:43, one speaker in Indonesian, a work briefing on
risk reports), key words in each transcript (✓ as said, as far as the context shows; ✗ wrong or missing):

| | Time | "tes baru" | Profil Risiko Badan | BPJS Ketenagakerjaan | akuisisi kepesertaan | retensi | JMO | kondisi saat ini | hasil audiensi ... di sini | Punctuation |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-ASR (two passes) | 20.3 s | ✓ | ✗ profesi | ✓ (ketenaga kerjaan) | ✓ | ✓ | ✓ | ✗ "dimana ko di saat ini" | ✓ | none, lowercase |
| Qwen3-ASR (one pass) | 18.8 s | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ Jemo | ✗ | ✓ | none |
| Whisper turbo | 7.5 s | ✓ | ✗ proses | ✗ Ketenagaan Kerjaan | ✗ keberstaan | ✓ | ✗ JEMO | ✗ | ✗ cut | yes, the end in fragments |
| Whisper large-v3 | 23.4 s | ✓ | ✗ proses | ✓ | ✗ keberstaan | ✓ | ✓ | ✓ | ✓ | yes, the middle in fragments |
| Gemma 4 E4B | 36.7 s | ✗ | ✓ | ✓ | ✗ kebersih-an | ✗ | ✓ | ✓ | ✓ (audensi) | yes |
| Gemma 4 12B | 75.0 s | ✗ | ✗ | ✗ BPGS katanaga kerja | ✗ akurasi keputusan | ✓ | ✗ jemo | ✗ | ✗ | yes; "Amerika", "tiga tuluhan", "I K C" |

The user confirmed "JMO" and "risiko **operasional**" (only Gemma 4 E4B wrote operasional; the others "personal").
With a glossary as its context (`terms` / the Transcripts tab's Glossary; `asr_testz --context`: "BPJS
Ketenagakerjaan, Profil Risiko Badan (PRB), risiko operasional, kepesertaan, aplikasi JMO, deputi bidang MRK")
Qwen3-ASR wrote every one of those as given, the rest of the text as before (still "pelakuan pelakuan", "dimana ko di
saat ini", no punctuation); 22.0 s. (The glossary held the right words, so this shows what a user's own list of
names and terms does, not an unaided result.)

No model invented sentences on this recording except Gemma 4 12B, whose text drifts (it has no audio encoder: raw
40-ms frames, see the Gemma 4 12B audio notes); on the first recording 12B wrote a short translated summary instead
of a transcript. Here Qwen3-ASR named the language Indonesian, and wrote it without punctuation, as it did when the
language was forced on the first recording.

Notes from the runs:

- The model names this speech **Malay** (it is Indonesian). The server reports it as `id` when the request's
  languages include `id` and not `ms` (`language: id,en`). Forcing the language (`language Indonesian<asr_text>`
  in the prompt, `asr_testz --lang id`) gave lowercase text without punctuation and the same loop, so the server never
  forces it.
- Over background sound alone (113-140 s as its own clip) it wrote one word ("Kerajaan."); after speech, in one
  context, it looped on the last word. Parts cut at pauses (`Options::part_seconds`, 20 s) were worse on this
  recording: the pauses after 68 s came too early to cut, and the 85-s last part repeated a sentence four times and
  lost three others. Off by default.
- No timestamps: one segment per part (Qwen's Qwen3-ForcedAligner-0.6B would give word times; not here). Subtitle
  formats spread a segment's time evenly over its cues.

## Where the code is

| File | What |
|---|---|
| zero/engine.h, zero/engine_qwen.h | the Qwen3 dense text model (`load_hparams_qwen3`, `layer_qwen3_attn`; q / k / v and gate / up stacked; f16 K rows, ops.cl -DK_F16) |
| zero/audio_qwen3a.h, zero/kernels/asr.cl | the audio encoder |
| zero/asr_qwen3.h | a transcription: prompt, decoding, output parsing, parts, model lookup |
| tools/server_whisper_asr.h | Qwen3-ASR and Gemma 4 E4B inside the speech service (included in its class) |
| tools/server_speech.h | starting the speech service, Gemma 4 E4B through the chat models, `/v1/audio/*` |
| zero/asr_testz.cpp, tools/ref_qwen3a.cpp, tools/ref_forward.cpp | the command line and the f32 references |

## Using it

Since 2026-10-28 (the user's choice after the comparisons above) the server's transcript models are **Qwen3-ASR
(the default), Gemma 4 E4B and Whisper large-v3**; Whisper turbo is no longer used when large-v3 is in the cache, and
Gemma 4 12B does not transcribe.

```
serverz -m <chat model> --whisper auto        (arc-server start -Whisper auto)
```

- `--whisper auto` loads faster-whisper large-v3 (turbo only when large-v3 is not in the cache) and, next to it,
  Qwen3-ASR 1.7B (else 0.6B) GGUF from the Hugging Face cache (`--asr auto` is the default with `--whisper`;
  `--asr <file.gguf>` a given one, `--asr off` none; the mmproj with projector `qwen3a` is found next to it). Gemma 4
  E4B is offered when the model list has it with its audio encoder. `/props` shows
  `"whisper": {"model": "whisper-large-v3", "asr": "qwen3-asr-1.7b", "chat_asr": "gemma4-e4b", ...}`.
  GPU memory: large-v3 and Qwen3-ASR 7.79 GiB together (turbo and Qwen3-ASR before f16 K: 5.31). Note: in the
  server with large-v3 loaded first, Qwen3-ASR wrote the 2:34 recording's 507 tokens in 24.1 s against 18.7 s on
  the command line (19.8 s in the server with turbo): not looked into yet.
- `POST /v1/audio/transcriptions`: no `model`, OpenAI's `whisper-1` or `qwen3-asr` -> Qwen3-ASR; `gemma4-e4b` /
  `e4b` -> Gemma 4 E4B (in parts of up to 2 minutes cut at quiet points, a segment each; the server loads E4B as its
  chat model when another one is loaded); `whisper` or its id -> Whisper large-v3 (word times); any other chat model:
  400. Every response format; `prompt` and `terms` become the context; translations stay Whisper's.
- Voice messages and audio files in a chat: written by Qwen3-ASR (unless the setting "The model hears it" is on and
  the chat's model hears audio).
- Transcripts tab: a model choice per file or recording (Qwen3-ASR / Gemma 4 E4B / Whisper large-v3), changeable any
  time; each file keeps the model it was added with. A live recording's text as it is spoken is Whisper's; the chosen
  model does the pass after Stop.

Measured through the server (2026-10-28): every route on a 14-s clip (default, `whisper-1`, `whisper`, `gemma4-e4b`,
`gemma4-12b` refused), the queue with each model; Gemma 4 E4B on the second recording in two parts 25.9 s, the same
text as in one piece (36.7 s).

Command line ([zero/asr_testz.cpp](../zero/asr_testz.cpp)):

```
asr_testz Qwen3-ASR-1.7B-Q8_0.gguf mmproj-Qwen3-ASR-1.7B-Q8_0.gguf recording.mp3 [--lang id] [--start S] [--seconds S]
          [--context "names, terms"] [--parts S] [--ref encoder.ref] [--no-text]
```

Checks: `ref_qwen3a mmproj.gguf audio out.ref [seconds]` writes the f32 encoder output, `asr_testz ... --seconds S
--ref out.ref --no-text` compares; `ref_forward model.gguf <ids> ref.bin` with `REF_ALL=1` and `runz model.gguf
--tokens-file ids --kl-ref ref.bin [--kl-decode]` for the text model.

## Left to do

- Timestamps (Qwen3-ForcedAligner-0.6B, or cutting at pauses done better than here).
- Memory: the encoder's f16 weights (0.6 GiB) could be int8 as the vision encoders' two-pass path has them.
- The server's slower decoding with Whisper large-v3 loaded before it (see Using it).
- Streaming (the user's friend runs it in vLLM for voice calls; Qwen's package feeds 2-s chunks).

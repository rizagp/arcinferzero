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
| `src/audio.h` (W4) | decoding to 16 kHz mono, streamed: WAV read in blocks, Media Foundation from the file itself (no copy of the upload's bytes for a file on disk), the resampling in pieces (bit-identical to the one-piece resampler: `tests/test_audio.cpp`); times on the file's timeline (a late audio track or a gap of more than 0.1 s becomes silence; an MP4 track's start delay from its edit list); the format read again when the decoder changes it; Ogg Opus repacked as WebM for Media Foundation |
| `tools/server_whisper.h` library (W4) | the saved transcripts and their queue: `/v1/transcripts` (below) |
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

**GPU memory of large-v3 (2026-10-30)**: 4.39 -> 3.94 GiB (NPU mode 3.28): the decoder layers' cross keys / values
are computed in f32 and packed to f16 one layer after the other, so they share one f32 buffer instead of one per layer
(0.46 GiB for 32). Logits of the first window (greedy, beam 5) and a 2:43 recording's beam-5 JSON / SRT / TXT
byte-identical. What is left: the decoder's self-attention KV cache in f32 for 5 sequences (1.37 GiB; f16 would
halve it, a change of the numbers to check against the f32 reference first), the weights in Q8_0 (~1.5 GiB), the
f16 cross keys / values of a window and of a part (0.46 GiB). No register spills in the Whisper kernels (the driver's
`ARC_ZE_KPROPS`).

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

**W3 (2026-10-28): the formats in their readers.** Checked here: `@remotion/captions` 4.0.538 (the channel's own copy)
reads the `captions` file (`createTikTokStyleCaptions`: pages) and parses the SRT (`parseSrt`: every cue,
`serializeSrt` round trip); Chrome's WebVTT parser reads the VTT (138 cues, 1,242 word timestamps, none refused);
SRT / VTT / SBV / SRV3 / ASS / TTML pass a structure check (`fmt_check.py`: timings, XML, ASS fields, karaoke `\k`
summing to each line, no overlap). Two faults found and fixed: a cue's second line could pass 42 characters (the
words were packed greedily into the first line: 14 cues of the 14-minute file); now a cue takes a word only while
its lines still fit, and two lines are evened out. And a VTT word timestamp could equal the cue's start (WebVTT asks
for later): each is now later than the cue's start and the previous one, earlier than its end. Left for the user:
uploading SRT / VTT to YouTube, importing SRT / ASS / TTML in Premiere and Resolve.

**W4 (2026-10-28): video files and the transcript library.** Every container of the test set (the 60-s clip with a
still image as video) transcribed, the words at the WAV's times:

| File | Before | Now |
|---|---|---|
| MP4 / MOV (H.264 + AAC), WebM (VP9 + Opus), MKV (H.264 + Opus), AVI (MP3, PCM), M4A, MP3 | read | the same samples (bit-identical transcripts) |
| MKV with 16-kHz AAC | 15 s of garbage (the decoder changes the format at the first sample: 32 kHz stereo was read) | read (the format read again) |
| MP4 / MKV whose audio starts 2 s into the video | the words 2 s early | on the video's timeline (+2.00 s) |
| Ogg Opus (.opus, .ogg: WhatsApp / Telegram voice notes) | "unrecognized audio format" | read |
| Ogg Vorbis | unrecognized | refused by name (no encoder here to test it) |

WMV / WMA: Media Foundation's own format, not in the test set (the ffmpeg at hand has no ASF muxer). A file on disk
is decoded as it is read: the CLI no longer reads a video of gigabytes into memory first, and 48-kHz samples of
hours never sit in memory at once.

The transcript library (`/v1/transcripts`, for the Transcript tab): each transcript in its own folder under
`%USERPROFILE%\.arcinferzero\transcripts\<id>\` (`meta.json`, the uploaded file, `transcript.json`); files uploaded
in pieces (any size the disk holds), a queue transcribed one file at a time with progress, cancel (what was done is
kept), delete (also while running), exports in every format with the cue options, the media with byte ranges, edits
saved over the transcript, and a restart that transcribes again what was queued or running. Checked end to end
against a server (`w4/api.py`, 31 checks): 3 files queued in order, progress while running (33x realtime), a cancel
at 25 %, 14 export formats, ranges (206 / 416), an edit and a rename, wrong offsets / sizes / languages / ids, a file
that is not audio, a restart in the middle of a transcription and of an upload, a delete while running.

**W5 (2026-10-28): the Transcripts tab** (`webui/index.html`, the rail's second button; on with `--whisper`):
files dropped or chosen are uploaded in 16-MB pieces into the server's queue, each with its progress bar
(uploading, waiting with its place, % with the speed, done with its length, languages and words); a transcript
opens with its media (`<video>` with byte ranges; a format the browser cannot play says so), the word being
played lit and followed, a word or a time clicked plays from there, low-probability words dotted, a segment
double-clicked edited (the words that stay keep their times, new ones are timed between their neighbours), find /
replace over the whole transcript (saved, and in every export), exports in the formats ticked with the cue options
(remembered). Checked in the browser: playback and highlight, an edit, a replace (3 found, 3 replaced, in the SRT),
an upload of 23 MB in two pieces, a stop at 9:54 with its text kept, the phone layout. A file of 56 minutes end
to end through the same API: uploaded in 0.4 s, transcribed in 129 s with beam 5 (26x realtime), 5,607 words.

**W6 (2026-10-28): live** (`/v1/live`, the tab's Live button): the microphone and / or a tab's or the screen's
sound (`getDisplayMedia` with "share tab audio") as one or two tracks at the sound card's rate, sent every 0.5 s;
the server writes the recording (WAV, sizes kept right as it grows) and cuts each track into utterances at its
pauses (10-ms energy against a noise floor; 0.6 s, or 0.3 s past 6 s; 15 s at most), transcribed greedy at once,
and the utterance still being said is transcribed again every second as the partial text (grey). Stop: the
recording queued for the accurate pass (beam 5, each track alone with its speaker, merged by time), then kept as
MP3 (Windows' own encoder, 192 kbps unless `mp3_kbps`; ~1.4 MB a minute in stereo). Marks (★) kept. The engine is
shared a window at a time (`Options::window_turn`): a live utterance goes first, and the queue's windows wait while
live utterances are queued. Two tracks sent in real time (60 s, 48 kHz) while the queue transcribed the 14-minute
file with beam 5:

| | Text behind the audio (median / p90) |
|---|---|
| utterances only (cut at 0.6 s or 25 s), nothing else running | final text 2.2 s / 16.5 s after the utterance's end |
| + partial text, the queue busy, live first only when waiting | 6.8 / 9.8 s |
| **+ the queue waiting while live utterances are queued** | **2.5 / 3.8 s** (the queue went on in the gaps) |
| the same, nothing else running | 2.0 / 3.3 s |

**W7 (2026-10-28): the Meetings tab** (the rail's third button): the same list for meetings only; "Start a meeting"
records the microphone as "Me" and the meeting's sound (a tab or the screen) as "Others"; a meeting's recording
dropped there goes the same way. Done: "Summarize" asks the loaded chat model for the summary, decisions, action
items and open questions with their [m:ss], in the meeting's language named (a 4B model answered an Indonesian
meeting in English when only told "the transcript's language"); a transcript longer than the context goes in
parts (one context's worth of characters each), then the parts' summaries together; kept as `summary.md`, shown
above the transcript. "Minutes DOCX / PDF": the summary and the transcript through `create_file` (in the Files
panel too). Checked with the browser's capture fed from two test files: Me / Others in the draft and the accurate
pass, a mark, the MP3, an Indonesian summary, a 15-KB DOCX; a 56-minute meeting summarized in 3 parts with the 4B
model (16K context) in ~6 minutes (each part's summary now at most 700 tokens).

**W9 (2026-10-28): speech in a chat.** A chat request's audio (a file part of an audio file: the web UI's
attachments and voice messages; or an `input_audio` part) is written by Whisper on the server, in `dispatch`
before the model's lock, when the loaded model does not hear audio or the request says `"speech": "whisper"`
(`"model"`: a model that hears it, Gemma 4 E4B / 12B, gets the clip). The text goes in as a text part (ended by a
blank line: the template joins a message's parts, and the question after it ran into the transcript's last word);
one of more than 12,000 characters becomes a text file in the chat's folder, read as a document. Transcripts are
cached by the clip's bytes (each turn sends the audio again). The web UI's setting **Speech** (Voice section):
Whisper for every model, or the model hears it. **Later** (2026-10-28): the speech model is Qwen3-ASR by default,
Whisper when Qwen3-ASR is off ([qwen3-asr.md](qwen3-asr.md#using-it)). Measured:

| | First turn | Next turn |
|---|---|---|
| 10-minute voice note (Ogg Opus), Qwen3.5 4B | 38.8 s (Whisper beam 5 + the answer) | 4.3 s (transcript cached) |
| the same, Qwen3.6 35B-A3B | 46.4 s | 5.8 s |
| 56-minute recording (MP3), 4B: the transcript as a document (7.7K tokens) | 150.7 s | 12.6 s |

Talk mode, Gemma 4 E4B, a 5-s spoken question to the answer's first token (median of 3): Whisper -> text 0.72 s
(0.55 s Whisper greedy + 0.17 s), the model hearing the clip 0.50 s (its first call 21.9 s: the audio encoder's
kernels built once). Whisper costs ~0.2 s and works with every model.

**W8 (2026-10-28): `arc-recorder.exe`** (`tools/arc_recorder.cpp`; `arc-server record [-App Zoom.exe] [-NoMic]
[-Title ...]`, `record-stop`, `record-status`): the microphone ("Me") and the meeting's sound ("Others": one app's
sound through WASAPI process loopback, its child processes included, or everything the speakers play) as a stereo
16-bit WAV at 48 kHz in `<data>\recordings\`, its sizes put right every second. Each packet goes to its place by its
time stamp (QPC), so a silent app, a headset change or a device back after a second leave silence, not a shift
between the tracks. The server is not involved while it records; at the end the recording goes to
`POST /v1/transcripts/import` (this computer only, a WAV in the recordings folder: moved into the library, the
accurate pass per track with Me / Others, then MP3), or, the server down, a sidecar `.json` makes the server take it
at its next start. Checked with a process playing the test clip: a 15-s recording handed over and transcribed
(26 words, the same as the clip's), MP3 kept (the encoder refuses mono at 192 kbps: the next lower bitrate it takes,
128); the server killed at 5 s and started again at 15 s of a 30-s recording: nothing lost (49 words); the server
down at the end: taken at its next start.

**3 hours unattended (2026-10-28, 01:19-04:19)**: `--seconds 10800`, the speakers' sound only (nothing played, so
WASAPI sent no packets: the length, the clock and the file were tested, not 3 hours of sound), alongside the GPU
tests of that night. The WAV came out 10,800.05 s long (1,036,804,786 bytes: the QPC placement lost or added
nothing), its note complete; no server ran, so it was taken at the next start and transcribed (6 min 10 s with
the start, beam 5 over silence), kept as MP3 (172.8 MB, its length checked against the recording's before the WAV
was deleted). It also showed a bug of both WAV writers (arc-recorder's and the live recordings'): the first
header was written at the file's start and the writing went back there, so the audio wrote over it and every
later header went over the first 44 bytes of audio (the data size 44 bytes more than the file held). Fixed: the
samples start after the header (checked: a 3-s recording's data size = its length - 44; a live WAV's bytes = the
bytes sent). After the self audit (2026-10-28) the recorder also follows a new default device (a headset plugged
in), finds an app started again, writes its note at the start (a recorder killed, a shutdown or a power cut: the
server takes the recording at its next start), never writes over a recording of the same minute; the three
checks above passed again with it (26, 50, 21 words).

**NPU mode** (14-minute file): NPU + iGPU greedy 81 s (10.5x realtime; OpenVINO NPU 144 s + 253 s compiling), beam 5
91 s; logits KL 0.00052 against the f32 reference (64 / 64 top tokens). It keeps the iGPU free for the LLM: 0.73 GiB
of GPU memory instead of 1.39 GiB.

**Server** (Qwen3.5 4B loaded, 60-s clip): json / srt / verbose_json with words / text / stream / translations
answered; beam 5 in 5.2 s with the upload (iGPU), 9.6 s in NPU mode; a chat request at the same time answered
normally; 6 uploads at once: 4 answered, 2 got 429; an unknown language, a non-audio file and an unknown format get
400.

**Text over silence and noise** (2026-10-28, a user's live recording in Indonesian): the quiet stretches came out
"Продолжение следует...", "Спасибо.", "Obrigada.", "MBC 뉴스 ...", "Thank you.", each with a no-speech probability
of 0.00. large-v3-turbo does this on 12 s of digital silence too, in the f32 CPU reference as well (no-speech
probability 1.5e-10), so `no_speech_threshold` never skipped a window. Now (src/whisper.h `drop_unvoiced`, on by
default): a segment with less than 0.3 s of voice under it (10-ms frames 12 dB over the recording's quietest tenth,
above -50 dB) is left out, a credit line ("Продолжение следует", "Thanks for watching", "MBC 뉴스") with less than
3 s, thanks in a language other than the speech's with less than 2 s, in its language with less than 0.5 s (a spoken
"terima kasih" stays). A live recording with language auto keeps the language of its first two clear utterances
(`live_lock`). Checked: 12 s of silence / quiet noise, auto and id: 6 of 6 hallucinated segments before, none now; an
80-s live recording of the 14-minute file with 8 s of silence, 8 s of noise and 4 s of silence at the end: the final
transcript's "Продолжение следует..." over the last silence gone, its 30 speech segments as before.

## Next steps

1. Speed: done (W1, above).
2. Mixed languages and jargon: done (W2, above).
3. Formats: checked in Remotion and Chrome and by structure (W3, above); YouTube, Premiere and Resolve by the user.
   Video files and the transcript library: done (W4, above).
4. The Transcripts tab, live recording, the Meetings tab with summaries and minutes: done (W5-W7, above); `@` on a
   recording in a chat: done (QUEUE 8). `arc-recorder.exe`: done (W8, above); the browser's live text does not
   come from its recordings (they are transcribed when they end).
5. Speech in a chat and the Talk mode choice: done (W9, above).
4. Then the web UI tabs (Transcript, Meeting), `arc-recorder.exe` (separate process, WASAPI process loopback +
   microphone, `recorder.json` + named pipe, reconnect after a server crash), Talk mode with the choice of Gemma
   hearing directly or Whisper -> any model.

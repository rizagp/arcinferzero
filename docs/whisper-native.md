# Whisper native: speech to text on Level Zero, transcripts and meetings

Status: plan (2026-10-25), nothing built yet. Whisper large-v3-turbo written from scratch on arcinferzero's Level Zero
engine, like the LLMs: no OpenVINO, no CTranslate2, no Python at run time. On top of it, two functions in the web UI
and one on the command line:

- **Transcript** ("Transkrip"): any audio or video file in, a transcript with word timestamps out (TXT, SRT, VTT,
  JSON, Markdown, and the inputs of the Remotion video pipeline);
- **Meeting** ("Rapat"): recording a meeting (Meet, Zoom, Teams, ... and the microphone), the live transcript, then
  an accurate pass, a summary, decisions and tasks;
- **`arc-transcribe`**: the same transcription from the command line, for scripted workflows (record -> transcribe
  -> Remotion input, in one line).

Both web UI functions open only once the encoder, the decoder and the timestamps are done and checked (phases 0-4).

## The five functions (the user, 2026-10-25)

| # | Function | Where | What it does |
|---|---|---|---|
| 1 | **Realtime transcript** | web UI (Transcript and Meeting tabs), `arc-transcribe --live` | text ~1 s after each sentence from the microphone and / or an app's sound (system / process loopback), recorded to `sound-recordings\` at the same time |
| 2 | **Accurate transcript after recording** | button after a realtime session, `arc-transcribe <recording>` | the whole recording again: full 30-s windows, beam 5, word timestamps; replaces the live draft (marks and edits kept) |
| 3 | **Audio upload** | drop / choose files, `arc-transcribe <file>` | wav, mp3, m4a / aac, flac, wma, ogg / opus, ... |
| 4 | **Video upload** | the same | mp4, mov, mkv, webm, wmv, avi, ...: the audio track is read, the video itself is not touched; timestamps match the video's timeline |
| 5 | **Output formats, chosen per export** | checkboxes in the web UI, `--format a,b,c` | Remotion, subtitles (SRT, VTT, YouTube), standard formats: table below |

### Output formats

| Format | File | For |
|---|---|---|
| Remotion captions | `.captions.json` | `Caption[]` of `@remotion/captions` (`text`, `startMs`, `endMs`, `timestampMs`, `confidence`), one per word: `<Captions>`, `createTikTokStyleCaptions` |
| Remotion pipeline | `.fw.json` | the JSON arcinferzero-channel's `edit.py` / `verify_cut.py` / `timeline.py` read (faster-whisper's: segments with words, `start` / `end` / `word` / `p`) |
| SubRip | `.srt` | the common subtitle file: YouTube, video editors (Premiere, DaVinci Resolve, CapCut), VLC |
| WebVTT | `.vtt` | web players, YouTube; with word timing (`<00:00:01.200>` cues) for karaoke-style highlighting |
| YouTube SBV | `.sbv` | YouTube's own subtitle format (SubViewer) |
| YouTube SRV3 / YTT | `.srv3` | YouTube's timed text with per-word timing (auto-caption style word-by-word display) |
| Advanced SubStation | `.ass` | styled subtitles (Aegisub, mpv, burned in with ffmpeg); `\k` word timing for karaoke |
| TTML / DFXP | `.ttml` | broadcast / streaming platforms, Premiere import |
| Plain text | `.txt` | the text only, one segment per line, optionally with `[hh:mm:ss]` |
| Markdown | `.md` | readable transcript with timestamps, speakers, marks; the meeting summary on top |
| JSON | `.json` | everything (segments, words with probabilities, languages, speakers, marks): the source the others are made from |
| CSV / TSV | `.csv`, `.tsv` | start, end, speaker, language, text per segment (spreadsheets) |

Subtitle cues (SRT, VTT, SBV, ASS, TTML) are cut from the word timestamps, not from Whisper's segments: at most 2
lines of 42 characters (YouTube's guidance), 1-7 s per cue, a break at sentence ends and pauses > 0.5 s, never inside
a word; all of it adjustable (`--max-chars`, `--max-lines`, `--max-cue-s`, or one word per cue for vertical
short-form video). Every export comes from the saved JSON, so a format can be added later without transcribing again.

### Progress

Every transcription that is not live shows its progress, the same numbers everywhere:
- **stage**: reading the file (decoding a video's audio track) -> transcribing -> word timestamps -> writing the
  formats;
- **position**: audio done / total (`00:09:55 / 00:13:58`, 71 %), from the windows finished;
- **speed and time left**: x realtime so far (e.g. 31x), elapsed, estimated time left;
- **the text so far**: segments appear as they are finished.

Where: a bar per file in the Transcript tab (several files queued, each with its own bar, one running at a time), in
the Meeting tab during the accurate pass; on the command line a one-line bar on stderr
(`02-clean.wav  [██████████░░░░]  71 %  09:55 / 13:58  31x  ~0:08 left`), or one JSON line per update with
`--progress` for scripts; through the API as server-sent `progress` events (`stream=true`).

## Why, and the target

Measured 2026-10-24 (arcinferzero-channel, `work/bench/hasil.md`): one 14-minute recording, Whisper large-v3-turbo
int8, this laptop.

| Engine | Transcription | x realtime | 1-hour meeting | Hard terms right | Word timestamps vs faster-whisper (mean / p90) |
|---|---|---|---|---|---|
| OpenVINO GenAI, iGPU, greedy | 57 s | 14.8x | ~4 min | 15 / 24 | 131 / 440 ms |
| OpenVINO, NPU | 144 s (+ 4 min compile) | 5.9x | ~10 min | 16 / 24 | 140 / 520 ms |
| faster-whisper, CPU, beam 5 | 247 s | 3.4x | ~17 min | 16 / 24 | (the reference) |

Where OpenVINO's 52.6 s on the iGPU go, and the target here (estimates from arcinferzero's measured throughput, to
be confirmed):

| Part | OpenVINO | Target | Why it can be faster |
|---|---|---|---|
| Encoder (32 windows of 30 s) | 25.1 s (785 ms per window, ~2.9 TOPS) | <= 12 s | it is a 1,500-row prompt pass; arcinferzero's int8 DP4a GEMMs run ~7 TOPS |
| Decoder (2,241 tokens) | 12.6 s (5.6 ms / token) | <= 9 s | ~170 MB of weights per token: 3-4 ms at the bandwidth arcinferzero reaches |
| Word timestamps (DTW) | 7.3 s (CPU) | <= 1 s | the alignment heads' cross-attention kept on the GPU, DTW written for it |
| Mel + the rest | ~7.6 s | <= 3 s | |
| **Total (greedy)** | **52.6 s** | **<= 25 s (>= 2x, ~34x realtime: 1-hour meeting ~2 min)** | |

## Greedy and beam 5

The decoder writes the text token by token. **Greedy** takes the single most likely token at every step: fastest,
and usually right. **Beam 5** keeps the 5 most likely partial sentences side by side and picks the best whole one at
the end: it recovers when the first guess of a word was wrong (names, rare terms, mixed languages), at more decoder
work. On the 14-minute file faster-whisper's beam 5 got 16 / 24 hard terms, greedy 13 / 24.

Here beam 5 costs less than 5x: the decoder is bandwidth-bound, and the 5 hypotheses run as 5 rows of one pass (the
weights read once for all, as the engine's batched decoding of two conversations already does). Expected: decoder
~1.5x slower, the whole file ~1.2x. Defaults: **Transcript and the meeting's accurate pass: beam 5**; the live meeting
transcript: greedy (speed matters there, the accurate pass fixes it afterwards). `--beam 1|5` overrides.

## The model

- **Whisper large-v3-turbo** (openai, MIT): 128 mel bins, encoder 32 layers (d 1280, 20 heads x 64, MLP 5120, GELU,
  pre-LayerNorm with bias), 1,500 encoder positions per 30-s window; decoder 4 layers (self-attention, causal, plus
  cross-attention to the encoder states), 448 text positions, vocabulary 51,866 (the output head tied to the token
  embedding), alignment heads (decoder layer, head) = (2,4) (2,11) (3,3) (3,6) (3,11) (3,14). 99 languages.
- **Weights**: the f16 CTranslate2 file already in the Hugging Face cache
  (`mobiuslabsgmbh/faster-whisper-large-v3-turbo/model.bin`, 1.62 GB, format version 6, `WhisperSpec`, 486 variables:
  name, shape, dtype, bytes), with its `tokenizer.json` and `config.json`; no download needed. Later also the
  original `openai/whisper-large-v3-turbo` safetensors (the same tensors; arcinferzero already reads safetensors for
  DFlash2).
- **On the GPU**: the encoder and decoder matrices as resident W8 (int8, per-32 scales, the engine's GEMM and GEMV
  forms): ~0.85 GB, plus per window the cross-attention K / V of the 4 decoder layers (1,500 x 1280 x 2 x 4, f16:
  ~30 MB) and the decoder's self-attention cache (448 positions, x 5 with beam 5). Loaded on demand next to the LLM
  and counted in the GPU budget (`--ctx auto` leaves room for it, as for the image encoders).

## The pipeline

1. **Audio and video files**: everything `src/audio.h` reads: WAV directly, and through Windows Media Foundation
   mp3, m4a / aac, flac, wma, and the audio track of video files (mp4, mov, wmv; mkv / webm / ogg / opus where
   Windows has the codec: to be checked in phase 4, with a built-in decoder for any that is missing). -> 16 kHz mono
   float, the same resampler the Gemma audio path uses. A file that cannot be opened fails with its name and the
   formats that work.
2. **Log-mel** (Whisper's exact recipe, not Gemma's): 400-sample periodic Hann window, hop 160, reflect-padded STFT
   (center), power spectrum, 128 Slaney-normalized mel filters (librosa `htk=False, norm="slaney"`, 0-8000 Hz),
   `log10(max(x, 1e-10))`, clamped to `max - 8`, then `(x + 4) / 4`; each window padded to 30 s = 3,000 frames. On
   the GPU (the FFT as a small kernel), checked against the CPU version value by value.
3. **Encoder** (per 30-s window, several windows per batch when the file is long): conv1 (128 -> 1280, kernel 3,
   pad 1) + GELU, conv2 (kernel 3, stride 2) + GELU as im2col + the int8 GEMM, + sinusoidal positions, 32 layers
   with **bidirectional** attention over 1,500 positions (no mask: a variant of the prompt attention kernels), final
   LayerNorm. New kernels: the convolutions, LayerNorm with bias (the vision encoders have one), GELU fused into the
   GEMM epilogue where it already exists.
4. **Decoder**: tokens `<|startoftranscript|> <|lang|> <|transcribe|>` (+ `<|notimestamps|>` when no timestamps are
   asked for), optionally preceded by `<|startofprev|>` and up to 224 tokens of the previous window's text or a
   glossary prompt. Per token: self-attention on its cache, cross-attention on the window's K / V (computed once per
   window right after the encoder), MLP, the head (51,866 x 1280: the largest read per token). Greedy or beam 5
   (above); timestamp rules as in openai/whisper (timestamp tokens <|0.00|> .. <|30.00|> in 0.02-s steps, in pairs,
   never decreasing; a timestamp is forced when their total probability beats every text token); the window
   advances to the last complete segment's end.
5. **Languages**: `auto` (default) detects the language **per 30-s window** from the language token's probabilities,
   so a recording that switches between Indonesian and English (or Javanese, Malay, ...) is written in each
   language as spoken; every segment carries its language code. `--lang id` (or any of the 99) fixes it, which is
   more robust for a single-language recording with many foreign terms. Translation to English (Whisper's translate
   task) on request.
6. **Robustness** (the openai / faster-whisper defaults): temperature fallback 0 -> 0.2 -> ... -> 1.0 when the text's
   gzip compression ratio > 2.4 or the mean log-probability < -1.0; a window whose no-speech probability > 0.6 and
   mean log-probability < -1.0 is skipped; repetition guard (the same segment text three times in a row ends the
   window). These keep repeated sentences and text invented in silence at zero on long recordings.
7. **Word timestamps**: during decoding the six alignment heads' cross-attention weights (tokens x 1,500) stay on the
   GPU; then normalized per head, median filter of width 7, averaged, DTW against the audio frames, words formed from
   the tokenizer's tokens (leading space = new word, punctuation joined to the word before), each word's start / end
   from the DTW path, plus each word's probability. On by default for Transcript (video editing needs them).
8. **Optional VAD**: silent stretches skipped before the encoder (energy + zero-crossing first; a native Silero-style
   model later if it helps). Off for exact timestamps, on for meetings.
9. **Glossary prompt**: a list of terms or a sample sentence the decoder sees first ("arcinferzero, Qwen, GGUF, SYCL,
   llama.cpp, Unsloth, SysMain"; for video narration also filler words "eh, em, hmm" so they get written and can be
   cut). Per call, or a default saved in the settings.

## Accuracy and checks (as for every arcinferzero model)

- **f32 reference**: `tools/ref_whisper` (CPU, f32, from the same weights): the mel, the encoder output, the decoder
  logits per step. The GPU engine against it: KL of the next-token distributions <= 0.015 (the project's limit),
  encoder output relative error reported per layer.
- **End to end against faster-whisper** (CPU, beam 5, f16: the transcript the videos are cut with) on the 14-minute
  recording, on a real meeting recording, and on a mixed Indonesian / English one: word differences <= OpenVINO's
  (8.7 %; with beam 5 close to 0 %), hard terms >= 15 / 24 (beam 5: >= 16), word timestamps mean <= 100 ms and
  p90 <= 300 ms (better than OpenVINO's 131 / 440), no repeated sentences, no text in silence, the language of each
  window right.
- **Deterministic**: the same file and options give the same text and timestamps bit for bit, run to run.
- **Bench**: `bench/run_bench.ps1` gets a `whisper` test (encoder ms per window, decoder ms per token, DTW ms, total,
  x realtime, greedy and beam 5), compared with OpenVINO GenAI from the channel's `.venv-ov` on the same file.

## Where the files go

| Folder | What |
|---|---|
| `%USERPROFILE%\arcinferzero\sound-recordings\` | recordings: `<yyyy-mm-dd hh.mm> <title>\` with `others.wav` (the meeting app) and `me.wav` (the microphone), or one `.wav` / the uploaded file for a single source |
| `%USERPROFILE%\arcinferzero\transcripts\` | transcripts: `<same name>.json` (everything: segments, words, languages, speakers, marks), `.txt`, `.srt`, `.vtt`, `.md`, and the meeting's `summary.md` |

Both are in arcinferzero's data folder (`C:\Users\<name>\arcinferzero\`, since 2026-10-25; visible in Explorer), next
to its caches and model packages. Both can be changed in the settings (and with `--out`).

## Command line: `arc-transcribe`

`bin\transcribez.exe`, installed as the `arc-transcribe` command next to `arc-server` and `pi-arc`. If the
arcinferzero server is running it sends the work there (the Whisper already loaded, no second copy in GPU memory);
otherwise it loads Whisper itself and exits when done.

```
arc-transcribe <file | folder | wildcard> ... [options]
  --lang auto|id|en|id,en     language (default auto, per 30-s window); a list: speech that mixes them, each
                              utterance's language checked (~3x slower; docs/whisper-native-status.md, W2)
  --beam 1|5                  greedy or beam 5 (default 5)
  --words / --no-words        word timestamps (default on)
  --terms "<list>"            a glossary kept before every window ("GPU, kernel, deploy", ~150 tokens at most);
                              --terms-file <file>. A bare list makes the text formal; terms inside a casual
                              sentence keep the casual spelling
  --prompt "<text>"           the first window's previous text (a sample sentence);  --prompt-file <file>
  --dual-language P           auto: a window whose likeliest language is below P (0.8) is split by language
  --previous-text             each window with the text before it as its prompt (off: it carried errors over
                              and dropped whole windows in the tests)
  --vad                       skip silence (meetings; off by default for exact timestamps)
  --translate                 to English instead of transcribing
  --from <s> --to <s>         only part of the file
  --format <list>             json, srt, vtt, sbv, srv3, ass, ttml, txt, md, csv, tsv, captions (Remotion
                              Caption[]), fw (the Remotion pipeline's JSON); default json,srt,txt
  --max-chars N --max-lines N --max-cue-s S   subtitle cue shape (default 42, 2, 7); --word-cues: one word per cue
  --out <folder | file>       default %USERPROFILE%\arcinferzero\transcripts\
  --live [--app chrome|zoom|...] [--mic]   realtime from the microphone / an app's sound, recorded to
                              sound-recordings\; Ctrl+C stops, then the accurate pass runs unless --no-accurate
  --progress                  progress and segments as JSON lines on stdout (default: a bar on stderr)
  --local / --server <url>    force the in-process engine or a server
```

The two Remotion formats (the full list is under "Output formats" above):
- `fw`: the JSON the Remotion channel pipeline reads today (`tools/audio/transcribe.py`'s output:
  `{"duration", "segments": [{"start", "end", "text", "words": [{"start", "end", "word", "p"}]}]}`), so
  `edit.py`, `verify_cut.py` and `timeline.py` work unchanged; each segment also carries `"lang"`.
- `captions`: `Caption[]` of `@remotion/captions` (`text`, `startMs`, `endMs`, `timestampMs`, `confidence`), one
  caption per word with the leading space Remotion's `createTikTokStyleCaptions` expects, for `<Captions>` /
  subtitle components directly.

Exit code 0 when every file was transcribed, 1 otherwise (each failure named on stderr). Example, the channel's
narration workflow without Python's faster-whisper:

```
arc-transcribe work\audio\02-clean.wav --lang id --prompt-file tools\audio\prompt.txt --format fw --out work\audio\transcript.json
arc-transcribe work\audio\03-edited.wav --lang id --format fw,captions --out work\audio\transcript-edited.json
```

Recording from the command line too: `arc-server record start [--app chrome] [--title "..."]` / `record stop`
(through `arc-recorder.exe`, below) writes into `sound-recordings\`; `arc-transcribe` on that folder then makes the
transcript. A Pi skill / tool can call the same command.

## Server API

Built (2026-10-26, `tools/server_whisper.h`): `serverz --whisper auto|<folder>` (default off),
`--whisper-device gpu|npu`, `--whisper-max-minutes N` (default 180); `arc-server -Whisper auto [-WhisperDevice npu]`.

- `POST /v1/audio/transcriptions` and `POST /v1/audio/translations`, multipart as OpenAI's API: `file`, `model`,
  `language`, `prompt`, `temperature` (0 = the fallback through higher temperatures), `response_format` = `json`
  (`{"text", "usage"}`) | `text` | `srt` | `vtt` | `verbose_json` (segments, and `words` with
  `timestamp_granularities[]=word`), `stream=true` (server-sent events `transcript.text.delta`, then
  `transcript.text.done` with `usage`). Extensions: `response_format` = `fw` | `captions` | `full_json` | `sbv` | `srv3`
  | `ass` | `ttml` | `md` | `csv` | `tsv`; `language` = `auto` or a list (`id,en`); `beam_size` 1..5 (default 5);
  `terms` (a glossary kept before every window); `dual_language` (0..1, default 0.8);
  `condition_on_previous_text=true` (off by default); `transcript.progress` events (`done`, `total` seconds) while streaming.
- Which model answers: Whisper, unless `model` names a chat model (the loaded one or one of `--list-models`: then the
  old path, Gemma 4 hearing the clip). Without `--whisper`, a `model` containing "whisper" gets a 400 that says so.
- Translation: large-v3-turbo was not trained for the translate task and mostly answers in the spoken language; a
  chat model translating the transcript is the better route.
- **Next to the LLM**: Whisper loads before the model; `--ctx auto` keeps its GPU memory free (iGPU 1.39 GiB, NPU mode
  0.73 GiB plus 0.63 GiB of RAM for the NPU's weights). Each Whisper step (a window's encoder, a decoder token) takes
  the model engine's lock, as a request's step does: both engines share one GPU command list. A model switch can come
  between two steps. `/props` has a `whisper` entry (model, device, max_minutes).
- **Safety**: the upload is held in memory within the HTTP body limit (256 MB) and decoded from memory by Media
  Foundation (WAV directly); no path or file name from the client is used; decoding stops past the length limit; one
  transcription runs at a time, at most 4 requests are running or waiting (429 past that); a client that closes its
  connection stops its transcription at the next window; the server listens on 127.0.0.1 unless `--host` says otherwise.

## The Transcript function ("Transkrip")

A tab in the web UI header, next to Chat and Meeting:

```
[ Chat ]  [ Transkrip ]  [ Rapat ]
Drop audio / video files here, or [Choose files]   or [● Live: mic / app sound]   Language: (•) auto ( ) id ( ) en ...
                                                           [✓] word timestamps  Beam: (•) 5 ( ) greedy
Glossary: arcinferzero, Qwen, GGUF, ...                    [✓] write filler words (eh, em, hmm)
───────────────────────────────────────────────────────────────────────────
02-clean.wav     13:58   ██████████░░  71 %   00:09:55 / 00:13:58
03-edited.wav    13:03   queued
───────────────────────────────────────────────────────────────────────────
00:00:01.20  id  Halo, jadi di video ini saya mau cerita...        (words: click to play from there)
Formats: [✓] SRT  [ ] VTT  [ ] YouTube SBV  [ ] YouTube SRV3  [ ] ASS  [ ] TTML  [✓] TXT  [ ] MD  [ ] CSV
         [ ] Remotion captions  [ ] Remotion pipeline (fw)  [✓] JSON      Cues: 42 chars x 2 lines, <= 7 s  [ ] one word
[Export]   Saved in transcripts\  (the format choice is remembered)
```

- Several files queued, each with its progress; the transcript appears segment by segment while it is processed.
- Playback: clicking a segment or a word plays the audio from that point; low-probability words are shaded, so
  wrong terms are easy to find; a find-and-replace for terms that applies to the text and to every export.
- Everything saved in `transcripts\` as it is made; a history of past transcripts.

## The Meeting function ("Rapat")

```
[ Chat ]  [ Transkrip ]  [ Rapat ]
● REC  00:42:17   Source: (•) Chrome  ( ) Zoom  ( ) all system audio   + my microphone ✓   ★ Mark
00:41:52  Others   ...so we move the deadline to the 15th, agreed?
00:42:05  Me       OK, I will tell the design team.
00:42:14  ▍ (listening...)
search in the transcript...
After the meeting: [Accurate pass] [Summarize] [Download TXT / SRT / MD]      History: 24 Oct · 22 Oct · ...
```

- **Recording that is never lost**: `arc-recorder.exe` (new, small, separate process) records WASAPI **process
  loopback** (only the chosen app: Chrome for Meet / Zoom web, Zoom, Teams, WhatsApp; Windows 10 2004+ / 11) and the
  microphone as two tracks, writes WAV to `sound-recordings\` continuously, survives a server restart, and follows a
  headset change. Microphone track = "Me", loopback = "Others": two speakers separated without a diarization model.
  `arc-server record start | stop`, or the button in the menu. Before it exists: the browser microphone and
  `getDisplayMedia` (screen + system audio) as the source.
- **Live transcript** ~1 s after each sentence: VAD pieces (the web UI's Talk mode already cuts them: 1 s of
  silence, 30 s at most, 300 ms before the voice), greedy, the previous text as context, text confirmed once two
  passes agree (LocalAgreement, as whisper-streaming), the language per piece. Marks (★), search, auto-scroll that
  stops when you scroll up.
- **After the meeting**: "Accurate pass" transcribes both tracks again with full windows, beam 5 and word
  timestamps (~2-3 min per hour of audio); "Summarize" asks the loaded LLM for a summary, decisions and tasks, each
  linked to its minute in the transcript, in the meeting's language. Long meetings: summaries per 10-15 minutes
  while recording, the final one from those; questions ("who agreed to the deadline?") search the transcript first
  and send only the matching parts; when the whole transcript fits the model's context, it becomes a cached prefix
  (asked once, follow-ups only pay for the question).
- **Old recordings**: an uploaded file (m4a from a phone, ...) goes through the same accurate pass and summary.
- **Local only**: recordings, transcripts and summaries never leave the laptop.

## Phases (each ends with its checks; the web UI functions open after phase 4)

| Phase | What | Done when |
|---|---|---|
| 0 | CT2 / safetensors loader, tokenizer, `tools/ref_whisper` (f32 CPU, mel -> text, greedy and beam 5) | the reference transcribes the 14-minute file like faster-whisper |
| 1 | GPU mel + encoder (convs, bidirectional attention, W8 GEMMs), several windows per batch | encoder output vs f32 within tolerance; <= 12 s for the 14-minute file |
| 2 | Decoder: greedy and beam 5 (hypotheses as rows of one pass), cross-attention K / V once per window, timestamp rules, fallback, no-speech, languages per window, prompt | KL <= 0.015 vs f32; transcript checks above; <= 9 s decoder greedy |
| 3 | Word timestamps (alignment heads on the GPU, DTW) | mean <= 100 ms, p90 <= 300 ms vs faster-whisper; <= 1 s |
| 4 | `transcribez` / `arc-transcribe` (all formats, progress bar) and the server endpoint (formats, streaming, memory budget, taking turns with the LLM) | total <= 25 s greedy for the 14-minute file; the channel pipeline runs on `fw` output unchanged; bench row; docs |
| 5 | Transcript tab: audio and video upload, queue with a progress bar per file, playback, term replace, the format choice (every format above), history; realtime from the microphone / app sound with the accurate pass after it | every format opens in its target (YouTube upload, Premiere / Resolve import, Remotion `<Captions>`) |
| 6 | Meeting tab with files and the browser microphone / screen audio: live transcript, accurate pass, summary | text ~1 s after a sentence |
| 7 | `arc-recorder.exe`: WASAPI process loopback + microphone, Me / Others, `arc-server record` | a 3-hour test recording without a gap |
| 8 | Summaries per segment while recording, questions over the transcript (search + cached prefix) | |
| later | Whisper large-v3 (full 32-layer decoder, slower, a little more accurate), diarization of the "Others" track, Meet captions for names (Chrome extension), the NPU (measured first: resident encoder weights gained nothing for the LLMs) | |

## Decided (the user, 2026-10-25)

- Two functions, Meeting and a general Transcript (any audio / video file, e.g. for videos), plus `arc-transcribe` on
  the command line feeding the Remotion pipeline.
- Multilingual: `auto` per window by default.
- Folders: `%USERPROFILE%\arcinferzero\sound-recordings\` and `%USERPROFILE%\arcinferzero\transcripts\`.
- The five functions: realtime, accurate after recording, audio upload, video upload, a choice of output formats
  (Remotion, subtitles incl. YouTube's, the common standard ones); a progress bar for every transcription.
- Beam 5 as the default for Transcript and the accurate pass, greedy for live (2026-10-27).
- Recordings are kept after their transcript, as MP3 (2026-10-27): WAV while recording (nothing lost in a crash),
  converted to MP3 with Windows' Media Foundation encoder after the accurate pass, at 192 kbps by default (~86 MB per
  hour per track), configurable: Settings in the web UI, `recorder.json`, `arc-transcribe --mp3-kbps` (96-320 kbps,
  or keep the WAV). 192 kbps needs MPEG-1 Layer III (32 / 44.1 / 48 kHz; 16 kHz stops at 160 kbps): the recording keeps the
  device's own rate (48 kHz), Whisper resamples it to 16 kHz.

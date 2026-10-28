# Using arcinferzero

arcinferzero is an LLM inference engine built specifically for this laptop (Intel Core Ultra 5 125H, integrated Intel
Arc GPU). Its focus is processing long prompts quickly, for example when an agent such as Pi reads many code files.
It runs GGUF models directly from the Hugging Face cache and serves an OpenAI-compatible API, so it can be used by
Pi, by other apps, or through its built-in web UI.

## 1. Setup (once)

Open PowerShell or Terminal in the arcinferzero folder, then run:

```powershell
.\setup.cmd
```

The setup is safe to run again (steps that are already done are skipped). It:

- checks the requirements: 64-bit Windows, an Intel Arc GPU, Visual Studio Build Tools (C++), Python and Pi, and
  says how to install whatever is missing;
- builds `bin\serverz.exe`, `runz.exe` and `clinfo.exe` when they are missing or older than their sources, then
  checks the GPU (OpenCL 3.0, DP4a, GPU memory);
- asks which models to download into the Hugging Face cache (`.\setup.cmd -Models 27b,4b` without asking; Gemma
  comes with its drafter and image projector; the 35B is downloaded and then converted). `huggingface_hub` /
  `numpy` are installed with pip when needed (it asks first);
- adds the `arcinferzero` provider to `~/.pi/agent/models.json` with every model it finds and each one's context
  (other providers are left unchanged, the previous file is kept as `models.json.bak`), and installs the `pi-arc`
  and `arc-server` commands into `~/.pi/agent/bin` (already on the PATH), so they can be called from any folder.

`.\setup.cmd -Tests` also runs the regression tests.

Download a model by hand: see [README, section 6](README.md#6-download-models) (every model, its shortcut and the
`hf download` command). `pi-arc 35b` is Qwen3.6 35B-A3B UD-IQ4_XS from Unsloth's MTP repository (MTP and images);
the `35b` choice in `setup.cmd` downloads ISTA-DASLab's GSQ 2-bit version instead and converts it once (no MTP;
[README, section 7](README.md#7-qwen36-35b-a3b-optional-convert-the-gsq-checkpoint)). Qwen3.8 Flash Next (`flash`)
and the DFlash2 drafters are not downloaded by setup.

## 2. Using Pi

1. Open a terminal in the project folder you want to work on. In File Explorer: open the folder, right-click an
   empty area, then choose **Open in Terminal**.
2. Run `pi-arc`. On first use, the server is started in the background first (5–40 s until "Ready" appears), then
   Pi opens.
3. Type your requests in Pi as usual.

| Command | What it does |
|---|---|
| `pi-arc` | The model that is running or was used last (the first time: choose from a list) |
| `pi-arc --pick` | Choose a model from the list of every downloaded model, by full name |
| `pi-arc 27b` | Qwen3.8 27B: very smart, slowest |
| `pi-arc 35b` | Qwen3.6 35B-A3B: smart and fast (mixture of experts, only ~3B active parameters per token) |
| `pi-arc flash` | Qwen3.8 Flash Next: large mixture of experts, slower (~6-7 tokens/s) |
| `pi-arc g12` | Gemma 4 12B: reads images (attach an image in Pi or the web UI) |
| `pi-arc e4b` | Gemma 4 E4B: small and fast (~32 tokens/s with its drafter); hears audio: the voice model for Talk in the web UI |
| `pi-arc g26` | Gemma 4 26B-A4B (Google QAT Q4_0): mixture of experts (~4B active), ~25 tokens/s with its drafter, reads images; 90K context with images, 118K with `--vision off` |
| `pi-arc 9b` | Qwen3.5 9B: faster, medium quality |
| `pi-arc 4b` | Qwen3.5 4B: fastest |
| `pi-arc --think off` | No thinking (reasoning) phase: much faster answers |
| `pi-arc --think medium` | Thinks longer for hard tasks (`off` / `minimal` / `low` / `medium` / `high`; default `low`) |
| `pi-arc 9b --think off` | A fast combination for everyday questions |
| `pi-arc -c` | Continue the last session in this folder (`pi-arc -r`: choose a session) |
| `pi-arc -p "explain app/auth.py"` | Other Pi arguments still work, written after the pi-arc options |
| `pi-arc --ctx 32768` | A different context size (default = the largest that fits, see section 5); `pi-arc --ctx auto` goes back to automatic |
| `pi-arc --no-mtp` | No speculative decoding (MTP / drafter) |
| `pi-arc --help` | Every pi-arc option, the server and the files it uses |

Inside Pi, `/model` shows every model; choosing one makes the server switch to that model (5–40 s), with its own
context. The next `pi-arc` uses the model chosen last.

The server keeps running after Pi is closed. The next session is ready right away without loading the model again,
and a repeated prompt (for example Pi's system prompt) comes from the cache. If `pi-arc` is called with another
model, the server switches models automatically.

Pi's system prompt is also saved to disk (`%USERPROFILE%\arcinferzero\zero\prefix`), so after the server restarts or
switches models, a new Pi session still starts within a few seconds (27B: ~5 s instead of ~40 s). Pi's system
prompt contains the project folder and its `AGENTS.md`, so the saved prompts are grouped per project folder: each
folder keeps its 2 latest versions and is not pushed out by other folders (27B: ~290 MB per file). All models
together are limited to 8 GB (the least recently used go first; change it with `--prefix-cache-gb`).

A new session always starts empty: Pi only sends its system prompt and your message, so the model does not see the
conversation of an earlier session. To continue a conversation, use `pi-arc -c` or `pi-arc -r`. What the server
processes then:

| Situation | What is processed (27B) |
|---|---|
| right after Pi is closed | only the new message (the server still holds that conversation) |
| after another session or project | only the new message: the session cache (on by default, 8 GB) saves the conversation that was left to disk and loads it again in seconds (`pi-arc --session-cache 0` turns it off) |
| after `arc-server stop` / `restart` or a model switch | only the new message: the conversation being held is saved to disk first and then loaded again (lost only if the server process is killed) |

Pressing Esc in Pi stops the server right away (while reading the prompt: within ~1 s; while writing the answer:
before the next token), so the next message does not have to wait. The next message also does not process the
whole conversation again: only the new message is processed.

## 3. Managing the server

**Through the window:** double-click `arc-launcher.cmd` in the project folder. The window shows:
- every model and its variants (MTP, images, auto context, VRAM at that context, compatible or not);
- for the selected model: how the auto context is counted (GPU memory minus 1.2 GiB for the driver, snapshots,
  image encoder, weights, MTP block and buffers, then the rest divided by the memory needed per 1,000 tokens);
- the server option fields: the main ones are shown directly (context, MTP drafts, thinking budget, prompt cache,
  access from other computers); the rest (port, chunk, image tokens, W8 weights, cache folders and sizes,
  `--no-swap`, f32 attention) appear after ticking "Show advanced options";
- monitoring of the running server: model, filled context, VRAM per part, number of requests, the last prompt and
  answer speed, the percentage of MTP drafts accepted.

All fields are filled in automatically with the best values for the selected model, so just choose a model and
press Start. Fields you change turn yellow and only those are sent to the server. "Set as default" saves the
changed fields and makes the selected model the default one (also for `pi-arc` without a model name); "Reset
options" goes back to the best values. The Start / Stop buttons use `arc-server`, so the server keeps running after
the window is closed. (The window text is in English: prompt cache = saved conversations, prefix cache = saved
system prompt.)

The thinking budget in the launcher only applies to clients that do not send their own budget (web UI, other
apps). Pi sends its own budget according to its `/thinking` level: minimal = thinking off, low 2,048, medium 8,192,
high/xhigh 16,384 tokens.

**Through the terminal:**

```powershell
arc-server status              # whether the server is running, which model is loaded, size of the conversation cache
arc-server stop                # stop the server and free the GPU memory (12–16 GB depending on the model)
arc-server clear-cache         # delete every saved conversation (session cache; on by default, turn off with pi-arc --session-cache 0)
arc-server start -Model 9b     # start with a particular model without opening Pi
arc-server start -Model 27b -Ctx 32768 -NoMtp
arc-server start -Model 27b -Ctx auto   # back to the automatic context (restarts a server with a fixed context)
arc-server restart             # restart, for example after the GPU memory setting was changed
arc-server start -Model 35b -MtpDrafts 2 -ThinkBudget 4096 -NoSwap   # other options (see arc-server help)
arc-server help                # every option
```

The server log is `%USERPROFILE%\arcinferzero\server.log`. The server uses port 8090.

**arcinferzero (the engine):** the server runs through Level Zero (Intel's GPU driver). For
MoE models (Qwen3.6 35B) the experts are kept on the SSD and only the frequently used experts are in GPU memory (a
cache): the context can reach 262K together with images and MTP.
The automatic context reserves 128K tokens, then sizes the expert cache (35B: 131,072 tokens with ~76-81 % of the
experts cached), because a small cache makes generation pause for SSD reads; 262,144 with `--ctx 262144`.

```powershell
arc-server start -Model 35b          # experts streamed from the SSD
pi-arc 35b
pi-arc 35b --no-stream               # without the SSD cache: every expert on the GPU (a smaller context)
```

On first use, arcinferzero writes the expert file once (35B UD-IQ4_XS: 16 GB, about 2.5 minutes; Flash Next: 56 GB, about 9.5 minutes) to
`%USERPROFILE%\arcinferzero\zero\`. This file can be deleted at any time; when it is missing, it is written again
automatically. The cache size is set automatically: arcinferzero uses the GPU memory Windows reports for the iGPU (your
shared-memory setting) less 1.2 GB for the driver, and the cache takes what the rest of the model leaves (it can be
changed: server options `--expert-cache-gb`, `--expert-cache-pct`). If GPU memory suddenly runs short because of other apps (e.g.
Chrome), the cache shrinks itself by 1 GB per step instead of crashing.

arcinferzero notes:
- The SSD cache is only for MoE models (35B, Qwen3.8 Flash Next and Gemma 4 26B; the experts of the 35B's MTP
  layer are in the same file). Dense models (27B, 9B, 4B, Gemma 12B / E4B) keep every weight on the GPU.
- Expert files are never deleted automatically: the next start uses them right away (35B about 10 s, including
  filling the GPU cache from the last list). They are rewritten only when the GGUF changes; files of an old version
  have to be deleted by hand from `%USERPROFILE%\arcinferzero\zero\`.
- Disk space: a MoE model's experts are stored twice, in the GGUF and in the expert file (Gemma 4 26B: 14.4 + 13.2
  GB). `bin\serverz.exe --pack <model.gguf>` turns them into a package in `%USERPROFILE%\arcinferzero\zero\models\`: a
  small GGUF with the other weights (26B: 1.6 GB) and the expert file (moved, not written again: seconds). Copy the
  `mmproj` file next to the package for images. The model list then uses the package, and the original GGUF can be
  moved to a backup disk. The package gives bit-identical logits and the same image answers (checked on the 26B,
  20 Oct; the 35B and Flash Next run from packages too).
- GPU kernels are compiled once and then saved (`%USERPROFILE%\arcinferzero\kernels`): the first request of a model on
  zero, the first image, or the first use after an Intel driver update feels slower, after that it is normal.

**DFlash2 drafter (`--dflash`, arcinferzero only: `serverz.exe` / `runz.exe`).** This drafter guesses 7 tokens at
once from the main model's hidden state, then the main model checks them in a single pass. Download the drafter
([README, section 6](README.md#6-download-models)), then pass its folder (the snapshot folder containing `config.json` and `model.safetensors`)
with `--dflash DIR`. The model's MTP block is still used: at each step the server picks DFlash2 while many of its
drafts are accepted, otherwise MTP. Results (greedy, answers identical to running without a drafter):

A model with a DFlash2 drafter is also listed (Pi's `/model`, `pi-arc --pick`) with one drafter only:
`<id>-nodf` drafts with the MTP block only, `<id>-df` with DFlash2 only (the MTP block is not loaded, which leaves a
little more GPU memory for the expert cache), e.g. `35b`, `35b-nodf`, `35b-df`; picking one in Pi switches the
server like another model does. On the command line: `--draft auto|mtp|dflash` (server and `runz`). Which one to
use: `-df` for code, `-nodf` for Indonesian and other prose (the drafter was trained mostly on English text and
code), the default (both) when the text is mixed. Speeds of each variant:
[docs/before-after.md, "Where it stands"](docs/before-after.md#where-it-stands).

**Gemma 4 runs in its accurate mode.** The attention is computed in float (f32 queries and keys, f16 values); on
the 26B and 12B every matmul input also goes through two int8 passes (the value, then its rounding residual).
E4B is accurate enough with one pass and keeps it (since 2026-10-23: its prompts ~1.5x faster). KL against an f32
reference on a 480-token code prompt: 26B 0.0042, 12B 0.0002, E4B 0.00088 (the limit kept is 0.015; the old fully
int8 mode gave 0.0985 / 0.033 / 0.0010). Speed: 26B with its drafter ~25 tokens/s, 4K prompt ~252 tokens/s; E4B
with its drafter ~32 tokens/s, a 14K prompt in ~23 s. Conversations saved by an older mode are not loaded (they are
processed again).

**The NPU next to the GPU (`-Npu on`).** `pi-arc 27b --npu on` (or `arc-server start -Model 27b -Npu on`) lets
the Intel AI Boost NPU compute part of every prompt chunk while the GPU computes the rest: Qwen3.8 27B prompts
+15-20 %, about 0.4 GB more RAM, generation unchanged. The setting stays with the server: Pi's `/model`, another
client's model, or `pi-arc` on another model keep it (models it does not help, such as the 4B or the 35B, run
without it and say so in the log); `arc-server restart` or a stop turns it off. The first prompt of a new length
class compiles the NPU's graphs once (10-30 s; cached in `%USERPROFILE%\arcinferzero\npu`).
[docs/npu-experiment.md](docs/npu-experiment.md).

**Two conversations at once (`-Slots 2`, arcinferzero).** By default the server holds one conversation and runs one
request at a time: a second Pi window waits until the first one's answer is finished, and switching between them
goes through the session cache. `pi-arc 35b --slots 2` (or `arc-server start -Model 35b -Slots 2`) makes the server
hold two conversations, each with its own attention cache, DeltaNet states and snapshots; start a second Pi in
another terminal with plain `pi-arc` and it shares that server. The GPU still runs one step at a time, but the two
requests take turns step by step (a piece of up to 8,192 prompt tokens, or one generated token / verification
pass), sharing the GPU time evenly, so a short question is answered while the other session reads a long prompt
(35B, a 46K-token prompt in one slot: a question sent 2 s later was answered after 9.6 s instead of after the whole
prompt). Each slot gets the context that fits two of them (`-Ctx auto`: 27B 2 x 49,152 tokens; streamed MoE models
aim at 128K in all and keep at least 32K per slot: 35B 2 x 65,536, its expert cache 76 -> 73 % of the experts, Gemma 4
26B-A4B 2 x 32,768, 72 -> 65 %). When both are writing their answers, their next tokens (with their
drafts) go through the model in one GPU pass, so the weights are read once for both: two answers at once, 27B
7.3 -> 10.5 tokens/s together for prose, 11.6 -> 15.4 for code; 35B 20.2 -> 24.7 and 29.0 -> 30.7; Gemma 4 26B-A4B
19.8 -> 26.2 and 19.6 -> 26.5 (Gemma's drafts are verified one slot at a time). Flash Next batches its slots too since
2026-10-27, without a drafter (with DFlash2 they take turns), but on this laptop two chats get only ~1.05x together:
each conversation reads its own missing experts from the SSD (1.43x when the experts are cached). The answers are the same as with one slot (greedy text identical; the
engine's logits bit-identical alone, interleaved or batched). `ARC_NO_BATCH=1` turns the batching off (turns only).

**Qwen3.8 Flash Next (arcinferzero only).** A MoE model with 512 experts fetched from the SSD (like the 35B). Run
`serverz.exe` with `-m` pointing to the first file (`...-IQ3_S-00001-of-00002.gguf`); the second file must be in the
same folder (or use `pi-arc flash` / `arc-server start -Model flash`). The first start writes a 56 GB expert file
to `%USERPROFILE%\arcinferzero\zero\` (~9.5 minutes); later loads through the server take ~21–26 s. It writes ~6–7
tokens/s (the expert cache holds about a fifth of the experts at 131K); prompts: 166 tokens/s at 4K, ~145-163 at
9-32K, ~123 at 125K. The automatic context is 131,072; up to 262,144 with `--ctx`. The difference from the CPU
reference: KL 0.0143 on 363 tokens of text, 0.0186 on a 3,584-token prompt. `ARC_FN_PRECISE=1` (set before starting
the server) makes it more exact (0.0127 at 3.5K) at about half the prompt speed. This model has no MTP block and
cannot read images; the packaged single file in `%USERPROFILE%\arcinferzero\zero\models` works too.

Stop the server (`arc-server stop`) before using the laptop for other heavy GPU work (games, video editing), or
before moving or rebuilding the arcinferzero folder.

## 4. Web UI and other apps

While the server is running:

- **Web UI**: open http://127.0.0.1:8090/ in a browser to chat directly. It is laid out like WhatsApp Web:
  - **Chats** on the left: every conversation is kept by the server in the data folder, **one folder per chat**:
    `%USERPROFILE%\arcinferzero\chats\<date> <first words> (<id>)\` holds `chat.json` and the chat's files under their
    own names (uploads, voice messages, documents the model wrote). Search the chats; the pencil starts a new one; the
    bin on a chat deletes its folder with its files (a file another chat uses too stays there).
  - **The model** in the chat header: click its name to pick another model or variant; the server loads it with the
    next message (5-40 s). Under the name: the context used, and whether thinking and web search are on.
  - **Attachments** with the paperclip, drag-and-drop or paste: text files, images (Gemma 4 12B / E4B / 26B, Qwen3.5
    4B / 9B, Qwen3.8 27B, Qwen3.6 35B), audio. Each file goes into the chat's folder; messages refer to it, and the model also gets its name,
    type, size and upload time.
  - **@ in the message** lists the files uploaded before (filtered as you type): Enter or a click attaches one again,
    without uploading it again (in another chat's folder it becomes a hard link: the bytes are kept once).
  - **Files** in the left rail: every saved file of every chat with its size, date and the chats that use it;
    filter, sort, search, attach it to the message, or delete files one by one, several at once, or all files no chat
    uses. Files put into a chat's folder by hand show up too.
  - A message's menu (the arrow on its bubble): **Edit** (correct it and send again) or **Delete** (the message with
    its answer, and its files when no other message uses them).
  - The model knows when each message was sent (day, date, time and time zone go with every message).
  - **Answers go on in the background**: open another chat while one answers and it keeps going ("answering…" in the
    list); open it again to watch it or Stop it. **Conversations at once** in the settings sets how many chats answer
    at the same time (the server loads the model again, 5-40 s; each chat then has 1/N of the context, and a chat
    too long for it is summarized before its next answer). With every slot answering, New chat is off and a
    message in another chat waits in its box until one ends (or Stop). A bar over the chat list shows
    what runs and has **Stop all**; an answer with no output from the server for 10 minutes stops by itself.
  - **Settings** behind the gear (bottom left; on a phone in the chats' header): **Thinking** as the model offers it
    (Qwen3.8: off / low / medium / high; the other models: on / off), **Files**, **Terminal**, **Web search**, the **Voice** of Talk mode
    (the Windows voice that reads answers aloud and its rate), **Advanced** (your own temperature instead of the
    model maker's recommended sampling, which the server takes from the model file; a system prompt), the theme,
    **Data** (what is stored; clear the chats, or the chats and their files).
  - There is no answer length limit to set: an answer may use the rest of the context (the reasoning is bounded by
    the server's thinking budget), and **Compact** in the chat header (or automatically when the context is nearly
    full) summarizes the earlier messages.
- **Recordings that end badly.** A meeting recorder (`arc-server record`) that is killed, a shutdown or a power
  cut leaves its WAV in `%USERPROFILE%\arcinferzero\recordings`: the server picks it up at its next start
  and transcribes it. A live recording whose browser tab closed or lost its connection is stopped once the page has sent
  nothing for 5 minutes, and transcribed as well; a recording of another tab can be stopped with its Stop
  button. Files and recordings of up to 6 hours are transcribed.
- **Web UI: other chats and recordings.** With **Files** on, the model can look into your other chats with it
  ("what did we decide in the chat about the budget?": it searches them and reads the exchange it needs) and
  into the recordings and meetings of the Transcripts / Meetings tabs. `@` in the message box lists them next to
  the saved files: picking one attaches it like a document (a long one as its outline, read in parts).
- **Web UI: Transcripts and Meetings** (the left rail's second and third buttons; the server started with
  `--whisper auto`, or `arc-server start -Whisper auto`; [docs/whisper-native.md](docs/whisper-native.md)).
  - **Transcripts**: drop audio or video files (MP3, M4A, WAV, Ogg / Opus voice notes, MP4, MOV, MKV, WebM, AVI,
    WMV) or choose them; they upload and wait their turn, each with its progress. Pick the **language** (detect,
    Indonesian, English, or **Indonesian + English mixed**, about 3x slower), **Accurate** (beam 5) or **Fast**, and
    a **Glossary** of names and terms the speech uses (written inside a casual sentence, the text keeps a casual
    spelling). Open a transcript: it plays with its media and lights the word being said; click a word or a time to
    play from there; double-click a line to correct it; **Find / Replace all** changes a term everywhere; tick the
    formats (SRT, WebVTT, YouTube SBV / SRV3, ASS, TTML, text, Markdown, CSV, Remotion captions / pipeline, JSON) and
    the subtitle shape, then **Export**. Everything is kept in `%USERPROFILE%\arcinferzero\transcripts\`.
  - **Live**: choose the microphone or a tab's / the screen's sound (Chrome asks what to share: tick "Share tab
    audio") and press **Live**: the text comes about 2 s behind, grey while a sentence is still being said;
    **★ Mark** marks a moment. **Stop**: the whole recording is transcribed again accurately, then kept as MP3.
  - **Meetings**: **Start a meeting** records your microphone as "Me" and the meeting's sound (the tab of Meet /
    Zoom web / Teams web, or the screen) as "Others"; use headphones, so the microphone does not hear the others
    too. Afterwards **Summarize** (the loaded chat model writes the summary, decisions, action items and open
    questions with their minute, in the meeting's language) and **Minutes DOCX / PDF** (also in the Files panel).
    A recording made elsewhere can be dropped in the Meetings list too. For a meeting in a desktop app (Zoom,
    Teams) or a long one: `arc-server record -App Zoom.exe -Title "weekly"` records your microphone and that app's
    sound in the background (without `-App`: everything the speakers play; `-NoMic`: the meeting's sound only),
    whatever the browser or the server do meanwhile; `arc-server record-stop` ends it, and it appears in the
    Meetings tab, transcribed as Me / Others.
  - **Audio in a chat**: an attached audio file or a voice message reaches every model as text written by
    Whisper (long recordings as a document the model reads by parts). Settings > Voice > **Speech**: "Whisper
    writes it" (every model), or "The model hears it" (Gemma 4 E4B / 12B get the sound itself; ~0.2 s quicker in
    Talk).
- **Web UI: web search.** Turn on **Web search** in the settings and the model can search the web (`web_search`) and
  read pages (`web_fetch`): each search or page shows as a card in its answer (with the links), and the answer goes
  on with what it found. The server does the requests itself (Brave Search, Bing when Brave gives nothing) and
  turns pages into plain text; it only reaches public http(s) addresses (nothing on this machine or the local
  network). Other clients can use the same endpoints: `POST /v1/tools/web_search` with `{"query": "..."}` and
  `POST /v1/tools/web_fetch` with `{"url": "...", "offset": 0}`. `--no-web-tools` turns them off.
- **Web UI: files and documents.** With **Files** on (the default) the model can list the saved files
  (`list_files`: by kind, name, upload time, this chat only), read them (`read_file`: DOCX, PDF, Excel XLSX / XLSM, PowerPoint
  PPTX (one heading per slide, with its speaker notes), text, Markdown, CSV; "read all the files I uploaded today" works; a spreadsheet reads as one heading per sheet and one
  line per row, "Nama: Budi | Nilai: 80" when it has a header row, dates as yyyy-mm-dd, formulas as their values) and write files (`create_document`: DOCX, PDF, DOC, RTF and XLSX made from Markdown; any
  text format, such as HTML, SVG, CSS, JS, Python, JSON, CSV or TXT, saved as written; a saved HTML / SVG file's card
  opens the page sandboxed; binary formats such as PPTX, PNG or ZIP are refused). A **long document** (more
  than ~6K tokens, e.g. a report of 100+ pages) is not put into the prompt whole, which would take minutes: the model
  gets its outline (chapters with pages) and reads the chapters or pages the question needs, or searches it
  (`search_file`: the passages with the question's words, with their pages), and says where its answer comes from.
  **Scanned PDFs** (pages that are images, without text) are read by the loaded model when it sees images (Qwen 3.5
  4B / 9B, Gemma 4 E4B / 12B / 26B with their mmproj): `read_file` with `pages` has it transcribe those pages from
  their images first, up to 4 a call (10-60 s a page: ~1,000 image tokens, then the text), and the text is kept with
  the file (the chat folder's `.pages\<id>.json`), so later reads, other chats and `search_file` use it without the
  vision model. JPEG, Flate and CCITT fax images are read; JBIG2 and JPEG 2000 scans are not. A model without an
  image encoder says which pages are scans and that a vision model is needed. With one slot the transcription is a
  conversation of its own, so the chat's prompt is processed again at its next turn (`--slots 2` or the session
  cache avoid that). Documents it writes: **DOCX** (opens in Word without
  repair), **PDF**, **DOC** (Word 97-2003 compatible, written as RTF, which Word opens as a document), RTF or Markdown,
  from Markdown with headings, bold / italic / underline / strike / highlight, colors and font sizes, lists, tables
  (column alignment, a header row repeated on each page), code, rules, centered or justified blocks and page breaks;
  page size and orientation, margins, font, line spacing, heading and table colors, header, footer and page numbers.
  **XLSX** (Excel, opens without repair): each Markdown table becomes a sheet named by the heading before it, with
  numbers, Rp amounts, percentages and dates as real values (so sums and filters work; Indonesian 1.234,5 or US
  1,234.5 told apart per table, or set with `number_format` id / us), a bold frozen header row with
  an autofilter; the text around the tables goes to a first "Notes" sheet. The document shows as a file card in the answer (open or download) and is saved in the chat's folder. Ask for a
  change ("the spacing is too tight", "make the title dark red") and the model reads the document's source and writes
  a new version next to it (`name (2).docx`).
  Long results of earlier questions (document parts, web pages, search passages, command output) go back to the
  model as a one-line note of what was read, so a few reads do not fill the context; the model reads a part again
  when it needs it. The last exchange and the answer running now get them whole, and the chat keeps them all.
- **Web UI: terminal.** The **Terminal** button above the chat opens the chat's terminal in a panel on the right: a
  PowerShell that starts in the chat's folder, so it can use the chat's files and the programs installed here
  (arcinferzero's own tools are on its PATH). Type commands there yourself (Enter runs, Up brings back the last one).
  The folder and the environment carry over from one command to the next: `cd`, `$env:X = ...` and an activated
  `.venv` (`.\.venv\Scripts\Activate.ps1`, later `deactivate`) stay; PowerShell variables and functions do not, and no
  program keeps running after its command. The terminal belongs to the chat: it is still there when the chat is opened
  again. **The model may use it too** (`run_command`, shown in the panel as AI), and **opens it itself** when it is
  closed (to make a `.venv` and install a package, say): that first command waits in its card for **Run** / **Don't
  run**, and the panel opens once you allow it. **Settings > Terminal**: **Ask each time** (every command waits for
  Run), **Run without asking** (only the first command of each terminal session asks), or **Model never** (the model
  neither opens nor uses it). Asked only to open, hide or exit the terminal, the model does that alone
  (`terminal_panel`; "close" hides it, the session goes on). The model is told the shell and Windows version in the
  session's first result, so it writes commands for Windows PowerShell 5.1 rather than bash. **Exit** (the power
  button) ends it: a command still running (yours or the model's, also one waiting for Run) is stopped, and the next
  session starts again in the chat's folder, its first command asking again; **×** only hides the panel.
- **Web UI: pages.** A code block of HTML or SVG in an answer has **Open** next to Copy and Download, and a saved
  `.html` / `.svg` file's chip opens the page (Ctrl+click: its source). It opens in a new tab inside a sandbox: its
  scripts run and it may load from `https:` sites (a CDN's chart library, images, fonts), but it cannot reach this
  server (no commands, chats or files), the web UI's storage or its window. A command stops after 2 minutes (the model may ask for up to
  10; yours: 10), together with every program it started; up to 64 KB of its output goes back to the model. Only the web UI
  on this computer can use it: the server refuses it to other computers (also with `--host`; such a server is opened by
  its IP address, http://192.168.x.x:port/, not by a name, against DNS rebinding), to pages of other sites,
  and entirely with `--no-terminal`.
- **Web UI: voice messages and talking.** The microphone in the message box records a voice message (the bin
  discards it, Send sends it). The headset in the chat header starts **Talk**: the microphone stays on, the end of
  each sentence is detected after a second of quiet, and the answer is read aloud while it is written. Speak while
  it talks to interrupt it; press the headset again to stop. With Whisper on the server (`--whisper auto`, or
  `arc-server start -Whisper auto`; [docs/whisper-native.md](docs/whisper-native.md)) your speech is written as text
  first, so both work with every model; without it they need Gemma 4 E4B or 12B, which hear the recording
  themselves. Something you said just before stopping Talk waits in the message box instead of being sent. Pick the reading voice and
  speed in the settings (Windows: Settings > Time & language > Speech adds an Indonesian voice). The audio goes only
  to this server. Headphones avoid the AI hearing itself.
  With **Web** ticked as well, searches run in the background while you talk: the AI says it is looking, and
  talks about the results when they arrive.
- **OpenAI API**: base URL `http://127.0.0.1:8090/v1`, any API key (for example `arcinferzero`), model `qwen3.8-27b`
  (or the model currently loaded; other models in the list are loaded automatically). Supports streaming,
  `reasoning_content`, tool calls, `usage`, repetition penalties (`frequency_penalty`, `repeat_penalty`),
  `logit_bias`, `logprobs`, several answers at once (`n`, also on `/v1/completions`), guaranteed-valid JSON answers
  (`response_format` with `json_schema`), and `tool_choice` (forcing a tool call). Limits: a request body up to
  256 MiB (larger: 413), all requests together up to 1 GiB at once (more: 503, try again); a prompt longer than the
  context gets llama-server's `exceed_context_size_error`, which Pi and other clients use to compact and retry.
- **Other formats** (same models and features):
  - **Ollama**: apps that have an "Ollama" option (e.g. Open WebUI, many editor plugins) just need to point to
    `http://127.0.0.1:8090` (`/api/chat`, `/api/generate`, `/api/tags`, ...).
  - **Anthropic**: `http://127.0.0.1:8090/v1/messages` (Claude format, including tool_use and thinking).
  - **OpenAI Responses**: `/v1/responses` (used e.g. by Codex CLI); the old **Completions**: `/v1/completions`
    (raw prompt, for code autocomplete).
  - **Audio transcription**: `/v1/audio/transcriptions` (upload an audio file, the result is text; needs Gemma 4
    12B / E4B, which have an audio encoder).
- Embeddings (`/v1/embeddings`) are not available yet: they need an embedding model with an architecture the engine
  does not support yet.
- **Web UI: automatic compaction.** The top right shows the context use ("context 12.3K / 40.9K"). If the next
  message would make the context almost full, the older messages are summarized by the model first, then the
  conversation continues with the summary + the last exchanges. The **Compact** button does this at any time;
  **Stop** cancels it.
- **Well-formed tool calls** (Gemma 4 and Qwen): while the model writes a tool call, the server only picks tokens
  that keep the call valid for the tool's schema (required arguments such as `path` cannot be left out, field names
  only those in the schema). The content is still the model's choice. Turn it off with the variable
  `ARC_TOOL_CONSTRAINT=0`. `ARC_LOG_TOOLCALLS=file.jsonl` logs the raw text of every tool call (to investigate
  failed tool calls).
- **Qwen3.6 35B: `pi-arc 35b --preserve-thinking`**: the thinking of earlier turns stays in the prompt (an official
  option of the Qwen3.6 template; Qwen3.8 27B always does this). Follow-up messages do not process anything again;
  the context fills faster.

Example from PowerShell:

```powershell
$body = @{ model = "qwen3.8-27b"; reasoning_effort = "none"
           messages = @(@{ role = "user"; content = "Hi! Explain arcinferzero in one sentence." }) } | ConvertTo-Json -Depth 4
(Invoke-RestMethod http://127.0.0.1:8090/v1/chat/completions -Method Post -ContentType "application/json" -Body $body).choices[0].message.content
```

From Python (the `openai` package):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="arcinferzero")
r = client.chat.completions.create(model="qwen3.8-27b", messages=[{"role": "user", "content": "Hi!"}],
                                   extra_body={"reasoning_effort": "none"})
print(r.choices[0].message.content)
```

## 5. Context size, and when a conversation gets long

Each model uses the largest context that fits the GPU memory currently available (set in Intel Graphics Software,
a 20.4 GiB GPU share on this laptop now; the amount the driver offers changes with what other apps run), minus 1.2 GiB
for the driver, or the model's training limit. The context adjusts itself when that setting is changed (run
`arc-server restart` afterwards). The launcher window shows the current numbers. Examples:

| Model | Default context |
|---|---|
| Qwen3.8 27B | 65,536 with images + MTP + DFlash2 (`27b`); 90,112 `27b-df`; 110,592 `27b-mtp` / `27b-nodf`; 143,360 `27b-vision`; 188,416 `27b-std` |
| Gemma 4 26B-A4B | 90,112 with images + drafter (`g26`); 118,784 `g26-mtp`; 102,400 `g26-vision`; 131,072 `g26-std` (generation at 36K of depth: 11.4 tok/s with a 49K context, 10.5 with 131K) |
| Qwen3.6 35B-A3B | 131,072 automatically (images + MTP + DFlash2, ~80 % of the experts cached); up to 262,144 with `--ctx` (smaller expert cache) |
| Qwen3.8 Flash Next | 131,072 automatically; up to 262,144 with `--ctx` |
| Qwen3.5 9B / 4B | 262,144 tokens (full) |
| Gemma 4 12B | 262,144 tokens (full) |
| Gemma 4 E4B | 131,072 tokens (full) |

arcinferzero uses the whole GPU share you set except 1.2 GiB for the driver, so the share decides how much RAM is left
for Windows: if the laptop becomes slow or stops responding while a model runs, lower the GPU memory setting.

The output limit (`maxTokens` in Pi) equals the context size: Pi itself clamps it to the room left. Very long
prompts still take time (see section 6), so a large context is mainly useful for long sessions that use the prefix
cache.

Pi handles a full context, not the server. When the conversation passes `context size − 16384` tokens, Pi
summarizes the older part and keeps the latest ~20,000 tokens. If a single request still exceeds the context, the
server sends the same error as llama.cpp ("the request exceeds the available context size"). Pi recognizes it,
then summarizes and retries automatically. You can also summarize at any time with the `/compact` command in Pi.

This summary takes time (a few minutes on the 27B), because the model has to read the part being summarized and
then process the new conversation again. The server cannot simply drop old tokens: the DeltaNet layers of
Qwen3.5/3.8 keep a state that cannot "forget" individual tokens.

## 6. How fast

Current numbers, measured on this laptop with arcinferzero (every model, how it was measured, and llama.cpp for
comparison: [docs/before-after.md, "Where it stands"](docs/before-after.md#where-it-stands)):

- Qwen3.6 35B-A3B (UD-IQ4_XS): reads a 32K prompt at 604 tokens/s, 128K at 324; writes 20.6 tokens/s, Python code
  ~50 with DFlash2, Indonesian ~22 with MTP; 12.6-13.0 tokens/s deep in a 128K context.
- Qwen3.8 27B: reads ~135 tokens/s; writes 4.9 tokens/s, code 18.1 with DFlash2 + MTP, Indonesian ~7 with MTP.
- Qwen3.8 Flash Next: reads ~164 tokens/s (4K) to 123 (125K); writes ~6-7 tokens/s.
- Gemma 4: 26B reads ~252 tokens/s (4K), writes ~25 with its drafter; 12B 137 / 19; E4B ~724 / 32.
- Images: a 1,064-token image takes ~7.4 s on Gemma 4 26B, ~3 s on E4B and the Qwen models.
- Very long prompts are read more slowly (attention gets heavier), and drafting helps code more than prose.

Tips:

- The longest part is usually the thinking phase. `--think off` or `--think minimal` speeds up simple answers.
- Follow-up requests in the same session only process the new tokens (prefix cache), so they are much faster than
  the first request.
- A new session in a project used before is fast: Pi's system prompt comes from the cache (memory or disk), so only
  your message is processed. The first session in a new project folder processes its system prompt once (27B:
  ~40 s).
- Going back to an old session after another session (or the web UI) loads its conversation from disk in seconds:
  the session cache is on by default (8 GB in `%USERPROFILE%\arcinferzero\zero\sessions`, delete with
  `arc-server clear-cache`, turn off with `pi-arc --session-cache 0`). `arc-server stop` / `restart` and a model
  switch first save the conversation the server holds, so afterwards that conversation is loaded too, not processed
  again.
- A new message in the same session (Qwen 4B / 9B / 35B) only processes the last turn again: the Qwen template drops
  the thinking of earlier turns as soon as you write again, and the server keeps the starting point of that turn
  (arcinferzero).
- Stop the server with `arc-server stop` (or the Stop button in the launcher), not by killing its process: only
  that way is the conversation being held saved first.
- A hot laptop slows down by 5–10 %. Do not run other GPU work at the same time.

## 7. Moving the arcinferzero folder

1. `arc-server stop`
2. Move the `arcinferzero` folder to its new location (File Explorer or `mv`).
3. Run setup again from the new location: `<new-folder>\setup.cmd -Models none`

Only the `pi-arc` / `arc-server` shortcut commands store the folder location; everything else uses relative paths.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `pi-arc` is not recognized | Run `setup.cmd` (section 1), then open a new terminal. |
| Setup fails at the build step | Visual Studio Build Tools (C++) is not installed; setup shows how to install it. The compiler output is in `%USERPROFILE%\arcinferzero\build.log`. |
| "The server stopped while loading" | Look at the log it shows. Usually there is not enough GPU memory: close heavy apps, or use `--ctx 32768` / a smaller model. |
| Windows is slow or stops responding while a model loads | The GPU memory is set too high (section 5): lower it in Intel Graphics Software, then `arc-server restart`. |
| `web_search` in Pi answers "No results found" or never finishes | Its default search engines are DuckDuckGo and Startpage; DuckDuckGo is blocked in Indonesia, so neither returns results. Add Yandex in `%USERPROFILE%\.pi\agent\settings.json`: `"webSearch": {"engines": ["yandex", "startpage", "duckduckgo"]}` (or with the `/search-config` command in Pi). |
| A new Pi session takes ~40 s before answering | Normal for the first session in a new project folder (the system prompt is processed once); later sessions in that folder are fast. |
| "Model file ... not found" | The model has not been downloaded to the Hugging Face cache yet (see [README, section 6](README.md#6-download-models)). |
| Models moved to another drive | Write the folders (one per line) in `%USERPROFILE%\arcinferzero\model-dirs.txt`, e.g. `D:\models` (a Hugging Face cache layout `models--...`, or `.gguf` files directly in that folder). The Hugging Face cache on C: is still searched first. A plain folder of `.gguf` files (a copied model) is found one level deep in the Hugging Face cache and in those folders too. A model that is in several places (the same file name and size, or the GGUF an arcinferzero package was made from) is listed once: the package, else the Hugging Face cache, else the C: copy. If that drive is removed, its models disappear from the list until the drive is connected again. |
| Pi stops with a context error | Type `/compact` in Pi, or start a new session. |
| The answer is very slow at the start | Normal for the first large prompt (see section 6); later ones use the cache. |
| Want to see what is happening | `arc-server status`, and the log in `%USERPROFILE%\arcinferzero\server.log` (prompt processing time, speed, MTP statistics per request). |

## 9. Rebuilding from source

Needs Visual Studio 2026 Build Tools (MSVC). From the arcinferzero folder, `.\setup.cmd -Models none` rebuilds only what
is needed (and stops the server first when `serverz.exe` is rebuilt too). Or by hand:

```powershell
arc-server stop
.\build.ps1 zero\serverz.cpp zero\runz.cpp
```

GPU kernels (`zero\kernels\*.cl`) are compiled by the driver
when the server starts, so kernel changes do not need a build.
Compiled kernels are saved in `%USERPROFILE%\arcinferzero\kernels`, so only the first use of each variant waits for
compilation.
Regression tests: `.\tests\run_tests.ps1` finds the models itself (`-ListModels` shows which), or give them:
`-Model4B`, `-Model27B`, `-Model35B`, `-ModelGemma` (12B), `-ModelFlashNext` (the packaged file first, else the
first part).

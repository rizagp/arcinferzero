# Running the server by hand, Pi and the CLI

(Moved from the README on 2026-10-31.)

## Run the server by hand

```powershell
.\bin\serverz.exe -m C:\path\to\model.gguf --port 8080 --ctx auto --mtp --alias qwen3.8-27b
.\bin\serverz.exe --help           # every option with its default
.\bin\serverz.exe --list-models    # the models it can load (Hugging Face cache, %USERPROFILE%\.inferzero\models, model-dirs.txt), as JSON
```

- **Speech to text only**: `serverz -m none` (`zero-server start -Model none`, the launcher's More >
  **Start speech to text only**) loads Qwen3-ASR alone (~3.2 GB of GPU memory; `--whisper auto` adds Whisper) for the Transcripts tab and
  `POST /v1/audio/transcriptions`. A chat model loads later, when a chat asks for one.
- Web UI: http://127.0.0.1:8080/: chats with images, audio and files (one folder per chat, kept by the server),
  voice messages and **Talk**, web search, documents the model reads and writes, a terminal, and with `--whisper` the
  **Transcripts** and **Meetings** tabs: [USAGE.md, section 4](../USAGE.md#4-web-ui-and-other-apps).
- OpenAI API: `POST /v1/chat/completions` (streaming, `reasoning_content`, tool calls streamed while they are
  generated, string arguments and array / object arguments alike (Pi's `edit` puts its whole change in an array),
  `usage`; the last streamed chunk carries `usage` and llama-server's `timings` together),
  `GET /v1/models` (every model of `--list-models`, the loaded one first; the served context as `context_length`,
  llama-server's `meta.n_ctx` and `max_model_len`), plus `GET /health`, `GET /props`, `POST /tokenize`, `GET` /
  `DELETE /sessions` (session cache size / clear).
- Web tools (the web UI's, usable by any client): `POST /v1/tools/web_search` `{"query": ...}` (Brave Search, Bing
  when Brave gives nothing) and `POST /v1/tools/web_fetch` `{"url": ..., "offset": 0, "links": false, "chat_id": ...}`
  (in windows of 12K characters, at most what the model's context takes in one read); public http(s) addresses only
  (every host and redirect is resolved first: nothing on this machine or the local network). `--no-web-tools` turns
  them off. What a URL gives:
  - a page: plain text with headings and lists; links to files and to GitHub issues / pull requests / commits (or
    written as a URL or `#123`) keep their URLs as `[text](url)`, the other links their text only, with a note of how
    many; `"links": true` gives every link's URL, to explore a site;
  - a text file (`.txt`, `.log`, `.csv`, JSON...): as it is, its name as the title;
  - a GitHub issue or pull request (`github.com/<owner>/<repo>/issues/<n>` or `/pull/<n>`): read from GitHub's API
    (no token: 60 requests an hour, 2-4 a thread) as Markdown, the description and every comment with its author and
    time, the files attached in the thread listed at the top, a pull request's review comments; the page when the API
    fails;
  - a PDF, Word, Excel or PowerPoint file (up to 20 MB): saved into the chat's files (`chat_id`; none: the uploads
    folder) and its text returned, as `read_file` reads it; `file_id`, `filename`, `kind` in the reply;
  - an image (PNG, JPEG, GIF, WebP, BMP, up to 20 MB): saved the same way; the web UI shows it to a model that sees
    images after that round of tools;
  - an audio file (MP3, M4A, AAC, WAV, FLAC, Ogg Opus, WMA, WebM, up to 60 MB): saved, and its first 30 minutes of
    speech as text by the speech service (Qwen3-ASR, else Whisper; the chat clips' cache); without the service, saved
    only. Ogg Vorbis is not decoded;
  - anything else (archives, programs, video): refused (415), without reading its body. A file's first bytes must be
    of its kind.

  The size each kind may have is decided from the headers before the body is read (a page or text 4 MB, a document or
  image 20 MB, audio 60 MB). A file is whole or nothing: one that announces more than its size (Content-Length) is not
  read, and one larger or cut is refused (413), one that stopped early (a stall, a dropped connection, the 2-minute
  deadline) too (504). A page or text is read up to its 4 MB, and the text says when it was cut. Any port of a public host, except those browsers refuse because other services listen
  there (SSH 22, SMTP 25, DNS 53, SMB 139 / 445, LDAP, IRC...: Chrome's restricted ports), on every redirect too. At
  most 4 fetches or views run at once; the others wait up to 2 minutes, then 429.
- `POST /v1/tools/web_view` `{"url": ..., "screen": 0, "links": false, "chat_id": ...}`: the page as a browser shows it
  (tools/web_view.h). Microsoft Edge runs headless (on every Windows 11; not found: 502): a screenshot of one screen
  (1280 x 900; `screen` 1-5: further down), saved in the chat's files like an image from web_fetch, and the page's text
  after its scripts ran (pages built by scripts read too), in the same windows and with the same links as web_fetch.
  5-15 s a page, one at a time. Edge's own requests go through a proxy inside the server (127.0.0.1, a free port) that
  connects to public addresses on web ports only (as web_fetch), checked on the address it connects to: a page cannot
  make Edge load anything on this machine or the local network, also through a redirect or a frame. The proxy serves
  only Edge's processes of a render running now (the process at the other end of each connection must be in that
  run's job), so no other program can use it. Each run has a new empty profile in %TEMP% (no cookies, logins or
  history; removed after, and ones left behind swept at every render once 30 minutes old), no extensions or sync,
  WebRTC's direct UDP off. Edge's processes are in a job: ended after 45 s at most, 3 GB of memory and 32 processes
  together, inheriting no handle of the server's but their output pipe; the server refuses to run Edge outside such a
  job. The URL reaches Edge's command line percent-encoded (no quote, space or control byte can leave its argument);
  the page's HTML is read from its first 4 MB.
- The web UI's memory and projects (`tools/server_memory.h`, `tools/server_projects.h`). They are served while a
  model loads, and only on this computer (not over the network), to the server's own page or a client that sends no
  Origin:
  - `GET` / `PUT /v1/memory[?project=<id>]` reads or sets the notes as one text (a project in the list);
  - `POST /v1/tools/remember` `{"fact", "project"?}` and `POST /v1/tools/forget` `{"match", "project"?}` add or
    remove a note; forget removes at most 3 notes a call, and in a project it falls back to the global notes;
  - `GET` / `POST /v1/projects` `{"folder", "name"?}` lists projects or adds one;
  - `GET` / `DELETE /v1/projects/<id>` reads a project with its AGENTS.md and memory, or removes it from the list;
  - `GET /v1/projects/browse?path=` returns the folder picker's listing;
  - the file tools take `project:<id>/<path>` together with `"project_id": "<id>"` (the chat's project: another
    project's files are refused), and `list_files` takes `{"scope": "project", "project_id"}`;
  - `run_command` takes `project_id`, so a project chat's terminal starts in its folder.
- Every reply carries `X-Content-Type-Options: nosniff` (a JSON or text reply is never run as a script or page).
  The web UI page also sends `X-Frame-Options: DENY` and `Content-Security-Policy: frame-ancestors 'none';
  object-src 'none'; base-uri 'none'`: no other site can show it in a frame and trick a click on it (delete chats,
  approve a command). OCR saves (`ocr_save`, into the chat's folder too) are refused with 403 from a page another
  local program serves; a client that sends no Origin, the server's own page and a remote session's page may save.
- Llama.cpp clients such as Unsloth Studio connect as to a llama-server: they read the context size and the token
  statistics from the fields above.
- Sampling and output (chat): `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`,
  `repeat_penalty` / `repeat_last_n`, `logit_bias`, `seed` (0 is a seed too), `stop`, `max_tokens`; `logprobs` /
  `top_logprobs`; `top_k` 0 or less: no top-k (not sent: the model's own default);
  `n` > 1 (without streaming; `/v1/completions` takes `n` too); `response_format` `json_object` / `json_schema` (sampling is constrained, so the answer
  always parses and fits the schema: types, properties / required / additionalProperties, items, enum, const,
  anyOf / oneOf, `$ref`); `tool_choice` `none` / `auto` / `required` / a named function (the answer starts with a
  call).
- Other API formats, on the same models and features (`tools/api_compat.h`):
  - OpenAI `POST /v1/completions` (a raw prompt, no chat template: code completion) and `POST /v1/responses`
    (Responses API: input items, function calls and their outputs, reasoning, streamed events).
  - Anthropic `POST /v1/messages` (system, image and tool_use / tool_result blocks, thinking, streamed events) and
    `POST /v1/messages/count_tokens`.
  - Ollama `POST /api/chat`, `POST /api/generate` (NDJSON streams, `images`, `tools`, `think`, `format`, `options`),
    `GET /api/tags`, `POST /api/show`, `GET /api/ps`, `GET /api/version`: apps with an Ollama setting connect to
    `http://127.0.0.1:8090`.
  - OpenAI audio: `POST /v1/audio/transcriptions` and `/v1/audio/translations` (multipart upload) with
    `--whisper auto`: Qwen3-ASR 1.7B by default, `model=gemma4-e4b` or `model=whisper` (Whisper large-v3, words
    with times, translations; its encoder on the NPU with `--whisper-device npu`); Gemma 4 12B is refused; json,
    text, srt, vtt, verbose_json, streaming, and subtitle / Remotion formats. Without `--whisper`: a model with an
    audio encoder (Gemma 4 E4B). Saved transcripts: `/v1/transcripts`; live recording: `/v1/live`. Which model
    answers what: [docs/qwen3-asr.md](qwen3-asr.md#using-it); the endpoints in full:
    [docs/whisper-native.md](whisper-native.md#server-api).
- Model switching: a request whose `model` is another model of `--list-models` (its id, `org/repo` name or file)
  makes the server unload the current model and load that one (context auto, same options); `--no-swap` keeps the
  loaded model.
- Thinking control: `chat_template_kwargs.enable_thinking`, `enable_thinking`, or `reasoning_effort`
  (`none`/`low`/`medium`/`high`); a reasoning budget (`thinking_token_budget`, `thinking_budget_tokens` or
  `reasoning_budget`, default `--think-budget`) closes the `<think>` block after that many tokens.
- Prefix reuse: consecutive requests that share a prefix (agent loops) only process the new tokens.
- Cancelling: when the client closes the connection (Pi: Esc), the server stops within a layer during prompt
  processing (27B: ~1 s) and before the next token during generation, so the next request starts right away. The
  prompt blocks that were complete stay processed. A reply the client drops after an abort or error costs nothing
  on the next request: the resume point sits at the start of the reply's turn header.
- System-prompt cache: the state after a client's system prompt (Pi: ~4K tokens; it contains the project folder
  and its `AGENTS.md`, so each project has its own) is saved to `%USERPROFILE%\.inferzero\zero\prefix\<model>` the first
  time it is processed, written in the background, and loaded when a request starts with it while the server holds
  something else: after a server start, a model switch or another project's conversation. Pi's are kept per project
  folder (read from the `<cwd>` section of its system prompt): each folder keeps its 2 latest versions, which other
  folders do not replace (27B: ~290 MB each); other clients' prompts: the 8 used last. All models together
  stay under `--prefix-cache-gb` (default 8; least recently used go first); `--no-prefix-cache` turns it off.
  Files: `<folder>-<folder hash>-<prompt hash>-<tokens>.arcp`.
- The NPU next to the GPU (`--npu on|off|force`, default off): the Intel AI Boost NPU computes part of each prompt
  chunk's matrix products while the GPU computes the rest, through Level Zero alone (Qwen3.8 27B and Gemma 4 12B;
  generation unchanged). Where it pays off, how it is turned on and off, and the results:
  [docs/npu-experiment.md](npu-experiment.md).
- Conversation slots (`--slots N`, inferzero): the server holds N conversations, each with its own KV cache and
  states; requests in different slots take turns step by step (by GPU time), so two clients answer at the same
  time. When several are generating, their next tokens and drafts run in one engine pass (Qwen dense and MoE, Gemma
  4, and Flash Next without a drafter; logits bit-identical to each slot alone; `ARC_NO_BATCH=1`: turns only). Each
  slot gets the context that fits N of them (streamed MoE models: 128K in all, at least 32K per slot). A step that
  fails ends its requests and starts their conversations over. Use and speeds:
  [USAGE.md, section 3](../USAGE.md#3-managing-the-server).
- Session cache: when a request belongs to another conversation (a new Pi session, or switching back to an older
  one), the server saves the conversation it holds to disk (KV cache, DeltaNet states; `%USERPROFILE%\.inferzero\zero\sessions`)
  and loads a saved one that matches the request, instead of processing its whole prompt again. New conversations
  resume after the system prompt they share (Qwen: a recurrent-state snapshot there, kept in host memory). Saving
  and loading take about a second; outputs are identical to never switching. `--session-cache-gb N` (the server's default 0 = off; zero-server / zero-pi and the launcher start it with 8 GB,
  `zero-pi --session-cache 0` = off). The size holds for all models together: when a save goes over it, other models'
  saved conversations go first (oldest first), then this model's least recently used ones; `zero-server clear-cache` deletes them all. `zero-server stop` /
  `restart` and a model switch save the conversation the server holds first (`POST /sessions/save`), so it is
  loaded afterwards instead of processed again; only a server that is killed loses it.
- Qwen turn snapshot (inferzero): the Qwen templates drop the reasoning of earlier turns as soon as the user
  writes again, so the server also snapshots the start of the last assistant turn and a new message only processes
  that turn again, not the whole conversation. `--preserve-thinking` (Qwen3.6; Qwen3.8 always does) keeps earlier
  reasoning in the prompt instead: nothing is processed again, the context fills faster.
- `--mtp`: speculative decoding. Qwen3.5/3.6/3.8: the model's own MTP (multi-token prediction) block in the GGUF
  (`nextn` tensors). Gemma 4: the assistant drafter `mtp-*.gguf` next to the model (found automatically), which
  reads the main model's KV cache. The drafts are verified in one main-model pass and the next token is still
  sampled from the model's distribution, so the output is unchanged (checked: identical answers at every draft
  count). `--mtp-drafts N`: inferzero 1–7 (default 7 for the 27B and whenever DFlash2 drafts, 5 for the 35B,
  3 for the 4B and Gemma 4 26B, 2 for the 9B and the other Gemma models).
- `--dflash DIR` (inferzero: `serverz`, `runz`): speculative decoding with a DFlash2 block drafter, `DIR` being
  the folder with its `config.json` and `model.safetensors` (the snapshot folder of `hf download
  z-lab/Qwen3.8-27B-DFlash2` for the 27B, `hf download incoai/Qwen3.6-35B-A3B-DFlash2` for the 35B). It drafts a
  block of 7 tokens per pass from the main model's hidden states. The model's MTP block stays loaded and each step
  picks the drafter: DFlash2 while its drafts are accepted, else MTP. Greedy answers are identical to plain decoding.
  A model with a DFlash2 drafter is also listed with one drafter only: `<id>-nodf` drafts with the MTP block only,
  `<id>-df` with DFlash2 only (the MTP block is not loaded); `--draft auto|mtp|dflash` (server, `runz`) does the
  same, default `auto` = both. Which variant for which text: [USAGE.md, section 3](../USAGE.md#3-managing-the-server);
  speeds of each variant: [docs/before-after.md](before-after.md#where-it-stands).
- Images (`image_url` content parts, data URLs) and audio (`input_audio`, `/v1/audio/transcriptions`) when the
  model's `mmproj-*.gguf` is next to it (or given with `--mmproj`): images on Gemma 4 12B / E4B / 26B, Qwen3.5 4B /
  9B, Qwen3.8 27B and Qwen3.6 35B, audio on Gemma 4 12B / E4B; the encoders run on the GPU.
- Constrained tool calls (Gemma 4 and Qwen): while the model writes a tool call, only tokens that keep the call
  valid for the tool's JSON schema are allowed (required arguments such as `path` cannot be left out, only known
  keys); the content stays the model's choice. `ARC_TOOL_CONSTRAINT=0` turns it off; `ARC_LOG_TOOLCALLS=file.jsonl`
  logs the raw text of every tool call.
- During a long prompt, streaming responses carry `: prefill N/M` progress comments (also shown in the web UI).
- Qwen3.8 Flash Next (inferzero only): pass the first part (`...-IQ3_S-00001-of-00002.gguf`) with `-m`; part 2
  must sit next to it (or use the packaged single file); context up to 262,144 with `--ctx`; `ARC_FN_PRECISE=1`
  (opt-in) is the more exact, slower mode. Its speed and KL: the tables above and "Supported models"; running it:
  [USAGE.md, section 3](../USAGE.md#3-managing-the-server); where its time goes and what was tried: QUEUE F21.13 /
  F21.15.

## Using Pi

`zero-pi` (step 9, or `setup.cmd`) starts the server in the background if it is not running, then Pi; the server
stays loaded between Pi sessions. Choosing the model, the context, the reasoning levels, what each new or resumed
session processes, Esc and compaction: [USAGE.md, section 2](../USAGE.md#2-using-pi) and
[section 5](../USAGE.md#5-context-size-and-when-a-conversation-gets-long).

Manual configuration, if you prefer: add a provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "inferzero": {
      "baseUrl": "http://127.0.0.1:8090/v1",
      "api": "openai-completions",
      "apiKey": "inferzero",
      "models": [
        { "id": "qwen3.8-27b", "reasoning": true, "input": ["text"], "contextWindow": 86016, "maxTokens": 86016,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "compat": { "thinkingFormat": "qwen", "supportsReasoningEffort": true,
                      "thinkingTokenBudgetField": "thinking_token_budget", "maxTokensField": "max_tokens" } }
      ]
    }
  }
}
```

then `pi --model inferzero/qwen3.8-27b --thinking low` (Pi sends the reasoning effort and budget; without an
effort the Qwen3.8 template thinks at `xhigh`, which can take very long on a laptop).

## CLI

`bin\runz.exe` runs the engine without the server; `--bench-pp` also prints the time of each generation step.

```powershell
.\bin\runz.exe model.gguf --bench-pp 2048 --n 64                       # prompt-processing and generation speed
.\bin\runz.exe model.gguf --tokens-file ids.txt --n 64 --show --prof   # time to first token + kernel profile
.\bin\runz.exe model.gguf --text-file prompt.txt --n 128 --prof-gen    # generation speed + per-kernel profile
.\bin\runz.exe model.gguf --perplexity corpus.txt --ppl-chunks 40      # perplexity (llama-perplexity method)
.\bin\runz.exe model.gguf --tokens-file ids.txt --kl-ref ref.bin       # KL vs an f32 reference (tools\ref_forward)
.\bin\runz.exe model.gguf --tokens-file ids.txt --kl-ref ref.bin --kl-decode   # the same, token by token
```

`--mtp` turns on speculative decoding, `--ctx N` sets the context; with `ARC_PROF_LAYERS=8` the profile is split
into groups of 8 layers.

Several GPUs and GPU tuning:

```powershell
.\bin\runz.exe model.gguf --bench-pp 512 --n 64 --gpus 2     # a Qwen model (dense or 35B-A3B) split over 2 GPUs (2, 3, 4, ...)
.\bin\serverz.exe -m model.gguf --gpus 4 --slots 4           # the server on a split model
.\bin\mgpu_probe.exe                                         # the GPUs, and what moving data between them costs
.\bin\event_probe.exe                                        # whether GPUs can sync without the CPU in the middle
.\bench\gemm_sweep.ps1 -Models model.gguf                     # the prompt GEMM's settings swept on this GPU
.\bench\decode_try.ps1 -Models model.gguf                     # generation: plain, MTP, DFlash2, GEMV switches
```

What to run on an Arc B-series card (B580 / B60 / B70): [docs/xe2-tuning.md](xe2-tuning.md).

`--gpus N` (runz and serverz) splits the model over N GPUs, and each reads 1/N of the weights per token:
- the heads and FFN columns are divided, and the 35B-A3B's experts too;
- `--slots` gives batched decoding over the split.

Not yet on a split model: MTP / DFlash2, images, and the prompt and session caches on disk. A model that cannot be
split loads on one GPU. A GPU that hangs makes `/health` answer 503 until a restart. Details and failsafes:
[docs/multi-gpu.md](multi-gpu.md). `ARC_TP_SIM=1` runs the split on one GPU (a correctness test).

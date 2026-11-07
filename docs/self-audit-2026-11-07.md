# Self audit, 2026-11-07

Everything changed since the audit of 2026-11-04 (commit 636dbe8 to 515d02c): remote access, speech models offered
from Settings, Transcribe again, the web UI, H8, XMX on iGPUs. Three independent static reviews, one each for the
server, the web UI with speech, and the engine with its kernels. Each finding below was checked in the code before it
was changed.

## Fixed

### Server and remote access

| Severity | Where | What went wrong | Now |
|---|---|---|---|
| High with `--host` | `tools/server.cpp` `/v1/remote/config` | It checked only that the request was not remote. A peer on the network could set the remote password and turn on full access, then sign in through a tunnel and run commands. | From this computer only (`local_peer`). |
| Medium | `tools/server_remote.h` `login` | A sign-in being checked (PBKDF2 takes a while) could add its session after a new password, "sign everyone out" or full access turned on. | A generation counter: such a sign-in gets 401. |
| Medium | `head_gate` | With access off or a host not listed, a body through a tunnel was read, up to 256 MiB, before the 403. | Refused before it is read. |
| Medium | `tools/server.cpp`, `tools/server_speech.h` | Every server has the speech service since 9c18778. With nothing of it loaded, a chat with audio and `/v1/audio/*` got "speech to text is unloaded" instead of reaching a model that hears. | As without the service, unless the request names a speech model. |
| High | `tools/server_whisper.h` `set_loaded` | The first load of an offered speech model skipped the memory check, and a failed GPU allocation ended the server. | The room check uses the model's largest file until a load has given its real size, and a failed allocation is an error the web UI can act on. |
| Low | `tools/server_remote.h` `blocked` | `/sessions` was open to a remote session without full access: it could wipe the conversation cache. | Fenced. |
| Low | `tools/server_remote.h` `gate` | A password could be sent over plain http on a named tunnel with "Always Use HTTPS" off. | Pages move to https, anything else is refused. |
| Low | `tools/server_remote.h` `config` | Settings changed in memory when `remote.json` could not be written, and came back different after a restart. | Nothing changes then, except turning access off. |
| Low | `src/http.h` | A reply to HEAD carried a body, left on a connection the tunnel reuses. | Headers only. |
| Low | `tools/server_file_tools.h` | `/files/` could be stored by a shared cache. The sandbox header never applied, as `mime_of` gives these types `text/plain`. | `Cache-Control: private`; the sandbox goes by the extension. |
| Low | `tools/server_ocr.h` | Full access could not save OCR text to a folder: it compared with `http://`. | `https://` for a remote page. |
| Low | `login` | Clearing the visitor table during a flood could leave a negative in-flight count. | Entries being checked stay. |

### Speech and web UI

| Severity | Where | What went wrong | Now |
|---|---|---|---|
| Medium | `webui/index.html` `trSavePrefs` | A `//` comment in mid-line hid `model`, `beam`, `keep`, `ch` and `terms`: since 636dbe8 they were not saved. | Saved. |
| Medium | `trOpen`, `trDirty`, `trEdit`, Replace | An edit open during Transcribe again could write the old text over the new pass. | No edit while the model loads; an edit still open when the pass ends is dropped, and the page says so. |
| Medium | `tools/server_whisper.h` | A pass again that wrote no text replaced the old transcript with an empty one. | The earlier text stays, and the record says why. |
| Medium | redo handler | The warning about edits missed those of this session, and a save could race the pass. | Edits are saved first and counted. |
| Medium | `trCaps`, recording | A model marked "(unloaded)" stayed selected, and Record failed. | A loaded model is preferred until the user picks one; Record loads the chosen one first. |
| Low | `tools/server_whisper.h` | A failed write of the new text marked a good record failed. A stopped pass of a two-track recording showed double the duration. Stop did nothing for a pass queued during the recording's encode. | All three fixed. |
| Low | `trState` | A failed pass again showed nothing on a failed or stopped record. | Shown. |
| Low | `trCaps` | The Whisper option could carry Qwen3-ASR's name. | Named Whisper. |
| Low | summary | An empty summary hid the summary and its earlier versions. | Refused by the page and the server. |
| Low | tool loop guard | A cycle A, B, A, B was never stopped, and three polls of a job whose output changes were stopped. | The same calls with the same results before them, a third time in the answer. |
| Low | messages | They said to start the server with `--asr auto` or `--whisper auto`. | They point to Settings. |

### Engine

The H8 rotation, its scales, buffer sizes and the float copy held up. Four low findings:

- The XMX speed check broke the `ARC_XMX_TRY` test modes. They skip it now, and keep testing 128 registers.
- A slower XMX was reported as a failed 2D load, and the check ran twice. It no longer retries.
- `aq_` did not count the shared expert's row and the MTP head's `2 * n_embd`. Both are in `aq_max_k()`, used by the three places that sized it. No model tested here changes in logits.
- `ARC_XTRACE` on a split model interleaved its shards in one file. Each shard has its own now.

## Checked

- **Remote access:** the 43 checks from when it was built, and new ones on a test server with a simulated tunnel: bodies
  refused before reading, `/sessions` fenced, plain http moved or refused, HEAD on a connection, sign-in against "sign
  everyone out", the audio routes with speech unloaded.
- **Failed speech load:** `ARC_SPEECH_OOM=1` (a new test switch), Qwen3-ASR and Whisper: 409 with the memory message, the
  server up. Whisper leaves about 0.1 GiB allocated after such a failure.
- **Load and unload** of both offered models.
- **Web UI in the browser:** no console errors, the model and glossary saved, Whisper's label right.
- **Logits:** the default path bit-identical on the 4B and 27B; H8's KL against f32 unchanged (27B 0.00113).
- **`tests\run_tests.ps1`:** 53 passed, 5 skipped (Gemma 12B / 26B not here; optional checks).

Not run:

- The `--host` fix against a real peer on the network: read through only.
- The Transcribe again changes through a real pass: read through only.
- A real tunnel: simulated by its headers, as the rule is not to open one from here.
- The real memory check against a full GPU. The test switch fails an allocation, it does not fill the budget.

## Not changed (noted)

- **Connections:** 64 in all, shared with local clients. Through a tunnel, about 65 parallel requests without a password
  could hold every slot while cloudflared keeps them idle. A lower `--proxy-keepalive-connections` in
  `scripts/remote-tunnel.ps1` or a separate cap would help; not tried without a tunnel to test on.
- **Sign-in lockout:** 40 wrong passwords lock every sign-in for 15 minutes, so anyone who knows the address can keep it
  locked. Sessions already signed in keep working, and a restart clears it. By design for now.
- **HSTS:** not sent. Plain http is moved to https instead.
- **Summaries and minutes:** no token limit (the user's choice) and no Stop. A model that repeats itself writes until the
  context is full.
- **An upload** with an unloaded model chosen can unload the chat model without asking, as Transcribe again does.
- **The XMX 2D prefetch** asks for one block past the surface on a row's last step. A B60 ran it correctly.
- **Not reviewed:** how the web UI renders stored chats. A remote session can write chat JSON that the local page later
  opens, so a flaw there would lead from remote to local powers. Worth its own review.

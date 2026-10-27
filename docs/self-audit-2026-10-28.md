# Self audit, 2026-10-28: Whisper, the transcript library, live recording, arc-recorder, the web UI

Scope: everything written for Whisper W0-W9 and queue items 8 / 8b (`git diff b66ff0f~1`, about 4,900 lines in 24
files). Method, as the DeepSeek audits: four independent static reviews, each over one area, every finding checked
against the code before it was fixed:

- the server library: `tools/server_whisper.h`, its routes in `tools/server.cpp`, `src/http.h`'s `send_file`
- audio and the recorder: `src/audio.h`, `tools/arc_recorder.cpp`, `scripts/arc-server.ps1`, `tests/test_audio.cpp`
- Whisper's core: `src/whisper.h`, `zero/whisper_engine.h`, `src/whisper_formats.h`, `zero/transcribez.cpp`, the
  virtual documents of `tools/server_files.h`
- the web UI's Transcripts / Meetings tabs, the Speech setting and the `@` menu (`webui/index.html`)

On top of the reviews: a mutation fuzz of the WAV, MP4 and Ogg parsers (200,000 inputs, checked iterators: no crash,
no hang, at most 0.05 s per input; it did not reach the MP4 wrap below, which needs one exact 64-bit size).

## Found and fixed

Severity as the reviews gave it. Every fix has a check: `tests/test_audio.cpp` (in `run_tests`), or a scripted API
or browser check (the last column).

### Server and HTTP

| ID | Severity | Problem | Fix | Checked |
|---|---|---|---|---|
| A-H1 | high | `mp4_audio_delay`: a box with a 64-bit size near 2^64 wrapped `a + len` to the file's start: a 32-byte upload looped forever, holding the transcription lock (every later request waited, the server could not stop) | the size checked against the room left (`len > b - a`), at most 100,000 boxes a level | test_audio |
| W1 | medium | chat audio: an audio file that exists but cannot be read was left in the request as audio, so `dispatch` found it again and again: recursion until the stack ran out | an unreadable audio file is an error (400) | API: a file locked without sharing -> 400 at once, server alive |
| W2 | medium | chat audio: no waiting cap and no lock: 64 requests on one 3-hour clip could ask for ~44 GB; a client that left still cost the whole transcription | the OpenAI endpoint's cap (429 past 4 waiting) and its lock, the cache read again after the lock, the client's leaving stops the clip | chat audio test: 37.2 s first turn, 4.1 s from the cache |
| W3 | medium | live: no limit per request or on the queue: one 256-MB POST queued hours of utterances, and live work goes first, so file jobs and chat audio waited for hours | at most 120 s of audio a POST (413), at most 64 utterances waiting: past that the draft leaves new ones out (`behind`), the WAV keeps every sample and the accurate pass after Stop has them | API: 121 s -> 413 |
| W4 / A-M2 | medium | recordings past 4 GB (6.2 h stereo): the header's size stayed at 0xFFFFFFF0, so the accurate pass and the MP3 read 4 GB and the WAV was deleted; arc-recorder's RIFF size wrapped to 20 | `wav_format` takes a size at the field's end (>= 0xFFFFFF00) as "to the end of the file"; both sizes held at their end; the WAV deleted only when the MP3 is as long as the recording (within max(2 s, 0.2 %)) | test_audio; API: a live recording -> MP3, length checked |
| W5 | medium | `write_atomic`: the rename over a file a reader held (no FILE_SHARE_DELETE) failed silently (the accurate transcript stayed in `.tmp`); the stream checked before its last bytes were flushed; one `.tmp` name for every writer | `files::write_shared`: a temporary file of its own, closed and checked, `MoveFileExW` with retries for 2 s; every read shared for write and delete (`files::read_shared`, also the file store's reads); a failed final transcript write marks the job failed | API suite (31 checks) |
| W6 | medium | delete (and the MP3's WAV cleanup) failed while a `<video>` held the media open: meta.json went first, the folder of gigabytes stayed, unknown to the library | media served shared for delete; a delete marks the record "deleting" on disk first, removes meta.json last; folders marked so, or without meta.json, swept at the next start; a WAV left next to its MP3 deleted at start | API: delete while the media is read -> file gone at once |
| W7 | low | a restart turned "deleting" into "cancelled": a deleted item came back | swept at start | (as W6) |
| W8 | low | ids: the second + 16 random bits from an unlocked generator, never checked: two in one second could overwrite each other | 24 random bits under a lock, the id's folder created with `create_directory` (an existing one never taken) | API suite |
| W9 | low | an upload piece that failed halfway left its bytes, the retry garbled the file; up to 256 MB written under the library lock | the file cut back to the bytes received first; written outside the lock (one piece at a time per file); a delete meanwhile ends it | API suite |
| W10 | low | header injection: an imported recording's name kept CR / LF, which the export's ASCII file name passed into `Content-Disposition` | one `clean_name` for every name and speaker (also cut at a character's start); the ASCII name keeps printable characters only | |
| W11 | low | a live recording whose page crashed or went offline stayed "live" until a restart | no audio for 5 minutes: stopped and transcribed; the page also stops it on `pagehide` (`sendBeacon`) and offers Stop on a recording of another page | |
| W12 | low | `metas_[tid]` after the lock was taken again made empty records when a delete came in between | found again, 404 | API suite |
| W13 | low | `Range: bytes=N-18446744073709551615`: `y + 1` wrapped, the length promised was 2^64 - N | an end past the file is the file's end | API: 206, the whole file |
| W14 | low | the chat audio cache emptied itself at 64 clips (a chat of 65 voice notes transcribed them all every turn); a 64-bit FNV key | the 64 used last kept; the key is the length and two hashes | |
| - | low | the live retry wrote a piece twice when its answer was lost after the write | the page sends the frames it sent before (`offset`), the server leaves out what it has | API: the same piece twice -> 6 s heard, not 9 |

### Audio and arc-recorder

| ID | Severity | Problem | Fix | Checked |
|---|---|---|---|---|
| A-M1 | medium | a 44-byte WAV of 65,535 channels allocated 17 GB | at most 64 channels; reads of ~1 MB, not 65,536 frames | test_audio |
| A-M3 | medium | Ogg: every zero lacing value became an empty packet (~80x the input in memory), the whole file repacked before the length limit | empty packets left out; repacking stops a second past the limit (and at 400 packets a second); the stream copied once less | test_audio |
| A-M4 | medium | arc-recorder: a headset plugged in mid-meeting moved the sound to a new default device and the "Others" track went silent; an app started again (new process) was never found | every 2 s: another default device is opened instead; the recorded app's end seen, its name looked up again until it runs | (needs the user's devices) |
| A-M5 | medium | a recording whose recorder was killed (console closed, logoff, shutdown, power cut) never reached the server | the note next to the WAV written at the start (`complete: false`, the recorder's process); the server takes it at its start once no recorder runs, sizes put right; close / logoff / shutdown wait for the file and the note | API: an unfinished note -> imported, 20 s transcribed |
| A-M6 | medium | a recording of the same minute and title truncated the one before | " (2)", " (3)", ... | |
| A-L1 | low | Media Foundation: a selected video stream never read kept its samples queued (an hours-long video) | only the audio stream selected | API suite (videos) |
| A-L2, L4, L6 | low | a 200-ms WASAPI buffer while the other track reopens; a title cut inside a UTF-8 character (the recorder exited); a late activation's client leaked | 2-s buffer; cut at a character's start; released by the handler | |
| A-L3 | low | a reported 2e9 Hz or 2^30 channels: huge allocation, divide by zero | 1-384 kHz, 1-64 channels checked before use | |
| A-L5 | low | packets from before the start piled up at sample 0; the device clock followed in 0.1-s jumps | dropped; 40-ms steps | |
| A-L7 | low | `arc-server record -App "My App"`: the name split in two | quoted | |
| note | - | a 1-track recording a little over 180 minutes (`--seconds 10800` runs ~0.1 s over) failed its transcription | the library's files take up to 6 hours | the 3-hour test (below) |

### Whisper core, formats, virtual documents

| ID | Severity | Problem | Fix | Checked |
|---|---|---|---|---|
| F1 | medium | subtitle cues of two speakers: the "no overlap" clamp across speakers made cues that end before they start (a short "Yeah." of Others between two cues of Me), and starts out of order | cues in time order; only a speaker's own cues are kept apart; never shorter than 10 ms | test_audio |
| F2 | medium | `id,en` per utterance: a run's words after its last closing timestamp were dropped (nothing decoded them again) | kept as a segment to the run's end | mixed and 14-minute files: the same as W2 |
| F3 | medium | chat documents: a heading or a `# comment` in an answer split the exchange's section, and the title made exchange N section N + 1 | message lines that start with `#` get a space; no title heading (read_file gives the title); a summary's headings two levels down | API: section 2 = exchange 2 |
| F4 | low | `transcribez --to`: the last window encoded the speech after the end | silence after the end | |
| F5 | low | the last window's final pause cut 0.15 s in: a window of silence after it, where a hallucination could come from | cut at the end | the same results as W2 |
| F6 | low | a run's segments could end past the run (no word timestamps) | clamped to the run | |
| F7 | low | `list_chats` / `search_chats` indexed every chat in the files' cache, pushing every file's index out | a cache of their own | |
| F8 | low | `list_chats` with a query matched the reasoning, tool output and base64 data | matched on title, snippet and the messages' text | API |
| F9 | low | `transcribez`: talk.mp4 and talk.wav wrote over each other's outputs; `--dump-logits` with a list fed token -1 to the GPU | named with their extension; refused | |
| F10 | low | a word holding a blank line ended an SRT / VTT cue inside it, and broke an ASS line | white space inside a word made one space | test_audio |
| F11 | low | `chat:CON` opened the console (Windows device names were valid ids) | refused | API: 404 |

### Web UI

| ID | Severity | Problem | Fix | Checked |
|---|---|---|---|---|
| U-H1 | high | an open that was slow or failed left the selection on the new transcript while the old one was shown: Delete (its dialog naming the shown one) deleted the other one, an edit was saved over the other one | the one being opened kept apart (`tr.want`); the shown one changes only when its record came | browser: a failed open, the selection kept |
| U-H2 | high | a file opened while uploading (or a live recording) kept an empty player for good: no playback, no click-to-seek | the media set once it is playable, and again when its file changed (WAV -> MP3) | browser: upload -> done -> playable |
| U-M1 | medium | live: after a server restart every POST was refused and retried forever, the backlog growing ~345 MB an hour, lost at Stop | 404 / 409: the capture stops, what the server did not get is saved as a WAV; at most 60 s a POST, backoff, ten minutes of backlog at most | |
| U-M2 | medium | a closed page left its recording "live" for good | `pagehide` stops it; Stop shown on another page's recording; the server's 5-minute idle stop | |
| U-M3 | medium | a double click on Live started two sessions, one orphaned with the microphone on | one start at a time, the button disabled | |
| U-M4 | medium | Summarize and the live Stop opened the transcript again: an edit in progress was lost, or the copy fetched before the save landed went over it | the summary shown in place; only the live one opened again; an open during an edit or a pending save keeps the text; saves one after another | browser: edit, open, save |
| U-M5 | medium | live: the whole transcript drawn again every 0.5 s | drawn only when the text changed; 0.34-s capture buffers | live test |
| U-M6 | medium | Talk stayed on during a meeting: its spoken answers went into the recording | stopped at Live | |
| U-L1..L9 | low | uploads not retried and options read late; export without the last edit; the Speech setting used the loaded model's hearing, not the chat's; an emptied line could not be removed, Shift+Enter glued words, drafts could be edited; long summaries overflowed a small context; the Files panel opened under the Transcripts view; list clicks lost while it redrew; 176.4 / 192-kHz sound cards refused; the `@` menu came back after it was closed | a piece sent again from where the server has it (3 tries), options taken at the drop, a warning before closing during an upload; export after the save; the chat's model; an emptied line removed (asked), innerText, finished transcripts only; part summaries merged in rounds; the Transcripts view left first; rows updated in place; 48-kHz capture; the input read again after the lists come | browser: edit, delete a line, Files, `@` |
| cosmetic | - | your own recording "in another window" while finishing; a minutes title lost ".30" of "14.30"; Copy / Download in a summary's code did nothing | fixed | |

## Left, on purpose or for later

- Uploads do not resume after the page is reloaded (the record says so; delete it and add the file again).
- Live capture stays on ScriptProcessor (an AudioWorklet would take it off the page's thread); the draft is drawn
  whole when it changes, not appended.
- `stop_library` is never reached: the HTTP loop does not return and there is no console handler, so every stop is
  abrupt; what that leaves is repaired at the next start (the queue, live recordings, deletes).
- arc-recorder's device clock is followed in 40-ms steps (no fine resampling); a device change is checked every 2 s.

## The 3-hour recording

See [whisper-native-status.md](whisper-native-status.md) (W8): arc-recorder ran 3 hours unattended alongside these
tests.

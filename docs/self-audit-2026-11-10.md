# Self audit 2026-11-10: the website, the logo and the renames

What was audited: the five commits of 2026-11-08 to 11-10 that added the website, changed the logo twice and renamed
the project and its commands:

- 2f77bb4 (the website on GitHub Pages);
- 476cadf and 4ee6693 (the logo);
- d859619 (arcinferzero -> inferzero);
- a7ce806 (the commands -> zero-*).

How:
- mechanical scans: names left over, doubled words, the recorder's exe in build / CI / package / setup, every
  relative Markdown link;
- four independent reviews, one per area (C++, PowerShell, web UI and site, docs), each finding checked in the code
  before it was fixed;
- then a rebuild, the scripts parsed, the commands run, and the logits compared bit for bit.

## Findings and fixes

| # | Severity | Where | What was wrong | Fix |
|---|---|---|---|---|
| C1 | high | `zero/gguf_pack.h` | The rename turned the package's key filter into `inferzero.`, but the keys it writes are still `arcinferzero.*` (kept so older engines read newer packages). The filter stripped nothing: a re-pack wrote every `arcinferzero.*` key twice, and a GSQ model's `arcinferzero.expert_format` was copied into the package, which changes `moe_std` and could move the logits | the filter is `arcinferzero.` again, as before the rename |
| C2 | medium | `tools/server_whisper.h` | At start the server asked only the new recorder event name. A recording an older `arc_recorder.exe` was still making counted as unfinished: its WAV was "fixed" and imported while still being written | the old event name is checked too |
| C3 | medium | `tools/zero_recorder.cpp` | `start` checked only the new event name, so it started a second recording next to an older recorder (both writing `recorder.json`) | `start` refuses while the old event exists |
| C4 | low | `zero/kernels/decode.cl` | A comment changed by the rename; the kernel cache key holds the source, so every kernel from that part would have compiled again once on every PC | the comment is back as it was: the file is the same as before the rename |
| C5 | low | `tools/server.cpp` | The data folder's models were named `arcinferzero/<name>`; a client that stored that name got "model not found" | `find_model` reads `arcinferzero/<name>` as `inferzero/<name>` |
| C6 | low | `tools/web_view.h` | Leftover browser profiles `%TEMP%\arcinferzero-view-*` were no longer cleaned up | both prefixes are cleaned up |
| S1 | medium | `bench/report.py`, `bench/review_data.py`, `bench/run_bench.ps1` | Every recorded result row says `"engine": "arcinferzero"`; the reports looked for `inferzero` only and showed an empty column | old rows are read as `inferzero`; `-Engines arcinferzero` still counts |
| S2 | medium | `scripts/zero-server.ps1`, `scripts/zero-launcher.ps1` | A build from before the rename has only `bin\arc_recorder.exe`: after a `git pull` without setup, the record menu disappeared and `record-stop` / `record-status` failed with a raw error | either exe is used; without one, `record-stop` / `record-status` say so plainly |
| S3 | low | `scripts/arc-server.ps1`, `scripts/pi-arc.ps1` | The alias scripts passed on `$LASTEXITCODE`, the last native command's code, not the script's (e.g. `record-stop` with nothing recording: 1 through the alias, 0 directly) | the alias runs the new script as its own process, as the `.cmd` shims do; both give 0 |
| S4 | low | `scripts/pi-models.ps1` | Pi's `settings.json` could still name the old provider (`defaultProvider`, `enabledModels "arcinferzero/<id>"`) | those are renamed too, with a `.bak` |
| S5 | low | `scripts/zero-launcher.ps1`, `setup.ps1`, `scripts/zero-server.ps1` | "arc-server help" in a help text, "an zero-server.cmd", User-Agent `inferzero-arc-server` | renamed |
| W1 | medium | `webui/index.html` | The system prompt explained the name as "arc + infer + zero", which no longer matches the name (it is there because a model once wrote the name wrong) | "infer + zero, formerly arcinferzero" |
| W2 | low | `site/index.html` | The expert-cache grid marked every highlighted tile as cached and never took that back: after a minute nearly all looked cached, next to the text "83 %" | only cached tiles are highlighted |
| W3 | low | `site/index.html` | "2.1 to 4.9 times faster prompts" above bars showing 2.7-4.9x (2K) and 5.0-7.4x (34K) | "2.7 to 7.4 times" |
| W4 | low | `site/index.html` | Numbers that disagreed within the page: the laptop 27B's generation 5.0 next to 5.1 in the table; several cards' "1.63x / 1.97x / 2.44x" against an older one-card run, next to today's 23.2; "the limit for every model is 0.015" next to Flash Next's ~0.02 | 5.1; the bars say "run of 2026-10-31"; the limits stated for both |
| W5 | low | `site/index.html` | The Copy button inside the tablist, tabs without their panels, no arrow keys, nothing shown when the clipboard is refused; a labelled `div` without a role | Copy beside the tabs; `aria-controls` / `tabpanel`; arrow keys; the text is selected with "Press Ctrl+C"; `role="img"` |
| D1 | medium | README.md, docs/install.md | The steps name `inferzero-<version>-win64.zip` and `zero-launcher.cmd`, but the latest release (2026.11.10.3) was packed before the rename | a line for packages from before the rename; **a new release is still needed** |
| D2 | medium | QUEUE.md, docs/whisper-native*.md | `zero-recorder.exe` in 9 places; the file is `zero_recorder.exe` | fixed |
| D3 | medium | docs/whisper-native.md | `zero-server record start [--app ...]` / `record stop`, which zero-server does not take (wrong before the rename too) | `zero-server record [-App x.exe] [-NoMic] [-Title ...]`, `record-stop`, `record-status` |
| D4 | low | docs/lessons-intel-arc-igpu.md, docs/review.md, docs/architecture.md, QUEUE.md | Links and mentions of `archive/inferzero-log.md`; the archived file kept its name, `arcinferzero-log.md` | fixed |
| D5 | low | QUEUE.md, docs/launcher-research.md, CHANGELOG.md | A history line naming `%LOCALAPPDATA%\inferzero` (it was `arcinferzero`); a planned `tools\arc_launcher.cpp` next to `zero-launcher.exe`; the rename's entry still listing the arc-* commands as kept, the newest entry naming `zero-recorder stop` as a command | fixed |

## Checked and clean

- **Data folder move:** `src/app_dir.h` and `scripts/app-dir.ps1`:
  - newest-first order, whole-name matching, JSON's doubled backslashes, the lower-case context file, `ARC_DATA_DIR`;
  - the expert files' fingerprint (size, time, stem) survives the move.
- **Storage and identifiers:**
  - the web UI's only stored key is `inferzero-prefs`, with its fallback;
  - no HTTP header, URL or JSON field changed;
  - the kept GGUF keys are read with their old names;
  - `ARC_*` switches are unchanged;
  - no identifier or macro changed: the name only appears in strings and comments.
- **Commands:**
  - the aliases forward positional values, named parameters and switches; `arc-launcher.cmd` works from any folder;
  - the old Pi shims point at existing scripts;
  - `zero_recorder.exe` is built by setup and CI and shipped in the package.
- **Logo:** `Draw-ArcMark` ran at 16-144 px in Windows PowerShell 5.1; the SVG is valid and draws the Z.
- **Website:** every link to a repository file exists; the HTML has no open tags or duplicate ids, and the script
  parses.

## Left as they are

- **A race in the data folder move** (from before this work):
  - when the scripts find the old folder busy, they use it for the rest of their run;
  - a new server started from them can move it meanwhile.
  - Both then use different folders until the next start.
- **Pi sessions saved with the old provider:** resuming one in plain `pi` (not through `zero-pi`) may ask for a model.
- **An `-Switch:False` from cmd through an alias script** binds as `$true` plus a stray value. Calling the script
  directly fails with an error instead. Low.

## Verified after the fixes

- **Build:** serverz, runz and zero_recorder build without errors; every changed PowerShell script parses.
- **Logits:** Qwen3.5 4B, a ~1,500-token prompt and 16 tokens, before the rename (package 4ee6693) and now, each with
  a data folder of its own:

  | Build | SHA-256 of the logits |
  |---|---|
  | 4ee6693 | 2B7DD157F54B5BA43B545E36A5BF40FA769F7043237A837B222B6FD5DAF69E35 |
  | now | 2B7DD157F54B5BA43B545E36A5BF40FA769F7043237A837B222B6FD5DAF69E35 |

- **Recorder:**
  - `zero_recorder.exe status` and `zero-server record-status` report "not recording";
  - `record-stop` with nothing recording gives exit code 0 directly and through `scripts\arc-server.ps1`.
- **Bench reports:** `bench/report.py` and `bench/review_data.py` read the old rows again. The review tables come out
  the same as `docs/review.md`.
- **Website:** the HTML parses with no open tags or duplicate ids, and the script passes `node --check`.

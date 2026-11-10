# Launcher and CLI: what a good first version needs, and a native Windows 11 look

Research report, 2026-11-08. **Status 2026-11-09:** stages 0 and 1 are built (CHANGELOG, 2026-11-09), except
`serverz --plan`: the memory estimate is computed from `--list-models`' numbers in `arc-server` and the launcher
instead. Stages 2 and 3 are not started.

## Summary

- The launcher already has most of what other local-LLM launchers offer in a first version: a model list with
  what each model supports, an automatic context with its GPU-memory breakdown, start / stop that outlives the
  window, a live view of the running server, a first-run "Get models" dialog and, since issue #6, "Copy command".
  Few launchers explain the memory as well.
- What is missing is mostly around the edges: named presets, a log view inside the window, a tray icon, an update
  check, a short "doctor" check, and a calmer layout (13 buttons in one row, three text panels of the same weight).
  The CLI lacks `list`, `logs`, `doctor`, `version`, a dry-run that only prints the command, and `--json` output.
  It is also only on PATH when Pi is installed.
- For the look, the cheapest real gain is to **keep the PowerShell + WinForms launcher and polish it** (Segoe UI
  Variable, a dark title bar, Mica and rounded corners through `DwmSetWindowAttribute`, a simple / advanced split,
  a tray icon). It ships with nothing new, because Windows PowerShell 5.1 and .NET Framework 4.8 are on every
  Windows 10 / 11 PC.
- For a launcher that really looks like a Windows 11 app, the best fit for this project is **a small C++ exe that
  hosts an HTML page in WebView2** (`bin\arc-launcher.exe`). It is built with the same `build.ps1` / MSVC toolchain
  as the engine and ships prebuilt in the zip. WebView2 is part of Windows 11. The page reuses the web UI's CSS
  skills, and the window gets Mica from DWM behind a transparent page. WPF with the Fluent theme or WinUI 3 look
  more native, but each needs a .NET or Windows App SDK runtime of 60-200 MB in the zip and a second toolchain.
  Tauri needs Rust.
- Recommended stages: **0** fix and polish what exists (days), **1** CLI additions (days), **2** the WebView2
  launcher next to the old one (one to two weeks), **3** tray icon and updates in the new launcher, and the old
  one kept as a fallback.

## 1. What exists today

| Part | What it is | Notes |
|---|---|---|
| `arc-launcher.cmd` | starts `scripts\arc-launcher.ps1` with `pwsh` when present, else `powershell`, hidden, STA | a PowerShell console can still flash or stay visible (issue #5 comment) |
| `scripts\arc-launcher.ps1` | a WinForms window, 727 lines | model grid; options panel (main + "Show advanced options"); "Selected model" panel (features, auto-context breakdown); "Running server" panel (memory per part, last request speeds, MTP acceptance, command line); a loading progress bar from the server's `progress:` lines; buttons Get models, Start, Speech only, Stop, Set as default, Reset options, Copy command, Refresh list, Open web UI, Open log, and Record / Stop recording with an App field |
| `scripts\launcher-firstrun.ps1` | first-run part | unblocks the zip's files once; checks the GPU driver with `bin\clinfo.exe` and links Intel's driver page; "Get models" dialog with a model picked for the PC's RAM, free disk space, speech add-on, desktop shortcut, background download with progress and resume; starts the model and opens the web UI afterwards (first run only since 2026-11-10) |
| `scripts\launcher-downloads.ps1` | Downloads dialog (2026-11-10) | the Get models button and More > Download models: several catalog models at once, or a pasted Hugging Face link (repository, folder or file; split parts and an mmproj ticked with it); Pause / Continue, a speed limit, a cut connection tried again without end; runs `scripts\get-models.ps1` in the background |
| `scripts\arc-server.ps1` | the CLI, 668 lines | `start`, `restart`, `stop`, `status`, `clear-cache`, `record`, `record-stop`, `record-status`, `help`; about 40 options; a typo gets "did you mean"; on / off values checked; the server is kept when nothing changed; options like `-Npu` and `-Whisper` carry over a restart; a numbered model menu when no `-Model` is given |
| `setup.cmd` / `setup.ps1` | one-time setup | keeps prebuilt exes, downloads models, connects Pi |
| `scripts\install-pi.ps1` | puts `pi-arc.cmd` and `arc-server.cmd` into `~\.pi\agent\bin` | **only when Pi is installed**: without Pi, `arc-server` is not a command |
| `scripts\make-package.ps1` | the release zip | tracked files plus nine prebuilt exes and `vcomp140.dll`; no Visual Studio, Python or runtime needed |

State lives in `%USERPROFILE%\.arcinferzero`: `server.json` (pid, model, options, command line), `launcher.json`
(saved overrides), `last-model.txt`, `server.log`.

### What the testers ran into

- **Issue #5** (PascalSoftwares, Arc 140T): a `-f` format bug threw an index error on every 2 s refresh; on Windows
  PowerShell 5.1 every start / stop was reported as "Failed" (exit code read too late) and the box repeated because
  the timer kept ticking; a git clone has no `bin\serverz.exe`, so the model list stayed empty with only one line
  saying why. All three are fixed, but they show where the launcher is weak: it runs on two PowerShell versions
  with different behaviour, it has no tests, and errors arrive as modal boxes.
- **Issue #6**: the command line was hidden. "Copy command" and the command line in `arc-server status` answer it.
  In the same thread the owner noted that `--ctx auto` with many on / off switches "might get wrong sometimes",
  which can crash the server. A preview of the memory before Start would help here (section 2, item 6).
- **Issue #3**: `setup.cmd` depended on the PowerShell version (a PowerShell 7 module path hid cmdlets in 5.1).

### Weak spots seen while reading the code

- **The UI thread waits on the network.** `Update-Status` calls `/v1/models` and `/props` with 1 s timeouts from the
  timer, on the UI thread, so a busy or hung server freezes the window for up to two seconds at a time.
- **Errors are modal boxes.** A timer that keeps ticking under a box caused the repeated boxes in #5.
- **Layout.** 13 buttons in one row, three equal panels of monospaced text, fixed pixel sizes (1320 x 860), and the
  WinForms default look. High-DPI screens (the #5 screenshots are about 1960 px wide) may blur, depending on whether
  the hosting `powershell.exe` / `pwsh.exe` is DPI-aware. This still needs a check at 150 %.
- **The first model list is slow.** It takes up to a minute per model while the auto context is computed, with only
  a line of text as feedback.
- **The log opens in Notepad**, as a snapshot.

## 2. What other launchers offer

| | LM Studio | Ollama | Jan | koboldcpp | llama-server | text-generation-webui |
|---|---|---|---|---|---|---|
| Kind | Electron desktop app + `lms` CLI | background service + CLI (+ a small desktop app) | Tauri desktop app | one exe; Tk launcher GUI, then a web UI | one exe, CLI flags; web UI | Python web UI, one-click installers |
| Model download | in-app catalog of Hugging Face models; `lms get` | `ollama pull` from its registry | Hub with size and "fits your hardware" | none (pick a file) | `-hf repo` downloads | download tab (Hugging Face) |
| Model pick | list with size, quant, fit | `ollama list` | list | file picker | `-m`, or router mode's `--models-dir` | dropdown |
| Context / offload | per-model load settings | `num_ctx` in a Modelfile or request | per-model settings | fields in the launcher | `-c`, `-ngl` | per-model fields |
| GPU memory view | estimate before load; live use | `ollama ps` (size, CPU / GPU split) | Settings > Hardware: live CPU / RAM / GPU | none | none | none |
| Start / stop | load / eject per model; `lms server start/stop` | the service is always on; `ollama stop <model>` | start / stop the local API server | Launch; closing the console stops it | process | process |
| Presets | per-model defaults and saved presets | Modelfiles | per-model settings | `.kcpps` files: save / load in the GUI, `--config`, `--exportconfig` | `--models-preset` INI (router mode) | saved settings per model |
| Logs | developer log tab | `server.log` files | log viewer | console | console | console |
| Tray icon | yes | yes (Windows / macOS) | no | no | no | no |
| Updates | in-app updater | in-app updater | in-app updater | prints a new-version note | none | `update_wizard` script |
| First run | onboarding, suggests a model | none (CLI) | onboarding, suggests a model | the launcher is the first run | none | installer script |

### How their CLIs are laid out

- **Ollama**: a verb per job, nouns as arguments: `serve`, `run <model>`, `pull`, `list` / `ls`, `ps`, `stop <model>`,
  `show`, `rm`, `cp`, `create`. `ollama <verb> --help` per verb. Settings come from environment variables
  (`OLLAMA_HOST`, `OLLAMA_CONTEXT_LENGTH`...), listed in `ollama serve --help`.
- **LM Studio `lms`**: `lms ls` (on disk), `lms ps` (loaded), `lms load <model> --context-length --gpu --ttl
  --identifier`, `lms unload [--all]`, `lms server start/stop/status`, `lms get`, `lms log stream`; `--json` on the
  listing commands.
- **llama-server**: one long flag list, every flag also an environment variable (`LLAMA_ARG_*`). Router mode
  (`--models-dir`, `--models-preset models.ini`, `--models-max`, `--models-autoload`) serves several models from one
  process, much like arcinferzero's own model switching. It is marked experimental.
- **koboldcpp**: the same flags in the GUI and on the command line. The GUI saves them as a `.kcpps` file that
  `--config` runs, and `--exportconfig` writes one from a command line. A preset is an ordinary file the user can
  copy, edit and double-click.

### What users expect from a first version

From the table, the testers' issues, and how they talk about the others:

1. Pick a model and press one button; see clearly whether the server is on, and its address.
2. Download a suggested model without knowing Hugging Face.
3. See whether a model fits before loading it (LM Studio's estimate and Jan's "fits" badge are their most praised
   points).
4. A saved setup they can start again: a preset, or a command line for a script (issue #6).
5. A log they can read and paste into a bug report.
6. The app tells them when an update is out.
7. The CLI follows the Ollama / `lms` pattern: `list`, `ps` / `status`, `start` / `load`, `stop`, `pull`, `logs`.

## 3. MVP feature list, ranked

Value: what it does for a new user or a tester. Effort: S (hours), M (a day or two), L (a week or more), counted
for the current PowerShell launcher unless noted. Ranked by value against effort.

| # | Feature | Today | Value | Effort | Notes |
|---|---|---|---|---|---|
| 1 | Pick a model, Start / Stop, server outlives the window | **have** | high | - | via `arc-server`; double-click starts |
| 2 | First run: driver check, suggested model download, start, open web UI | **have** | high | - | `launcher-firstrun.ps1` |
| 3 | Clear server state and address | **have** | high | - | status line, API address with Copy |
| 4 | Copy the start command / show the server's command line | **have** | high | - | issue #6 |
| 5 | Errors in the window, not modal boxes; the UI never waits on the network | **missing** | high | S-M | a message bar with "Show log"; status calls in a background job / runspace |
| 6 | Fits-or-not before Start: the planned GPU memory for the options as set, with a warning when the context or switches leave too little | **partly** | high | M | the auto-context breakdown exists per model, but not for overridden options; needs a `serverz --plan` (or `--list-models` with options) that prints the memory for given options without loading |
| 7 | Simple view first: model, context, Start; the rest one click away | **partly** | high | S | "Show advanced options" exists, but 13 buttons and three panels show at once |
| 8 | Log view inside the window (live tail, copy, "open folder") | **partly** | medium-high | S | today: Notepad snapshot |
| 9 | Named presets (save / load / delete; a preset is also a `.cmd` or JSON file the CLI takes) | **partly** | medium-high | M | "Set as default" keeps one unnamed set in `launcher.json` |
| 10 | "Doctor": driver version, GPU and its memory setting, NPU, XMX, exe present and version, port free, other `serverz` / `runz` running, folder writable | **partly** | medium-high | S-M | parts exist (`clinfo` check, missing-exe message); one report to paste into issues would have shortened #5 |
| 11 | Update check: compare the package version (`bin\PREBUILT.txt`) with the latest GitHub release and show a link | **missing** | medium | S | read-only `api.github.com/repos/rizagp/arcinferzero/releases/latest`; no self-update in a first version |
| 12 | Tray icon: on / off state, Start last model, Stop, Open web UI, Quit window | **missing** | medium | S-M | WinForms `NotifyIcon` exists in .NET Framework |
| 13 | Model management: show in folder, size on disk, remove (with a confirmation, to the Recycle Bin) | **partly** | medium | M | listed, no actions |
| 14 | Live GPU memory bar and speed of the running server | **partly** | medium | S | as text with a `#####` bar; a real bar and a small tok/s line |
| 15 | Windows 11 look: Segoe UI Variable, dark mode, Mica, rounded corners, DPI-aware | **missing** | medium | S (title bar, Mica) to L (full dark theme) | section 5 |
| 16 | Start with Windows / start the last model at sign-in | **missing** | low-medium | S | a shortcut in the Startup folder or a scheduled task; off by default |
| 17 | Speech only, Record meeting | **have** | niche | - | keep, but move under a "Speech" section |
| 18 | Multi-GPU options (`--gpus`) in the launcher | **missing** | niche (B60 users) | S | issue #4 users run the CLI anyway |
| 19 | Self-update (download and replace the folder) | **missing** | low now | L | risky with a running server and a zip install; later |

Recommended MVP cut: items 5-12, plus 14 and the cheap part of 15. Item 6 is the most valuable new one: the
owner's own note in #6 is that wrong switch combinations can crash the server or the whole system.

## 4. CLI changes worth making

`arc-server` is already careful: typos get suggestions, values are checked, and the help has "WRITING OPTIONS" and
"WHEN IT FAILS" sections. The gaps are in the commands it has, and in where it is found.

1. **On PATH without Pi.** Put `arc-server.cmd` in the package root (next to `setup.cmd`), and let setup add a
   folder to the user PATH with its own question. `install-pi.ps1` keeps the Pi-only `pi-arc`. Today a user without
   Pi has to type `powershell -NoProfile -ExecutionPolicy Bypass -File ...\scripts\arc-server.ps1`.
2. **`arc-server list`** (alias `ls`): the supported models with id, shortcut, size, auto context and variants. It
   is the numbered menu without the question. `--json` passes `serverz --list-models` through.
3. **`arc-server start ... -DryRun`** (or `print-command`): checks the options and prints the `serverz.exe` command
   and the planned memory without starting anything. The launcher's "Copy command" and the fits-or-not preview
   (item 6) can use the same path.
4. **`arc-server logs [-Follow] [-Lines N]`**: the log without knowing where it is; `-Follow` tails it.
5. **`arc-server doctor`**: the report of item 10, plain text, made to paste into a GitHub issue.
6. **`arc-server version`** (and `--version` on `serverz.exe`): the package version and commit from `PREBUILT.txt`,
   the driver version, and whether a newer release exists (only when asked: `version -Check`).
7. **`arc-server pull <key>`** (alias `get`): `scripts\get-models.ps1` with the catalog keys (`4b`, `35b`, `asr`...),
   plus `pull -List` for the catalog. Same names as the shortcuts.
8. **Presets**: `arc-server start -Preset work` reads `%USERPROFILE%\.arcinferzero\presets\work.json` (the
   launcher's options, model included); `arc-server save-preset work -Model 35b -Ctx 65536 -Npu on` writes one.
   Command-line options override the preset. The launcher's "Set as default" becomes the preset `default`, so
   both share one format, in the way koboldcpp's `.kcpps` files work.
9. **`--json` on `status`** (and `list`): one JSON object for scripts, instead of the current mix of text and a
   compressed `/props` dump.
10. **Shorter `help`.** The help is about 150 lines. Show the commands and the ten common options by default, with
    `help options`, `help paths` and `help errors` for the rest (the current sections, unchanged).
11. **Exit codes that mean something**: 0 ok, 1 wrong option, 2 server failed to load, 3 timeout, 4 port in use, so
    scripts and the launcher can tell them apart without parsing text.
12. **Drafter naming.** The owner's comment in #6 speaks of one `--drafter` switch for MTP and DFlash2, while
    `arc-server` has `-NoMtp`, `-DFlash` and the `-nodf` / `-df` model suffixes, and `serverz` has `--draft
    auto|mtp|dflash`. A single `-Drafter auto|mtp|dflash|off|<folder>` in `arc-server`, with the old names kept as
    aliases, would match what users are told.

Not worth copying: Ollama's always-on service (the GPU memory is shared with Windows on an iGPU, so a server that
is always loaded costs the user RAM), and environment variables for every option (the `ARC_*` variables are
engine switches for testing and should stay that way).

## 5. A native look on Windows 11: the options

The hard constraint: **the zip must run without Visual Studio, Python or a runtime install, and the exes ship
prebuilt.** A runtime may ship inside the zip, but it counts against the download.

### A. Keep PowerShell + WinForms, add Windows 11 styling

- **How:** `Add-Type` with a few lines of C# P/Invoke (it compiles with the C# compiler that ships with .NET
  Framework, so no SDK is needed): `DwmSetWindowAttribute` with `DWMWA_USE_IMMERSIVE_DARK_MODE` (20) for a dark
  title bar, `DWMWA_WINDOW_CORNER_PREFERENCE` (33) for rounded corners, and `DWMWA_SYSTEMBACKDROP_TYPE` (38) =
  `DWMSBT_MAINWINDOW` (2) for Mica. Mica and the backdrop need Windows 11 build 22000+ (22621+ for attribute 38);
  older builds keep today's look. Add Segoe UI Variable (Windows 11) with Segoe UI as the fallback, flat buttons
  with an accent colour, more padding, and a `NotifyIcon`.
- **Limits:** Mica only shows where the controls are transparent, and WinForms controls on .NET Framework paint
  their own opaque backgrounds. So in practice Mica shows on the title bar and the window margins, not behind the
  grid. A full dark mode means repainting `DataGridView`, `ComboBox`, `NumericUpDown` and scrollbars by hand.
  The built-in `Application.SetColorMode` only exists in .NET 9+ WinForms, where it is experimental, and Windows
  PowerShell runs on .NET Framework. The UI also still runs on two PowerShell versions (5.1 and 7) with different
  behaviour, the source of issues #3 and #5.
- **Cost:** build none; package nothing new (a few KB of script); maintenance as today. **S** for the title bar,
  corners, fonts, layout and tray icon; **L** for a complete dark theme.

### B. WPF with the Fluent theme (.NET 9 / 10)

- **How:** `ThemeMode="System"` gives Windows 11 controls, light and dark, and the accent colour. Mica needs a
  library (FluentWpfCore, MicaWPF) or the same DWM call as in A.
- **Limits:** the Fluent theme is still marked experimental (`WPF0001`), and it needs .NET 9+, so the zip has to
  carry the .NET desktop runtime self-contained. That is roughly 60-150 MB, since WinForms and WPF do not support
  trimming or Native AOT. The other way is asking users to install the .NET Desktop Runtime, which breaks the "no
  install" rule. It adds the .NET SDK to the build, a new language (C# / XAML), and a second build system next to
  `build.ps1`.
- **Cost:** build **M-L**; package +60-150 MB; maintenance: a second toolchain with yearly .NET upgrades.

### C. WinUI 3 / Windows App SDK

- **How:** the real Windows 11 controls (NavigationView, InfoBar, Mica built in). Unpackaged and self-contained
  with `WindowsAppSDKSelfContained`, and since Windows App SDK 1.5 also as a single file.
- **Limits:** the biggest runtime of the options (the Windows App SDK plus .NET, well over 100 MB unpacked). The
  heaviest toolchain (Visual Studio workload, NuGet, XAML compiler). Unpackaged apps lose some identity-based
  features. The most "native" result, but heavy for a launcher with one main window.
- **Cost:** build **L**; package +100-200 MB; maintenance high (SDK versions move fast).

### D. A web page served by `serverz` itself

- **How:** a `/launcher` page in the existing web UI: model list, options, restart with new options. The server
  already lists models (`/v1/models`) and switches between them on request.
- **Limits:** it cannot start a server that is not running. That is the launcher's main job (and the first-run
  case), so something else still has to exist. The page would also give the HTTP server the power to restart
  itself with arbitrary options, which matters when it listens on `0.0.0.0` or through the Cloudflare tunnel
  (`docs/remote-access.md`). It suits a "change model / options while running" panel in the web UI. It cannot
  replace the launcher.
- **Cost:** build **M**; package nothing; maintenance shared with the web UI.

### E. Tauri or a WebView2 host

- **Tauri 2:** a Rust exe (a few MB) using the system WebView2, which Windows 11 includes. A good result, but it
  adds Rust, Cargo and Node to a C++ project, plus a second build to keep working.
- **A C++ WebView2 host (recommended):** the same idea without Rust. A Win32 window in C++ (built by
  `build.ps1`, like the other exes) creates a WebView2 control and shows an HTML / CSS / JS page shipped in the
  zip (`launcher\index.html`). The build needs the WebView2 SDK header and its static loader lib. The SDK is a
  NuGet package that can be committed to `third_party\` or fetched by `build.ps1`, and the zip needs nothing
  extra. With a transparent WebView2 background and `DWMSBT_MAINWINDOW`, Mica shows behind the whole page,
  rounded corners come free on Windows 11, and dark mode follows `prefers-color-scheme`. The page talks to the
  host through `window.chrome.webview.postMessage`, and the host runs `arc-server.ps1` (or later starts
  `serverz.exe` itself), reads `server.json` and polls `/props` off the UI thread.
- **Limits:** WebView2 is missing on some Windows 10 LTSC installs, so the host checks for it and falls back to the
  PowerShell launcher. A Fluent look in CSS is hand-made, though Fluent UI Web Components can be shipped as one JS
  file. The exe is new C++ code, but small (window, WebView2, process start, a JSON bridge).
- **Cost:** build **M** (about a week for the host plus the page); package +~1 MB; maintenance: one C++ file, the
  rest is HTML / CSS / JS like `webui/index.html`.
- **The cheapest variant**, for a quick try: `msedge --app=file:///.../launcher/index.html` gives a window without
  browser chrome, with no new exe. But it has no host to start processes, so it only works next to a small local
  service, and it has no Mica.

### Comparison

| Option | Looks like Windows 11 | Zip size added | Build toolchain | Runs on Win 10 | Effort | Fits this project |
|---|---|---|---|---|---|---|
| A. WinForms + DWM | partly (title bar, corners, fonts) | ~0 | none | yes | S (L for full dark) | **yes, now** |
| B. WPF Fluent | mostly | 60-150 MB | .NET SDK, C# / XAML | yes | M-L | weak |
| C. WinUI 3 | fully | 100-200 MB | VS workload, Windows App SDK | yes (1809+) | L | weak |
| D. Page in serverz | web look | 0 | none | yes | M | as an extra panel only |
| E1. Tauri | mostly (Mica via plugin) | ~5-10 MB | Rust, Node | yes, with WebView2 | M-L | adds Rust |
| E2. C++ + WebView2 | mostly (Mica, dark, rounded) | ~1 MB | MSVC (as today) + WebView2 SDK | yes, with WebView2 | M | **yes, next** |

## 6. Recommendation and stages

**Polish the existing launcher first, then replace its window with a C++ + WebView2 one that uses the same
`arc-server` commands**, and keep the PowerShell launcher as the fallback. That way:
- the zip stays free of runtimes;
- the build stays MSVC only;
- the UI is written in the HTML / CSS this project already writes well (`webui/index.html`);
- the window gets real Mica and dark mode on Windows 11.

Each stage can ship on its own.

### Stage 0: fix and polish the current launcher (2-3 days)

- Move the `/v1/models` and `/props` polling off the UI thread (a runspace or background job writing a file,
  as the model list already does); replace modal error boxes with a message bar plus "Show log" / "Copy details".
- A simple layout: the model list; under it context, the API address and one large Start / Stop; options, the
  memory breakdown and the server monitor in tabs; secondary buttons in a "More" menu.
- Windows 11 touches through DWM (dark title bar, rounded corners, Mica where it shows), Segoe UI Variable, and a
  check at 150 % scaling.
- A log tab with a live tail.
- A tray icon (on / off, Start last, Stop, Open web UI).
- An update check against the latest GitHub release, shown as a link.

### Stage 1: CLI (2-3 days, in parallel with stage 0)

- `arc-server.cmd` in the package root and an optional PATH entry without Pi.
- `list`, `logs`, `doctor`, `version`, `pull`, `-DryRun`, `--json` on `status` / `list`, presets (`-Preset`,
  `save-preset`, shared with the launcher), exit codes, a shorter `help`.
- `serverz --plan` (or options on `--list-models`) for the planned memory of a given option set. This is the base
  for the fits-or-not preview in both the CLI and the launcher.

### Stage 2: the WebView2 launcher (1-2 weeks)

- `tools\arc_launcher.cpp` builds `bin\arc-launcher.exe`: a Win32 window with Mica, WebView2 with a transparent
  background, and a small message bridge (list models, start, stop, status, open log, copy command). Starting and
  stopping still go through `arc-server.ps1`, so the CLI and the window cannot drift apart.
- `launcher\index.html` (plus CSS / JS): the stage-0 layout, with light / dark from the system, the accent colour,
  and a real GPU memory bar with the fits-or-not preview.
- `arc-launcher.cmd` starts the exe when WebView2 is there, else the PowerShell window.
- `make-package.ps1` adds the exe and the `launcher\` folder; `build.ps1` learns the WebView2 SDK.

### Stage 3: later

- Start with Windows (off by default), model management (show in folder, to the Recycle Bin), named presets in the
  UI, `--gpus` for multi-GPU users.
- A "while running" panel in the web UI (option D) for changing the model or context without the launcher, only on
  `127.0.0.1` and never through the tunnel.
- Self-update only if the testers ask for it.

### Not recommended

WPF / WinUI 3 for now: the runtime they add to the zip is larger than every other part except the models. They
also bring a second toolchain, for a window with one list and a few fields. Revisit only if the launcher grows into
a full app with chat, which the web UI already covers.

## Sources

Local:
- `scripts/arc-launcher.ps1`, `scripts/launcher-firstrun.ps1`, `scripts/arc-server.ps1`, `scripts/install-pi.ps1`,
  `scripts/make-package.ps1`, `scripts/model-catalog.ps1`, `setup.cmd`, `arc-launcher.cmd`, `USAGE.md` sections 1-3
  and 8, `tools/server.cpp` (routes and `--list-models`).
- GitHub issues [#3](https://github.com/rizagp/arcinferzero/issues/3),
  [#5](https://github.com/rizagp/arcinferzero/issues/5), [#6](https://github.com/rizagp/arcinferzero/issues/6) and
  [#4](https://github.com/rizagp/arcinferzero/issues/4), with their comments.

Other launchers and CLIs:
- LM Studio `lms`: https://lmstudio.ai/docs/cli, https://lmstudio.ai/blog/lms, https://github.com/lmstudio-ai/lms
- Ollama CLI: https://fossies.org/linux/ollama/docs/cli.mdx (mirror of `docs/cli.mdx`),
  https://www.glukhov.org/post/2024/12/ollama-cheatsheet/
- Jan: https://jan.ai/docs/desktop/settings, https://docs.jan.ai/local-server/settings,
  https://www.storagereview.com/review/jan-v0-8-6-hands-on-running-local-llms-on-windows
- koboldcpp `.kcpps`, `--exportconfig`, `--showgui`: https://newreleases.io/project/github/LostRuins/koboldcpp/release/v1.85,
  https://newreleases.io/project/github/LostRuins/koboldcpp/release/v1.75
- llama-server router mode: https://glukhov.org/llm-hosting/llama-cpp/llama-server-router-mode/,
  https://github.com/ggml-org/llama.cpp/issues/18035

Windows UI:
- DWM attributes: https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute,
  https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwm_systembackdrop_type
- WinForms dark mode (.NET 9, experimental): https://learn.microsoft.com/dotnet/desktop/winforms/whats-new/net90
- WPF Fluent theme (.NET 9): https://learn.microsoft.com/dotnet/desktop/wpf/whats-new/net90, MicaWPF
  https://github.com/Simnico99/MicaWPF
- WinUI 3 unpackaged / self-contained: https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/unpackage-winui-app,
  https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/project-properties
- WebView2 in Windows 11 and Tauri's install modes: https://v2.tauri.app/distribute/windows-installer/,
  https://v1.tauri.app/v1/guides/building/windows

Sizes for self-contained .NET apps (60-150 MB) come from general sizing guides and from WinForms / WPF not
supporting trimming or Native AOT. They should be measured with a sample publish before any decision that rests
on them.

# Installing arcinferzero

(Moved from the README on 2026-10-31.)

Everything below runs in **Windows PowerShell** from the project folder (the folder that contains `build.ps1`).
If PowerShell refuses to run the scripts ("running scripts is disabled on this system"), start them as
`powershell -ExecutionPolicy Bypass -File <script> <arguments>`, or allow local scripts once for your user with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Ready-to-use package (no compiler, no Python, no terminal)

The package (`arcinferzero-<version>-win64.zip`, 6 MB, **download: https://github.com/rizagp/arcinferzero/releases/latest**)
holds the programs already built. On a laptop with an Intel
Core Ultra (Arc graphics) and Intel's graphics driver, nothing else is needed:

1. Unpack the zip anywhere (right-click, "Extract All...", e.g. to `C:\arcinferzero`).
2. Double-click `arc-launcher.cmd`. Windows may ask once whether to run it (the zip came from the internet: "More
   info", "Run anyway"); the launcher then unblocks its own files.
3. The first time, a window offers a model: the one that suits the laptop's RAM is ticked (4B on 16 GB, the 35B on
   32 GB), each with a line on what it is good at; speech to text (recordings, meetings, voice messages) is an
   add-on, and a desktop shortcut is offered. "Download and start" fetches it with a progress bar (a cut connection
   goes on where it stopped), starts it, and opens the web UI (`http://127.0.0.1:8090/`) in the browser.

Later: double-click the shortcut (or `arc-launcher.cmd`), select a model, Start; "Get models" adds more. Without
the Intel driver the launcher says so and offers Intel's download page.

Behind it: the programs carry the C++ runtime inside (`/MT`) and ship the OpenMP runtime `vcomp140.dll` next to them
(no Visual C++ Redistributable); models download with `scripts\get-models.ps1` / `scripts\hf-get.ps1` (`curl.exe`,
part of Windows; every file's SHA256 checked) into the Hugging Face cache, from the list in
`scripts\model-catalog.ps1`. `setup.cmd` works too (the same models, plus Pi); `bin\PREBUILT.txt` marks the package,
so setup keeps the shipped programs (`setup.cmd -Rebuild` builds them from the sources in the package: step 3).
The package is made with `scripts\make-package.ps1` (into `dist\`).

## Quick setup

From the sources: once the graphics driver, Visual Studio Build Tools (C++) and optionally Pi are installed (steps
1–3 and 9.1 below), one command does the rest:

```powershell
.\setup.cmd                    # checks, builds, asks which models to download, connects Pi
.\setup.cmd -Models 27b,4b     # download these without asking (27b 35b 35b-gsq 9b 4b g12 e4b, whisper asr: speech to text; all, none)
.\setup.cmd -Tests             # also build the test tools and run the regression tests (needs the 4B)
```

It checks the requirements (and says where to get what is missing), builds `bin\serverz.exe`, `runz.exe`,
`clinfo.exe`, `transcribez.exe` and `arc_recorder.exe` when they are missing or older than the sources, checks the GPU (OpenCL 3.0, DP4a, sub-groups, GPU
memory), downloads the chosen models into the Hugging Face cache without Python (Gemma with its drafter and image
projector; `35b-gsq` is the one model converted with Python, then it installs `huggingface_hub` / `numpy` with pip
after asking), and
connects Pi (`pi-arc`, `arc-server`, Pi's model list with each model's context). Running it again only does what
is missing: a model the server already finds (in the Hugging Face cache, as a package in
`%USERPROFILE%\.arcinferzero\zero\models` or in a folder of `model-dirs.txt`) is not downloaded again. Double-clicked,
its window stays open at the end so the result can be read; it ends with setup's exit code (1 when a step failed).
Afterwards the easiest start is `arc-launcher.cmd` (see the package's steps above). Steps 1–10 describe the same by
hand.

## 1. Requirements

| | Needed | Tested with |
|---|---|---|
| OS | Windows 11, 64-bit | Windows 11 Home 25H2 (build 26200) |
| GPU | Intel Core Ultra with Intel Arc integrated graphics: Meteor Lake / Arrow Lake (Xe-LPG), and Lunar Lake / Panther Lake (Xe2 / Xe3: the 8-lane kernels are built with 16 lanes there, and the prompt GEMMs use XMX after a start-up check; a user ran the 35B on an Arc B390 at 1,182 tok/s prompt, 24 tok/s generation, before the XMX path existed). The engine needs OpenCL 3.0 with `cl_intel_subgroups` and `cl_khr_integer_dot_product` (DP4a). Discrete Arc GPUs are not supported. | Core Ultra 5 125H, Arc iGPU 112 EU |
| Memory | 16 GB works for the 4B / E4B models; 32 GB for the 12B–35B models and long contexts. The iGPU can use part of the RAM (`bin\clinfo.exe`, "global mem": set in Intel Graphics Software; 20.4 GiB of 32 GB on the test laptop now); arcinferzero uses all of it but 1.2 GiB for the driver, the rest of the RAM stays with Windows. Each model's context follows that setting. | 32 GB LPDDR5x-6400 |
| CPU | AVX2 (every Core Ultra has it; the build uses `/arch:AVX2`) | |
| Graphics driver | Intel Arc & Iris Xe graphics driver (it contains the OpenCL runtime; nothing else to install) | 32.0.101.9033 |
| Compiler | Visual Studio 2026 Build Tools with the **Desktop development with C++** workload (MSVC + Windows SDK); not for the ready-to-use package | VS Build Tools 18 |
| Python | only for the `35b-gsq` conversion (Python 3.10 or newer with `huggingface_hub` and `numpy`); downloads use `curl.exe` | Python 3.14 |
| Pi (optional) | the Pi coding agent, which runs on Node.js | Node.js 24 |
| Disk | 3–13 GB per model | |

## 2. Graphics driver

Install or update the Intel graphics driver from Intel's download center ("Intel Arc & Iris Xe Graphics - Windows")
or with the Intel Driver & Support Assistant, then restart. The OpenCL runtime comes with the driver; there is no
SDK to install: arcinferzero loads `OpenCL.dll` at run time and the driver compiles the kernels.

## 3. Compiler

Install **Visual Studio 2026 Build Tools** (free, from visualstudio.microsoft.com, "Tools for Visual Studio" →
"Build Tools for Visual Studio 2026") and select the workload **Desktop development with C++** (it includes MSVC
and the Windows SDK). `build.ps1` finds the compiler itself (`vswhere`: any Visual Studio or Build Tools
installation with `VC\Auxiliary\Build\vcvars64.bat`, Community and Professional included); to use a particular
one, set `$env:ARC_VCVARS` to its `vcvars64.bat`.

## 4. Get the code

Copy or clone the `arcinferzero` folder anywhere (the scripts find everything relative to their own location). It
already contains everything it needs to build (`third_party\` holds minja and nlohmann/json).

## 5. Build

```powershell
.\build.ps1 zero\serverz.cpp zero\runz.cpp tools\clinfo.cpp zero\transcribez.cpp tools\arc_recorder.cpp
```

This takes a few minutes and writes `bin\serverz.exe` / `bin\runz.exe` (arcinferzero: the HTTP server and the
command-line tool for runs, benchmarks and accuracy checks) and `bin\clinfo.exe`. `zero\serverz.cpp` compiles
`tools\server.cpp` (the HTTP API) on the Level Zero engine. Run the programs from the project folder, because they
read `zero\kernels\*.cl` and `webui\` from there. They are built with the C++ runtime inside (`/MT`) and the OpenMP
runtime `vcomp140.dll`, which `build.ps1` copies into `bin\` from Visual Studio's redistributable folder: the folder
runs on a PC without Visual Studio or the Visual C++ Redistributable.

Check that the GPU is usable:

```powershell
.\bin\clinfo.exe
```

It should list `Intel(R) Arc(TM) Graphics`, `OpenCL 3.0 NEO`, `cl_khr_integer_dot_product yes`,
`cl_intel_subgroups yes` and sub-group sizes `8 16 32`.

Optional tools (tests and benchmarks):

```powershell
.\build.ps1 tools\tok_test.cpp tools\chat_render.cpp tools\ref_forward.cpp tools\gguf_dump.cpp tools\tool_stream_test.cpp
.\build.ps1 zero\vision_testz.cpp tools\gemma4v_ref.cpp tools\gemma4a_ref.cpp
.\build.ps1 bench\gemv_bench.cpp bench\attn_dec_bench.cpp bench\attn_bench.cpp bench\bw_bench.cpp
```

`zero\vision_testz` times an image through a model's encoder (Gemma 4 or Qwen3-VL `mmproj`, `ARC_PROF=1` per kernel)
and with `ARC_VIS_LM=file` writes the model's log-probabilities for 32 tokens after the image, to compare two encoder
builds or against `tools\gemma4v_ref` (the Gemma 4 encoder in f32 on the CPU); an audio clip with E4B's `mmproj`
the same against `tools\gemma4a_ref` (its audio conformer in f32; `ARC_AUDIO_EMB` runs the model on those rows). `runz --bench-batch N --prof-gen`
shows each kernel's GPU time of a decoding step alone and batched; `ARC_KLIST=1` prints the cache file and build
options of every kernel, to read the driver's own assembly with Intel's `ocloc disasm`
([docs/kernel-audit-2026-10-27.md](kernel-audit-2026-10-27.md)). The source builds warning-free at
`/W4 /permissive-`; `.github/workflows/ci.yml` builds every exe and runs the model-free unit tests on each push.

## 6. Download models

Without Python: `powershell -File scripts\get-models.ps1 4b asr` (the keys of `setup.cmd -Models`, from
`scripts\model-catalog.ps1`), or any repository's files with
`powershell -File scripts\hf-get.ps1 <org/name> <file> ...` (curl.exe, resumed after a cut, SHA256 checked, into
the Hugging Face cache). With Python, install the Hugging Face command-line tool once:

```powershell
python -m pip install -U huggingface_hub
```

Then download the models you want. `hf download` puts them in the Hugging Face cache
(`%USERPROFILE%\.cache\huggingface\hub`), which is where `arc-server` / `pi-arc` look for them, so no paths have to
be configured. For the Gemma models, the drafter (`mtp-*.gguf`) and the image projector (`mmproj-*.gguf`) must sit
in the same folder as the model: download them from the same repository in the same command.

| Alias | Model | Command | Size |
|---|---|---|---|
| `27b` | Qwen3.8 27B GSQ-RCO (with MTP head) | `hf download ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | 12.1 GB |
| `9b` | Qwen3.5 9B (with MTP head) | `hf download unsloth/Qwen3.5-9B-MTP-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf` | 6.1 GB |
| `4b` | Qwen3.5 4B (with MTP head) | `hf download unsloth/Qwen3.5-4B-MTP-GGUF Qwen3.5-4B-UD-Q4_K_XL.gguf` | 3.0 GB |
| `g12` | Gemma 4 12B QAT (Google, Q4_0) + image projector; drafter from Unsloth, copied into the model's snapshot folder | `hf download google/gemma-4-12B-it-qat-q4_0-gguf` and `hf download unsloth/gemma-4-12B-it-qat-GGUF mtp-gemma-4-12B-it.gguf` | 7.0 + 0.2 + 0.3 GB |
| `e4b` | Gemma 4 E4B QAT (Google, Q4_0); drafter from Unsloth, copied into the model's snapshot folder | `hf download google/gemma-4-E4B-it-qat-q4_0-gguf gemma-4-E4B_q4_0-it.gguf gemma-4-E4B-it-mmproj.gguf` (images, audio) and `hf download unsloth/gemma-4-E4B-it-qat-GGUF mtp-gemma-4-E4B-it.gguf` | 5.2 + 0.1 GB |
| `g26` | Gemma 4 26B-A4B QAT (Google, Q4_0) + image projector; drafter `mtp-gemma-4-26B-A4B-it.gguf` from Unsloth's Gemma 4 26B-A4B QAT GGUF repository, copied next to the model (optional: ~25 tok/s with it) | `hf download google/gemma-4-26B-A4B-it-qat-q4_0-gguf` | 14.4 + 1.2 GB |
| `35b` | Qwen3.6 35B-A3B UD-IQ4_XS (Unsloth's MTP repository: MTP block + image projector) | `hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-IQ4_XS.gguf mmproj-BF16.gguf` | 18.2 GB + projector |
| `35b-mtp` | Qwen3.6 35B-A3B UD-Q3_K_M of the same repository | `hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q3_K_M.gguf mmproj-BF16.gguf` | |
| (`35b`) | Qwen3.6 35B-A3B, 2-bit GSQ, no MTP (what `setup.cmd -Models 35b` downloads) | see step 7 (download 12.6 GB, then convert) | 11.7 GB |
| `flash` | Qwen3.8 Flash Next (arcinferzero only); images: an image projector copied into the model's folder (the server looks only next to the model; with a package: next to the package) | `hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --include "IQ3_S/*"`; projector `hf download bartowski/Qwen3.8-Flash-Next-GGUF mmproj-Qwen3.8-Flash-Next-f16.gguf` | 54.8 + 28.8 GB + 0.9 GB |
| | DFlash2 drafters (optional, see `--dflash` below) | `hf download z-lab/Qwen3.8-27B-DFlash2` / `hf download incoai/Qwen3.6-35B-A3B-DFlash2` | 3.85 / 1.05 GB |
| | Speech to text (optional, `arc-server start -Whisper auto`): Whisper large-v3, and Qwen3-ASR 1.7B (the default speech model; its mmproj next to it) | `hf download Systran/faster-whisper-large-v3` and `hf download unslothai/Qwen3-ASR-1.7B-GGUF --include "*Q8_0*" "*mmproj*"` | 2.9 + 2.4 GB |

With their `mmproj` file next to the model, Gemma 4 12B and E4B read images and audio, and Gemma 4 26B, Qwen3.5
4B / 9B, Qwen3.8 27B and Qwen3.6 35B read images (all encoders run on the GPU).
Transcripts use Qwen3-ASR (the default), Whisper large-v3 (turbo only when large-v3 is not cached) or Gemma 4 E4B
([docs/qwen3-asr.md](qwen3-asr.md#using-it)); Gemma 4 12B is refused for transcripts: it has no audio encoder
(its projector maps raw 40-ms waveform frames straight into the model), and on noisy recordings of mixed Indonesian
and English it made up sentences or repeated a word where E4B's conformer transcribed them.

Shortcuts (`pi-arc <shortcut>`, `arc-server start -Model <shortcut>`): `27b`, `35b`, `35b-s`, `35b-mtp`, `flash`,
`9b`, `4b`, `g12`, `e4b`, `g26`. Other GGUFs of these architectures get an id from their file name (`pi-arc --pick`
lists them). `35b` is the UD-IQ4_XS file above, or UD-Q3_K_M / the GSQ file when that is the one
downloaded (with several, the first found keeps the shortcut); `35b-s` is Unsloth's UD-Q3_K_S. Variant suffixes:
`-mtp` (MTP, no images), `-vision` (images, no MTP), `-std` (neither), and for a model with a DFlash2 drafter
`-nodf` / `-df` (see `--dflash` below).

## 7. Qwen3.6-35B-A3B (optional): convert the GSQ checkpoint

ISTA-DASLab publishes this model only as a safetensors checkpoint (compressed-tensors: 2-bit routed experts
`{-2,-1,0,1} x bf16 scale per 128`, int8 attention / DeltaNet / shared experts / lm_head). Convert it once
(about 2.5 minutes; it needs `numpy`, and the tokenizer metadata comes from any GGUF with the same vocabulary, e.g.
the 4B from step 6):

```powershell
python -m pip install -U numpy
hf download ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ
$snap = (Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub\models--ISTA-DASLab--Qwen3.6-35B-A3B-2Bit-GSQ\snapshots" | Select-Object -First 1).FullName
$vocab = (Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub" -Recurse -Filter "Qwen3.5-4B-UD-Q4_K_XL.gguf" | Select-Object -First 1).FullName
New-Item -ItemType Directory -Force "$env:USERPROFILE\.arcinferzero\models" | Out-Null
python tools\convert_gsq.py $snap "$env:USERPROFILE\.arcinferzero\models\Qwen3.6-35B-A3B-GSQ2-arc.gguf" --vocab-from $vocab
```

The int8 matrices become Q8_0 with the same int8 values (f16 scales), the experts keep their exact 2-bit codes in
the layout of `zero/kernels/moe.cl`, and the tensor transforms follow llama.cpp's converter (DeltaNet V-head order,
norm weights + 1, A = -exp(A_log)). The checkpoint's MTP block has no routed experts, so this model runs without
speculative decoding. `arc-server` / `pi-arc` find the converted file in `%USERPROFILE%\.arcinferzero\models`.

## 8. First run

```powershell
.\scripts\arc-server.ps1 start -Model e4b      # or 27b, 35b, flash, 9b, 4b, g12, g26
```

This starts arcinferzero. The first start of each model compiles its GPU
kernels (up to a minute or two); they are cached in `%USERPROFILE%\.arcinferzero\kernels`, so later starts take 5–40 s
(mostly reading the model file). The first arcinferzero start of a mixture-of-experts model also writes its expert
file once (35B UD-IQ4_XS: 16 GB, about 2.5 minutes; Qwen3.8 Flash Next: 56 GB, about 9.5 minutes; later loads through the
server ~21–26 s) to `%USERPROFILE%\.arcinferzero\zero\`; it can be deleted at any time and is written again when missing. Then open the web
UI at http://127.0.0.1:8090/ or call the OpenAI API at `http://127.0.0.1:8090/v1`. `arc-server status` shows what
is running and `arc-server stop` frees the GPU memory.

**Launcher window.** `arc-launcher.cmd` (double-click it in the project folder) opens a window to pick a model,
set its options and start or stop the server: [USAGE.md, section 3](../USAGE.md#3-managing-the-server).

## 9. Pi (optional)

1. Install Pi by following the instructions on https://pi.dev and run it once, so that `%USERPROFILE%\.pi\agent`
   exists (with `bin` on the PATH).
2. Add arcinferzero to Pi (one time, `setup.cmd` does it too; it builds `bin\serverz.exe` if it is missing, adds the
   `arcinferzero` provider with every model it finds and its context for the current GPU memory to
   `~/.pi/agent/models.json` (the previous file is kept as `models.json.bak`; `pi-arc` refreshes the list), and
   installs the `pi-arc` and `arc-server` commands):

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\install-pi.ps1
   ```

3. Open a new terminal and start Pi from any project folder:

   ```powershell
   pi-arc                  # the model the server runs or used last (the first time: choose from a list)
   pi-arc 27b              # a model by its shortcut (step 6);  pi-arc --pick = choose from every downloaded model
   pi-arc --help           # every option
   ```

   Inside Pi, `/model` switches between all of them (the server loads the chosen one). Every pi-arc and
   arc-server command: [USAGE.md, sections 2 and 3](../USAGE.md#2-using-pi).

## 10. Check the installation (optional)

With the test tools from step 5 built, the regression tests compare every kernel with a CPU reference, the
tokenizer and chat templates with the reference output, the model's next-token distributions with an f32 CPU
implementation, and the server API:

```powershell
powershell -ExecutionPolicy Bypass -File tests\run_tests.ps1               # arcinferzero, models found automatically
powershell -ExecutionPolicy Bypass -File tests\run_tests.ps1 -SkipGpu      # only the checks without a GPU / model
powershell -ExecutionPolicy Bypass -File tests\run_tests.ps1 -ListModels   # which model each check would use
```

It runs the model-free checks first (HTTP parser, API formats, JSON, sampler, expert-cache pools, tool-call
streaming, attention and routing kernels, chat templates against the GGUFs' Jinja templates), then per model found
(4B, 27B, 35B, Flash Next, Gemma 4 12B / E4B / 26B) the KL and greedy checks and MTP / DFlash2 =
plain decoding, then a server smoke test; a missing model is a SKIP, not a failure. It ends with `ALL TESTS PASSED`. Stop `arc-server` first: the
test uses port 8091 and needs the GPU memory.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cannot load OpenCL.dll` / `no OpenCL GPU found` | Install or update the Intel graphics driver (step 2) and restart. |
| `build failed` right away / `No Visual Studio C++ compiler found` | Visual Studio's C++ workload is missing (step 3); to use a particular installation, set `$env:ARC_VCVARS` to its `vcvars64.bat`. `setup.cmd` checks this. The compiler output of `setup.cmd` is in `%USERPROFILE%\.arcinferzero\build.log`. |
| `running scripts is disabled on this system` | Run the script through `powershell -ExecutionPolicy Bypass -File ...` (see the top of this section). |
| `Model file ... not found` | Download it with `hf download` (step 6); `arc-server` searches `%USERPROFILE%\.arcinferzero\models` and the Hugging Face cache (models on another drive: [USAGE.md, section 8](../USAGE.md#8-troubleshooting)). |
| The first start of a model takes minutes | The driver compiles the kernels once; later starts use `%USERPROFILE%\.arcinferzero\kernels`. |
| Gemma does not use its drafter | `mtp-gemma-4-*-it.gguf` must be in the same folder as the model GGUF (download it from the same repository). |
| Later problems (the server stops while loading, Windows gets slow, Pi's `web_search`, slow first sessions, models on another drive) | [USAGE.md, section 8](../USAGE.md#8-troubleshooting). |

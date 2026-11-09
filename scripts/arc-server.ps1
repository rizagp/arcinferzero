# Starts / stops the arcinferzero server in the background for Pi and other OpenAI-compatible clients.
#
#   arc-server start                  asks which model to load: every supported GGUF model in the Hugging Face cache
#                                     (and %USERPROFILE%\.arcinferzero\models), listed by its full name; Enter keeps the
#                                     running one. Waits until the server is ready.
#   arc-server start -Model 27b       the same without asking: a shortcut (27b 35b 35b-s 35b-mtp flash 9b 4b g12 e4b g26,
#                                     with -mtp / -vision / -std / -nodf / -df for a variant), a model id, part of a model's
#                                     name (e.g. swift) or a path to a .gguf file
#        [-SessionCacheGb N]          save conversations to disk when switching (N GB, all models together; default 8, 0 = off)
#        [-Vision on|off]             image / audio encoders (default on; off: no mmproj, more context)
#        [-Ctx N|auto] [-Chunk N] [-NoMtp]  context (default: auto, the largest that fits the GPU memory),
#                                     prompt chunk size, no speculative decoding
#        [-NoStream]                  MoE experts on the GPU instead of streamed from the SSD (bin\serverz.exe,
#                                     Level Zero: the engine; the OpenCL one, bin\server.exe, was removed 2026-10-20)
#   arc-server stop                   stop the server
#   arc-server status                 show what is running and the saved conversations
#   arc-server clear-cache            delete the saved conversations (session cache) of all models
#   arc-server record [-App Zoom.exe] [-NoMic] [-Title "..."]   record a meeting (bin\arc_recorder.exe, in the
#                                     background): the microphone as "Me", the app's sound (or the speakers') as
#                                     "Others"; record-stop ends it and the server transcribes it (the Meetings tab);
#                                     record-status
#   arc-server start -Whisper auto    also speech to text (Qwen3-ASR by default, Whisper, Gemma 4 E4B): the web UI's
#                                     Transcripts / Meetings tabs, voice messages, /v1/audio/transcriptions
#   arc-server start ... -DryRun      check the options, print the server's command line and a GPU memory estimate
#   arc-server start -Preset work     the options saved with save-preset -Preset work (the command line wins)
#   arc-server list | logs | doctor | version [-Check] | pull <keys> | presets | save-preset   (help: arc-server help)
#
# Exit codes: 0 ok, 1 a wrong option or value (the message says which), 2 the server stopped while loading,
# 3 not ready within 10 minutes, 4 the port is used by another program.
#
# The server keeps running after Pi exits, so the next session skips the model load and reuses the prompt cache.
# Log: %USERPROFILE%\.arcinferzero\server.log
param(
  [ValidateSet("start", "stop", "status", "restart", "clear-cache", "record", "record-stop", "record-status", "list", "ls",
    "logs", "log", "doctor", "version", "--version", "pull", "get", "presets", "save-preset", "help", "--help", "-h")] [string]$Action = "start",
  [string]$Model = "",
  [string]$Ctx = "",
  [int]$Chunk = 0,
  [double]$SessionCacheGb = 8,
  [int]$Port = 8090,
  [switch]$NoMtp,
  [string]$Vision = "",
  [switch]$NoStream,
  [int]$Slots = 0,
  [string]$Npu = "",
  [switch]$Precise,
  [switch]$NoPrecise,
  [string]$Whisper = "",
  [string]$WhisperDevice = "",
  [string]$Asr = "",
  [int]$WhisperMaxMinutes = 0,
  [int]$AsrSlots = 0,
  # the other server options (0 / empty / off = the server's default)
  [int]$MtpDrafts = 0,
  [string]$DFlash = "",
  [double]$DriverReserveGb = -1,
  [int]$ThinkBudget = 8192,
  [int]$ImageMinTokens = 0,
  [int]$ImageMaxTokens = 0,
  [double]$BudgetGb = 0,
  [switch]$AttnF32,
  [switch]$PreserveThinking,
  [switch]$NoPrefixCache,
  [double]$PrefixCacheGb = -1,
  [switch]$NoSwap,
  [string]$ListenHost = "",
  [string]$SessionDir = "",
  # record: the meeting app (its sound is "Others"; none: everything the speakers play), no microphone, a title
  [string]$App = "",
  [switch]$NoMic,
  [string]$Title = "",
  [switch]$Quiet,
  # (added last: the positions of the options above stay as they were)
  [string]$Preset = "",   # start / restart: the options of a saved preset; save-preset: its name
  [switch]$DryRun,        # start / restart: print the command and the memory estimate, start nothing
  [switch]$Json,          # status, list, doctor, version, presets, -DryRun: JSON for scripts
  [switch]$Follow,        # logs: keep printing new lines
  [int]$Lines = 40,       # logs: how many lines
  [switch]$Check,         # version: ask GitHub whether a newer package is out
  [string]$Drafter = ""   # auto | mtp | dflash | off | <DFlash2 folder>
)
$ErrorActionPreference = "Stop"
# (an option this script does not have went to $args and was ignored: "-Whispr on" started the server without
#  speech. Now it stops, with the option meant: one dash or two, any case, the closest name)
if ($args.Count) {
  $names = @((Get-Command $PSCommandPath).Parameters.Keys | Where-Object { $_ -notin [Management.Automation.PSCmdlet]::CommonParameters })
  $bad = [string]$args[0]
  $key = ($bad -replace '^-+', '').ToLower()
  function Edit-Distance([string]$x, [string]$y) {
    $d = 0..$y.Length
    for ($p = 1; $p -le $x.Length; $p++) {
      $prev = $d[0]; $d[0] = $p
      for ($j = 1; $j -le $y.Length; $j++) {
        $t = $d[$j]; $d[$j] = [Math]::Min([Math]::Min($d[$j] + 1, $d[$j - 1] + 1), $prev + [int]($x[$p - 1] -ne $y[$j - 1])); $prev = $t
      }
    }
    return $d[$y.Length]
  }
  $best = $names | Sort-Object { Edit-Distance $key ($_.ToLower() -replace '-', '') } | Select-Object -First 1
  $hint = if ($best -and (Edit-Distance $key ($best.ToLower() -replace '-', '')) -le 3) { ": did you mean -$($best)?" } else { "" }
  $what = if ($bad -like "-*") { "unknown option $bad" } else { "'$bad' is a value without an option before it (a path with spaces goes in quotes)" }
  throw "arc-server: $what$hint (arc-server help lists the options)"
}
$root = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot "app-dir.ps1")
$stateDir = Get-AppDir  # (%USERPROFILE%\.arcinferzero; renamed from arcinfer once)
New-Item -ItemType Directory -Force $stateDir | Out-Null

# ---- presets: %USERPROFILE%\.arcinferzero\presets\<name>.json, {"Model": "35b", "Ctx": "65536", "Npu": "on", "NoMtp": true}:
# arc-server's own option names (the launcher's "Save as preset" writes the same). start -Preset <name> takes them;
# an option given on the command line wins over the preset's.
$presetDir = Join-Path $stateDir "presets"
$presetKeys = @("Model", "Ctx", "Chunk", "SessionCacheGb", "Port", "NoMtp", "Vision", "NoStream", "Slots", "Npu", "Precise",
  "NoPrecise", "Whisper", "WhisperDevice", "Asr", "WhisperMaxMinutes", "AsrSlots", "MtpDrafts", "DFlash", "DriverReserveGb",
  "ThinkBudget", "ImageMinTokens", "ImageMaxTokens", "BudgetGb", "AttnF32", "PreserveThinking", "NoPrefixCache",
  "PrefixCacheGb", "NoSwap", "ListenHost", "SessionDir", "Drafter")
function Preset-Path([string]$n) {
  $n = "$n".Trim()
  if ($n -notmatch '^[\w][\w .-]{0,63}$') { throw "-Preset: a name of letters, digits, spaces, dots and dashes (got '$n')" }
  return Join-Path $presetDir "$n.json"
}
if ($Preset -and $Action -in "start", "restart") {
  $pp = Preset-Path $Preset
  if (-not (Test-Path -LiteralPath $pp)) { throw "-Preset: no preset '$Preset' ($pp); arc-server presets lists them" }
  try { $po = Get-Content -LiteralPath $pp -Raw | ConvertFrom-Json } catch { throw "preset '$Preset' ($pp) is not valid JSON: $($_.Exception.Message)" }
  foreach ($p in $po.PSObject.Properties) {
    $k = @($presetKeys | Where-Object { $_ -eq $p.Name })[0]  # (the script's spelling: -eq ignores case)
    if (-not $k) { throw "preset '$Preset': unknown option '$($p.Name)' ($pp)" }
    if ($PSBoundParameters.ContainsKey($k)) { continue }  # (the command line wins)
    $v = $p.Value
    if ((Get-Variable $k).Value -is [switch]) { $v = [switch][bool]$v }
    Set-Variable -Name $k -Value $v  # (the parameter's type converts it: "4096" for -Chunk becomes 4096)
    $PSBoundParameters[$k] = $v      # (counts as given: a running server with other options is restarted)
  }
}
# -Drafter auto|mtp|dflash|off|<folder>: one name for the drafters. off = -NoMtp, a folder = -DFlash <folder>; mtp /
# dflash / auto pick the model's variant below (the -nodf / -df / plain ids)
$drafterPick = ""
if ($Drafter) {
  $l = $Drafter.Trim().ToLower()
  if ($l -in "off", "no", "none", "0", "false") { $NoMtp = [switch]$true }
  elseif ($l -in "auto", "mtp", "dflash", "on") { $drafterPick = $(if ($l -eq "on") { "auto" } else { $l }) }
  elseif (Test-Path -LiteralPath $Drafter.Trim().Trim('"') -PathType Container) { $DFlash = $Drafter.Trim().Trim('"'); $PSBoundParameters["DFlash"] = $DFlash }
  else { throw "-Drafter: expected auto, mtp, dflash, off or a DFlash2 drafter's folder (got '$Drafter')" }
}
# on / off options: on, yes, true, 1 / off, no, false, 0 (any case); a wrong value says what is expected
function Switch-Value([string]$name, [string]$v, [string[]]$more = @()) {
  $l = "$v".Trim().ToLower()
  if ($l -eq "") { return "" }
  if ($l -in "on", "yes", "true", "1") { return "on" }
  if ($l -in "off", "no", "false", "0") { return "off" }
  if ($l -in $more) { return $l }
  throw "-${name}: expected $((@("on", "off") + $more) -join ', ') (got '$v')"
}
$Vision = Switch-Value "Vision" $Vision
$Npu = Switch-Value "Npu" $Npu
$l = "$WhisperDevice".Trim().ToLower()
if ($l -and $l -notin "gpu", "npu") { throw "-WhisperDevice: expected gpu or npu (got '$WhisperDevice')" }
$WhisperDevice = $l
# -Whisper / -Asr: a word (auto / on / off) or a path. A path's last backslash goes (written in quotes as "C:\folder\" it
# turned the closing quote into part of the path for the server), and the path is checked here: a folder with
# model.bin for -Whisper, a .gguf file for -Asr
function Speech-Value([string]$name, [string]$v, [bool]$folder) {
  $l = "$v".Trim().ToLower()
  if ($l -in "", "auto", "off") { return $l }
  if ($l -in "on", "yes", "true", "1") { return "auto" }
  if ($l -in "no", "false", "0") { return "off" }
  $p = "$v".Trim().Trim('"').TrimEnd('\', '/')
  if (-not (Test-Path -LiteralPath $p)) { throw "-${name}: no $(if ($folder) { 'folder' } else { 'file' }) '$p' (auto, on, off or $(if ($folder) { 'a faster-whisper folder with model.bin' } else { 'a Qwen3-ASR .gguf file' }))" }
  if ($folder) {
    if (-not (Test-Path -LiteralPath $p -PathType Container)) { throw "-${name}: '$p' is a file: give the folder that holds model.bin, $(Split-Path $p)" }
    if (-not (Test-Path -LiteralPath (Join-Path $p "model.bin"))) { throw "-${name}: '$p' holds no model.bin: give the faster-whisper folder itself (e.g. ...\models--Systran--faster-whisper-large-v3\snapshots\<hash>)" }
  } else {
    if (Test-Path -LiteralPath $p -PathType Container) { throw "-${name}: '$p' is a folder: give the .gguf file in it ($((Get-ChildItem -LiteralPath $p -Filter *.gguf | Where-Object { $_.Name -notlike 'mmproj*' } | Sort-Object { $_.Name -notmatch 'Q8' } | Select-Object -First 1).Name))" }
    if ($p -notlike "*.gguf") { throw "-${name}: '$p' is no .gguf file" }
  }
  return (Resolve-Path -LiteralPath $p).Path
}
$Whisper = Speech-Value "Whisper" $Whisper $true
$Asr = Speech-Value "Asr" $Asr $false
# -Ctx: a number of tokens, or auto (which also restarts a server running a fixed context); not given (or 0): a
# running server is kept as it is, a new one starts with auto
$Ctx = "$Ctx".Trim().ToLower()
if ($Ctx -eq "0") { $Ctx = "" }
if ($Ctx -and $Ctx -ne "auto" -and $Ctx -notmatch '^\d+$') { throw "-Ctx: a number of tokens or auto (got '$Ctx')" }
# the engine: arcinferzero (Level Zero); -NoStream keeps every expert on the GPU
$Engine = if ($NoStream) { "zero-nostream" } else { "zero" }
# ($moe: the model has routed experts, from serverz --list-models; only those keep their experts on the SSD)
function Engine-Text([string]$e, [bool]$moe = $false) {
  switch ($e) {
    "zero" { if ($moe) { "arcinferzero (Level Zero; experts on the SSD with a GPU cache)" } else { "arcinferzero (Level Zero)" } }
    "zero-nostream" { "arcinferzero without the SSD cache (Level Zero; every weight on the GPU)" }
    default { "an older server ($e)" }
  }
}
$exe = Join-Path $root "bin\serverz.exe"
$log = Join-Path $stateDir "server.log"
$stateFile = Join-Path $stateDir "server.json"
$base = "http://127.0.0.1:$Port"
# saved conversations (server --session-cache-gb, default 8 GB)
$sessDir = if ($SessionDir) { $SessionDir } else { Join-Path $stateDir "zero\sessions" }  # (status / clear-cache: -SessionDir's folder)

# The models come from `serverz.exe --list-models` (Hugging Face cache + %USERPROFILE%\.arcinferzero\models): name, file,
# alias (= the model id of the API and Pi), shortcut, the context --ctx auto chose for the current GPU memory. The
# context is auto unless -Ctx is given: the largest that fits the GPU memory currently set less 0.5 GiB for the
# driver, at most the trained one (Intel Graphics Software can change the GPU memory; it is then
# computed again). Smaller prompt chunks (-Chunk 1024) free ~1.1 GiB on the 27B for more context.
$lastFile = Join-Path $stateDir "last-model.txt"  # the model started last (pi-arc starts it again by default)

function Say($msg) { if (-not $Quiet) { Write-Host $msg } }

# The id of the model the server has loaded: the entry marked loaded (the server lists every model it can switch
# to), else the first (servers that list only the loaded one).
function Get-Running {
  try { $d = @((Invoke-RestMethod "$base/v1/models" -TimeoutSec 2).data) } catch { return $null }
  $l = @($d | Where-Object { $_.loaded -eq $true })
  if ($l.Count) { return $l[0].id } else { return "none" }  # (-Model none, or a failed switch: no chat model)
}

# Model files compared by full path: serverz.exe lists Hugging Face cache paths with mixed separators
# (C:\Users\me/.cache/huggingface/hub\models--...), a resolved path has only backslashes.
function Same-Path($a, $b) {
  if (-not $a -or -not $b) { return $false }
  try { return [IO.Path]::GetFullPath($a) -eq [IO.Path]::GetFullPath($b) } catch { return $a -eq $b }
}

function Stop-Arc {
  $stopped = $false
  # the conversation the server holds goes to the session cache first (when it has one), so that continuing it after
  # the restart loads it instead of processing it again
  try { $null = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/sessions/save" -TimeoutSec 60 } catch {}
  if (Test-Path $stateFile) {
    $st = Get-Content $stateFile -Raw | ConvertFrom-Json
    # (only the server of this port: -Port 8091 left the 8090 one, which the state file describes, running)
    if (-not $st.port -or "$($st.port)" -eq "$Port") {
      $p = Get-Process -Id $st.pid -ErrorAction SilentlyContinue
      if ($p -and $p.ProcessName -eq "serverz") { Stop-Process -Id $p.Id -Force; $stopped = $true }
      Remove-Item $stateFile -Force
    }
  }
  # a server started by hand on the same port
  Get-CimInstance Win32_Process -Filter "Name = 'serverz.exe'" |
    Where-Object { $_.ExecutablePath -like "$root\bin\serverz.exe" -and $_.CommandLine -match "--port\s+$Port\b" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $stopped = $true }
  # A new server that allocates its GPU memory while Windows is still releasing the old one's can run at half speed
  # for its whole life (measured: Qwen3.6 35B on arcinferzero 26.4 -> 11.9 tok/s when started at once, normal with
  # 5 s in between): the process ends first, then a short pause.
  if ($stopped) {
    Get-Process -Name server, serverz -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$root\bin\server*.exe" } |
      Wait-Process -Timeout 20 -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 5
    Say "arcinferzero server stopped."
  }
  return $stopped
}

# The models serverz.exe finds (Hugging Face cache + arcinferzero's model folder), with their alias.
function Get-Models {
  $all = @(& $exe --list-models | ConvertFrom-Json | ForEach-Object { $_ })  # (Windows PowerShell: unroll the array)
  foreach ($m in $all) { $m | Add-Member ctx 0 -Force }  # (only -Ctx sets a fixed context here)
  return $all
}

# Pi's model list (~/.pi/agent/models.json, provider "arcinferzero") follows what is on disk: a model deleted from the
# Hugging Face cache leaves Pi's /model, a new one appears; each with its context (the running one's from the server)
function Sync-PiModels {
  if (-not (Test-Path (Join-Path $env:USERPROFILE ".pi\agent"))) { return }
  . (Join-Path $PSScriptRoot "pi-models.ps1")
  $cfgPath = Join-Path $env:USERPROFILE ".pi\agent\models.json"
  $before = if (Test-Path $cfgPath) { (Get-Item $cfgPath).LastWriteTime } else { $null }
  $id = Get-Running
  $n = 0
  if ($id) { try { $n = [int](Invoke-RestMethod "$base/props" -TimeoutSec 5).n_ctx } catch {} }
  try { $ids = @(Update-PiModels $exe $base $id $n) } catch { Say "  (Pi's model list not updated: $_)"; return }
  if ((Test-Path $cfgPath) -and (Get-Item $cfgPath).LastWriteTime -ne $before) { Say "Pi's model list updated: $($ids.Count) models" }
}

function Pick-Model($models, $runningPath, $runningAlias) {
  $ok = @($models | Where-Object { $_.supported })
  Write-Host ""
  Write-Host "Models (Hugging Face cache and $stateDir\models):"
  for ($i = 0; $i -lt $ok.Count; $i++) {
    $m = $ok[$i]
    $tags = @("{0:N1} GB" -f ($m.bytes / 1e9))
    if ($m.ctx_train) { $tags += "max context length {0:N0}" -f $m.ctx_train }
    if ($m.tags) { $tags += $m.tags }
    if ($m.ctx) { $tags += "auto context {0:N0}" -f $m.ctx }
    if ($m.short) { $tags += "shortcut $($m.short)" } else { $tags += "id $($m.alias)" }
    $mark = if ($runningAlias -and $m.alias -eq $runningAlias) { "  <- running" } else { "" }
    Write-Host ("  {0,2}) {1}" -f ($i + 1), $m.name) -NoNewline -ForegroundColor Cyan
    Write-Host ("   {0}  [{1}]{2}" -f $m.file, ($tags -join ", "), $mark)
  }
  $skip = @($models | Where-Object { -not $_.supported })
  if ($skip.Count) { Write-Host ("  (not supported by arcinferzero: " + (($skip | ForEach-Object { "$($_.name) [$($_.arch)]" }) -join ", ") + ")") }
  $prompt = if ($runningPath) { "Model number (Enter = keep the running model)" } else { "Model number" }
  while ($true) {
    $a = Read-Host $prompt
    if (-not $a -and $runningPath) { return $null }
    $n = 0
    if ([int]::TryParse($a, [ref]$n) -and $n -ge 1 -and $n -le $ok.Count) { return $ok[$n - 1] }
    Write-Host "Type a number from 1 to $($ok.Count)."
  }
}

if ($Action -in "help", "--help", "-h") {
  $short = @"
arc-server: the arcinferzero server in the background (port $Port), for Pi, the web UI and other OpenAI-compatible
clients. Options take one dash or two, any case.

  arc-server start [-Model <shortcut|id|file>] [options]   start, or switch the model; waits until ready
  arc-server restart [options]     restart with other options          arc-server stop      stop, free the GPU memory
  arc-server status [-Json]        what runs, its command line         arc-server list      the models on this PC
  arc-server logs [-Follow]        the server's log                    arc-server doctor    a check to paste into a bug report
  arc-server version [-Check]      this package, and whether a newer one is out
  arc-server pull <keys>           download models: 4b 9b 35b 27b g12 g31 e4b asr whisper (pull alone lists them)
  arc-server presets               the saved presets;  save-preset -Preset <name> [options] saves one
  arc-server clear-cache           delete saved conversations;  record / record-stop / record-status: meetings

Common options for start / restart:
  -Model 27b | 35b | 9b | 4b | g12 | e4b | g26 | flash | none    (arc-server list: every id and shortcut)
  -Ctx <N|auto>      context in tokens (auto: the largest that fits)
  -Preset <name>     the options of a saved preset (an option on the command line wins)
  -Drafter auto|mtp|dflash|off|<folder>    speculative decoding: which drafter, or none
  -Vision on|off     images / audio input (off: more context)
  -Npu on|off        part of each prompt on the NPU (27B, Gemma 4 12B)
  -Whisper auto|off  speech to text (Transcripts / Meetings tabs)
  -Slots <N>         conversations at once          -ListenHost 0.0.0.0   reachable from other computers
  -DryRun            check the options, print the server's command line and a GPU memory estimate; start nothing

More: arc-server help options (every option) | help writing (values, paths) | help errors (what a message means)
      | help all. Exit codes: 0 ok, 1 wrong option, 2 the server stopped while loading, 3 timeout, 4 port in use.
"@
  $full = @"
arc-server: the arcinferzero server in the background (port $Port), for Pi and other OpenAI-compatible clients.
Options take one dash or two, any case (-Whisper auto, --whisper auto); WRITING OPTIONS below says how to write
values and paths, WHEN IT FAILS what an error means.

  arc-server start [options]     start, or switch to another model; waits until ready. Also brings Pi's model
                                 list up to date (a model deleted from the Hugging Face cache leaves it, a new
                                 one appears), even when the server is already running.
  arc-server restart [options]   restart with new options
  arc-server status              running model, its context and chunk size, saved conversations
  arc-server stop                stop the server and free the GPU memory
  arc-server clear-cache         delete saved conversations (session cache). Saved system prompts (per project
                                 folder, 8 GB in all, reused across restarts) are in %USERPROFILE%\.arcinferzero\zero\prefix;
                                 deleting that folder is safe while the server is stopped.
  arc-server record [-App Zoom.exe] [-NoMic] [-Title "weekly"]
                                 record a meeting in the background (bin\arc_recorder.exe): your microphone as
                                 "Me", the app's sound (Zoom.exe, Teams.exe, chrome.exe...; without -App: all the
                                 speakers play) as "Others", into %USERPROFILE%\.arcinferzero\recordings\. It goes on
                                 if the server stops or restarts.
  arc-server record-stop         end it: the server transcribes it (Me / Others) in the Meetings tab, now or
                                 at its next start
  arc-server record-status       whether a recording runs, and where
  arc-server list [-Json]        the models serverz.exe finds: id, shortcut, size, auto context, MTP, images
                                 (also ls; -Json: serverz --list-models as it is)
  arc-server logs [-Lines N] [-Follow]   the last N lines of the server's log (default 40); -Follow keeps
                                 printing new ones (Ctrl+C ends it)
  arc-server doctor [-Json]      checks the exes, the GPU driver, the NPU, memory and disk, the server and its
                                 port, other serverz / runz processes: a report to paste into a GitHub issue
  arc-server version [-Check]    this package's version and commit; -Check asks GitHub for the latest release
  arc-server pull <key,key>      download models of the catalog into the Hugging Face cache (no Python):
                                 4b 9b 35b 27b g12 g31 e4b asr whisper; pull alone lists them (also get)
  arc-server presets [-Json]     the saved presets (%USERPROFILE%\.arcinferzero\presets\<name>.json)
  arc-server save-preset -Preset <name> [options]   save the options given (the model too) as a preset;
                                 start -Preset <name> uses them. Delete one: delete its .json file.
  arc-server help [options|writing|errors|all]   help (also --help, -h)

Options for start / restart:
  -Model <shortcut|text|path>    27b 35b 35b-s 35b-mtp flash 9b 4b g12 e4b g26; a variant with -mtp (MTP, no images),
                                 -vision (images, no MTP) or -std (neither), e.g. 9b-std, 27b-std; a model id
                                 (as in Pi's /model), part of a model's name (e.g. swift) or a .gguf path.
                                 none: speech to text only (Qwen3-ASR; -Whisper auto adds Whisper), the web UI's
                                 Transcripts tab; a chat model loads later when a chat picks one.
                                 -nodf / -df (models with a DFlash2 drafter): the MTP block only / DFlash2 only.
                                 Without it: a numbered list of the models in the Hugging Face cache.
  -Preset <name>                 the options saved with save-preset (options on the command line win)
  -Drafter auto|mtp|dflash|off|<folder>   which drafter speculative decoding uses: auto (the model's best), mtp
                                 (the MTP block or Gemma drafter only = the -nodf id), dflash (DFlash2 only = -df),
                                 off (= -NoMtp), or a DFlash2 drafter's folder (= -DFlash <folder>)
  -DryRun                        check everything, print the serverz.exe command line and a GPU memory estimate
                                 (with -Json as JSON), and start or stop nothing
  -Ctx <N|auto>                  context in tokens, or auto = the largest that fits the GPU memory currently
                                 set, less 0.5 GiB for the driver (computed on the first start
                                 of a model, remembered, recomputed when the GPU memory changes). Not given:
                                 a running server keeps its context, a new one starts with auto. -Ctx auto
                                 also restarts a server that runs a fixed context.
  -Chunk <N>                     prompt chunk size (default per model: 27B 4096; 35B 8192; Gemma 4 26B and Flash
                                 Next 2048; 4B / 9B / E4B 512; Gemma with images at least 2048); smaller = less
                                 GPU memory, more context
  -SessionCacheGb <N>            save conversations to disk when switching sessions (GB, all models together; default 8, 0 = off)
  -NoMtp                         no speculative decoding
  -Slots <N>                     hold N conversations at once (default 1): two Pi sessions each keep their own
                                 KV cache and states and take turns step by step (a prompt piece, a token), so
                                 neither waits for the other's whole answer; while both write, their next tokens
                                 run in one pass. Each slot gets the
                                 context that fits N of them (-Ctx auto; 27B: 2 x 49,152, 35B: 2 x 65,536, g26 2 x 32,768)
  -Npu on|off                    the NPU (Intel AI Boost) computes part of each prompt chunk next to the GPU
                                 (Qwen3.8 27B +15 %, Gemma 4 12B +10 %, 0.4 GB of RAM; other models stay off). Kept when
                                 the model changes (pi-arc, Pi's /model) and on a restart; off again after stop
  -Precise / -NoPrecise          Qwen3.8 Flash Next's precise mode (BF16 hyper-connection rows and router, two int8
                                 passes everywhere): KL 0.0186 -> 0.0127 at 3.5K tokens (under the 0.015 limit),
                                 prompts about half as fast; other models are unchanged. Kept like -Npu
  -Whisper auto|on|off|<folder>  speech to text at /v1/audio/transcriptions (OpenAI's API) and the Transcripts tab:
                                 Qwen3-ASR (default), Gemma 4 E4B or Whisper large-v3 (turbo when large-v3 is not
                                 in the Hugging Face cache; <folder>: a faster-whisper folder); Whisper and Qwen3-ASR
                                 take ~8 GB of GPU memory off the context (docs/qwen3-asr.md). Kept like -Npu
  -WhisperDevice gpu|npu         Whisper's encoder on the iGPU (default) or the NPU
  -Asr auto|off|<file.gguf>      Qwen3-ASR, the default speech model (auto: from the Hugging Face cache when it is
                                 there; off: Whisper alone). Without -Whisper: Qwen3-ASR alone (~3.7 GB of GPU
                                 memory instead of ~9.8; no translations, no live text, no word times). Kept like -Npu
  -AsrSlots <1-4>                Qwen3-ASR parts of a clip longer than 4 minutes decoded together (default 1; each
                                 more ~0.6 GB of GPU memory; 4: a 14-minute file 129 -> 80 s). Kept like -Asr
  -WhisperMaxMinutes <N>        the longest upload or chat clip (default 180); the Transcripts library takes files up
                                 to 6 hours, or N when it is more
  -NoStream                      every expert on the GPU: no expert file, no SSD reads; MoE models then get a
                                 smaller context. Without it, MoE models (35B, Flash Next, Gemma 4 26B) keep their
                                 experts (MTP layer included) on the SSD and the most used ones in a GPU cache
                                 and read the next layer's missing experts while the GPU computes (35B and Flash
                                 Next: 131,072 tokens of context by default, up to 262,144 with -Ctx). The first start writes the expert file (35B: 16 GB, ~2.5 min; Flash Next:
                                 56 GB, ~9.5 min; Gemma 4 26B: 13 GB, ~1 min) to %USERPROFILE%\.arcinferzero\zero (safe
                                 to delete: it is written again).
  -Vision on|off                 image / audio input (default on). off: the projector (mmproj) is not loaded and
                                 --ctx auto keeps no GPU memory for it, so the context is longer
  -Port <N>                      port (default 8090; also for status / stop)
  -MtpDrafts <1..7>              drafts per step (default: 7 for Qwen 27B-class and with DFlash2, 5
                                 for Qwen MoE, 3 for 4B and Gemma 4 26B, 2 otherwise)
  -DriverReserveGb <N>           GPU memory left to the driver, GiB (default 0.5; 0 to 8)
  -DFlash <folder>|off           a DFlash2 drafter of your own: its snapshot folder (with config.json and
                                 model.safetensors), e.g. for a fine-tuned variant the model list does not
                                 pair with one; off: no drafter. It stays with the server: a model
                                 picked inside it (Pi's /model) uses it too, until the next start
  -ThinkBudget <N>               default reasoning budget in tokens (default 8192; requests can override it)
  -ImageMinTokens / -ImageMaxTokens <N>  tokens per image (defaults: Qwen 256..2048, Gemma 4 12B 1120..2048)
  -BudgetGb <GB>                 keep decoded weights resident up to GB (default: chosen with the context)
  -AttnF32                       f32 attention (tests)
  -PreserveThinking              earlier turns keep their reasoning (Qwen3.6 35B; Qwen3.8 27B always does)
  -NoPrefixCache                 do not keep system prompts on disk;  -PrefixCacheGb <N>  their disk space (8)
  -NoSwap                        the server keeps its model (requests for another model do not switch)
  -ListenHost <addr>             listen address (default 127.0.0.1; 0.0.0.0 = other computers too)
  -SessionDir <folder>           where saved conversations go (default %USERPROFILE%\.arcinferzero\zero\sessions)
  -Quiet                         no messages
A running server is kept when the model and every option given match; otherwise it is restarted with them.

WRITING OPTIONS
  dashes and case               -Whisper auto, --whisper auto and -whisper AUTO are the same here (and in pi-arc).
                                bin\serverz.exe itself takes two dashes only: --whisper auto.
  on / off                      on, yes, true, 1 or off, no, false, 0 (-Npu, -Vision; -Whisper on = -Whisper auto)
  switches                      no value after them: -NoMtp (or --nomtp), -NoStream, -AttnF32 (pi-arc also takes
                                --no-mtp, --no-stream)
  paths                         in quotes when they have spaces: -Model "C:\My models\Qwen3.5-4B-UD-Q4_K_XL.gguf".
                                Never end a quoted path with a backslash: "C:\folder\" turns the closing quote into
                                part of the path; write "C:\folder". Forward slashes work too: C:/models/x.gguf.
  what each path is             -Model       a .gguf FILE, not its folder (a model in parts: its -00001-of-0000N part)
                                -Whisper     a FOLDER that holds model.bin and tokenizer.json (faster-whisper), e.g.
                                             %USERPROFILE%\.cache\huggingface\hub\models--Systran--faster-whisper-large-v3\snapshots\<hash>
                                -Asr         a Qwen3-ASR .gguf FILE (its mmproj-*.gguf next to it)
                                -SessionDir  a folder (made when it is missing)
                                Usually none is needed: shortcuts (-Model 35b) and auto find them in the Hugging Face
                                cache, %USERPROFILE%\.arcinferzero\models and the folders in model-dirs.txt.

WHEN IT FAILS (the message, then what to do)
  -Whisper: no folder / holds no model.bin / is a file
                                give the faster-whisper folder itself (the one with model.bin in it), or auto
  -Asr: is a folder / is no .gguf file
                                give the .gguf file, e.g. ...\Qwen3-ASR-1.7B-Q8_0.gguf, or auto
  -Npu: expected on, off (got 'x')   and the like: the value after the option is not one it takes
  unknown option -X: did you mean -Y?
                                arc-server has no option X (a typo, or a bin\serverz.exe option that arc-server does
                                not pass on): the closest one is named; the list is above
  'x' is a value without an option before it
                                a value in the wrong place, or a path with spaces outside quotes
  Cannot convert value "x" to type "System.Int32"
                                the option wants a number: -Slots 2, -Chunk 4096
  'x' matches no model / matches N models
                                the list printed with it has every id and shortcut: use one of them
  GPU memory estimate: ... may not fit
                                a warning only (the start goes on): the context asked, with speech to text and
                                slots, looks larger than the GPU memory. A smaller -Ctx, -Vision off or -Ctx auto;
                                check first with -DryRun
  -Preset: no preset 'x'        arc-server presets lists them; save one with save-preset -Preset x [options]
  -Drafter dflash: no DFlash2 drafter was found
                                give the drafter's folder: -Drafter <folder>, or take -Drafter auto / mtp
  The server stopped while loading. Last log lines: ...   (exit code 2)
                                the server's own message, e.g. "-m: no file ...", "--whisper needs a value",
                                "unknown option --wisper (did you mean --whisper ...?)", "... has a quote in it"
                                (a path ending in a backslash), or the GPU memory: a smaller -Ctx, or -Vision off
  The server did not become ready within 10 minutes   (exit code 3)
                                the log (arc-server logs, or $log) says how far it got; the first start of a MoE
                                model writes its expert file (35B ~2.5 min, Flash Next ~9.5 min)
  Port N is used by <program>   (exit code 4) another program listens there: stop it, or -Port <another>
  anything else                 arc-server doctor checks the exes, the driver, the GPU, the port and other GPU
                                programs; paste its report into a GitHub issue

EXIT CODES (for scripts): 0 ok, 1 a wrong option or value, 2 the server stopped while loading, 3 not ready within
  10 minutes, 4 the port is used by another program.


A window for all of this: arc-launcher.cmd in the project folder (models with their memory and automatic
context, every option filled in with the best value (speech to text too) and whether it fits the GPU memory,
presets, Start / Stop, the running server's memory, speeds and live log, a tray icon; More: speech to text only,
Copy the start command, Record, a doctor report, the update check).

The server is started as bin\serverz.exe with --ctx, --mtp (when the model has
an MTP block or a Gemma drafter), --draft for the -nodf / -df variants, --think-budget 8192 and the options above;
its other options (bin\serverz.exe --help) keep their defaults.

Log: $log
Everything (Pi launcher, server options, files): pi-arc --help
"@
  # help <topic>: a part of the full text (its sections start at these lines)
  $topic = "$Model".Trim().ToLower()
  # (a heading at the start of its line: the first lines name WRITING OPTIONS and WHEN IT FAILS too)
  $at = { param($s) $mt = [regex]::Match($full, '(?m)^' + [regex]::Escape($s)); if ($mt.Success) { $mt.Index } else { $full.Length } }
  $iOpt = & $at "Options for start / restart:"; $iWr = & $at "WRITING OPTIONS"; $iErr = & $at "WHEN IT FAILS"; $iEnd = & $at "A window for all of this"
  switch ($topic) {
    "" { Write-Host $short }
    { $_ -in "options", "option", "opts" } { Write-Host $full.Substring($iOpt, $iWr - $iOpt).TrimEnd() }
    { $_ -in "writing", "paths", "values" } { Write-Host $full.Substring($iWr, $iErr - $iWr).TrimEnd() }
    { $_ -in "errors", "error", "fails" } { Write-Host $full.Substring($iErr, $iEnd - $iErr).TrimEnd() }
    { $_ -in "all", "full" } { Write-Host $full }
    default { throw "arc-server help: topics are options, writing, errors, all (got '$topic')" }
  }
  return
}

# ---- the commands that start nothing: list, logs, doctor, version, pull, presets, save-preset
function Out-Json($o) { ConvertTo-Json -InputObject $o -Depth 8 }
function Get-ModelsListed { @(& $exe --list-models | ConvertFrom-Json | ForEach-Object { $_ }) }  # (with the auto context)
function Get-Version {
  $pb = Join-Path $root "bin\PREBUILT.txt"
  $v = [ordered]@{ version = ""; commit = ""; built = ""; prebuilt = (Test-Path $pb); folder = $root }
  if ($v.prebuilt) {
    $l = (Get-Content $pb -TotalCount 1).Trim()
    if ($l -match '^arcinferzero (\S+) \(built ([^,]+), commit (\w+)\)') { $v.version = $Matches[1]; $v.built = $Matches[2]; $v.commit = $Matches[3] }
  } else {
    try {
      $c = (& git -C $root rev-parse --short HEAD 2>$null)
      if ($c) { $v.commit = "$c".Trim(); $v.version = "git " + $v.commit; $v.built = "built from source" }
    } catch {}
    if (-not $v.version) { $v.version = "unknown (no bin\PREBUILT.txt, no git)" }
  }
  $e = Join-Path $root "bin\serverz.exe"
  $v.serverz = if (Test-Path $e) { (Get-Item $e).LastWriteTime.ToString("yyyy-MM-dd HH:mm") } else { "missing" }
  return $v
}
# the latest release on GitHub against this package: up to date when its zip is this commit (or, in a git clone, an
# ancestor of HEAD), newer when it came out after this package was built
function Get-UpdateInfo($v) {
  $r = Invoke-RestMethod "https://api.github.com/repos/rizagp/arcinferzero/releases/latest" -TimeoutSec 10 -Headers @{ "User-Agent" = "arcinferzero-arc-server" }
  $names = @($r.assets | ForEach-Object { $_.name })
  $relCommit = ""
  foreach ($n in $names) { if ($n -match '-([0-9a-f]{7,})-win64\.zip$') { $relCommit = $Matches[1] } }
  $newer = $true
  if ($v.commit -and $relCommit -and ($relCommit.StartsWith($v.commit) -or $v.commit.StartsWith($relCommit))) { $newer = $false }
  elseif (-not $v.prebuilt -and $relCommit) {
    & git -C $root merge-base --is-ancestor $relCommit HEAD 2>$null
    if ($LASTEXITCODE -eq 0) { $newer = $false }
  } elseif ($v.built -match '^\d{4}-\d\d-\d\d') {
    try { $newer = [datetime]$r.published_at -gt [datetime]$v.built.Substring(0, 10) } catch {}
  }
  return [ordered]@{ latest = $r.tag_name; published = $r.published_at; url = $r.html_url; commit = $relCommit; newer = $newer }
}
switch ($Action) {
  { $_ -in "list", "ls" } {
    if (-not (Test-Path $exe)) { throw "$exe not found: unpack the ready package from https://github.com/rizagp/arcinferzero/releases (or build it: setup.cmd)" }
    $all = Get-ModelsListed
    if ($Json) { Out-Json $all; return }
    $run = Get-Running
    $ok = @($all | Where-Object { $_.supported })
    Write-Host ("{0,-38} {1,-12} {2,8} {3,12}  {4,-4} {5,-6} {6}" -f "id", "shortcut", "file", "auto ctx", "MTP", "images", "")
    foreach ($m in $ok) {
      Write-Host ("{0,-38} {1,-12} {2,8} {3,12}  {4,-4} {5,-6} {6}" -f $m.alias, $(if ($m.short) { $m.short } else { "-" }), ("{0:N1} GB" -f ($m.bytes / 1e9)),
        $(if ($m.ctx) { "{0:N0}" -f $m.ctx } else { "-" }), $(if ($m.mtp) { "yes" } else { "-" }), $(if ($m.vision) { "yes" } else { "-" }),
        $(if ($run -and $m.alias -eq $run) { "<- running" } else { "" }))
    }
    $skip = @($all | Where-Object { -not $_.supported })
    if ($skip.Count) { Write-Host ("not supported: " + (($skip | ForEach-Object { "$($_.file) [$($_.arch)]" }) -join ", ")) }
    Write-Host "$($ok.Count) models / variants. Start one: arc-server start -Model <id or shortcut>"
    return
  }
  { $_ -in "logs", "log" } {
    if (-not (Test-Path $log)) { Say "No log yet ($log): the server has not been started."; return }
    if ($Follow) { Get-Content -LiteralPath $log -Tail ([math]::Max(1, $Lines)) -Wait }
    else { Get-Content -LiteralPath $log -Tail ([math]::Max(1, $Lines)) }
    return
  }
  { $_ -in "version", "--version" } {
    $v = Get-Version
    if ($Check) { try { $v.update = Get-UpdateInfo $v } catch { $v.update = [ordered]@{ error = $_.Exception.Message } } }
    if ($Json) { Out-Json $v; return }
    Write-Host "arcinferzero $($v.version)$(if ($v.commit -and $v.version -notmatch $v.commit) { ", commit $($v.commit)" })$(if ($v.built) { " ($($v.built))" })"
    Write-Host "  folder: $($v.folder)   bin\serverz.exe: $($v.serverz)"
    if ($v.update) {
      if ($v.update.error) { Write-Host "  could not ask GitHub: $($v.update.error)" }
      elseif ($v.update.newer) { Write-Host "  a newer package is out: $($v.update.latest) ($($v.update.published)): $($v.update.url)" -ForegroundColor Yellow }
      else { Write-Host "  up to date (latest release $($v.update.latest))" }
    }
    return
  }
  { $_ -in "pull", "get" } {
    . (Join-Path $PSScriptRoot "model-catalog.ps1")
    $keys = @("$Model" -split '[\s,;]+' | Where-Object { $_ })
    if (-not $keys.Count) {
      $hub = Get-HfHub
      Write-Host ("{0,-8} {1,7}  {2,-6} {3}" -f "key", "size", "", "model")
      foreach ($m in $ArcModels) { Write-Host ("{0,-8} {1,7}  {2,-6} {3}" -f $m.key, ("{0:N1} GB" -f $m.gb), $(if (Test-ArcModel $m $hub) { "have" } else { "" }), $m.name) }
      Write-Host "Download: arc-server pull <key> (several: pull 4b,asr). A cut download goes on where it stopped."
      return
    }
    $bad = @($keys | Where-Object { $k = $_.ToLower(); -not ($ArcModels | Where-Object { $_.key -eq $k }) })
    if ($bad.Count) { throw "pull: unknown model $($bad -join ', ') (the keys: $(($ArcModels | ForEach-Object { $_.key }) -join ', '))" }
    & (Join-Path $PSScriptRoot "get-models.ps1") @keys
    $rc = $LASTEXITCODE
    if ($rc -ne 0) { Write-Host "pull: $rc model(s) not finished; run it again to go on where it stopped." -ForegroundColor Red; exit 1 }
    Write-Host "Done. Start it: arc-server start -Model $($keys[0])"
    return
  }
  "presets" {
    $ps = @(Get-ChildItem -LiteralPath $presetDir -Filter *.json -ErrorAction SilentlyContinue | Sort-Object Name)
    $out = [ordered]@{}
    foreach ($f in $ps) { try { $out[$f.BaseName] = Get-Content -LiteralPath $f.FullName -Raw | ConvertFrom-Json } catch { $out[$f.BaseName] = "not valid JSON" } }
    if ($Json) { Out-Json $out; return }
    if (-not $ps.Count) { Write-Host "No presets yet. Save one: arc-server save-preset -Preset <name> -Model 35b -Ctx 65536 ..."; return }
    foreach ($k in $out.Keys) {
      $o = $out[$k]
      $t = if ($o -is [string]) { $o } else { ($o.PSObject.Properties | ForEach-Object { if ($_.Value -is [bool]) { if ($_.Value) { "-$($_.Name)" } } else { "-$($_.Name) $($_.Value)" } }) -join " " }
      Write-Host ("  {0,-20} {1}" -f $k, $t)
    }
    Write-Host "Use one: arc-server start -Preset <name>   (folder: $presetDir)"
    return
  }
  "save-preset" {
    if (-not $Preset) { throw "save-preset: give its name: arc-server save-preset -Preset <name> -Model ... [options]" }
    $pp = Preset-Path $Preset
    $o = [ordered]@{}
    foreach ($k in $presetKeys) {
      if (-not $PSBoundParameters.ContainsKey($k)) { continue }
      $v = $PSBoundParameters[$k]
      if ($v -is [switch]) { $v = [bool]$v }
      $o[$k] = $v
    }
    if (-not $o.Count) { throw "save-preset: no options given to save (e.g. -Model 35b -Ctx 65536 -Npu on)" }
    New-Item -ItemType Directory -Force $presetDir | Out-Null
    $existed = Test-Path -LiteralPath $pp
    [IO.File]::WriteAllText($pp, (Out-Json $o))
    Say "Preset '$Preset' $(if ($existed) { 'replaced' } else { 'saved' }): $pp"
    Say "  start it: arc-server start -Preset `"$Preset`""
    return
  }
  "doctor" {
    $rep = New-Object System.Collections.Generic.List[object]
    function Add-Check([string]$what, [string]$state, [string]$text) { $rep.Add([pscustomobject]@{ check = $what; state = $state; text = $text }) }
    $v = Get-Version
    Add-Check "package" "ok" "arcinferzero $($v.version)$(if ($v.commit) { ", commit $($v.commit)" }), folder $root"
    try { $os = Get-CimInstance Win32_OperatingSystem; Add-Check "windows" "ok" "$($os.Caption) build $($os.BuildNumber), $([math]::Round($os.TotalVisibleMemorySize / 1MB)) GB RAM ($([math]::Round($os.FreePhysicalMemory / 1MB, 1)) GB free)" } catch {}
    Add-Check "powershell" "ok" "$($PSVersionTable.PSEdition) $($PSVersionTable.PSVersion)"
    foreach ($n in "serverz", "runz", "transcribez", "arc_recorder", "clinfo") {
      $p = Join-Path $root "bin\$n.exe"
      if (Test-Path $p) {
        $blocked = $false
        try { $blocked = [bool](Get-Item -LiteralPath $p -Stream Zone.Identifier -ErrorAction Stop) } catch {}
        Add-Check "bin\$n.exe" $(if ($blocked) { "warn" } else { "ok" }) ("{0:yyyy-MM-dd HH:mm}" -f (Get-Item $p).LastWriteTime + $(if ($blocked) { " (marked as downloaded: run arc-launcher.cmd once, or Unblock-File)" } else { "" }))
      } elseif ($n -in "serverz", "clinfo") {
        Add-Check "bin\$n.exe" "FAIL" "missing: a copy from git has no exes; take the package from https://github.com/rizagp/arcinferzero/releases or run setup.cmd"
      }
    }
    try {
      $gpus = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "Intel" })
      foreach ($g in $gpus) { Add-Check "gpu" "ok" "$($g.Name), driver $($g.DriverVersion) ($("{0:yyyy-MM-dd}" -f $g.DriverDate))" }
      if (-not $gpus.Count) { Add-Check "gpu" "FAIL" "no Intel GPU in Windows' list of display adapters" }
    } catch {}
    $ci = Join-Path $root "bin\clinfo.exe"
    if (Test-Path $ci) {
      $co = @(& $ci 2>&1 | ForEach-Object { "$_" })
      $devs = @($co | Where-Object { $_ -match '^\s+device: ' } | ForEach-Object { ($_ -replace '^\s+device: ', '').Trim() })
      if ($LASTEXITCODE -eq 0 -and $devs.Count) {
        $mem = @($co | Where-Object { $_ -match 'global mem: ([\d.]+ GiB)' } | ForEach-Object { $Matches[1] })
        $xmx = @($co | Where-Object { $_ -match 'cl_intel_subgroup_matrix_multiply_accumulate\s+(\S+)' } | ForEach-Object { $Matches[1] })
        $sg = @($co | Where-Object { $_ -match 'sub-group sizes: (.*)$' } | ForEach-Object { $Matches[1].Trim() })
        for ($i = 0; $i -lt $devs.Count; $i++) {
          Add-Check "opencl" "ok" ("{0}: GPU memory {1}, sub-groups {2}, XMX (DPAS) {3}" -f $devs[$i], $(if ($i -lt $mem.Count) { $mem[$i] } else { "?" }), $(if ($i -lt $sg.Count) { $sg[$i] } else { "?" }), $(if ($i -lt $xmx.Count -and $xmx[$i] -eq "yes") { "yes" } else { "no" }))
        }
      } else { Add-Check "opencl" "FAIL" "the GPU driver offers no OpenCL device: install Intel's graphics driver and restart" }
    }
    try {
      # ("NPU" as a whole word: "Microsoft Input Configuration Device" holds it too)
      # (not $npu: that is -Npu, a string, case-insensitively)
      $npuDevs = @(Get-PnpDevice -PresentOnly -ErrorAction Stop | Where-Object { $_.Class -eq "ComputeAccelerator" -or "$($_.FriendlyName)" -match 'AI Boost|\bNPU\b' })
      if ($npuDevs.Count) { foreach ($dev in $npuDevs) { Add-Check "npu" $(if ("$($dev.Status)" -eq "OK") { "ok" } else { "warn" }) "$($dev.FriendlyName): $($dev.Status)" } }
      else { Add-Check "npu" "info" "none found (only -Npu on needs one)" }
    } catch {}
    try {
      . (Join-Path $PSScriptRoot "model-catalog.ps1")
      $hub = Get-HfHub
      foreach ($dir in @($hub, $stateDir) | Select-Object -Unique) {
        $q = (Split-Path $dir -Qualifier)
        $dr = Get-PSDrive $q.TrimEnd(":") -ErrorAction SilentlyContinue
        if ($dr) { Add-Check "disk" $(if ($dr.Free -lt 20GB) { "warn" } else { "ok" }) ("{0} ({1}): {2:N0} GB free" -f $dir, $q, ($dr.Free / 1GB)) }
      }
    } catch {}
    try { $t = Join-Path $stateDir "doctor.tmp"; Set-Content $t "x"; Remove-Item $t; Add-Check "data folder" "ok" "$stateDir (writable)" } catch { Add-Check "data folder" "FAIL" "$stateDir is not writable: $($_.Exception.Message)" }
    $run = Get-Running
    if ($run) {
      $pr = $null; try { $pr = Invoke-RestMethod "$base/props" -TimeoutSec 3 } catch {}
      $mt = if ($pr -and $pr.memory) { ", GPU memory {0:N1} of {1:N1} GiB" -f ($pr.memory.total / 1GB), ($pr.memory.device / 1GB) } else { "" }
      Add-Check "server" "ok" "running on $base, model $run$(if ($pr) { ", context $($pr.n_ctx)" })$mt"
    } else {
      Add-Check "server" "info" "not running on port $Port"
      try {
        $lis = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop)
        foreach ($c in $lis) { $pn = (Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue).ProcessName; Add-Check "port" "FAIL" "port $Port is used by $pn (pid $($c.OwningProcess)): stop it, or -Port <another>" }
      } catch {}
    }
    $procs = @(Get-CimInstance Win32_Process -Filter "Name = 'serverz.exe' OR Name = 'runz.exe' OR Name = 'llama-server.exe' OR Name = 'ollama.exe'" -ErrorAction SilentlyContinue)
    foreach ($p in $procs) { Add-Check "process" $(if ($p.Name -eq "serverz.exe") { "info" } else { "warn" }) "$($p.Name) pid $($p.ProcessId): $($p.CommandLine)" }
    if ($procs.Count -gt 1) { Add-Check "process" "warn" "$($procs.Count) GPU programs run at once: they share the GPU memory (a cause of device-lost errors)" }
    $onPath = [bool](Get-Command arc-server -ErrorAction SilentlyContinue)
    Add-Check "commands" $(if ($onPath) { "ok" } else { "info" }) $(if ($onPath) { "arc-server is on PATH" } else { "arc-server is not on PATH: run it as $root\arc-server.cmd (setup offers to add the folder)" })
    if (Test-Path $log) {
      $errs = @(Get-Content -LiteralPath $log -Tail 400 -ErrorAction SilentlyContinue | Where-Object { $_ -match '(?i)error|device lost|failed|out of memory' } | Select-Object -Last 5)
      foreach ($e in $errs) { Add-Check "log" "warn" $e }
    }
    if ($Json) { Out-Json $rep.ToArray(); return }  # (@($rep) on a List: "Argument types do not match" in PowerShell 5.1)
    Write-Host "arcinferzero doctor ($(Get-Date -Format 'yyyy-MM-dd HH:mm')): paste this into a GitHub issue"
    foreach ($r in $rep) {
      $c = switch ($r.state) { "FAIL" { "Red" } "warn" { "Yellow" } default { "Gray" } }
      Write-Host ("  {0,-5} {1,-14} {2}" -f $r.state, $r.check, $r.text) -ForegroundColor $c
    }
    if (@($rep | Where-Object { $_.state -eq "FAIL" }).Count) { exit 1 }
    return
  }
}

switch ($Action) {
  "status" {
    $id = Get-Running
    if ($Json) {
      $o = [ordered]@{ running = [bool]$id; base = $base; port = $Port; model = $id; props = $null; state = $null }
      if ($id) { try { $o.props = Invoke-RestMethod "$base/props" -TimeoutSec 2 } catch {} }
      if ($id -and (Test-Path $stateFile)) { try { $o.state = Get-Content $stateFile -Raw | ConvertFrom-Json } catch {} }
      $files = @(Get-ChildItem $sessDir -Recurse -Filter *.arcs -ErrorAction SilentlyContinue)
      $o.saved_conversations = [ordered]@{ count = $files.Count; bytes = [double](($files | Measure-Object Length -Sum).Sum); folder = $sessDir }
      Out-Json $o
      return
    }
    if ($id) {
      $p = Invoke-RestMethod "$base/props" -TimeoutSec 2
      Say "arcinferzero server running on $base : model $id"
      if (Test-Path $lastFile) { Say ("  " + (Get-Content $lastFile -Raw).Trim()) }  # (also after switching in Pi)
      Say ("  " + ($p | ConvertTo-Json -Compress -Depth 3))
      if ((Test-Path $stateFile) -and "$((Get-Content $stateFile -Raw | ConvertFrom-Json).extra)" -match "--npu on") {
        Say "  NPU: on (-Npu on; kept when the model changes and on a restart, off after stop)"
      }
      if ((Test-Path $stateFile) -and "$((Get-Content $stateFile -Raw | ConvertFrom-Json).extra)" -match "--precise") {
        Say "  precise mode: on (-Precise; Flash Next; kept like -Npu, -NoPrecise turns it off)"
      }
      if ((Test-Path $stateFile) -and ($c = (Get-Content $stateFile -Raw | ConvertFrom-Json).cmd)) { Say "  command: $c" }
    } else { Say "arcinferzero server is not running." }
    $files = @(Get-ChildItem $sessDir -Recurse -Filter *.arcs -ErrorAction SilentlyContinue)
    Say ("Saved conversations: {0} ({1:N1} GiB) in {2}" -f $files.Count, (($files | Measure-Object Length -Sum).Sum / 1GB), $sessDir)
    return
  }
  "clear-cache" {
    $n = 0; $bytes = 0
    if (Get-Running) {  # the running server forgets its saved conversations and deletes their files
      try { $r = Invoke-RestMethod -Method Delete "$base/sessions" -TimeoutSec 600; $n += $r.deleted; $bytes += $r.freed_bytes } catch {}
    }
    $files = @(Get-ChildItem $sessDir -Recurse -Filter *.arcs -ErrorAction SilentlyContinue)  # other models'
    $n += $files.Count; $bytes += ($files | Measure-Object Length -Sum).Sum
    $files | ForEach-Object { [IO.File]::Delete($_.FullName) }
    Say ("Deleted {0} saved conversations ({1:N1} GiB)." -f $n, ($bytes / 1GB))
    return
  }
  "stop" { if (-not (Stop-Arc)) { Say "arcinferzero server is not running." }; return }
  "record" {
    $rec = Join-Path $root "bin\arc_recorder.exe"
    if (-not (Test-Path $rec)) { throw "$rec not found: build it with .\build.ps1 tools\arc_recorder.cpp (or run setup)" }
    $a = @("start")
    if ($App) { $a += "--app", ('"' + $App.Replace('"', '') + '"') }  # (quoted: Start-Process splits a name with a space)
    if ($NoMic) { $a += "--no-mic" }
    if ($Title) { $a += "--title", ('"' + $Title.Replace('"', "'") + '"') }
    $log = Join-Path $stateDir "recorder.log"
    $p = Start-Process $rec -ArgumentList $a -WindowStyle Hidden -PassThru -RedirectStandardError $log
    Start-Sleep -Milliseconds 1500
    if ($p.HasExited) { Get-Content $log | ForEach-Object { Say $_ }; throw "the recorder did not start" }
    Get-Content $log | ForEach-Object { Say $_ }
    Say "arc-server record-stop ends it."
    return
  }
  "record-stop" { & (Join-Path $root "bin\arc_recorder.exe") stop; return }
  "record-status" { & (Join-Path $root "bin\arc_recorder.exe") status; return }
}

# ---- start / restart
if (-not (Test-Path $exe)) { throw "$exe not found: build it with .\build.ps1 zero\serverz.cpp (or run setup)" }
$running = Get-Running
$st = if ($running -and (Test-Path $stateFile)) { Get-Content $stateFile -Raw | ConvertFrom-Json } else { $null }
$models = Get-Models
# -Model none: speech to text only (Qwen3-ASR; -Whisper auto for Whisper too), a chat model loaded later by the first
# chat request naming one (the web UI's model menu)
$none = [pscustomobject]@{ name = "speech to text only"; file = ""; path = "none"; alias = "none"; mtp = $false; ctx = 0 }
if ($Model -eq "none" -or (-not $Model -and $Action -eq "restart" -and $running -eq "none")) {
  $m = $none
} elseif (-not $Model -and $Action -eq "restart" -and $running) {
  # (restart without -Model: the running model again, with the options given; no menu)
  $m = $models | Where-Object { $_.alias -eq $running } | Select-Object -First 1
  if (-not $m -and $st) { $m = $models | Where-Object { Same-Path $_.path $st.path } | Select-Object -First 1 }
  if (-not $m) { throw "the running model $running is not in the model list: arc-server restart -Model <id>" }
} elseif (-not $Model) {
  # (no one to type a number: a script, a scheduled task or a tool with its input redirected, or -NonInteractive. The
  #  menu asked again on every empty answer, at full CPU, forever: now the running server stays, else the model used
  #  last starts, as Enter / pi-arc do)
  $noConsole = [Console]::IsInputRedirected -or (@([Environment]::GetCommandLineArgs()) -match '^-noni').Count -gt 0
  if ($noConsole) {
    if ($running) { Say "No -Model given: keeping $running."; return }
    $lp = if (Test-Path $lastFile) { (Get-Content $lastFile -Raw).Trim() } else { "" }
    $m = if ($lp) { $models | Where-Object { $_.supported -and (Same-Path $_.path $lp) } | Select-Object -First 1 } else { $null }
    if (-not $m) { throw "No -Model given and no console to choose one in: arc-server start -Model <shortcut, id or file> (run 'arc-server start' in a console for the list)" }
    Say "No -Model given: starting the model used last, $($m.alias)"
  } else {
    $m = Pick-Model $models $(if ($st) { $st.path } else { $null }) $running
    if (-not $m) { Say "Keeping $running."; return }
  }
} elseif (Test-Path $Model -PathType Leaf) {
  $full = (Resolve-Path $Model).Path
  $m = $models | Where-Object { Same-Path $_.path $full } | Select-Object -First 1
  if (-not $m) { $m = [pscustomobject]@{ name = $full; file = Split-Path $full -Leaf; path = $full; mtp = $true; alias = [IO.Path]::GetFileNameWithoutExtension($full).ToLower(); ctx = 0 } }
} else {
  $m = $models | Where-Object { $_.supported -and ($_.short -eq $Model.ToLower() -or $_.alias -eq $Model.ToLower()) } | Select-Object -First 1
  if (-not $m) {
    $hits = @($models | Where-Object { $_.supported -and ($_.name + " " + $_.file) -like "*$Model*" })
    # (the variants of one file all match its name: then the file's main entry, the one listed first)
    $files = @($hits | ForEach-Object { $_.path } | Select-Object -Unique)
    if ($files.Count -eq 1) { $hits = @($hits[0]) }
    if ($hits.Count -ne 1) {
      $pool = if ($hits.Count) { $hits } else { @($models | Where-Object { $_.supported }) }
      $list = ($pool | ForEach-Object { "  {0,-34} {1,-14} {2}" -f $_.alias, $(if ($_.short) { $_.short } else { "-" }), $_.file }) -join "`n"
      $what = if ($hits.Count) { "matches $($hits.Count) models" } else { "matches no model" }
      throw "'$Model' $what. Use an id or a shortcut:`n  id                                 shortcut       file`n$list"
    }
    $m = $hits[0]
  }
}
# -Vision off / -NoMtp: the variant of the same file without the image projector / the MTP block (every
# combination a file supports is listed by the server: <id>, <id>-mtp, <id>-vision, <id>-std)
if (($Vision -eq "off" -and $m.vision) -or ($NoMtp -and $m.mtp) -or ($Vision -eq "on" -and -not $m.vision)) {
  $wantVision = $(if ($Vision -eq "off") { $false } elseif ($Vision -eq "on") { $true } else { [bool]$m.vision })
  $wantMtp = [bool]$m.mtp -and -not $NoMtp
  $v = $models | Where-Object { $_.supported -and (Same-Path $_.path $m.path) -and [bool]$_.vision -eq $wantVision -and [bool]$_.mtp -eq $wantMtp } | Select-Object -First 1
  if ($v) { $m = $v }
}
# -Drafter mtp / dflash / auto: the variant of the same file (and the same images / MTP choice) with that drafter
if ($drafterPick -and $m -ne $none -and $m.mtp) {
  $v = $models | Where-Object { $_.supported -and (Same-Path $_.path $m.path) -and [bool]$_.vision -eq [bool]$m.vision -and $_.mtp -and "$($_.draft)" -eq $drafterPick } | Select-Object -First 1
  if ($v) { $m = $v }
  elseif ($drafterPick -eq "dflash") { throw "-Drafter dflash: no DFlash2 drafter was found for $($m.alias); give its folder: -Drafter <folder>" }
  # (mtp / auto on a model without a DFlash2 drafter: its MTP block or Gemma drafter is the only one, as it is)
}
# a restart (another model, other options) keeps the running server's slots unless -Slots is given: pi-arc on
# another model dropped them (and the second Pi / client then waited for the only slot)
if (-not $PSBoundParameters.ContainsKey("Slots") -and $st -and "$($st.extra)" -match "--slots (\d+)") { $Slots = [int]$Matches[1] }
# -Npu on: the NPU next to the GPU for prompts (serverz --npu on). A start on another model (pi-arc, Pi's /model
# inside the server) and a restart keep a running server's setting (-Npu off turns it off); a start after stop is off
# unless -Npu on is given (a restart dropped it: the 27B lost the NPU when -Whisper was added)
if (-not $Npu -and $Action -in "start", "restart" -and $st -and "$($st.extra)" -match "--npu on") { $Npu = "on" }
# -Precise: Flash Next's precise mode (serverz --precise), kept the same way (-NoPrecise turns it off)
if (-not $Precise -and -not $NoPrecise -and $Action -in "start", "restart" -and $st -and "$($st.extra)" -match "--precise") { $Precise = [switch]$true }
if ($NoPrecise) { $Precise = [switch]$false }
# -Whisper: kept the same way (with its device; -Whisper off turns it off)
if (-not $Whisper -and $Action -in "start", "restart" -and $st -and "$($st.extra)" -match '--whisper ("[^"]*"|[^\s|]+)') {
  $Whisper = $Matches[1].Trim('"')
  # ([^\s|]: the state's extra ends "...|think N", which "npu|think" did not match as npu: Whisper went back to the GPU)
  if (-not $WhisperDevice -and "$($st.extra)" -match "--whisper-device ([^\s|]+)") { $WhisperDevice = $Matches[1] }
  if ($WhisperMaxMinutes -le 0 -and "$($st.extra)" -match "--whisper-max-minutes ([0-9]+)") { $WhisperMaxMinutes = [int]$Matches[1] }
}
# -Asr: kept the same way, with or without Whisper (without it: Qwen3-ASR alone); -Asr off turns it off
if (-not $Asr -and $Action -in "start", "restart" -and $st -and "$($st.extra)" -match '--asr ("[^"]*"|[^\s|]+)') { $Asr = $Matches[1].Trim('"') }
if ($AsrSlots -le 0 -and $Action -in "start", "restart" -and $st -and "$($st.extra)" -match "--asr-slots ([0-9]+)") { $AsrSlots = [int]$Matches[1] }
if ($AsrSlots -lt 0 -or $AsrSlots -gt 4) { throw "-AsrSlots: expected 1 to 4 (got $AsrSlots)" }
$ctxArg = if ($Ctx) { $Ctx } elseif ($Slots -gt 1) { "auto" } elseif ($m.ctx) { "$($m.ctx)" } else { "auto" }  # (the listed context is for one slot)
$novision = $(if ($m.vision -eq $false) { "1" } else { "0" })
# the other options, as server arguments (also kept in the state file: a change restarts the server)
$extra = @()
if ($MtpDrafts -gt 0) { $extra += @("--mtp-drafts", $MtpDrafts) }
# -DFlash <folder>: a DFlash2 drafter of your own (the folder with config.json and model.safetensors), e.g. for a
# fine-tuned variant the model list does not pair with one; -DFlash off: no drafter
if ($DFlash -eq "off") { $extra += "--no-dflash" }
elseif ($DFlash) {
  # (an absolute path: the server starts in the arcinferzero folder, where a relative one would point elsewhere; and not
  #  with -NoMtp, since a drafter turns speculative decoding on. Self audit 2026-11-08)
  if ($NoMtp) { throw "-DFlash with -NoMtp: a drafter means speculative decoding (use -DFlash off for none)" }
  if (-not (Test-Path -LiteralPath (Join-Path $DFlash "config.json"))) { throw "-DFlash: no config.json in $DFlash (give the drafter's snapshot folder)" }
  $DFlash = (Resolve-Path -LiteralPath $DFlash).Path
  $extra += @("--dflash", "`"$($DFlash.TrimEnd('\'))`"")
}
# -DriverReserveGb N: GPU memory left to the driver (default 0.5 GiB; kept with the server like the others)
if ($DriverReserveGb -ge 0) {
  if ($DriverReserveGb -gt 8) { throw "-DriverReserveGb: 0 to 8 GiB (got $DriverReserveGb)" }
  $extra += @("--driver-reserve-gb", $DriverReserveGb.ToString([Globalization.CultureInfo]::InvariantCulture))
}
if ($Slots -gt 1) { $extra += @("--slots", $Slots) }
if ($Npu -eq "on") { $extra += @("--npu", "on") }
if ($Precise) { $extra += "--precise" }
if ($Whisper -and $Whisper -ne "off") {
  $extra += @("--whisper", $(if ($Whisper -eq "auto") { "auto" } else { "`"$Whisper`"" }))
  if ($WhisperDevice -eq "npu") { $extra += @("--whisper-device", "npu") }
}
if ($Asr -and ($Asr -ne "off" -or ($Whisper -and $Whisper -ne "off"))) { $extra += @("--asr", $(if ($Asr -in "auto", "off") { $Asr } else { "`"$Asr`"" })) }
if (($Whisper -and $Whisper -ne "off") -or ($Asr -and $Asr -ne "off")) {
  if ($WhisperMaxMinutes -gt 0) { $extra += @("--whisper-max-minutes", $WhisperMaxMinutes) }
  if ($AsrSlots -gt 1) { $extra += @("--asr-slots", $AsrSlots) }
}
if ($ImageMinTokens -gt 0) { $extra += @("--image-min-tokens", $ImageMinTokens) }
if ($ImageMaxTokens -gt 0) { $extra += @("--image-max-tokens", $ImageMaxTokens) }
if ($BudgetGb -gt 0) { $extra += @("--budget", $BudgetGb) }
if ($AttnF32) { $extra += "--attn-f32" }
if ($PreserveThinking) { $extra += "--preserve-thinking" }
if ($NoPrefixCache) { $extra += "--no-prefix-cache" }
if ($PrefixCacheGb -ge 0) { $extra += @("--prefix-cache-gb", $PrefixCacheGb) }
if ($NoSwap) { $extra += "--no-swap" }
if ($ListenHost) { $extra += @("--host", $ListenHost) }
if ($SessionDir) { $extra += @("--session-dir", "`"$SessionDir`"") }
if ($NoStream) { $extra += "--no-stream" }  # (in the key: a running server with the SSD cache is restarted without it)
$extraKey = "$($extra -join ' ')|think $ThinkBudget"
# (compared with the running server's only when given: pi-arc's plain start keeps a server started with other options)
$extraGiven = @("MtpDrafts", "DFlash", "DriverReserveGb", "ThinkBudget", "ImageMinTokens", "ImageMaxTokens", "BudgetGb", "AttnF32", "PreserveThinking", "NoPrefixCache",
  "PrefixCacheGb", "NoSwap", "ListenHost", "SessionDir", "Slots", "NoStream", "Npu", "Precise", "NoPrecise", "Whisper", "WhisperDevice", "Asr", "AsrSlots", "WhisperMaxMinutes" | Where-Object { $PSBoundParameters.ContainsKey($_) }).Count -gt 0
# the running server is kept unless an option given now differs from what it runs with (empty / 0 = not given)
$same = $running -eq $m.alias -and (-not $Ctx -or ($st -and "$($st.ctx)" -eq $ctxArg)) -and
  ($Chunk -eq 0 -or ($st -and "$($st.chunk)" -eq "$Chunk")) -and (-not $PSBoundParameters.ContainsKey("SessionCacheGb") -or ($st -and "$($st.sess)" -eq "$SessionCacheGb")) -and
  (-not $extraGiven -or ($st -and "$(if ($st.extra) { $st.extra } else { '|think 8192' })" -eq $extraKey)) -and
  (-not $st -or "$(if ($st.engine) { $st.engine } else { 'opencl' })" -eq $Engine)
$useMtp = [bool]$m.mtp  # (the variant's setting; -NoMtp picked the variant without it)
$srvArgs = @("-m", "`"$($m.path)`"", "--port", $Port, "--ctx", $ctxArg, "--alias", $m.alias, "--think-budget", $ThinkBudget) + $extra
if ($useMtp) { $srvArgs += "--mtp" }
if ($useMtp -and $m.draft -and $m.draft -ne "auto") { $srvArgs += @("--draft", $m.draft) }  # (-nodf: MTP block only, -df: DFlash2 only)
if ($Chunk -gt 0) { $srvArgs += @("--chunk", $Chunk) }
if ($SessionCacheGb -gt 0) { $srvArgs += @("--session-cache-gb", $SessionCacheGb) }  # saved conversations on disk (default 8 GB, 0 = off)
if ($novision -eq "1") { $srvArgs += "--no-mmproj" }
$srvCmd = "`"$exe`" " + ($srvArgs -join " ")  # (issue #6: the server's own command line, also in server.json)

# A GPU memory estimate from the model's --list-models numbers (how --ctx auto counted): the fixed parts (weights,
# MTP block / drafter, working buffers at the default chunk, snapshots, encoders) + the KV cache for the context
# asked (per slot) + speech to text (~8 GiB with Whisper, ~3.7 GiB Qwen3-ASR alone; docs/qwen3-asr.md), against the
# GPU memory less the driver's share. With -Ctx auto the server makes it fit itself. An estimate: another chunk or
# -BudgetGb moves it.
function Get-Estimate {
  $mm = $m.memory
  if (-not $mm -or -not $mm.device) { return $null }
  $keep = if ($DriverReserveGb -ge 0) { $DriverReserveGb * 1GB } else { [double]$mm.keep_free }
  $freeGpu = [double]$mm.device - $keep
  $fixed = [double]$mm.weights + $mm.mtp + $mm.buffers + $mm.snapshots + $mm.encoders
  $speech = if ($Whisper -and $Whisper -ne "off") { 8GB } elseif ($Asr -and $Asr -ne "off") { 3.7GB } else { 0 }
  $o = [ordered]@{ device = [double]$mm.device; driver_reserve = $keep; fixed = $fixed; speech = $speech; ctx = $ctxArg; slots = [math]::Max(1, $Slots) }
  if ($ctxArg -eq "auto") {
    $o.context = [double]$mm.context; $o.need = $fixed + $mm.context; $o.fits = $true
    $o.note = "auto: the server picks the largest context that fits$(if ($speech) { ' (smaller with speech to text on)' })"
  } else {
    $o.context = [double]$ctxArg / 1000 * $mm.context_per_1k * $o.slots
    $o.need = $fixed + $o.context + $speech
    $o.fits = $o.need -le $freeGpu
    $o.note = if ($o.fits) { "fits" } else { "may not fit: a smaller -Ctx, -Vision off or -Ctx auto" }
  }
  $o.free = $freeGpu
  return $o
}
$est = if ($m -ne $none) { Get-Estimate } else { $null }
function Est-Text($e) {
  if (-not $e) { return "no memory estimate for this model" }
  "GPU memory estimate: {0:N1} of {1:N1} GiB (weights and buffers {2:N1}, context {3:N1}{4}){5}: {6}" -f ($e.need / 1GB), ($e.free / 1GB),
  ($e.fixed / 1GB), ($e.context / 1GB), $(if ($e.speech) { ", speech {0:N1}" -f ($e.speech / 1GB) } else { "" }), $(if ($e.slots -gt 1) { ", $($e.slots) slots" } else { "" }), $e.note
}
if ($DryRun) {
  $what = if ($Action -eq "start" -and $same) { "keep the running server" } elseif ($running) { "switch the running server ($running)" } else { "start the server" }
  if ($Json) {
    Out-Json ([ordered]@{ action = $what; model = $m.alias; file = $m.path; port = $Port; ctx = $ctxArg; command = $srvCmd; estimate = $est })
    return
  }
  Say "Dry run: would $what with $($m.alias), context $ctxArg, port $Port"
  Say "  command: $srvCmd"
  Say "  $(Est-Text $est)"
  return
}
if ($Action -eq "start" -and $same) {
  Say "Server already running: $running on $base"
  Say "  engine: $(Engine-Text $Engine ([bool]$m.moe))"
  Sync-PiModels
  return
}
if ($running) {
  $was = if ($st -and $st.engine) { $st.engine } else { "opencl" }  # (a server.json without engine: the old OpenCL one)
  Say "Switching the server: $running ($(Engine-Text $was))"
  Say "                  to: $($m.alias) ($(Engine-Text $Engine ([bool]$m.moe)))"
  Stop-Arc | Out-Null
}
elseif ($Action -eq "restart") { Stop-Arc | Out-Null }

Say "Starting $(if ($NoStream) { 'arcinferzero (no SSD cache)' } else { 'arcinferzero' }): $($m.name) as $($m.alias), context $ctxArg, port $Port$(if ($useMtp) { ', speculative decoding' })"
Say "  engine: $(Engine-Text $Engine ([bool]$m.moe))"
if (-not $NoStream) {
  Say "  (MoE models: the first start writes the expert file to $stateDir\zero, 35B: 16 GB / ~2.5 min, Flash Next: 56 GB / ~9.5 min, once)"
}
Say "  (loading takes 5-40 s; log: $log)"
Say "  command: $srvCmd"
if ($est) {
  if ($est.fits) { Say "  $(Est-Text $est)" }
  elseif (-not $Quiet) { Write-Host "  $(Est-Text $est)" -ForegroundColor Yellow }
}
# (exit 4: another program listens on the port, so the new server could not; ours was stopped above)
try {
  $lis = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop)
  if ($lis.Count) {
    $pn = (Get-Process -Id $lis[0].OwningProcess -ErrorAction SilentlyContinue).ProcessName
    Write-Host "Port $Port is used by $pn (pid $($lis[0].OwningProcess)): stop it, or start with -Port <another>." -ForegroundColor Red
    exit 4
  }
} catch {}
$p = Start-Process -FilePath $exe -ArgumentList $srvArgs -WorkingDirectory $root -WindowStyle Hidden -PassThru `
  -RedirectStandardError $log -RedirectStandardOutput (Join-Path $stateDir "server.out.log")
if ($m -ne $none) { Set-Content $lastFile $m.path -Encoding utf8 }  # (none: the chat model used last stays the default)
@{ pid = $p.Id; engine = $Engine; model = $m.name; path = $m.path; alias = $m.alias; ctx = $ctxArg; chunk = $(if ($Chunk -gt 0) { "$Chunk" } else { "default" }); sess = "$SessionCacheGb"; novision = $novision; extra = $extraKey; port = $Port; cmd = $srvCmd; started = (Get-Date).ToString("s") } |
  ConvertTo-Json | Set-Content $stateFile -Encoding utf8

$t0 = Get-Date
# the server's "progress: P % <step>" lines (src/load_progress.h) as a progress bar while it loads
$barActivity = "Loading $($m.alias)"
$script:lastProgress = $null  # (kept: more than 40 other lines after it would hide it from the tail)
$showBar = -not [Console]::IsOutputRedirected  # (the launcher runs this hidden, its output in a file: its own bar)
function Show-LoadProgress {
  if (-not $showBar) { return }
  $line = Get-Content $log -Tail 40 -ErrorAction SilentlyContinue | Where-Object { $_ -match '^progress: ' } | Select-Object -Last 1
  if ($line) { $script:lastProgress = $line } else { $line = $script:lastProgress }
  $secs = [int]((Get-Date) - $t0).TotalSeconds
  if ($line -match '^progress: (-?\d+) % (.*)$') {
    $pct = [int]$Matches[1]
    if ($pct -ge 0) { Write-Progress -Activity $barActivity -Status "$($Matches[2])  ($pct %, $secs s)" -PercentComplete ([math]::Min(100, $pct)) }
    else { Write-Progress -Activity $barActivity -Status "$($Matches[2])  ($secs s)" }
  } else {
    Write-Progress -Activity $barActivity -Status "starting the server  ($secs s)"
  }
}
while ($true) {
  Start-Sleep -Milliseconds 700
  Show-LoadProgress
  if ($p.HasExited) {
    Write-Progress -Activity $barActivity -Completed
    Write-Host "The server stopped while loading. Last log lines:" -ForegroundColor Red
    Get-Content $log -Tail 60 | Where-Object { $_ -notmatch '^progress: ' } | Select-Object -Last 15 | ForEach-Object { Write-Host "  $_" }
    Remove-Item $stateFile -Force -ErrorAction SilentlyContinue
    exit 2
  }
  try { if ((Invoke-RestMethod "$base/health" -TimeoutSec 2).status -eq "ok") { break } } catch {}
  if (((Get-Date) - $t0).TotalSeconds -gt 600) {
    Write-Progress -Activity $barActivity -Completed
    Write-Host "The server did not become ready within 10 minutes; see $log" -ForegroundColor Red
    exit 3
  }
}
Write-Progress -Activity $barActivity -Completed
$nctx = (Invoke-RestMethod "$base/props" -TimeoutSec 5).n_ctx
if ($m -eq $none) {
  Say ("Ready after {0:N0} s: speech to text only (web UI {1}/, Transcripts tab); a chat model loads when a chat asks for one" -f ((Get-Date) - $t0).TotalSeconds, $base)
  return
}
Say ("Ready after {0:N0} s: {1}, context {2:N0} (web UI {3}/)" -f ((Get-Date) - $t0).TotalSeconds, $m.alias, $nctx, $base)
Say "  engine: $(Engine-Text $Engine ([bool]$m.moe))"
if ($true) {
  # the engine's own lines: the expert cache (MoE) or that the model has no experts to stream
  $zl = @(Get-Content $log -ErrorAction SilentlyContinue | Where-Object { $_ -match "^arcinferzero: (expert cache|cache filled|expert file written|memory limit)" })
  if ($zl.Count) {
    foreach ($l in $zl) { Say ("  " + ($l -replace "^arcinferzero: ", "") -replace ", file .*$", "") }
  } elseif (-not $NoStream) {
    Say "  dense model: no MoE experts, so no SSD cache"
  }
}
Sync-PiModels

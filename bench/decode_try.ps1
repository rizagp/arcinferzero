# Token generation settings on this GPU (made for GPUs with memory of their own, e.g. Arc Pro B60): the same greedy
# answer to one coding question plain, with the model's MTP block, with a DFlash2 drafter, and with the GEMV's
# tuning switches, each the best of 2 runs. Paste the whole output into the GitHub issue.
#   .\bench\decode_try.ps1 -Models C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf [-DFlash <DFlash2 drafter folder>] [-N 256]
# Runs (each: bin\runz.exe <model> --text-file <question> --n N):
#   plain          no drafts
#   mtp            --mtp (the model's own MTP block, when it has one)
#   mtp7           --mtp --mtp-drafts 7
#   dflash         --dflash DIR --draft auto (with -DFlash)
#   GEMV_GRF=256   plain, the single-token GEMV with 256 registers per thread (ARC_GEMV_GRF=256)
#   GVROWS=1/2/4   plain, rows per sub-group of the codebook GEMV (ARC_GVROWS; default: by matrix size)
#   mtp MT_GRF=256 mtp, the 2-8 token GEMV of the verification with 256 registers, as before 2026-11-03 (ARC_GEMV_MT_GRF=256)
#   mtp MT_GRF=128 the same with 128 registers (the discrete cards' default; iGPUs default to 256)
#   GEMV_PF=0      plain, without the single-token GEMV issuing all its rows' reads of a step first (ARC_GEMV_PF=0;
#                  on by default on discrete cards since 2026-11-08)
#   GV_IQ3S=1      plain, IQ3_S weights through gemv_v (straight into registers) instead of gemv's local window
#                  (ARC_GV_IQ3S=1); also with GVROWS=2 / 4 (ARC_GVROWS: its rows per sub-group)
#   BATCH=16/32/128  plain, kernels recorded into command lists of up to n kernels, each appended to the immediate
#                  list when full (ARC_BATCH=n; off by default); also "mtp BATCH=32"
#   XALL=1         MoE models: with every expert in the GPU's expert cache, no expert ids read on the host per
#                  layer (ARC_XALL=1; off by default); also with BATCH=32. These rows say whether the tokens match plain.
#   (at the end, per model: a --prof-gen table, the GPU time of a generation step by kernel)
#   MOE_GR_GU=2/4, MOE_GR_DN=2/8   MoE models: rows per sub-group of the experts' gate / up and down GEMVs for one
#                  token (ARC_MOE_GR_GU, default 1; ARC_MOE_GR_DN, default 4)
param(
  [Parameter(Mandatory = $true)][string[]]$Models,
  [string]$DFlash = "",
  [int]$N = 256
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$runz = Join-Path $root "bin\runz.exe"
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found (run from the arcinferzero folder)" }
$vars = @("ARC_GEMV_GRF", "ARC_GVROWS", "ARC_GEMV_MT_GRF", "ARC_GEMV_PF", "ARC_GV_IQ3S", "ARC_MOE_GR_GU", "ARC_MOE_GR_DN", "ARC_BATCH", "ARC_XALL")
# (Qwen chat format, thinking off: the answer starts right away)
$q = Join-Path ([IO.Path]::GetTempPath()) "arcinferzero-decode-try.txt"
$text = "<|im_start|>user`nWrite a Python module with a class LRUCache (get, put, a capacity, O(1) operations using " +
        "a dict and a doubly linked list), docstrings, and unit tests with unittest.<|im_end|>`n<|im_start|>assistant`n" +
        "<think>`n`n</think>`n`n"
[IO.File]::WriteAllText($q, $text, (New-Object Text.UTF8Encoding $false))

function Invoke-Run([string]$model, [string[]]$extra, [hashtable]$envs) {
  $best = $null
  for ($i = 0; $i -lt 2; ++$i) {
    foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
    # (stderr lines of runz are output, not errors that stop the script; the variables are removed even after Ctrl+C)
    $eap = $ErrorActionPreference
    try {
      $ErrorActionPreference = "Continue"
      foreach ($k in $envs.Keys) { Set-Item "Env:$k" $envs[$k] }
      $o = & $runz $model --text-file $q --n $N @extra 2>&1 | Out-String
    } finally {
      $ErrorActionPreference = $eap
      foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
    }
    $m = [regex]::Match($o, "generation: \d+ tokens in [\d.]+ ms \(([\d.]+) t/s\)")
    if (-not $m.Success) {
      $err = ($o -split "`n" | Where-Object { $_ -match "error|cannot|failed|exception" } | Select-Object -First 1)
      return [pscustomobject]@{ tps = 0; note = "failed: $err" }
    }
    $a = [regex]::Match($o, "MTP: \d+ drafts, \d+ accepted \(([\d.]+) %\), ([\d.]+) tokens per main-model pass")
    $note = if ($a.Success) { "$($a.Groups[1].Value) % accepted, $($a.Groups[2].Value) tokens per pass" } else { "" }
    $g = [regex]::Match($o, "greedy:([ \d]*)")
    $r = [pscustomobject]@{ tps = [double]$m.Groups[1].Value; note = $note; moe = $o -match "expert cache"; ids = $g.Groups[1].Value.Trim() }
    if ($null -eq $best -or $r.tps -gt $best.tps) { $best = $r }
  }
  $best
}

$inv = [Globalization.CultureInfo]::InvariantCulture
function Line([string]$what, $r) { [string]::Format($inv, "{0,-22} {1,8:F2} tok/s  {2}", $what, $r.tps, $r.note) }
# (switches that must not change the answer: their tokens against plain's)
function Same([string]$what, $r, $base) {
  $n = if ($r.tps -eq 0) { $r.note } elseif ($r.ids -eq $base.ids) { "same tokens as plain" } else { "TOKENS DIFFER from plain" }
  [string]::Format($inv, "{0,-22} {1,8:F2} tok/s  {2}", $what, $r.tps, $n)
}
& (Join-Path $root "bin\mgpu_probe.exe") 2>&1 | Select-String -Pattern "^\s+GPU \d+:" | ForEach-Object { $_.Line.Trim() }
"generation of $N tokens (greedy), best of 2 runs"
foreach ($model in $Models) {
  if (-not (Test-Path $model)) { "skipped (not found): $model"; continue }
  ""
  "=== " + (Split-Path $model -Leaf)
  $null = Invoke-Run $model @("--n", "8") @{}  # (warm-up: the first run after a start is slower)
  $plain = Invoke-Run $model @() @{}
  Line "plain" $plain
  Line "mtp" (Invoke-Run $model @("--mtp") @{})
  Line "mtp7" (Invoke-Run $model @("--mtp", "--mtp-drafts", "7") @{})
  if ($DFlash) { Line "dflash" (Invoke-Run $model @("--dflash", $DFlash, "--draft", "auto") @{}) }
  Line "GEMV_GRF=256" (Invoke-Run $model @() @{ ARC_GEMV_GRF = "256" })
  foreach ($g in 1, 2, 4) { Line "GVROWS=$g" (Invoke-Run $model @() @{ ARC_GVROWS = "$g" }) }
  Line "mtp MT_GRF=256" (Invoke-Run $model @("--mtp") @{ ARC_GEMV_MT_GRF = "256" })
  Line "mtp MT_GRF=128" (Invoke-Run $model @("--mtp") @{ ARC_GEMV_MT_GRF = "128" })
  Line "GEMV_PF=0" (Invoke-Run $model @() @{ ARC_GEMV_PF = "0" })
  Line "GV_IQ3S=1" (Invoke-Run $model @() @{ ARC_GV_IQ3S = "1" })
  foreach ($g in 2, 4) { Line "GV_IQ3S=1 GVROWS=$g" (Invoke-Run $model @() @{ ARC_GV_IQ3S = "1"; ARC_GVROWS = "$g" }) }
  foreach ($b in 16, 32, 128) { Same "BATCH=$b" (Invoke-Run $model @() @{ ARC_BATCH = "$b" }) $plain }
  Line "mtp BATCH=32" (Invoke-Run $model @("--mtp") @{ ARC_BATCH = "32" })
  if ($plain.moe) {  # (MoE models: the experts' per-tile GEMVs of one token, rows per sub-group)
    Same "XALL=1" (Invoke-Run $model @() @{ ARC_XALL = "1" }) $plain
    Same "XALL=1 BATCH=32" (Invoke-Run $model @() @{ ARC_XALL = "1"; ARC_BATCH = "32" }) $plain
    foreach ($g in 2, 4) { Line "MOE_GR_GU=$g" (Invoke-Run $model @() @{ ARC_MOE_GR_GU = "$g" }) }
    foreach ($g in 2, 8) { Line "MOE_GR_DN=$g" (Invoke-Run $model @() @{ ARC_MOE_GR_DN = "$g" }) }
  }
  # (where a generation step's GPU time goes, per kernel, plain: --prof-gen over 32 steps after a 128-token prompt)
  "--- generation profile (plain, 32 steps)"
  $eap = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    $o = & $runz $model --bench-pp 128 --n 32 --prof-gen 2>&1 | Out-String
  } finally { $ErrorActionPreference = $eap }
  $o -split "`n" | Where-Object { $_ -match "^tg\d+:|tg steps|weights read|GPU time by kernel|^\s+\w+\s+[\d.]+ ms" } | ForEach-Object { $_.TrimEnd() }
  # (kernel launches and host waits per token: plain, and for MoE models with ARC_XALL=1)
  "--- launches per token"
  $runs = @(@{ n = "plain"; e = @{} })
  if ($plain.moe) { $runs += @{ n = "XALL=1"; e = @{ ARC_XALL = "1" } } }
  foreach ($c in $runs) {
    $eap = $ErrorActionPreference
    try {
      $ErrorActionPreference = "Continue"
      $env:ARC_LAUNCH_STATS = "1"
      foreach ($k in $c.e.Keys) { Set-Item "Env:$k" $c.e[$k] }
      $o = & $runz $model --bench-pp 128 --n 32 2>&1 | Out-String
    } finally {
      $ErrorActionPreference = $eap
      Remove-Item Env:ARC_LAUNCH_STATS -ErrorAction SilentlyContinue
      foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
    }
    $o -split "`n" | Where-Object { $_ -match "^launches:|expert cache" } | ForEach-Object { "$($c.n): " + $_.Trim() }
  }
}
Remove-Item $q -ErrorAction SilentlyContinue

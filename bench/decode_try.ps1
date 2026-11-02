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
#   mtp MT_GRF=128 mtp, the 2-8 token GEMV of the verification with the default register file (ARC_GEMV_MT_GRF=128)
#   GEMV_PF=1      plain, the single-token GEMV issuing all its rows' reads of a step first (ARC_GEMV_PF=1)
param(
  [Parameter(Mandatory = $true)][string[]]$Models,
  [string]$DFlash = "",
  [int]$N = 256
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$runz = Join-Path $root "bin\runz.exe"
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found (run from the arcinferzero folder)" }
$vars = @("ARC_GEMV_GRF", "ARC_GVROWS", "ARC_GEMV_MT_GRF", "ARC_GEMV_PF")
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
    $r = [pscustomobject]@{ tps = [double]$m.Groups[1].Value; note = $note }
    if ($null -eq $best -or $r.tps -gt $best.tps) { $best = $r }
  }
  $best
}

$inv = [Globalization.CultureInfo]::InvariantCulture
function Line([string]$what, $r) { [string]::Format($inv, "{0,-16} {1,8:F2} tok/s  {2}", $what, $r.tps, $r.note) }
& (Join-Path $root "bin\mgpu_probe.exe") 2>&1 | Select-String -Pattern "^\s+GPU \d+:" | ForEach-Object { $_.Line.Trim() }
"generation of $N tokens (greedy), best of 2 runs"
foreach ($model in $Models) {
  if (-not (Test-Path $model)) { "skipped (not found): $model"; continue }
  ""
  "=== " + (Split-Path $model -Leaf)
  $null = Invoke-Run $model @("--n", "8") @{}  # (warm-up: the first run after a start is slower)
  Line "plain" (Invoke-Run $model @() @{})
  Line "mtp" (Invoke-Run $model @("--mtp") @{})
  Line "mtp7" (Invoke-Run $model @("--mtp", "--mtp-drafts", "7") @{})
  if ($DFlash) { Line "dflash" (Invoke-Run $model @("--dflash", $DFlash, "--draft", "auto") @{}) }
  Line "GEMV_GRF=256" (Invoke-Run $model @() @{ ARC_GEMV_GRF = "256" })
  foreach ($g in 1, 2, 4) { Line "GVROWS=$g" (Invoke-Run $model @() @{ ARC_GVROWS = "$g" }) }
  Line "mtp MT_GRF=128" (Invoke-Run $model @("--mtp") @{ ARC_GEMV_MT_GRF = "128" })
  Line "GEMV_PF=1" (Invoke-Run $model @() @{ ARC_GEMV_PF = "1" })
}
Remove-Item $q -ErrorAction SilentlyContinue

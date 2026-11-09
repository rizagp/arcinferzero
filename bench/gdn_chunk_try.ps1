# The chunked DeltaNet (QUEUE F33.1) against the sequential kernel on this GPU: prompt speed, the DeltaNet kernel's own
# time, and accuracy (KL against the sequential path over 4 perplexity chunks of USAGE.md). Made for the Pro B60
# (issue #4); runs on any GPU. Paste the whole output into the GitHub issue.
#   .\bench\gdn_chunk_try.ps1 -Model C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf [-MoeModel C:\hf\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf]
# Modes (ARC_GDN_CHUNK):
#   0  gdn_recur, the sequential kernel (the default)
#   1  gdn_chunk: chunks of 32 tokens, 4 work-items per state row, local rows read element by element
#   2  gdn_chunk, local rows read 16 floats at a time
#   3  gdn_chunk_x: the chunk's products on XMX (16-lane DPAS; checked at start against gdn_chunk, which it falls back
#      to when they differ or there is no XMX: the "path" column says which ran)
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [string]$MoeModel = "",
  [int[]]$Sizes = @(512, 4096, 16384)
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$runz = Join-Path $root "bin\runz.exe"
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found (run from the arcinferzero folder)" }
$inv = [Globalization.CultureInfo]::InvariantCulture

function Invoke-Runz([int]$mode, [string[]]$a) {
  $eap = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    $env:ARC_GDN_CHUNK = "$mode"
    return (& $runz @a 2>&1 | Out-String)
  } finally {
    $ErrorActionPreference = $eap
    Remove-Item Env:ARC_GDN_CHUNK -ErrorAction SilentlyContinue
  }
}
function Path-Of([string]$o, [int]$mode) {
  if ($mode -eq 0) { return "gdn_recur" }
  if ($mode -eq 3 -and $o -match "chunked DeltaNet on XMX \(results checked") { return "gdn_chunk_x (XMX)" }
  if ($o -match "chunked DeltaNet on XMX differs") { return "gdn_chunk (XMX differed!)" }
  if ($o -match "chunked DeltaNet on XMX does not build") { return "gdn_chunk (XMX did not build)" }
  return "gdn_chunk"
}
# best of 3 prompt passes; with -Prof the GPU time of the DeltaNet kernel of the last pass
function Invoke-Bench([string]$model, [int]$mode, [int]$pp, [switch]$Prof) {
  $a = @($model, "--bench-pp", $pp, "--n", 1)
  if ($Prof) { $a += "--prof" }
  $o = Invoke-Runz $mode $a
  $tps = [regex]::Matches($o, "pp\d+ run \d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s") | ForEach-Object { [double]$_.Groups[1].Value }
  $k = [regex]::Match($o, "\n\s+(gdn_recur|gdn_chunk_x|gdn_chunk)\s+([\d.]+) ms\s+([\d.]+)%")
  $kt = if ($k.Success) { "{0} {1} ms ({2} %)" -f $k.Groups[1].Value, $k.Groups[2].Value, $k.Groups[3].Value } else { "" }
  if (-not $tps) {
    $err = ($o -split "`n" | Where-Object { $_ -match "error|cannot|failed|exception" } | Select-Object -First 1)
    return [pscustomobject]@{ tps = 0; path = "failed: $err"; kern = "" }
  }
  [pscustomobject]@{ tps = ($tps | Measure-Object -Maximum).Maximum; path = (Path-Of $o $mode); kern = $kt }
}
function Line([string]$what, $r) { [string]::Format($inv, "{0,-26} {1,9:F1} tok/s  {2,-28} {3}", $what, $r.tps, $r.path, $r.kern) }

& (Join-Path $root "bin\mgpu_probe.exe") 2>&1 | Select-String -Pattern "^\s+GPU \d+:" | ForEach-Object { $_.Line.Trim() }
foreach ($m in @($Model, $MoeModel)) {
  if (-not $m) { continue }
  if (-not (Test-Path $m)) { "skipped (not found): $m"; continue }
  ""
  "=== " + (Split-Path $m -Leaf) + ": prompt speed, best of 3 passes"
  $null = Invoke-Bench $m 0 512  # (warm-up: the first run after a start is slower)
  foreach ($pp in $Sizes) {
    foreach ($mode in 0, 1, 2, 3) { Line ("{0,5} tokens, mode {1}" -f $pp, $mode) (Invoke-Bench $m $mode $pp) }
  }
  Line ("{0,5} tokens, mode 0 again" -f $Sizes[-1]) (Invoke-Bench $m 0 $Sizes[-1])
  ""
  "--- the DeltaNet kernel's GPU time, 4096-token prompt (--prof)"
  foreach ($mode in 0, 1, 2, 3) { Line ("mode {0}" -f $mode) (Invoke-Bench $m $mode 4096 -Prof) }
  ""
  "--- accuracy against mode 0: KL over 4 perplexity chunks of USAGE.md"
  $kl = Join-Path $env:TEMP "gdn_chunk_try.kl"
  $text = Join-Path $root "USAGE.md"
  $o = Invoke-Runz 0 @($m, "--perplexity", $text, "--ppl-chunks", 4, "--ppl-kl-save", $kl)
  $ppl = [regex]::Match($o, "perplexity: ([\d.]+)")
  "mode 0  perplexity " + $(if ($ppl.Success) { $ppl.Groups[1].Value } else { "?" })
  foreach ($mode in 1, 2, 3) {
    $o = Invoke-Runz $mode @($m, "--perplexity", $text, "--ppl-chunks", 4, "--ppl-kl", $kl)
    $line = ($o -split "`n" | Where-Object { $_ -match "KL vs base" } | Select-Object -Last 1)
    $ppl = [regex]::Match($o, "perplexity: ([\d.]+)")
    "mode {0}  {1}  perplexity {2}  ({3})" -f $mode, $(if ($line) { $line.Trim() } else { "no KL line" }),
      $(if ($ppl.Success) { $ppl.Groups[1].Value } else { "?" }), (Path-Of $o $mode)
  }
  Remove-Item $kl -ErrorAction SilentlyContinue
}

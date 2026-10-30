# Sweeps the prompt GEMM's settings on this GPU (made for GPUs with XMX: Arc B-series / Xe2, Xe3) and prints which
# give the fastest prompt processing, one setting at a time from the defaults, then the best of each together.
# Each run is bin\runz.exe --bench-pp (the best of 3 passes); the engine checks every XMX build against DP4a at start
# and falls back to DP4a if they differ, which the table shows. Paste the whole output into the GitHub issue.
#   .\bench\gemm_sweep.ps1 -Models C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf, C:\hf\Qwen3.5-4B-UD-Q4_K_XL.gguf
#   [-Tokens 4096]   prompt length per run
#   [-Quick]         fewer values (about half the runs)
# Settings (environment variables the engine reads; defaults in brackets):
#   XMX         ARC_NO_XMX=1 for DP4a only [XMX when the GPU has it]
#   TM          ARC_GEMM_TM, tokens per sub-group [16]
#   RG          ARC_GEMM_RG, token rows swept together per weight slice [4]
#   WGY         ARC_GEMM_WGY, most sub-groups per work-group [8]
#   SPLIT       ARC_GEMM_SPLIT, superblocks per split-K piece, 0 = none [17]
#   CHUNK       runz --chunk, tokens per prompt chunk [model's default]
param(
  [Parameter(Mandatory = $true)][string[]]$Models,
  [int]$Tokens = 4096,
  [switch]$Quick
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$runz = Join-Path $root "bin\runz.exe"
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found (run from the arcinferzero folder)" }
$vars = @("ARC_NO_XMX", "ARC_GEMM_TM", "ARC_GEMM_RG", "ARC_GEMM_WGY", "ARC_GEMM_SPLIT")

# one run: the settings in $set (name -> value; CHUNK goes to the command line), returns tok/s and the GEMM path
function Invoke-Run([string]$model, [hashtable]$set) {
  foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  $args2 = @($model, "--bench-pp", $Tokens, "--n", 1)
  foreach ($k in $set.Keys) {
    switch ($k) {
      "XMX" { if ($set[$k] -eq "off") { $env:ARC_NO_XMX = "1" } }
      "TM" { $env:ARC_GEMM_TM = "$($set[$k])" }
      "RG" { $env:ARC_GEMM_RG = "$($set[$k])" }
      "WGY" { $env:ARC_GEMM_WGY = "$($set[$k])" }
      "SPLIT" { $env:ARC_GEMM_SPLIT = "$($set[$k])" }
      "CHUNK" { $args2 += @("--chunk", $set[$k]) }
    }
  }
  $o = & $runz @args2 2>&1 | Out-String
  foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  $tps = [regex]::Matches($o, "pp\d+ run \d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s") | ForEach-Object { [double]$_.Groups[1].Value }
  $path = if ($o -match "XMX on for the prompt GEMMs") { "XMX" } elseif ($o -match "XMX results differ") { "DP4a (XMX differed!)" } else { "DP4a" }
  if (-not $tps) {
    $err = ($o -split "`n" | Where-Object { $_ -match "error|cannot|failed|exception" } | Select-Object -First 1)
    return [pscustomobject]@{ tps = 0; path = "failed: $err" }
  }
  [pscustomobject]@{ tps = ($tps | Measure-Object -Maximum).Maximum; path = $path }
}

$grid = [ordered]@{
  XMX   = @("on", "off")
  TM    = @(16, 32)
  RG    = if ($Quick) { @(2, 4, 8) } else { @(1, 2, 4, 8) }
  WGY   = if ($Quick) { @(2, 8) } else { @(1, 2, 4, 8, 16) }
  SPLIT = if ($Quick) { @(0, 17) } else { @(0, 9, 17, 34) }
  CHUNK = if ($Quick) { @(2048, 4096) } else { @(1024, 2048, 4096, 8192) }
}
$defaults = @{ XMX = "on"; TM = 16; RG = 4; WGY = 8; SPLIT = 17 }

$inv = [Globalization.CultureInfo]::InvariantCulture
function Line([string]$what, $r) { [string]::Format($inv, "{0,-28} {1,10:F1} tok/s  {2}", $what, $r.tps, $r.path) }
& (Join-Path $root "bin\mgpu_probe.exe") 2>&1 | Select-String -Pattern "^\s+GPU \d+:" | ForEach-Object { $_.Line.Trim() }
"prompt: $Tokens tokens; tok/s = best of 3 passes"
foreach ($model in $Models) {
  if (-not (Test-Path $model)) { "skipped (not found): $model"; continue }
  ""
  "=== " + (Split-Path $model -Leaf)
  $null = Invoke-Run $model @{}  # (warm-up: the first run after a start is slower)
  $base = Invoke-Run $model @{}
  Line "defaults" $base
  $best = @{}
  foreach ($p in $grid.Keys) {
    $bv = $null; $bt = 0
    foreach ($v in $grid[$p]) {
      if ($defaults.ContainsKey($p) -and "$($defaults[$p])" -eq "$v") { $r = $base }  # (the default: measured above)
      else { $r = Invoke-Run $model @{ $p = $v } }
      Line "$p=$v" $r
      if ($r.tps -gt $bt) { $bt = $r.tps; $bv = $v }
    }
    $best[$p] = $bv
  }
  $set = @{}
  foreach ($p in $best.Keys) { if ($null -ne $best[$p]) { $set[$p] = $best[$p] } }
  $r = Invoke-Run $model $set
  $desc = ($set.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join " "
  Line "best of each together" $r
  "  ($desc)"
  Line "defaults again (drift)" (Invoke-Run $model @{})
}

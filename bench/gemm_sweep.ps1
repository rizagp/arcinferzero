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
#   GRF         ARC_GEMM_GRF, registers per thread of the dense GEMM: 256 or 128 (the default file) [128]
#   MOEGRF      ARC_MOE_GRF, the same for the MoE expert GEMMs [256]; MoE models only
#   MOECG       ARC_MOE_CG, sub-groups per work-group along the columns of the MoE expert GEMM's tiles [1]; MoE only
#   MOEWY       ARC_MOE_WY, sub-groups per work-group along the tiles (mostly the same expert's) [1]; MoE only
#   CG          ARC_GEMM_CG, 16-column groups per work-group: 1, 2 or 4 [1]
#   ACT         runz --act256: one activation scale per 256 values instead of 32 [32]; less accurate, so a winning
#               ACT=256 needs a KL check before it could become a default (not part of "best of each together")
#   CHUNK       runz --chunk, tokens per prompt chunk [model's default]
#   X16         ARC_X16=1: the f16 XMX GEMM (kernels/gemm_x16.cl) for W8 weights, e.g. the 27B's IQ3_S [off]
#   X16TM       with X16 on: ARC_X16_TM, tokens per sub-group 8, 16 or 32 [32]
#   X16GRF      with X16 on: ARC_X16_GRF, registers per thread 256 or 128 [256]
#   X16CG       with X16 on: ARC_X16_CG, 16-column groups per sub-group 1, 2 or 4 [1]
#   H8          ARC_H8=1: the H8 GEMM (QUEUE 31.12, zero/engine_h8.h): weights and activations Hadamard-rotated, int8
#               with one scale per 256 values, no per-16 / per-32 rescaling [off]
#   X2D         the XMX GEMM's activations by 2D block loads (QUEUE 31.8; ARC_GEMM_2D=0 turns them off); "on": without
#               the prefetch of the next 64 bytes (ARC_GEMM_2D_PF=0); "pf": with it [pf]
param(
  [Parameter(Mandatory = $true)][string[]]$Models,
  [int]$Tokens = 4096,
  [switch]$Quick
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$runz = Join-Path $root "bin\runz.exe"
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found (run from the arcinferzero folder)" }
$vars = @("ARC_NO_XMX", "ARC_GEMM_TM", "ARC_GEMM_RG", "ARC_GEMM_WGY", "ARC_GEMM_SPLIT", "ARC_GEMM_GRF", "ARC_MOE_GRF", "ARC_MOE_CG", "ARC_MOE_WY", "ARC_GEMM_CG",
          "ARC_X16", "ARC_X16_TM", "ARC_X16_GRF", "ARC_X16_CG", "ARC_H8", "ARC_GEMM_2D", "ARC_GEMM_2D_PF")

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
      "GRF" { $env:ARC_GEMM_GRF = "$($set[$k])" }
      "MOEGRF" { $env:ARC_MOE_GRF = "$($set[$k])" }
      "MOECG" { $env:ARC_MOE_CG = "$($set[$k])" }
      "MOEWY" { $env:ARC_MOE_WY = "$($set[$k])" }
      "CG" { $env:ARC_GEMM_CG = "$($set[$k])" }
      "ACT" { if ("$($set[$k])" -eq "256") { $args2 += "--act256" } }
      "CHUNK" { $args2 += @("--chunk", $set[$k]) }
      "X16" { if ($set[$k] -eq "on") { $env:ARC_X16 = "1" } }
      "X16TM" { $env:ARC_X16 = "1"; $env:ARC_X16_TM = "$($set[$k])" }
      "X16GRF" { $env:ARC_X16 = "1"; $env:ARC_X16_GRF = "$($set[$k])" }
      "X16CG" { $env:ARC_X16 = "1"; $env:ARC_X16_CG = "$($set[$k])" }
      "H8" { $env:ARC_H8 = $(if ($set[$k] -eq "on") { "1" } else { "0" }) }  # (off: ARC_H8=0, H8 being the default for Qwen dense models on discrete cards)
      "X2D" { $env:ARC_GEMM_2D = $(if ($set[$k] -eq "off") { "0" } else { "1" }); $env:ARC_GEMM_2D_PF = $(if ($set[$k] -eq "pf") { "1" } else { "0" }) }
    }
  }
  # (stderr lines of runz are output here, not errors that stop the sweep; the variables are removed even after
  #  Ctrl+C, so later runs in this shell do not pick them up)
  $eap = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    $o = & $runz @args2 2>&1 | Out-String
  } finally {
    $ErrorActionPreference = $eap
    foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  }
  $tps = [regex]::Matches($o, "pp\d+ run \d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s") | ForEach-Object { [double]$_.Groups[1].Value }
  $path = if ($o -match "XMX on for the prompt GEMMs") { "XMX" } elseif ($o -match "XMX results differ") { "DP4a (XMX differed!)" } else { "DP4a" }
  if ($o -match "f16 XMX on for the W8") { $path += " + f16 XMX (W8)" }
  elseif ($o -match "f16 XMX results differ") { $path += " (f16 XMX differed!)" }
  elseif ($o -match "f16 XMX kernels do not build") { $path += " (f16 XMX did not build!)" }
  if ($o -match "H8 on for the prompt GEMMs") { $path += " + H8" }
  if ($o -match "with 2D block loads") { $path += " + 2D loads" }
  elseif ($o -match "2D block loads \(ARC_GEMM_2D\) do not build or differ") { $path += " (2D loads differed!)" }
  if (-not $tps) {
    $err = ($o -split "`n" | Where-Object { $_ -match "error|cannot|failed|exception" } | Select-Object -First 1)
    return [pscustomobject]@{ tps = 0; path = "failed: $err"; moe = $false }
  }
  [pscustomobject]@{ tps = ($tps | Measure-Object -Maximum).Maximum; path = $path; moe = $o -match "expert cache" }
}

$grid = [ordered]@{
  XMX   = @("on", "off")
  TM    = @(16, 32)
  RG    = if ($Quick) { @(2, 4, 8) } else { @(1, 2, 4, 8) }
  WGY   = if ($Quick) { @(2, 8) } else { @(1, 2, 4, 8, 16) }
  SPLIT = if ($Quick) { @(0, 17) } else { @(0, 9, 17, 34) }
  GRF   = @(256, 128)
  MOEGRF = @(256, 128)
  MOECG = @(1, 2, 4)
  MOEWY = if ($Quick) { @(1, 4) } else { @(1, 2, 4, 8) }
  CG   = if ($Quick) { @(1, 2) } else { @(1, 2, 4) }
  ACT   = @(32, 256)
  CHUNK = if ($Quick) { @(2048, 4096) } else { @(1024, 2048, 4096, 8192) }
  X16   = @("off", "on")
  X16TM = @(8, 16, 32)
  X16GRF = @(256, 128)
  X16CG = @(1, 2, 4)
  H8    = @("off", "on")
  X2D   = @("pf", "on", "off")
}
$defaults = @{ XMX = "on"; TM = 16; RG = 4; WGY = 8; SPLIT = 17; GRF = 128; MOEGRF = 256; MOECG = 1; MOEWY = 1; ACT = 32; X16 = "off"; X2D = "pf" }  # (no H8: on by default for Qwen dense models on discrete cards, off otherwise, so both are measured)

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
  $x16_ran = $false
  foreach ($p in $grid.Keys) {
    if ($p -in @("MOEGRF", "MOECG", "MOEWY") -and -not $base.moe) { continue }  # (a dense model has no expert GEMMs)
    # (the f16 GEMM's own settings only where it ran: a model without W8 weights never takes it)
    if ($p -in @("X16TM", "X16GRF", "X16CG") -and -not $x16_ran) { continue }
    $bv = $null; $bt = 0
    foreach ($v in $grid[$p]) {
      if ($defaults.ContainsKey($p) -and "$($defaults[$p])" -eq "$v") { $r = $base }  # (the default: measured above)
      else { $r = Invoke-Run $model @{ $p = $v } }
      Line "$p=$v" $r
      if ($p -eq "X16" -and $v -eq "on") { $x16_ran = $r.path -match "f16 XMX \(W8\)" }
      if ($r.tps -gt $bt) { $bt = $r.tps; $bv = $v }
    }
    $best[$p] = $bv
  }
  $set = @{}
  foreach ($p in $best.Keys) { if ($null -ne $best[$p] -and $p -ne "ACT") { $set[$p] = $best[$p] } }
  # (the f16 GEMM's settings turn it on: only when it won, and never together with DP4a only)
  if ($best["X16"] -ne "on" -or $best["XMX"] -eq "off") { $set.Remove("X16"); $set.Remove("X16TM"); $set.Remove("X16GRF"); $set.Remove("X16CG") }
  $r = Invoke-Run $model $set
  $desc = ($set.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join " "
  Line "best of each together" $r
  "  ($desc)"
  Line "defaults again (drift)" (Invoke-Run $model @{})
}

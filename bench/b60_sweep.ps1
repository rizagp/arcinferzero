# Sweeps the prompt GEMM's settings on a discrete Arc card (made for the Pro B60, issue #4) around the two best paths
# so far: the default (XMX with 2D loads and their prefetch: 764 tok/s on the 27B, 4K prompt) and H8 (ARC_H8=1,
# Hadamard-rotated int8 with one scale per column and token: 918 tok/s). Each setting is changed alone from each base,
# the winners are put together, then measured at prompt sizes of real use, and their accuracy is checked against the
# default by KL. With -MoeModel also the 35B-A3B's expert GEMM shapes (ARC_MOE_CG / ARC_MOE_WY) and H8 there.
# Paste the whole output into the GitHub issue.
#   .\bench\b60_sweep.ps1 -Model C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf [-MoeModel C:\hf\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf]
#   [-Tokens 4096]  phase A's prompt length   [-Quick]  fewer values, no 16K row   [-Gpus 2,4]  phase E's GPU counts
# A setting wins only 3 % ahead of its base; "best" is the faster of all winners together and the single best change.
# Settings (defaults on discrete cards in brackets):
#   TM      ARC_GEMM_TM, tokens per sub-group [16]      WGY    ARC_GEMM_WGY, sub-groups per work-group [8]
#   RG      ARC_GEMM_RG, rows per weight slice [4]       CG     ARC_GEMM_CG, column groups per work-group [1]
#   GRF     ARC_GEMM_GRF, registers per thread [128]     SPLIT  ARC_GEMM_SPLIT, superblocks per split-K piece [17]
#   X2D     2D loads: pf (with the prefetch) / on / off [pf]
#   WPF     ARC_GEMM_WPF, the weights prefetched n 64-value chunks ahead, 0 = off [0]
#   FUSE    ARC_H8_FUSE, H8's weight rotation in the decode pass (1) or after it (0) [1]; H8 only, bit-identical
#   MOECG / MOEWY  ARC_MOE_CG / ARC_MOE_WY, sub-groups per work-group of the expert GEMM [1 / 1] (35B only)
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [string]$MoeModel = "",
  [int]$Tokens = 4096,
  [int[]]$Gpus = @(2, 4),
  [switch]$Quick
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$runz = Join-Path $root "bin\runz.exe"
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found (run from the arcinferzero folder)" }
if (-not (Test-Path $Model)) { throw "model not found: $Model" }
$vars = @("ARC_H8", "ARC_GEMM_TM", "ARC_GEMM_WGY", "ARC_GEMM_RG", "ARC_GEMM_CG", "ARC_GEMM_GRF", "ARC_GEMM_SPLIT", "ARC_GEMM_2D",
          "ARC_GEMM_2D_PF", "ARC_MOE_CG", "ARC_MOE_WY", "ARC_GEMM_WPF", "ARC_H8_FUSE")
$inv = [Globalization.CultureInfo]::InvariantCulture

# the environment of a setting set (name -> value as in the table above)
function Env-Of([hashtable]$set) {
  $e = @{}
  foreach ($k in $set.Keys) {
    $v = "$($set[$k])"
    switch ($k) {
      "H8" { if ($v -eq "on") { $e.ARC_H8 = "1" } }
      "TM" { $e.ARC_GEMM_TM = $v }
      "WGY" { $e.ARC_GEMM_WGY = $v }
      "RG" { $e.ARC_GEMM_RG = $v }
      "CG" { $e.ARC_GEMM_CG = $v }
      "GRF" { $e.ARC_GEMM_GRF = $v }
      "SPLIT" { $e.ARC_GEMM_SPLIT = $v }
      "X2D" { $e.ARC_GEMM_2D = $(if ($v -eq "off") { "0" } else { "1" }); $e.ARC_GEMM_2D_PF = $(if ($v -eq "pf") { "1" } else { "0" }) }
      "WPF" { $e.ARC_GEMM_WPF = $v }
      "FUSE" { $e.ARC_H8_FUSE = $v }
      "MOECG" { $e.ARC_MOE_CG = $v }
      "MOEWY" { $e.ARC_MOE_WY = $v }
    }
  }
  $e
}
# one runz call with the settings $set and the arguments $a; returns its output
function Invoke-Runz([hashtable]$set, [string[]]$a) {
  foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  $e = Env-Of $set
  foreach ($k in $e.Keys) { Set-Item "Env:$k" $e[$k] }
  $eap = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    return (& $runz @a 2>&1 | Out-String)
  } finally {
    $ErrorActionPreference = $eap
    foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  }
}
# a prompt benchmark: best of 3 passes, and the path the engine took
function Invoke-Bench([string]$model, [hashtable]$set, [int]$pp, [int]$chunk = 0, [int]$gpus = 1) {
  $a = @($model, "--bench-pp", $pp, "--n", 1)
  if ($chunk -gt 0) { $a += @("--chunk", $chunk) }
  if ($gpus -gt 1) { $a += @("--gpus", $gpus) }
  $o = Invoke-Runz $set $a
  $tps = [regex]::Matches($o, "pp\d+ run \d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s") | ForEach-Object { [double]$_.Groups[1].Value }
  $path = if ($o -match "XMX on for the prompt GEMMs") { "XMX" } elseif ($o -match "XMX results differ") { "DP4a (XMX differed!)" } else { "DP4a" }
  if ($o -match "with 2D block loads") { $path += " + 2D" } elseif ($o -match "2D block loads \(ARC_GEMM_2D\) do not build or differ") { $path += " (2D differed!)" }
  if ($o -match "H8 on for the prompt GEMMs") { $path += " + H8" }
  if (-not $tps) {
    $err = ($o -split "`n" | Where-Object { $_ -match "error|cannot|failed|exception" } | Select-Object -First 1)
    return [pscustomobject]@{ tps = 0; path = "failed: $err" }
  }
  [pscustomobject]@{ tps = ($tps | Measure-Object -Maximum).Maximum; path = $path }
}
function Line([string]$what, $r) { [string]::Format($inv, "{0,-40} {1,9:F1} tok/s  {2}", $what, $r.tps, $r.path) }
function Desc([hashtable]$set) { if ($set.Count -eq 0) { "defaults" } else { ($set.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join " " } }

$grid = [ordered]@{
  TM    = @(16, 32)
  WGY   = if ($Quick) { @(4, 8, 16) } else { @(2, 4, 8, 16) }
  RG    = if ($Quick) { @(2, 4, 8) } else { @(1, 2, 4, 8) }
  CG    = @(1, 2, 4)
  GRF   = @(128, 256)
  SPLIT = if ($Quick) { @(0, 17) } else { @(0, 9, 17, 34) }
  X2D   = @("pf", "on", "off")
  WPF   = if ($Quick) { @(0, 4) } else { @(0, 2, 4, 8) }
  FUSE  = @(1, 0)
}
$defaults = @{ TM = 16; WGY = 8; RG = 4; CG = 1; GRF = 128; SPLIT = 17; X2D = "pf"; WPF = 0; FUSE = 1 }

& (Join-Path $root "bin\mgpu_probe.exe") 2>&1 | Select-String -Pattern "^\s+GPU \d+:" | ForEach-Object { $_.Line.Trim() }
"model: " + (Split-Path $Model -Leaf) + "; phase A prompt: $Tokens tokens; tok/s = best of 3 passes"
$null = Invoke-Bench $Model @{} 512  # (warm-up: the first run after a start is slower)
$together = @{}
foreach ($base in @(@{ name = "default (no H8)"; set = @{} }, @{ name = "H8"; set = @{ H8 = "on" } })) {
  ""
  "=== A. $($base.name): one setting at a time"
  $b = Invoke-Bench $Model $base.set $Tokens
  Line "base" $b
  $bestSet = $base.set.Clone()
  $single = $base.set.Clone(); $singleT = $b.tps
  foreach ($p in $grid.Keys) {
    if ($p -eq "FUSE" -and -not $base.set.ContainsKey("H8")) { continue }  # (H8's switch: nothing to measure without it)
    $bv = $defaults[$p]; $bt = $b.tps
    foreach ($v in $grid[$p]) {
      if ("$v" -eq "$($defaults[$p])") { continue }  # (the base: measured above)
      $s = $base.set.Clone(); $s[$p] = $v
      $r = Invoke-Bench $Model $s $Tokens
      Line "$p=$v" $r
      # (a winner must be 3 % ahead of the base, past the run-to-run drift)
      if ($r.tps -gt $bt * 1.03 -and $r.tps -gt $b.tps * 1.03) { $bt = $r.tps; $bv = $v }
      if ($r.tps -gt $singleT -and $r.tps -gt $b.tps * 1.03) { $singleT = $r.tps; $single = $s.Clone() }
    }
    if ("$bv" -ne "$($defaults[$p])") { $bestSet[$p] = $bv }
  }
  # (winners alone do not always add up: settings of the same work-group can work against each other, so the single
  #  best change is kept too, and whichever of the two is faster is "best")
  $r = Invoke-Bench $Model $bestSet $Tokens
  Line "best of each together" $r
  "  (" + (Desc $bestSet) + ")"
  Line "base again (drift)" (Invoke-Bench $Model $base.set $Tokens)
  "  single best change: " + (Desc $single)
  $together[$base.name] = $(if ($r.tps -ge $singleT) { $bestSet } else { $single })
  "  taken as best: " + (Desc $together[$base.name])
}

""
"=== B. prompt sizes of real use (default chunks unless noted)"
$sets = [ordered]@{ "default" = @{}; "H8" = @{ H8 = "on" }; "best without H8" = $together["default (no H8)"]; "best with H8" = $together["H8"] }
$sizes = @(@(512, 0, "512 (a chat turn)"), @(4096, 0, "4096 (a document)"), @(16384, 0, "16384 (a long prompt)"), @(16384, 4096, "16384 in 4096-token chunks"))
if ($Quick) { $sizes = $sizes[0..1] }
foreach ($z in $sizes) {
  foreach ($n in $sets.Keys) { Line ("{0,-27} {1,-16}" -f $z[2], $n) (Invoke-Bench $Model $sets[$n] $z[0] $z[1]) }
}

""
"=== C. accuracy against the default: KL over 4 perplexity chunks of USAGE.md"
$kl = Join-Path $env:TEMP "b60_sweep_base.kl"
$text = Join-Path $root "USAGE.md"
$o = Invoke-Runz @{} @($Model, "--perplexity", $text, "--ppl-chunks", 4, "--ppl-kl-save", $kl)
($o -split "`n" | Where-Object { $_ -match "^Final|perplexity|PPL" } | Select-Object -Last 1) | ForEach-Object { "default: " + $_.Trim() }
foreach ($n in @("H8", "best without H8", "best with H8")) {
  $o = Invoke-Runz $sets[$n] @($Model, "--perplexity", $text, "--ppl-chunks", 4, "--ppl-kl", $kl)
  $line = ($o -split "`n" | Where-Object { $_ -match "KL vs base" } | Select-Object -Last 1)
  $ppl = [regex]::Match($o, "perplexity: ([\d.]+)")
  "{0,-18} {1}  perplexity {2}  ({3})" -f $n, $(if ($line) { $line.Trim() } else { "no KL line" }),
    $(if ($ppl.Success) { $ppl.Groups[1].Value } else { "?" }), (Desc $sets[$n])
}
Remove-Item $kl -ErrorAction SilentlyContinue

if ($MoeModel -ne "") {
  ""
  "=== D. " + (Split-Path $MoeModel -Leaf) + ": the expert GEMM's work-group shapes, 16384-token prompt"
  if (-not (Test-Path $MoeModel)) { "skipped (not found): $MoeModel" }
  else {
    $null = Invoke-Bench $MoeModel @{} 512
    $moe = [ordered]@{
      "defaults (1 x 1)" = @{}; "MOECG=2" = @{ MOECG = 2 }; "MOECG=4" = @{ MOECG = 4 }; "MOEWY=2" = @{ MOEWY = 2 }
      "MOEWY=4" = @{ MOEWY = 4 }; "MOEWY=8" = @{ MOEWY = 8 }; "MOECG=2 MOEWY=4" = @{ MOECG = 2; MOEWY = 4 }
      "MOECG=4 MOEWY=4" = @{ MOECG = 4; MOEWY = 4 }; "H8" = @{ H8 = "on" }
    }
    foreach ($n in $moe.Keys) { Line $n (Invoke-Bench $MoeModel $moe[$n] 16384) }
    Line "defaults again (drift)" (Invoke-Bench $MoeModel @{} 16384)
  }
}

""
"=== E. several GPUs (--gpus), the 27B at 4096 and 16384 tokens" + $(if ($MoeModel -ne "") { ", the 35B at 16384" } else { "" })
$ngpu = @(& (Join-Path $root "bin\mgpu_probe.exe") 2>&1 | Select-String -Pattern "^\s+GPU \d+:").Count
foreach ($g in $Gpus) {
  if ($g -gt $ngpu) { "skipped --gpus $g ($ngpu GPUs found)"; continue }
  foreach ($n in @("default", "best with H8")) {
    foreach ($pp in @(4096, 16384)) { Line ("--gpus {0} 27B {1,-14} {2,5}" -f $g, $n, $pp) (Invoke-Bench $Model $sets[$n] $pp 0 $g) }
  }
  if ($MoeModel -ne "" -and (Test-Path $MoeModel)) {
    Line ("--gpus {0} 35B default        16384" -f $g) (Invoke-Bench $MoeModel @{} 16384 0 $g)
    Line ("--gpus {0} 35B MOEWY=4        16384" -f $g) (Invoke-Bench $MoeModel @{ MOEWY = 4 } 16384 0 $g)
  }
}
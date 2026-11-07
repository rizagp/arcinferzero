# Sweeps the prompt GEMM's and the GEMV's settings on an iGPU with XMX (made for the Arc 140T, Arrow Lake H, whose XMX
# needs 8-lane sub-groups: zero/kernels/gemm_x8.cl), then measures the best ones at prompt sizes of real use. XMX is
# forced (ARC_XMX_FORCE=1) except in the DP4a rows; the engine still checks every XMX build against DP4a at start and
# stays on DP4a if they differ, which the "path" column shows. Paste the whole output into the GitHub issue.
#   .\bench\igpu_sweep.ps1 -Model C:\path\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
#   [-Quick]   phase A with fewer settings, phase B without the chunk rows
# Phase A, the prompt GEMM, a 2048-token prompt (best of 3 passes):
#   DP4a         ARC_NO_XMX=1, the default on 8-lane GPUs
#   XMX          gemm_x8 as built: 256 registers, 16 tokens per sub-group, one 8-column half per sub-group
#   X8W          ARC_X8_W=1: one sub-group for both halves of 16 columns, the activations loaded once (ocloc: 28 % fewer
#                memory messages per DPAS, no spills)
#   TM32         ARC_GEMM_TM=32: 32 tokens per sub-group
#   GRF128       ARC_GEMM_GRF=128: the default register file (more threads; spills with X8W or TM32)
#   CG2 / CG4    ARC_GEMM_CG: 16-column groups per work-group
#   WGY2 / 4 / 16  ARC_GEMM_WGY: most sub-groups per work-group along the tokens [8]
#   RG2 / RG8    ARC_GEMM_RG: token rows swept together per weight slice [4]
#   H8           ARC_H8=1 on top (rotated int8, one scale per column and token)
# Phase B, prompt sizes of real use with DP4a, XMX and phase A's best: 512 (a chat turn), 2048 (a document), 8192 (a
#   long prompt, in the default 4096-token chunks), and the 8192 prompt in 1024 / 2048-token chunks.
# Phase C, generation (the GEMV, 64 tokens after a 128-token prompt; XMX forced so the GEMV switches apply):
#   GEMV_GRF=256 (its register file), GVROWS=1 / 2 / 4 (rows per sub-group of the codebook GEMV, default by matrix size),
#   GEMV_PF=1 (every row's reads of a step issued first). Generation already reads ~85 % of a 140T's bandwidth, so
#   little is expected here.
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [switch]$Quick
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$runz = Join-Path $root "bin\runz.exe"
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found (run from the arcinferzero folder)" }
if (-not (Test-Path $Model)) { throw "model not found: $Model" }
$vars = @("ARC_NO_XMX", "ARC_XMX_FORCE", "ARC_X8_W", "ARC_GEMM_TM", "ARC_GEMM_GRF", "ARC_GEMM_CG", "ARC_GEMM_WGY", "ARC_GEMM_RG",
          "ARC_H8", "ARC_GEMV_GRF", "ARC_GVROWS", "ARC_GEMV_PF")
$inv = [Globalization.CultureInfo]::InvariantCulture

# one run: environment $env (name -> value), prompt $pp tokens, $tg generated, optional chunk; returns prompt tok/s
# (best of the passes), generation tok/s, the median step and the path the engine took
function Invoke-Run([hashtable]$envs, [int]$pp, [int]$tg, [int]$chunk = 0) {
  foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  foreach ($k in $envs.Keys) { Set-Item "Env:$k" "$($envs[$k])" }
  $a = @($Model, "--bench-pp", $pp, "--n", $tg)
  if ($chunk -gt 0) { $a += @("--chunk", $chunk) }
  $eap = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    $o = & $runz @a 2>&1 | Out-String
  } finally {
    $ErrorActionPreference = $eap
    foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  }
  $pps = [regex]::Matches($o, "pp\d+ run \d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s") | ForEach-Object { [double]$_.Groups[1].Value }
  $tgm = [regex]::Match($o, "tg\d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s")
  $med = [regex]::Match($o, "median ([\d.]+) ms")
  $path = if ($o -match "XMX on for the prompt GEMMs") { "XMX" } elseif ($o -match "XMX results differ") { "DP4a (XMX differed!)" }
          elseif ($o -match "XMX kernels do not build") { "DP4a (XMX did not build!)" } else { "DP4a" }
  if ($o -match "H8 on for the prompt GEMMs") { $path += " + H8" }
  if (-not $pps) {
    $err = ($o -split "`n" | Where-Object { $_ -match "error|cannot|failed|exception" } | Select-Object -First 1)
    return [pscustomobject]@{ pp = 0; tg = 0; step = 0; path = "failed: $err" }
  }
  [pscustomobject]@{
    pp = ($pps | Measure-Object -Maximum).Maximum
    tg = if ($tgm.Success) { [double]$tgm.Groups[1].Value } else { 0 }
    step = if ($med.Success) { [double]$med.Groups[1].Value } else { 0 }
    path = $path
  }
}
function Line([string]$what, $r) { [string]::Format($inv, "{0,-34} {1,9:F1} tok/s  {2}", $what, $r.pp, $r.path) }
function LineTg([string]$what, $r) { [string]::Format($inv, "{0,-34} {1,7:F2} tok/s  median step {2,6:F1} ms  {3}", $what, $r.tg, $r.step, $r.path) }

$xmx = @{ ARC_XMX_FORCE = 1 }
function With([hashtable]$extra) { $h = $xmx.Clone(); foreach ($k in $extra.Keys) { $h[$k] = $extra[$k] }; $h }
$phaseA = [ordered]@{
  "DP4a"            = @{ ARC_NO_XMX = 1 }
  "XMX"             = $xmx
  "X8W"             = (With @{ ARC_X8_W = 1 })
  "TM32"            = (With @{ ARC_GEMM_TM = 32 })
  "X8W TM32"        = (With @{ ARC_X8_W = 1; ARC_GEMM_TM = 32 })
  "GRF128"          = (With @{ ARC_GEMM_GRF = 128 })
  "CG2"             = (With @{ ARC_GEMM_CG = 2 })
  "X8W CG2"         = (With @{ ARC_X8_W = 1; ARC_GEMM_CG = 2 })
  "WGY4"            = (With @{ ARC_GEMM_WGY = 4 })
  "X8W WGY4"        = (With @{ ARC_X8_W = 1; ARC_GEMM_WGY = 4 })
  "X8W RG8"         = (With @{ ARC_X8_W = 1; ARC_GEMM_RG = 8 })
  "X8W H8"          = (With @{ ARC_X8_W = 1; ARC_H8 = 1 })
}
if (-not $Quick) {
  $phaseA["CG4"] = (With @{ ARC_GEMM_CG = 4 })
  $phaseA["WGY2"] = (With @{ ARC_GEMM_WGY = 2 })
  $phaseA["X8W WGY16"] = (With @{ ARC_X8_W = 1; ARC_GEMM_WGY = 16 })
  $phaseA["X8W RG2"] = (With @{ ARC_X8_W = 1; ARC_GEMM_RG = 2 })
  $phaseA["X8W GRF128"] = (With @{ ARC_X8_W = 1; ARC_GEMM_GRF = 128 })
}

& (Join-Path $root "bin\mgpu_probe.exe") 2>&1 | Select-String -Pattern "^\s+GPU \d+:" | ForEach-Object { $_.Line.Trim() }
"model: " + (Split-Path $Model -Leaf)
""
"=== A. prompt GEMM settings, 2048-token prompt (best of 3 passes)"
$null = Invoke-Run @{ ARC_NO_XMX = 1 } 512 1  # (warm-up: the first run after a start is slower)
$res = [ordered]@{}
foreach ($n in $phaseA.Keys) {
  $r = Invoke-Run $phaseA[$n] 2048 1
  $res[$n] = $r
  Line $n $r
}
$best = ($res.GetEnumerator() | Where-Object { $_.Key -ne "DP4a" -and $_.Value.path -like "XMX*" } |
         Sort-Object { $_.Value.pp } -Descending | Select-Object -First 1)
$bestName = if ($best) { $best.Key } else { "XMX" }
Line "DP4a again (drift)" (Invoke-Run @{ ARC_NO_XMX = 1 } 2048 1)

""
"=== B. prompt sizes of real use: DP4a, XMX, and phase A's best ($bestName)"
$cases = @(@(512, 0, "512 (chat turn)"), @(2048, 0, "2048 (document)"), @(8192, 0, "8192 (long, default chunks)"))
if (-not $Quick) { $cases += , @(8192, 1024, "8192 in 1024-token chunks"); $cases += , @(8192, 2048, "8192 in 2048-token chunks") }
foreach ($c in $cases) {
  foreach ($n in @("DP4a", "XMX", $bestName) | Select-Object -Unique) {
    Line ("{0,-28} {1}" -f $c[2], $n) (Invoke-Run $phaseA[$n] $c[0] 1 $c[1])
  }
}

""
"=== C. generation (GEMV), 64 tokens after a 128-token prompt"
$phaseC = [ordered]@{
  "DP4a default"   = @{ ARC_NO_XMX = 1 }
  "XMX forced"     = $xmx
  "GEMV_GRF=256"   = (With @{ ARC_GEMV_GRF = 256 })
  "GVROWS=1"       = (With @{ ARC_GVROWS = 1 })
  "GVROWS=2"       = (With @{ ARC_GVROWS = 2 })
  "GVROWS=4"       = (With @{ ARC_GVROWS = 4 })
  "GEMV_PF=1"      = (With @{ ARC_GEMV_PF = 1 })
}
foreach ($n in $phaseC.Keys) { LineTg $n (Invoke-Run $phaseC[$n] 128 64) }
LineTg "DP4a default again (drift)" (Invoke-Run @{ ARC_NO_XMX = 1 } 128 64)

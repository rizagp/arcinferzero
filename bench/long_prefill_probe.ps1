# Where does a long prompt's time go? Per-kernel profile of the prompt at growing depth (QUEUE row 0, long prompts).
#
#   .\bench\long_prefill_probe.ps1 -Model C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
#   .\bench\long_prefill_probe.ps1 -Model C:\hf\gemma-4-12b-it-qat-q4_0.gguf -Depths 65536,131072   (default: 64K, 131K, 262K)
#
# Wanted most (issue #4): Gemma 4 26B-A4B and 31B (head-512 global layers, `attn_fa_gh`), Gemma 4 12B, Qwen3.8 27B,
# each at 64K, 131K and 262K. 262K has not been measured on any card for models of 26B and up: whether it loads, how
# much memory it needs and its speed are all unknown, so a FAILED line (with the error) is a result too: post it.
# For each depth N it runs `runz <model> --bench-pp N --n 0 --prof` once and writes the whole output (GPU, switches,
# time to first token, per-kernel table) to long_prefill_<model>_<date>.txt, then prints a summary line per depth:
# prompt tok/s and the share of the attention kernels (attn_*). Attach the .txt file to GitHub issue #4.
# A 131K prompt takes minutes (27B: ~5 min at 420 tok/s) and 262K several times that; a depth that fails (memory or
# an unsupported context) prints a FAILED line with the error, and the others still run.
# Stop any serverz / runz first (a second program on the GPU gives device-lost errors).
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [int[]]$Depths = @(65536, 131072, 262144),
  [string]$OutDir = ".",
  [int]$Gen = 32,          # tokens generated after each prompt: generation speed and bandwidth at that depth
  [switch]$SkipPeak        # skip bin\xw_peak.exe (the card's int8 XMX peak and the prompt GEMMs' % of it)
)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$runz = Join-Path $root "bin\runz.exe"
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found" }
if (Get-Process serverz, runz -ErrorAction SilentlyContinue) { Write-Warning "a serverz / runz is running: stop it first" }
$inv = [Globalization.CultureInfo]::InvariantCulture
$name = [IO.Path]::GetFileNameWithoutExtension($Model)
$out = Join-Path $OutDir ("long_prefill_{0}_{1}.txt" -f $name, (Get-Date -Format "yyyyMMdd"))
$env:ARC_BENCH_RUNS = "1"
# Targets on a B70 (the user, 2026-11-14): 82 % of the XMX peak in the prompt GEMMs, 82 % of the bandwidth in generation.
$xw = Join-Path $root "bin\xw_peak.exe"
if (-not $SkipPeak -and (Test-Path $xw)) {
  "=== xw_peak (int8 XMX peak, the prompt GEMMs' TOPS and % of it)  $(Get-Date -Format s)" | Add-Content $out
  $p = & $xw --iters 3 2>&1 | Out-String
  $p | Add-Content $out
  $pk = [regex]::Match($p, "int8 XMX peak \(measured\): ([\d.]+) TOPS")
  if ($pk.Success) { "int8 XMX peak: {0} TOPS (measured)" -f $pk.Groups[1].Value }
}
foreach ($d in $Depths) {
  "=== depth $d  $(Get-Date -Format s)" | Add-Content $out
  $o = & $runz $Model --bench-pp $d --n $Gen --prof --ctx ($d + 512 + $Gen) 2>&1 | Out-String
  $o | Add-Content $out
  $pp = [regex]::Match($o, "pp\d+ run 0: ([\d.]+) ms\s+->\s+([\d.]+) tokens/s")
  $attn = 0.0
  foreach ($m in [regex]::Matches($o, "(?m)^\s*attn\w*\s+([\d.]+) ms\s+([\d.]+)%")) { $attn += [double]::Parse($m.Groups[2].Value, $inv) }
  $tg = [regex]::Match($o, "tg\d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s")
  $bw = [regex]::Match($o, "weights read: .*?([\d.]+) GB/s \((\d+) % of")
  $genTxt = if ($tg.Success) { "   gen {0} tok/s" -f $tg.Groups[1].Value } else { "" }
  if ($bw.Success) { $genTxt += " ({0} % of the measured bandwidth, weights only: the KV reads come on top)" -f $bw.Groups[2].Value }
  if ($pp.Success) { "depth {0,7}: prompt {1,8} tok/s   attention kernels {2,5:N1} % of GPU time{3}" -f $d, $pp.Groups[2].Value, $attn, $genTxt }
  else { "depth {0,7}: FAILED: {1}" -f $d, (($o -split "`n" | Where-Object { $_.Trim() } | Select-Object -Last 2) -join " | ") }
}
Remove-Item Env:ARC_BENCH_RUNS -ErrorAction SilentlyContinue
"Results: $out  (attach it to issue #4)"

# Sweeps the prompt GEMM of an iGPU (made for the Arc 140T, issue #5): four bases, XMX with the 128-register file (which
# took the 140T's XMX kernel from 108 to 165 tok/s on the 27B) and DP4a at its own 256, each without and with H8
# (Hadamard-rotated int8, ARC_H8=1); DP4a with 128 registers is measured once (it spills). For each base every tile
# setting is changed alone, the two that gained
# most are crossed in a small grid, and the best set is kept. Then the other H8 forms (per-256 scales, random signs) on
# the faster H8 base, and the KL and perplexity of every best set against the default DP4a path. The engine checks
# every XMX build against DP4a at start and stays on DP4a if they differ ("XMX differed!" in the path column).
#   .\bench\igpu_grf_sweep.ps1 -Model <27B gguf>          about 80 runs (on a 140T roughly 70-80 minutes)
#   [-Quick]     fewer values, no grid (about 45 runs)   [-Tokens 2048]  prompt length
# Settings changed (ARC_GEMM_*; the defaults in brackets): TM tokens per sub-group [16], WGY sub-groups per work-group
# [8], RG token rows per weight slice [4], CG column groups per work-group [1], SPLIT superblocks per split-K piece [17].
# A setting counts as a gain only 2 % ahead of its base (run-to-run drift on a 140T: ~1-6 %).
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [int]$Tokens = 2048,
  [switch]$Quick
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$runz = Join-Path $root "bin\runz.exe"
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found (run from the arcinferzero folder)" }
if (-not (Test-Path $Model)) { throw "model not found: $Model" }
$vars = @("ARC_NO_XMX", "ARC_XMX_FORCE", "ARC_X16", "ARC_X16_TM", "ARC_X16_GRF", "ARC_GEMM_GRF", "ARC_H8", "ARC_H8_SB", "ARC_H8_D", "ARC_GEMM_TM", "ARC_GEMM_WGY",
          "ARC_GEMM_RG", "ARC_GEMM_CG", "ARC_GEMM_SPLIT")
$names = @{ TM = "ARC_GEMM_TM"; WGY = "ARC_GEMM_WGY"; RG = "ARC_GEMM_RG"; CG = "ARC_GEMM_CG"; SPLIT = "ARC_GEMM_SPLIT" }
$inv = [Globalization.CultureInfo]::InvariantCulture

function Invoke-Runz([hashtable]$envs, [string[]]$a) {
  foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  foreach ($k in $envs.Keys) { Set-Item "Env:$k" "$($envs[$k])" }
  $eap = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    return (& $runz @a 2>&1 | Out-String)
  } finally {
    $ErrorActionPreference = $eap
    foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  }
}
# the environment of a base plus tile settings (TM / WGY / RG / CG / SPLIT -> value)
function Env-Of([hashtable]$base, [hashtable]$tiles) {
  $e = $base.Clone()
  foreach ($k in $tiles.Keys) { $e[$names[$k]] = $tiles[$k] }
  $e
}
function Invoke-Bench([hashtable]$envs) {
  $o = Invoke-Runz $envs @($Model, "--bench-pp", $Tokens, "--n", 1)
  $tps = [regex]::Matches($o, "pp\d+ run \d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s") | ForEach-Object { [double]$_.Groups[1].Value }
  $path = if ($o -match "XMX on for the prompt GEMMs") { "XMX" } elseif ($o -match "(?<!f16 )XMX results differ") { "DP4a (XMX differed!)" }
          elseif ($o -match "(?<!f16 )XMX kernels do not build") { "DP4a (XMX did not build!)" } else { "DP4a" }
  if ($o -match "H8 on for the prompt GEMMs") { $path += " + H8" }
  if ($o -match "f16 XMX on for the W8") { $path += " + f16 XMX" }
  elseif ($o -match "f16 XMX results differ") { $path += " (f16 XMX differed!)" }
  elseif ($o -match "f16 XMX kernels do not build") { $path += " (f16 XMX did not build!)" }
  if (-not $tps) {
    $err = ($o -split "`n" | Where-Object { $_ -match "error|cannot|failed|exception" } | Select-Object -First 1)
    return [pscustomobject]@{ tps = 0; path = "failed: $err" }
  }
  [pscustomobject]@{ tps = ($tps | Measure-Object -Maximum).Maximum; path = $path }
}
function Desc([hashtable]$t) { if ($t.Count -eq 0) { "-" } else { ($t.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join " " } }
function Line([string]$what, $r) { [string]::Format($inv, "{0,-40} {1,9:F1} tok/s  {2}", $what, $r.tps, $r.path) }

$grid = [ordered]@{
  TM    = @(32)
  WGY   = if ($Quick) { @(4, 16) } else { @(2, 4, 16) }
  RG    = if ($Quick) { @(2, 8) } else { @(1, 2, 8) }
  CG    = if ($Quick) { @(2) } else { @(2, 4) }
  SPLIT = if ($Quick) { @(0) } else { @(0, 9, 34) }
}
$bases = [ordered]@{
  "XMX GRF128"     = @{ ARC_XMX_FORCE = 1; ARC_GEMM_GRF = 128 }
  "XMX GRF128 H8"  = @{ ARC_XMX_FORCE = 1; ARC_GEMM_GRF = 128; ARC_H8 = 1 }
  "DP4a"           = @{ ARC_NO_XMX = 1 }
  "DP4a H8"        = @{ ARC_NO_XMX = 1; ARC_H8 = 1 }
}
# (DP4a with 128 registers: measured once, not swept. ocloc for arl-h (and the 125H) shows it spilling in every layout,
#  750 scratch messages for the 27B's; on the 125H it ran 6x slower, 120 -> 19 tok/s. DP4a is swept at its 256)
$probes = [ordered]@{
  "DP4a GRF128"    = @{ ARC_NO_XMX = 1; ARC_GEMM_GRF = 128 }
  "DP4a GRF128 H8" = @{ ARC_NO_XMX = 1; ARC_GEMM_GRF = 128; ARC_H8 = 1 }
}

& (Join-Path $root "bin\mgpu_probe.exe") 2>&1 | Select-String -Pattern "^\s+GPU \d+:" | ForEach-Object { $_.Line.Trim() }
"model: " + (Split-Path $Model -Leaf) + "; prompt $Tokens tokens; tok/s = best of 3 passes"
$null = Invoke-Runz @{ ARC_NO_XMX = 1 } @($Model, "--bench-pp", 512, "--n", 1)  # (warm-up)
""
"=== references"
$ref = Invoke-Bench @{ ARC_NO_XMX = 1 }
Line "DP4a default (256 registers)" $ref
foreach ($n in $probes.Keys) { Line "$n (expected to spill)" (Invoke-Bench $probes[$n]) }

$best = [ordered]@{}
foreach ($bn in $bases.Keys) {
  $base = $bases[$bn]
  ""
  "=== $bn"
  $b = Invoke-Bench $base
  Line "base" $b
  $res = @{}   # (setting -> value -> tok/s)
  $top = @{ tps = $b.tps; tiles = @{} }
  foreach ($p in $grid.Keys) {
    $res[$p] = @{}
    foreach ($v in $grid[$p]) {
      $t = @{ $p = $v }
      $r = Invoke-Bench (Env-Of $base $t)
      Line "$p=$v" $r
      $res[$p]["$v"] = $r.tps
      if ($r.tps -gt $top.tps) { $top = @{ tps = $r.tps; tiles = $t } }
    }
  }
  # the settings that gained (2 % past the base), by their best value
  $gains = @($grid.Keys | ForEach-Object {
    $p = $_; $bv = ($res[$p].GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 1)
    if ($bv -and $bv.Value -gt $b.tps * 1.02) { [pscustomobject]@{ p = $p; v = $bv.Name; t = $bv.Value } }
  } | Sort-Object t -Descending)
  if (-not $Quick -and $gains.Count -ge 2) {
    # (the two best settings crossed: each one's two best values, against each other)
    $p1 = $gains[0].p; $p2 = $gains[1].p
    $v1 = @($res[$p1].GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 2 | ForEach-Object { $_.Name })
    $v2 = @($res[$p2].GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 2 | ForEach-Object { $_.Name })
    foreach ($a in $v1) { foreach ($c in $v2) {
      $t = @{ $p1 = $a; $p2 = $c }
      $r = Invoke-Bench (Env-Of $base $t)
      Line ("grid " + (Desc $t)) $r
      if ($r.tps -gt $top.tps) { $top = @{ tps = $r.tps; tiles = $t } }
    } }
  }
  if ($gains.Count -ge 1) {
    # (every gaining setting at its best value together)
    $t = @{}; foreach ($g in $gains) { $t[$g.p] = $g.v }
    if ($t.Count -gt 1) {
      $r = Invoke-Bench (Env-Of $base $t)
      Line ("all gains " + (Desc $t)) $r
      if ($r.tps -gt $top.tps) { $top = @{ tps = $r.tps; tiles = $t } }
    }
  }
  $again = Invoke-Bench $base
  Line "base again (drift)" $again
  [string]::Format($inv, "  best: {0} -> {1:F1} tok/s ({2:+0.0;-0.0} % against DP4a default)", (Desc $top.tiles), $top.tps,
                   100.0 * ($top.tps / $ref.tps - 1))
  $best[$bn] = $top
}

""
"=== f16 XMX (ARC_X16=1, gemm_x16.cl at 8 lanes: f16 weights and activations, the DPAS summing the whole row)"
$x16 = @{ ARC_X16 = 1 }
$xb = Invoke-Bench $x16
Line "base (32 tokens, 256 registers)" $xb
$xtop = @{ tps = $xb.tps; env = $x16 }
$xgrid = [ordered]@{
  "X16_TM=16" = @{ ARC_X16_TM = 16 }; "X16_TM=8" = @{ ARC_X16_TM = 8 }; "X16_GRF=128" = @{ ARC_X16_GRF = 128 }
  "WGY=2" = @{ ARC_GEMM_WGY = 2 }; "WGY=4" = @{ ARC_GEMM_WGY = 4 }; "RG=2" = @{ ARC_GEMM_RG = 2 }; "RG=8" = @{ ARC_GEMM_RG = 8 }
  "SPLIT=0" = @{ ARC_GEMM_SPLIT = 0 }
}
if (-not $Quick) { $xgrid["WGY=16"] = @{ ARC_GEMM_WGY = 16 }; $xgrid["X16_TM=16 X16_GRF=128"] = @{ ARC_X16_TM = 16; ARC_X16_GRF = 128 } }
foreach ($n in $xgrid.Keys) {
  $e = $x16.Clone(); foreach ($k in $xgrid[$n].Keys) { $e[$k] = $xgrid[$n][$k] }
  $r = Invoke-Bench $e
  Line $n $r
  if ($r.tps -gt $xtop.tps) { $xtop = @{ tps = $r.tps; env = $e } }
}
Line "base again (drift)" (Invoke-Bench $x16)
[string]::Format($inv, "  best: {0} -> {1:F1} tok/s ({2:+0.0;-0.0} % against DP4a default)",
                 (($xtop.env.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join " "), $xtop.tps,
                 100.0 * ($xtop.tps / $ref.tps - 1))

""
"=== other H8 forms, on the faster H8 base with its best tiles"
$hb = if ($best["XMX GRF128 H8"].tps -ge $best["DP4a H8"].tps) { "XMX GRF128 H8" } else { "DP4a H8" }
$h8sets = [ordered]@{
  "$hb (one scale per column and token)" = (Env-Of $bases[$hb] $best[$hb].tiles)
  "$hb + ARC_H8_SB=1 (per-256 scales)"   = (Env-Of ($bases[$hb] + @{ ARC_H8_SB = 1 }) $best[$hb].tiles)
  "$hb + ARC_H8_D=1 (random signs)"      = (Env-Of ($bases[$hb] + @{ ARC_H8_D = 1 }) $best[$hb].tiles)
}
foreach ($n in $h8sets.Keys) { Line $n (Invoke-Bench $h8sets[$n]) }

""
"=== accuracy against the default DP4a path: KL over 4 perplexity chunks of USAGE.md"
$kl = Join-Path $env:TEMP "igpu_grf_sweep_base.kl"
$text = Join-Path $root "USAGE.md"
$o = Invoke-Runz @{ ARC_NO_XMX = 1 } @($Model, "--perplexity", $text, "--ppl-chunks", 4, "--ppl-kl-save", $kl)
$p0 = [regex]::Match($o, "perplexity: ([\d.]+)")
"DP4a default: perplexity " + $(if ($p0.Success) { $p0.Groups[1].Value } else { "?" })
$klsets = [ordered]@{}
foreach ($bn in $bases.Keys) { $klsets["$bn, best tiles"] = (Env-Of $bases[$bn] $best[$bn].tiles) }
$klsets["f16 XMX, best"] = $xtop.env
$klsets["$hb + ARC_H8_SB=1"] = $h8sets["$hb + ARC_H8_SB=1 (per-256 scales)"]
$klsets["$hb + ARC_H8_D=1"] = $h8sets["$hb + ARC_H8_D=1 (random signs)"]
foreach ($n in $klsets.Keys) {
  $o = Invoke-Runz $klsets[$n] @($Model, "--perplexity", $text, "--ppl-chunks", 4, "--ppl-kl", $kl)
  $line = ($o -split "`n" | Where-Object { $_ -match "KL vs base" } | Select-Object -Last 1)
  $p = [regex]::Match($o, "perplexity: ([\d.]+)")
  "{0,-34} {1}  perplexity {2}" -f $n, $(if ($line) { $line.Trim() } else { "no KL line" }), $(if ($p.Success) { $p.Groups[1].Value } else { "?" })
}
Remove-Item $kl -ErrorAction SilentlyContinue

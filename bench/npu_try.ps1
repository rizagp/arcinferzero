# The NPU next to the GPU for prompts (--npu on, docs/npu-experiment.md) on this machine: the prompt speed with the NPU
# off, on, and at other shares, and the KL against NPU-off. Made for Core Ultra laptops with a newer NPU than the one it
# was tuned on (Lunar Lake, Panther Lake: Arc 140V / B390 and their NPUs); paste the whole output into a GitHub issue.
#   .\bench\npu_try.ps1 -Model <a dense model: Qwen3.8 27B, Gemma 4 12B> [-Tokens 4096,8192]
# The shares were set on a Core Ultra 5 125H, whose GPU does ~8.6 int8 TOPS: 1/3 of the FFN gate / up columns, 27.5 %
# of down, 29 % of DeltaNet's qkv go to the NPU, where both finish together. A GPU with matrix engines (XMX) is several
# times faster, so with those shares it would wait for the NPU: the sweep scales all three together (scale 1 = those
# shares). "waited" is the engine's own count of the time the GPU waited for the NPU after its part, over the whole
# run: it never reaches 0 (each NPU run has a fixed latency: the 125H's 27B at 4K waited 27 s at x0.3, 47 s at x1,
# 73 s at x1.3), so compare it between rows; the fastest row is the answer. The 125H, 2026-11-13: off 128.4 tok/s,
# x0.3 / 0.5 / 0.7 / 1 / 1.3 129.3 / 137.9 / 133.5 / 151.4 / 125.8, KL 0.0012-0.0034. The NPU turns H8 off (both
# convert the same weights), so on a GPU where H8 is the default the comparison includes that.
# KL: two 4,096-token perplexity chunks of USAGE.md (the NPU takes chunks of 2,048+ rows), against the NPU off; the
# engine's limit is 0.015.
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [int[]]$Tokens = @(4096, 8192)
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$runz = Join-Path $root "bin\runz.exe"
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found (run from the inferzero folder)" }
if (-not (Test-Path $Model)) { throw "model not found: $Model" }
$vars = @("ARC_NPU_SHARE", "ARC_NPU_DOWN", "ARC_NPU_QKV")
$inv = [Globalization.CultureInfo]::InvariantCulture
$tmp = Join-Path ([IO.Path]::GetTempPath()) "inferzero-npu-try"
New-Item -ItemType Directory $tmp -Force | Out-Null
$text = Join-Path $root "USAGE.md"

function Invoke-Runz([string]$npu, [hashtable]$envs, [string[]]$a) {
  foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  $eap = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"  # (runz's stderr lines are output, not errors)
    foreach ($k in $envs.Keys) { Set-Item "Env:$k" "$($envs[$k])" }
    return (& $runz $Model @a --npu $npu 2>&1 | ForEach-Object { "$_" })
  } finally {
    $ErrorActionPreference = $eap
    foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  }
}
# (scale s: the three shares at s times the 125H's)
function Shares([double]$s) {
  @{ ARC_NPU_SHARE = (1.0 / 3.0 * $s).ToString("0.###", $inv); ARC_NPU_DOWN = (0.275 * $s).ToString("0.###", $inv);
     ARC_NPU_QKV = (0.29 * $s).ToString("0.###", $inv) }
}
$sets = [ordered]@{
  "NPU off"         = @{ npu = "off"; e = @{} }
  "NPU on"          = @{ npu = "on"; e = @{} }
  "on, shares x0.3" = @{ npu = "on"; e = (Shares 0.3) }
  "on, shares x0.5" = @{ npu = "on"; e = (Shares 0.5) }
  "on, shares x0.7" = @{ npu = "on"; e = (Shares 0.7) }
  "on, shares x1.3" = @{ npu = "on"; e = (Shares 1.3) }
}
"inferzero npu_try: $(Split-Path $Model -Leaf), $(Get-Date -Format s)"
& (Join-Path $root "bin\mgpu_probe.exe") 2>&1 | Select-String -Pattern "^\s+GPU \d+:" | ForEach-Object { $_.Line.Trim() }
# (what the engine says about the NPU: found or not, the shares, the memory; the first run compiles its graphs)
$o = Invoke-Runz "on" @{} @("--bench-pp", "$($Tokens[0])", "--n", "0")
$o | Where-Object { $_ -match "ARC_NPU|NPU|H8|XMX on" } | Select-Object -First 12 | ForEach-Object { "  $_" }
if ($o -match "no NPU|only; off|off for this model") { "The NPU is not used with this model or machine (see above): nothing to measure."; exit 0 }

"--- prompt speed (best of runz's 3 passes); waited = the GPU waiting for the NPU, whole run"
$rows = @()
foreach ($t in $Tokens) {
  foreach ($n in $sets.Keys) {
    $o = Invoke-Runz $sets[$n].npu $sets[$n].e @("--bench-pp", "$t", "--n", "0")
    $r = @($o | ForEach-Object { if ($_ -match 'pp\d+ run \d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s') { [double]::Parse($Matches[1], $inv) } })
    $best = if ($r.Count) { ($r | Measure-Object -Maximum).Maximum } else { 0 }
    $w = ($o | Where-Object { $_ -match "ARC_NPU: \d+ NPU runs" } | Select-Object -Last 1)
    $wt = if ($w -match "(\d+) NPU runs, (\d+) ms waited") { "$($Matches[1]) NPU runs, waited $($Matches[2]) ms" } elseif ($sets[$n].npu -eq "off") { "" } else { "no NPU runs" }
    $err = if ($best -eq 0) { "failed: " + (($o | Where-Object { $_ -match "error|cannot|failed|exception" } | Select-Object -First 1)) } else { "" }
    [string]::Format($inv, "{0,6} tokens  {1,-16} {2,9:F1} tok/s  {3}{4}", $t, $n, $best, $wt, $err)
  }
}
"--- accuracy: KL against NPU off, 2 perplexity chunks of 4,096 tokens (USAGE.md)"
$kl = Join-Path $tmp "off.kl"
Remove-Item $kl -ErrorAction SilentlyContinue
$o = Invoke-Runz "off" @{} @("--perplexity", $text, "--ppl-ctx", "4096", "--ppl-chunks", "2", "--chunk", "4096", "--ppl-kl-save", $kl)
$p = [regex]::Match(($o -join "`n"), "perplexity: ([\d.]+)")
"  NPU off: perplexity $(if ($p.Success) { $p.Groups[1].Value } else { '?' })"
foreach ($n in $sets.Keys) {
  if ($sets[$n].npu -eq "off") { continue }
  $o = Invoke-Runz $sets[$n].npu $sets[$n].e @("--perplexity", $text, "--ppl-ctx", "4096", "--ppl-chunks", "2", "--chunk", "4096", "--ppl-kl", $kl)
  $line = ($o | Where-Object { $_ -match "KL vs base" } | Select-Object -Last 1)
  $p = [regex]::Match(($o -join "`n"), "perplexity: ([\d.]+)")
  "  {0,-16} {1}  perplexity {2}" -f $n, $(if ($line) { $line.Trim() } else { "no KL line" }), $(if ($p.Success) { $p.Groups[1].Value } else { "?" })
}
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue

# The H8 prompt GEMMs with wider XMX tiles (gemm_xw, ARC_GEMM_XW=1, QUEUE 33.4) against the default, on a GPU with
# 16-lane XMX (made for the Arc Pro B60). Paste the whole output into the GitHub issue.
#   .\bench\xw_sweep.ps1 -Model <Qwen3.8 27B or another Qwen dense gguf> [-Tokens 512,4096,16384]
# For each tile shape (tokens x columns per sub-group: 32 x 64 the default, 32 x 32, 16 x 64, 32 x 16) and the
# default path: the prompt speed at each length (runz --bench-pp, the best of its 3 runs), and the logits of a real
# prompt against the default path's, which must be identical bit for bit (the engine also checks gemm_xw at start,
# bit for bit: "the H8 GEMMs on XMX with ... tiles" in the log, or "gemm_xw differs" and the default kept).
# Then one --prof run of the default and of the fastest shape at 4096 tokens: the GPU time by kernel.
# H8 is on by default on discrete cards for the Qwen dense models; on another GPU set $env:ARC_H8 = 1 first.
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [int[]]$Tokens = @(512, 4096, 16384)
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$runz = Join-Path $root "bin\runz.exe"
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found (run from the inferzero folder)" }
if (-not (Test-Path $Model)) { throw "model not found: $Model" }
$vars = @("ARC_GEMM_XW", "ARC_GEMM_XW_TM", "ARC_GEMM_XW_NC", "ARC_GEMM_XW_GRF")
$inv = [Globalization.CultureInfo]::InvariantCulture
$tmp = Join-Path ([IO.Path]::GetTempPath()) "inferzero-xw-sweep"
New-Item -ItemType Directory $tmp -Force | Out-Null
$text = Join-Path $tmp "prompt.txt"
[IO.File]::WriteAllText($text, ((Get-Content (Join-Path $root "USAGE.md") -TotalCount 150) -join "`n"), (New-Object Text.UTF8Encoding $false))

function Invoke-Runz([hashtable]$envs, [string[]]$a) {
  foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  $eap = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"  # (runz's stderr lines are output, not errors)
    foreach ($k in $envs.Keys) { Set-Item "Env:$k" "$($envs[$k])" }
    return (& $runz $Model @a 2>&1 | ForEach-Object { "$_" })
  } finally {
    $ErrorActionPreference = $eap
    foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  }
}
function Best-PP($o) {
  $r = @($o | ForEach-Object { if ($_ -match 'pp\d+ run \d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s') { [double]::Parse($Matches[1], $inv) } })
  if ($r.Count) { ($r | Measure-Object -Maximum).Maximum } else { $null }
}

$sets = [ordered]@{
  "default"  = @{}
  "xw 32x64" = @{ ARC_GEMM_XW = 1 }
  "xw 32x32" = @{ ARC_GEMM_XW = 1; ARC_GEMM_XW_NC = 2 }
  "xw 16x64" = @{ ARC_GEMM_XW = 1; ARC_GEMM_XW_TM = 16 }
  "xw 32x16" = @{ ARC_GEMM_XW = 1; ARC_GEMM_XW_NC = 1 }
}
"inferzero xw_sweep: $(Split-Path $Model -Leaf), $(Get-Date -Format s)"
$res = @{}
$base = $null
foreach ($name in $sets.Keys) {
  $e = $sets[$name]
  # (the logits of a real prompt: every row of a 4-token answer, against the default's)
  $d = Join-Path $tmp ("logits_" + ($name -replace '\W', '_') + ".bin")
  $o = Invoke-Runz $e @("--text-file", $text, "--n", "4", "--dump-logits", $d)
  $note = ($o | Where-Object { $_ -match "gemm_xw|H8 on|XMX on" }) -join " | "
  $h = if (Test-Path $d) { (Get-FileHash $d).Hash } else { "" }
  if ($name -eq "default") { $base = $h }
  $same = if (-not $h) { "no logits" } elseif ($h -eq $base) { "identical" } else { "DIFFER" }
  "[$name] $note"
  $row = [ordered]@{ set = $name; logits = $same }
  foreach ($t in $Tokens) {
    $o = Invoke-Runz $e @("--bench-pp", "$t", "--n", "4")
    $row["pp$t"] = Best-PP $o
  }
  $res[$name] = $row
}
""
"| set | logits vs default | " + (($Tokens | ForEach-Object { "pp$_ tok/s" }) -join " | ") + " |"
"|---|---|" + (($Tokens | ForEach-Object { "---:" }) -join "|") + "|"
foreach ($name in $sets.Keys) {
  $r = $res[$name]
  "| $name | $($r.logits) | " + (($Tokens | ForEach-Object { $v = $r["pp$_"]; if ($null -eq $v) { "-" } else { $v.ToString("N1", $inv) } }) -join " | ") + " |"
}
# the GPU time by kernel at 4096 tokens: the default and the fastest xw shape there
$mid = if ($Tokens -contains 4096) { 4096 } else { $Tokens[0] }
$fast = $sets.Keys | Where-Object { $_ -ne "default" -and $res[$_].logits -eq "identical" -and $res[$_]["pp$mid"] } |
        Sort-Object { - $res[$_]["pp$mid"] } | Select-Object -First 1
foreach ($name in @("default") + @($fast | Where-Object { $_ })) {
  ""
  "== --prof, $name, $mid tokens"
  Invoke-Runz $sets[$name] @("--bench-pp", "$mid", "--n", "4", "--prof") | Where-Object { $_ -match "%|ms|tok" } | Select-Object -First 40
}
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue

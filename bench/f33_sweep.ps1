# The chunked DeltaNet (QUEUE F33.1) and the prompt attention on XMX (F33.2) on this GPU, for every model given:
# prompt speed, the kernels' own time, and accuracy (DeltaNet: KL against the sequential kernel over 4 perplexity
# chunks of USAGE.md; attention: logits bit for bit against attn_q8). Made for the Pro B60 (issue #4) and the Arc 140T
# (issue #5); runs on any GPU. Paste the whole output into the GitHub issue.
#   .\bench\f33_sweep.ps1 -Models C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf, C:\hf\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
#   [-Sizes 512,4096,16384,65536]  prompt lengths (attention and DeltaNet weigh more in long prompts)
#   [-Quick]  512 and 4096 only
#   [-Emulate]  also the XMX kernels with DPAS emulated (a GPU without XMX: correctness, not speed)
#   [-OldRunz <path to another build's runz.exe>]  the default path's logits against that build, bit for bit
# DeltaNet modes (ARC_GDN_CHUNK; models without DeltaNet, e.g. Gemma 4, skip them):
#   0  gdn_recur, the sequential kernel (the default)
#   1  gdn_chunk: chunks of 32 tokens, 4 work-items per state row, local rows read element by element
#   2  gdn_chunk, local rows read 16 floats at a time
#   3  on XMX: gdn_chunk_x (16-lane DPAS: B580 / B60) or gdn_chunk_x8 (8 lanes: the 140T), checked at start against
#      gdn_chunk; when they differ or there is no XMX it runs gdn_recur ("path" says which ran)
# Attention (ARC_ATTN_XMX=1): attn_q8's integer dot products as int8 DPAS, bit-identical; 16-lane DPAS and Qwen models
# only (on the 140T it stays off: its attention kernels run 16-lane sub-groups, its DPAS takes 8).
param(
  [Parameter(Mandatory = $true)][string[]]$Models,
  [int[]]$Sizes = @(512, 4096, 16384, 65536),
  [switch]$Quick,
  [switch]$Emulate,
  [string]$OldRunz = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$runz = Join-Path $root "bin\runz.exe"
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found (run from the inferzero folder)" }
if ($Quick) { $Sizes = @(512, 4096) }
$inv = [Globalization.CultureInfo]::InvariantCulture
$vars = @("ARC_GDN_CHUNK", "ARC_GDN_XEMU", "ARC_ATTN_XMX", "ARC_ATTN_XEMU")
$text = Join-Path $root "USAGE.md"
$tmp = Join-Path $env:TEMP "inferzero-f33"
New-Item -ItemType Directory -Force $tmp | Out-Null
# (a second program on the GPU spoils the numbers and has caused device-lost errors: wait for them to end)
if (Get-Process serverz, runz -ErrorAction SilentlyContinue) {
  "waiting: serverz / runz is running (stop the server first)"
  while (Get-Process serverz, runz -ErrorAction SilentlyContinue) { Start-Sleep 30 }
}

function Invoke-Runz([hashtable]$envs, [string[]]$a, [string]$exe = $runz) {
  $eap = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
    foreach ($k in $envs.Keys) { Set-Item "Env:$k" $envs[$k] }
    return (& $exe @a 2>&1 | Out-String)
  } finally {
    $ErrorActionPreference = $eap
    foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  }
}
# which DeltaNet / attention kernel a run took, from its start-up lines
function Path-Of([string]$o, [hashtable]$e) {
  $p = @()
  if ($e.ARC_GDN_CHUNK) {
    $x = [regex]::Match($o, "chunked DeltaNet on XMX, (\d+) lanes")
    if ($e.ARC_GDN_CHUNK -eq "3" -and $x.Success) { $p += "gdn XMX $($x.Groups[1].Value)-lane" + $(if ($e.ARC_GDN_XEMU) { " (emulated)" } else { "" }) }
    # (with its size: ~1e-3 says precision (an f16 DPAS that drops the low halves' bits), ~1 a wrong layout; the 140T
    #  reported only the word, issue #5)
    elseif ($o -match "chunked DeltaNet on XMX differs from gdn_chunk \(([^ ]+) of the result\)") { $p += "gdn_recur (XMX DIFFERED by $($Matches[1]))" }
    elseif ($o -match "chunked DeltaNet on XMX differs") { $p += "gdn_recur (XMX DIFFERED)" }
    elseif ($o -match "chunked DeltaNet on XMX does not build") { $p += "gdn_recur (XMX did not build)" }
    elseif ($e.ARC_GDN_CHUNK -eq "3") { $p += "gdn_recur (no XMX here)" }
    else { $p += "gdn_chunk" + $(if ($e.ARC_GDN_CHUNK -eq "2") { " vec" } else { "" }) }
  }
  if ($e.ARC_ATTN_XMX) {
    if ($o -match "prompt attention on XMX( \(DPAS emulated: tests\))? \(bit-identical") { $p += "attn XMX" + $(if ($e.ARC_ATTN_XEMU) { " (emulated)" } else { "" }) }
    elseif ($o -match "prompt attention on XMX differs") { $p += "attn_q8 (XMX DIFFERED)" }
    elseif ($o -match "prompt attention on XMX does not build") { $p += "attn_q8 (XMX did not build)" }
    else { $p += "attn_q8 (XMX not used here)" }
  }
  if (-not $p) { "default" } else { $p -join " + " }
}
function Invoke-Bench([string]$model, [hashtable]$e, [int]$pp, [switch]$Prof) {
  $a = @($model, "--bench-pp", $pp, "--n", 1)
  if ($Prof) { $a += "--prof" }
  $o = Invoke-Runz $e $a
  $tps = [regex]::Matches($o, "pp\d+ run \d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s") | ForEach-Object { [double]$_.Groups[1].Value }
  $k = [regex]::Match($o, "\n\s+(gdn_recur|gdn_chunk_x8|gdn_chunk_x|gdn_chunk)\s+([\d.]+) ms\s+([\d.]+)%")
  $at = [regex]::Match($o, "\n\s+attention\s+([\d.]+) ms\s+([\d.]+)%")
  $kern = @()
  if ($k.Success) { $kern += "{0} {1} ms ({2} %)" -f $k.Groups[1].Value, $k.Groups[2].Value, $k.Groups[3].Value }
  if ($at.Success) { $kern += "attention {0} ms ({1} %)" -f $at.Groups[1].Value, $at.Groups[2].Value }
  if (-not $tps) {
    $err = ($o -split "`n" | Where-Object { $_ -match "error|cannot|failed|exception" } | Select-Object -First 1)
    return [pscustomobject]@{ tps = 0; path = "FAILED: $err"; kern = ""; gdn = $false }
  }
  [pscustomobject]@{ tps = ($tps | Measure-Object -Maximum).Maximum; path = (Path-Of $o $e); kern = ($kern -join ", "); gdn = $o -match "gdn_recur|gdn_chunk" }
}
function Line([string]$what, $r) { [string]::Format($inv, "{0,-34} {1,9:F1} tok/s  {2,-34} {3}", $what, $r.tps, $r.path, $r.kern) }
function Same-File([string]$a, [string]$b) {
  (Test-Path $a) -and (Test-Path $b) -and (Get-Item $a).Length -gt 0 -and (Get-FileHash $a).Hash -eq (Get-FileHash $b).Hash
}

& (Join-Path $root "bin\mgpu_probe.exe") 2>&1 | Select-String -Pattern "^\s+GPU \d+:" | ForEach-Object { $_.Line.Trim() }
"prompt sizes: $($Sizes -join ', '); tok/s = best of 3 passes" + $(if ($Emulate) { "; XMX also emulated" } else { "" })
foreach ($m in $Models) {
  ""
  if (-not (Test-Path $m)) { "=== skipped (not found): $m"; continue }
  "=== " + (Split-Path $m -Leaf)
  $probe = Invoke-Bench $m @{} 512 -Prof  # (warm-up, and whether the model has DeltaNet layers)
  $hasGdn = $probe.gdn
  $sets = [ordered]@{ "default" = @{} }
  if ($hasGdn) {
    $sets["gdn 1"] = @{ ARC_GDN_CHUNK = "1" }
    $sets["gdn 2"] = @{ ARC_GDN_CHUNK = "2" }
    $sets["gdn 3 (XMX)"] = @{ ARC_GDN_CHUNK = "3" }
    if ($Emulate) { $sets["gdn 3 emulated"] = @{ ARC_GDN_CHUNK = "3"; ARC_GDN_XEMU = "1" } }
  } else { "  (no DeltaNet layers: the chunked DeltaNet does not apply)" }
  $sets["attn XMX"] = @{ ARC_ATTN_XMX = "1" }
  if ($Emulate) { $sets["attn XMX emulated"] = @{ ARC_ATTN_XMX = "1"; ARC_ATTN_XEMU = "1" } }
  if ($hasGdn) { $sets["gdn 3 + attn XMX"] = @{ ARC_GDN_CHUNK = "3"; ARC_ATTN_XMX = "1" } }
  "--- prompt speed"
  foreach ($pp in $Sizes) {
    foreach ($n in $sets.Keys) {
      if ($n -match "emulated" -and $pp -gt 4096) { continue }  # (emulation is slow: correctness only)
      Line ("{0,5} tokens  {1}" -f $pp, $n) (Invoke-Bench $m $sets[$n] $pp)
    }
  }
  Line ("{0,5} tokens  default again (drift)" -f $Sizes[-1]) (Invoke-Bench $m @{} $Sizes[-1])
  "--- the kernels' GPU time, 4096-token prompt (--prof)"
  foreach ($n in $sets.Keys) { Line $n (Invoke-Bench $m $sets[$n] 4096 -Prof) }
  "--- accuracy"
  if ($hasGdn) {
    $kl = Join-Path $tmp "base.kl"
    $o = Invoke-Runz @{} @($m, "--perplexity", $text, "--ppl-chunks", 4, "--ppl-kl-save", $kl)
    $ppl = [regex]::Match($o, "perplexity: ([\d.]+)")
    "DeltaNet, KL against gdn_recur over 4 perplexity chunks of USAGE.md (gdn_recur: perplexity $($ppl.Groups[1].Value))"
    foreach ($n in $sets.Keys) {
      if ($n -notmatch "^gdn") { continue }
      $o = Invoke-Runz $sets[$n] @($m, "--perplexity", $text, "--ppl-chunks", 4, "--ppl-kl", $kl)
      $line = ($o -split "`n" | Where-Object { $_ -match "KL vs base" } | Select-Object -Last 1)
      $ppl = [regex]::Match($o, "perplexity: ([\d.]+)")
      "  {0,-20} {1}  perplexity {2}  [{3}]" -f $n, $(if ($line) { $line.Trim() } else { "no KL line" }),
        $(if ($ppl.Success) { $ppl.Groups[1].Value } else { "?" }), (Path-Of $o $sets[$n])
    }
    Remove-Item $kl -ErrorAction SilentlyContinue
  }
  $d0 = Join-Path $tmp "default.bin"
  Remove-Item $d0 -ErrorAction SilentlyContinue
  $null = Invoke-Runz @{} @($m, "--text-file", $text, "--n", 16, "--dump-logits", $d0)
  foreach ($n in $sets.Keys) {
    if ($n -notmatch "^attn") { continue }
    $d1 = Join-Path $tmp "attn.bin"
    Remove-Item $d1 -ErrorAction SilentlyContinue
    $o = Invoke-Runz $sets[$n] @($m, "--text-file", $text, "--n", 16, "--dump-logits", $d1)
    "attention, logits (USAGE.md as the prompt, 16 tokens), {0,-18}: {1}  [{2}]" -f $n,
      $(if (Same-File $d0 $d1) { "bit-identical to the default" } else { "DIFFER (or a run failed)" }), (Path-Of $o $sets[$n])
  }
  if ($OldRunz) {
    $d2 = Join-Path $tmp "old.bin"
    Remove-Item $d2 -ErrorAction SilentlyContinue
    $null = Invoke-Runz @{} @($m, "--text-file", $text, "--n", 16, "--dump-logits", $d2) $OldRunz
    "default path, logits against $OldRunz : " + $(if (Same-File $d0 $d2) { "bit-identical" } else { "DIFFER (or a run failed)" })
  }
}
Remove-Item $tmp -Recurse -ErrorAction SilentlyContinue

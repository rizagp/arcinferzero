# Benchmarks for the website: per model, prompt processing and generation from short prompts to 131K tokens, the
# generation modes (plain, MTP, DFlash2), and the KL against an f32 CPU reference. Writes one JSON object per
# measurement to <Out>\results.jsonl and the tables to <Out>\summary.md.
#   .\bench\website_bench.ps1 [-Models a.gguf,b.gguf] [-Sizes 128,512,2048,8192,32768,131072] [-Out bench\results\website]
#   [-SkipKl] [-SkipModes] [-Gen 64] [-Cooldown 20]
# A. Prompt sizes: runz --bench-pp N --n Gen. Prompt tok/s (best of 2 runs up to 8K, one run above), time to the first
#    token (N / prompt tok/s), and generation tok/s at that depth (Gen tokens after the N-token prompt), plain decoding.
# B. Generation modes, short context: a 256-token coding answer, plain / --mtp / --dflash (when the model has an MTP
#    block or a DFlash2 drafter is in the Hugging Face cache), best of 2.
# C. KL: 480 tokens of USAGE.md through tools\ref_forward (f32 activations, dequantized weights; kept in
#    <Out>\ref so later runs reuse it) and runz --kl-ref: mean / max KL and how often the top token agrees.
# Without -Models: the usual models found in the Hugging Face cache and %USERPROFILE%\.arcinferzero\zero\models.
# Every run checks for other GPU programs first (a second program on the GPU spoils the numbers).
param(
  [string[]]$Models = @(),
  [int[]]$Sizes = @(128, 512, 2048, 8192, 32768, 131072),
  [string]$Out = "bench\results\website",
  [int]$Gen = 64,
  [int]$Cooldown = 20,
  [switch]$SkipKl,
  [switch]$SkipModes
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$runz = Join-Path $root "bin\runz.exe"
$ref = Join-Path $root "bin\ref_forward.exe"
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found" }
New-Item -ItemType Directory -Force $Out, (Join-Path $Out "ref") | Out-Null
$jsonl = Join-Path $Out "results.jsonl"
$hub = Join-Path $env:USERPROFILE ".cache\huggingface\hub"
$pkg = Join-Path $env:USERPROFILE ".arcinferzero\zero\models"
$inv = [Globalization.CultureInfo]::InvariantCulture

if (-not $Models.Count) {
  $names = "Qwen3.5-4B-UD-Q4_K_XL.gguf", "Qwen3.5-9B-UD-Q4_K_XL.gguf", "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
    "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.gguf", "gemma-4-E4B_q4_0-it.gguf",
    "gemma-4-12b-it-qat-q4_0.gguf", "gemma-4-26b-a4b-it-qat-q4_0.gguf"
  foreach ($n in $names) {
    $f = @(Get-ChildItem $pkg -Recurse -Filter $n -ErrorAction SilentlyContinue) + @(Get-ChildItem $hub -Recurse -Filter $n -ErrorAction SilentlyContinue) |
      Select-Object -First 1
    if ($f) { $Models += $f.FullName }
  }
}
# DFlash2 drafters in the Hugging Face cache, by model
$drafters = @{ "27B" = "models--z-lab--Qwen3.8-27B-DFlash2"; "35B-A3B" = "models--incoai--Qwen3.6-35B-A3B-DFlash2";
               "Flash-Next" = "models--incoai--Qwen3.8-Flash-Next-DFlash2" }
function Drafter([string]$model) {
  foreach ($k in $drafters.Keys) {
    if ($model -like "*$k*") {
      $d = Get-ChildItem (Join-Path $hub "$($drafters[$k])\snapshots") -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
      if ($d -and (Test-Path (Join-Path $d.FullName "config.json"))) { return $d.FullName }
    }
  }
  return $null
}
function WaitFree {
  while (Get-Process serverz, runz, zero, zdec, ref_forward -ErrorAction SilentlyContinue) { Start-Sleep 30 }
  if ($Cooldown -gt 0) { Start-Sleep $Cooldown }
}
function Run([string[]]$a, [hashtable]$envs = @{}) {
  WaitFree
  foreach ($k in $envs.Keys) { Set-Item "Env:$k" $envs[$k] }
  $eap = $ErrorActionPreference
  try { $ErrorActionPreference = "Continue"; return (& $runz @a 2>&1 | Out-String) }
  finally { $ErrorActionPreference = $eap; foreach ($k in $envs.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue } }
}
function Save($obj) { ($obj | ConvertTo-Json -Compress) | Add-Content -Path $jsonl -Encoding utf8 }
function Num($o, [string]$re) { $m = [regex]::Match($o, $re); if ($m.Success) { [double]::Parse($m.Groups[1].Value, $inv) } else { $null } }

$q = Join-Path ([IO.Path]::GetTempPath()) "arcinferzero-website-q.txt"
$text = "<|im_start|>user`nWrite a Python module with a class LRUCache (get, put, a capacity, O(1) operations using " +
        "a dict and a doubly linked list), docstrings, and unit tests with unittest.<|im_end|>`n<|im_start|>assistant`n" +
        "<think>`n`n</think>`n`n"
[IO.File]::WriteAllText($q, $text, (New-Object Text.UTF8Encoding $false))
$kltext = Join-Path ([IO.Path]::GetTempPath()) "arcinferzero-website-kl.txt"
[IO.File]::WriteAllText($kltext, (Get-Content (Join-Path $root "USAGE.md") -Raw).Substring(0, 2600))

$gpu = (& (Join-Path $root "bin\mgpu_probe.exe") 2>&1 | Select-String -Pattern "^\s+GPU 0:" | Select-Object -First 1)
$gpu = if ($gpu) { $gpu.Line.Trim() } else { "unknown GPU" }
$commit = (git rev-parse --short HEAD 2>$null)
$stamp = Get-Date -Format s
"# website benchmark $stamp, commit $commit, $gpu" | Add-Content (Join-Path $Out "log.txt")

foreach ($m in $Models) {
  if (-not (Test-Path $m)) { "skipped (not found): $m"; continue }
  $name = [IO.Path]::GetFileNameWithoutExtension($m)
  "=== $name"
  # ---- A. prompt sizes
  foreach ($n in $Sizes) {
    $runs = if ($n -le 8192) { "2" } else { "1" }
    $o = Run @($m, "--bench-pp", $n, "--n", $Gen) @{ ARC_BENCH_RUNS = $runs }
    $pps = [regex]::Matches($o, "pp\d+ run \d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s") | ForEach-Object { [double]::Parse($_.Groups[1].Value, $inv) }
    $pp = if ($pps) { ($pps | Measure-Object -Maximum).Maximum } else { $null }
    $tg = Num $o "tg\d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s"
    $err = if (-not $pp) { ($o -split "`n" | Where-Object { $_ -match "error|cannot|failed|exception|out of" } | Select-Object -First 1) } else { $null }
    Save @{ time = $stamp; commit = $commit; gpu = $gpu; model = $name; test = "size"; tokens = $n; pp = $pp; tg = $tg;
            ttft_s = $(if ($pp) { [math]::Round($n / $pp, 2) } else { $null }); error = $err }
    [string]::Format($inv, "  {0,7} tokens: prompt {1,8:F1} tok/s, first token {2,7:F2} s, generation {3,6:F2} tok/s {4}", $n, $pp,
                     $(if ($pp) { $n / $pp } else { 0 }), $tg, $err)
  }
  # ---- B. generation modes
  if (-not $SkipModes) {
    $dir = Drafter $m
    $modes = [ordered]@{ "plain" = @() ; "mtp" = @("--mtp") }
    if ($dir) { $modes["dflash"] = @("--dflash", $dir, "--draft", "dflash"); $modes["mtp+dflash"] = @("--dflash", $dir, "--draft", "auto") }
    foreach ($k in $modes.Keys) {
      $best = $null; $acc = $null
      foreach ($i in 1, 2) {
        $o = Run (@($m, "--text-file", $q, "--n", "256") + $modes[$k])
        $t = Num $o "generation: \d+ tokens in [\d.]+ ms \(([\d.]+) t/s\)"
        if ($t -and ($null -eq $best -or $t -gt $best)) { $best = $t; $acc = Num $o "\(([\d.]+) %\), [\d.]+ tokens per main-model pass" }
      }
      if ($k -eq "mtp" -and $null -eq $best) { continue }  # (no MTP block)
      Save @{ time = $stamp; commit = $commit; gpu = $gpu; model = $name; test = "mode"; mode = $k; tg = $best; accepted_pct = $acc }
      [string]::Format($inv, "  {0,-11} {1,6:F2} tok/s {2}", $k, $best, $(if ($acc) { "($acc % of drafts accepted)" } else { "" }))
    }
  }
  # ---- C. KL against f32
  if (-not $SkipKl -and (Test-Path $ref)) {
    $ids = Join-Path $Out "ref\$name.ids.txt"
    $rb = Join-Path $Out "ref\$name.f32.bin"
    if (-not (Test-Path $rb)) {
      $null = Run @($m, "--text-file", $kltext, "--dump-ids", $ids)
      $t = ((Get-Content $ids -Raw).Trim() -split ",")
      [IO.File]::WriteAllText($ids, ($t[0..([math]::Min(479, $t.Count - 1))] -join ","))
      WaitFree
      $env:REF_ALL = "1"
      $eap = $ErrorActionPreference
      try { $ErrorActionPreference = "Continue"; & $ref $m "@$ids" $rb 2>&1 | Select-Object -Last 1 | Out-Null }
      finally { $ErrorActionPreference = $eap; Remove-Item Env:REF_ALL -ErrorAction SilentlyContinue }
    }
    if (Test-Path $rb) {
      $o = Run @($m, "--tokens-file", $ids, "--kl-ref", $rb)
      $kl = Num $o "KL\(ref \|\| arcinferzero\) = ([\d.]+)"
      $kmax = Num $o "max ([\d.]+)\)"
      $agree = [regex]::Match($o, "top token agrees (\d+) / (\d+)")
      $top = if ($agree.Success) { [math]::Round(100.0 * [int]$agree.Groups[1].Value / [int]$agree.Groups[2].Value, 1) } else { $null }
      Save @{ time = $stamp; commit = $commit; gpu = $gpu; model = $name; test = "kl"; kl = $kl; kl_max = $kmax; top_agree_pct = $top }
      [string]::Format($inv, "  KL against f32: mean {0}, max {1}, top token agrees {2} %", $kl, $kmax, $top)
    } else { "  KL: no f32 reference (ref_forward failed or does not take this model)" }
  }
}
Remove-Item $q, $kltext -ErrorAction SilentlyContinue

# ---- summary.md from every row of this run
$rows = Get-Content $jsonl -Encoding utf8 | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.time -eq $stamp }
$md = @("# arcinferzero benchmarks", "", "$gpu; commit $commit; $stamp.", "")
$md += "## Prompt processing (tok/s) and time to first token"
$md += ""; $md += "| Model | " + (($Sizes | ForEach-Object { "$_" }) -join " | ") + " |"; $md += "|---|" + (($Sizes | ForEach-Object { "---" }) -join "|") + "|"
foreach ($mn in ($rows | Select-Object -ExpandProperty model -Unique)) {
  $cells = foreach ($n in $Sizes) { $r = $rows | Where-Object { $_.model -eq $mn -and $_.test -eq "size" -and $_.tokens -eq $n } | Select-Object -First 1
    if ($r -and $r.pp) { [string]::Format($inv, "{0:F0} ({1:F1} s)", $r.pp, $r.ttft_s) } else { "–" } }
  $md += "| $mn | " + ($cells -join " | ") + " |"
}
$md += ""; $md += "## Generation (tok/s) at that context depth, plain decoding"
$md += ""; $md += "| Model | " + (($Sizes | ForEach-Object { "$_" }) -join " | ") + " |"; $md += "|---|" + (($Sizes | ForEach-Object { "---" }) -join "|") + "|"
foreach ($mn in ($rows | Select-Object -ExpandProperty model -Unique)) {
  $cells = foreach ($n in $Sizes) { $r = $rows | Where-Object { $_.model -eq $mn -and $_.test -eq "size" -and $_.tokens -eq $n } | Select-Object -First 1
    if ($r -and $r.tg) { [string]::Format($inv, "{0:F1}", $r.tg) } else { "–" } }
  $md += "| $mn | " + ($cells -join " | ") + " |"
}
$md += ""; $md += "## Generation modes (256-token coding answer, tok/s)"
$md += ""; $md += "| Model | plain | MTP | DFlash2 | MTP + DFlash2 |"; $md += "|---|---|---|---|---|"
foreach ($mn in ($rows | Select-Object -ExpandProperty model -Unique)) {
  $cells = foreach ($k in "plain", "mtp", "dflash", "mtp+dflash") { $r = $rows | Where-Object { $_.model -eq $mn -and $_.test -eq "mode" -and $_.mode -eq $k } | Select-Object -First 1
    if ($r -and $r.tg) { [string]::Format($inv, "{0:F1}{1}", $r.tg, $(if ($r.accepted_pct) { " ($($r.accepted_pct) %)" } else { "" })) } else { "–" } }
  $md += "| $mn | " + ($cells -join " | ") + " |"
}
$md += ""; $md += "## Accuracy: KL against an f32 reference (480 tokens of USAGE.md)"
$md += ""; $md += "| Model | KL mean | KL max | top token agrees |"; $md += "|---|---|---|---|"
foreach ($r in ($rows | Where-Object { $_.test -eq "kl" })) {
  $md += [string]::Format($inv, "| {0} | {1:F5} | {2:F4} | {3:F1} % |", $r.model, $r.kl, $r.kl_max, $r.top_agree_pct)
}
[IO.File]::WriteAllLines((Join-Path $Out "summary.md"), $md)
"summary: $(Join-Path $Out 'summary.md')"

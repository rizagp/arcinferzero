# Every benchmark for the testers in one go (run-tests.cmd runs this): the models found on this PC, then
#   1. bench\f33_sweep.ps1   the new XMX switches (Qwen models)
#   2. bench\website_bench.ps1   prompt 128 to 131,072 tokens, generation at each depth, MTP / DFlash2, KL against f32
# and one zip on the Desktop with everything (user names taken out of the paths) to drag into the GitHub issue.
#   run-tests.cmd            everything (long: overnight)
#   run-tests.cmd quick      a shorter round (about an hour)
#   run-tests.cmd quick 27B  only the models whose file name has 27B in it
# (ASCII only: Windows PowerShell 5.1 reads scripts without a BOM as ANSI)
param([string]$Mode = "full", [string]$Only = "")  # (Only: a part of a model name, e.g. 27B: just that model)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$quick = $Mode -match "^(quick|-quick|q)$"
$stamp = Get-Date -Format "yyyy-MM-dd-HHmm"
$out = Join-Path $root "bench\results\run-$stamp"
New-Item -ItemType Directory -Force $out | Out-Null
$log = Join-Path $out "run.log"
function Say([string]$t) { Write-Host $t; Add-Content -Path $log -Value $t -Encoding utf8 }
# (each line shown and kept in the log, as it comes)
function Run-Step([string]$name, [scriptblock]$what) {
  Say ""; Say "==== $name ($(Get-Date -Format 'HH:mm'))"
  $eap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
  try { & $what 2>&1 | ForEach-Object { Say "$_" } }
  catch { Say "FAILED: $($_.Exception.Message)" }
  finally { $ErrorActionPreference = $eap }
}

Say "arcinferzero tests, $(Get-Date -Format 'yyyy-MM-dd HH:mm'), mode: $(if ($quick) { 'quick' } else { 'full' })"
# ---- no sleep while this runs (released when the window closes)
try {
  Add-Type -Name Pw -Namespace ArcTests -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
  [void][ArcTests.Pw]::SetThreadExecutionState([uint32]"0x80000001")
} catch {}

# ---- this PC (for the results; nothing personal)
$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim()
$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$gpus = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "Intel" } | ForEach-Object { "$($_.Name) (driver $($_.DriverVersion))" })
Say "CPU: $cpu; RAM: $ram GB"
foreach ($g in $gpus) { Say "GPU: $g" }
$probe = Join-Path $root "bin\mgpu_probe.exe"
if (Test-Path $probe) { & $probe 2>&1 | Select-String -Pattern "^\s+GPU \d+:" | ForEach-Object { Say ("arcinferzero sees " + $_.Line.Trim()) } }

# ---- the server must be off: a second GPU program spoils the numbers
if (Get-Process serverz, runz -ErrorAction SilentlyContinue) {
  Say ""; Say "The arcinferzero server (or another test) is running. Stop it (launcher: Stop server), then this goes on by itself."
  while (Get-Process serverz, runz -ErrorAction SilentlyContinue) { Start-Sleep 10 }
}

# ---- the models on this PC (the same names website_bench looks for)
$hub = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE } elseif ($env:HF_HOME) { Join-Path $env:HF_HOME "hub" } else { Join-Path $env:USERPROFILE ".cache\huggingface\hub" }
$pkg = Join-Path $env:USERPROFILE ".arcinferzero\zero\models"
$names = "Qwen3.5-4B-UD-Q4_K_XL.gguf", "Qwen3.5-9B-UD-Q4_K_XL.gguf", "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
  "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.gguf", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
  "gemma-4-E4B_q4_0-it.gguf", "gemma-4-12b-it-qat-q4_0.gguf", "gemma-4-26b-a4b-it-qat-q4_0.gguf", "gemma-4-26B_q4_0-it.gguf"
# (and every folder the server reads, model-dirs.txt, e.g. C:\hf: serverz --list-models)
$listed = @()
$sz = Join-Path $root "bin\serverz.exe"
if (Test-Path $sz) {
  $eap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
  try { $listed = @(& $sz --list-models 2>$null | ConvertFrom-Json | ForEach-Object { $_ } | Where-Object { $_.supported -and $_.path }) } catch {}
  $ErrorActionPreference = $eap
}
$models = @()
foreach ($n in $names) {
  $f = @(Get-ChildItem $pkg -Recurse -Filter $n -ErrorAction SilentlyContinue) + @(Get-ChildItem $hub -Recurse -Filter $n -ErrorAction SilentlyContinue) | Select-Object -First 1
  if ($f) { $f = $f.FullName }
  else { $f = $listed | Where-Object { (Split-Path $_.path -Leaf) -eq $n } | Select-Object -First 1 -ExpandProperty path }
  if ($f -and -not ($models | Where-Object { $_ -like "*Flash-Next*" -and $f -like "*Flash-Next*" })) { $models += $f }
}
if ($Only) { $models = @($models | Where-Object { (Split-Path $_ -Leaf) -like "*$Only*" }) }
if (-not $models) { Say "No model found. Download one first (launcher: Get models), then run this again."; Read-Host "Enter to close"; exit 1 }
Say ""; Say "Models found:"
foreach ($m in $models) { Say ("  " + (Split-Path $m -Leaf)) }
$qwen = @($models | Where-Object { (Split-Path $_ -Leaf) -like "Qwen*" })
Say ""
Say $(if ($quick) { "This takes about an hour. Keep the PC plugged in; it will not sleep." } else { "This takes many hours (overnight is best). Keep the PC plugged in; it will not sleep." })
Say "You can leave it alone: at the end a zip file appears on the Desktop."

# ---- 1. the XMX switches (Qwen models only: Gemma has no DeltaNet and its attention is not on XMX)
if ($qwen) {
  Run-Step "1/2  bench\f33_sweep.ps1 (the new XMX switches)" {
    if ($quick) { & (Join-Path $root "bench\f33_sweep.ps1") -Models $qwen -Quick }
    else { & (Join-Path $root "bench\f33_sweep.ps1") -Models $qwen }
  }
}
# ---- 2. the website numbers
$wb = Join-Path $out "website"
Run-Step "2/2  bench\website_bench.ps1 (speeds from short to long prompts, KL)" {
  if ($quick) { & (Join-Path $root "bench\website_bench.ps1") -Models $models -Out $wb -Sizes 128, 2048, 32768 -SkipKl }
  else { & (Join-Path $root "bench\website_bench.ps1") -Models $models -Out $wb }
}

# ---- one zip on the Desktop: the user's folder in paths written as %USERPROFILE%, no f32 reference files (large)
Say ""; Say "==== Packing the results"
$pack = Join-Path $env:TEMP "arcinferzero-results-$stamp"
Remove-Item $pack -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $pack | Out-Null
Get-ChildItem $out -Recurse -File | Where-Object { $_.Extension -in ".log", ".md", ".jsonl", ".txt" -and $_.FullName -notmatch "\\ref\\" } | ForEach-Object {
  $rel = $_.FullName.Substring($out.Length).TrimStart("\")
  $dst = Join-Path $pack $rel
  New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
  $t = [IO.File]::ReadAllText($_.FullName)
  $t = $t.Replace($env:USERPROFILE, "%USERPROFILE%").Replace(($env:USERPROFILE -replace "\\", "/"), "%USERPROFILE%")
  [IO.File]::WriteAllText($dst, $t, (New-Object Text.UTF8Encoding $false))
}
$desk = [Environment]::GetFolderPath("Desktop")
$zip = Join-Path $desk "arcinferzero-results-$stamp.zip"
Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $pack "*") -DestinationPath $zip
Remove-Item $pack -Recurse -Force -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "Done. Your results: $zip" -ForegroundColor Green
Write-Host "Please drag that file into a comment on the GitHub issue. Thank you!" -ForegroundColor Green
try { Start-Process explorer.exe -ArgumentList "/select,`"$zip`"" } catch {}

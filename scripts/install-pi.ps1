# One-time setup of inferzero for Pi on this laptop (setup.cmd runs it too):
#   - builds bin\serverz.exe if it is missing,
#   - adds the "inferzero" provider to ~/.pi/agent/models.json: every model serverz.exe finds (Hugging Face cache and
#     %USERPROFILE%\.inferzero\models) with its context for the current GPU memory (other providers are kept; the
#     previous file is saved as models.json.bak). zero-pi refreshes the list on every start.
#   - installs the `zero-pi` and `zero-server` commands into ~/.pi/agent/bin (already on PATH).
# Run:  powershell -ExecutionPolicy Bypass -File scripts\install-pi.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$piDir = Join-Path $env:USERPROFILE ".pi\agent"
$binDir = Join-Path $piDir "bin"
if (-not (Test-Path $binDir)) { throw "Pi is not installed ($binDir not found): install it from https://pi.dev and run it once." }

# 1. server binary
$exe = Join-Path $root "bin\serverz.exe"
if (-not (Test-Path $exe)) {
  Write-Host "Building bin\serverz.exe..."
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "build.ps1") zero\serverz.cpp
  if ($LASTEXITCODE -ne 0) { throw "build failed" }
}

# 2. Pi model configuration
. (Join-Path $PSScriptRoot "pi-models.ps1")
$ids = @(Update-PiModels $exe "http://127.0.0.1:8090")
$cfgPath = Join-Path $piDir "models.json"
if ($ids.Count) { Write-Host "Pi models: $(($ids | ForEach-Object { "inferzero/$_" }) -join ', ')  ($cfgPath)" }
else { Write-Host "Pi models: none yet (download a model, then run this again or just start zero-pi)" }

# 3. commands on PATH
# (the shims hold this folder's path: run this again after moving or renaming the folder. The old names pi-arc and
#  arc-server, from before 2026-11-10, are removed if an earlier run installed them)
foreach ($old in "pi-arc", "arc-server") { Remove-Item (Join-Path $binDir "$old.cmd") -ErrorAction SilentlyContinue }
foreach ($name in "zero-pi", "zero-server") {
  $target = Join-Path $root "scripts\$name.ps1"
  # (set PSModulePath=: Windows PowerShell with its own modules, not a path PowerShell 7 left behind: GitHub #3)
  "@echo off`r`nset `"PSModulePath=`"`r`npowershell -NoProfile -ExecutionPolicy Bypass -File `"$target`" %*`r`n" | Set-Content (Join-Path $binDir "$name.cmd") -Encoding ascii -NoNewline
}
Write-Host "Commands installed: zero-pi, zero-server  ($binDir)"

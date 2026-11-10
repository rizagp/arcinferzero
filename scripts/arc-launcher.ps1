# The old name (before 2026-11-10): the launcher's script is zero-launcher.ps1 now (started by zero-launcher.cmd).
& (Join-Path $PSScriptRoot "zero-launcher.ps1") @args
exit $LASTEXITCODE

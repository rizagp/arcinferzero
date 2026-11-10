# The old name (before 2026-11-10): the script is zero-pi.ps1 now. Kept so Pi shims installed before still work.
& (Join-Path $PSScriptRoot "zero-pi.ps1") @args
exit $LASTEXITCODE

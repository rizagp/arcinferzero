# The old name (before 2026-11-10): the script is zero-server.ps1 now. Kept so commands people copied still work,
# e.g. .\scripts\arc-server.ps1 start -Model ...
& (Join-Path $PSScriptRoot "zero-server.ps1") @args
exit $LASTEXITCODE

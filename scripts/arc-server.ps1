# The old name (before 2026-11-10): the script is zero-server.ps1 now. Kept so commands people copied still work,
# e.g. .\scripts\arc-server.ps1 start -Model ...
# (its own process, as the .cmd shims run it: the exit code is the script's, not a native command's inside it)
& (Get-Process -Id $PID).Path -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "zero-server.ps1") @args
exit $LASTEXITCODE

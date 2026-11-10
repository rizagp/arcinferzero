# The old name (before 2026-11-10): the script is zero-pi.ps1 now. Kept so Pi shims installed before still work.
# (its own process, as the .cmd shims run it: the exit code is the script's, not a native command's inside it)
& (Get-Process -Id $PID).Path -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "zero-pi.ps1") @args
exit $LASTEXITCODE

# Downloads models of scripts\model-catalog.ps1 by key, one after another, into the Hugging Face cache (no Python).
# The launcher runs it in the background and reads its lines: "START <key>", "DONE <key>", "FAIL <key> <why>".
#   powershell -File scripts\get-models.ps1 4b asr whisper
#   powershell -File scripts\get-models.ps1 -List      the catalog as JSON (with "have": already downloaded)
param([switch]$List, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Keys)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "model-catalog.ps1")
if ($List) {
  $ArcModels | ForEach-Object { $_ | Add-Member have (Test-ArcModel $_) -PassThru -Force } | ConvertTo-Json -Depth 4
  exit 0
}
$failed = 0
foreach ($k in $Keys) {
  $m = $ArcModels | Where-Object { $_.key -eq $k.ToLower() } | Select-Object -First 1
  if (-not $m) { Write-Output "FAIL $k unknown model: $(($ArcModels | ForEach-Object { $_.key }) -join ', ')"; $failed++; continue }
  Write-Output "START $($m.key)"
  try {
    $snap = Install-ArcModel $m
    Write-Output "DONE $($m.key) $snap"
  } catch {
    Write-Output "FAIL $($m.key) $($_.Exception.Message)"
    $failed++
  }
}
exit $failed

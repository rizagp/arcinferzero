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
# files from a Hugging Face link (the launcher's Downloads dialog): "hf:<org>/<name>/<path in the repository>", grouped
# by repository; an image projector in another folder than the repository's model files is copied next to them (the
# server looks for it in the model's folder)
$hf = @($Keys | Where-Object { $_ -like "hf:*" })
$Keys = @($Keys | Where-Object { $_ -notlike "hf:*" })
foreach ($g in ($hf | ForEach-Object { $s = $_.Substring(3) -split "/", 3; [pscustomobject]@{ repo = "$($s[0])/$($s[1])"; path = $s[2] } } | Group-Object repo)) {
  $key = "hf:$($g.Name)"
  Write-Output "START $key"
  try {
    [string[]]$files = @($g.Group | ForEach-Object { $_.path })
    $snap = Get-HfFiles $g.Name $files
    $models = @($files | Where-Object { $_ -notmatch "mmproj" })
    foreach ($p in @($files | Where-Object { $_ -match "mmproj" })) {
      foreach ($dir in @($models | ForEach-Object { Split-Path ($_ -replace "/", "\") } | Sort-Object -Unique)) {
        if ($dir -ne (Split-Path ($p -replace "/", "\"))) {  # (Split-Path of a bare name: "", the snapshot itself)
          # (both inside the snapshot: Get-HfLocalPath refuses a path that leaves it, audit 2026-11-12 #12)
          $to = if ($dir) { Get-HfLocalPath $snap $dir } else { $snap }
          Copy-Item (Get-HfLocalPath $snap $p) (Join-Path $to (Split-Path $p -Leaf)) -Force
        }
      }
    }
    Write-Output "DONE $key $snap"
  } catch {
    Write-Output "FAIL $key $($_.Exception.Message)"
    $failed++
  }
}
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

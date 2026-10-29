# The data folder %USERPROFILE%\.arcinferzero (kernel cache, model packages, caches, logs, settings, transcripts,
# recordings), or $env:ARC_DATA_DIR, as the exes find it (src/app_dir.h). Until 2026-10-29 it was
# %USERPROFILE%\arcinferzero, until 2026-10-25 %LOCALAPPDATA%\arcinferzero (until 2026-10-24 %LOCALAPPDATA%\arcinfer):
# when only an old one exists it is moved once (the same drive: instant) and the paths in last-model.txt /
# launcher.json / ctx-auto.txt / server.json follow it. If the move fails (a file held open by a running older server)
# the old folder is used as it is. Dot-sourced by setup.ps1, the scripts and tests\run_tests.ps1.
function Get-AppDir {
  if ($env:ARC_DATA_DIR) { return $env:ARC_DATA_DIR }
  $now = Join-Path $env:USERPROFILE ".arcinferzero"
  if (Test-Path -LiteralPath $now) { return $now }
  $olds = @(@{ path = (Join-Path $env:USERPROFILE "arcinferzero"); mixed = "" })
  if ($env:LOCALAPPDATA) {
    foreach ($name in "arcinferzero", "arcinfer") { $olds += @{ path = (Join-Path $env:LOCALAPPDATA $name); mixed = "$env:LOCALAPPDATA/$name" } }
  }
  foreach ($o in $olds) {
    $old = $o.path
    if (-not (Test-Path -LiteralPath $old -PathType Container)) { continue }
    try { Move-Item -LiteralPath $old -Destination $now -ErrorAction Stop } catch { return $old }
    foreach ($f in "last-model.txt", "launcher.json", "ctx-auto.txt", "server.json") {
      $p = Join-Path $now $f
      if (-not (Test-Path -LiteralPath $p)) { continue }
      $s = [IO.File]::ReadAllText($p)
      # (the old path where it is not already the new one's end: "...\arcinferzero" inside "...\.arcinferzero")
      $pairs = @(, @($old.Replace('\', '\\'), $now.Replace('\', '\\'))) + @(, @($old, $now))
      if ($o.mixed) { $pairs += @(, @($o.mixed, $now)) }
      $t = $s
      foreach ($pair in $pairs) { $t = [regex]::Replace($t, '(?i)(?<!\.)' + [regex]::Escape($pair[0]), $pair[1].Replace('$', '$$')) }
      if ($t -ne $s) { [IO.File]::WriteAllText($p, $t) }
    }
    return $now
  }
  return $now
}

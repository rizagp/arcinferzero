# The Visual Studio C++ compiler for build.ps1 and setup.ps1 (dot-sourced). Find-VcVars returns the vcvars64.bat
# to call before cl, searched in this order:
#   1. $env:ARC_VCVARS (the path of a vcvars64.bat),
#   2. vswhere (the installer's copy in Program Files (x86) or Program Files, or one on PATH): every installation,
#      Build Tools and previews included,
#   3. the usual folders on every fixed drive, for an installation vswhere does not list (moved to another drive,
#      a copied Build Tools folder): <drive>:\Program Files [(x86)]\Microsoft Visual Studio\<version>\<edition>,
#      <drive>:\Microsoft Visual Studio\<version>\<edition>, <drive>:\<folder>\Microsoft Visual Studio\<version>\<edition>
#      and <drive>:\<folder> itself (e.g. D:\BuildTools); the newest vcvars64.bat first.
# Returns "" when this shell is already set up (a Developer prompt: cl.exe on PATH and VCToolsInstallDir set), so
# cl runs without vcvars, and $null when nothing is found. Find-VsWhere puts vswhere on PATH (vcvars64.bat calls it).
function Find-VsWhere {
  foreach ($pf in ${env:ProgramFiles(x86)}, $env:ProgramFiles) {
    if (-not $pf) { continue }
    $v = Join-Path $pf "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $v) { return $v }
  }
  $c = Get-Command vswhere.exe -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  return $null
}

function Find-VcVars {
  if ($env:ARC_VCVARS -and (Test-Path $env:ARC_VCVARS)) { return $env:ARC_VCVARS }
  $vw = Find-VsWhere
  if ($vw) {
    $env:PATH = "$(Split-Path $vw);$env:PATH"
    foreach ($p in @(& $vw -all -products * -prerelease -property installationPath)) {
      $v = Join-Path $p "VC\Auxiliary\Build\vcvars64.bat"
      if (Test-Path $v) { return $v }
    }
  }
  $tail = "VC\Auxiliary\Build\vcvars64.bat"
  foreach ($d in [IO.DriveInfo]::GetDrives() | Where-Object { $_.DriveType -eq 'Fixed' -and $_.IsReady }) {
    $r = $d.RootDirectory.FullName
    $found = foreach ($pat in "Program Files*\Microsoft Visual Studio\*\*\$tail", "Microsoft Visual Studio\*\*\$tail",
                              "*\Microsoft Visual Studio\*\*\$tail", "*\$tail") {
      Get-Item -Path (Join-Path $r $pat) -ErrorAction SilentlyContinue
    }
    $v = $found | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($v) { return $v.FullName }
  }
  $cl = Get-Command cl.exe -ErrorAction SilentlyContinue  # (the 64-bit compiler only: ...\bin\Host*\x64\cl.exe)
  if ($cl -and $env:VCToolsInstallDir -and $cl.Source -match '\\x64\\cl\.exe$') { return "" }
  return $null
}

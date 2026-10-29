# A ready-to-use package of arcinferzero: dist\arcinferzero-<version>-win64.zip. Unpacked anywhere, setup.cmd (or
# arc-launcher.cmd) runs without Visual Studio or Python: the exes come built (static C++ runtime, the OpenMP
# runtime vcomp140.dll next to them), bin\PREBUILT.txt tells setup to keep them, and models download with
# scripts\hf-get.ps1. What it holds: every file git tracks (sources too, so setup -Rebuild still builds; nothing
# untracked: no recordings, reports or models), plus bin\ with serverz, runz, transcribez, arc_recorder, clinfo.
#   powershell -File scripts\make-package.ps1            build the five exes, then the zip
#   powershell -File scripts\make-package.ps1 -NoBuild   the exes as they are in bin\
param([switch]$NoBuild)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot
Set-Location $root
$exes = @("zero\serverz", "zero\runz", "zero\transcribez", "tools\arc_recorder", "tools\clinfo")
if (-not $NoBuild) {
  & (Join-Path $root "build.ps1") @($exes | ForEach-Object { "$_.cpp" })
  if (-not $?) { throw "build failed" }
}
$commit = (git rev-parse --short HEAD).Trim()
$dirty = (git status --porcelain --untracked-files=no) -ne $null
$version = (Get-Date -Format "yyyy.MM.dd") + "-" + $commit + $(if ($dirty) { "-dirty" } else { "" })
if ($dirty) { Write-Warning "uncommitted changes go into the package (its version says -dirty): commit first for a release" }
$name = "arcinferzero-$version-win64"
$dist = Join-Path $root "dist"
$stage = Join-Path $dist $name
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force $stage | Out-Null
# the tracked files, as they are in the working tree
$files = @(git ls-files)
foreach ($f in $files) {
  $src = Join-Path $root $f
  if (-not (Test-Path -LiteralPath $src -PathType Leaf)) { continue }  # (deleted, not yet committed)
  $dst = Join-Path $stage $f
  New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
  Copy-Item -LiteralPath $src $dst
}
# the built exes and the OpenMP runtime, and the marker setup reads
$bin = Join-Path $stage "bin"
New-Item -ItemType Directory -Force $bin | Out-Null
foreach ($e in $exes) {
  $exe = Join-Path $root ("bin\" + (Split-Path $e -Leaf) + ".exe")
  if (-not (Test-Path $exe)) { throw "$exe is missing: build it (or run without -NoBuild)" }
  Copy-Item $exe $bin
}
$omp = Join-Path $root "bin\vcomp140.dll"
if (-not (Test-Path $omp)) { throw "bin\vcomp140.dll is missing: build.ps1 copies it from Visual Studio's Redist folder" }
Copy-Item $omp $bin
@("arcinferzero $version (built $(Get-Date -Format 'yyyy-MM-dd HH:mm'), commit $commit)",
  "The exes in this folder are built: setup keeps them (setup -Rebuild builds them from the sources instead).") |
  Set-Content (Join-Path $bin "PREBUILT.txt") -Encoding utf8
# the zip
$zip = Join-Path $dist "$name.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
Write-Host ("{0} ({1:N1} MB): unpack it anywhere, then run setup.cmd (or arc-launcher.cmd once a model is there)" -f $zip, ((Get-Item $zip).Length / 1MB))

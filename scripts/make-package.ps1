# A ready-to-use package of inferzero: dist\inferzero-<version>-win64.zip. Unpacked anywhere, setup.cmd (or
# zero-launcher.cmd) runs without Visual Studio or Python: the exes come built (static C++ runtime, the OpenMP
# runtime vcomp140.dll next to them), bin\PREBUILT.txt tells setup to keep them, and models download with
# scripts\hf-get.ps1. What it holds: every file git tracks (sources too, so setup -Rebuild still builds; nothing
# untracked: no recordings, reports or models), plus bin\ with serverz, runz, transcribez, zero_recorder, clinfo,
# mgpu_probe, event_probe, replay_probe, membw_probe, gpu_bench, xw_peak, ref_forward, gguf_dump, import_autoround_pack
# (setup -Models flash-ar4).
#   powershell -File scripts\make-package.ps1            build the exes, then the zip
#   powershell -File scripts\make-package.ps1 -NoBuild   the exes as they are in bin\
#   powershell -File scripts\make-package.ps1 -Release 2026.11.13.01   the release's number in the package's name
#     (yyyy.MM.dd.NN, NN two digits: names and tags sort by release, 09 before 13; issue #4, 2026-11-12)
param([switch]$NoBuild, [string]$Release = "")
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot
Set-Location $root
$exes = @("zero\serverz", "zero\runz", "zero\transcribez", "tools\zero_recorder", "tools\clinfo", "zero\mgpu_probe", "zero\event_probe", "bench\replay_probe", "bench\membw_probe", "bench\gpu_bench",
          "bench\xw_peak", "tools\ref_forward", "tools\gguf_dump", "tools\import_autoround_pack")  # (ref_forward: the f32 CPU reference of bench\website_bench.ps1's KL, 2026-11-09; gguf_dump: a model's keys and tensors, for reports; xw_peak: the 140T's XMX GEMM in TOPS, 2026-11-12)
if (-not $NoBuild) {
  $srcs = @($exes | ForEach-Object { "$_.cpp" })  # (splatted from a variable: @(...) inline is one argument)
  & (Join-Path $root "build.ps1") @srcs
  if (-not $?) { throw "build failed" }
}
$commit = (git rev-parse --short HEAD).Trim()
$dirty = (git status --porcelain --untracked-files=no) -ne $null
if ($Release -and $Release -notmatch '^\d{4}\.\d{2}\.\d{2}\.\d{2}$') { throw "-Release: yyyy.MM.dd.NN with two digits for NN, e.g. 2026.11.13.01 (got $Release)" }
$version = $(if ($Release) { $Release } else { Get-Date -Format "yyyy.MM.dd" }) + "-" + $commit + $(if ($dirty) { "-dirty" } else { "" })
if ($dirty) { Write-Warning "uncommitted changes go into the package (its version says -dirty): commit first for a release" }
$name = "inferzero-$version-win64"
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
@("inferzero $version (built $(Get-Date -Format 'yyyy-MM-dd HH:mm'), commit $commit)",
  "The exes in this folder are built: setup keeps them (setup -Rebuild builds them from the sources instead).") |
  Set-Content (Join-Path $bin "PREBUILT.txt") -Encoding utf8
# the zip
$zip = Join-Path $dist "$name.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
# older packages go: dist\ kept every one, a full copy of the sources each (43 folders, audit 2026-11-12 #11). The
# newest ARC_DIST_KEEP (default 3) folders and zips stay; the released zips are on GitHub. Only names of this
# script's own form are touched.
$keep = if ($env:ARC_DIST_KEEP -match '^\d+$') { [Math]::Max(1, [int]$env:ARC_DIST_KEEP) } else { 3 }
$pkgRe = '^(arc)?inferzero-\d{4}\.\d{2}\.\d{2}(\.\d{2})?-[0-9a-f]{7}(-dirty)?-win64$'
foreach ($kind in @("dir", "zip")) {
  $items = @(Get-ChildItem $dist | Where-Object {
      if ($kind -eq "dir") { $_.PSIsContainer -and $_.Name -match $pkgRe } else { -not $_.PSIsContainer -and $_.BaseName -match $pkgRe -and $_.Extension -eq ".zip" } } |
    Sort-Object LastWriteTime -Descending)
  foreach ($old in ($items | Select-Object -Skip $keep)) {
    if ($old.FullName -eq $stage -or $old.FullName -eq $zip) { continue }
    Remove-Item $old.FullName -Recurse -Force -ErrorAction SilentlyContinue
  }
}
Write-Host ("{0} ({1:N1} MB): unpack it anywhere, then run setup.cmd (or zero-launcher.cmd once a model is there)" -f $zip, ((Get-Item $zip).Length / 1MB))

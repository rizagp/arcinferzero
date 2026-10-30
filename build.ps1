# Builds one or more arcinferzero targets with MSVC (Visual Studio Build Tools with the C++ workload).
# Usage: .\build.ps1 tools\clinfo.cpp [bench\gemv_bench.cpp ...]
# The compiler is found by scripts\vcvars.ps1 (vswhere, then the usual folders on every drive, then a Developer
# prompt's own cl); $env:ARC_VCVARS = the path of a vcvars64.bat overrides it.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Sources)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
. (Join-Path $PSScriptRoot "scripts\vcvars.ps1")
$vcvars = Find-VcVars
if ($null -eq $vcvars) {
  throw "No Visual Studio C++ compiler found: install Visual Studio Build Tools with the 'Desktop development with C++' workload (README, step 3), or set `$env:ARC_VCVARS to a vcvars64.bat."
}
$out = Join-Path $root "bin"
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($src in $Sources) {
  $full = Join-Path $root $src
  $name = [IO.Path]::GetFileNameWithoutExtension($full)
  # (_CRT_SECURE_NO_WARNINGS: no "getenv / fopen may be unsafe" deprecation notes)
  # (the f32 references and checks, ref_forward / tests / vision_testz: /fp:precise; the rest /fp:fast;
  #  /utf-8: the sources are UTF-8, their string literals stay byte for byte;
  #  /W4 /permissive-: our own sources build warning-free, third_party is an external include without warnings)
  $fp = if ($name -match '^(ref_forward|ref_whisper|test_|vision_test)') { '/fp:precise' } else { '/fp:fast' }
  # (/bigobj: serverz has more sections than a default object file holds;
  #  /MT: the C++ runtime inside each exe, so a PC without the Visual C++ Redistributable runs them; /openmp: the
  #  OpenMP runtime vcomp140.dll, which may be shipped next to them (copied into bin below). The LLVM OpenMP of
  #  /openmp:llvm has no redistributable DLL. Logits bit-identical, speeds the same (2026-10-29))
  $pre = if ($vcvars) { "`"$vcvars`" >nul && " } else { "" }  # ("": a Developer prompt, cl already set up)
  # (/d1trimfile: __FILE__ in error messages relative to the source folder, so the exes hold no user's path; the
  #  doubled backslash keeps the closing quote a quote)
  $cmd = "$pre" + "cl /nologo /utf-8 /std:c++20 /O2 /Oi /arch:AVX2 $fp /EHsc /bigobj /MT /W4 /permissive- /Zc:__cplusplus /d1trimfile:`"$root\\`" /external:I`"$root\third_party`" /external:W0 /openmp /D_CRT_SECURE_NO_WARNINGS /I`"$root\third_party`" " +
         "/Fe`"$out\$name.exe`" /Fo`"$out\\`" `"$full`" " +
         "/link /MANIFEST:EMBED /MANIFESTINPUT:`"$root\tools\utf8.manifest`""  # (UTF-8 code page: paths with any characters)
  cmd /c $cmd
  if ($LASTEXITCODE -ne 0) { throw "build failed: $src" }
}
# the OpenMP runtime the exes load (vcomp140.dll, redistributable: Visual Studio's Redist folder), next to them
if (-not (Test-Path (Join-Path $out "vcomp140.dll"))) {
  $pre = if ($vcvars) { "`"$vcvars`" >nul && " } else { "" }
  # (read with set after vcvars ran: an %...% in the same command line is expanded before it runs)
  $redist = ((cmd /c "$pre set VCToolsRedistDir") | Where-Object { $_ -like "VCToolsRedistDir=*" } | Select-Object -First 1) -replace '^VCToolsRedistDir=', ''
  $dll = if ($redist) { Get-ChildItem -Path (Join-Path $redist "x64") -Recurse -Filter vcomp140.dll -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch 'debug|onecore' } | Select-Object -First 1 }
  if ($dll) { Copy-Item $dll.FullName $out } else { Write-Warning "vcomp140.dll not found under ${redist}: the exes need it (Visual C++ Redistributable)" }
}

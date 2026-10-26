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
  $fp = if ($name -match '^(ref_forward|test_|vision_test)') { '/fp:precise' } else { '/fp:fast' }
  # (/bigobj: serverz has more sections than a default object file holds)
  $pre = if ($vcvars) { "`"$vcvars`" >nul && " } else { "" }  # ("": a Developer prompt, cl already set up)
  $cmd = "$pre" + "cl /nologo /utf-8 /std:c++20 /O2 /Oi /arch:AVX2 $fp /EHsc /bigobj /MD /W4 /permissive- /Zc:__cplusplus /external:I`"$root\third_party`" /external:W0 /openmp:llvm /D_CRT_SECURE_NO_WARNINGS /I`"$root\third_party`" " +
         "/Fe`"$out\$name.exe`" /Fo`"$out\\`" `"$full`" " +
         "/link /MANIFEST:EMBED /MANIFESTINPUT:`"$root\tools\utf8.manifest`""  # (UTF-8 code page: paths with any characters)
  cmd /c $cmd
  if ($LASTEXITCODE -ne 0) { throw "build failed: $src" }
}

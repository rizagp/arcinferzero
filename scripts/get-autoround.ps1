# Qwen3.8 Flash Next with Intel's AutoRound INT4 experts (pull request #22), made ready for inferzero in one step:
#   powershell -File scripts\get-autoround.ps1 [-DeleteSource] [-Yes]     (or: setup -Models flash-ar4)
# Intel's checkpoint is safetensors, not GGUF, so it is imported:
#   1. the AutoRound checkpoint's expert shards (~78 GB, at the revision the importer checks; shards without experts
#      are not downloaded) and the model-info.json its checks read (each file's size and SHA-256, from the API);
#   2. tools\import_autoround_experts.py (Python 3.11+, NumPy): their INT4 experts into the AR4_128 expert file
#      (~64 GB), every code and scale checked;
#   3. Unsloth's UD-Q4_K_XL Flash Next (~111 GB, 4 parts) for everything but the experts, and its image projector;
#   4. bin\import_autoround_pack.exe: the package GGUF (~32 GB) next to the expert file, in
#      %USERPROFILE%\.inferzero\zero\models\Qwen3.8-Flash-Next-AutoRound-INT4\, where the server lists it
#      (shortcut flash-ar4; precise mode turns on by itself for it).
# Afterwards the AutoRound shards are no longer needed: deleted when you agree at the prompt (-DeleteSource: without
# asking; -Yes alone keeps them). The UD-Q4_K_XL stays (its own model). Disk: ~78 GB of shards + 64 GB expert file +
# 111 GB UD-Q4_K_XL + 32 GB package = ~285 GB at the most, ~207 GB once the shards are deleted.
param([switch]$DeleteSource, [switch]$Yes)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot
. (Join-Path $PSScriptRoot "hf-get.ps1")
. (Join-Path $PSScriptRoot "app-dir.ps1")

$ArRepo = "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound"
$ArRevision = "4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897"  # (tools\import_autoround_experts.py REVISION)
$Q4Repo = "unsloth/Qwen3.8-Flash-Next-GGUF"
$Q4Folder = "UD-Q4_K_XL"
$ArName = "Qwen3.8-Flash-Next-AutoRound-INT4"

function Get-AutoRoundDir { Join-Path (Get-AppDir) "zero\models\$ArName" }
function Test-AutoRound {
  $d = Get-AutoRoundDir
  return (Test-Path (Join-Path $d "$ArName.gguf")) -and (Test-Path (Join-Path $d "$ArName.experts"))
}

function Install-AutoRound([switch]$DeleteSource, [switch]$Yes) {
  $dir = Get-AutoRoundDir
  $gguf = Join-Path $dir "$ArName.gguf"
  $experts = Join-Path $dir "$ArName.experts"
  if (Test-AutoRound) { Write-Host "    $ArName is there: $dir"; return $dir }

  # Python 3.11+ (hashlib.file_digest) with NumPy, and the packer
  $py = $null
  foreach ($c in "python", "py -3") {
    [string[]]$parts = $c -split " "
    $rest = @($parts | Select-Object -Skip 1)
    try {
      $v = "$(& $parts[0] @rest -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null)".Trim()
      if ($LASTEXITCODE -eq 0 -and $v -match '^3\.(\d+)$' -and [int]$Matches[1] -ge 11) { $py = $parts; break }
    } catch {}
  }
  if (-not $py) { throw "the AutoRound import needs Python 3.11 or newer (https://www.python.org/downloads/, tick 'Add to PATH')" }
  $pyRest = @($py | Select-Object -Skip 1)
  $ErrorActionPreference = "Continue"
  & $py[0] @pyRest -c "import numpy" 2>$null | Out-Null
  $hasNumpy = $LASTEXITCODE -eq 0
  $ErrorActionPreference = "Stop"
  if (-not $hasNumpy) {
    Write-Host "    installing NumPy (pip) for the import"
    & $py[0] @pyRest -m pip install -U numpy
    if ($LASTEXITCODE -ne 0) { throw "pip install numpy failed" }
  }
  $pack = Join-Path $root "bin\import_autoround_pack.exe"
  if (-not (Test-Path $pack)) {
    Write-Host "    building bin\import_autoround_pack.exe"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "build.ps1") "tools\import_autoround_pack.cpp"
    if (-not (Test-Path $pack)) { throw "bin\import_autoround_pack.exe could not be built (it needs Visual Studio's C++ tools, or a package from the releases page)" }
  }
  New-Item -ItemType Directory -Force $dir | Out-Null

  # 1 + 2: the expert file (skipped when a run before made it)
  $arSnap = $null
  if (-not (Test-Path $experts)) {
    Write-Host "    1/4 the AutoRound checkpoint's experts ($ArRepo at $($ArRevision.Substring(0, 7)))"
    $arSnap = Get-HfFiles $ArRepo @("config.json", "model.safetensors.index.json") -Revision $ArRevision
    # (the shards that hold experts, from the index: shard 16, ~102 GB of other weights, is not needed)
    #  (a regex, not ConvertFrom-Json: Windows PowerShell's stops at ~2 MB of JSON, the index is 20 MB)
    $index = [IO.File]::ReadAllText((Join-Path $arSnap "model.safetensors.index.json"))
    $shards = @([regex]::Matches($index, '"[^"]*\.mlp\.experts\.[^"]*"\s*:\s*"([^"/\\]+\.safetensors)"') |
        ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
    if (-not $shards) { throw "$ArRepo`: no expert shards found in its index" }
    $null = Get-HfFiles $ArRepo $shards -Revision $ArRevision
    # (model-info.json: each file's size and SHA-256 as the API gives them, which the importer checks every shard against)
    $token = Get-HfToken
    $hdr = @{}
    if ($token) { $hdr.Authorization = "Bearer $token" }
    $info = Invoke-RestMethod "https://huggingface.co/api/models/$ArRepo/revision/$ArRevision`?blobs=true" -Headers $hdr -TimeoutSec 60
    [IO.File]::WriteAllText((Join-Path $arSnap "model-info.json"), ($info | ConvertTo-Json -Depth 20))
    Write-Host "    2/4 importing the INT4 experts into $experts (reads ~78 GB, checks every code and scale)"
    & $py[0] @pyRest (Join-Path $root "tools\import_autoround_experts.py") $arSnap $experts --experimental-format
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $experts)) { throw "the expert import failed (run this again: the downloads stay)" }
  } else { Write-Host "    1-2/4 the expert file is there: $experts" }

  # 3 + 4: the rest of the model from the UD-Q4_K_XL, and the package GGUF
  Write-Host "    3/4 the rest of the model: $Q4Repo $Q4Folder"
  $q4Snap = Get-HfFiles $Q4Repo @("$Q4Folder/", "mmproj-F16.gguf")
  $first = @(Get-ChildItem (Join-Path $q4Snap $Q4Folder) -Filter "*-00001-of-*.gguf")[0]
  if (-not $first) { throw "$Q4Repo ${Q4Folder}: its first part was not found in $q4Snap" }
  Write-Host "    4/4 the package: $gguf"
  if (Test-Path $gguf) { Remove-Item $gguf }  # (only a package this step wrote: the folder is its own)
  & $pack $first.FullName $gguf "$ArName.experts" $ArRevision
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $gguf)) { throw "writing the package failed" }
  Copy-Item (Join-Path $q4Snap "mmproj-F16.gguf") (Join-Path $dir "mmproj-F16.gguf") -Force  # (images)

  # the AutoRound shards: not needed any more
  if ($DeleteSource -or -not $Yes) {
    if (-not $arSnap) { $arSnap = Join-Path (Get-HfHub) ("models--" + ($ArRepo -replace "/", "--") + "\snapshots\$ArRevision") }
    $st = @(Get-ChildItem $arSnap -Filter "*.safetensors" -ErrorAction SilentlyContinue)
    if ($st) {
      $gb = ($st | ForEach-Object { (Get-Item $_.FullName).Length } | Measure-Object -Sum).Sum / 1e9
      $del = [bool]$DeleteSource
      if (-not $del) {
        $a = Read-Host ("    Delete the AutoRound checkpoint's {0} shards ({1:N1} GB)? inferzero reads only the expert file now [Y/n]" -f $st.Count, $gb)
        $del = $a -eq "" -or $a -match '^[yY]'
      }
      if ($del) {
        # (the cache's blobs too: a snapshot file can be a link to one)
        foreach ($f in $st) {
          $it = Get-Item $f.FullName
          if ($it.LinkType -and $it.Target) { Remove-Item -LiteralPath ([string]@($it.Target)[0]) -ErrorAction SilentlyContinue }
          Remove-Item -LiteralPath $f.FullName
        }
        Write-Host ("    deleted {0:N1} GB of AutoRound shards" -f $gb)
      }
    }
  }
  Write-Host "    ready: zero-server start -Model flash-ar4 (or pick it in the launcher)"
  return $dir
}

if ($MyInvocation.InvocationName -ne ".") {
  $null = Install-AutoRound -DeleteSource:$DeleteSource -Yes:$Yes
}

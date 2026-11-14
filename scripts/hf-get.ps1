# Files of a Hugging Face model repository, without Python: into the Hugging Face cache layout the server reads
# (<hub>\models--<org>--<name>\snapshots\<commit>\<file>, refs\main = the commit), downloaded with curl.exe (part of
# Windows 10 / 11), resumed where a cut download stopped (<file>.part), each file's size and, for LFS files (the
# models), its SHA256 checked against the repository's before it takes its name. A token, for repositories that ask
# you to accept their terms first (Gemma): $env:HF_TOKEN, or the file %USERPROFILE%\.cache\huggingface\token that
# `hf auth login` writes. Dot-sourced by setup.ps1; also on its own:
#   powershell -File scripts\hf-get.ps1 unsloth/Qwen3.5-4B-MTP-GGUF Qwen3.5-4B-UD-Q4_K_XL.gguf mmproj-F16.gguf
# A folder of the repository downloads every file in it, as a split model's parts are kept (no merging needed):
#   powershell -File scripts\hf-get.ps1 unsloth/DeepSeek-V4.1-Flash-1017-GGUF UD-Q8_K_XL
# Before anything is downloaded the free space of the drive is checked; at the end the command that runs the model.
function Get-HfHub {
  if ($env:HF_HUB_CACHE) { return $env:HF_HUB_CACHE }
  if ($env:HF_HOME) { return (Join-Path $env:HF_HOME "hub") }
  return (Join-Path $env:USERPROFILE ".cache\huggingface\hub")
}
# a file's SHA256 by .NET, not Get-FileHash: a Windows PowerShell started from PowerShell 7 can inherit a module path
# under which that cmdlet is missing ("The term 'Get-FileHash' is not recognized", GitHub issue #3)
function Get-Sha256([string]$path) {
  $sha = [System.Security.Cryptography.SHA256]::Create()
  $fs = [System.IO.File]::Open($path, "Open", "Read", "Read")
  try { return (($sha.ComputeHash($fs) | ForEach-Object { $_.ToString("x2") }) -join "") }
  finally { $fs.Dispose(); $sha.Dispose() }
}
function Get-HfToken {
  if ($env:HF_TOKEN) { return $env:HF_TOKEN }
  $f = Join-Path $env:USERPROFILE ".cache\huggingface\token"
  if (Test-Path $f) { return (Get-Content $f -Raw).Trim() }
  return ""
}
# A repository file's path inside the snapshot folder, or an error: a name from a pasted link or a repository's own
# listing may hold "..", a drive letter or a UNC prefix, and Join-Path would then write outside the cache (audit
# 2026-11-12 #12; the C++ side refuses the same with files::safe_name / proj::)
function Get-HfLocalPath([string]$snap, [string]$f) {
  $rel = $f -replace "/", "\"
  if ($rel -match '^\\' -or $rel -match '^[A-Za-z]:' -or $rel -match '(^|\\)\.\.(\\|$)' -or $rel -match '[<>:"|?*]') {
    throw "$f is not a path inside the repository"
  }
  $root = [IO.Path]::GetFullPath($snap).TrimEnd('\') + '\'
  $full = [IO.Path]::GetFullPath((Join-Path $snap $rel))
  if (-not $full.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { throw "$f leaves the cache folder" }
  return $full
}
# downloads $Files of $Repo (paths as the repository lists them, e.g. IQ3_S/x.gguf) and returns the snapshot folder
# (-Revision: a commit to take instead of the newest, e.g. the one tools\import_autoround_experts.py checks)
function Get-HfFiles([string]$Repo, [string[]]$Files, [string]$Hub = (Get-HfHub), [string]$Revision = "main") {
  $token = Get-HfToken
  $hdr = @{}
  if ($token) { $hdr.Authorization = "Bearer $token" }
  $api = "https://huggingface.co/api/models/$Repo"
  try {
    $sha = (Invoke-RestMethod "$api/revision/$Revision" -Headers $hdr -TimeoutSec 60).sha
    # (Windows PowerShell hands a JSON array over as one item: unrolled into its entries)
    $tree = @(Invoke-RestMethod "$api/tree/$sha`?recursive=1" -Headers $hdr -TimeoutSec 60 | ForEach-Object { $_ })
  } catch {
    $code = try { [int]$_.Exception.Response.StatusCode } catch { 0 }
    if ($code -in 401, 403) {
      throw "$Repo needs you to sign in and accept its terms on huggingface.co first, then a token: set HF_TOKEN (a read token from https://huggingface.co/settings/tokens) and run this again"
    }
    throw "cannot reach huggingface.co for $Repo ($($_.Exception.Message)): check the internet connection, then run this again"
  }
  $base = Join-Path $Hub ("models--" + ($Repo -replace "/", "--"))
  $snap = Join-Path $base "snapshots\$sha"
  New-Item -ItemType Directory -Force $snap, (Join-Path $base "refs") | Out-Null
  if ($Revision -eq "main") { [IO.File]::WriteAllText((Join-Path $base "refs\main"), $sha) }
  $curl = Join-Path $env:SystemRoot "System32\curl.exe"
  if (-not (Test-Path $curl)) { $curl = "curl.exe" }
  # (a folder, "UD-Q8_K_XL" or "UD-Q8_K_XL/": every file in it, in name order, so a split model's parts come whole)
  $Files = @(foreach ($f in $Files) {
      $dir = $f.TrimEnd("/")
      $inside = @($tree | Where-Object { $_.type -eq "file" -and $_.path.StartsWith("$dir/") } | ForEach-Object { $_.path } | Sort-Object)
      if (-not ($tree | Where-Object { $_.type -eq "file" -and $_.path -eq $f }) -and $inside.Count) { $inside } else { $f }
    })
  # (the room the downloads need, against the drive's free space: a 558 GB model that stops at 90 % helps nobody)
  $need = [int64]0
  foreach ($f in $Files) {
    $e = $tree | Where-Object { $_.path -eq $f } | Select-Object -First 1
    if (-not $e) { continue }
    $dst = Get-HfLocalPath $snap $f
    $have = if (Test-Path $dst) { (Get-Item $dst).Length } elseif (Test-Path "$dst.part") { (Get-Item "$dst.part").Length } else { 0 }
    $need += [math]::Max([int64]0, [int64]$e.size - $have)
  }
  # (best effort: when the drive's free space cannot be found, a relative path for one, the check is skipped; before,
  #  Split-Path -Qualifier's error on a relative or UNC path stopped every download. A UNC share, \\server\share: its
  #  free space from Windows' FileSystemObject, as [IO.DriveInfo] refuses such a path in PowerShell 5.1 and 7 alike)
  $free = $null
  $where = ""
  try {
    if ($snap -match '^\\\\[^\\?.]') {
      $where = [IO.Path]::GetPathRoot($snap).TrimEnd('\')
      $free = (New-Object -ComObject Scripting.FileSystemObject).GetDrive($where).AvailableSpace
    } else {
      $drive = Get-PSDrive -Name ((Split-Path $snap -Qualifier -ErrorAction Stop).TrimEnd(":")) -ErrorAction Stop
      $free = $drive.Free
      $where = "$($drive.Name):"
    }
  } catch { $free = $null }
  if ($null -ne $free) {
    if ($need -gt $free) {
      throw ("these files need {0:N1} GB more, and {1} has {2:N1} GB free: make room, or set HF_HOME to a folder on a larger drive" -f ($need / 1e9), $where, ($free / 1e9))
    }
    if ($need -gt 0) { Write-Host ("    {0:N1} GB to download ({1} {2:N1} GB free)" -f ($need / 1e9), $where, ($free / 1e9)) }
  }
  foreach ($f in $Files) {
    $e = $tree | Where-Object { $_.path -eq $f } | Select-Object -First 1
    if (-not $e) { throw "$Repo has no file $f" }
    $dst = Get-HfLocalPath $snap $f
    New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
    $size = [int64]$e.size
    if ((Test-Path $dst) -and (Get-Item $dst).Length -eq $size) { Write-Host "    $f is there"; continue }
    $part = "$dst.part"
    $have = if (Test-Path $part) { (Get-Item $part).Length } else { 0 }
    if ($have -gt $size) { Remove-Item $part; $have = 0 }
    if ($have -lt $size) {
      Write-Host ("    {0} ({1:N2} GB{2})" -f $f, ($size / 1e9), $(if ($have) { ", resuming at {0:N2} GB" -f ($have / 1e9) } else { "" }))
      # (--speed-time: a connection that stalls, under 1 byte/s for 2 minutes, ends with an error instead of waiting
      #  forever, so the launcher's Downloads dialog tries again)
      $a = @("-L", "--fail", "--retry", "5", "--retry-delay", "3", "--speed-limit", "1", "--speed-time", "120", "--progress-bar", "-o", $part)
      if ($have) { $a += @("-C", "-") }
      # (a speed limit: $env:ARC_HF_RATE in curl's form, e.g. 5M = 5 MiB/s; the launcher's Downloads dialog sets it)
      if ($env:ARC_HF_RATE -match '^\d+[KkMmGg]?$') { $a += @("--limit-rate", $env:ARC_HF_RATE) }
      if ($token) { $a += @("-H", "Authorization: Bearer $token") }
      & $curl @a "https://huggingface.co/$Repo/resolve/$sha/$f"
      if ($LASTEXITCODE -ne 0) { throw "downloading $f failed (curl exit $LASTEXITCODE): run this again, it goes on where it stopped" }
    }
    if ((Get-Item $part).Length -ne $size) { throw "$f is $((Get-Item $part).Length) bytes, the repository says ${size}: run this again" }
    if ($e.lfs -and $e.lfs.oid) {
      Write-Host "    checking $f (SHA256)..."
      $h = Get-Sha256 $part
      if ($h -ne $e.lfs.oid.ToLower()) { Remove-Item $part; throw "$f arrived damaged (SHA256 differs): run this again to download it anew" }
    }
    Move-Item $part $dst -Force
  }
  return $snap
}
if ($MyInvocation.InvocationName -ne "." -and $args.Count -ge 2) {
  $ErrorActionPreference = "Stop"
  $snap = Get-HfFiles $args[0] @($args | Select-Object -Skip 1)
  $snap
  # (the model to give runz / serverz: a split model's first part, else the first GGUF that is not an image encoder)
  $g = @(Get-ChildItem $snap -Recurse -Filter *.gguf | Where-Object { $_.Name -notmatch "^mmproj" } | Sort-Object FullName)
  $first = @($g | Where-Object { $_.Name -match "-00001-of-\d{5}\.gguf$" }) + $g | Select-Object -First 1
  if ($first) { Write-Host "    run it with: bin\runz.exe `"$($first.FullName)`" `"The capital of France is`" --n 32" }
}

# Files of a Hugging Face model repository, without Python: into the Hugging Face cache layout the server reads
# (<hub>\models--<org>--<name>\snapshots\<commit>\<file>, refs\main = the commit), downloaded with curl.exe (part of
# Windows 10 / 11), resumed where a cut download stopped (<file>.part), each file's size and, for LFS files (the
# models), its SHA256 checked against the repository's before it takes its name. A token, for repositories that ask
# you to accept their terms first (Gemma): $env:HF_TOKEN, or the file %USERPROFILE%\.cache\huggingface\token that
# `hf auth login` writes. Dot-sourced by setup.ps1; also on its own:
#   powershell -File scripts\hf-get.ps1 unsloth/Qwen3.5-4B-MTP-GGUF Qwen3.5-4B-UD-Q4_K_XL.gguf mmproj-F16.gguf
function Get-HfHub {
  if ($env:HF_HUB_CACHE) { return $env:HF_HUB_CACHE }
  if ($env:HF_HOME) { return (Join-Path $env:HF_HOME "hub") }
  return (Join-Path $env:USERPROFILE ".cache\huggingface\hub")
}
function Get-HfToken {
  if ($env:HF_TOKEN) { return $env:HF_TOKEN }
  $f = Join-Path $env:USERPROFILE ".cache\huggingface\token"
  if (Test-Path $f) { return (Get-Content $f -Raw).Trim() }
  return ""
}
# downloads $Files of $Repo (paths as the repository lists them, e.g. IQ3_S/x.gguf) and returns the snapshot folder
function Get-HfFiles([string]$Repo, [string[]]$Files, [string]$Hub = (Get-HfHub)) {
  $token = Get-HfToken
  $hdr = @{}
  if ($token) { $hdr.Authorization = "Bearer $token" }
  $api = "https://huggingface.co/api/models/$Repo"
  try {
    $sha = (Invoke-RestMethod "$api/revision/main" -Headers $hdr -TimeoutSec 60).sha
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
  [IO.File]::WriteAllText((Join-Path $base "refs\main"), $sha)
  $curl = Join-Path $env:SystemRoot "System32\curl.exe"
  if (-not (Test-Path $curl)) { $curl = "curl.exe" }
  foreach ($f in $Files) {
    $e = $tree | Where-Object { $_.path -eq $f } | Select-Object -First 1
    if (-not $e) { throw "$Repo has no file $f" }
    $dst = Join-Path $snap ($f -replace "/", "\")
    New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
    $size = [int64]$e.size
    if ((Test-Path $dst) -and (Get-Item $dst).Length -eq $size) { Write-Host "    $f is there"; continue }
    $part = "$dst.part"
    $have = if (Test-Path $part) { (Get-Item $part).Length } else { 0 }
    if ($have -gt $size) { Remove-Item $part; $have = 0 }
    if ($have -lt $size) {
      Write-Host ("    {0} ({1:N2} GB{2})" -f $f, ($size / 1e9), $(if ($have) { ", resuming at {0:N2} GB" -f ($have / 1e9) } else { "" }))
      $a = @("-L", "--fail", "--retry", "5", "--retry-delay", "3", "--progress-bar", "-o", $part)
      if ($have) { $a += @("-C", "-") }
      if ($token) { $a += @("-H", "Authorization: Bearer $token") }
      & $curl @a "https://huggingface.co/$Repo/resolve/$sha/$f"
      if ($LASTEXITCODE -ne 0) { throw "downloading $f failed (curl exit $LASTEXITCODE): run this again, it goes on where it stopped" }
    }
    if ((Get-Item $part).Length -ne $size) { throw "$f is $((Get-Item $part).Length) bytes, the repository says ${size}: run this again" }
    if ($e.lfs -and $e.lfs.oid) {
      Write-Host "    checking $f (SHA256)..."
      $h = (Get-FileHash $part -Algorithm SHA256).Hash.ToLower()
      if ($h -ne $e.lfs.oid.ToLower()) { Remove-Item $part; throw "$f arrived damaged (SHA256 differs): run this again to download it anew" }
    }
    Move-Item $part $dst -Force
  }
  return $snap
}
if ($MyInvocation.InvocationName -ne "." -and $args.Count -ge 2) {
  $ErrorActionPreference = "Stop"
  Get-HfFiles $args[0] @($args | Select-Object -Skip 1)
}

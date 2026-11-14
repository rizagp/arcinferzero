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
# 111 GB UD-Q4_K_XL + 32 GB package = ~290 GB at the most, ~207 GB once the shards are deleted.
# The steps are Install-AutoRound in scripts\autoround.ps1, which setup.ps1 dot-sources too. This file is the command
# only: dot-sourced, its param block would set the caller's own $DeleteSource / $Yes (dot-source autoround.ps1).
param([switch]$DeleteSource, [switch]$Yes)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "autoround.ps1")

if ($MyInvocation.InvocationName -ne ".") {
  $null = Install-AutoRound -DeleteSource:$DeleteSource -Yes:$Yes
}

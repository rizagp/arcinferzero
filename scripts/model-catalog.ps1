# The models setup and the launcher offer: each one's repository and files (image projectors included), its size,
# the RAM it wants, a line for people choosing one, and Install-ArcModel, which downloads it into the Hugging Face
# cache (scripts\hf-get.ps1: no Python) with a Gemma drafter copied next to its model. 35b-gsq (converted with
# Python) is setup's own step. Dot-sourced by setup.ps1, get-models.ps1 and the launcher.
. (Join-Path $PSScriptRoot "hf-get.ps1")
$ArcModels = @(
  [pscustomobject]@{ key = "4b"; name = "Qwen3.5 4B (fastest; MTP, reads images)"; repo = "unsloth/Qwen3.5-4B-MTP-GGUF"; gb = 3.7; ram = 16
    plain = "Small and fast: everyday chat, summaries, pictures. Runs on any laptop with 16 GB."
    files = @("Qwen3.5-4B-UD-Q4_K_XL.gguf", "mmproj-F16.gguf") },
  [pscustomobject]@{ key = "9b"; name = "Qwen3.5 9B (MTP, reads images)"; repo = "unsloth/Qwen3.5-9B-MTP-GGUF"; gb = 7.1; ram = 16
    plain = "Better answers than 4B, a bit slower. 16 GB laptops."
    files = @("Qwen3.5-9B-UD-Q4_K_XL.gguf", "mmproj-F16.gguf") },
  [pscustomobject]@{ key = "35b"; name = "Qwen3.6 35B-A3B (fast mixture of experts; MTP, reads images)"; repo = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"; gb = 19.1; ram = 32
    plain = "The best all-rounder on a 32 GB laptop: smart and still quick (most of it stays on the SSD)."
    files = @("Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", "mmproj-F16.gguf") },
  [pscustomobject]@{ key = "27b"; name = "Qwen3.8 27B (best answers, slowest; MTP, reads images)"; repo = "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF"; gb = 13.1; ram = 32
    plain = "The most careful answers (code, reasoning), but slow. 32 GB laptops."
    files = @("Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf", "mmproj-Qwen3.8-27B-BF16.gguf") },
  # Google's QAT Q4_0 release has no drafter: Unsloth's copy of the assistant drafter is put next to the model
  [pscustomobject]@{ key = "g12"; name = "Gemma 4 12B (reads images; drafter)"; repo = "google/gemma-4-12B-it-qat-q4_0-gguf"; gb = 7.4; ram = 32
    plain = "Google's Gemma: good with pictures and documents, many languages."
    files = @("gemma-4-12b-it-qat-q4_0.gguf", "mmproj-gemma-4-12b-it-qat-q4_0.gguf", "mtp-gemma-4-12B-it.gguf")
    drafter = @{ repo = "unsloth/gemma-4-12B-it-qat-GGUF"; file = "mtp-gemma-4-12B-it.gguf" } },
  # (its projector: images and audio, also for transcripts with -Whisper)
  [pscustomobject]@{ key = "e4b"; name = "Gemma 4 E4B (images, audio; drafter)"; repo = "google/gemma-4-E4B-it-qat-q4_0-gguf"; gb = 6.2; ram = 16
    plain = "Google's small Gemma: hears audio and sees pictures. 16 GB laptops."
    files = @("gemma-4-E4B_q4_0-it.gguf", "gemma-4-E4B-it-mmproj.gguf", "mtp-gemma-4-E4B-it.gguf")
    drafter = @{ repo = "unsloth/gemma-4-E4B-it-qat-GGUF"; file = "mtp-gemma-4-E4B-it.gguf" } },
  # speech to text (arc-server -Whisper auto): Whisper large-v3, and Qwen3-ASR, the default speech model
  [pscustomobject]@{ key = "whisper"; name = "Whisper large-v3 (speech to text: words' times, live text)"; repo = "Systran/faster-whisper-large-v3"; gb = 3.1; ram = 16; speech = $true
    plain = "Speech to text with every word's time (subtitles, live text while you talk)."
    files = @("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.json") },
  [pscustomobject]@{ key = "asr"; name = "Qwen3-ASR 1.7B (speech to text: the default model)"; repo = "unslothai/Qwen3-ASR-1.7B-GGUF"; gb = 2.4; ram = 16; speech = $true
    plain = "Speech to text for recordings and meetings (Indonesian and English mixed, noisy rooms)."
    files = @("Qwen3-ASR-1.7B-Q8_0.gguf", "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf") }
)
# whether a model's files are all in the cache already
function Test-ArcModel($m, [string]$Hub = (Get-HfHub)) {
  $snaps = Join-Path $Hub ("models--" + ($m.repo -replace "/", "--") + "\snapshots")
  if (-not (Test-Path $snaps)) { return $false }
  foreach ($f in $m.files) { if (-not (Get-ChildItem $snaps -Recurse -Filter (Split-Path $f -Leaf) -ErrorAction SilentlyContinue)) { return $false } }
  return $true
}
# downloads a model's files (and a Gemma drafter into the model's folder, where the engine looks for it); returns
# its snapshot folder
function Install-ArcModel($m) {
  [string[]]$own = $m.files
  if ($m.drafter) { $own = @($m.files | Where-Object { $_ -ne $m.drafter.file }) }
  $snap = Get-HfFiles $m.repo $own
  if ($m.drafter) {
    $dsnap = Get-HfFiles $m.drafter.repo @($m.drafter.file)
    Copy-Item (Join-Path $dsnap $m.drafter.file) (Join-Path $snap $m.drafter.file) -Force
  }
  return $snap
}

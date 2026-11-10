# Pi's model list for inferzero (provider "inferzero" in ~/.pi/agent/models.json): every model serverz.exe can load,
# so Pi's /model switches between them (the server loads the requested one). Context window: each model's own, as
# `serverz.exe --list-models` computes it for the current GPU memory; the running model's from the server.
# Dot-sourced by pi-arc.ps1 and install-pi.ps1. Other providers in models.json are kept.
function Update-PiModels([string]$ServerExe, [string]$Base, [string]$RunningId = "", [int]$RunningCtx = 0) {
  $all = @(& $ServerExe --list-models | ConvertFrom-Json | ForEach-Object { $_ }) | Where-Object { $_.supported }  # (unroll)
  $cfgPath = Join-Path $env:USERPROFILE ".pi\agent\models.json"
  $cfg = if (Test-Path $cfgPath) { Get-Content $cfgPath -Raw | ConvertFrom-Json } else { [pscustomobject]@{} }
  if (-not $cfg.PSObject.Properties["providers"]) { $cfg | Add-Member providers ([pscustomobject]@{}) }
  # (the provider was called "arcinfer" until 2026-10-24 and "arcinferzero" until 2026-11-10, the project's renames:
  #  those entries are replaced)
  $renamed = $false
  foreach ($old in "arcinfer", "arcinferzero") {
    if ($cfg.providers.PSObject.Properties[$old]) { $cfg.providers.PSObject.Properties.Remove($old); $renamed = $true }
  }
  if (-not $cfg.providers.PSObject.Properties["inferzero"]) {
    $cfg.providers | Add-Member inferzero ([pscustomobject]@{ baseUrl = "$Base/v1"; api = "openai-completions"; apiKey = "inferzero"; models = @() })
  }
  $list = New-Object System.Collections.Generic.List[object]
  # several files of one repository (e.g. Qwen3.6 35B UD-Q3_K_M and UD-Q3_K_S): the file name tells them apart in /model
  # (every variant of a file shares its name: count distinct files)
  $shared = @($all | Group-Object name | Where-Object { @($_.Group | Select-Object -ExpandProperty path -Unique).Count -gt 1 } | ForEach-Object { $_.Name })
  foreach ($m in $all) {
    $n = if ($RunningId -and $m.alias -eq $RunningId -and $RunningCtx -gt 0) { $RunningCtx } elseif ($m.ctx -gt 0) { [int]$m.ctx } else { [int][Math]::Min(32768, $m.ctx_train) }
    $list.Add([pscustomobject][ordered]@{
      # e.g. "unsloth/Qwen3.5-9B-MTP-GGUF - MTP + vision - 143K (inferzero)" (ASCII: console output / PS 5 encodings)
      id = $m.alias; name = ("{0}{1}{2} - {3}K (inferzero)" -f $m.name, $(if ($shared -contains $m.name) { " " + ($m.file -replace '\.gguf$', '') } else { "" }), $(if ($m.tags) { " - $($m.tags)" } else { "" }), [int]($n / 1024)); reasoning = $true
      input = [string[]]$(if ($m.vision) { "text", "image" } else { "text" })  # typed: stays a JSON array with one item
      contextWindow = $n; maxTokens = $n
      cost = [ordered]@{ input = 0; output = 0; cacheRead = 0; cacheWrite = 0 }
      compat = [ordered]@{ thinkingFormat = "qwen"; supportsReasoningEffort = $true; thinkingTokenBudgetField = "thinking_token_budget"; maxTokensField = "max_tokens" }
    })
  }
  $json = ($cfg | ConvertTo-Json -Depth 10)
  $cfg.providers.inferzero.models = $list.ToArray()
  $new = ($cfg | ConvertTo-Json -Depth 10)
  if ($new -ne $json -or $renamed) {
    if (Test-Path $cfgPath) { Copy-Item $cfgPath "$cfgPath.bak" -Force }
    [IO.File]::WriteAllText($cfgPath, $new, (New-Object Text.UTF8Encoding($false)))  # no BOM (Node JSON.parse)
  }
  return @($list | ForEach-Object { $_.id })
}

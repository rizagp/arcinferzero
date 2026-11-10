# Opens the web UI to the internet through Cloudflare (docs/remote-access.md), once remote access is set up in the
# web UI (Settings > Remote access: a password and the host names).
#   .\scripts\remote-tunnel.ps1                 a quick tunnel: a random https://<words>.trycloudflare.com address
#                                               (no Cloudflare account; host name *.trycloudflare.com in Settings)
#   .\scripts\remote-tunnel.ps1 -Name chat      a named tunnel of your Cloudflare account ("cloudflared tunnel login",
#                                               "cloudflared tunnel create chat", a DNS route to it first)
#   -Port 8080                                  the server's port (default: server.json's, else 8080)
# Needs cloudflared (winget install --id Cloudflare.cloudflared). Ctrl+C closes the tunnel.
param([string]$Name = "", [int]$Port = 0)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "app-dir.ps1")
$data = Get-AppDir
$cf = Get-Command cloudflared -ErrorAction SilentlyContinue
if (-not $cf) {
  Write-Host "cloudflared is not installed. Install it with:  winget install --id Cloudflare.cloudflared" -ForegroundColor Yellow
  exit 1
}
if ($Port -le 0) {
  $Port = 8080
  $sj = Join-Path $data "server.json"
  if (Test-Path -LiteralPath $sj) {
    try { $p = (Get-Content -LiteralPath $sj -Raw | ConvertFrom-Json).port; if ($p) { $Port = [int]$p } } catch {}
  }
}
# remote access must be on with a password, else every request through the tunnel is refused (403)
$rj = Join-Path $data "remote.json"
$cfg = $null
if (Test-Path -LiteralPath $rj) { try { $cfg = Get-Content -LiteralPath $rj -Raw | ConvertFrom-Json } catch {} }
if (-not $cfg -or -not $cfg.enabled -or -not $cfg.hash) {
  Write-Host "Remote access is off. In the web UI on this computer: Settings > Remote access: a password, the host name" -ForegroundColor Yellow
  Write-Host "(*.trycloudflare.com for a quick tunnel), the switch on, Save. Then run this again." -ForegroundColor Yellow
  exit 1
}
if (-not $Name -and -not ($cfg.hosts -contains "*.trycloudflare.com")) {
  Write-Host "A quick tunnel's address is <words>.trycloudflare.com: add *.trycloudflare.com to the host names in Settings > Remote access." -ForegroundColor Yellow
  exit 1
}
try { $null = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "http://127.0.0.1:$Port/health" } catch {
  Write-Host "No inferzero server answers on port $Port." -ForegroundColor Yellow
  exit 1
}
Write-Host "Tunnel to http://127.0.0.1:$Port (Ctrl+C closes it). Sign in with the remote password; the address stays private" -ForegroundColor Cyan
Write-Host "only as long as you do not share it." -ForegroundColor Cyan
# (the Host header is left as the visitor sent it: the server needs it to tell a remote request from a local one)
if ($Name) { & $cf.Source tunnel --no-autoupdate run --url "http://127.0.0.1:$Port" $Name }
else { & $cf.Source tunnel --no-autoupdate --url "http://127.0.0.1:$Port" }

# The launcher's part for people new to it (dot-sourced by arc-launcher.ps1, after its window and buttons exist):
#  - the package's files unblocked once (a zip from the internet marks each file as downloaded: Windows then warns
#    before every script and program, or refuses them);
#  - the Intel GPU driver checked (bin\clinfo.exe), with the driver's download page when it is missing;
#  - "Get models": a short list in plain words (the catalog of scripts\model-catalog.ps1), the one that suits this
#    laptop's RAM ticked, speech to text as an add-on, a desktop shortcut; the download runs in the background
#    (scripts\get-models.ps1, no Python) with a progress bar; then the model starts and the web UI opens;
#  - with no model on the disk the dialog opens by itself.
# (ASCII only: Windows PowerShell 5.1 reads scripts without a BOM as ANSI)
. (Join-Path $PSScriptRoot "model-catalog.ps1")

# ---- unblock once per folder the package was unpacked to
$unblockMark = Join-Path $stateDir "unblocked.txt"
$doneFor = if (Test-Path $unblockMark) { (Get-Content $unblockMark -Raw).Trim() } else { "" }
if ($doneFor -ne $root) {
  Get-ChildItem $root -Recurse -File -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue
  Set-Content $unblockMark $root
}

# ---- the GPU driver: clinfo finds the Arc GPU through OpenCL, or the driver is missing / too old
function Test-GpuDriver {
  $ci = Join-Path $root "bin\clinfo.exe"
  if (-not (Test-Path $ci)) { return $true }
  $out = (& $ci 2>&1 | Out-String)
  if ($LASTEXITCODE -eq 0 -and $out -match "device:") { return $true }
  $r = [System.Windows.Forms.MessageBox]::Show(
    "No Intel Arc GPU was found through its driver.`n`narcinferzero needs a laptop with an Intel Core Ultra (Arc graphics) and Intel's graphics driver. Open Intel's driver page now? (Install it, restart, then open arcinferzero again.)",
    "arcinferzero", "YesNo", "Warning")
  if ($r -eq "Yes") { Start-Process "https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html" }
  return $false
}

# ---- a desktop shortcut to this window
function New-DesktopShortcut {
  $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "arcinferzero.lnk"
  $ws = New-Object -ComObject WScript.Shell
  $s = $ws.CreateShortcut($lnk)
  $s.TargetPath = Join-Path $root "arc-launcher.cmd"
  $s.WorkingDirectory = $root
  $s.WindowStyle = 7  # (minimized: the .cmd only starts the window)
  $s.Description = "arcinferzero: local AI on the Intel Arc GPU"
  $s.Save()
}

# ---- what the dialog downloaded: started when the model list shows it
$script:wantStart = $null   # catalog entry of the chat model to start
$script:wantSpeech = $false
$script:openWeb = $false    # the web UI opens once a start ends well

function Show-GetModels([bool]$first) {
  $ramGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
  $hub = Get-HfHub
  $drive = Get-PSDrive ((Split-Path $hub -Qualifier).TrimEnd(":")) -ErrorAction SilentlyContinue
  $chat = @($ArcModels | Where-Object { -not $_.speech })
  $speech = @($ArcModels | Where-Object { $_.speech })
  $rec = if ($ramGb -ge 30) { "35b" } else { "4b" }

  $d = New-Object System.Windows.Forms.Form
  $d.Text = $(if ($first) { "Welcome to arcinferzero: get a model" } else { "Get models" })
  $d.Size = New-Object System.Drawing.Size(760, 560)
  $d.StartPosition = "CenterParent"
  $d.Font = $form.Font
  $d.FormBorderStyle = "FixedDialog"; $d.MaximizeBox = $false; $d.MinimizeBox = $false
  $p = New-Object System.Windows.Forms.FlowLayoutPanel
  $p.Dock = "Fill"; $p.FlowDirection = "TopDown"; $p.WrapContents = $false; $p.AutoScroll = $true
  $p.Padding = New-Object System.Windows.Forms.Padding(14, 10, 14, 10)
  $intro = New-Object System.Windows.Forms.Label
  $intro.AutoSize = $true; $intro.MaximumSize = New-Object System.Drawing.Size(700, 0)
  $intro.Text = $(if ($first) { "arcinferzero runs AI models on this laptop's own GPU: nothing goes to the internet. First it needs a model. " } else { "" }) +
    "This laptop has $ramGb GB of RAM; the model that suits it is ticked. Free disk space: " +
    $(if ($drive) { "{0:N0} GB" -f ($drive.Free / 1GB) } else { "?" }) + ". You can add more models later (Get models)."
  $p.Controls.Add($intro)
  $radios = @()
  foreach ($m in $chat) {
    $have = Test-ArcModel $m $hub
    $rb = New-Object System.Windows.Forms.RadioButton
    $rb.AutoSize = $true; $rb.MaximumSize = New-Object System.Drawing.Size(700, 0)
    $rb.Margin = New-Object System.Windows.Forms.Padding(3, 8, 3, 0)
    $tooBig = $ramGb -lt $m.ram - 2
    $rb.Text = "{0}  ({1:N1} GB{2})`n      {3}{4}" -f ($m.name -replace ' \(.*$', ''), $m.gb, $(if ($have) { ", already here" } else { "" }), $m.plain,
      $(if ($tooBig) { "  Needs $($m.ram) GB of RAM." } else { "" })
    $rb.Tag = $m
    $rb.Enabled = -not $tooBig
    $rb.Checked = ($m.key -eq $rec)
    $p.Controls.Add($rb); $radios += $rb
  }
  $sp = New-Object System.Windows.Forms.CheckBox
  $sp.AutoSize = $true; $sp.MaximumSize = New-Object System.Drawing.Size(700, 0); $sp.Margin = New-Object System.Windows.Forms.Padding(3, 14, 3, 0)
  $spGb = ($speech | Measure-Object gb -Sum).Sum
  $sp.Text = "Also speech to text ({0:N1} GB): transcripts of recordings and meetings, voice messages (Qwen3-ASR and Whisper)" -f $spGb
  $sp.Checked = -not $first -and $false
  $p.Controls.Add($sp)
  $sc = New-Object System.Windows.Forms.CheckBox
  $sc.AutoSize = $true; $sc.Text = "Put an arcinferzero shortcut on the desktop"; $sc.Margin = New-Object System.Windows.Forms.Padding(3, 8, 3, 0)
  $sc.Checked = $first -and -not (Test-Path (Join-Path ([Environment]::GetFolderPath("Desktop")) "arcinferzero.lnk"))
  $sc.Visible = $sc.Checked
  $p.Controls.Add($sc)
  $bar = New-Object System.Windows.Forms.ProgressBar
  $bar.Width = 700; $bar.Height = 18; $bar.Margin = New-Object System.Windows.Forms.Padding(3, 16, 3, 0); $bar.Visible = $false
  $p.Controls.Add($bar)
  $info = New-Object System.Windows.Forms.Label
  $info.AutoSize = $true; $info.MaximumSize = New-Object System.Drawing.Size(700, 0); $info.ForeColor = [System.Drawing.Color]::DimGray
  $p.Controls.Add($info)
  $btns = New-Object System.Windows.Forms.FlowLayoutPanel
  $btns.Dock = "Bottom"; $btns.Height = 44; $btns.FlowDirection = "RightToLeft"; $btns.Padding = New-Object System.Windows.Forms.Padding(8)
  $go = New-Object System.Windows.Forms.Button; $go.Text = "Download and start"; $go.Width = 160
  $go.Font = New-Object System.Drawing.Font("Segoe UI", 9, [System.Drawing.FontStyle]::Bold)
  $close = New-Object System.Windows.Forms.Button; $close.Text = $(if ($first) { "Later" } else { "Close" }); $close.Width = 100
  $btns.Controls.AddRange(@($go, $close))
  $d.Controls.Add($p); $d.Controls.Add($btns)
  $d.AcceptButton = $go

  $state = @{ proc = $null; out = (Join-Path $env:TEMP "arc-get-models.log"); keys = @(); total = 0.0; dirs = @(); chat = $null }
  $t = New-Object System.Windows.Forms.Timer
  $t.Interval = 1000
  $t.Add_Tick({
      if (-not $state.proc) { return }
      # (progress: the bytes in the repositories' cache folders, .part files included, against their sizes)
      $now = 0.0
      foreach ($dir in $state.dirs) {
        if (Test-Path $dir) { $now += (Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum }
      }
      $bar.Value = [int][math]::Min(100, [math]::Max(0, 100 * ($now - $state.start) / [math]::Max(1, $state.total)))
      $lines = @(Get-Content $state.out -ErrorAction SilentlyContinue)
      $cur = ($lines | Where-Object { $_ -like "START *" } | Select-Object -Last 1) -replace '^START ', ''
      $info.Text = "Downloading $cur ... {0:N1} of {1:N1} GB (a cut connection is fine: Download again goes on where it stopped)" -f (($now - $state.start) / 1e9), ($state.total / 1e9)
      if ($state.proc.HasExited) {
        $t.Stop()
        $fails = @($lines | Where-Object { $_ -like "FAIL *" })
        $state.proc = $null
        if ($fails.Count) {
          $info.Text = "Not finished: " + (($fails | ForEach-Object { $_ -replace '^FAIL ', '' }) -join "; ")
          $go.Enabled = $true; $go.Text = "Download again"
          return
        }
        $bar.Value = 100
        $script:wantStart = $state.chat
        $script:wantSpeech = $state.speech
        $script:openWeb = $true
        $d.Close()
        Start-List
      }
    })
  $go.Add_Click({
      $sel = $radios | Where-Object { $_.Checked } | Select-Object -First 1
      $keys = @()
      if ($sel) { $keys += $sel.Tag.key }
      if ($sp.Checked) { $keys += @($speech | ForEach-Object { $_.key }) }
      if (-not $keys) { return }
      $ms = @($ArcModels | Where-Object { $keys -contains $_.key })
      $need = ($ms | Where-Object { -not (Test-ArcModel $_ $hub) } | Measure-Object gb -Sum).Sum
      if ($drive -and $drive.Free / 1GB -lt $need + 5) {
        [void][System.Windows.Forms.MessageBox]::Show(("Not enough free disk space: {0:N0} GB needed, {1:N0} GB free on {2}:." -f ($need + 5), ($drive.Free / 1GB), $drive.Name), "arcinferzero")
        return
      }
      if ($sc.Checked) { try { New-DesktopShortcut } catch {} }
      $state.chat = if ($sel) { $sel.Tag } else { $null }
      $state.speech = $sp.Checked
      $state.dirs = @($ms | ForEach-Object { Join-Path $hub ("models--" + ($_.repo -replace "/", "--")) })
      $state.total = $need * 1e9
      $state.start = 0.0
      foreach ($dir in $state.dirs) { if (Test-Path $dir) { $state.start += (Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum } }
      Remove-Item $state.out -ErrorAction SilentlyContinue
      $a = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$(Join-Path $PSScriptRoot 'get-models.ps1')`"") + $keys
      $state.proc = Start-Process -FilePath $shell -ArgumentList $a -WindowStyle Hidden -PassThru -RedirectStandardOutput $state.out `
        -RedirectStandardError (Join-Path $env:TEMP "arc-get-models.err")
      $go.Enabled = $false
      $bar.Visible = $true; $bar.Value = 0
      $info.Text = "Starting the download..."
      $t.Start()
    })
  $close.Add_Click({ $d.Close() })
  $d.Add_FormClosing({
      param($s, $e)
      if ($state.proc -and -not $state.proc.HasExited) {
        $r = [System.Windows.Forms.MessageBox]::Show("Stop the download? (Started again later, it goes on where it stopped.)", "arcinferzero", "YesNo")
        if ($r -ne "Yes") { $e.Cancel = $true; return }
        try { Stop-Process -Id $state.proc.Id -Force } catch {}
        Get-CimInstance Win32_Process -Filter "ParentProcessId = $($state.proc.Id)" -ErrorAction SilentlyContinue | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force } catch {} }
      }
      $t.Stop()
    })
  [void]$d.ShowDialog($form)
}

# ---- after the model list: start what the dialog downloaded, or open the dialog when no model is there
function After-List {
  $ok = @($script:models | Where-Object { $_.supported })
  if ($script:wantStart) {
    $w = $script:wantStart
    $script:wantStart = $null
    $dirKey = "models--" + ($w.repo -replace "/", "--")
    $row = $null
    foreach ($r in $grid.Rows) {
      $m = $r.Tag
      if ($m.supported -and $m.path -and $m.path -like "*$dirKey*" -and (Split-Path $m.path -Leaf) -eq $w.files[0]) { $row = $r; break }
    }
    if ($row) {
      $grid.ClearSelection(); $row.Selected = $true
      if ($script:wantSpeech) { Put-Field "whisper" "auto"; $script:over["whisper"] = "auto"; Show-Mark "whisper"; Update-Speech }
      Start-Selected
    }
    return
  }
  if (-not $ok.Count -and -not $script:askedFirst) {
    $script:askedFirst = $true
    Show-GetModels $true
  }
}

# ---- buttons: Get models, and the web UI opened once a start ends well
$getBtn = New-Button "Get models" 110
$getBtn.Font = New-Object System.Drawing.Font("Segoe UI", 9, [System.Drawing.FontStyle]::Bold)
$buttons.Controls.Add($getBtn)
$buttons.Controls.SetChildIndex($getBtn, 0)
$getBtn.Add_Click({ Show-GetModels $false })
$openWebChk = New-Check "Open the web UI when started"
$openWebChk.Checked = $true
$buttons.Controls.Add($openWebChk)

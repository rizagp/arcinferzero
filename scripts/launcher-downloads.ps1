# The launcher's Downloads dialog (dot-sourced by arc-launcher.ps1 after launcher-firstrun.ps1; the "Get models"
# button and More > "Download models..." open it):
#  - the models of scripts\model-catalog.ps1, several at once, the ones already here marked;
#  - or a Hugging Face link pasted in (a repository, a folder of it, or one file: .../blob/main/x.gguf,
#    .../resolve/main/x.gguf): its GGUF files listed with their sizes, the linked one ticked with the other parts of a
#    split file and an image projector (mmproj);
#  - Pause / Continue: Pause stops curl and keeps the <file>.part files, Continue goes on where they end (curl -C -);
#  - a speed limit (curl --limit-rate through $env:ARC_HF_RATE), changed while a download runs: it goes on at once
#    with the new limit.
# The download itself: scripts\get-models.ps1 (catalog keys, and "hf:<org>/<name>/<path>" for a link's files) in the
# background, into the Hugging Face cache the server reads. (ASCII only: Windows PowerShell 5.1 reads scripts without
# a BOM as ANSI)

# a link (or "org/name") -> @{ repo; path }: path "" for the repository, a folder or a file in it
function Read-HfLink([string]$t) {
  $t = $t.Trim()
  $m = [regex]::Match($t, '^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/(?:models/)?([\w.\-]+)/([\w.\-]+)(?:/(?:blob|resolve|tree)/[^/?#]+(?:/([^?#]*))?)?')
  if ($m.Success) { return @{ repo = "$($m.Groups[1].Value)/$($m.Groups[2].Value)"; path = [uri]::UnescapeDataString($m.Groups[3].Value).Trim("/") } }
  $m = [regex]::Match($t, '^([\w.\-]+)/([\w.\-]+)$')
  if ($m.Success) { return @{ repo = $t; path = "" } }
  return $null
}
# the files of a repository (recursive), as the Hugging Face API lists them
function Get-HfTree([string]$repo) {
  $hdr = @{}
  $tok = Get-HfToken
  if ($tok) { $hdr.Authorization = "Bearer $tok" }
  return @(Invoke-RestMethod "https://huggingface.co/api/models/$repo/tree/main?recursive=1" -Headers $hdr -TimeoutSec 60 | ForEach-Object { $_ })
}
# the parts of a split GGUF that belong with a path (x-00001-of-00002.gguf -> both parts), else the path alone
function Get-SplitSet([string]$p, $paths) {
  $m = [regex]::Match($p, '^(.*)-\d{5}-of-(\d{5})\.gguf$')
  if (-not $m.Success) { return @($p) }
  return @($paths | Where-Object { $_ -match ('^' + [regex]::Escape($m.Groups[1].Value) + '-\d{5}-of-' + $m.Groups[2].Value + '\.gguf$') })
}
# a catalog model already here: in the Hugging Face cache, or kept as a package (serverz --pack) in
# %USERPROFILE%\.arcinferzero\zero\models (one GGUF for a split model: Flash Next)
function Test-ArcHave($m, [string]$hub) {
  if (Test-ArcModel $m $hub) { return $true }
  $pk = Join-Path $env:USERPROFILE ".arcinferzero\zero\models"
  if (-not (Test-Path $pk)) { return $false }
  $main = Split-Path $m.files[0] -Leaf
  $one = $main -replace "-00001-of-\d{5}\.gguf$", ".gguf"
  return [bool](Get-ChildItem $pk -Recurse -File -Filter *.gguf -ErrorAction SilentlyContinue | Where-Object { $_.Name -in $main, $one })
}
# a process and everything it started (get-models.ps1 -> curl.exe)
function Stop-Tree([int]$id) {
  Get-CimInstance Win32_Process -Filter "ParentProcessId = $id" -ErrorAction SilentlyContinue | ForEach-Object { Stop-Tree $_.ProcessId }
  try { Stop-Process -Id $id -Force -ErrorAction Stop } catch {}
}

function Show-Downloads {
  $hub = Get-HfHub
  $drive = Get-PSDrive ((Split-Path $hub -Qualifier).TrimEnd(":")) -ErrorAction SilentlyContinue
  $W = 760
  $d = New-Object System.Windows.Forms.Form
  $d.Text = "Download models"
  $d.AutoScaleMode = "None"
  $d.Size = New-Object System.Drawing.Size((S ($W + 40)), (S 680))
  $d.StartPosition = "CenterParent"
  $d.Font = $form.Font
  $d.BackColor = $Pal.bg; $d.ForeColor = $Pal.text
  Set-WindowStyle $d
  $d.FormBorderStyle = "FixedDialog"; $d.MaximizeBox = $false; $d.MinimizeBox = $false
  $p = New-Object System.Windows.Forms.FlowLayoutPanel
  $p.Dock = "Fill"; $p.FlowDirection = "TopDown"; $p.WrapContents = $false; $p.AutoScroll = $true
  $p.Padding = New-Object System.Windows.Forms.Padding((S 14), (S 10), (S 14), (S 10))
  function Lbl([string]$t, [bool]$sub = $false, [int]$top = 8) {
    $l = New-Object System.Windows.Forms.Label
    $l.AutoSize = $true; $l.MaximumSize = New-Object System.Drawing.Size((S $W), 0); $l.Text = $t
    $l.Margin = New-Object System.Windows.Forms.Padding((S 3), (S $top), (S 3), 0)
    if ($sub) { $l.ForeColor = $Pal.sub }
    $p.Controls.Add($l); $l
  }
  function New-List([int]$h) {
    $c = New-Object System.Windows.Forms.CheckedListBox
    $c.Width = S $W; $c.Height = S $h; $c.CheckOnClick = $true; $c.IntegralHeight = $false
    $c.BackColor = $Pal.card; $c.ForeColor = $Pal.text; $c.BorderStyle = "FixedSingle"
    $p.Controls.Add($c); $c
  }
  [void](Lbl ("Models from the list. Free disk space: " + $(if ($drive) { "{0:N0} GB on {1}:" -f ($drive.Free / 1GB), $drive.Name } else { "?" })) $false 0)
  $cat = New-List 190
  $items = @()  # what each row of a list stands for
  foreach ($m in $ArcModels) {
    $have = Test-ArcHave $m $hub
    [void]$cat.Items.Add(("{0}  ({1:N1} GB){2}" -f $m.name, $m.gb, $(if ($have) { "  - already here" } else { "" })))
  }
  [void](Lbl "Or a Hugging Face link: a model's page, a folder of it, or one file (copy it from the address bar or the file's download link)" $false 12)
  $row = New-Object System.Windows.Forms.FlowLayoutPanel
  $row.AutoSize = $true; $row.WrapContents = $false; $row.Margin = New-Object System.Windows.Forms.Padding(0)
  $link = New-Object System.Windows.Forms.TextBox
  $link.Width = S ($W - 130); $link.BackColor = $Pal.input; $link.ForeColor = $Pal.text; $link.BorderStyle = "FixedSingle"
  $look = New-Button "Look up" 110
  $row.Controls.AddRange(@($link, $look))
  $p.Controls.Add($row)
  $hfList = New-List 150
  $hfInfo = Lbl "" $true 4
  $rowS = New-Object System.Windows.Forms.FlowLayoutPanel
  $rowS.AutoSize = $true; $rowS.WrapContents = $false; $rowS.Margin = New-Object System.Windows.Forms.Padding(0, (S 12), 0, 0)
  $sl = New-Object System.Windows.Forms.Label
  $sl.AutoSize = $true; $sl.Text = "Speed limit"; $sl.Margin = New-Object System.Windows.Forms.Padding((S 3), (S 6), (S 8), 0)
  $speed = New-Object System.Windows.Forms.ComboBox
  $speed.DropDownStyle = "DropDownList"; $speed.Width = S 140; $speed.BackColor = $Pal.input; $speed.ForeColor = $Pal.text
  $rates = [ordered]@{ "No limit" = ""; "1 MB/s" = "1M"; "2 MB/s" = "2M"; "5 MB/s" = "5M"; "10 MB/s" = "10M"; "20 MB/s" = "20M"; "50 MB/s" = "50M" }
  foreach ($k in $rates.Keys) { [void]$speed.Items.Add($k) }
  $speed.SelectedIndex = 0
  $rowS.Controls.AddRange(@($sl, $speed))
  $p.Controls.Add($rowS)
  $bar = New-Object System.Windows.Forms.ProgressBar
  $bar.Width = S $W; $bar.Height = S 18; $bar.Margin = New-Object System.Windows.Forms.Padding((S 3), (S 14), (S 3), 0); $bar.Visible = $false
  $p.Controls.Add($bar)
  $info = Lbl "" $true 6
  $btns = New-Object System.Windows.Forms.FlowLayoutPanel
  $btns.Dock = "Bottom"; $btns.Height = S 48; $btns.FlowDirection = "RightToLeft"; $btns.Padding = New-Object System.Windows.Forms.Padding((S 8))
  $go = New-Button "Download" 120 -Primary
  $pause = New-Button "Pause" 110; $pause.Enabled = $false
  $close = New-Button "Close" 100
  $btns.Controls.AddRange(@($go, $pause, $close))
  $d.Controls.Add($p); $d.Controls.Add($btns)

  # st: the download. tokens: what get-models.ps1 is given; done: those finished; dirs: the cache folders whose bytes
  # show the progress; total: the bytes to download; base: the bytes there before the first start (not reset by Continue)
  $st = @{ proc = $null; tokens = @(); done = @{}; fails = @(); dirs = @(); total = 0.0; base = 0.0; paused = $false; retryAt = $null; retries = 0; why = ""
    out = (Join-Path $env:TEMP "arc-downloads.log"); last = 0.0; lastT = [DateTime]::Now; hf = @(); hfRepo = "" }
  # (a .part file's size from the file itself: a directory listing gives the size of the read before, while curl
  #  writes it, so the bar and the speed lagged one tick)
  function Bytes-Now {
    $n = 0.0
    foreach ($dir in $st.dirs) {
      if (-not (Test-Path $dir)) { continue }
      foreach ($f in @(Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue)) {
        $n += $(if ($f.Extension -eq ".part") { (New-Object IO.FileInfo $f.FullName).Length } else { $f.Length })
      }
    }
    $n
  }
  function Start-Run {
    $left = @($st.tokens | Where-Object { -not $st.done.ContainsKey(($_ -replace '^(hf:[^/]+/[^/]+)/.*$', '$1')) })
    if (-not $left) { return }
    Remove-Item $st.out -ErrorAction SilentlyContinue
    $st.fails = @()
    $rate = $rates[[string]$speed.SelectedItem]
    $a = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$(Join-Path $PSScriptRoot 'get-models.ps1')`"") + @($left | ForEach-Object { "`"$_`"" })
    # (the child inherits the environment: the limit set just for its start)
    $old = $env:ARC_HF_RATE
    if ($rate) { $env:ARC_HF_RATE = $rate } else { Remove-Item Env:ARC_HF_RATE -ErrorAction SilentlyContinue }
    try {
      $st.proc = Start-Process -FilePath $shell -ArgumentList $a -WindowStyle Hidden -PassThru -RedirectStandardOutput $st.out `
        -RedirectStandardError (Join-Path $env:TEMP "arc-downloads.err")
    } finally { if ($old) { $env:ARC_HF_RATE = $old } else { Remove-Item Env:ARC_HF_RATE -ErrorAction SilentlyContinue } }
    $st.paused = $false
    $st.last = Bytes-Now; $st.lastT = [DateTime]::Now
    $pause.Text = "Pause"; $pause.Enabled = $true; $go.Enabled = $false
    $bar.Visible = $true
    $info.Text = "Starting the download..."
    $t.Start()
  }
  function Stop-Run {
    if ($st.proc -and -not $st.proc.HasExited) { Stop-Tree $st.proc.Id }
    $st.proc = $null
  }
  # the log of the current run: DONE / FAIL lines into st
  function Read-Log {
    foreach ($l in @(Get-Content $st.out -ErrorAction SilentlyContinue)) {
      if ($l -match '^DONE (\S+)') { $st.done[$Matches[1]] = $true }
      elseif ($l -match '^FAIL (\S+) (.*)$') { if ($st.fails -notcontains $l) { $st.fails += $l } }
    }
  }
  $t = New-Object System.Windows.Forms.Timer
  $t.Interval = 1000
  $t.Add_Tick({
      try {
        if (-not $st.proc) {
          # (a cut connection: started again when its wait is over, for as long as it takes; Pause stops that)
          if ($st.retryAt) {
            $left = [int][math]::Ceiling(($st.retryAt - [DateTime]::Now).TotalSeconds)
            if ($left -le 0) { $st.retryAt = $null; Start-Run }
            else { $info.Text = "The connection was cut ({0}). Trying again in {1} s (attempt {2}); it goes on where it stopped. Pause stops this." -f $st.why, $left, ($st.retries + 1) }
          }
          return
        }
        $now = Bytes-Now
        if ($now -gt $st.last + 1MB) { $st.retries = 0 }  # (data again: the next cut waits 5 s again)
        $sec = ([DateTime]::Now - $st.lastT).TotalSeconds
        $rateNow = if ($sec -gt 0.2) { [math]::Max(0, ($now - $st.last) / $sec) } else { 0 }
        $st.last = $now; $st.lastT = [DateTime]::Now
        $got = [math]::Max(0, $now - $st.base)
        $bar.Value = [int][math]::Min(100, [math]::Max(0, 100 * $got / [math]::Max(1, $st.total)))
        $lines = @(Get-Content $st.out -ErrorAction SilentlyContinue)
        $cur = ($lines | Where-Object { $_ -like "START *" } | Select-Object -Last 1) -replace '^START ', ''
        $info.Text = "Downloading {0} ... {1:N2} of {2:N2} GB, {3:N1} MB/s" -f $cur, ($got / 1e9), ($st.total / 1e9), ($rateNow / 1MB)
        if ($st.proc.HasExited) {
          $t.Stop()
          $st.proc = $null
          Read-Log
          $pause.Enabled = $false
          # (a failure no new try mends, shown at once: a repository that asks for a token, a file it does not have,
          #  a model name the catalog does not know; anything else, the connection above all, is tried again without
          #  end, waiting 5 s, 10, 20, 40, then 60 s between tries)
          $fatal = @($st.fails | Where-Object { $_ -match "sign in|has no file|unknown model|not found" })
          if ($st.fails.Count -and -not $fatal.Count) {
            $st.why = (($st.fails | Select-Object -Last 1) -replace '^FAIL \S+ ', '')
            if ($st.why.Length -gt 90) { $st.why = $st.why.Substring(0, 90) + "..." }
            $st.fails = @()
            $wait = [math]::Min(60, 5 * [math]::Pow(2, [math]::Min($st.retries, 4)))
            $st.retries++
            $st.retryAt = [DateTime]::Now.AddSeconds($wait)
            $pause.Enabled = $true; $pause.Text = "Pause"
            $t.Start()
            return
          }
          if ($st.fails.Count) {
            $info.Text = "Not finished: " + (($st.fails | ForEach-Object { $_ -replace '^FAIL ', '' }) -join "; ") + " (Download again goes on where it stopped)"
            $st.fails = @()
            $go.Enabled = $true; $go.Text = "Download again"
            return
          }
          $bar.Value = 100
          $info.Text = "Done. The model list is read again: select the model and press Start."
          $go.Enabled = $true; $go.Text = "Download"
          Start-List
        }
      } catch { $info.Text = "Error: $($_.Exception.Message)" }
    })

  $look.Add_Click({
      $r = Read-HfLink $link.Text
      if (-not $r) { $hfInfo.Text = "Not a Hugging Face link: e.g. https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF"; $hfInfo.ForeColor = $Pal.bad; return }
      $hfInfo.ForeColor = $Pal.sub; $hfInfo.Text = "Reading $($r.repo) ..."; $d.Refresh()
      try { $tree = Get-HfTree $r.repo }
      catch {
        $code = try { [int]$_.Exception.Response.StatusCode } catch { 0 }
        $hfInfo.ForeColor = $Pal.bad
        $hfInfo.Text = $(if ($code -in 401, 403) { "$($r.repo) asks you to sign in and accept its terms on huggingface.co, then a read token in HF_TOKEN (or hf auth login)." }
          elseif ($code -eq 404) { "$($r.repo) not found (a private repository needs a token in HF_TOKEN)." }
          else { "Cannot reach huggingface.co: $($_.Exception.Message)" })
        return
      }
      $g = @($tree | Where-Object { $_.type -eq "file" -and $_.path -like "*.gguf" } | Sort-Object path)
      $hfList.Items.Clear(); $st.hf = @(); $st.hfRepo = $r.repo
      if (-not $g) { $hfInfo.ForeColor = $Pal.bad; $hfInfo.Text = "$($r.repo) has no GGUF file."; return }
      $paths = @($g | ForEach-Object { $_.path })
      # (ticked: the linked file with its split parts, or a linked folder's files; with either, one image projector)
      $tick = @()
      if ($r.path -and ($paths -contains $r.path)) { $tick = @(Get-SplitSet $r.path $paths) }  # (@(): one path comes back as a string, and += then joined text
      elseif ($r.path) { $tick = @($paths | Where-Object { $_ -like "$($r.path)/*" -and $_ -notmatch "mmproj" }) }
      if ($tick) {
        $mp = @($paths | Where-Object { $_ -match "mmproj" })
        $pick = @($mp | Where-Object { $_ -match "(?i)f16" -and $_ -notmatch "(?i)bf16" }) + @($mp | Where-Object { $_ -match "(?i)bf16" }) + $mp | Select-Object -First 1
        if ($pick) { $tick += $pick }
      }
      foreach ($e in $g) {
        $have = [bool](Get-ChildItem (Join-Path $hub ("models--" + ($r.repo -replace "/", "--") + "\snapshots")) -Recurse -Filter (Split-Path $e.path -Leaf) -ErrorAction SilentlyContinue | Where-Object { $_.Length -eq [int64]$e.size })
        $i = $hfList.Items.Add(("{0}  ({1:N2} GB){2}" -f $e.path, ($e.size / 1e9), $(if ($have) { "  - already here" } else { "" })))
        if ($tick -contains $e.path) { $hfList.SetItemChecked($i, $true) }
        $st.hf += [pscustomobject]@{ path = $e.path; size = [double]$e.size; have = $have }
      }
      $hfInfo.Text = "$($g.Count) GGUF files in $($r.repo). Tick what to download (a split model: one part ticks all of them; the image projector, mmproj, for pictures)."
    })
  $link.Add_KeyDown({ param($s, $e) if ($e.KeyCode -eq "Enter") { $e.SuppressKeyPress = $true; $look.PerformClick() } })

  $go.Add_Click({
      $tok = @(); $dirs = @(); $total = 0.0; $disk = 0.0
      for ($i = 0; $i -lt $cat.Items.Count; $i++) {
        if (-not $cat.GetItemChecked($i)) { continue }
        $m = $ArcModels[$i]
        if (Test-ArcHave $m $hub) { continue }
        $tok += $m.key; $total += $m.gb * 1e9; $disk += $(if ($m.disk) { $m.disk } else { $m.gb })
        foreach ($rp in @($m.repo, $m.drafter.repo, $m.extra.repo)) { if ($rp) { $dirs += Join-Path $hub ("models--" + ($rp -replace "/", "--")) } }
      }
      $paths = @($st.hf | ForEach-Object { $_.path })
      $want = @()
      for ($i = 0; $i -lt $hfList.Items.Count; $i++) { if ($hfList.GetItemChecked($i)) { $want += @(Get-SplitSet $st.hf[$i].path $paths) } }
      foreach ($w in @($want | Sort-Object -Unique)) {
        $e = $st.hf | Where-Object { $_.path -eq $w } | Select-Object -First 1
        $tok += "hf:$($st.hfRepo)/$w"
        if (-not $e.have) { $total += $e.size; $disk += $e.size / 1e9 }
      }
      if ($want) { $dirs += Join-Path $hub ("models--" + ($st.hfRepo -replace "/", "--")) }
      if (-not $tok) { $info.Text = "Nothing to download: tick a model that is not here yet, or look up a link and tick its files."; return }
      if ($drive -and $drive.Free / 1GB -lt $disk + 5) {
        [void][System.Windows.Forms.MessageBox]::Show(("Not enough free disk space: {0:N0} GB needed, {1:N0} GB free on {2}:." -f ($disk + 5), ($drive.Free / 1GB), $drive.Name), "arcinferzero")
        return
      }
      $st.tokens = $tok; $st.done = @{}; $st.fails = @(); $st.dirs = @($dirs | Sort-Object -Unique); $st.total = $total
      $st.base = Bytes-Now
      $st.retries = 0; $st.retryAt = $null
      $bar.Value = 0
      Start-Run
    })
  $pause.Add_Click({
      if ($st.paused) { Start-Run; return }
      Stop-Run; $t.Stop(); Read-Log
      $st.retryAt = $null  # (also during the wait after a cut connection: no further try)
      $st.paused = $true
      $pause.Text = "Continue"
      $info.Text = "Paused: {0:N2} of {1:N2} GB. Continue goes on where it stopped (also after closing this window: Download again)." -f ([math]::Max(0, (Bytes-Now) - $st.base) / 1e9), ($st.total / 1e9)
    })
  # (a new limit while a download runs: stopped and started again with it, the .part files kept)
  $speed.Add_SelectedIndexChanged({ if ($st.proc -and -not $st.proc.HasExited) { Stop-Run; $t.Stop(); Read-Log; Start-Run } })
  $close.Add_Click({ $d.Close() })
  $d.Add_FormClosing({
      param($s, $e)
      if ($st.proc -and -not $st.proc.HasExited) {
        $r = [System.Windows.Forms.MessageBox]::Show("Stop the download? (Started again later, it goes on where it stopped.)", "arcinferzero", "YesNo")
        if ($r -ne "Yes") { $e.Cancel = $true; return }
        Stop-Run
      }
      $t.Stop()
    })
  [void]$d.ShowDialog($form)
}

# ---- in the More menu, first (the Get models button opens it too: launcher-firstrun.ps1; the welcome dialog stays
#      for the first run)
$dlItem = Add-Item $more "Download models..." { Show-Downloads }
$more.Items.Remove($dlItem); $more.Items.Insert(0, $dlItem)

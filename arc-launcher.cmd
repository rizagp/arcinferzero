@echo off
rem inferzero launcher window (scripts\arc-launcher.ps1): models, options, server on / off.
rem The PowerShell behind the window runs in a console without a window (conhost --headless): with Windows Terminal
rem as the default terminal, -WindowStyle Hidden still opened a terminal window next to the launcher (GitHub #5).
rem Windows without conhost --headless: the old way, the console hidden by the script (-HideConsole).
set "ps=powershell"
where pwsh >nul 2>nul && set "ps=pwsh"
if "%ps%"=="powershell" set "PSModulePath="
set "head="
if exist "%SystemRoot%\System32\conhost.exe" set "head=%SystemRoot%\System32\conhost.exe --headless"
start "" %head% %ps% -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -STA -File "%~dp0scripts\arc-launcher.ps1" -HideConsole

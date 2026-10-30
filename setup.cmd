@echo off
rem arcinferzero setup (see setup.ps1): setup, setup -Models 27b,4b, setup -Tests, setup -NoPi
rem (Windows PowerShell gets its own module path: one inherited from PowerShell 7 can hide cmdlets, GitHub #3)
set "PSModulePath="
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
set rc=%errorlevel%
rem (double-clicked: Explorer runs it as cmd /c "...setup.cmd": the window would close before the result is read)
echo %cmdcmdline% | find /i "/c" >nul && (echo. & pause)
exit /b %rc%

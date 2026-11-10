@echo off
rem zero-server from the inferzero folder (scripts\zero-server.ps1), with or without Pi: zero-server help
rem (setup offers to put this folder on PATH; Windows PowerShell gets its own module path, GitHub #3)
setlocal
set "PSModulePath="
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\zero-server.ps1" %*
exit /b %errorlevel%

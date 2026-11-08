@echo off
rem arc-server from the arcinferzero folder (scripts\arc-server.ps1), with or without Pi: arc-server help
rem (setup offers to put this folder on PATH; Windows PowerShell gets its own module path, GitHub #3)
setlocal
set "PSModulePath="
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\arc-server.ps1" %*
exit /b %errorlevel%

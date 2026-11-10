@echo off
rem The old name (before 2026-11-10): the command is zero-server now (zero-server.cmd). Kept so scripts still work.
call "%~dp0zero-server.cmd" %*
exit /b %errorlevel%

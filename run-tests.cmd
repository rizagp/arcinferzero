@echo off
rem arcinferzero benchmarks for testers (bench\run_tests.ps1): double-click, leave it running, then drag the zip
rem that appears on the Desktop into the GitHub issue.
rem   run-tests.cmd          everything (overnight)
rem   run-tests.cmd quick    a shorter round (about an hour)
rem   run-tests.cmd quick 27B   only the models with 27B in their file name
set "PSModulePath="
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bench\run_tests.ps1" %*
pause

#!/usr/bin/env sh
# inferzero on Linux: pick a model, download it, start the server (scripts/zero_launcher.py; --help for the options).
# Needs python3 (standard library only). The Windows launcher is zero-launcher.cmd.
# (No cd: a relative --model path stays relative to where you ran it. A symlink to this file, e.g. in ~/bin, is
#  followed to the package.)
self=$(readlink -f "$0" 2>/dev/null || printf '%s' "$0")
here=$(dirname "$self")
command -v python3 >/dev/null 2>&1 || { echo "zero-launcher: python3 not found (sudo apt install python3)" >&2; exit 1; }
[ -f "$here/scripts/zero_launcher.py" ] || { echo "zero-launcher: $here/scripts/zero_launcher.py not found (run it from the unpacked package)" >&2; exit 1; }
exec python3 -I "$here/scripts/zero_launcher.py" "$@"

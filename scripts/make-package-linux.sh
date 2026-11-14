#!/usr/bin/env bash
# The Linux package: dist/inferzero-<version>-linux-x64.tar.gz, the Linux side of scripts/make-package.ps1. Unpacked
# anywhere, ./zero-launcher.sh runs without a compiler: bin/ holds runz, serverz and mgpu_probe built by ./build.sh,
# bin/PREBUILT.txt says so. It holds every file git tracks (the sources too, so ./build.sh still builds), nothing
# untracked (no recordings, reports or models).
#   scripts/make-package-linux.sh                       build, then the archive
#   scripts/make-package-linux.sh --no-build            the programs as they are in bin/
#   scripts/make-package-linux.sh --release 2026.11.14.13   the release's number in the name (yyyy.MM.dd.NN)
# Built on Ubuntu 24.04 (glibc 2.39), it runs there and on newer systems. It needs the system's libstdc++ and GCC's
# OpenMP runtime (libgomp1: present on desktops, maybe not on a minimal install; ./zero-launcher.sh --check says) and
# Python 3 for the launcher. The GPU runtime is the user's: Intel's compute runtime and Level Zero loader.
set -euo pipefail
cd "$(dirname "$0")/.."
build=1; release=""
while [ $# -gt 0 ]; do
  case "$1" in
    --no-build) build=0 ;;
    --release) [ $# -ge 2 ] || { echo "--release needs a number (yyyy.MM.dd.NN)" >&2; exit 2; }; release="$2"; shift ;;
    *) echo "make-package-linux.sh: unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
if [ -n "$release" ] && ! [[ "$release" =~ ^[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{2}$ ]]; then
  echo "--release: yyyy.MM.dd.NN with two digits for NN, e.g. 2026.11.14.13 (got $release)" >&2; exit 2
fi
progs=(runz serverz mgpu_probe)
[ $build -eq 1 ] && ./build.sh "${progs[@]}"
commit=$(git rev-parse --short HEAD)
dirty=""; [ -n "$(git status --porcelain --untracked-files=no)" ] && dirty="-dirty"
[ -n "$dirty" ] && echo "warning: uncommitted changes go into the package (its version says -dirty): commit first for a release" >&2
version="${release:-$(date +%Y.%m.%d)}-$commit$dirty"
name="inferzero-$version-linux-x64"
stage="dist/$name"
rm -rf "$stage"; mkdir -p "$stage/bin"
# the tracked files, as they are in the working tree (a deleted, not yet committed one skipped)
git ls-files -z | while IFS= read -r -d '' f; do
  [ -f "$f" ] || continue
  mkdir -p "$stage/$(dirname "$f")"
  cp -p "$f" "$stage/$f"
done
for p in "${progs[@]}"; do
  [ -x "bin/$p" ] || { echo "bin/$p is missing: build it (or run without --no-build)" >&2; exit 1; }
  cp -p "bin/$p" "$stage/bin/"
done
chmod +x "$stage/build.sh" "$stage/zero-launcher.sh" "$stage/scripts/make-package-linux.sh"
printf 'inferzero %s (built %s, commit %s, %s)\nThe programs in this folder are built: ./zero-launcher.sh runs them (./build.sh builds them from the sources instead).\n' \
  "$version" "$(date '+%Y-%m-%d %H:%M')" "$commit" "$("${CXX:-g++}" --version | head -1)" > "$stage/bin/PREBUILT.txt"
# (owned by root in the archive, not the builder's uid: unpacked by root into /opt, the files must not belong to
#  whichever user has that uid there)
tar -C dist --owner=0 --group=0 --numeric-owner -czf "dist/$name.tar.gz" "$name"
# (older packages go, as make-package.ps1 does: the newest ARC_DIST_KEEP (default 3) folders and archives of this
#  script's own name form stay; anything else in dist/ is left alone)
keep="${ARC_DIST_KEEP:-3}"; [[ "$keep" =~ ^[0-9]+$ ]] || keep=3; [ "$keep" -ge 1 ] || keep=1
own='^inferzero-[0-9]{4}\.[0-9]{2}\.[0-9]{2}(\.[0-9]{2})?-[0-9a-f]{7,}(-dirty)?-linux-x64'
(cd dist && ls -1dt -- */ 2>/dev/null | sed 's#/$##' | grep -E "$own\$" | tail -n +$((keep + 1)) | xargs -r rm -rf --)
(cd dist && ls -1t -- *.tar.gz 2>/dev/null | grep -E "$own\.tar\.gz\$" | tail -n +$((keep + 1)) | xargs -r rm -f --)
echo "dist/$name.tar.gz ($(du -h "dist/$name.tar.gz" | cut -f1)): unpack it anywhere, then ./zero-launcher.sh"

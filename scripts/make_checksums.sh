#!/usr/bin/env bash
# Regenerate checksums.sha256 over every tracked file (LF basis).
#
# checksums.sha256 must never hash itself (self-reference). Run after ANY
# content change to committed files; CI verifies with `sha256sum -c`.
set -euo pipefail
cd "$(dirname "$0")/.."

git ls-files -z | grep -zv '^checksums.sha256$' | LC_ALL=C sort -z \
  | xargs -0 sha256sum > checksums.sha256

echo "wrote $(wc -l < checksums.sha256) entries; verifying..."
sha256sum -c --quiet checksums.sha256 && echo "ALL OK"

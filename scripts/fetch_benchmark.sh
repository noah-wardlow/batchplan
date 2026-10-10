#!/usr/bin/env bash
# Downloads the MotionBenchMaker and MπNets Panda problem sets, which robometrics packages as plain
# YAML (MIT, https://github.com/fishbotics/robometrics), and verifies their checksums.
#
#   scripts/fetch_benchmark.sh [dir=data/robometrics]
set -euo pipefail
COMMIT=81e3d1d605de84100d8ab880b43096aba221a48b
DIR="${1:-data/robometrics}"
mkdir -p "$DIR"

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

while read -r sum file; do
  path="$DIR/$file"
  if [[ -f "$path" && "$(sha256 "$path")" == "$sum" ]]; then
    echo "$path is up to date"
    continue
  fi
  curl -fsSL -o "$path.part" "https://raw.githubusercontent.com/fishbotics/robometrics/$COMMIT/robometrics/content/dataset/$file"
  got="$(sha256 "$path.part")"
  if [[ "$got" != "$sum" ]]; then
    rm -f "$path.part"
    echo "checksum mismatch for $file: expected $sum, got $got" >&2
    exit 1
  fi
  mv "$path.part" "$path"
  echo "fetched $path"
done <<'SUMS'
5165ad4fb55f93c63dbbdda6f25a14f3ceb24ee3305e27ad317f38f1e0d6ed0d mb_set.yaml
9189186d83e51600a2c768aa7657933aa052150501852771994757052b797bbf mpinets_set.yaml
SUMS

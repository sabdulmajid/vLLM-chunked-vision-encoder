#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="${PYTHON_DEV_CACHE_DIR:-$REPO_ROOT/.deps/python-dev}"
EXTRACTED_DIR="$TARGET_DIR/extracted"
HEADER_PATH="$EXTRACTED_DIR/usr/include/python3.12/Python.h"

if [[ -f /usr/include/python3.12/Python.h ]]; then
  exit 0
fi

if [[ -f "$HEADER_PATH" ]]; then
  exit 0
fi

download_package() {
  local package_name="$1"
  local version=""

  while read -r version; do
    [[ -n "$version" ]] || continue
    if apt download "${package_name}=${version}"; then
      return 0
    fi
  done < <(apt-cache madison "$package_name" | awk '{print $3}' | uniq)

  echo "Unable to download ${package_name} from configured apt sources." >&2
  return 1
}

mkdir -p "$TARGET_DIR"
pushd "$TARGET_DIR" >/dev/null
download_package "libpython3.12-dev"
download_package "python3.12-dev"
rm -rf "$EXTRACTED_DIR"
mkdir -p "$EXTRACTED_DIR"
for deb in ./*.deb; do
  dpkg-deb -x "$deb" "$EXTRACTED_DIR"
done
popd >/dev/null

if [[ ! -f "$HEADER_PATH" ]]; then
  echo "Python 3.12 development headers were not extracted as expected." >&2
  exit 1
fi

#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 || -z $1 ]]; then
  echo "Usage: bash $0 <models-directory>" >&2
  exit 2
fi

root_dir=$(unset CDPATH; cd -- "$(dirname -- "$0")" && pwd)
destination=$1
base_url=$(<"$root_dir/models/source-url.txt")
checksum=(sha256sum)
if ! command -v sha256sum >/dev/null 2>&1; then
  checksum=(shasum -a 256)
fi

matches_checksum() {
  local actual
  actual=$("${checksum[@]}" <"$2") || return 1
  [[ ${actual%% *} == "$1" ]]
}

temporary=
trap 'if [[ -n $temporary ]]; then rm -f -- "$temporary"; fi' EXIT
while read -r expected name; do
  if [[ ! $expected =~ ^[a-f0-9]{64}$ || ! $name =~ ^v[12]/[a-z_]+\.pt$ ]]; then
    echo "Invalid entry in the model manifest" >&2
    exit 1
  fi
  target=$destination/$name
  if [[ -f $target && ! -L $target ]] && matches_checksum "$expected" "$target"; then
    printf '%s: OK (already downloaded)\n' "$name"
    continue
  fi
  mkdir -p -- "$(dirname -- "$target")"
  temporary=$(mktemp "$target.part.XXXXXX")
  curl --disable --fail --silent --show-error --location \
    --proto '=https' --proto-redir '=https' --retry 2 --connect-timeout 20 \
    --max-time 600 --output "$temporary" "$base_url/$name"
  if ! matches_checksum "$expected" "$temporary"; then
    printf '%s: SHA-256 mismatch\n' "$name" >&2
    exit 1
  fi
  mv -f -- "$temporary" "$target"
  temporary=
  printf '%s: OK\n' "$name"
done <"$root_dir/models/manifest.sha256"

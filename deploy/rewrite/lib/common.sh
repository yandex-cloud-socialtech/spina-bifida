#!/usr/bin/env bash

REWRITE_SOURCE_DIR=${REWRITE_SOURCE_DIR:-$(unset CDPATH; cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
SPINA_INSTALL_ROOT=${SPINA_INSTALL_ROOT:-/opt/spina-bifida}
SPINA_STATE_DIR=${SPINA_STATE_DIR:-/var/lib/spina-bifida/install-state}
SPINA_BACKUP_ROOT=${SPINA_BACKUP_ROOT:-/var/backups/spina-bifida}
APP_CONTAINER_UID=${APP_CONTAINER_UID:-10001}
APP_CONTAINER_GID=${APP_CONTAINER_GID:-10001}

log() {
  printf '%s\n' "$*" >&2
}

die() {
  log "ERROR: $*"
  return 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

run_root() {
  if [[ ${SPINA_TEST_MODE:-0} == 1 || $(id -u) == 0 ]]; then
    "$@"
  elif [[ ${SPINA_SUDO+x} && -z ${SPINA_SUDO} ]]; then
    "$@"
  else
    "${SPINA_SUDO:-sudo}" "$@"
  fi
}

validate_platform() {
  local os_file=${OS_RELEASE_FILE:-/etc/os-release}
  local arch=${SPINA_ARCH:-$(uname -m)}
  [[ -r $os_file ]] || die "OS metadata is unavailable: $os_file" || return

  local ID="" VERSION_ID=""
  # shellcheck disable=SC1090
  source "$os_file"
  [[ $ID == ubuntu ]] || die "Unsupported OS: ${ID:-unknown}; use Ubuntu 22.04 or 24.04" || return
  case "$VERSION_ID" in
    22.04 | 24.04) ;;
    *) die "Unsupported Ubuntu version: ${VERSION_ID:-unknown}; use 22.04 or 24.04" || return ;;
  esac
  case "$arch" in
    x86_64 | amd64) ;;
    *) die "Unsupported architecture: $arch; use x86_64" || return ;;
  esac
}

ensure_private_directory() {
  if mkdir -p "$1" 2>/dev/null && chmod 0700 "$1" 2>/dev/null; then
    return
  fi
  run_root install -d -m 0700 "$1"
}

state_write() {
  local name=$1 value=$2 temporary
  ensure_private_directory "$SPINA_STATE_DIR"
  temporary="$SPINA_STATE_DIR/.${name}.tmp"
  printf '%s\n' "$value" >"$temporary"
  chmod 0600 "$temporary"
  mv "$temporary" "$SPINA_STATE_DIR/$name"
}

state_read() {
  local path="$SPINA_STATE_DIR/$1"
  [[ -r $path ]] && sed -n '1p' "$path"
}

sha256_file() {
  if command_exists sha256sum; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify_release_bundle() {
  local expected_version=${1:-} checksums="$REWRITE_SOURCE_DIR/release-checksums.sha256"
  local manifest="$REWRITE_SOURCE_DIR/release-manifest.yaml" actual_version
  [[ -f $checksums ]] || return 0
  if command_exists sha256sum; then
    (cd "$REWRITE_SOURCE_DIR" && sha256sum -c release-checksums.sha256)
  else
    (cd "$REWRITE_SOURCE_DIR" && shasum -a 256 -c release-checksums.sha256)
  fi
  actual_version=$(awk '/^application_version:/ {print $2; exit}' "$manifest")
  if [[ -n $expected_version && $actual_version != "$expected_version" ]]; then
    die "Release manifest version $actual_version does not match requested $expected_version"
  fi
}

release_image() {
  local key=$1 manifest="$REWRITE_SOURCE_DIR/release-manifest.yaml"
  awk -v key="$key" '$1 == key ":" {print $2; exit}' "$manifest"
}

random_secret() {
  umask 077
  openssl rand -base64 48 | tr -d '\n'
}

env_value() {
  local key=$1 file=${2:-$SPINA_INSTALL_ROOT/.env}
  [[ -r $file ]] || return 1
  sed -n "s/^${key}=//p" "$file" | tail -1
}

set_env_value() {
  local key=$1 value=$2 file=${3:-$SPINA_INSTALL_ROOT/.env} temporary
  [[ $value != *$'\n'* && $value != *$'\r'* ]] || die "Invalid newline in $key" || return
  temporary="${file}.tmp"
  awk -v key="$key" -v value="$value" '
    BEGIN { found = 0 }
    index($0, key "=") == 1 { print key "=" value; found = 1; next }
    { print }
    END { if (!found) print key "=" value }
  ' "$file" >"$temporary"
  chmod 0600 "$temporary"
  mv "$temporary" "$file"
}

compose() {
  (
    cd "$SPINA_INSTALL_ROOT" || exit
    if [[ $(state_read accelerator 2>/dev/null || true) == gpu ]]; then
      docker_command compose --env-file .env -f compose.yaml -f compose.gpu.yaml "$@"
    else
      docker_command compose --env-file .env -f compose.yaml "$@"
    fi
  )
}

configured_parallel_jobs() {
  local config="$SPINA_INSTALL_ROOT/config/app.yaml" value
  value=$(awk '/^[[:space:]]*max_parallel_jobs:/ {print $2; exit}' "$config" 2>/dev/null || true)
  value=${value:-1}
  case "$value" in
    1 | 2 | 3 | 4) printf '%s\n' "$value" ;;
    *) die "inference.max_parallel_jobs must be an integer from 1 through 4" ;;
  esac
}

start_compose_stack() {
  local parallel
  parallel=$(configured_parallel_jobs) || return
  compose up -d --scale "worker=$parallel"
}

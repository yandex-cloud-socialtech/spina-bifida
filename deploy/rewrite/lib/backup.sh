#!/usr/bin/env bash

backup_timestamp() {
  date -u '+%Y%m%dT%H%M%SZ'
}

write_inner_checksums() {
  local root=$1 item
  (
    cd "$root" || exit
    : >SHA256SUMS
    while IFS= read -r item; do
      printf '%s  %s\n' "$(sha256_file "$item")" "${item#./}" >>SHA256SUMS
    done < <(find . -type f ! -name SHA256SUMS | LC_ALL=C sort)
  )
}

verify_inner_checksums() {
  local root=$1 expected relative actual
  while read -r expected relative; do
    [[ -n ${relative:-} && -f "$root/$relative" ]] || die "Backup checksum entry is missing: ${relative:-unknown}" || return
    actual=$(sha256_file "$root/$relative")
    [[ $actual == "$expected" ]] || die "Backup checksum mismatch: $relative" || return
  done <"$root/SHA256SUMS"
}

backup_test_data() {
  local staging=$1 source="$SPINA_INSTALL_ROOT/data/app.sqlite3"
  mkdir -p "$staging/feedback"
  if [[ -f $source ]]; then
    sqlite3 "$source" ".backup '$staging/app.sqlite3'"
  else
    sqlite3 "$staging/app.sqlite3" 'PRAGMA user_version;'
  fi
  cp -R "$SPINA_INSTALL_ROOT/data/feedback/." "$staging/feedback/" 2>/dev/null || true
}

restore_helper_compose() {
  local extracted=$1
  shift
  (
    cd "$extracted/runtime" || exit
    APP_CONFIG_PATH="$extracted/app.yaml" APP_ENV_FILE="$extracted/.env" \
      docker_command compose --env-file "$extracted/.env" \
      -f "$extracted/runtime/compose.yaml" "$@"
  )
}

backup_live_data() {
  local staging=$1 restore_helper=${2:-}
  compose stop worker >/dev/null
  BACKUP_WORKER_STOPPED=1
  mkdir -p "$staging/feedback"
  if [[ -n $restore_helper ]]; then
    restore_helper_compose "$restore_helper" run --rm --no-deps --user 0 \
      --volume "$staging:/snapshot" api python -c \
      "import shutil,sqlite3; source=sqlite3.connect('/data/db/app.sqlite3'); target=sqlite3.connect('/snapshot/app.sqlite3'); source.backup(target); target.close(); source.close(); shutil.copytree('/data/feedback', '/snapshot/feedback', dirs_exist_ok=True)"
  else
    compose run --rm --no-deps --user 0 --volume "$staging:/snapshot" api python -c \
      "import shutil,sqlite3; source=sqlite3.connect('/data/db/app.sqlite3'); target=sqlite3.connect('/snapshot/app.sqlite3'); source.backup(target); target.close(); source.close(); shutil.copytree('/data/feedback', '/snapshot/feedback', dirs_exist_ok=True)"
  fi
}

backup_runtime_bundle() {
  local staging=$1 file
  mkdir -p "$staging/runtime/config" "$staging/runtime/models"
  for file in install.sh compose.yaml compose.gpu.yaml .env.example release-manifest.yaml; do
    [[ -f $SPINA_INSTALL_ROOT/$file ]] || die "Installed runtime file is missing: $file" || return
    cp "$SPINA_INSTALL_ROOT/$file" "$staging/runtime/$file"
  done
  [[ -d $SPINA_INSTALL_ROOT/lib && -d $SPINA_INSTALL_ROOT/traefik ]] \
    || die "Installed runtime directories are incomplete" || return
  cp -R "$SPINA_INSTALL_ROOT/lib" "$SPINA_INSTALL_ROOT/traefik" "$staging/runtime/"
  cp "$SPINA_INSTALL_ROOT/config/app.example.yaml" "$staging/runtime/config/app.example.yaml"
  cp "$SPINA_INSTALL_ROOT/models/manifest.sha256" "$staging/runtime/models/manifest.sha256"
}

backup_create() {
  local prefix=${1:-backup} timestamp staging archive manifest_version retention stale
  local restore_helper=${2:-} resume_workers=${3:-1} default_model_version schema_version
  timestamp=$(backup_timestamp)
  ensure_private_directory "$SPINA_BACKUP_ROOT"
  staging=$(mktemp -d "${TMPDIR:-/tmp}/spina-backup.XXXXXX")
  chmod 0700 "$staging"
  BACKUP_WORKER_STOPPED=0
  trap 'if [[ ${BACKUP_WORKER_STOPPED:-0} == 1 && ${resume_workers:-1} == 1 ]]; then compose start worker >/dev/null 2>&1 || true; fi; rm -rf "$staging"' RETURN

  if [[ ${SPINA_TEST_MODE:-0} == 1 ]]; then
    backup_test_data "$staging"
  else
    backup_live_data "$staging" "$restore_helper"
  fi
  cp "$SPINA_INSTALL_ROOT/config/app.yaml" "$staging/app.yaml"
  cp "$SPINA_INSTALL_ROOT/.env" "$staging/.env"
  backup_runtime_bundle "$staging"
  chmod 0600 "$staging/.env"
  manifest_version=$(env_value APP_VERSION 2>/dev/null || printf unknown)
  default_model_version=$(awk '
    /^[[:space:]]*default_model_version[[:space:]]*:/ {
      value = $0
      sub(/^[^:]*:[[:space:]]*/, "", value)
      sub(/[[:space:]]+#.*$/, "", value)
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      first = substr(value, 1, 1)
      last = substr(value, length(value), 1)
      if ((first == "\"" && last == "\"") || (first == "\047" && last == "\047")) {
        value = substr(value, 2, length(value) - 2)
      }
      print value
      exit
    }
  ' "$staging/app.yaml")
  default_model_version=${default_model_version:-v2}
  schema_version=$(python3 - "$staging/app.sqlite3" <<'PY'
import sqlite3
import sys

try:
    with sqlite3.connect(sys.argv[1]) as connection:
        value = connection.execute("SELECT version_num FROM alembic_version").fetchone()
except sqlite3.Error:
    value = None
print(value[0] if value else "unknown")
PY
  )
  printf 'format=2\ncreated_at=%s\napplication_version=%s\nmodel_versions=v1,v2\ndefault_model_version=%s\nschema_version=%s\n' \
    "$timestamp" "$manifest_version" "$default_model_version" "$schema_version" \
    >"$staging/MANIFEST.txt"
  write_inner_checksums "$staging"

  archive="$SPINA_BACKUP_ROOT/${prefix}-${timestamp}.tar.gz"
  tar -C "$staging" -czf "$archive" .
  chmod 0600 "$archive"
  printf '%s  %s\n' "$(sha256_file "$archive")" "$(basename "$archive")" >"${archive}.sha256"
  chmod 0600 "${archive}.sha256"

  if [[ $BACKUP_WORKER_STOPPED == 1 ]]; then
    if [[ $resume_workers == 1 ]]; then
      compose start worker >/dev/null
    fi
    BACKUP_WORKER_STOPPED=0
  fi
  state_write last-backup "$archive"
  retention=$(awk '/^[[:space:]]*retention_count:/ {print $2; exit}' "$SPINA_INSTALL_ROOT/config/app.yaml")
  retention=${retention:-7}
  while IFS= read -r stale; do
    [[ -n $stale ]] || continue
    rm -f -- "$stale" "${stale}.sha256"
  done < <(find "$SPINA_BACKUP_ROOT" -maxdepth 1 -type f -name 'backup-*.tar.gz' \
    -print | LC_ALL=C sort -r | tail -n "+$((retention + 1))")
  printf '%s\n' "$archive"
}

verify_archive_checksum() {
  local archive=$1 expected actual
  local checksum_file="${archive}.sha256"
  [[ -f $archive && -f $checksum_file ]] || die "Backup archive or checksum file is missing" || return
  expected=$(awk 'NR == 1 {print $1}' "$checksum_file")
  actual=$(sha256_file "$archive")
  [[ $actual == "$expected" ]] || die "Backup archive checksum mismatch" || return
}

restore_test_data() {
  local extracted=$1
  ensure_private_directory "$SPINA_INSTALL_ROOT/data/feedback"
  cp "$extracted/app.sqlite3" "$SPINA_INSTALL_ROOT/data/app.sqlite3"
  rm -rf "$SPINA_INSTALL_ROOT/data/feedback"
  mkdir -m 0700 "$SPINA_INSTALL_ROOT/data/feedback"
  cp -R "$extracted/feedback/." "$SPINA_INSTALL_ROOT/data/feedback/" 2>/dev/null || true
}

restore_live_data() {
  local extracted=$1
  compose stop worker api >/dev/null
  compose run --rm --no-deps --user 0 --volume "$extracted:/restore:ro" api python -c \
    "from pathlib import Path; import os,shutil; db=Path('/data/db/app.sqlite3'); feedback=Path('/data/feedback'); db.unlink(missing_ok=True); feedback.mkdir(mode=0o700, parents=True, exist_ok=True); [(shutil.rmtree(item) if item.is_dir() else item.unlink()) for item in feedback.iterdir()]; shutil.copy2('/restore/app.sqlite3', db); shutil.copytree('/restore/feedback', feedback, dirs_exist_ok=True); paths=[db,feedback,*feedback.rglob('*')]; [(os.chown(path, $APP_CONTAINER_UID, $APP_CONTAINER_GID), os.chmod(path, 0o700 if path.is_dir() else 0o600)) for path in paths]"
}

restore_runtime_bundle() {
  local extracted=$1
  [[ -d $extracted/runtime/lib && -d $extracted/runtime/traefik ]] \
    || die "Backup runtime bundle is incomplete" || return
  cp "$extracted/runtime/install.sh" "$extracted/runtime/compose.yaml" \
    "$extracted/runtime/compose.gpu.yaml" "$extracted/runtime/.env.example" \
    "$extracted/runtime/release-manifest.yaml" "$SPINA_INSTALL_ROOT/"
  cp -R "$extracted/runtime/lib" "$extracted/runtime/traefik" "$SPINA_INSTALL_ROOT/"
  cp "$extracted/runtime/config/app.example.yaml" "$SPINA_INSTALL_ROOT/config/app.example.yaml"
  cp "$extracted/runtime/models/manifest.sha256" "$SPINA_INSTALL_ROOT/models/manifest.sha256"
  chmod 0755 "$SPINA_INSTALL_ROOT/install.sh"
}

extract_archive_safely() {
  local archive=$1 destination=$2
  python3 - "$archive" "$destination" <<'PY'
import sys
import tarfile
from pathlib import Path

archive = Path(sys.argv[1])
destination = Path(sys.argv[2]).resolve()
with tarfile.open(archive, "r:gz") as bundle:
    for member in bundle.getmembers():
        target = (destination / member.name).resolve()
        if target != destination and destination not in target.parents:
            raise SystemExit(f"unsafe backup path: {member.name}")
        if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported backup entry: {member.name}")
    bundle.extractall(destination)
PY
}

restore_archive() {
  local archive=$1 extracted emergency
  verify_archive_checksum "$archive"
  extracted=$(mktemp -d "${TMPDIR:-/tmp}/spina-restore.XXXXXX")
  chmod 0700 "$extracted"
  trap 'rm -rf "$extracted"' RETURN
  extract_archive_safely "$archive" "$extracted"
  verify_inner_checksums "$extracted"

  emergency=$(backup_create pre-restore "$extracted" 0)
  log "Emergency snapshot: $emergency"

  restore_runtime_bundle "$extracted"
  cp "$extracted/app.yaml" "$SPINA_INSTALL_ROOT/config/app.yaml"
  cp "$extracted/.env" "$SPINA_INSTALL_ROOT/.env"
  chmod 0600 "$SPINA_INSTALL_ROOT/.env"
  secure_app_config
  if [[ ${SPINA_TEST_MODE:-0} == 1 ]]; then
    restore_test_data "$extracted"
  else
    restore_live_data "$extracted"
  fi
  if [[ ${SPINA_TEST_MODE:-0} != 1 ]]; then
    start_application
  fi
  log "Restore completed from $archive"
}

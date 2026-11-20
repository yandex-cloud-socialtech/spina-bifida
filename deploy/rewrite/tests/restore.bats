#!/usr/bin/env bats

setup() {
  export REWRITE_ROOT="$BATS_TEST_DIRNAME/.."
  export TEST_ROOT="$BATS_TEST_TMPDIR/root"
  mkdir -p "$TEST_ROOT/install/data/feedback" "$TEST_ROOT/install/config" \
    "$TEST_ROOT/install/models" "$TEST_ROOT/install/lib" "$TEST_ROOT/install/traefik" \
    "$TEST_ROOT/backups"
  printf 'APP_SESSION_SECRET=abcdefghijklmnopqrstuvwxyz1234567890\n' >"$TEST_ROOT/install/.env"
  printf 'access:\n  mode: ip\n' >"$TEST_ROOT/install/config/app.yaml"
  printf 'example\n' >"$TEST_ROOT/install/config/app.example.yaml"
  printf 'manifest\n' >"$TEST_ROOT/install/models/manifest.sha256"
  printf '#!/usr/bin/env bash\n' >"$TEST_ROOT/install/install.sh"
  chmod +x "$TEST_ROOT/install/install.sh"
  printf 'services: old\n' >"$TEST_ROOT/install/compose.yaml"
  printf 'services: old\n' >"$TEST_ROOT/install/compose.gpu.yaml"
  printf 'example\n' >"$TEST_ROOT/install/.env.example"
  printf 'application_version: old\n' >"$TEST_ROOT/install/release-manifest.yaml"
  printf 'old-lib\n' >"$TEST_ROOT/install/lib/common.sh"
  printf 'old-route\n' >"$TEST_ROOT/install/traefik/http.yaml"
  sqlite3 "$TEST_ROOT/install/data/app.sqlite3" 'create table sample(id integer); insert into sample values (1);'
  env SPINA_TEST_MODE=1 SPINA_INSTALL_ROOT="$TEST_ROOT/install" \
    SPINA_BACKUP_ROOT="$TEST_ROOT/backups" SPINA_STATE_DIR="$TEST_ROOT/state" \
    "$REWRITE_ROOT/install.sh" backup >/dev/null
  export ARCHIVE
  ARCHIVE=$(find "$TEST_ROOT/backups" -name '*.tar.gz' -type f | head -1)
}

@test "restore rejects an archive checksum mismatch" {
  printf 'tampered' >>"$ARCHIVE"
  run env SPINA_TEST_MODE=1 SPINA_INSTALL_ROOT="$TEST_ROOT/install" \
    SPINA_BACKUP_ROOT="$TEST_ROOT/backups" SPINA_STATE_DIR="$TEST_ROOT/state" \
    "$REWRITE_ROOT/install.sh" restore "$ARCHIVE"
  [ "$status" -ne 0 ]
  [[ "$output" == *"checksum"* ]]
}

@test "restore creates an emergency snapshot and restores selected data" {
  sqlite3 "$TEST_ROOT/install/data/app.sqlite3" 'delete from sample; insert into sample values (9);'
  printf 'services: new\n' >"$TEST_ROOT/install/compose.yaml"
  printf 'APP_SESSION_SECRET=abcdefghijklmnopqrstuvwxyz1234567890\nAPP_VERSION=9.9.9\nBACKEND_CPU_IMAGE=missing.example/new@sha256:deadbeef\n' >"$TEST_ROOT/install/.env"
  run env SPINA_TEST_MODE=1 SPINA_INSTALL_ROOT="$TEST_ROOT/install" \
    SPINA_BACKUP_ROOT="$TEST_ROOT/backups" SPINA_STATE_DIR="$TEST_ROOT/state" \
    "$REWRITE_ROOT/install.sh" restore "$ARCHIVE"
  [ "$status" -eq 0 ]
  [ "$(sqlite3 "$TEST_ROOT/install/data/app.sqlite3" 'select id from sample;')" = 1 ]
  grep -q 'services: old' "$TEST_ROOT/install/compose.yaml"
  emergency=$(find "$TEST_ROOT/backups" -name 'pre-restore-*.tar.gz' -type f | head -1)
  [ -n "$emergency" ]
  tar -xOzf "$emergency" ./runtime/compose.yaml | grep -q 'services: new'
  tar -xOzf "$emergency" ./.env | grep -q 'missing.example/new'
  tar -xOzf "$emergency" ./app.sqlite3 >"$BATS_TEST_TMPDIR/emergency.sqlite3"
  [ "$(sqlite3 "$BATS_TEST_TMPDIR/emergency.sqlite3" 'select id from sample;')" = 9 ]
}

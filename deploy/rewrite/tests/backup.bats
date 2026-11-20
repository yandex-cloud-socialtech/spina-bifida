#!/usr/bin/env bats

setup() {
  export REWRITE_ROOT="$BATS_TEST_DIRNAME/.."
  export TEST_ROOT="$BATS_TEST_TMPDIR/root"
  mkdir -p "$TEST_ROOT/install/data/feedback" "$TEST_ROOT/install/config" \
    "$TEST_ROOT/install/models" "$TEST_ROOT/install/lib" "$TEST_ROOT/install/traefik" \
    "$TEST_ROOT/backups"
  printf 'APP_SESSION_SECRET=test-secret-not-printed\n' >"$TEST_ROOT/install/.env"
  printf 'APP_VERSION=1.2.3\n' >>"$TEST_ROOT/install/.env"
  printf 'access:\n  mode: ip\ninference:\n  default_model_version: "v2"   # selected\n' \
    >"$TEST_ROOT/install/config/app.yaml"
  printf 'example\n' >"$TEST_ROOT/install/config/app.example.yaml"
  printf 'manifest\n' >"$TEST_ROOT/install/models/manifest.sha256"
  printf '#!/usr/bin/env bash\n' >"$TEST_ROOT/install/install.sh"
  chmod +x "$TEST_ROOT/install/install.sh"
  printf 'services: {}\n' >"$TEST_ROOT/install/compose.yaml"
  printf 'services: {}\n' >"$TEST_ROOT/install/compose.gpu.yaml"
  printf 'example\n' >"$TEST_ROOT/install/.env.example"
  printf 'application_version: 1.2.3\n' >"$TEST_ROOT/install/release-manifest.yaml"
  printf 'old-lib\n' >"$TEST_ROOT/install/lib/common.sh"
  printf 'old-route\n' >"$TEST_ROOT/install/traefik/http.yaml"
  printf 'feedback' >"$TEST_ROOT/install/data/feedback/record.json"
  sqlite3 "$TEST_ROOT/install/data/app.sqlite3" 'create table sample(id integer); insert into sample values (1);'
}

@test "backup includes consistent database configuration feedback and checksums" {
  run env SPINA_TEST_MODE=1 SPINA_INSTALL_ROOT="$TEST_ROOT/install" \
    SPINA_BACKUP_ROOT="$TEST_ROOT/backups" SPINA_STATE_DIR="$TEST_ROOT/state" \
    "$REWRITE_ROOT/install.sh" backup
  [ "$status" -eq 0 ]
  archive=$(find "$TEST_ROOT/backups" -name '*.tar.gz' -type f | head -1)
  [ -n "$archive" ]
  [ "$(stat -c '%a' "$archive" 2>/dev/null || stat -f '%Lp' "$archive")" = 600 ]
  tar -tzf "$archive" | grep -q 'app.sqlite3'
  tar -tzf "$archive" | grep -q 'app.yaml'
  tar -tzf "$archive" | grep -q '.env'
  tar -tzf "$archive" | grep -q 'feedback/record.json'
  tar -tzf "$archive" | grep -q 'MANIFEST.txt'
  tar -tzf "$archive" | grep -q 'SHA256SUMS'
  tar -tzf "$archive" | grep -q 'runtime/compose.yaml'
  tar -xOzf "$archive" ./MANIFEST.txt | grep -q 'application_version=1.2.3'
  tar -xOzf "$archive" ./MANIFEST.txt | grep -q '^model_versions=v1,v2$'
  tar -xOzf "$archive" ./MANIFEST.txt | grep -q '^default_model_version=v2$'
  ! tar -xOzf "$archive" ./MANIFEST.txt | grep -q '^model_version='
  tar -xOzf "$archive" ./MANIFEST.txt | grep -q 'schema_version=unknown'
  [[ "$output" != *"test-secret-not-printed"* ]]
}

@test "backup canonicalizes every supported YAML scalar form" {
  for scalar in 'v2' "'v2'" '"v2"' 'v2   # selected' \
    "'v2'   # selected" '"v2"   # selected'; do
    printf 'access:\n  mode: ip\ninference:\n  default_model_version: %s   \n' "$scalar" \
      >"$TEST_ROOT/install/config/app.yaml"
    run env SPINA_TEST_MODE=1 SPINA_INSTALL_ROOT="$TEST_ROOT/install" \
      SPINA_BACKUP_ROOT="$TEST_ROOT/backups" SPINA_STATE_DIR="$TEST_ROOT/state" \
      "$REWRITE_ROOT/install.sh" backup
    [ "$status" -eq 0 ]
    archive=$(find "$TEST_ROOT/backups" -name '*.tar.gz' -type f | head -1)
    [ -n "$archive" ]
    tar -xOzf "$archive" ./MANIFEST.txt | grep -q '^default_model_version=v2$'
  done
}

@test "backup records v2 when configuration omits the default model" {
  printf 'access:\n  mode: ip\ninference:\n  device: cpu\n' \
    >"$TEST_ROOT/install/config/app.yaml"

  run env SPINA_TEST_MODE=1 SPINA_INSTALL_ROOT="$TEST_ROOT/install" \
    SPINA_BACKUP_ROOT="$TEST_ROOT/backups" SPINA_STATE_DIR="$TEST_ROOT/state" \
    "$REWRITE_ROOT/install.sh" backup

  [ "$status" -eq 0 ]
  archive=$(find "$TEST_ROOT/backups" -name '*.tar.gz' -type f | head -1)
  [ -n "$archive" ]
  tar -xOzf "$archive" ./MANIFEST.txt | grep -q '^default_model_version=v2$'
}

#!/usr/bin/env bats

@test "status help is available from a release bundle" {
  run "$BATS_TEST_DIRNAME/../install.sh" status --help
  [ "$status" -eq 0 ]
  [[ "$output" == *"application/model/schema"* ]]
}

@test "test-mode status omits secrets" {
  root="$BATS_TEST_TMPDIR/root"
  mkdir -p "$root/config" "$root/data"
  printf 'APP_VERSION=0.1.0\nAPP_SESSION_SECRET=never-show-this\n' >"$root/.env"
  printf 'inference:\n  device: cpu\n' >"$root/config/app.yaml"
  run env SPINA_TEST_MODE=1 SPINA_INSTALL_ROOT="$root" SPINA_BACKUP_ROOT="$root/backups" \
    "$BATS_TEST_DIRNAME/../install.sh" status
  [ "$status" -eq 0 ]
  [[ "$output" == *"application_version=0.1.0"* ]]
  [[ "$output" == *"model_versions=v1,v2"* ]]
  [[ "$output" == *"default_model_version=v2"* ]]
  [[ "$output" != "model_version="* ]]
  [[ "$output" != *$'\nmodel_version='* ]]
  [[ "$output" != *"never-show-this"* ]]
}

@test "status canonicalizes every supported YAML scalar form" {
  root="$BATS_TEST_TMPDIR/root"
  mkdir -p "$root/config"
  for scalar in 'v2' "'v2'" '"v2"' 'v2   # selected' \
    "'v2'   # selected" '"v2"   # selected'; do
    printf 'inference:\n  default_model_version: %s   \n' "$scalar" >"$root/config/app.yaml"
    run env SPINA_TEST_MODE=1 SPINA_INSTALL_ROOT="$root" \
      SPINA_BACKUP_ROOT="$root/backups" "$BATS_TEST_DIRNAME/../install.sh" status
    [ "$status" -eq 0 ]
    [[ "$output" == *"default_model_version=v2"* ]]
    [[ "$output" != *"default_model_version='v2'"* ]]
    [[ "$output" != *'default_model_version="v2"'* ]]
    [[ "$output" != *"# selected"* ]]
  done
}

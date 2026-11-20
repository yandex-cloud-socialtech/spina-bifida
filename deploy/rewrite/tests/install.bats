#!/usr/bin/env bats

setup() {
  export REWRITE_ROOT="$BATS_TEST_DIRNAME/.."
  export TEST_ROOT="$BATS_TEST_TMPDIR/root"
  mkdir -p "$TEST_ROOT"
}

@test "requires an explicit subcommand" {
  run "$REWRITE_ROOT/install.sh"
  [ "$status" -ne 0 ]
  [[ "$output" == *"install|update|backup|restore|status"* ]]
}

@test "accepts Ubuntu 22.04 and 24.04 on x86_64" {
  for fixture in os-release-22.04 os-release-24.04; do
    run env OS_RELEASE_FILE="$BATS_TEST_DIRNAME/fixtures/$fixture" SPINA_ARCH=x86_64 \
      bash -c 'source "$1/lib/common.sh"; validate_platform' _ "$REWRITE_ROOT"
    [ "$status" -eq 0 ]
  done
}

@test "rejects unsupported platform before mutation" {
  printf 'ID=debian\nVERSION_ID=12\n' >"$TEST_ROOT/os-release"
  run env OS_RELEASE_FILE="$TEST_ROOT/os-release" SPINA_ARCH=x86_64 \
    bash -c 'source "$1/lib/common.sh"; validate_platform' _ "$REWRITE_ROOT"
  [ "$status" -ne 0 ]
  [ ! -e "$TEST_ROOT/mutated" ]
}

@test "test-mode install is idempotent and never prints its secret" {
  common=(
    env SPINA_TEST_MODE=1 SPINA_INSTALL_ROOT="$TEST_ROOT/app"
    SPINA_STATE_DIR="$TEST_ROOT/state" SPINA_ETC_ROOT="$TEST_ROOT/etc"
    OS_RELEASE_FILE="$BATS_TEST_DIRNAME/fixtures/os-release-24.04" SPINA_ARCH=x86_64
    APP_ACCESS_MODE=ip ADMIN_EMAIL=admin@example.test MODELS_DIR="$REWRITE_ROOT/models"
  )
  run "${common[@]}" "$REWRITE_ROOT/install.sh" install --force-cpu
  [ "$status" -eq 0 ]
  secret=$(sed -n 's/^APP_SESSION_SECRET=//p' "$TEST_ROOT/app/.env")
  [ "${#secret}" -ge 43 ]
  [[ "$output" != *"$secret"* ]]
  [ "$(stat -c '%a' "$TEST_ROOT/app/config/app.yaml" 2>/dev/null || stat -f '%Lp' "$TEST_ROOT/app/config/app.yaml")" = 600 ]

  run "${common[@]}" "$REWRITE_ROOT/install.sh" install --force-cpu
  [ "$status" -eq 0 ]
  [ "$(sed -n 's/^APP_SESSION_SECRET=//p' "$TEST_ROOT/app/.env")" = "$secret" ]
  [ "$(cat "$TEST_ROOT/state/accelerator")" = cpu ]
}

@test "configured parallel jobs scales the worker service" {
  mkdir -p "$TEST_ROOT/app/config"
  printf 'inference:\n  max_parallel_jobs: 4\n' >"$TEST_ROOT/app/config/app.yaml"

  run env SPINA_INSTALL_ROOT="$TEST_ROOT/app" bash -c '
    source "$1/lib/common.sh"
    compose() { printf "%s\\n" "$*"; }
    start_compose_stack
  ' _ "$REWRITE_ROOT"

  [ "$status" -eq 0 ]
  [ "$output" = "up -d --scale worker=4" ]
}

@test "external TLS uses supplied certificate without an ACME router" {
  mkdir -p "$TEST_ROOT/app" "$TEST_ROOT/tls"
  printf 'test certificate' >"$TEST_ROOT/tls/fullchain.pem"
  printf 'test private key' >"$TEST_ROOT/tls/privkey.pem"
  printf 'TRAEFIK_TLS_MODE=external\nTRAEFIK_CERTS_DIR=%s\n' "$TEST_ROOT/tls" >"$TEST_ROOT/app/.env"
  common=(env SPINA_TEST_MODE=1 SPINA_INSTALL_ROOT="$TEST_ROOT/app"
    SPINA_STATE_DIR="$TEST_ROOT/state" SPINA_ETC_ROOT="$TEST_ROOT/etc"
    OS_RELEASE_FILE="$BATS_TEST_DIRNAME/fixtures/os-release-24.04" SPINA_ARCH=x86_64
    APP_ACCESS_MODE=domain APP_DOMAIN=demo.example.test ACME_EMAIL=ops@example.test)
  run "${common[@]}" "$REWRITE_ROOT/install.sh" install --force-cpu
  [ "$status" -eq 0 ]
  grep -q 'certFile: /certs/fullchain.pem' "$TEST_ROOT/app/traefik/domain.active.yaml"
  ! grep -q 'certResolver:' "$TEST_ROOT/app/traefik/domain.active.yaml"
  cp "$TEST_ROOT/app/traefik/domain.active.yaml" "$TEST_ROOT/previous-dynamic.yaml"
  rm "$TEST_ROOT/tls/privkey.pem"
  run "${common[@]}" "$REWRITE_ROOT/install.sh" install --force-cpu
  [ "$status" -ne 0 ]
  [[ "$output" == *"External TLS requires"* ]]
  cmp "$TEST_ROOT/previous-dynamic.yaml" "$TEST_ROOT/app/traefik/domain.active.yaml"
}

#!/usr/bin/env bats

setup() {
  export REWRITE_ROOT="$BATS_TEST_DIRNAME/.."
  export SPINA_INSTALL_ROOT="$BATS_TEST_TMPDIR/install"
  export COMPOSE_PS_FIXTURE="$BATS_TEST_TMPDIR/compose-ps.json"
  mkdir -p "$SPINA_INSTALL_ROOT/config"
  printf 'inference:\n  max_parallel_jobs: 1\n' >"$SPINA_INSTALL_ROOT/config/app.yaml"
}

run_health_check() {
  run bash -c '
    source "$REWRITE_ROOT/lib/common.sh"
    source "$REWRITE_ROOT/lib/health.sh"
    compose() { cat "$COMPOSE_PS_FIXTURE"; }
    require_compose_services_healthy
  '
}

@test "stack health requires every configured worker replica" {
  printf 'inference:\n  max_parallel_jobs: 2\n' >"$SPINA_INSTALL_ROOT/config/app.yaml"
  printf '%s\n' '[
    {"Service":"traefik","State":"running","Health":"healthy","ExitCode":0},
    {"Service":"frontend","State":"running","Health":"healthy","ExitCode":0},
    {"Service":"api","State":"running","Health":"healthy","ExitCode":0},
    {"Service":"worker","State":"running","Health":"healthy","ExitCode":0},
    {"Service":"worker","State":"running","Health":"healthy","ExitCode":0},
    {"Service":"migrate","State":"exited","Health":"","ExitCode":0}
  ]' >"$COMPOSE_PS_FIXTURE"

  run_health_check

  [ "$status" -eq 0 ]
}

@test "stack health rejects fewer workers than configured" {
  printf 'inference:\n  max_parallel_jobs: 2\n' >"$SPINA_INSTALL_ROOT/config/app.yaml"
  printf '%s\n' '[
    {"Service":"traefik","State":"running","Health":"healthy","ExitCode":0},
    {"Service":"frontend","State":"running","Health":"healthy","ExitCode":0},
    {"Service":"api","State":"running","Health":"healthy","ExitCode":0},
    {"Service":"worker","State":"running","Health":"healthy","ExitCode":0},
    {"Service":"migrate","State":"exited","Health":"","ExitCode":0}
  ]' >"$COMPOSE_PS_FIXTURE"

  run_health_check

  [ "$status" -ne 0 ]
  [[ "$output" == *"expected=2 actual=1"* ]]
}

@test "stack health requires every long-running service and successful migration" {
  printf '%s\n' '[
    {"Service":"traefik","State":"running","Health":"healthy","ExitCode":0},
    {"Service":"frontend","State":"running","Health":"healthy","ExitCode":0},
    {"Service":"api","State":"running","Health":"healthy","ExitCode":0},
    {"Service":"worker","State":"running","Health":"healthy","ExitCode":0},
    {"Service":"migrate","State":"exited","Health":"","ExitCode":0}
  ]' >"$COMPOSE_PS_FIXTURE"

  run_health_check

  [ "$status" -eq 0 ]
}

@test "stack health rejects an unhealthy worker" {
  printf '%s\n' '[
    {"Service":"traefik","State":"running","Health":"healthy","ExitCode":0},
    {"Service":"frontend","State":"running","Health":"healthy","ExitCode":0},
    {"Service":"api","State":"running","Health":"healthy","ExitCode":0},
    {"Service":"worker","State":"running","Health":"unhealthy","ExitCode":0},
    {"Service":"migrate","State":"exited","Health":"","ExitCode":0}
  ]' >"$COMPOSE_PS_FIXTURE"

  run_health_check

  [ "$status" -ne 0 ]
  [[ "$output" == *"worker"* ]]
}

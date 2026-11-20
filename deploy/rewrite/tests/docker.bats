#!/usr/bin/env bats

setup() {
  export REWRITE_ROOT="$BATS_TEST_DIRNAME/.."
  export TEST_ROOT="$BATS_TEST_TMPDIR/root"
  mkdir -p "$TEST_ROOT/state"
}

@test "CPU source build exports each shared image once" {
  run env REWRITE_ROOT="$REWRITE_ROOT" SPINA_STATE_DIR="$TEST_ROOT/state" bash -c '
    source "$REWRITE_ROOT/lib/common.sh"
    source "$REWRITE_ROOT/lib/docker.sh"
    env_value() { printf "build\n"; }
    compose() { printf "%s\n" "$*"; }
    state_write accelerator cpu
    prepare_images
  '

  [ "$status" -eq 0 ]
  [ "$output" = "build api frontend" ]
}

@test "GPU source build exports CPU and GPU images once" {
  run env REWRITE_ROOT="$REWRITE_ROOT" SPINA_STATE_DIR="$TEST_ROOT/state" bash -c '
    source "$REWRITE_ROOT/lib/common.sh"
    source "$REWRITE_ROOT/lib/docker.sh"
    env_value() { printf "build\n"; }
    compose() { printf "%s\n" "$*"; }
    state_write accelerator gpu
    prepare_images
  '

  [ "$status" -eq 0 ]
  [ "$output" = "build api worker frontend" ]
}

@test "repository setup does not leak a RETURN trap into its caller" {
  run env \
    REWRITE_ROOT="$REWRITE_ROOT" \
    OS_RELEASE_FILE="$BATS_TEST_DIRNAME/fixtures/os-release-24.04" \
    bash -c '
      set -u
      source "$REWRITE_ROOT/lib/common.sh"
      source "$REWRITE_ROOT/lib/docker.sh"
      run_root() { :; }
      curl() {
        local output=""
        while (($#)); do
          if [[ $1 == --output ]]; then
            output=$2
            shift 2
          else
            shift
          fi
        done
        printf key >"$output"
      }
      dpkg() { printf "amd64\n"; }
      wrapper() { install_docker_repository; :; }
      wrapper
    '

  [ "$status" -eq 0 ]
}

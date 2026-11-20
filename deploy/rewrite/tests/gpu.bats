#!/usr/bin/env bats

setup() {
  export REWRITE_ROOT="$BATS_TEST_DIRNAME/.."
  export TEST_ROOT="$BATS_TEST_TMPDIR/root"
  mkdir -p "$TEST_ROOT/state" "$TEST_ROOT/bin"
}

@test "no NVIDIA PCI device selects CPU" {
  run env SPINA_GPU_PRESENT=0 SPINA_STATE_DIR="$TEST_ROOT/state" \
    bash -c 'source "$1/lib/common.sh"; source "$1/lib/gpu.sh"; select_accelerator 0' _ "$REWRITE_ROOT"
  [ "$status" -eq 0 ]
  [ "$output" = cpu ]
}

@test "force CPU overrides detected NVIDIA" {
  run env SPINA_GPU_PRESENT=1 SPINA_STATE_DIR="$TEST_ROOT/state" \
    bash -c 'source "$1/lib/common.sh"; source "$1/lib/gpu.sh"; select_accelerator 1' _ "$REWRITE_ROOT"
  [ "$status" -eq 0 ]
  [ "$output" = cpu ]
  [ "$(cat "$TEST_ROOT/state/force-cpu")" = 1 ]
}

@test "missing driver records a resumable reboot marker" {
  printf '#!/bin/sh\nexit 0\n' >"$TEST_ROOT/bin/ubuntu-drivers"
  printf '#!/bin/sh\nexit 0\n' >"$TEST_ROOT/bin/apt-get"
  chmod +x "$TEST_ROOT/bin/ubuntu-drivers" "$TEST_ROOT/bin/apt-get"
  run env PATH="$TEST_ROOT/bin:/usr/bin:/bin" SPINA_GPU_PRESENT=1 \
    SPINA_STATE_DIR="$TEST_ROOT/state" SPINA_SUDO= \
    bash -c 'source "$1/lib/common.sh"; source "$1/lib/gpu.sh"; ensure_nvidia_driver' _ "$REWRITE_ROOT"
  [ "$status" -eq 75 ]
  [ -f "$TEST_ROOT/state/reboot-required" ]
}

@test "broken NVIDIA container runtime fails with exact remediation" {
  mkdir -p "$TEST_ROOT/bin"
  printf '#!/bin/sh\nexit 0\n' >"$TEST_ROOT/bin/nvidia-smi"
  printf '#!/bin/sh\nexit 0\n' >"$TEST_ROOT/bin/nvidia-ctk"
  printf '#!/bin/sh\nexit 1\n' >"$TEST_ROOT/bin/docker"
  chmod +x "$TEST_ROOT/bin/"*
  run env PATH="$TEST_ROOT/bin:/usr/bin:/bin" SPINA_GPU_PRESENT=1 SPINA_SUDO= \
    bash -c 'source "$1/lib/common.sh"; source "$1/lib/gpu.sh"; validate_nvidia_runtime' _ "$REWRITE_ROOT"
  [ "$status" -ne 0 ]
  [[ "$output" == *"--force-cpu"* ]]
  [[ "$output" == *"nvidia-container-toolkit"* ]]
}

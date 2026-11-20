#!/usr/bin/env bash

gpu_present() {
  if [[ ${SPINA_GPU_PRESENT+x} ]]; then
    [[ $SPINA_GPU_PRESENT == 1 ]]
    return
  fi
  command_exists lspci && lspci -nn | grep -qi 'NVIDIA'
}

nvidia_driver_healthy() {
  command_exists nvidia-smi && nvidia-smi >/dev/null 2>&1
}

ensure_nvidia_driver() {
  nvidia_driver_healthy && return 0
  log "NVIDIA device detected; installing the Ubuntu recommended driver"
  run_root apt-get update
  run_root apt-get install -y ubuntu-drivers-common
  run_root ubuntu-drivers install
  state_write reboot-required "nvidia-driver"
  log "NVIDIA driver installed. Reboot the VM and rerun install."
  return 75
}

install_nvidia_toolkit() {
  local key_temp list_temp keyring=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  key_temp=$(mktemp)
  list_temp=$(mktemp)
  trap 'rm -f "$key_temp" "$list_temp"' RETURN

  curl --fail --silent --show-error --location \
    https://nvidia.github.io/libnvidia-container/gpgkey --output "$key_temp"
  run_root gpg --dearmor --yes --output "$keyring" "$key_temp"
  curl --fail --silent --show-error --location \
    https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    --output "$list_temp"
  sed "s#deb https://#deb [signed-by=$keyring] https://#g" "$list_temp" >"${list_temp}.signed"
  run_root install -m 0644 "${list_temp}.signed" /etc/apt/sources.list.d/nvidia-container-toolkit.list
  run_root apt-get update
  run_root apt-get install -y nvidia-container-toolkit
}

ensure_nvidia_toolkit() {
  command_exists nvidia-ctk || install_nvidia_toolkit
  run_root nvidia-ctk runtime configure --runtime=docker
  run_root systemctl restart docker
}

validate_nvidia_runtime() {
  local image=${GPU_TEST_IMAGE:-nvidia/cuda:12.8.1-base-ubuntu24.04}
  if ! nvidia_driver_healthy || ! command_exists nvidia-ctk \
    || ! docker_command run --rm --gpus all "$image" nvidia-smi >/dev/null 2>&1; then
    die "NVIDIA runtime is broken. Repair nvidia-container-toolkit and Docker, or rerun install --force-cpu."
    return
  fi
}

select_accelerator() {
  local force_cpu=${1:-0}
  if [[ $force_cpu == 1 ]]; then
    state_write force-cpu 1
    state_write accelerator cpu
    printf 'cpu\n'
    return
  fi
  if ! gpu_present; then
    state_write accelerator cpu
    printf 'cpu\n'
    return
  fi
  ensure_nvidia_driver || return
  ensure_nvidia_toolkit
  validate_nvidia_runtime
  rm -f "$SPINA_STATE_DIR/reboot-required" "$SPINA_STATE_DIR/force-cpu"
  state_write accelerator gpu
  printf 'gpu\n'
}

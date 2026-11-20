#!/usr/bin/env bash

docker_command() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    run_root docker "$@"
  fi
}

docker_is_compatible() {
  command_exists docker || return 1
  local version
  version=$(docker_command version --format '{{.Server.Version}}' 2>/dev/null) || return 1
  [[ ${version%%.*} -ge 24 ]] || return 1
  docker_command compose version >/dev/null 2>&1
}

install_docker_repository() (
  local keyring=/etc/apt/keyrings/docker.asc source_file=/etc/apt/sources.list.d/docker.sources
  local temporary source_temp codename arch
  temporary=$(mktemp)
  source_temp=$(mktemp)
  trap 'rm -f "$temporary" "$source_temp"' EXIT

  run_root apt-get update
  run_root apt-get install -y ca-certificates curl
  curl --fail --silent --show-error --location https://download.docker.com/linux/ubuntu/gpg \
    --output "$temporary"
  run_root install -d -m 0755 /etc/apt/keyrings
  run_root install -m 0644 "$temporary" "$keyring"

  # shellcheck source=/etc/os-release
  # shellcheck disable=SC1091
  source "${OS_RELEASE_FILE:-/etc/os-release}"
  codename=${UBUNTU_CODENAME:-$VERSION_CODENAME}
  arch=$(dpkg --print-architecture)
  printf '%s\n' \
    'Types: deb' \
    'URIs: https://download.docker.com/linux/ubuntu' \
    "Suites: $codename" \
    'Components: stable' \
    "Architectures: $arch" \
    "Signed-By: $keyring" >"$source_temp"
  run_root install -m 0644 "$source_temp" "$source_file"
)

install_docker_packages() {
  install_docker_repository
  run_root apt-get update
  run_root apt-get install -y \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  run_root systemctl enable --now docker
}

ensure_docker() {
  local action=${1:-install}
  if [[ $action != update ]] && docker_is_compatible; then
    log "Docker is compatible: $(docker_command version --format '{{.Server.Version}}')"
    return
  fi
  install_docker_packages
  docker_is_compatible || die "Docker daemon or Compose plugin is unavailable after installation"
}

prepare_images() {
  local mode
  mode=$(env_value APP_IMAGE_MODE 2>/dev/null || printf build)
  if [[ $mode == pull ]]; then
    validate_pull_references
    compose pull
  elif [[ $mode == build ]]; then
    if [[ $(state_read accelerator 2>/dev/null || true) == gpu ]]; then
      compose build api worker frontend
    else
      compose build api frontend
    fi
  else
    die "APP_IMAGE_MODE must be build or pull"
  fi
}

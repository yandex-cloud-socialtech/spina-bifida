#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(unset CDPATH; cd -- "$(dirname -- "$0")" && pwd)
export REWRITE_SOURCE_DIR=$SCRIPT_DIR
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/docker.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/gpu.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/backup.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/health.sh"

usage() {
  cat <<'EOF'
Subcommands: install|update|backup|restore|status

Usage:
  ./install.sh install [--force-cpu]
  ./install.sh update --version X.Y.Z [--force-cpu]
  ./install.sh backup
  ./install.sh restore <archive.tar.gz>
  ./install.sh status

Run install/update/restore as root. The installer supports Ubuntu 22.04/24.04 x86_64.
EOF
}

status_help() {
  echo "Reports application/model/schema versions, CPU/CUDA, service health, queue depth, volume use and backup/upload state."
}

require_root() {
  if [[ ${SPINA_TEST_MODE:-0} != 1 && $(id -u) -ne 0 ]]; then
    die "Run this command with sudo: sudo $0 $*"
  fi
}

copy_runtime_bundle() {
  [[ $SCRIPT_DIR == "$SPINA_INSTALL_ROOT" ]] && return
  ensure_private_directory "$SPINA_INSTALL_ROOT"
  mkdir -p "$SPINA_INSTALL_ROOT/config" "$SPINA_INSTALL_ROOT/models"
  cp "$SCRIPT_DIR/install.sh" "$SCRIPT_DIR/compose.yaml" "$SCRIPT_DIR/compose.gpu.yaml" \
    "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/release-manifest.yaml" "$SPINA_INSTALL_ROOT/"
  cp -R "$SCRIPT_DIR/lib" "$SCRIPT_DIR/traefik" "$SPINA_INSTALL_ROOT/"
  cp "$SCRIPT_DIR/models/manifest.sha256" "$SPINA_INSTALL_ROOT/models/manifest.sha256"
  if [[ ! -f $SPINA_INSTALL_ROOT/config/app.example.yaml ]]; then
    cp "$SCRIPT_DIR/config/app.example.yaml" "$SPINA_INSTALL_ROOT/config/app.example.yaml"
  fi
}

secure_app_config() {
  local config="$SPINA_INSTALL_ROOT/config/app.yaml"
  [[ $APP_CONTAINER_UID =~ ^[0-9]+$ && $APP_CONTAINER_GID =~ ^[0-9]+$ ]] \
    || die "APP container UID/GID must be numeric"
  chmod 0600 "$config"
  if [[ ${SPINA_TEST_MODE:-0} != 1 ]]; then
    chown "$APP_CONTAINER_UID:$APP_CONTAINER_GID" "$config"
  fi
}

configure_environment() {
  local env_file="$SPINA_INSTALL_ROOT/.env" secret
  if [[ ! -f $env_file ]]; then
    cp "$SCRIPT_DIR/.env.example" "$env_file"
    chmod 0600 "$env_file"
  fi
  secret=$(env_value APP_SESSION_SECRET "$env_file" 2>/dev/null || true)
  if [[ ${#secret} -lt 43 ]]; then
    set_env_value APP_SESSION_SECRET "$(random_secret)" "$env_file"
  fi
  set_env_value APP_CONFIG_PATH ./config/app.yaml "$env_file"
  set_env_value APP_ENV_FILE .env "$env_file"
  set_env_value MODELS_DIR "${MODELS_DIR:-./models}" "$env_file"
}

configure_access() {
  local mode=${APP_ACCESS_MODE:-} domain=${APP_DOMAIN:-} email=${ACME_EMAIL:-}
  local config="$SPINA_INSTALL_ROOT/config/app.yaml" dynamic tls_mode certs_dir
  tls_mode=$(env_value TRAEFIK_TLS_MODE 2>/dev/null || printf acme)
  tls_mode=${tls_mode:-acme}
  [[ $tls_mode == acme || $tls_mode == external ]] || die "TRAEFIK_TLS_MODE must be acme or external"
  if [[ -z $mode && -t 0 ]]; then
    read -r -p 'Access mode [ip/domain] (ip): ' mode
  fi
  mode=${mode:-ip}
  [[ $mode == ip || $mode == domain ]] || die "APP_ACCESS_MODE must be ip or domain"
  if [[ $mode == domain ]]; then
    if [[ -z $domain && -t 0 ]]; then read -r -p 'Domain: ' domain; fi
    if [[ -z $email && -t 0 ]]; then read -r -p 'ACME email: ' email; fi
    [[ $domain =~ ^[A-Za-z0-9.-]+$ && $domain == *.* ]] || die "A valid APP_DOMAIN is required"
    [[ $email =~ ^[^[:space:]@]+@[^[:space:]@]+\.[^[:space:]@]+$ ]] || die "A valid ACME_EMAIL is required"
  fi

  [[ -f $config ]] || cp "$SCRIPT_DIR/config/app.example.yaml" "$config"
  awk -v mode="$mode" -v domain="$domain" -v email="$email" '
    /^access:/ { in_access = 1; print; next }
    in_access && /^  mode:/ { print "  mode: " mode; next }
    in_access && /^  domain:/ { print "  domain: " domain; next }
    in_access && /^  acme_email:/ { print "  acme_email: " email; in_access = 0; next }
    { print }
  ' "$config" >"${config}.tmp"
  mv "${config}.tmp" "$config"
  secure_app_config

  if [[ $mode == domain ]]; then
    if [[ $tls_mode == external ]]; then
      certs_dir=$(env_value TRAEFIK_CERTS_DIR 2>/dev/null || printf /etc/spina-bifida/tls)
      certs_dir=${certs_dir:-/etc/spina-bifida/tls}
      [[ -s $certs_dir/fullchain.pem && -s $certs_dir/privkey.pem ]] || die "External TLS requires fullchain.pem and privkey.pem in $certs_dir"
    fi
    dynamic="$SPINA_INSTALL_ROOT/traefik/domain.active.yaml"
    sed "s/example\.invalid/$domain/g" "$SPINA_INSTALL_ROOT/traefik/domain.yaml" >"$dynamic"
    if [[ $tls_mode == external ]]; then
      sed '/^        certResolver: letsencrypt$/d; s/^      tls:$/      tls: {}/' "$dynamic" >"${dynamic}.tmp"
      mv "${dynamic}.tmp" "$dynamic"
      cat >>"$dynamic" <<'YAML'

tls:
  certificates:
    - certFile: /certs/fullchain.pem
      keyFile: /certs/privkey.pem
YAML
    fi
    chmod 0600 "$dynamic"
    set_env_value TRAEFIK_DYNAMIC_CONFIG ./traefik/domain.active.yaml
    set_env_value TRAEFIK_ACME_EMAIL "$email"
  else
    set_env_value TRAEFIK_DYNAMIC_CONFIG ./traefik/http.yaml
    set_env_value TRAEFIK_ACME_EMAIL ''
  fi
}

configure_inference_device() {
  local accelerator=$1 config="$SPINA_INSTALL_ROOT/config/app.yaml" device=cpu
  [[ $accelerator == gpu ]] && device=cuda
  awk -v device="$device" '
    /^inference:/ { in_inference = 1; print; next }
    in_inference && /^  device:/ { print "  device: " device; in_inference = 0; next }
    { print }
  ' "$config" >"${config}.tmp"
  mv "${config}.tmp" "$config"
  secure_app_config
}

verify_model_files() {
  local models_dir manifest
  [[ ${SPINA_TEST_MODE:-0} == 1 ]] && return
  models_dir=$(env_value MODELS_DIR)
  [[ $models_dir == /* ]] || models_dir="$SPINA_INSTALL_ROOT/${models_dir#./}"
  manifest="$SPINA_INSTALL_ROOT/models/manifest.sha256"
  [[ -f $manifest ]] || die "Model checksum manifest is missing: $manifest"
  if find "$models_dir" -type l -print -quit | grep -q .; then
    die "Model directory must not contain symbolic links: $models_dir"
  fi
  (cd "$models_dir" && sha256sum -c "$manifest") || die "Model verification failed in $models_dir"
  find "$models_dir" -type d -exec chown "root:$APP_CONTAINER_GID" {} +
  find "$models_dir" -type d -exec chmod 0750 {} +
  find "$models_dir" -type f -exec chown "root:$APP_CONTAINER_GID" {} +
  find "$models_dir" -type f -exec chmod 0640 {} +
}

validate_pull_references() {
  local key reference
  for key in BACKEND_CPU_IMAGE BACKEND_GPU_IMAGE FRONTEND_IMAGE; do
    reference=$(env_value "$key")
    [[ $reference == *@sha256:* ]] || die "$key must be pinned by digest in pull mode"
  done
}

start_application() {
  compose stop worker api >/dev/null 2>&1 || true
  compose run --rm migrate
  compose run --rm --no-deps api flask --app spina_bifida:create_app recover-jobs
  start_compose_stack
  require_stack_health
}

create_first_admin() {
  local admin_email=${ADMIN_EMAIL:-}
  if [[ -z $admin_email && -t 0 ]]; then
    read -r -p 'First administrator email (leave blank to skip): ' admin_email
  fi
  [[ -n $admin_email ]] || {
    log "Administrator creation skipped. Later run: cd $SPINA_INSTALL_ROOT && sudo ./install.sh status, then docker compose run --rm api flask --app spina_bifida:create_app create-admin"
    return
  }
  [[ $admin_email =~ ^[^[:space:]@]+@[^[:space:]@]+\.[^[:space:]@]+$ ]] || die "Invalid ADMIN_EMAIL"
  [[ -t 0 ]] || {
    log "ADMIN_EMAIL was recorded, but password entry requires a TTY. Run create-admin interactively."
    return
  }
  compose run --rm api flask --app spina_bifida:create_app create-admin --email "$admin_email"
}

install_application() {
  local force_cpu=$1 selected
  require_root install
  validate_platform
  verify_release_bundle
  copy_runtime_bundle
  configure_environment
  configure_access
  if [[ ${SPINA_TEST_MODE:-0} == 1 ]]; then
    selected=$(select_accelerator "$force_cpu")
    configure_inference_device "$selected"
    log "Test-mode installation complete; accelerator=$selected"
    return
  fi
  ensure_docker install
  selected=$(select_accelerator "$force_cpu") || return
  configure_inference_device "$selected"
  log "Selected accelerator: $selected"
  verify_model_files
  prepare_images
  start_application
  create_first_admin
  state_write installed-version "$(env_value APP_VERSION)"
  log "Installation complete"
}

update_application() {
  local version=$1 force_cpu=$2 backup image_mode selected
  require_root update
  validate_platform
  [[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9]+)*$ ]] || die "update requires --version X.Y.Z"
  [[ -f $SPINA_INSTALL_ROOT/.env ]] || die "Application is not installed at $SPINA_INSTALL_ROOT"
  verify_release_bundle "$version"
  ensure_docker update
  selected=$(select_accelerator "$force_cpu") || return
  configure_inference_device "$selected"
  verify_model_files
  backup=$(backup_create backup)
  copy_runtime_bundle
  set_env_value APP_VERSION "$version"
  image_mode=$(env_value APP_IMAGE_MODE 2>/dev/null || printf build)
  if [[ $image_mode == pull ]]; then
    set_env_value BACKEND_CPU_IMAGE "$(release_image backend_cpu)"
    set_env_value BACKEND_GPU_IMAGE "$(release_image backend_gpu)"
    set_env_value FRONTEND_IMAGE "$(release_image frontend)"
  else
    set_env_value BACKEND_CPU_IMAGE "spina-bifida-backend:${version}-cpu"
    set_env_value BACKEND_GPU_IMAGE "spina-bifida-backend:${version}-gpu"
    set_env_value FRONTEND_IMAGE "spina-bifida-frontend:${version}"
  fi
  if ! prepare_images || ! start_application; then
    log "Update failed. Previous images remain available. Restore with: sudo $SPINA_INSTALL_ROOT/install.sh restore $backup"
    return 1
  fi
  state_write installed-version "$version"
  log "Update complete: $version"
}

COMMAND=${1:-}
[[ -n $COMMAND ]] || { usage; exit 2; }
shift
FORCE_CPU=0
VERSION=
POSITIONAL=()
while (($#)); do
  case "$1" in
    --force-cpu) FORCE_CPU=1 ;;
    --version) shift; VERSION=${1:-} ;;
    --help | -h)
      if [[ $COMMAND == status ]]; then status_help; else usage; fi
      exit 0
      ;;
    --*) die "Unknown option: $1"; usage; exit 2 ;;
    *) POSITIONAL+=("$1") ;;
  esac
  shift
done

case "$COMMAND" in
  install) install_application "$FORCE_CPU" ;;
  update) update_application "$VERSION" "$FORCE_CPU" ;;
  backup)
    require_root backup
    backup_create backup
    ;;
  restore)
    require_root restore
    [[ ${#POSITIONAL[@]} -eq 1 ]] || die "restore requires one archive path"
    restore_archive "${POSITIONAL[0]}"
    ;;
  status) status_report ;;
  help) usage ;;
  *) usage; exit 2 ;;
esac

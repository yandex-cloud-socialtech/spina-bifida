#!/usr/bin/env bash

health_probe() {
  local config=${SPINA_INSTALL_ROOT}/config/app.yaml mode domain
  if [[ -n ${HEALTH_URL:-} ]]; then
    curl --fail --silent --show-error --max-time 3 "$HEALTH_URL" >/dev/null 2>&1
    return
  fi

  mode=$(awk '
    /^access:/ { in_access = 1; next }
    in_access && /^  mode:/ { print $2; exit }
    in_access && /^[^[:space:]]/ { exit }
  ' "$config" 2>/dev/null || true)
  if [[ $mode == domain ]]; then
    domain=$(awk '
      /^access:/ { in_access = 1; next }
      in_access && /^  domain:/ { print $2; exit }
      in_access && /^[^[:space:]]/ { exit }
    ' "$config" 2>/dev/null || true)
    [[ -n $domain ]] || return 1
    curl --fail --silent --show-error --max-time 3 \
      --resolve "$domain:443:127.0.0.1" "https://$domain/health/ready" >/dev/null 2>&1
    return
  fi

  curl --fail --silent --show-error --max-time 3 \
    http://127.0.0.1/health/ready >/dev/null 2>&1
}

wait_for_health() {
  local attempts=${HEALTH_ATTEMPTS:-60}
  local count=1
  while (( count <= attempts )); do
    if health_probe; then
      return 0
    fi
    sleep 2
    count=$((count + 1))
  done
  die "Healthcheck failed after $attempts attempts"
}

require_compose_services_healthy() {
  local services expected_workers
  services=$(compose ps --all --format json) || die "Unable to inspect Compose services" || return
  expected_workers=$(configured_parallel_jobs) || return
  COMPOSE_PS_JSON=$services EXPECTED_WORKERS=$expected_workers python3 - <<'PY'
import json
import os
import sys

raw = os.environ.get("COMPOSE_PS_JSON", "").strip()
try:
    parsed = json.loads(raw)
    records = parsed if isinstance(parsed, list) else [parsed]
except json.JSONDecodeError:
    try:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        print(f"Invalid docker compose ps JSON: {error}", file=sys.stderr)
        raise SystemExit(1) from error

by_service = {}
for record in records:
    by_service.setdefault(str(record.get("Service", "")), []).append(record)
failures = []
for service in ("traefik", "frontend", "api"):
    service_records = by_service.get(service, [])
    if len(service_records) != 1:
        failures.append(f"{service}: expected=1 actual={len(service_records)}")
        continue
    record = service_records[0]
    state = str(record.get("State", "")).lower()
    health = str(record.get("Health", "")).lower()
    if state != "running" or health != "healthy":
        failures.append(f"{service}: state={state or 'unknown'} health={health or 'unknown'}")

expected_workers = int(os.environ.get("EXPECTED_WORKERS", "1"))
workers = by_service.get("worker", [])
if len(workers) != expected_workers:
    failures.append(f"worker: expected={expected_workers} actual={len(workers)}")
for index, record in enumerate(workers, start=1):
    state = str(record.get("State", "")).lower()
    health = str(record.get("Health", "")).lower()
    if state != "running" or health != "healthy":
        failures.append(
            f"worker[{index}]: state={state or 'unknown'} health={health or 'unknown'}"
        )

migrate_records = by_service.get("migrate", [])
if len(migrate_records) != 1:
    failures.append(f"migrate: expected=1 actual={len(migrate_records)}")
else:
    migrate = migrate_records[0]
    state = str(migrate.get("State", "")).lower()
    try:
        exit_code = int(migrate.get("ExitCode", -1))
    except (TypeError, ValueError):
        exit_code = -1
    if state != "exited" or exit_code != 0:
        failures.append(f"migrate: state={state or 'unknown'} exit_code={exit_code}")

if failures:
    print("Compose health failed: " + "; ".join(failures), file=sys.stderr)
    raise SystemExit(1)
PY
}

wait_for_compose_health() {
  local attempts=${HEALTH_ATTEMPTS:-60} count=1
  while (( count <= attempts )); do
    if require_compose_services_healthy >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
    count=$((count + 1))
  done
  require_compose_services_healthy || true
  die "Compose services did not become healthy after $attempts attempts"
}

require_stack_health() {
  wait_for_compose_health
  wait_for_health
}

status_report() {
  local env_file="$SPINA_INSTALL_ROOT/.env" config_file="$SPINA_INSTALL_ROOT/config/app.yaml"
  local version=unknown default_model=v2 device=unknown schema=unknown queue=unknown upload=unknown
  [[ -r $env_file ]] && version=$(env_value APP_VERSION "$env_file" 2>/dev/null || printf unknown)
  if [[ -r $config_file ]]; then
    default_model=$(awk '
      /^[[:space:]]*default_model_version[[:space:]]*:/ {
        value = $0
        sub(/^[^:]*:[[:space:]]*/, "", value)
        sub(/[[:space:]]+#.*$/, "", value)
        sub(/^[[:space:]]+/, "", value)
        sub(/[[:space:]]+$/, "", value)
        first = substr(value, 1, 1)
        last = substr(value, length(value), 1)
        if ((first == "\"" && last == "\"") || (first == "\047" && last == "\047")) {
          value = substr(value, 2, length(value) - 2)
        }
        print value
        exit
      }
    ' "$config_file")
    default_model=${default_model:-v2}
    device=$(awk '/^[[:space:]]*device:/ {print $2; exit}' "$config_file")
  fi
  if [[ ${SPINA_TEST_MODE:-0} != 1 ]] && docker_is_compatible; then
    compose ps || true
    schema=$(compose exec -T api python -c \
      "import sqlite3; c=sqlite3.connect('/data/db/app.sqlite3'); print(c.execute('select version_num from alembic_version').fetchone()[0])" \
      2>/dev/null || printf unavailable)
    queue=$(compose exec -T api python -c \
      "import sqlite3; c=sqlite3.connect('/data/db/app.sqlite3'); print(c.execute(\"select count(*) from analysis_jobs where state='queued'\").fetchone()[0])" \
      2>/dev/null || printf unavailable)
    upload=$(compose exec -T api python -c \
      "from pathlib import Path; p=Path('/data/db/object-storage-last-success'); print(p.read_text().strip() if p.exists() else 'never')" \
      2>/dev/null || printf unavailable)
  fi
  printf 'application_version=%s\nmodel_versions=v1,v2\ndefault_model_version=%s\nschema_version=%s\ndevice=%s\nqueue_depth=%s\nlast_backup=%s\nlast_object_storage_upload=%s\n' \
    "${version:-unknown}" "$default_model" "$schema" "${device:-unknown}" "$queue" \
    "$(state_read last-backup 2>/dev/null || printf never)" "$upload"
  if [[ -d $SPINA_INSTALL_ROOT/data ]]; then
    du -sh "$SPINA_INSTALL_ROOT/data" 2>/dev/null | awk '{print "data_usage=" $1}'
  fi
}

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(unset CDPATH; cd -- "$(dirname -- "$0")/.." && pwd)
TEST_ROOT=$(mktemp -d)
trap 'rm -rf "$TEST_ROOT"' EXIT

mkdir -p "$TEST_ROOT/config"
cat >"$TEST_ROOT/config/app.yaml" <<'YAML'
access:
  mode: domain
  domain: sb-dev.example.test
  acme_email: ops@example.test
YAML

export SPINA_INSTALL_ROOT="$TEST_ROOT"
# shellcheck source=../lib/common.sh
source "$ROOT_DIR/lib/common.sh"
# shellcheck source=../lib/health.sh
source "$ROOT_DIR/lib/health.sh"

CURL_ARGUMENTS=
curl() {
  CURL_ARGUMENTS=$*
  return 0
}

HEALTH_ATTEMPTS=1 wait_for_health

if [[ " $CURL_ARGUMENTS " != *" --resolve sb-dev.example.test:443:127.0.0.1 "* ]] ||
  [[ " $CURL_ARGUMENTS " != *" https://sb-dev.example.test/health/ready "* ]]; then
  printf 'domain health probe did not use local TLS routing: %s\n' "$CURL_ARGUMENTS" >&2
  exit 1
fi

printf 'Domain health probe verified\n'

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(unset CDPATH; cd -- "$(dirname -- "$0")/.." && pwd)
WITH_MODELS=0
WITH_GPU=0
RELEASE_TMP=$(mktemp -d "${TMPDIR:-/tmp}/spina-release-verify.XXXXXX")
trap 'rm -rf "$RELEASE_TMP"' EXIT

usage() {
  echo "Usage: $0 [--with-models] [--with-gpu]"
}

while (($#)); do
  case "$1" in
    --with-models) WITH_MODELS=1 ;;
    --with-gpu) WITH_GPU=1; WITH_MODELS=1 ;;
    --help | -h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

for command in python3 npm npx docker sqlite3 shellcheck bats git; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "Required command is missing: $command" >&2
    exit 1
  }
done

BACKEND_PYTHON=${BACKEND_PYTHON:-$ROOT_DIR/apps/backend/.venv/bin/python}
[[ -x $BACKEND_PYTHON ]] || {
  echo "Backend virtualenv is missing; see docs/rewrite/DEVELOPMENT.md" >&2
  exit 1
}

run "$BACKEND_PYTHON" -m ruff check "$ROOT_DIR/apps/backend/src" "$ROOT_DIR/apps/backend/tests"
run "$BACKEND_PYTHON" -m mypy "$ROOT_DIR/apps/backend/src/spina_bifida"
run "$BACKEND_PYTHON" -m pytest -q -m 'not ml' "$ROOT_DIR/apps/backend/tests"
run "$BACKEND_PYTHON" -m pytest -q \
  "$ROOT_DIR/apps/backend/tests/integration/test_migrations.py::test_upgrade_from_empty_database_creates_expected_tables"

if [[ $WITH_MODELS == 1 ]]; then
  [[ -n ${SPINA_BIFIDA_MODELS_ROOT:-} ]] || {
    echo "SPINA_BIFIDA_MODELS_ROOT is required with --with-models" >&2
    exit 1
  }
  MODELS_ROOT=$(unset CDPATH; cd -- "$SPINA_BIFIDA_MODELS_ROOT" && pwd)
  run docker build --target test-cpu -t spina-bifida-backend:verify-test-cpu \
    "$ROOT_DIR/apps/backend"
  run docker run --rm \
    --mount "type=bind,src=$ROOT_DIR,dst=/workspace,readonly" \
    --mount "type=bind,src=$MODELS_ROOT,dst=/models,readonly" \
    --env MPLCONFIGDIR=/tmp/matplotlib \
    --env PYTHONPATH=/workspace/apps/backend/src \
    --env SPINA_BIFIDA_MODELS_ROOT=/models \
    --workdir /workspace/apps/backend \
    spina-bifida-backend:verify-test-cpu \
    python -m pytest -q -m ml tests/ml
else
  echo "SKIP: real model weights and golden ML suite (use --with-models)"
fi

(
  cd "$ROOT_DIR/apps/frontend"
  run npm run typecheck
  run npm test -- --run
  run npm run build
)

run docker build --target test-e2e -t spina-bifida-frontend:verify-e2e \
  "$ROOT_DIR/apps/frontend"
run docker run --rm --init --ipc=host spina-bifida-frontend:verify-e2e

(
  cd "$ROOT_DIR/deploy/rewrite"
  run bash tests/test-compose.sh
  run bash tests/test-health-domain.sh
)
run docker build --target runtime-cpu -t spina-bifida-backend:verify-cpu "$ROOT_DIR/apps/backend"
run docker build -t spina-bifida-frontend:verify "$ROOT_DIR/apps/frontend"
run shellcheck \
  "$ROOT_DIR/deploy/rewrite/install.sh" \
  "$ROOT_DIR/deploy/rewrite/download-models.sh" \
  "$ROOT_DIR/deploy/rewrite/lib/"*.sh \
  "$ROOT_DIR/deploy/rewrite/build-release.sh" \
  "$ROOT_DIR/deploy/rewrite/tests/test-compose.sh" \
  "$ROOT_DIR/scripts/verify-rewrite.sh"
run bats "$ROOT_DIR/deploy/rewrite/tests/"*.bats

TEST_CPU_REF="verify.invalid/spina/backend-cpu@sha256:$(printf '1%.0s' {1..64})"
TEST_GPU_REF="verify.invalid/spina/backend-gpu@sha256:$(printf '2%.0s' {1..64})"
TEST_FRONTEND_REF="verify.invalid/spina/frontend@sha256:$(printf '3%.0s' {1..64})"
RELEASE_OUTPUT_DIR="$RELEASE_TMP" \
RELEASE_BACKEND_CPU_IMAGE="$TEST_CPU_REF" \
RELEASE_BACKEND_GPU_IMAGE="$TEST_GPU_REF" \
RELEASE_FRONTEND_IMAGE="$TEST_FRONTEND_REF" \
  run "$ROOT_DIR/deploy/rewrite/build-release.sh" 0.1.0
(
  cd "$RELEASE_TMP"
  if command -v sha256sum >/dev/null 2>&1; then
    run sha256sum -c spina-bifida-0.1.0.tar.gz.sha256
  else
    run shasum -a 256 -c spina-bifida-0.1.0.tar.gz.sha256
  fi
)
run tar -C "$RELEASE_TMP" -xzf "$RELEASE_TMP/spina-bifida-0.1.0.tar.gz"
(
  cd "$RELEASE_TMP/spina-bifida-0.1.0"
  if command -v sha256sum >/dev/null 2>&1; then
    run sha256sum -c release-checksums.sha256
  else
    run shasum -a 256 -c release-checksums.sha256
  fi
  run ./install.sh status --help
  if find . -type f \( -name '.env' -o -name '*.pt' -o -name '*.pyc' \) -print | grep -q .; then
    echo "Release archive contains a forbidden runtime or secret artifact" >&2
    exit 1
  fi
  grep -Eq '^  backend_cpu: .+@sha256:[a-f0-9]{64}$' release-manifest.yaml
  grep -Eq '^  backend_gpu: .+@sha256:[a-f0-9]{64}$' release-manifest.yaml
  grep -Eq '^  frontend: .+@sha256:[a-f0-9]{64}$' release-manifest.yaml
)

if [[ $WITH_GPU == 1 ]]; then
  run docker build --target runtime-gpu -t spina-bifida-backend:verify-gpu "$ROOT_DIR/apps/backend"
  run docker run --rm --gpus all spina-bifida-backend:verify-gpu python -c \
    "import torch; assert torch.cuda.is_available(); print(torch.version.cuda)"
else
  echo "SKIP: NVIDIA runtime and CUDA image suite (use --with-gpu on an NVIDIA host)"
fi

run git -C "$ROOT_DIR" diff --check
if git -C "$ROOT_DIR" grep -nE \
  '(AKIA[0-9A-Z]{16}|BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|APP_SESSION_SECRET=[A-Za-z0-9+/]{40,})' \
  -- . ':(exclude)*.example' ':(exclude)*.bats'; then
  echo "Potential secret material found in tracked files" >&2
  exit 1
fi
echo "Secret-pattern scan passed"
echo "Rewrite verification passed"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(unset CDPATH; cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

export APP_ENV_FILE=${APP_ENV_FILE:-.env.example}
export APP_CONFIG_PATH=${APP_CONFIG_PATH:-./config/app.example.yaml}

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to validate Compose" >&2
  exit 1
fi

docker compose --env-file .env.example -f compose.yaml config --quiet
docker compose --env-file .env.example -f compose.yaml -f compose.gpu.yaml config --quiet

CPU_CONFIG=$(mktemp)
GPU_CONFIG=$(mktemp)
trap 'rm -f "$CPU_CONFIG" "$GPU_CONFIG"' EXIT

docker compose --env-file .env.example -f compose.yaml config --format json >"$CPU_CONFIG"
docker compose --env-file .env.example -f compose.yaml -f compose.gpu.yaml config --format json >"$GPU_CONFIG"

python3 - "$CPU_CONFIG" "$GPU_CONFIG" <<'PY'
import json
import os
import sys
from pathlib import Path

cpu = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gpu = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


services = cpu["services"]
require(set(services) == {"traefik", "frontend", "migrate", "api", "worker"}, "unexpected services")
require(services["migrate"].get("restart") in (None, "no"), "migrate must be one-shot")
require(
    services["api"].get("depends_on", {}).get("migrate", {}).get("condition")
    == "service_completed_successfully",
    "api must wait for a successful migration",
)

for name, service in services.items():
    if name != "traefik":
        require(not service.get("ports"), f"{name} must not publish host ports")
    for mount in service.get("volumes", []):
        source = str(mount.get("source", "")) if isinstance(mount, dict) else str(mount)
        require("docker.sock" not in source, "Docker socket mounts are forbidden")

published = {
    int(port.get("published", 0))
    for port in services["traefik"].get("ports", [])
    if isinstance(port, dict)
}
require(published == {80, 443}, "Traefik must publish exactly ports 80 and 443")

traefik_command = services["traefik"].get("command", [])
require(
    not any(argument.lower().startswith("--configfile=") for argument in traefik_command),
    "Traefik must not mix a file-based static config with ACME CLI flags",
)
for required_argument in (
    "--entrypoints.web.address=:80",
    "--entrypoints.websecure.address=:443",
    "--providers.file.filename=/etc/traefik/dynamic/active.yaml",
    "--ping=true",
    "--certificatesresolvers.letsencrypt.acme.storage=/letsencrypt/acme.json",
    "--certificatesresolvers.letsencrypt.acme.httpchallenge.entrypoint=web",
):
    require(required_argument in traefik_command, f"missing Traefik argument: {required_argument}")

for name in ("frontend", "api", "worker"):
    require("healthcheck" in services[name], f"{name} requires a healthcheck")

worker_mounts = services["worker"].get("volumes", [])
model_mounts = [
    mount
    for mount in worker_mounts
    if isinstance(mount, dict) and mount.get("target") == "/models"
]
require(len(model_mounts) == 1 and model_mounts[0].get("read_only"), "models must be mounted read-only")
for name in ("api", "migrate"):
    require(
        all(not isinstance(mount, dict) or mount.get("target") != "/models" for mount in services[name].get("volumes", [])),
        f"{name} must not mount models",
    )

required_volumes = {"db", "jobs", "feedback", "logs", "backups"}
require(required_volumes <= set(cpu.get("volumes", {})), "required named volumes are missing")

cpu_devices = services["worker"].get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices", [])
gpu_devices = gpu["services"]["worker"].get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices", [])
require(not cpu_devices, "CPU Compose must not reserve a GPU")
require(
    any("gpu" in device.get("capabilities", []) for device in gpu_devices),
    "GPU overlay must reserve an NVIDIA GPU",
)

# Empty TORCH_CPU_INDEX_URL must remain empty for local-wheel builds.
backend_args = services["api"]["build"]["args"]
frontend_args = services["frontend"]["build"]["args"]
for key, default in (
    ("PYTHON_IMAGE", "python:3.12-slim-bookworm"),
    ("DEBIAN_MIRROR", "http://deb.debian.org/debian"),
    ("DEBIAN_SECURITY_MIRROR", "http://deb.debian.org/debian-security"),
    ("TORCH_CPU_INDEX_URL", "https://download.pytorch.org/whl/cpu"),
    ("PIP_FIND_LINKS", ""),
):
    expected = os.environ.get("BUILD_" + key, default)
    require(backend_args[key] == expected, f"incorrect backend build argument: {key}")
for key, default in (
    ("NODE_IMAGE", "node:22-bookworm-slim"),
    ("NGINX_IMAGE", "nginxinc/nginx-unprivileged:1.27-alpine"),
    ("NPM_REGISTRY", "https://registry.npmjs.org"),
):
    require(frontend_args[key] == os.environ.get("BUILD_" + key, default), f"incorrect frontend build argument: {key}")
require(any(v.get("target") == "/certs" and v.get("read_only") for v in services["traefik"]["volumes"]), "external TLS certificates must be read-only")
for name in ("api", "worker", "migrate", "frontend"):
    require(not any(v.get("target") == "/certs" for v in services[name].get("volumes", [])), "TLS private key leaked to application")
print("Compose topology, build arguments and certificate isolation verified")
PY

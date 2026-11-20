#!/usr/bin/env python3
"""Root-only, credential-free VM provisioning payload, run by systemd."""
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tarfile

UPLOAD = Path(__file__).resolve().parent
SOURCE = Path("/opt/spina-bifida-src")
INSTALL = SOURCE / "deploy/rewrite"
MODELS = Path("/opt/spina-bifida-models")
ADMIN_FILE = Path("/etc/spina-bifida/admin.json")


def command(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def extract(archive, target):
    with tarfile.open(archive) as bundle:
        # git archive plus our two data directories contain no external links.
        for member in bundle.getmembers():
            path = (target / member.name).resolve()
            if not path.is_relative_to(target.resolve()) or member.issym() or member.islnk():
                raise RuntimeError("Unsafe archive member")
        bundle.extractall(target, filter="data")


def main():
    cfg = json.loads((UPLOAD / "runtime.json").read_text())
    os.umask(0o077)
    SOURCE.mkdir(parents=True, exist_ok=True)
    extract(UPLOAD / "source.tar.gz", SOURCE)
    extract(UPLOAD / "payload.tar", UPLOAD)
    MODELS.mkdir(exist_ok=True)
    for version in ("v1", "v2"):
        shutil.copytree(UPLOAD / "models" / version, MODELS / version, dirs_exist_ok=True)
    wheels = SOURCE / "apps/backend/requirements/wheels"
    shutil.copytree(UPLOAD / "wheels", wheels, dirs_exist_ok=True)
    version = "yc-" + cfg["source_commit"][:12]
    env_file = INSTALL / ".env"
    # Preserve generated session secret and runtime settings across retries.
    if not env_file.exists():
        shutil.copyfile(INSTALL / ".env.example", env_file)
    settings = cfg["build_env"] | {
        "APP_VERSION": version,
        "BACKEND_CPU_IMAGE": "spina-bifida-backend:" + version + "-cpu",
        "FRONTEND_IMAGE": "spina-bifida-frontend:" + version,
        "BUILD_TORCH_CPU_INDEX_URL": "",
        "BUILD_PIP_FIND_LINKS": "/build/requirements/wheels",
        "TRAEFIK_TLS_MODE": "external" if cfg["certificate_dir"] else "acme",
        "TRAEFIK_CERTS_DIR": "/etc/spina-bifida/tls",
    }
    lines = env_file.read_text().splitlines()
    lines = [line for line in lines if line.split("=", 1)[0] not in settings]
    lines += [key + "=" + value for key, value in settings.items()]
    env_file.write_text("\n".join(lines) + "\n")
    env_file.chmod(0o600)
    env = os.environ | {
        "SPINA_INSTALL_ROOT": str(INSTALL), "MODELS_DIR": str(MODELS),
        "APP_ACCESS_MODE": "domain" if cfg["domain"] else "ip",
        "APP_DOMAIN": cfg["domain"], "ACME_EMAIL": cfg["acme_email"],
    }
    command(["bash", "./install.sh", "install", "--force-cpu"], cwd=INSTALL, env=env)
    compose = ["docker", "compose", "--env-file", ".env", "-f", "compose.yaml"]
    if not ADMIN_FILE.exists():
        ADMIN_FILE.parent.mkdir(mode=0o700, exist_ok=True)
        password = secrets.token_urlsafe(24)
        # Write before CLI: a network/process failure must not silently reset a password.
        credentials = {"email": cfg["admin_email"], "password": password}
        with ADMIN_FILE.open("x") as output:
            json.dump(credentials, output)
    credentials = json.loads(ADMIN_FILE.read_text())
    exists = command([*compose, "exec", "-T", "api", "python", "-c",
        "import sqlite3,sys; c=sqlite3.connect('/data/db/app.sqlite3'); "
        "print(int(c.execute('select 1 from users where email=?', (sys.argv[1].lower(),)).fetchone() is not None))",
        cfg["admin_email"]], cwd=INSTALL, capture_output=True, text=True).stdout.strip()
    if exists != "1":
        password = credentials["password"]
        result = subprocess.run([*compose, "run", "--rm", "--no-deps", "-T", "api", "flask", "--app",
            "spina_bifida:create_app", "create-admin", "--email", cfg["admin_email"]], cwd=INSTALL,
            input=password + "\n" + password + "\nru\n", text=True, capture_output=True)
        if result.returncode:
            # Never include interactive command output or password in journald.
            raise RuntimeError("Administrator creation failed; inspect database before retry")
    # Password stays on VM: smoke tests use it locally, never in process argv/logs.
    command(["python3", str(UPLOAD / "smoke.py")], cwd=INSTALL)
    (SOURCE / "DEPLOYED_SOURCE_COMMIT").write_text(cfg["source_commit"] + "\n")
    # Drop staging duplicates after success; canonical models/wheels remain for rebuilds.
    for name in ("models", "wheels"):
        shutil.rmtree(UPLOAD / name)
    for name in ("source.tar.gz", "payload.tar"):
        (UPLOAD / name).unlink(missing_ok=True)
    print("Cloud deployment and real model smoke checks completed", flush=True)


if __name__ == "__main__":
    main()

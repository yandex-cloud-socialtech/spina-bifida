#!/usr/bin/env python3
"""Provision one CPU stand with yc/OpenSSH; no cloud credentials leave this host."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REMOTE = "/home/yc-user/spina-upload"
DEFAULTS = {
    "models_bucket": "", "models_prefix": "", "zone": "ru-central1-d",
    "cores": 4, "memory_gb": 8, "disk_gb": 50, "ssh_cidr": "0.0.0.0/0",
    "subnet_cidr": "10.79.0.0/24", "subnet_id": "", "yc_profile": "",
    "domain": "", "acme_email": "", "certificate_dir": "", "build_env": {},
}
# Official CPU wheel checksums; checked before any model/VM mutation.
WHEELS = {
    "torch-2.7.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl": "8f8b3cfc53010a4b4a3c7ecb88c212e9decc4f5eeb6af75c3c803937d2d60947",
    "torchvision-0.22.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl": "b5fa7044bd82c6358e8229351c98070cf3a7bf4a6e89ea46352ae6c65745ef94",
}
BUILD_ENV = {
    "BUILD_PYTHON_IMAGE": "cr.yandex/mirror/library/python:3.12-slim-bookworm",
    "BUILD_DEBIAN_MIRROR": "https://mirror.yandex.ru/debian",
    "BUILD_DEBIAN_SECURITY_MIRROR": "https://mirror.yandex.ru/debian-security",
    "BUILD_DOCKERFILE_FRONTEND": "dockerhub.timeweb.cloud/docker/dockerfile:1.7",
    "BUILD_NODE_IMAGE": "dockerhub.timeweb.cloud/library/node:22-bookworm-slim",
    "BUILD_NGINX_IMAGE": "dockerhub.timeweb.cloud/nginxinc/nginx-unprivileged:1.27-alpine",
    "BUILD_NPM_REGISTRY": "https://registry.npmmirror.com",
    "TRAEFIK_IMAGE": "dockerhub.timeweb.cloud/library/traefik:v3.3",
}


class DeployError(Exception):
    pass


def config_hash(cfg, *, identity=False):
    # Access and mirror settings can be repaired without replacing the VM.
    values = {key: value for key, value in cfg.items()
              if not identity or key not in ("domain", "acme_email", "certificate_dir", "build_env")}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def cloud_config(public_key):
    return "#cloud-config\n" + json.dumps({
        "users": [{"name": "yc-user", "groups": ["sudo"], "shell": "/bin/bash",
                   "sudo": ["ALL=(ALL) NOPASSWD:ALL"], "lock_passwd": True,
                   "ssh_authorized_keys": [public_key.strip()]}],
        "ssh_pwauth": False, "disable_root": True,
    }) + "\n"


def check_cloud_init(result):
    info = json.loads(result.stdout)
    errors = info.get("errors", []) + [error for section in info.values() if isinstance(section, dict)
                                       for error in section.get("errors", [])]
    if result.returncode not in (0, 2) or info.get("status") != "done" or errors:
        raise DeployError("cloud-init did not complete successfully; inspect cloud-init status --long on VM")
    if result.returncode == 2:
        print("cloud-init finished with recoverable warnings; no failed modules", flush=True)


def run(args, *, data=None, capture=True, check=True, timeout=None):
    result = subprocess.run([str(a) for a in args], input=data, stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None, timeout=timeout)
    if check and result.returncode:
        # Never echo stdin: it can contain a TLS private key.
        detail = (result.stderr or b"").decode(errors="replace")[-2500:]
        raise DeployError(f"{args[0]} failed ({result.returncode}): {detail}")
    return result


def load_config(path):
    supplied = json.loads(path.read_text())
    if not isinstance(supplied, dict):
        raise DeployError("Configuration must be a JSON object")
    allowed = set(DEFAULTS) | {"folder_id", "name", "admin_email"}
    if set(supplied) - allowed:
        raise DeployError(f"Unknown configuration fields: {sorted(set(supplied) - allowed)}")
    cfg = DEFAULTS | supplied
    for key, default in DEFAULTS.items():
        if isinstance(default, str) and not isinstance(cfg[key], str):
            raise DeployError(f"{key} must be a string")
    for key in ("folder_id", "name", "admin_email"):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise DeployError(f"Required string: {key}")
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,39}[a-z0-9]", cfg["name"]):
        raise DeployError("name: 3..41 lowercase letters, digits, hyphens")
    if not re.fullmatch(r"[a-z0-9]{10,40}", cfg["folder_id"]):
        raise DeployError("Invalid folder_id")
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", cfg["admin_email"]):
        raise DeployError("Invalid admin_email")
    if cfg["zone"] not in ("ru-central1-a", "ru-central1-b", "ru-central1-d"):
        raise DeployError("Unsupported zone")
    for key, minimum in (("cores", 2), ("memory_gb", 4), ("disk_gb", 30)):
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise DeployError(f"{key} must be an integer >= {minimum}")
    for key in ("ssh_cidr", "subnet_cidr"):
        if ipaddress.ip_network(cfg[key]).version != 4:
            raise DeployError(f"{key} must be IPv4")
    if cfg["models_bucket"] and not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", cfg["models_bucket"]):
        raise DeployError("Invalid models_bucket")
    prefix = cfg["models_prefix"].strip("/")
    if ".." in prefix.split("/"):
        raise DeployError("Unsafe models_prefix")
    if prefix and not cfg["models_bucket"]:
        raise DeployError("models_prefix requires models_bucket")
    cfg["models_prefix"] = prefix + "/" if prefix else ""
    if cfg["domain"]:
        if not re.fullmatch(r"[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", cfg["domain"]):
            raise DeployError("Invalid domain")
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", cfg["acme_email"]):
            raise DeployError("HTTPS requires acme_email")
    elif cfg["certificate_dir"]:
        raise DeployError("certificate_dir requires domain")
    if not isinstance(cfg["build_env"], dict) or set(cfg["build_env"]) - set(BUILD_ENV):
        raise DeployError("build_env may only override documented image/package mirrors")
    for value in cfg["build_env"].values():
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_./:+@=-]+", value):
            raise DeployError("Invalid mirror value")
    if cfg["certificate_dir"]:
        cfg["certificate_dir"] = str((path.parent / Path(cfg["certificate_dir"]).expanduser()).resolve())
    return cfg


def atomic_json(path, value):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def verified_download(path, expected, download):
    if path.is_file() and digest(path) == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    download(temporary)
    if digest(temporary) != expected:
        temporary.unlink()
        raise DeployError(f"SHA-256 mismatch: {path.name}; original file preserved")
    os.replace(temporary, path)


def public_models_url(commit):
    source = f"{commit}:deploy/rewrite/models/source-url.txt"
    url = run(["git", "-C", ROOT, "show", source]).stdout.decode().strip()
    if not re.fullmatch(r"https://[a-zA-Z0-9.-]+(?:/[a-zA-Z0-9._-]+)+", url):
        raise DeployError("Invalid public model URL")
    return url + "/"


def download_https(url, target):
    # System curl uses the OS trust store on macOS as well as Linux.
    run(["curl", "--disable", "--fail", "--silent", "--show-error", "--location",
         "--proto", "=https", "--proto-redir", "=https", "--retry", "2",
         "--connect-timeout", "20", "--max-time", "600", "--output", target, url])


class Deployment:
    def __init__(self, cfg, directory, yc="yc"):
        self.cfg, self.directory, self.yc_bin = cfg, directory, yc
        self.state_file = directory / "state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}

    def save(self):
        atomic_json(self.state_file, self.state)

    def yc(self, *args):
        command = [self.yc_bin]
        if self.cfg["yc_profile"]:
            command += ["--profile", self.cfg["yc_profile"]]
        result = run([*command, *args, "--format", "json"])
        return json.loads(result.stdout or b"{}")

    def owned(self, resource):
        return resource.get("labels", {}).get("spina-deployment") == self.state.get("deployment_id")

    def ensure(self, key, kind, name, args):
        # A label survives a lost create response; names alone never establish ownership.
        resources = self.yc(*kind, "list", "--folder-id", self.cfg["folder_id"])
        matches = [item for item in resources if item.get("name") == name]
        if len(matches) > 1 or (matches and not self.owned(matches[0])):
            raise DeployError(f"Unrelated resource already uses name {name}")
        if matches:
            resource = matches[0]
        elif key in self.state["resources"]:
            raise DeployError(f"Resource {key} disappeared; refusing automatic replacement")
        else:
            print(f"Create {key}: {name}", flush=True)
            resource = self.yc(*kind, "create", "--name", name, "--folder-id", self.cfg["folder_id"],
                               "--labels", f"spina-deployment={self.state['deployment_id']}", *args)
        self.state["resources"][key] = resource["id"]
        self.save()
        return resource

    def ssh(self, command, *, data=None, check=True):
        return run(["ssh", "-i", self.directory / "ssh_key", "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                    "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
                    "-o", f"UserKnownHostsFile={self.directory / 'known_hosts'}",
                    f"yc-user@{self.state['ip']}", command], data=data, check=check)

    def upload(self, path, destination):
        # scp does not accept stdin; SSH streaming keeps all paths quoted and secrets out of argv.
        with path.open("rb") as stream:
            command = ["ssh", "-i", str(self.directory / "ssh_key"), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                       "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new", "-o",
                       f"UserKnownHostsFile={self.directory / 'known_hosts'}", f"yc-user@{self.state['ip']}",
                       "umask 077; cat > " + shlex.quote(destination)]
            result = subprocess.run(command, stdin=stream, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode:
                raise DeployError(f"SSH upload failed: {destination}")

    def artifacts(self, commit):
        print("Download and verify 10 model files from Object Storage", flush=True)
        manifest = run(["git", "-C", ROOT, "show", f"{commit}:deploy/rewrite/models/manifest.sha256"]).stdout.decode()
        bucket = self.cfg["models_bucket"]
        if bucket:
            def download_model(target, name):
                self.yc("storage", "s3api", "get-object", "--bucket", bucket,
                        "--key", self.cfg["models_prefix"] + name, str(target))
        else:
            model_url = public_models_url(commit)

            def download_model(target, name):
                download_https(model_url + name, target)

        for line in manifest.splitlines():
            expected, name = line.split()
            if not re.fullmatch(r"v[12]/[a-z_]+\.pt", name):
                raise DeployError("Unsafe model manifest")
            verified_download(self.directory / "models" / name, expected,
                              lambda target, key=name: download_model(target, key))
        print("Download and verify official Linux CPU PyTorch wheels", flush=True)
        for name, expected in WHEELS.items():
            url = "https://download-r2.pytorch.org/whl/cpu/" + name.replace("+", "%2B")
            verified_download(self.directory / "wheels" / name, expected,
                              lambda target, source=url: download_https(source, target))
        run(["git", "-C", ROOT, "archive", "--format=tar.gz", "--output", self.directory / "source.tar.gz", commit])
        for name in ("bootstrap.py", "smoke.py"):
            (self.directory / name).write_bytes(run(["git", "-C", ROOT, "show", f"{commit}:deploy/rewrite/yc/{name}"]).stdout)
        with tarfile.open(self.directory / "payload.tar", "w") as archive:
            archive.add(self.directory / "models", arcname="models")
            archive.add(self.directory / "wheels", arcname="wheels")

    def apply(self, commit, timeout):
        fingerprint = config_hash(self.cfg, identity=True)
        runtime_fingerprint = config_hash(self.cfg)
        if self.state and (self.state.get("config_sha256") != fingerprint or self.state.get("source_commit") != commit):
            raise DeployError("State belongs to another configuration/commit; use the original configuration or a new stand name")
        if not self.state:
            self.state = {"deployment_id": uuid.uuid4().hex[:20], "config_sha256": fingerprint,
                          "source_commit": commit, "resources": {}, "stage": "preflight"}
            self.save()
        if self.state.get("destroyed"):
            raise DeployError("This state was destroyed; use a new state directory for a new stand")
        if self.state.get("stage") == "ready" and self.state.get("runtime_sha256") == runtime_fingerprint:
            self.status()
            return
        if self.cfg["certificate_dir"]:
            for name in ("fullchain.pem", "privkey.pem"):
                if not (Path(self.cfg["certificate_dir"]) / name).is_file():
                    raise DeployError(f"Missing external TLS file: {name}")
        if not (self.directory / "ssh_key").exists():
            run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", self.cfg["name"], "-f", self.directory / "ssh_key"])
        # Fail on inaccessible/mismatching artifacts before creating billable resources.
        self.artifacts(commit)
        subnet_id = self.cfg["subnet_id"]
        if subnet_id:
            subnet = self.yc("vpc", "subnet", "get", "--id", subnet_id)
            if subnet["zone_id"] != self.cfg["zone"] or subnet["folder_id"] != self.cfg["folder_id"]:
                raise DeployError("Existing subnet must be in configured zone and folder")
            network_id = subnet["network_id"]
        else:
            network = self.ensure("network", ["vpc", "network"], self.cfg["name"] + "-net", [])
            network_id = network["id"]
            subnet = self.ensure("subnet", ["vpc", "subnet"], self.cfg["name"] + "-subnet",
                                 ["--network-id", network_id, "--zone", self.cfg["zone"], "--range", self.cfg["subnet_cidr"]])
            subnet_id = subnet["id"]
        rules = []
        for port, cidr in ((22, self.cfg["ssh_cidr"]), (80, "0.0.0.0/0"), (443, "0.0.0.0/0")):
            rules += ["--rule", f"direction=ingress,port={port},protocol=tcp,v4-cidrs=[{cidr}]"]
        rules += ["--rule", "direction=egress,from-port=0,to-port=65535,protocol=any,v4-cidrs=[0.0.0.0/0]"]
        group = self.ensure("security_group", ["vpc", "security-group"], self.cfg["name"] + "-sg", ["--network-id", network_id, *rules])
        metadata = self.directory / "cloud-init.yaml"
        metadata.write_text(cloud_config((self.directory / "ssh_key.pub").read_text()))
        vm = self.ensure("vm", ["compute", "instance"], self.cfg["name"], [
            "--zone", self.cfg["zone"], "--platform", "standard-v3", "--cores", str(self.cfg["cores"]),
            "--memory", str(self.cfg["memory_gb"]) + "GB", "--core-fraction", "100",
            "--create-boot-disk", f"type=network-ssd,size={self.cfg['disk_gb']}GB,image-family=ubuntu-2404-lts,image-folder-id=standard-images,auto-delete=true",
            "--network-interface", f"subnet-id={subnet_id},nat-ip-version=ipv4,security-group-ids={group['id']}",
            "--metadata-from-file", "user-data=" + str(metadata)])
        if vm.get("status") != "RUNNING":
            raise DeployError("VM is not RUNNING; start/repair it explicitly before retry")
        self.state["ip"] = vm["network_interfaces"][0]["primary_v4_address"]["one_to_one_nat"]["address"]
        self.state["stage"] = "vm-created"
        self.save()
        print(f"VM {vm['id']}: {self.state['ip']}; waiting for SSH/cloud-init", flush=True)
        deadline = time.monotonic() + 600
        while self.ssh("true", check=False).returncode:
            if time.monotonic() > deadline:
                raise DeployError("SSH unavailable after 10 minutes; state saved, retry apply")
            time.sleep(5)
        check_cloud_init(self.ssh("sudo -n cloud-init status --wait --format=json", check=False))
        if self.cfg["domain"]:
            print(f"HTTPS: DNS A for {self.cfg['domain']} must point to {self.state['ip']}. "
                  "Configure it now; installation waits for a valid certificate.", flush=True)
        active = self.ssh("sudo -n systemctl is-active spina-bootstrap.service", check=False)
        if self.state.get("runtime_sha256") != runtime_fingerprint:
            if active.stdout.strip() == b"activating":
                raise DeployError("Previous configuration still installs; wait for it to finish before changing access/mirrors")
            if active.stdout.strip() == b"active":
                self.ssh("sudo -n systemctl stop spina-bootstrap.service")
                active = subprocess.CompletedProcess([], 3, b"inactive", b"")
        if active.stdout.strip() not in (b"active", b"activating"):
            self.ssh(f"install -d -m 0700 {REMOTE}")
            runtime = self.cfg | {"source_commit": commit, "build_env": BUILD_ENV | self.cfg["build_env"]}
            atomic_json(self.directory / "runtime.json", runtime)
            for filename in ("source.tar.gz", "payload.tar", "runtime.json"):
                self.upload(self.directory / filename, f"{REMOTE}/{filename}")
            self.upload(self.directory / "bootstrap.py", f"{REMOTE}/bootstrap.py")
            self.upload(self.directory / "smoke.py", f"{REMOTE}/smoke.py")
            if self.cfg["certificate_dir"]:
                self.ssh("sudo -n install -d -m 0700 /etc/spina-bifida/tls")
                for filename in ("fullchain.pem", "privkey.pem"):
                    path = Path(self.cfg["certificate_dir"]) / filename
                    self.ssh("sudo -n sh -c " + shlex.quote("umask 077; cat > /etc/spina-bifida/tls/" + filename), data=path.read_bytes())
            self.ssh("sudo -n systemctl reset-failed spina-bootstrap.service 2>/dev/null || true")
            self.state["runtime_sha256"] = runtime_fingerprint
            self.save()
            self.ssh(f"sudo -n systemd-run --no-block --unit spina-bootstrap --property=Type=oneshot --property=RemainAfterExit=yes /usr/bin/python3 {REMOTE}/bootstrap.py")
        self.state["stage"] = "installing"
        self.save()
        print("Server installs in systemd; logs: sudo journalctl -fu spina-bootstrap", flush=True)
        deadline = time.monotonic() + timeout
        while True:
            info = self.ssh("sudo -n systemctl show spina-bootstrap.service -p ActiveState -p SubState -p Result").stdout.decode()
            if "ActiveState=failed" in info:
                raise DeployError("Server installation failed; use logs, fix the cause and repeat apply. Resources retained.")
            if "SubState=exited" in info and "Result=success" in info:
                break
            if time.monotonic() > deadline:
                raise DeployError("Wait timeout; server continues in systemd. Repeat apply to resume waiting.")
            time.sleep(10)
        self.check_public()
        self.state["stage"] = "ready"
        self.save()
        self.status()

    def url(self):
        return "https://" + self.cfg["domain"] if self.cfg["domain"] else "http://" + self.state["ip"]

    def refresh_vm(self):
        if not self.state.get("resources", {}).get("vm"):
            raise DeployError("No VM in this state")
        vm = self.yc("compute", "instance", "get", "--id", self.state["resources"]["vm"])
        if not self.owned(vm) or vm.get("status") != "RUNNING":
            raise DeployError("VM is not running or ownership label changed")
        self.state["ip"] = vm["network_interfaces"][0]["primary_v4_address"]["one_to_one_nat"]["address"]

    def check_public(self):
        # Default SSL validation is mandatory in domain mode.
        for path in ("/", "/health/ready"):
            response = run(["curl", "--disable", "--fail", "--silent", "--show-error", "--max-time", "20", self.url() + path])
            if path.endswith("ready") and json.loads(response.stdout).get("status") != "ready":
                raise DeployError("Public healthcheck failed")

    def status(self):
        if not self.state.get("ip") or self.state.get("destroyed"):
            print(json.dumps(self.state, indent=2))
            return
        self.refresh_vm()
        self.check_public()
        report = self.ssh("sudo -n env SPINA_INSTALL_ROOT=/opt/spina-bifida-src/deploy/rewrite "
                          "bash -c 'cd /opt/spina-bifida-src/deploy/rewrite && bash ./install.sh status'").stdout.decode()
        print(report)
        print(f"URL: {self.url()}\nAdmin: {self.cfg['admin_email']}\nCredentials: /etc/spina-bifida/admin.json on VM (root, 0600)")
        print(f"SSH: ssh -i {shlex.quote(str(self.directory / 'ssh_key'))} yc-user@{self.state['ip']}")

    def destroy(self, confirmation):
        if confirmation != self.cfg["name"]:
            raise DeployError("destroy requires --confirm <exact stand name>; VM data will be deleted")
        kinds = {"vm": ["compute", "instance"], "security_group": ["vpc", "security-group"],
                 "subnet": ["vpc", "subnet"], "network": ["vpc", "network"]}
        for key, kind in kinds.items():
            resource_id = self.state.get("resources", {}).get(key)
            if not resource_id:
                continue
            items = self.yc(*kind, "list", "--folder-id", self.cfg["folder_id"])
            resource = next((item for item in items if item["id"] == resource_id), None)
            if resource and not self.owned(resource):
                raise DeployError(f"Refusing deletion of resource without this deployment's label: {resource_id}")
            if resource:
                print(f"Delete {key}: {resource_id}", flush=True)
                self.yc(*kind, "delete", "--id", resource_id)
            del self.state["resources"][key]
            self.save()
        self.state["destroyed"] = True
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "apply", "status", "logs", "destroy"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--yc", default="yc")
    parser.add_argument("--source-ref", help="default: recorded commit on retry, otherwise HEAD")
    parser.add_argument("--timeout", type=int, default=7200, help="server wait seconds")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    directory = (args.state_dir or args.config.resolve().parent / ".state" / cfg["name"]).resolve()
    recorded = json.loads((directory / "state.json").read_text()) if (directory / "state.json").exists() else {}
    source_ref = args.source_ref or recorded.get("source_commit", "HEAD")
    commit = run(["git", "-C", ROOT, "rev-parse", "--verify", source_ref + "^{commit}"]).stdout.decode().strip()
    if args.action == "plan":
        models = (f"s3://{cfg['models_bucket']}/{cfg['models_prefix']}"
                  if cfg["models_bucket"] else public_models_url(commit))
        print(json.dumps({"name": cfg["name"], "folder_id": cfg["folder_id"], "zone": cfg["zone"],
              "cpu_ram_disk": [cfg["cores"], cfg["memory_gb"], cfg["disk_gb"]], "existing_subnet": cfg["subnet_id"],
              "create": ["VM with auto-delete SSD", "security group 22/80/443"] + ([] if cfg["subnet_id"] else ["network", "subnet"]),
              "models": models, "domain": cfg["domain"],
              "source_commit": commit, "state_directory": str(directory), "cloud_credentials_on_vm": False}, indent=2))
        return
    for tool in (args.yc, "ssh", "ssh-keygen", "git", "curl"):
        if not shutil.which(tool):
            raise DeployError(f"Missing command: {tool}")
    def execute():
        deployment = Deployment(cfg, directory, args.yc)
        fingerprint = config_hash(cfg, identity=True)
        if deployment.state and deployment.state.get("config_sha256") != fingerprint:
            raise DeployError("Configuration differs from this state; refusing to operate on another target")
        if args.action == "apply":
            deployment.apply(commit, args.timeout)
        elif args.action == "destroy":
            deployment.destroy(args.confirm)
        elif args.action == "logs":
            deployment.refresh_vm()
            result = deployment.ssh("sudo -n journalctl -u spina-bootstrap.service --no-pager -n 100")
            print(result.stdout.decode())
        else:
            deployment.status()
    if args.action in ("status", "logs"):
        # State snapshots are replaced atomically; read-only inspection must work during apply.
        execute()
        return
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    with (directory / "lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DeployError("Another process owns this state") from error
        execute()


if __name__ == "__main__":
    try:
        main()
    except (DeployError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

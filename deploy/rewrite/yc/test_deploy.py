"""Cloud safety and resume regressions; python3 -m unittest discover -s deploy/rewrite/yc."""
import contextlib
import fcntl
import importlib.util
import io
import json
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("cloud_deploy", Path(__file__).with_name("deploy.py"))
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


class SimulatedCloud(deploy.Deployment):
    def __init__(self, cfg, directory):
        super().__init__(cfg, directory)
        self.cloud, self.calls, self.transfers = {}, [], []

    def yc(self, *args):
        self.calls.append(args)
        kind, operation = tuple(args[:2]), args[2]
        records = self.cloud.setdefault(kind, [])
        if operation == "list":
            return records
        if kind == ("vpc", "subnet") and operation == "get":
            return {"id": "existing", "zone_id": self.cfg["zone"], "folder_id": self.cfg["folder_id"], "network_id": "external-network"}
        if operation == "create":
            if kind == ("vpc", "security-group"):
                for index, value in enumerate(args):
                    if value == "--rule":
                        rule = args[index + 1]
                        if "port=" not in rule:
                            raise deploy.DeployError("yc requires a port or from/to-port range")
            record = {"id": str(len(self.calls)), "name": args[args.index("--name") + 1],
                      "labels": {"spina-deployment": self.state["deployment_id"]}}
            if kind == ("compute", "instance"):
                record |= {"status": "RUNNING", "network_interfaces": [{"primary_v4_address": {"one_to_one_nat": {"address": "192.0.2.7"}}}]}
            records.append(record)
            return record
        if operation == "delete":
            resource_id = args[args.index("--id") + 1]
            self.cloud[kind] = [item for item in records if item["id"] != resource_id]
            return {}
        raise AssertionError(args)

    def artifacts(self, commit):
        self.transfers.append("artifacts verified")

    def ssh(self, command, **kwargs):
        if "cloud-init status" in command:
            return subprocess.CompletedProcess([], 0, b'{"status":"done","errors":[]}', b"")
        if "is-active" in command:
            return subprocess.CompletedProcess([], 3, b"inactive\n", b"")
        if "systemctl show" in command:
            return subprocess.CompletedProcess([], 0, b"ActiveState=active\nSubState=exited\nResult=success\n", b"")
        return subprocess.CompletedProcess([], 0, b"", b"")

    def upload(self, path, destination):
        self.transfers.append(destination)

    def check_public(self):
        pass

    def status(self):
        pass


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"folder_id": "b1g00000000000000000", "name": "sb-test", "admin_email": "admin@example.com", "models_bucket": "test-models"}))
        self.cfg = deploy.load_config(self.config)

    def client(self):
        directory = self.root / "state"
        directory.mkdir()
        (directory / "ssh_key").write_text("fixture")
        (directory / "ssh_key.pub").write_text("ssh-ed25519 public-key-fixture")
        return SimulatedCloud(self.cfg, directory)

    def test_plan_has_no_cloud_calls_or_state_files(self):
        before = set(self.root.rglob("*"))
        with patch("sys.argv", ["deploy.py", "plan", "--config", str(self.config)]), patch.object(deploy, "run") as command:
            command.return_value = subprocess.CompletedProcess([], 0, b"012345\n", b"")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                deploy.main()
        self.assertEqual(set(self.root.rglob("*")), before)
        self.assertFalse(json.loads(output.getvalue())["cloud_credentials_on_vm"])
        self.assertEqual(command.call_count, 1)  # git rev-parse only

    def test_configuration_rejects_command_injection_and_wrong_fields(self):
        for extra in ({"admin_password": "forbidden"}, {"build_env": {"BUILD_PYTHON_IMAGE": "$(touch /tmp/pwn)"}},
                      {"cores": True}, {"models_prefix": "../private"}, {"domain": "bad;domain"}, {"yc_profile": []}):
            self.config.write_text(json.dumps(self.cfg | extra))
            with self.assertRaises((deploy.DeployError, ValueError)):
                deploy.load_config(self.config)

    def test_plan_defaults_to_public_model_downloads(self):
        config = dict(self.cfg)
        del config["models_bucket"]
        self.config.write_text(json.dumps(config))
        src_url = "https://storage.example.org/model-release/2026-11"
        with patch("sys.argv", ["deploy.py", "plan", "--config", str(self.config)]), patch.object(deploy, "run") as command:
            command.side_effect = [
                subprocess.CompletedProcess([], 0, b"012345\n", b""),
                subprocess.CompletedProcess([], 0, src_url.encode(), b""),
            ]
            with contextlib.redirect_stdout(io.StringIO()) as output:
                deploy.main()
        self.assertEqual(json.loads(output.getvalue())["models"], src_url + "/")
        self.assertEqual(command.call_args.args[0][-1], "012345:deploy/rewrite/models/source-url.txt")

    def test_models_prefix_requires_custom_bucket(self):
        self.config.write_text(json.dumps(self.cfg | {"models_bucket": "", "models_prefix": "v2"}))
        with self.assertRaisesRegex(deploy.DeployError, "models_prefix requires models_bucket"):
            deploy.load_config(self.config)

    def test_status_is_available_while_apply_owns_lock(self):
        directory = self.root / "state"
        directory.mkdir()
        with (directory / "lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch("sys.argv", ["deploy.py", "status", "--config", str(self.config), "--state-dir", str(directory)]), \
                 patch.object(deploy, "run") as command, patch.object(deploy.shutil, "which", return_value="fixture"), \
                 patch.object(deploy.Deployment, "status") as status:
                command.return_value = subprocess.CompletedProcess([], 0, b"commit\n", b"")
                deploy.main()
                status.assert_called_once()

    def test_corrupted_download_preserves_existing_file(self):
        path = self.root / "model.pt"
        path.write_bytes(b"previous")
        with self.assertRaises(deploy.DeployError):
            deploy.verified_download(path, "0" * 64, lambda p: p.write_bytes(b"corrupt"))
        self.assertEqual(path.read_bytes(), b"previous")

    def test_verified_cache_avoids_network(self):
        path = self.root / "model.pt"
        path.write_bytes(b"model")
        deploy.verified_download(path, deploy.digest(path), lambda p: self.fail("unexpected network"))

    def test_cloud_init_rejects_failed_modules_but_accepts_finished_warning(self):
        for code, info in ((1, {"status": "error"}), (2, {"status": "done", "errors": ["failed"]}),
                           (0, {"status": "done", "modules-final": {"errors": ["failed"]}})):
            with self.assertRaises(deploy.DeployError):
                deploy.check_cloud_init(subprocess.CompletedProcess([], code, json.dumps(info).encode()))
        with contextlib.redirect_stdout(io.StringIO()):
            deploy.check_cloud_init(subprocess.CompletedProcess([], 2, b'{"status":"done","errors":[]}'))

    def test_ssh_cloud_config_has_no_datasource_or_password_login(self):
        config = json.loads(deploy.cloud_config("ssh-ed25519 public-key").split("\n", 1)[1])
        self.assertNotIn("datasource", config)
        self.assertFalse(config["ssh_pwauth"])
        self.assertTrue(config["users"][0]["lock_passwd"])

    def test_existing_foreign_resource_is_never_adopted(self):
        client = self.client()
        client.state = {"deployment_id": "mine", "resources": {}}
        client.cloud[("vpc", "network")] = [{"name": "sb-test-net", "id": "foreign", "labels": {}}]
        with self.assertRaises(deploy.DeployError):
            client.ensure("network", ["vpc", "network"], "sb-test-net", [])
        self.assertFalse(any("create" in call for call in client.calls))

    def test_lost_create_response_recovers_by_label(self):
        client = self.client()
        client.state = {"deployment_id": "mine", "resources": {}}
        client.cloud[("vpc", "network")] = [{"name": "sb-test-net", "id": "ours", "labels": {"spina-deployment": "mine"}}]
        self.assertEqual(client.ensure("network", ["vpc", "network"], "sb-test-net", [])["id"], "ours")
        self.assertFalse(any("create" in call for call in client.calls))

    def test_deleted_resource_is_not_replaced(self):
        client = self.client()
        client.state = {"deployment_id": "mine", "resources": {"network": "missing"}}
        with self.assertRaises(deploy.DeployError):
            client.ensure("network", ["vpc", "network"], "sb-test-net", [])

    def test_apply_creates_one_stand_without_vm_service_account_and_retry_is_noop(self):
        client = self.client()
        with contextlib.redirect_stdout(io.StringIO()):
            client.apply("commit", 10)
            count = len(client.calls)
            client.apply("commit", 10)
        self.assertEqual(len(client.calls), count)
        self.assertEqual(len(client.state["resources"]), 4)
        vm_command = next(call for call in client.calls if call[:3] == ("compute", "instance", "create"))
        self.assertNotIn("--service-account-id", vm_command)
        self.assertNotIn("--service-account-name", vm_command)
        self.assertEqual(client.state["stage"], "ready")
        self.assertEqual(stat.S_IMODE(client.state_file.stat().st_mode), 0o600)

    def test_destroy_requires_exact_name_and_refuses_foreign_labels(self):
        client = self.client()
        client.state = {"deployment_id": "mine", "resources": {"vm": "foreign"}}
        client.cloud[("compute", "instance")] = [{"id": "foreign", "name": "sb-test", "labels": {}}]
        with self.assertRaises(deploy.DeployError):
            client.destroy("sb-other")
        with self.assertRaises(deploy.DeployError):
            client.destroy("sb-test")
        self.assertFalse(any("delete" in call for call in client.calls))

    def test_existing_subnet_is_used_and_never_deleted(self):
        client = self.client()
        client.cfg = client.cfg | {"subnet_id": "existing"}
        with contextlib.redirect_stdout(io.StringIO()):
            client.apply("commit", 10)
            self.assertEqual(set(client.state["resources"]), {"security_group", "vm"})
            client.destroy("sb-test")
        for call in client.calls:
            self.assertFalse(call[:3] in (("vpc", "subnet", "delete"), ("vpc", "network", "delete")))

    def test_destroy_is_resumable_and_keeps_model_bucket(self):
        client = self.client()
        with contextlib.redirect_stdout(io.StringIO()):
            client.apply("commit", 10)
            client.destroy("sb-test")
            client.destroy("sb-test")
        self.assertTrue(client.state["destroyed"])
        self.assertEqual(client.state["resources"], {})
        self.assertFalse(any("storage" in call for call in client.calls))


if __name__ == "__main__":
    unittest.main()

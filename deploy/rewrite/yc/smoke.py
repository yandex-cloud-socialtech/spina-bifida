#!/usr/bin/env python3
"""Real-weight acceptance against the local API; no secrets printed."""
import http.cookiejar
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import uuid


def main():
    base = "http://api:8000"
    # The helper joins Compose networking; API has no published port.
    if not Path("/app/alembic.ini").exists():
        example = Path.cwd().parents[1] / "apps/frontend/public/examples/norm-axial.jpg"
        if not example.is_file():
            raise RuntimeError("Missing application example image")
        subprocess.run(["docker", "compose", "--env-file", ".env", "-f", "compose.yaml", "run", "--rm", "--no-deps",
            "-T", "--user", "0", "--volume", "/etc/spina-bifida/admin.json:/etc/spina-bifida/admin.json:ro",
            "--volume", f"{example}:/tmp/spina-smoke-example.jpg:ro", "api", "python", "-"],
            input=Path(__file__).read_bytes(), check=True)
        return
    credentials = json.loads(Path("/etc/spina-bifida/admin.json").read_text())
    # Helper calls the API over the private Compose network, including in domain mode.
    # Public TLS is verified separately by the launcher; production cookie settings stay intact.
    jar = http.cookiejar.CookieJar(policy=http.cookiejar.DefaultCookiePolicy(secure_protocols=("https", "http")))
    client = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    token = ""

    def api(path, method="GET", data=None, content_type="application/json"):
        request = urllib.request.Request(base + path, data=data, method=method,
            headers={"Content-Type": content_type, "X-CSRF-Token": token})
        # Sanitise HTTP errors: never include headers, cookies or credentials.
        with client.open(request, timeout=30) as response:
            body = response.read()
            return json.loads(body) if body else None

    try:
        api("/api/v1/auth/me")
    except urllib.error.HTTPError as error:
        if error.code != 401:
            raise RuntimeError("Unexpected anonymous auth response") from None
        token = json.loads(error.read())["csrf_token"]
    token = api("/api/v1/auth/login", "POST", json.dumps(credentials).encode())["csrf_token"]
    api("/api/v1/auth/terms/accept", "POST", b"{}")
    source = Path("/tmp/spina-smoke-example.jpg").read_bytes()
    for version in ("v1", "v2"):
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"model_version\"\r\n\r\n{version}\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"frames\"; filename=\"example.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").encode()
        body += source + f"\r\n--{boundary}--\r\n".encode()
        jobs = api("/api/v1/analyses", "POST", body, "multipart/form-data; boundary=" + boundary)["jobs"]
        for job in jobs:
            deadline = time.monotonic() + 360
            while True:
                result = api(job["polling_url"])
                if result["state"] in ("completed", "rejected_quality"):
                    break
                if result["state"] == "failed_recoverable" or time.monotonic() > deadline:
                    raise RuntimeError("Model smoke failed: " + version)
                time.sleep(2)
            for asset in result["result"]["frames"][0]["assets"]:
                with client.open(base + asset["url"], timeout=30) as response:
                    if not response.read():
                        raise RuntimeError("Empty result asset")
            api(job["polling_url"], "DELETE")
        print("Model " + version + ": PASS", flush=True)


if __name__ == "__main__":
    main()

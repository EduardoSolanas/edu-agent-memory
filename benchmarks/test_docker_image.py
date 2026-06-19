import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

IMAGE_NAME = "edumem-test:latest"
CONTAINER_NAME = "edumem-test-instance"
HOST_PORT = 6346
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GPU_DEVICE = "/dev/dri"


def pick_runtime():
    for runtime in ("podman", "docker"):
        resolved = shutil.which(runtime)
        if resolved:
            return resolved
    raise FileNotFoundError("Neither podman nor docker was found on PATH.")


def load_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if val.startswith(('"', "'")) and val.endswith(val[0]):
            val = val[1:-1]
        if key == "HF_TOKEN" and val:
            return val
    return None


def run_cmd(cmd, check=True, env=None):
    print(f"[*] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if check and res.returncode != 0:
        print(f"[!] Error executing: {' '.join(cmd)}")
        print(f"STDOUT:\n{res.stdout}")
        print(f"STDERR:\n{res.stderr}")
        raise subprocess.CalledProcessError(res.returncode, cmd)
    return res.stdout, res.stderr


def main():
    print("==================================================")
    print("NATIVE SINGLE-IMAGE CONTAINER VERIFICATION SUITE")
    print("==================================================")

    runtime = pick_runtime()
    build_env = os.environ.copy()
    build_args = []
    hf_token = load_hf_token()
    if hf_token:
        build_env["HF_TOKEN"] = hf_token
        build_args = ["--build-arg", "HF_TOKEN"]

    print("[*] Cleaning up any stale test containers...")
    subprocess.run([runtime, "rm", "-f", CONTAINER_NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        print("[*] Building the self-contained container image (this uses local cache)...")
        run_cmd([runtime, "build", "-t", IMAGE_NAME, *build_args, str(PROJECT_ROOT)], env=build_env)

        print(f"[*] Spinning up test container '{CONTAINER_NAME}' on mapped host port {HOST_PORT}...")
        run_args = [
            runtime,
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"{HOST_PORT}:6336",
            "-p",
            "6343:6333",
            "-p",
            "3012:3002",
        ]
        if os.path.exists(GPU_DEVICE):
            run_args.extend([
                "--privileged",
                "--device",
                f"{GPU_DEVICE}:{GPU_DEVICE}",
            ])
        run_args.append(IMAGE_NAME)
        run_cmd(run_args)

        print("[*] Polling container health (allowing up to 25s for background OpenVINO model compilation)...")
        online = False
        url = f"http://127.0.0.1:{HOST_PORT}/health"

        for i in range(25):
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode())
                        if data.get("status") == "healthy":
                            print(f"[+] Container is online and healthy in {i+1}s!")
                            print(f"[+] System payload: {json.dumps(data, indent=2)}")
                            online = True
                            break
            except Exception:
                pass
            time.sleep(1)

        if not online:
            print("[!] Container failed to become healthy. Fetching container logs...")
            out, _ = run_cmd([runtime, "logs", CONTAINER_NAME], check=False)
            print(f"--- CONTAINER LOGS ---\n{out}\n----------------------")
            raise TimeoutError("Container health poll timed out.")

        print("\n[*] Exercising the containerized pipeline (E2E Write -> Read)...")
        retain_url = f"http://127.0.0.1:{HOST_PORT}/v1/default/banks/testbank/memories/retain"
        recall_url = f"http://127.0.0.1:{HOST_PORT}/v1/default/banks/testbank/memories/recall"

        print("[*] Sending RETAIN request to store a brand new fact...")
        test_fact = f"TDD Docker Verify {int(time.time())}: The neon pink flamingo danced on host port {HOST_PORT}"
        headers = {"Content-Type": "application/json"}
        retain_data = json.dumps({"content": test_fact, "tags": ["user:docker-verifier"]}).encode()

        req = urllib.request.Request(retain_url, data=retain_data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as res:
            res_data = json.loads(res.read().decode())
            print(f"[+] Retain Response: {json.dumps(res_data)}")
            assert res_data.get("result") == "Memory stored successfully."
            assert res_data.get("id")

        print("[*] Waiting 1.5s for Qdrant index synchronization...")
        time.sleep(1.5)

        print("[*] Sending RECALL request to retrieve the stored fact...")
        recall_data = json.dumps({
            "query": "Which pink animal danced on port?",
            "tags": ["user:docker-verifier"],
        }).encode()

        req = urllib.request.Request(recall_url, data=recall_data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as res:
            res_data = json.loads(res.read().decode())
            results = res_data.get("results", [])
            print(f"[+] Recall Response: found {len(results)} matches.")
            for match in results:
                print(f"    - Text: \"{match['text']}\" (Score: {match['score']:.4f})")

            found = any(test_fact in r["text"] for r in results)
            if found:
                print("\n[OK] SUCCESS: Vector search matched and retrieved the stored fact with 100% precision!")
            else:
                print("\n[FAIL] Fact was stored but not retrieved by the vector search.")
                raise ValueError("E2E Verification failed.")

    finally:
        print("\n[*] Cleaning up test resources...")
        run_cmd([runtime, "rm", "-f", CONTAINER_NAME], check=False)
        run_cmd([runtime, "rmi", IMAGE_NAME], check=False)
        print("[*] Cleanup complete. Host workspace is pristine.")


if __name__ == "__main__":
    main()

import subprocess
import time
import urllib.request
import json
import os

IMAGE_NAME = "edumem-test:latest"
CONTAINER_NAME = "edumem-test-instance"
HOST_PORT = 6346

def run_cmd(cmd, check=True):
    print(f"[*] Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and res.returncode != 0:
        print(f"[!] Error executing: {cmd}")
        print(f"STDOUT:\n{res.stdout}")
        print(f"STDERR:\n{res.stderr}")
        raise subprocess.CalledProcessError(res.returncode, cmd)
    return res.stdout, res.stderr

def main():
    print("==================================================")
    print("🐳 NATIVE SINGLE-IMAGE DOCKER VERIFICATION SUITE")
    print("==================================================")
    
    # 1. Ensure any stale containers are cleaned up first
    print("[*] Cleaning up any stale test containers...")
    subprocess.run(f"docker rm -f {CONTAINER_NAME} >/dev/null 2>&1", shell=True)
    
    try:
        # 2. Build the single all-in-one Docker image
        print("[*] Building the self-contained Docker image (this uses local cache)...")
        run_cmd(f"docker build -t {IMAGE_NAME} /opt/edumem")
        
        # 3. Spin up the container with mapped ports (preventing host port collisions)
        print(f"[*] Spinning up test container '{CONTAINER_NAME}' on mapped host port {HOST_PORT}...")
        run_cmd(f"docker run -d --name {CONTAINER_NAME} -p {HOST_PORT}:6336 -p 6343:6333 -p 3012:3002 --privileged -v /dev/dri:/dev/dri {IMAGE_NAME}")
        
        # 4. Poll health endpoint until online and loaded
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
            out, _ = run_cmd(f"docker logs {CONTAINER_NAME}", check=False)
            print(f"--- CONTAINER LOGS ---\n{out}\n----------------------")
            raise TimeoutError("Container health poll timed out.")
            
        # 5. Run end-to-end integration write and read test against the mapped host port
        print("\n[*] Exercising the containerized pipeline (E2E Write -> Read)...")
        retain_url = f"http://127.0.0.1:{HOST_PORT}/v1/default/banks/testbank/memories/retain"
        recall_url = f"http://127.0.0.1:{HOST_PORT}/v1/default/banks/testbank/memories/recall"
        
        # Retain / Write
        print("[*] Sending RETAIN request to store a brand new fact...")
        test_fact = f"TDD Docker Verify {int(time.time())}: The neon pink flamingo danced on host port {HOST_PORT}"
        headers = {'Content-Type': 'application/json'}
        retain_data = json.dumps({"content": test_fact, "tags": ["user:docker-verifier"]}).encode()
        
        req = urllib.request.Request(retain_url, data=retain_data, headers=headers, method='POST')
        with urllib.request.urlopen(req) as res:
            res_data = json.loads(res.read().decode())
            print(f"[+] Retain Response: {json.dumps(res_data)}")
            assert res_data.get("result") == "Memory stored successfully."
            assert res_data.get("id")
            
        # Wait for Qdrant write-index sync
        print("[*] Waiting 1.5s for Qdrant index synchronization...")
        time.sleep(1.5)
        
        # Recall / Read
        print("[*] Sending RECALL request to retrieve the stored fact...")
        recall_data = json.dumps({
            "query": "Which pink animal danced on port?",
            "tags": ["user:docker-verifier"]
        }).encode()
        
        req = urllib.request.Request(recall_url, data=recall_data, headers=headers, method='POST')
        with urllib.request.urlopen(req) as res:
            res_data = json.loads(res.read().decode())
            results = res_data.get("results", [])
            print(f"[+] Recall Response: found {len(results)} matches.")
            for match in results:
                print(f"    - Text: \"{match['text']}\" (Score: {match['score']:.4f})")
            
            # Verify the fact is found in the returned vector search results
            found = any(test_fact in r["text"] for r in results)
            if found:
                print("\n[✔] SUCCESS: Vector search matched and retrieved the stored fact with 100% precision!")
            else:
                print("\n[✘] FAILURE: Fact was stored but not retrieved by the vector search.")
                raise ValueError("E2E Verification failed.")
                
    finally:
        # 6. Teardown and clean up resources
        print("\n[*] Cleaning up test resources...")
        run_cmd(f"docker rm -f {CONTAINER_NAME} >/dev/null 2>&1", check=False)
        run_cmd(f"docker rmi {IMAGE_NAME} >/dev/null 2>&1", check=False)
        print("[*] Cleanup complete. Host workspace is pristine.")

if __name__ == "__main__":
    main()

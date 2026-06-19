#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import shutil

# Determine paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = "/opt/edumem"
if not os.path.exists(PROJECT_ROOT):
    # Fallback to parent of script dir if not running in standard location
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

GTE_MODEL = "sentence-transformers/all-mpnet-base-v2"
GTE_DIR = os.path.join(MODELS_DIR, "gte-modernbert-ov")

ETTIN_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ETTIN_DIR = os.path.join(MODELS_DIR, "ettin-17m-ov")

def load_hf_token():
    # 1. Check in os.environ
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    # 2. Try parsing /opt/edumem/.env or .env in current working dir
    env_paths = [
        os.path.join(os.getcwd(), ".env"),
        "/opt/edumem/.env",
        os.path.join(PROJECT_ROOT, ".env")
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("export "):
                            line = line[7:].strip()
                        if "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip()
                            if val.startswith(('"', "'")) and val.endswith(val[0]):
                                val = val[1:-1]
                            if key == "HF_TOKEN" and val:
                                return val
            except Exception as e:
                print(f"Error reading {env_path}: {e}")
    return None

def install_optimum():
    print("Checking for optimum-cli...")
    # Determine local virtual env pip/optimum-cli paths
    venv_pip = os.path.join(VENV_DIR, "bin", "pip")
    venv_optimum_cli = os.path.join(VENV_DIR, "bin", "optimum-cli")
    
    # Check if optimum-cli is already available
    if os.path.exists(venv_optimum_cli) and os.access(venv_optimum_cli, os.X_OK):
        print(f"Found optimum-cli at {venv_optimum_cli}")
        return venv_optimum_cli
    
    system_optimum_cli = shutil.which("optimum-cli")
    if system_optimum_cli:
        print(f"Found system optimum-cli at {system_optimum_cli}")
        return system_optimum_cli
        
    print("optimum-cli not found. Installing optimum-intel[openvino]...")
    if os.path.exists(venv_pip):
        pip_cmd = venv_pip
    else:
        pip_cmd = shutil.which("pip3") or shutil.which("pip")
        if not pip_cmd:
            raise RuntimeError("Could not find a valid pip/pip3 executable to install dependencies.")
            
    print(f"Using pip: {pip_cmd}")
    try:
        subprocess.run([pip_cmd, "install", "optimum-intel[openvino]"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install optimum-intel[openvino]: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Re-check paths after installation
    if os.path.exists(venv_optimum_cli):
        return venv_optimum_cli
    system_optimum_cli = shutil.which("optimum-cli")
    if system_optimum_cli:
        return system_optimum_cli
        
    # Fallback to assuming we can run 'optimum-cli' from PATH or venv
    return venv_optimum_cli

def _build_export_command(optimum_cli, model_name, task, out_dir):
    cmd = [
        optimum_cli,
        "export", "openvino",
        "--model", model_name,
        "--task", task,
        "--weight-format", "fp16",
    ]
    cmd.append(out_dir)
    return cmd

def run_export(optimum_cli, model_name, task, out_dir, hf_token):
    print(f"Exporting model {model_name} (Task: {task}) to {out_dir}...")
    
    # Create output directory's parent if it doesn't exist
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    
    # Construct subprocess environment
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        # Also set HUGGING_FACE_HUB_TOKEN just in case
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
        
    cmd = _build_export_command(optimum_cli, model_name, task, out_dir)
    
    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"Successfully exported {model_name} to {out_dir}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error executing optimum-cli: {e}", file=sys.stderr)
        sys.exit(1)

def is_model_present(directory):
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return False
    try:
        files = os.listdir(directory)
        return len(files) > 0
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Download and export HuggingFace models to OpenVINO format.")
    parser.add_argument("--force", action="store_true", help="Force export even if target folders exist.")
    args = parser.parse_args()
    
    hf_token = load_hf_token()
    if not hf_token:
        print("Warning: HF_TOKEN not found in .env or environment variables. Some models might fail to download if restricted.")
    else:
        print("HF_TOKEN successfully loaded.")
        
    # Check what needs to be built
    gte_exists = is_model_present(GTE_DIR)
    ettin_exists = is_model_present(ETTIN_DIR)
    
    need_gte = args.force or not gte_exists
    need_ettin = args.force or not ettin_exists
    
    if not need_gte and not need_ettin:
        print("Both OpenVINO models (gte-modernbert-ov and ettin-17m-ov) already exist.")
        print("Skipping download and export. Use --force to overwrite.")
        sys.exit(0)
        
    # Install optimum-intel[openvino] if needed
    optimum_cli = install_optimum()
    
    if need_gte:
        if os.path.exists(GTE_DIR):
            print(f"Removing existing directory: {GTE_DIR}")
            shutil.rmtree(GTE_DIR, ignore_errors=True)
        run_export(
            optimum_cli,
            GTE_MODEL,
            "feature-extraction",
            GTE_DIR,
            hf_token,
        )
    else:
        print(f"Skipping GTE ModernBERT export (already exists at {GTE_DIR})")
        
    if need_ettin:
        if os.path.exists(ETTIN_DIR):
            print(f"Removing existing directory: {ETTIN_DIR}")
            shutil.rmtree(ETTIN_DIR, ignore_errors=True)
        run_export(optimum_cli, ETTIN_MODEL, "text-classification", ETTIN_DIR, hf_token)
    else:
        print(f"Skipping Ettin Reranker export (already exists at {ETTIN_DIR})")
        
    print("Model preparation completed successfully!")

if __name__ == "__main__":
    main()

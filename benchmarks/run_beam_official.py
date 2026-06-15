#!/usr/bin/env python3
"""Official BEAM Benchmark Runner for edumem with Provider Selection.

Allows running with local Qwen3.6 proxy (NAN Builders), official OpenAI,
Google Gemini (OpenAI-compatible), or models via OpenRouter.
"""
import os
import sys
import argparse
from pathlib import Path
import subprocess

WORKDIR = Path("/opt/edumem")

def load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k] = v.strip().strip('"').strip("'")
    return env

def main():
    parser = argparse.ArgumentParser(description="Official edumem BEAM Evaluation Runner")
    parser.add_argument("--provider", default="nan", choices=["nan", "openai", "openrouter", "gemini"], 
                        help="LLM provider: nan (local Qwen), openai (official ChatGPT), openrouter (OpenRouter), or gemini (Google)")
    parser.add_argument("--model", help="Answering/Judging model (defaults: qwen3.6 for nan, gpt-4o for openai, openai/gpt-4o for openrouter, gemini-2.5-flash for gemini)")
    parser.add_argument("--api-key", help="Override API key for the selected provider")
    parser.add_argument("--scales", default="100K", help="BEAM Scales (100K, 500K, 1M, 10M)")
    parser.add_argument("--sample", type=int, default=1, help="Conversations per scale")
    parser.add_argument("--dry-run", action="store_true", help="Download dataset and print stats, don't run")
    args = parser.parse_args()

    # Load local .env
    dot_env = load_env(WORKDIR / ".env")
    
    # Resolve provider details
    if args.provider == "gemini":
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        model = args.model or "gemini-2.5-flash"
        api_key = (
            args.api_key or 
            os.getenv("GOOGLE_API_KEY") or 
            dot_env.get("GOOGLE_API_KEY")
        )
        if not api_key:
            print("[!] Error: No Gemini API key found in env or .env (expected GOOGLE_API_KEY)")
            sys.exit(1)

    elif args.provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
        model = args.model or "openai/gpt-4o-mini"
        api_key = (
            args.api_key or 
            os.getenv("OPENROUTER_API_KEY") or 
            dot_env.get("OPENROUTER_API_KEY")
        )
        if not api_key:
            print("[!] Error: No OpenRouter API key found in env or .env (expected OPENROUTER_API_KEY)")
            sys.exit(1)
            
    elif args.provider == "openai":
        base_url = "https://api.openai.com/v1"
        model = args.model or "gpt-4o"
        api_key = (
            args.api_key or 
            os.getenv("OPENAI_API_KEY") or 
            dot_env.get("OPENAI_API_KEY")
        )
        if not api_key:
            print("[!] Error: No OpenAI API key found in env or .env (expected OPENAI_API_KEY)")
            sys.exit(1)
            
    else:
        # Default NAN provider
        base_url = dot_env.get("CHAT_MODEL_BASE_URL", "https://api.nan.builders/v1")
        model = args.model or "qwen3.6"
        api_key = (
            args.api_key or 
            os.getenv("NAN_APY_KEY") or 
            os.getenv("NAN_API_KEY") or 
            dot_env.get("NAN_APY_KEY") or 
            dot_env.get("NAN_API_KEY")
        )
        if not api_key:
            print("[!] Error: No NAN Builders API key found in env or .env (expected NAN_APY_KEY)")
            sys.exit(1)

    # Core environment mapping for evaluate_beam_end_to_end.py
    env_overrides = {
        "OPENROUTER_API_KEY": api_key,
        "OPENROUTER_BASE_URL": base_url,
        # Enable local_llm.py remote summarization/consolidation workers
        "EDUMEM_LLM_BASE_URL": base_url,
        "EDUMEM_LLM_API_KEY": api_key,
        "EDUMEM_LLM_MODEL": model,
        "EDUMEM_LLM_ENABLED": "true",
        "PYTHONPATH": str(WORKDIR / "benchmarks")
    }

    python_env = WORKDIR / ".venv" / "bin" / "python3"
    script_path = WORKDIR / ".venv/lib/python3.13/site-packages/tools/evaluate_beam_end_to_end.py"

    cmd = [
        str(python_env),
        str(script_path),
        "--scales", args.scales,
        "--sample", str(args.sample),
        "--model", model,
        "--pure-recall"
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    print("======================================================================")
    print("🎯 STARTING OFFICIAL BEAM EVALUATION RUNNER")
    print(f"[*] Provider: {args.provider.upper()}")
    print(f"[*] Scales: {args.scales} (Sample Size: {args.sample})")
    print(f"[*] Model: {model}")
    print(f"[*] Base URL: {base_url}")
    print("======================================================================\n")

    env = os.environ.copy()
    env.update(env_overrides)

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[!] BEAM evaluation exited with non-zero code: {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()

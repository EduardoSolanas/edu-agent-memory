#!/usr/bin/env python3
"""Official BEAM benchmark runner for edumem.

This runner keeps the official BEAM invocation in one place, wires the shipped
reranker and embedding endpoint, and fails fast when the expected local
services are not healthy.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib import request

WORKDIR = Path(__file__).resolve().parents[1]
SHIPPED_RERANKER_URL = "http://localhost:3002/rerank"
SHIPPED_EMBEDDING_BASE_URL = "http://localhost:3002"
SHIPPED_EMBEDDING_MODEL = "Alibaba-NLP/gte-modernbert-base"
SHIPPED_EMBEDDING_DIM = "768"


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key] = value.strip().strip('"').strip("'")
    return env


def _http_json(url: str, payload: dict | None = None) -> object:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, headers=headers)
    with request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _preflight_embedding(embedding_base_url: str, embedding_model: str, embedding_dim: int) -> None:
    health_url = embedding_base_url.rstrip("/") + "/health"
    embedding_url = embedding_base_url.rstrip("/") + "/v1/embeddings"
    try:
        health = _http_json(health_url)
        if not isinstance(health, dict) or health.get("status") not in {"ok", "healthy"}:
            raise ValueError(f"unexpected health payload: {health}")
        payload = {"input": ["beam preflight"], "model": embedding_model}
        data = _http_json(embedding_url, payload)
        vector = data["data"][0]["embedding"]
        if len(vector) != embedding_dim:
            raise ValueError(f"expected {embedding_dim}-dimensional embeddings, got {len(vector)}")
    except Exception as exc:
        raise RuntimeError(
            f"embedding preflight failed for {embedding_url} "
            f"(model={embedding_model}, dim={embedding_dim}): {exc}"
        ) from exc


def _preflight_reranker(reranker_url: str) -> None:
    reranker_base = reranker_url.rsplit("/rerank", 1)[0].rstrip("/")
    health_url = reranker_base + "/health" if reranker_base else None
    try:
        if health_url and health_url != reranker_url:
            health = _http_json(health_url)
            if not isinstance(health, dict) or health.get("status") not in {"ok", "healthy"}:
                raise ValueError(f"unexpected health payload: {health}")
        data = _http_json(
            reranker_url,
            {"query": "beam preflight", "texts": ["alpha", "beta"]},
        )
        if not isinstance(data, list) or not data:
            raise ValueError("reranker returned no scored results")
        if not all(isinstance(item, dict) and "index" in item and "score" in item for item in data):
            raise ValueError("reranker response shape was invalid")
    except Exception as exc:
        raise RuntimeError(f"reranker preflight failed for {reranker_url}: {exc}") from exc


def _env_value(dot_env: dict[str, str], key: str, default: str) -> str:
    return os.environ.get(key) or dot_env.get(key) or default


def _resolve_embedding_settings(dot_env: dict[str, str]) -> tuple[str, str, int]:
    embedding_base_url = _env_value(dot_env, "EDUMEM_EMBEDDING_API_URL", SHIPPED_EMBEDDING_BASE_URL)
    embedding_model = _env_value(dot_env, "EDUMEM_EMBEDDING_MODEL", SHIPPED_EMBEDDING_MODEL)
    embedding_dim = int(_env_value(dot_env, "EDUMEM_EMBEDDING_DIM", SHIPPED_EMBEDDING_DIM))
    return embedding_base_url, embedding_model, embedding_dim


def _build_runner_env(
    dot_env: dict[str, str],
    base_url: str,
    api_key: str,
    model: str,
    judge_model: str,
    reranker_url: str,
    embedding_base_url: str,
    embedding_model: str,
    embedding_dim: int,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(dot_env)
    env.update(
        {
            "OPENROUTER_API_KEY": api_key,
            "OPENROUTER_BASE_URL": base_url,
            "EDUMEM_LLM_BASE_URL": base_url,
            "EDUMEM_LLM_API_KEY": api_key,
            "EDUMEM_LLM_MODEL": model,
            "EDUMEM_LLM_ENABLED": "true",
            "EDUMEM_JUDGE_MODEL": judge_model,
            "EDUMEM_RERANKER_URL": reranker_url,
            "EDUMEM_EMBEDDING_API_URL": embedding_base_url,
            "EDUMEM_EMBEDDING_MODEL": embedding_model,
            "EDUMEM_EMBEDDING_DIM": str(embedding_dim),
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{WORKDIR}:{WORKDIR / 'benchmarks'}",
        }
    )
    return env


def _build_evaluator_command(
    script_path: Path,
    model: str,
    judge_model: str,
    scales: str,
    sample: int,
    dry_run: bool,
    *,
    start_index: int = 0,
    case_index: int | None = None,
    output_dir: Path | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--scales",
        scales,
        "--sample",
        str(sample),
        "--model",
        model,
        "--judge-model",
        judge_model,
        "--pure-recall",
    ]
    if start_index != 0:
        cmd.extend(["--start-index", str(start_index)])
    if case_index is not None:
        cmd.extend(["--case-index", str(case_index)])
    if dry_run:
        cmd.append("--dry-run")
    if output_dir is not None:
        cmd.extend(["--output-dir", str(output_dir)])
    return cmd


def _sample_selection_message(sample: int, case_index: int | None, start_index: int) -> str | None:
    if sample == 1 and case_index is None:
        return (
            "--sample is a prefix count, so --sample 1 selects the first "
            f"conversation from --start-index={start_index}. Use --case-index "
            "to target one exact conversation."
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Official edumem BEAM Evaluation Runner")
    parser.add_argument(
        "--provider",
        default="nan",
        choices=["nan", "openai", "openrouter", "gemini"],
        help="LLM provider: nan (DeepSeek via NAN), openai (official ChatGPT), openrouter (OpenRouter), or gemini (Google)",
    )
    parser.add_argument(
        "--model",
        help="Answering model (defaults: deepseek-v4-flash for nan, gpt-4o for openai, openai/gpt-4o for openrouter, gemini-2.5-flash for gemini)",
    )
    parser.add_argument("--judge-model", help="Judge model (defaults to --model)")
    parser.add_argument("--api-key", help="Override API key for the selected provider")
    parser.add_argument("--scales", default="100K", help="BEAM scales (100K, 500K, 1M, 10M)")
    parser.add_argument("--sample", type=int, default=1, help="Prefix count of conversations per scale from --start-index")
    parser.add_argument("--start-index", type=int, default=0, help="Prefix start index for deterministic conversation selection")
    parser.add_argument("--case-index", type=int, default=None, help="Select exactly one conversation by index per scale")
    parser.add_argument("--dry-run", action="store_true", help="Download dataset and print stats, don't run")
    parser.add_argument(
        "--allow-no-reranker",
        action="store_true",
        help="Continue when the shipped reranker is unavailable",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for results. Defaults to results/<timestamp>_<model>/")
    args = parser.parse_args()

    dot_env = load_env(WORKDIR / ".env")

    if args.provider == "gemini":
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        model = args.model or "gemini-2.5-flash"
        api_key = args.api_key or os.getenv("GOOGLE_API_KEY") or dot_env.get("GOOGLE_API_KEY")
        if not api_key:
            print("[!] Error: No Gemini API key found in env or .env (expected GOOGLE_API_KEY)")
            sys.exit(1)
    elif args.provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
        model = args.model or "openai/gpt-4o-mini"
        api_key = args.api_key or os.getenv("OPENROUTER_API_KEY") or dot_env.get("OPENROUTER_API_KEY")
        if not api_key:
            print("[!] Error: No OpenRouter API key found in env or .env (expected OPENROUTER_API_KEY)")
            sys.exit(1)
    elif args.provider == "openai":
        base_url = "https://api.openai.com/v1"
        model = args.model or "gpt-4o"
        api_key = args.api_key or os.getenv("OPENAI_API_KEY") or dot_env.get("OPENAI_API_KEY")
        if not api_key:
            print("[!] Error: No OpenAI API key found in env or .env (expected OPENAI_API_KEY)")
            sys.exit(1)
    else:
        base_url = dot_env.get("CHAT_MODEL_BASE_URL", "https://api.nan.builders/v1")
        model = args.model or "deepseek-v4-flash"
        api_key = (
            args.api_key
            or os.getenv("NAN_API_KEY")
            or os.getenv("NAN_APY_KEY")
            or dot_env.get("NAN_API_KEY")
            or dot_env.get("NAN_APY_KEY")
        )
        if not api_key:
            print("[!] Error: No NAN Builders API key found in env or .env (expected NAN_API_KEY)")
            sys.exit(1)

    judge_model = args.judge_model or model
    reranker_url = _env_value(dot_env, "EDUMEM_RERANKER_URL", SHIPPED_RERANKER_URL)
    embedding_base_url, embedding_model, embedding_dim = _resolve_embedding_settings(dot_env)

    env = _build_runner_env(
        dot_env,
        base_url=base_url,
        api_key=api_key,
        model=model,
        judge_model=judge_model,
        reranker_url=reranker_url,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )

    try:
        _preflight_embedding(embedding_base_url, embedding_model, embedding_dim)
        print(
            f"[preflight] embedding {embedding_base_url.rstrip('/')}/v1/embeddings "
            f"model={embedding_model} dim={embedding_dim} status=healthy"
        )
    except RuntimeError as exc:
        print(f"[preflight] embedding {embedding_base_url.rstrip('/')}/v1/embeddings status=unhealthy")
        print(f"[!] {exc}")
        sys.exit(1)

    try:
        _preflight_reranker(reranker_url)
        print(f"[preflight] reranker {reranker_url} status=healthy")
    except RuntimeError as exc:
        print(f"[preflight] reranker {reranker_url} status=unhealthy")
        if not args.allow_no_reranker:
            print(f"[!] {exc}")
            sys.exit(1)
        print(f"[*] Warning: {exc} (continuing because --allow-no-reranker was set)")

    script_path = WORKDIR / "tools/evaluate_beam_end_to_end.py"
    cmd = _build_evaluator_command(
        script_path,
        model,
        judge_model,
        args.scales,
        args.sample,
        args.dry_run,
        start_index=args.start_index,
        case_index=args.case_index,
        output_dir=args.output_dir,
    )

    print("======================================================================")
    print("STARTING OFFICIAL BEAM EVALUATION RUNNER")
    print(f"[*] Provider: {args.provider.upper()}")
    print(f"[*] Scales: {args.scales} (Prefix Count: {args.sample})")
    sample_message = _sample_selection_message(args.sample, args.case_index, args.start_index)
    if sample_message:
        print(f"[*] Note: {sample_message}")
    print(f"[*] Model: {model}")
    print(f"[*] Judge Model: {judge_model}")
    print(f"[*] Base URL: {base_url}")
    print(f"[*] Embedding: {embedding_base_url.rstrip('/')}/v1/embeddings")
    print(f"[*] Reranker: {reranker_url}")
    if args.output_dir:
        print(f"[*] Output Dir: {args.output_dir}")
    print("======================================================================\n")

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"\n[!] BEAM evaluation exited with non-zero code: {exc.returncode}")
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()

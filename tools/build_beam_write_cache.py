#!/usr/bin/env python3
"""Build a generated write-cache artifact for one BEAM conversation.

This captures the LLM-generated write-side payloads during ingest, stores them
as JSON artifacts, and derives a stable replay contract that tests can assert
against later without calling the live models again.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        for prefix in ("NAN_APY_KEY=", "EDUMEM_LLM_API_KEY="):
            if line.startswith(prefix):
                key = line[len(prefix):].strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault("EDUMEM_LLM_API_KEY", key)
                break


def _configure_env() -> None:
    _load_env_file()
    os.environ.setdefault("EDUMEM_LLM_BASE_URL", "https://api.nan.builders/v1")
    os.environ.setdefault("EDUMEM_LLM_MODEL", "qwen3.6")
    os.environ.setdefault("EDUMEM_EXTRACTION_MODEL", os.environ["EDUMEM_LLM_MODEL"])
    os.environ.setdefault("EDUMEM_EXTRACTION_TIMEOUT", "300")
    os.environ.setdefault("EDUMEM_EMBEDDING_API_URL", "http://127.0.0.1:3002")
    os.environ.setdefault("EDUMEM_EMBEDDING_MODEL", "Alibaba-NLP/gte-modernbert-base")
    os.environ.setdefault("EDUMEM_EMBEDDINGS_VIA_API", "1")
    os.environ.setdefault("EDUMEM_RERANKER_URL", "http://127.0.0.1:3002/rerank")
    os.environ.setdefault("EDUMEM_LLM_FACT_CONSOLIDATION", "1")


_configure_env()


from tools.beam_write_cache import build_generated_write_cache_for_conversation  # noqa: E402
from tools.evaluate_beam_end_to_end import load_beam_dataset  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a generated BEAM write-cache artifact for one conversation.",
    )
    parser.add_argument(
        "--scale",
        default="100K",
        help="BEAM dataset scale to load (default: 100K).",
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=0,
        help="Zero-based conversation index inside the selected scale (default: 0).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output cache directory (default: tests/.beam_write_cache/beam_<scale>_conv<index>).",
    )
    return parser.parse_args()


def _default_out_dir(scale: str, case_index: int) -> Path:
    scale_slug = scale.replace("/", "_")
    return PROJECT_ROOT / "tests" / ".beam_write_cache" / f"beam_{scale_slug}_conv{case_index}"


def main() -> int:
    args = _parse_args()
    out_dir = args.out_dir or _default_out_dir(args.scale, args.case_index)

    if out_dir.exists():
        print(f"[write-cache] ERROR: output directory already exists: {out_dir}")
        print("[write-cache] Remove it first if you want to rebuild this cache.")
        return 1

    print(f"[write-cache] Loading BEAM {args.scale} up to case {args.case_index}...")
    data = load_beam_dataset([args.scale], max_conversations=args.case_index + 1)
    conversations = data.get(args.scale, [])
    if args.case_index >= len(conversations):
        print(
            f"[write-cache] ERROR: case index {args.case_index} is out of range "
            f"for scale {args.scale} ({len(conversations)} loaded)."
        )
        return 1

    conversation = conversations[args.case_index]
    messages = conversation.get("messages", [])
    conversation_id = str(conversation.get("id", args.case_index))
    session_id = f"write-cache-{args.scale.lower()}-conv{args.case_index}"
    model = os.environ["EDUMEM_LLM_MODEL"]

    print(f"[write-cache] Output dir : {out_dir}")
    print(f"[write-cache] Model      : {model}")
    print(f"[write-cache] Base URL   : {os.environ['EDUMEM_LLM_BASE_URL']}")
    print(f"[write-cache] Embed URL  : {os.environ['EDUMEM_EMBEDDING_API_URL']}")
    print(f"[write-cache] Reranker   : {os.environ['EDUMEM_RERANKER_URL']}")
    print(f"[write-cache] Session    : {session_id}")
    print(f"[write-cache] Conversation: {conversation_id}")
    print(f"[write-cache] Messages   : {len(messages)}")
    print(f"[write-cache] Questions  : {len(conversation.get('questions', []))}")
    print(f"[write-cache] Key set    : {'yes' if os.environ.get('EDUMEM_LLM_API_KEY') else 'NO - will fail'}")

    result = build_generated_write_cache_for_conversation(
        messages,
        out_dir,
        session_id=session_id,
        conversation_id=conversation_id,
        scale=args.scale,
        llm_model=model,
    )

    print(f"[write-cache] Wrote cache to {result['cache_dir']}")
    print(f"[write-cache] Recorded operations: {result['operation_count']}")
    print(f"[write-cache] Final DB   : {result['final_db']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

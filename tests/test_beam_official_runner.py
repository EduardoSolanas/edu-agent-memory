from __future__ import annotations

import socket
from pathlib import Path

import pytest

from benchmarks import run_beam_official as runner


def _closed_local_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_runner_helpers_build_expected_env_and_command():
    embedding_base_url, embedding_model, embedding_dim = runner._resolve_embedding_settings({})
    assert embedding_base_url == "http://127.0.0.1:3002"
    assert embedding_model == "Alibaba-NLP/gte-modernbert-base"
    assert embedding_dim == 768

    env = runner._build_runner_env(
        {},
        base_url="https://api.nan.builders/v1",
        api_key="test-nan-key",
        model="deepseek-v4-flash",
        judge_model="deepseek-v4-flash",
        reranker_url="http://127.0.0.1:3002/rerank",
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )
    assert env["OPENROUTER_BASE_URL"] == "https://api.nan.builders/v1"
    assert env["EDUMEM_LLM_MODEL"] == "deepseek-v4-flash"
    assert env["EDUMEM_JUDGE_MODEL"] == "deepseek-v4-flash"
    assert env["EDUMEM_RERANKER_URL"] == "http://127.0.0.1:3002/rerank"
    assert env["EDUMEM_EMBEDDING_MODEL"] == "Alibaba-NLP/gte-modernbert-base"
    assert env["EDUMEM_EMBEDDING_DIM"] == "768"

    command = runner._build_evaluator_command(
        Path("/tmp/tools/evaluate_beam_end_to_end.py"),
        model="deepseek-v4-flash",
        judge_model="deepseek-v4-flash",
        scales="100K",
        sample=1,
        dry_run=True,
        start_index=4,
        case_index=7,
    )
    assert command[0].endswith("python.exe") or command[0].endswith("python") or command[0].endswith("python3")
    assert command[1] == "-u"
    assert command[2].endswith("evaluate_beam_end_to_end.py")
    assert command[command.index("--model") + 1] == "deepseek-v4-flash"
    assert command[command.index("--judge-model") + 1] == "deepseek-v4-flash"
    assert "--pure-recall" in command
    assert command[command.index("--start-index") + 1] == "4"
    assert command[command.index("--case-index") + 1] == "7"
    assert "--dry-run" in command
    assert "prefix count" in runner._sample_selection_message(1, None, 0).lower()
    assert runner._sample_selection_message(1, 7, 4) is None


def test_runner_fails_on_closed_local_ports():
    closed_port = _closed_local_port()
    embedding_url = f"http://127.0.0.1:{closed_port}"
    reranker_url = f"{embedding_url}/rerank"

    with pytest.raises(RuntimeError, match="embedding preflight failed"):
        runner._preflight_embedding(embedding_url, "Alibaba-NLP/gte-modernbert-base", 768)

    with pytest.raises(RuntimeError, match="reranker preflight failed"):
        runner._preflight_reranker(reranker_url)

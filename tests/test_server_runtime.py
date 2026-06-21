import ast
from pathlib import Path

from server_text import sanitize_rerank_text


ROOT = Path(__file__).resolve().parents[1]


def test_native_inference_routes_are_synchronous():
    tree = ast.parse((ROOT / "server.py").read_text(encoding="utf-8"))
    async_functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
    }

    assert "openai_embeddings" not in async_functions
    assert "rerank" not in async_functions
    assert "chat_completions" not in async_functions


def test_sanitizer_maps_empty_and_whitespace_to_safe_value():
    assert sanitize_rerank_text("") == "empty"
    assert sanitize_rerank_text(" \n\t ") == "empty"


def test_sanitizer_collapses_whitespace_and_pathological_repetition():
    assert sanitize_rerank_text("alpha\n\t beta") == "alpha beta"
    assert sanitize_rerank_text("aaaaaaaaaa ========== z") == "aa == z"


def test_sanitizer_preserves_short_character_runs_in_values_and_identifiers():
    text = "1000 coool version_1111111"

    assert sanitize_rerank_text(text) == text


def test_sanitizer_bounds_utf8_bytes_and_preserves_head_and_tail():
    text = "HEAD " + ("middle " * 100) + "TAIL"
    result = sanitize_rerank_text(text, max_utf8_bytes=80)

    assert len(result.encode("utf-8")) <= 80
    assert result.startswith("HEAD")
    assert result.endswith("TAIL")
    assert " ... " in result


def test_sanitizer_preserves_both_ends_of_long_identifier():
    identifier = "prefix_" + ("Ab9x" * 500) + "_suffix"
    result = sanitize_rerank_text(identifier, max_utf8_bytes=96)

    assert len(result.encode("utf-8")) <= 96
    assert result.startswith("prefix_")
    assert result.endswith("_suffix")


def test_sanitizer_bounds_random_subword_like_unicode():
    text = "漢字Ab9_" * 2000
    result = sanitize_rerank_text(text, max_utf8_bytes=257)

    assert len(result.encode("utf-8")) <= 257
    assert result != "empty"
    assert result.endswith("Ab9_")


def test_sanitizer_rejects_a_budget_too_small_for_head_and_tail():
    import pytest

    with pytest.raises(ValueError, match="at least 7"):
        sanitize_rerank_text("long enough to truncate", max_utf8_bytes=6)

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

import server_nvidia as nvidia

ROOT = Path(__file__).resolve().parents[1]


def test_linear_layer_norm_and_gelu_helpers_work_together():
    values = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)

    linear = nvidia._linear(
        values,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        np.array([0.5, -0.5], dtype=np.float32),
    )
    assert np.allclose(linear, np.array([[1.5, -2.5]], dtype=np.float32))

    gelu = nvidia._gelu(np.array([[-1.0, 0.0, 1.0]], dtype=np.float32))
    assert gelu.shape == (1, 3)
    assert gelu[0, 1] == 0.0
    assert gelu[0, 2] > 0.8

    normed = nvidia._layer_norm(
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        eps=0.0,
    )
    assert np.allclose(normed.mean(axis=-1), np.array([0.0], dtype=np.float32), atol=1e-6)


def test_sigmoid_maps_logits_into_probability_range():
    values = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    scores = nvidia._sigmoid(values)

    assert scores[0] < 0.001
    assert scores[1] == 0.5
    assert scores[2] > 0.999


def test_sanitize_rerank_inputs_bounds_query_and_documents():
    query = "HEAD " + ("middle " * 100) + "TAIL"
    texts = [
        "DOC " + ("payload " * 100) + "TAIL",
        " \n\t ",
    ]

    safe_query, safe_texts = nvidia._sanitize_rerank_inputs(
        query,
        texts,
        query_max_utf8_bytes=48,
        text_max_utf8_bytes=64,
    )

    assert len(safe_query.encode("utf-8")) <= 48
    assert safe_query.startswith("HEAD")
    assert safe_query.endswith("TAIL")
    assert len(safe_texts[0].encode("utf-8")) <= 64
    assert safe_texts[0].startswith("DOC")
    assert safe_texts[0].endswith("TAIL")
    assert safe_texts[1] == "empty"


def test_iter_rerank_batches_groups_similar_lengths_and_caps_batch_size():
    texts = [
        "alpha " * 200,
        "bravo " * 180,
        "charlie " * 24,
        "delta " * 20,
        "echo " * 4,
    ]

    batches = list(nvidia._iter_rerank_batches(texts, batch_size=2))

    assert batches == [
        ([0, 1], [texts[0], texts[1]]),
        ([2, 3], [texts[2], texts[3]]),
        ([4], [texts[4]]),
    ]


def test_onnx_reranker_score_uses_sanitization_and_batch_helpers():
    tree = ast.parse((ROOT / "server_nvidia.py").read_text(encoding="utf-8"))
    reranker_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OnnxReranker"
    )
    score_fn = next(
        node for node in reranker_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "score"
    )

    called = set()
    for node in ast.walk(score_fn):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)

    assert "_sanitize_rerank_inputs" in called
    assert "_iter_rerank_batches" in called

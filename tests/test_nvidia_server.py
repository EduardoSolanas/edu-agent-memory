from __future__ import annotations

import numpy as np

import server_nvidia as nvidia


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

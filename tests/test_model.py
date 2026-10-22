"""CNN forward pass, probabilities, checkpoint reload, and CSP features."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from mindmove.baseline import build_csp_lda
from mindmove.data import EpochTable
from mindmove.inference import binary_cursor_offset, load_cnn_checkpoint, predict_baseline, predict_cnn, save_cnn_checkpoint
from mindmove.models import EEGCNN
from mindmove.train import fit_cnn


def test_forward_shape_and_probabilities():
    torch.manual_seed(0)
    model = EEGCNN().eval()
    batch = torch.randn(4, 64, 320)
    logits = model(batch)
    assert logits.shape == (4, 2)
    probabilities = torch.softmax(logits, dim=-1)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_checkpoint_reload_reproduces_predictions(tmp_path):
    torch.manual_seed(42)
    model = EEGCNN()
    model.train()
    features = torch.randn(8, 64, 320)
    labels = torch.randint(0, 2, (8,))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.functional.cross_entropy(model(features), labels)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(features[:4])
    path = tmp_path / "cnn.pt"
    mean = np.zeros((1, 64, 1), dtype=np.float32)
    std = np.ones((1, 64, 1), dtype=np.float32)
    save_cnn_checkpoint(
        path,
        model,
        {
            "norm_mean": mean,
            "norm_std": std,
            "train_trial_ids": ["S001_R04_T000"],
            "history": [],
            "best_epoch": 1,
        },
    )
    loaded, payload = load_cnn_checkpoint(path, "cpu")
    with torch.no_grad():
        actual = loaded(features[:4])
    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)
    prediction, proba = predict_cnn(
        loaded,
        features[:4].numpy(),
        payload["norm_mean"],
        payload["norm_std"],
        "cpu",
    )
    assert prediction.shape == (4,)
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    with torch.no_grad():
        manual = torch.softmax(loaded(features[:4]), dim=-1).numpy()
    assert np.allclose(manual, proba, atol=1e-5)


def test_inference_functions_do_not_take_labels():
    for function in (predict_cnn, predict_baseline):
        names = set(inspect.signature(function).parameters)
        assert names.isdisjoint({"y", "y_true", "labels", "target", "ground_truth"})


def test_cursor_offset_uses_only_probabilities():
    assert binary_cursor_offset([0.2, 0.8]) == pytest.approx(0.6)
    assert binary_cursor_offset([0.5, 0.5]) == pytest.approx(0.0)
    assert binary_cursor_offset([1.0, 0.0]) == pytest.approx(-1.0)


def test_csp_lda_learns_a_spatial_difference_without_labels_at_predict_time():
    train_x, train_y = _separable_epochs(60, seed=0)
    test_x, test_y = _separable_epochs(40, seed=1)
    pipeline = build_csp_lda({"model": {"csp_components": 4, "csp_reg": "ledoit_wolf"}})
    pipeline.fit(train_x, train_y)
    features = pipeline.named_steps["csp"].transform(train_x[:3])
    assert features.shape == (3, 4)
    prediction, proba = predict_baseline(pipeline, test_x)
    assert proba.shape == (len(test_y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    accuracy = float(np.mean(prediction == test_y))
    assert accuracy >= 0.8


def test_training_loop_normalizes_with_training_ids_only(tmp_path):
    cfg = {
        "training": {
            "seed": 42,
            "batch_size": 8,
            "learning_rate": 0.001,
            "epochs": 2,
            "patience": 2,
            "device": "cpu",
        },
        "model": {
            "type": "cnn1d",
            "num_classes": 2,
            "dropout": 0.0,
            "conv1_channels": 8,
            "conv1_kernel": 5,
            "conv2_channels": 8,
            "conv2_kernel": 3,
            "pool_kernel": 2,
            "csp_components": 4,
            "csp_reg": "ledoit_wolf",
        },
    }
    train = _tiny_table(subjects=[1], runs=[4], n_per=8, seed=0)
    val = _tiny_table(subjects=[1], runs=[8], n_per=8, seed=1)
    result = fit_cnn(train, val, cfg, tmp_path)
    loaded, payload = load_cnn_checkpoint(Path(result["path"]), "cpu")
    assert set(payload["train_trial_ids"]) == set(train.trial_ids().tolist())
    assert set(payload["train_trial_ids"]).isdisjoint(val.trial_ids().tolist())
    stored = np.load(tmp_path / "norm_stats.npz")
    assert set(stored["train_trial_ids"].tolist()) == set(train.trial_ids().tolist())
    again, _payload = predict_cnn(loaded, train.data[:4], payload["norm_mean"], payload["norm_std"], "cpu")
    assert again.shape == (4,)


def _separable_epochs(n_epochs: int, seed: int):
    rng = np.random.default_rng(seed)
    half = n_epochs // 2
    labels = np.array([0] * half + [1] * (n_epochs - half), dtype=np.int64)
    data = rng.normal(scale=0.4, size=(n_epochs, 8, 128)).astype(np.float64)
    for index, label in enumerate(labels):
        channel = 0 if label == 0 else 1
        data[index, channel] += rng.normal(scale=3.0, size=128)
    return data, labels


def _tiny_table(subjects, runs, n_per: int, seed: int) -> EpochTable:
    rng = np.random.default_rng(seed)
    parts = []
    for subject in subjects:
        for run in runs:
            labels = np.arange(n_per) % 2
            data = rng.normal(scale=1.0, size=(n_per, 8, 64)).astype(np.float32)
            parts.append(
                EpochTable(
                    data=data,
                    labels=labels.astype(np.int64),
                    subjects=np.full(n_per, subject),
                    runs=np.full(n_per, run),
                    trials=np.arange(n_per),
                    onsets=np.arange(n_per, dtype=float) * 8.0,
                    times=np.arange(64) / 160.0 + 1.0,
                    ch_names=[f"CH{i}" for i in range(8)],
                    sfreq=160.0,
                )
            )
    return parts[0]


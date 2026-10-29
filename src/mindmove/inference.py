"""Load saved decoders and predict without ground-truth labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mindmove.models import EEGCNN
from mindmove.preprocessing import apply_channel_norm


def save_cnn_checkpoint(path: Path, model: EEGCNN, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "spec": dict(model.spec),
    }
    payload.update(extra)
    torch.save(payload, path)


def load_cnn_checkpoint(path: Path, device) -> tuple[EEGCNN, dict]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model = EEGCNN(**payload["spec"])
    model.load_state_dict(payload["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model, payload


def predict_cnn(model: EEGCNN, data, norm_mean, norm_std, device, batch_size: int = 64):
    """Class index and probabilities for each epoch.

    ``data`` is (epochs, channels, time) or (channels, time). Labels are not
    an argument. Normalization uses the supplied training statistics.
    """
    batch = _as_batch(data)
    mean, std = _norm_arrays(norm_mean, norm_std)
    normalized = apply_channel_norm(batch, mean, std)
    model.eval()
    device = torch.device(device)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(normalized), int(batch_size)):
            tensor = torch.from_numpy(normalized[start : start + int(batch_size)]).to(device)
            probabilities = torch.softmax(model(tensor), dim=-1).cpu().numpy()
            chunks.append(probabilities)
    proba = np.concatenate(chunks, axis=0)
    prediction = proba.argmax(axis=1).astype(np.int64)
    return prediction, proba


def predict_baseline(pipeline, data):
    """Class index and probabilities from a fitted CSP + LDA pipeline.

    Labels are not an argument.
    """
    batch = _as_batch(data).astype(np.float64, copy=False)
    raw_proba = pipeline.predict_proba(batch)
    proba = align_binary_proba(pipeline.named_steps["lda"].classes_, raw_proba)
    prediction = proba.argmax(axis=1).astype(np.int64)
    return prediction, proba


def align_binary_proba(classes, proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    aligned = np.zeros((proba.shape[0], 2), dtype=np.float64)
    for column, class_index in enumerate(int(item) for item in classes):
        if class_index not in (0, 1):
            raise ValueError(f"Unexpected class index {class_index}.")
        aligned[:, class_index] = proba[:, column]
    return aligned


def binary_cursor_offset(proba) -> float:
    """Map class probabilities to a cursor in [-1, 1]. Positive is right."""
    values = np.asarray(proba, dtype=float)
    if values.shape != (2,):
        raise ValueError(f"Expected two class probabilities, got {values.shape}.")
    return float(np.clip(values[1] - values[0], -1.0, 1.0))


def assert_same_training_ids(saved_ids, current_ids, what: str) -> None:
    saved = set(str(item) for item in saved_ids)
    current = set(str(item) for item in current_ids)
    if saved != current:
        raise RuntimeError(
            f"{what} was fit on a different training split than the current config. "
            "Retrain before evaluation so normalization and the decoder match these trials."
        )


def _as_batch(data) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected EEG shaped (epochs, channels, time), got {array.shape}.")
    return np.ascontiguousarray(array)


def _norm_arrays(mean, std) -> tuple[np.ndarray, np.ndarray]:
    center = _to_numpy(mean).astype(np.float32)
    scale = _to_numpy(std).astype(np.float32)
    if center.ndim == 1:
        center = center.reshape(1, -1, 1)
    if scale.ndim == 1:
        scale = scale.reshape(1, -1, 1)
    return center, scale


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)

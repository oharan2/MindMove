"""Matplotlib figures for evaluation and the dashboard."""

from __future__ import annotations

import matplotlib

try:
    matplotlib.use("Agg")
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from mindmove.preprocessing import CLASS_NAMES, SENSORIMOTOR_CHANNELS


def choose_display_channels(ch_names, preferred=SENSORIMOTOR_CHANNELS) -> list[str]:
    found = [name for name in preferred if name in ch_names]
    if found:
        return found
    return list(ch_names[:3])


def plot_training_curves(history: list[dict], best_epoch: int):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    epochs = [row["epoch"] for row in history]
    ax.plot(epochs, [row["train_loss"] for row in history], label="Train", marker="o", ms=3)
    ax.plot(epochs, [row["val_loss"] for row in history], label="Validation", marker="o", ms=3)
    if best_epoch:
        ax.axvline(best_epoch, color="0.4", ls="--", label=f"Best epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("CNN training and validation loss")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_model_comparison(rows: list[dict]):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = [row["name"] for row in rows]
    positions = np.arange(len(names))
    width = 0.36
    ax.bar(positions - width / 2, [row["accuracy"] for row in rows], width, label="Accuracy")
    ax.bar(
        positions + width / 2,
        [row["balanced_accuracy"] for row in rows],
        width,
        label="Balanced accuracy",
    )
    ax.axhline(0.5, color="0.35", ls="--", lw=1, label="Nominal chance 0.50")
    ax.set_xticks(positions, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Held-out test scores")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrices(matrices: dict[str, list]):
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.4 * len(matrices), 4.0))
    if len(matrices) == 1:
        axes = [axes]
    for ax, (name, matrix) in zip(axes, matrices.items()):
        values = np.asarray(matrix, dtype=float)
        ax.imshow(values, cmap="Blues")
        ax.set_xticks([0, 1], list(CLASS_NAMES))
        ax.set_yticks([0, 1], list(CLASS_NAMES))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                ax.text(column, row, str(int(values[row, column])), ha="center", va="center")
    fig.suptitle("Test-set confusion matrices")
    fig.tight_layout()
    return fig


def plot_per_subject(subjects: list[int], series: dict[str, list[float]]):
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    positions = np.arange(len(subjects))
    width = 0.8 / max(len(series), 1)
    for index, (name, values) in enumerate(series.items()):
        ax.bar(positions + (index - (len(series) - 1) / 2) * width, values, width, label=name)
    ax.axhline(0.5, color="0.35", ls="--", lw=1, label="Nominal chance 0.50")
    ax.set_xticks(positions, [str(subject) for subject in subjects], rotation=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy")
    ax.set_title("Test-set accuracy by subject")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_waveform_examples(times, left, right, ch_names, left_title: str, right_title: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    _stack_plot(axes[0], times, left, ch_names, left_title)
    _stack_plot(axes[1], times, right, ch_names, right_title)
    axes[1].set_xlabel("Time after cue (s)")
    fig.suptitle("Held-out epochs, 8–30 Hz, average reference")
    fig.tight_layout()
    return fig


def plot_trial_waveforms(times, data, ch_names, title: str):
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    _stack_plot(ax, times, data, ch_names, title)
    ax.set_xlabel("Time after cue (s)")
    fig.tight_layout()
    return fig


def plot_spectrum(data, sfreq: float, ch_names, title: str):
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    microvolts = np.asarray(data, dtype=float) * 1e6
    nperseg = min(microvolts.shape[-1], max(16, int(sfreq)))
    for index, name in enumerate(ch_names):
        freqs, psd = welch(microvolts[index], fs=float(sfreq), nperseg=nperseg)
        ax.plot(freqs, psd, label=name, lw=1.2)
    ax.axvline(8, color="0.5", ls="--", lw=0.8)
    ax.axvline(30, color="0.5", ls="--", lw=0.8)
    ax.set_xlim(0, 40)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def save_figure(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _stack_plot(ax, times, data, ch_names, title: str) -> None:
    microvolts = np.asarray(data, dtype=float) * 1e6
    spans = microvolts.max(axis=1) - microvolts.min(axis=1)
    spacing = max(float(np.max(spans)) * 1.15, 1.0) if microvolts.size else 1.0
    for index, name in enumerate(ch_names):
        ax.plot(times, microvolts[index] + index * spacing, label=name, lw=1.0)
    ax.set_yticks([index * spacing for index in range(len(ch_names))], list(ch_names))
    ax.set_ylabel("Amplitude (µV, offset)")
    ax.set_title(title)
    ax.legend(loc="upper right")

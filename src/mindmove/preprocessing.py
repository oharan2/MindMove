"""EEG loading, filtering, epoching, and training-only normalization.

T1 and T2 are imagined left and right hand only in runs 4, 8, and 12.
Other PhysioNet runs use the same annotation names for different tasks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Bump this when the epoch definition changes so cached files are rebuilt.
PIPELINE_VERSION = 1

# Names produced by mne.datasets.eegbci.standardize on the PhysioNet 64-channel montage.
CANONICAL_CHANNELS: tuple[str, ...] = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FT8",
    "T7", "T8", "T9", "T10",
    "TP7", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
    "Iz",
)

SENSORIMOTOR_CHANNELS: tuple[str, ...] = ("C3", "Cz", "C4")
CLASS_NAMES: tuple[str, ...] = ("left", "right")
DISPLAY_NAMES: tuple[str, ...] = ("imagined left hand", "imagined right hand")
IMAGERY_RUNS: tuple[int, ...] = (4, 8, 12)
NORM_EPS = np.float32(1e-6)


class RecordingRejected(Exception):
    """A recording cannot be used. The reason is safe to store in the manifest."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def class_name(label: int) -> str:
    return CLASS_NAMES[int(label)]


def display_name(label: int) -> str:
    return DISPLAY_NAMES[int(label)]


def ensure_imagery_run(run: int) -> None:
    if int(run) not in IMAGERY_RUNS:
        raise ValueError(
            f"Run {run} is not a left/right hand imagery run. "
            "T1/T2 mean imagined left/right only in runs 4, 8, and 12."
        )


def annotation_to_label(description: str, run: int) -> int | None:
    """Map an annotation to a class index.

    Returns 0 for T1 (left), 1 for T2 (right), and None for rest or anything else.
    Raises if ``run`` is not a left/right imagery run, because the same codes
    mean other tasks in the rest of the dataset.
    """
    ensure_imagery_run(run)
    text = str(description).strip().upper()
    if text == "T1":
        return 0
    if text == "T2":
        return 1
    return None


def epoch_window_samples(preprocessing: dict[str, Any]) -> int:
    """Sample count of the half-open window [epoch_start, epoch_end)."""
    start = float(preprocessing["epoch_start"])
    end = float(preprocessing["epoch_end"])
    sfreq = float(preprocessing["sfreq"])
    n_samples = int(round((end - start) * sfreq))
    if n_samples <= 1:
        raise ValueError("The epoch window is too short for this sampling rate.")
    return n_samples


def fit_channel_norm(data: np.ndarray, eps: float = float(NORM_EPS)) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation of training epochs.

    ``data`` has shape (epochs, channels, time). Statistics are reduced over
    epochs and time, so the caller must pass training epochs only.
    Returned arrays have shape (1, channels, 1). Standard deviation uses
    NumPy's population definition (ddof=0) and is floored at ``eps``.
    """
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] == 0:
        raise ValueError(f"Expected a non-empty array (epochs, channels, time), got {array.shape}.")
    mean = array.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = array.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std = np.maximum(std, np.float32(eps))
    return mean, std


def apply_channel_norm(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    center = np.asarray(mean, dtype=np.float32)
    scale = np.asarray(std, dtype=np.float32)
    return (values - center) / scale


def preprocess_recording(raw, subject: int, run: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """Filter one recording and return labeled epochs.

    The function copies ``raw`` before modifying it. Filtering is applied to
    this recording alone. Rest annotations are not turned into epochs.
    """
    import mne
    from mne.datasets.eegbci import standardize

    ensure_imagery_run(run)
    pre = cfg["preprocessing"]
    if str(pre["reference"]).lower() != "average":
        raise ValueError("Only an average reference is implemented.")
    target_sfreq = float(pre["sfreq"])
    n_samples = epoch_window_samples(pre)
    epoch_start = float(pre["epoch_start"])
    epoch_end = float(pre["epoch_end"])

    raw = raw.copy()
    if getattr(raw, "preload", True) is False:
        raw.load_data()
    observed_sfreq = float(raw.info["sfreq"])
    if observed_sfreq <= 0 or observed_sfreq > 10000:
        raise RecordingRejected(f"implausible sampling rate {observed_sfreq}")

    standardize(raw)
    missing = [name for name in CANONICAL_CHANNELS if name not in raw.ch_names]
    if missing:
        preview = ", ".join(raw.ch_names[:12])
        raise RecordingRejected(
            f"missing {len(missing)} expected channels (first missing: {missing[:6]}); "
            f"file starts with [{preview}]"
        )
    raw.pick(list(CANONICAL_CHANNELS), verbose="ERROR")
    raw.set_channel_types(
        {name: "eeg" for name in raw.ch_names},
        on_unit_change="ignore",
        verbose="ERROR",
    )
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")

    resampled = False
    if abs(float(raw.info["sfreq"]) - target_sfreq) > 1e-3:
        raw.resample(target_sfreq, verbose="ERROR")
        resampled = True
    if abs(float(raw.info["sfreq"]) - target_sfreq) > 1e-3:
        raise RecordingRejected(
            f"sampling rate is {raw.info['sfreq']} after resampling, not {target_sfreq}"
        )

    raw.filter(
        l_freq=float(pre["low_freq"]),
        h_freq=float(pre["high_freq"]),
        picks="eeg",
        verbose="ERROR",
    )
    continuous = raw.get_data()
    if not np.isfinite(continuous).all():
        raise RecordingRejected("non-finite values after filtering")
    flat = [
        name
        for name, deviation in zip(raw.ch_names, continuous.std(axis=1))
        if float(deviation) < 1e-15
    ]
    if flat:
        raise RecordingRejected(f"flat channels after filtering: {flat}")

    events, ann_counts, unknown, n_rest, n_dup, _cue_onsets = _imagery_events(raw, int(run))
    present = set(events[:, 2].tolist())
    event_id = {}
    if 1 in present:
        event_id["left"] = 1
    if 2 in present:
        event_id["right"] = 2
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=epoch_start,
        tmax=epoch_end,
        baseline=None,
        picks="eeg",
        preload=True,
        proj=False,
        reject=None,
        flat=None,
        detrend=None,
        on_missing="ignore",
        reject_by_annotation=False,
        event_repeated="error",
        verbose="ERROR",
    )
    if len(epochs) == 0:
        raise RecordingRejected(
            f"S{int(subject):03d} run {int(run)} produced no epochs inside the recording "
            f"(annotation counts={ann_counts})"
        )
    data = np.asarray(epochs.get_data(copy=True), dtype=np.float64)
    times = np.asarray(epochs.times, dtype=np.float64)
    data, times = trim_inclusive_endpoint(data, times, n_samples, epoch_end)
    finite = np.isfinite(data).all(axis=(1, 2))
    n_dropped_nonfinite = int((~finite).sum())
    data = data[finite]
    kept_events = epochs.events[finite]
    selection = np.asarray(epochs.selection, dtype=np.int64)[finite]
    if data.shape[0] == 0:
        raise RecordingRejected("all epochs contained non-finite samples")
    if data.shape[1:] != (len(CANONICAL_CHANNELS), n_samples):
        raise RecordingRejected(f"incompatible epoch shape {data.shape}")
    if abs(float(times[0]) - epoch_start) > (0.5 / target_sfreq):
        raise RecordingRejected(f"epoch time axis starts at {times[0]}, not {epoch_start}")

    codes = kept_events[:, 2]
    if not np.isin(codes, (1, 2)).all():
        raise RecordingRejected(f"unexpected event codes {sorted(set(codes.tolist()))}")
    labels = np.where(codes == 1, 0, 1).astype(np.int64)
    onsets = (kept_events[:, 0].astype(np.float64) - float(raw.first_samp)) / target_sfreq
    n_outside = int(len(events) - len(epochs))

    return {
        "data": np.ascontiguousarray(data, dtype=np.float32),
        "labels": labels,
        "trials": selection.astype(np.int64),
        "onsets": onsets.astype(np.float64),
        "times": np.ascontiguousarray(times, dtype=np.float64),
        "ch_names": list(raw.ch_names),
        "sfreq": target_sfreq,
        "observed_sfreq": observed_sfreq,
        "resampled": bool(resampled),
        "n_rest": int(n_rest),
        "n_unknown": int(len(unknown)),
        "unknown_descriptions": list(unknown),
        "annotation_counts": {str(key): int(value) for key, value in ann_counts.items()},
        "n_dropped_nonfinite": n_dropped_nonfinite,
        "n_dropped_outside": n_outside,
        "n_duplicate_events": int(n_dup),
    }


def trim_inclusive_endpoint(
    data: np.ndarray,
    times: np.ndarray,
    n_samples: int,
    epoch_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop MNE's extra sample at the inclusive ``tmax``.

    A 2.000 s window at 160 Hz contains 320 samples on a half-open interval.
    ``Epochs(..., tmax=3.0)`` also keeps the sample at exactly 3.0 s, which
    makes 321 samples. That endpoint sample is removed.
    """
    times = np.asarray(times, dtype=np.float64)
    if data.shape[-1] == n_samples + 1 and float(times[-1]) >= float(epoch_end) - 1e-6:
        data = data[..., :n_samples]
        times = times[:n_samples]
    if data.shape[-1] != n_samples or len(times) != n_samples:
        raise RecordingRejected(
            f"expected {n_samples} samples, got {data.shape[-1]} "
            f"covering {float(times[0]):.4f}s to {float(times[-1]):.4f}s"
        )
    if float(times[-1]) > float(epoch_end) + 1e-6:
        raise RecordingRejected(
            f"last sample at {float(times[-1]):.4f}s is past epoch_end {epoch_end}"
        )
    return data, times


def _imagery_events(raw, run: int):
    descriptions = [str(item).strip() for item in raw.annotations.description]
    counts: dict[str, int] = {}
    rows: list[tuple[float, int]] = []
    unknown: set[str] = set()
    for description, onset in zip(descriptions, raw.annotations.onset):
        counts[description] = counts.get(description, 0) + 1
        label = annotation_to_label(description, run)
        if label is None:
            if description.upper() not in {"T0", "T1", "T2"}:
                unknown.add(description)
            continue
        rows.append((float(onset), int(label)))
    if not rows:
        raise RecordingRejected(f"no T1/T2 events; annotation counts={counts}")
    rows.sort(key=lambda item: item[0])
    onsets = [item[0] for item in rows]
    labels = [item[1] for item in rows]
    samples = np.asarray(raw.time_as_index(onsets, use_rounding=True), dtype=np.int64)
    keep: list[int] = []
    seen: set[int] = set()
    n_dup = 0
    for index, sample in enumerate(samples.tolist()):
        if int(sample) in seen:
            n_dup += 1
            continue
        seen.add(int(sample))
        keep.append(index)
    samples = samples[keep]
    labels_kept = [labels[index] for index in keep]
    onsets_kept = [onsets[index] for index in keep]
    codes = np.where(np.asarray(labels_kept) == 0, 1, 2).astype(np.int64)
    events = np.column_stack(
        [samples, np.zeros(len(samples), dtype=np.int64), codes]
    )
    n_rest = sum(count for description, count in counts.items() if description.upper() == "T0")
    return (
        np.ascontiguousarray(events),
        counts,
        sorted(unknown),
        int(n_rest),
        int(n_dup),
        np.asarray(onsets_kept, dtype=np.float64),
    )

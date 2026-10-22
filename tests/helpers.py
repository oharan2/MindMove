"""Synthetic recordings shared by unit tests."""

from __future__ import annotations

import numpy as np


def preprocessing_config(**overrides):
    preprocessing = {
        "sfreq": 160,
        "low_freq": 8,
        "high_freq": 30,
        "epoch_start": 1.0,
        "epoch_end": 3.0,
        "reference": "average",
    }
    preprocessing.update(overrides)
    return {"preprocessing": preprocessing}


def synthetic_raw(sfreq: float = 160.0, seed: int = 0):
    import mne

    from mindmove.preprocessing import CANONICAL_CHANNELS

    rng = np.random.default_rng(seed)
    n_times = int(90.0 * sfreq)
    data = rng.normal(scale=2e-5, size=(len(CANONICAL_CHANNELS), n_times))
    info = mne.create_info(list(CANONICAL_CHANNELS), sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    onsets = [5, 12, 19, 26, 33, 40, 47]
    descriptions = ["T0", "T1", "T0", "T2", "T1", "T2", "FOO"]
    raw.set_annotations(mne.Annotations(onsets, [4.0] * len(onsets), descriptions))
    return raw

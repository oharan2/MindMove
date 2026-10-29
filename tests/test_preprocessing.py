"""Label mapping, rest exclusion, epoch length, and training-only normalization."""

from __future__ import annotations

import numpy as np
import pytest

from mindmove.preprocessing import (
    CANONICAL_CHANNELS,
    annotation_to_label,
    epoch_window_samples,
    fit_channel_norm,
    apply_channel_norm,
    preprocess_recording,
)
from tests.helpers import preprocessing_config, synthetic_raw

DOTTED_EEG_NAMES = [
    "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.",
    "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..", "C6..",
    "Cp5.", "Cp3.", "Cp1.", "Cpz.", "Cp2.", "Cp4.", "Cp6.",
    "Fp1.", "Fpz.", "Fp2.",
    "Af7.", "Af3.", "Afz.", "Af4.", "Af8.",
    "F7..", "F5..", "F3..", "F1..", "Fz..", "F2..", "F4..", "F6..", "F8..",
    "Ft7.", "Ft8.",
    "T7..", "T8..", "T9..", "T10.",
    "Tp7.", "Tp8.",
    "P7..", "P5..", "P3..", "P1..", "Pz..", "P2..", "P4..", "P6..", "P8..",
    "Po7.", "Po3.", "Poz.", "Po4.", "Po8.",
    "O1..", "Oz..", "O2..", "Iz..",
]


def test_t1_t2_mapping_for_imagery_runs():
    for run in (4, 8, 12):
        assert annotation_to_label("T1", run) == 0
        assert annotation_to_label(" t2 ", run) == 1
        assert annotation_to_label("T0", run) is None
        assert annotation_to_label("FOO", run) is None


def test_other_runs_are_rejected():
    with pytest.raises(ValueError, match="runs 4, 8, and 12"):
        annotation_to_label("T1", 3)
    with pytest.raises(ValueError, match="runs 4, 8, and 12"):
        annotation_to_label("T1", 6)


def test_default_window_has_320_samples():
    assert epoch_window_samples(preprocessing_config()["preprocessing"]) == 320


def test_rest_is_excluded_and_shape_matches_the_cnn(processed_epochs):
    epochs = processed_epochs
    assert list(epochs["labels"]) == [0, 1, 0, 1]
    assert epochs["n_rest"] == 2
    assert epochs["n_unknown"] == 1
    assert epochs["data"].shape == (4, 64, 320)
    assert epochs["data"].dtype == np.float32
    assert np.isfinite(epochs["data"]).all()
    assert epochs["times"][0] == pytest.approx(1.0, abs=1e-6)
    assert epochs["times"][-1] < 3.0
    assert epochs["times"][-1] == pytest.approx(1.0 + 319 / 160, abs=1e-4)
    assert list(epochs["onsets"]) == pytest.approx([12, 26, 33, 40], abs=1e-6)
    assert "T0" not in set(epochs["labels"])


def test_non_imagery_recording_is_rejected_before_use():
    with pytest.raises(ValueError, match="runs 4, 8, and 12"):
        preprocess_recording(synthetic_raw(), subject=1, run=5, cfg=preprocessing_config())


def test_resample_to_target_rate_keeps_320_samples():
    raw = synthetic_raw(sfreq=128.0, seed=1)
    epochs = preprocess_recording(raw, subject=1, run=8, cfg=preprocessing_config())
    assert epochs["resampled"] is True
    assert epochs["observed_sfreq"] == pytest.approx(128.0)
    assert epochs["sfreq"] == pytest.approx(160.0)
    assert epochs["data"].shape[1:] == (64, 320)
    assert list(epochs["labels"]) == [0, 1, 0, 1]


def test_canonical_names_match_mne_standardize():
    import mne
    from mne.datasets.eegbci import standardize

    info = mne.create_info(DOTTED_EEG_NAMES, 160, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((64, 8)), info, verbose="ERROR")
    standardize(raw)
    assert raw.ch_names == list(CANONICAL_CHANNELS)
    assert len(set(CANONICAL_CHANNELS)) == 64
    assert {"C3", "Cz", "C4"} <= set(CANONICAL_CHANNELS)


def test_normalization_uses_training_values_only():
    train = np.zeros((4, 2, 5), dtype=np.float32)
    test = np.full((4, 2, 5), 10.0, dtype=np.float32)
    mean, std = fit_channel_norm(train)
    assert np.allclose(mean, 0.0)
    leaked_mean, _leaked_std = fit_channel_norm(np.concatenate([train, test], axis=0))
    assert not np.allclose(mean, leaked_mean)
    normalized = apply_channel_norm(test, mean, std)
    refit_mean, refit_std = fit_channel_norm(test)
    refit = apply_channel_norm(test, refit_mean, refit_std)
    assert not np.allclose(normalized, refit)
    own_mean, own_std = fit_channel_norm(train)
    assert np.allclose(mean, own_mean)
    assert np.allclose(std, own_std)

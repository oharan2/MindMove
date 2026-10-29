"""Shared synthetic EEG fixtures. These tests do not download PhysioNet."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import mne

from tests.helpers import preprocessing_config, synthetic_raw

mne.set_log_level("ERROR")


@pytest.fixture(scope="session")
def processed_epochs():
    from mindmove.preprocessing import preprocess_recording

    raw = synthetic_raw()
    return preprocess_recording(raw, subject=1, run=4, cfg=preprocessing_config())

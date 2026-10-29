"""Split integrity and the checked-in experiment configs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mindmove.config import load_config, validate_config
from mindmove.data import EpochTable, assert_no_overlapping_windows, concatenate_tables, split_table
from mindmove.metrics import compute_metrics, majority_class

ROOT = Path(__file__).resolve().parents[1]


def test_held_out_runs_are_disjoint_and_may_share_subjects():
    table = _table(subjects=(1, 2), runs=(4, 8, 12))
    train, val, test = split_table(table, _held_out_config())
    assert set(train.trial_ids()).isdisjoint(val.trial_ids())
    assert set(train.trial_ids()).isdisjoint(test.trial_ids())
    assert set(val.trial_ids()).isdisjoint(test.trial_ids())
    assert set(train.subjects.tolist()) & set(test.subjects.tolist())
    assert set(zip(train.subjects.tolist(), train.runs.tolist())).isdisjoint(
        set(zip(test.subjects.tolist(), test.runs.tolist()))
    )
    assert set(train.runs.tolist()) == {4}
    assert set(val.runs.tolist()) == {8}
    assert set(test.runs.tolist()) == {12}


def test_unseen_subjects_do_not_share_people():
    table = _table(subjects=(1, 2, 3), runs=(4, 8))
    cfg = _held_out_config()
    cfg["evaluation"] = {
        "protocol": "unseen_subjects",
        "train_subjects": [1],
        "val_subjects": [2],
        "test_subjects": [3],
        "runs": [4, 8],
    }
    train, val, test = split_table(table, cfg)
    assert set(train.subjects.tolist()).isdisjoint(val.subjects.tolist())
    assert set(train.subjects.tolist()).isdisjoint(test.subjects.tolist())
    assert set(val.subjects.tolist()).isdisjoint(test.subjects.tolist())
    assert set(train.runs.tolist()) == {4, 8}
    assert set(test.runs.tolist()) == {4, 8}


def test_overlapping_windows_are_rejected():
    table = _table(subjects=(1,), runs=(4,), gap=0.5)
    with pytest.raises(ValueError, match="overlap"):
        assert_no_overlapping_windows(table, 2.0)


def test_default_config_matches_the_plan():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg["dataset"]["subjects"] == list(range(1, 21))
    assert cfg["dataset"]["train_runs"] == [4]
    assert cfg["dataset"]["val_runs"] == [8]
    assert cfg["dataset"]["test_runs"] == [12]
    assert cfg["preprocessing"]["sfreq"] == 160
    assert cfg["preprocessing"]["low_freq"] == 8
    assert cfg["preprocessing"]["high_freq"] == 30
    assert cfg["preprocessing"]["epoch_start"] == 1.0
    assert cfg["preprocessing"]["epoch_end"] == 3.0
    assert str(cfg["preprocessing"]["reference"]) == "average"
    assert cfg["training"]["seed"] == 42
    assert cfg["training"]["batch_size"] == 32
    assert cfg["training"]["learning_rate"] == pytest.approx(0.001)
    assert cfg["training"]["epochs"] == 50
    assert cfg["training"]["patience"] == 8
    assert cfg["model"]["dropout"] == pytest.approx(0.4)
    assert cfg["model"]["num_classes"] == 2
    assert cfg["evaluation"]["protocol"] == "held_out_run"


def test_smoke_config_does_not_change_preprocessing():
    default = load_config(ROOT / "configs" / "default.yaml")
    smoke = load_config(ROOT / "configs" / "smoke.yaml")
    assert smoke["preprocessing"] == default["preprocessing"]
    assert smoke["dataset"]["subjects"] == [1]
    assert smoke["dataset"]["train_runs"] == default["dataset"]["train_runs"]
    assert smoke["training"]["epochs"] < default["training"]["epochs"]
    assert smoke["paths"]["artifacts"] != default["paths"]["artifacts"]
    assert smoke["paths"]["cache"] == default["paths"]["cache"]


def test_subject_holdout_config_is_disjoint():
    cfg = load_config(ROOT / "configs" / "subject_holdout.yaml")
    assert cfg["evaluation"]["protocol"] == "unseen_subjects"
    train = set(cfg["evaluation"]["train_subjects"])
    val = set(cfg["evaluation"]["val_subjects"])
    test = set(cfg["evaluation"]["test_subjects"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert set(cfg["evaluation"]["runs"]) == {4, 8, 12}


def test_overlapping_or_non_imagery_runs_are_rejected():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["dataset"]["test_runs"] = [4]
    with pytest.raises(ValueError, match="both"):
        validate_config(cfg)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["dataset"]["train_runs"] = [4, 6]
    with pytest.raises(ValueError, match="left/right"):
        validate_config(cfg)


def test_metrics_of_perfect_predictions_and_majority_tie():
    labels = np.array([0, 1, 0, 1])
    proba_subjects = np.array([1, 1, 2, 2])
    perfect = compute_metrics(labels, labels.copy(), proba_subjects)
    assert perfect["accuracy"] == 1.0
    assert perfect["balanced_accuracy"] == 1.0
    assert perfect["confusion_matrix"] == [[2, 0], [0, 2]]
    assert majority_class(labels) == 0
    majority = np.zeros_like(labels)
    scored = compute_metrics(labels, majority, proba_subjects)
    assert scored["accuracy"] == 0.5


def _held_out_config():
    return {
        "preprocessing": {"epoch_start": 1.0, "epoch_end": 3.0, "sfreq": 160},
        "dataset": {
            "subjects": [1, 2],
            "train_runs": [4],
            "val_runs": [8],
            "test_runs": [12],
        },
        "evaluation": {"protocol": "held_out_run"},
    }


def _table(subjects, runs, gap: float = 8.0) -> EpochTable:
    parts = []
    for subject in subjects:
        for run in runs:
            n_epochs = 2
            parts.append(
                EpochTable(
                    data=np.zeros((n_epochs, 2, 4), dtype=np.float32),
                    labels=np.array([0, 1], dtype=np.int64),
                    subjects=np.full(n_epochs, subject),
                    runs=np.full(n_epochs, run),
                    trials=np.arange(n_epochs),
                    onsets=np.arange(n_epochs, dtype=float) * gap,
                    times=np.arange(4) / 160.0 + 1.0,
                    ch_names=["C3", "C4"],
                    sfreq=160.0,
                )
            )
    return concatenate_tables(parts)

"""Download PhysioNet EEG recordings and cache labeled epochs.

Training, validation, and test trials come from different runs (or, in the
optional protocol, different subjects). Windows from one recording are never
randomly assigned to more than one split.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mindmove.config import (
    artifacts_dir,
    cache_dir,
    configure_logging,
    dataset_path,
    ensure_artifact_dirs,
    iter_subject_runs,
    load_config,
    build_parser,
)
from mindmove.preprocessing import (
    CANONICAL_CHANNELS,
    PIPELINE_VERSION,
    RecordingRejected,
    epoch_window_samples,
    preprocess_recording,
)

logger = logging.getLogger(__name__)


@dataclass
class EpochTable:
    data: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    runs: np.ndarray
    trials: np.ndarray
    onsets: np.ndarray
    times: np.ndarray
    ch_names: list[str]
    sfreq: float

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.subjects = np.asarray(self.subjects, dtype=np.int64)
        self.runs = np.asarray(self.runs, dtype=np.int64)
        self.trials = np.asarray(self.trials, dtype=np.int64)
        self.onsets = np.asarray(self.onsets, dtype=np.float64)
        self.times = np.asarray(self.times, dtype=np.float64)
        self.ch_names = [str(name) for name in self.ch_names]
        self.sfreq = float(self.sfreq)
        n_epochs = int(self.labels.shape[0])
        if self.data.ndim != 3:
            raise ValueError(f"Epoch data must have shape (epochs, channels, time), got {self.data.shape}.")
        if self.data.shape[0] != n_epochs:
            raise ValueError("The data array and the labels have different lengths.")
        if self.data.shape[1] != len(self.ch_names):
            raise ValueError("The channel axis does not match the channel names.")
        if self.data.shape[2] != len(self.times):
            raise ValueError("The time axis does not match the time vector.")
        for name, array in (
            ("subjects", self.subjects),
            ("runs", self.runs),
            ("trials", self.trials),
            ("onsets", self.onsets),
        ):
            if len(array) != n_epochs:
                raise ValueError(f"{name} has length {len(array)} for {n_epochs} epochs.")
        if not np.isfinite(self.data).all():
            raise ValueError("Epoch data contains non-finite values.")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def trial_ids(self) -> np.ndarray:
        return np.array(
            [
                f"S{int(subject):03d}_R{int(run):02d}_T{int(trial):03d}"
                for subject, run, trial in zip(self.subjects, self.runs, self.trials)
            ]
        )

    def distribution(self) -> dict[str, int]:
        return {
            "n": int(len(self)),
            "n_left": int(np.sum(self.labels == 0)),
            "n_right": int(np.sum(self.labels == 1)),
        }

    def subset(self, mask: np.ndarray) -> EpochTable:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self),):
            raise ValueError("Selection mask does not match the number of epochs.")
        return EpochTable(
            data=self.data[mask].copy(),
            labels=self.labels[mask].copy(),
            subjects=self.subjects[mask].copy(),
            runs=self.runs[mask].copy(),
            trials=self.trials[mask].copy(),
            onsets=self.onsets[mask].copy(),
            times=self.times.copy(),
            ch_names=list(self.ch_names),
            sfreq=self.sfreq,
        )

    def select(self, subjects=None, runs=None) -> EpochTable:
        mask = np.ones(len(self), dtype=bool)
        if subjects is not None:
            mask &= np.isin(self.subjects, np.asarray(list(subjects), dtype=np.int64))
        if runs is not None:
            mask &= np.isin(self.runs, np.asarray(list(runs), dtype=np.int64))
        return self.subset(mask)


def concatenate_tables(tables: list[EpochTable]) -> EpochTable:
    if not tables:
        raise ValueError("No epoch tables to combine.")
    first = tables[0]
    for other in tables[1:]:
        if other.ch_names != first.ch_names:
            raise ValueError("Recordings do not share the same channel names and order.")
        if abs(other.sfreq - first.sfreq) > 1e-6:
            raise ValueError("Recordings do not share the same sampling rate.")
        if other.times.shape != first.times.shape or not np.allclose(other.times, first.times, atol=1e-6):
            raise ValueError("Recordings do not share the same epoch time axis.")
    if len(tables) == 1:
        return tables[0]
    return EpochTable(
        data=np.concatenate([item.data for item in tables], axis=0),
        labels=np.concatenate([item.labels for item in tables]),
        subjects=np.concatenate([item.subjects for item in tables]),
        runs=np.concatenate([item.runs for item in tables]),
        trials=np.concatenate([item.trials for item in tables]),
        onsets=np.concatenate([item.onsets for item in tables]),
        times=first.times.copy(),
        ch_names=list(first.ch_names),
        sfreq=first.sfreq,
    )


def assert_no_overlapping_windows(table: EpochTable, window_seconds: float) -> None:
    if window_seconds <= 0:
        raise ValueError("The analysis window must be positive.")
    for subject in np.unique(table.subjects):
        for run in np.unique(table.runs):
            mask = (table.subjects == subject) & (table.runs == run)
            if int(mask.sum()) < 2:
                continue
            gaps = np.diff(np.sort(table.onsets[mask].astype(float)))
            if np.any(gaps < window_seconds - 1e-6):
                raise ValueError(
                    f"Epochs overlap in subject {int(subject)} run {int(run)}: "
                    f"the shortest cue gap is {float(gaps.min()):.3f}s and the "
                    f"analysis window is {window_seconds:.3f}s."
                )


def assert_split_integrity(
    train: EpochTable,
    val: EpochTable,
    test: EpochTable,
    protocol: str,
    window_seconds: float,
) -> None:
    parts = {"train": train, "validation": val, "test": test}
    id_sets: dict[str, set[str]] = {}
    for name, part in parts.items():
        if len(part) == 0:
            raise ValueError(f"The {name} split is empty.")
        labels = set(int(item) for item in np.unique(part.labels))
        if labels != {0, 1}:
            raise ValueError(f"The {name} split must contain both classes, found {sorted(labels)}.")
        assert_no_overlapping_windows(part, window_seconds)
        identifiers = part.trial_ids()
        if len(set(identifiers.tolist())) != len(identifiers):
            raise ValueError(f"The {name} split contains duplicate trial ids.")
        id_sets[name] = set(identifiers.tolist())
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        shared = id_sets[left] & id_sets[right]
        if shared:
            raise ValueError(f"{left} and {right} share trial ids, for example {sorted(shared)[:5]}.")
    if protocol == "held_out_run":
        recordings = {
            name: set(zip(part.subjects.tolist(), part.runs.tolist()))
            for name, part in parts.items()
        }
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            shared = recordings[left] & recordings[right]
            if shared:
                raise ValueError(f"{left} and {right} share recordings {sorted(shared)[:5]}.")
    elif protocol == "unseen_subjects":
        people = {name: set(int(item) for item in part.subjects) for name, part in parts.items()}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            shared = people[left] & people[right]
            if shared:
                raise ValueError(f"{left} and {right} share subjects {sorted(shared)}.")
    else:
        raise ValueError(f"Unknown evaluation protocol '{protocol}'.")


def split_table(table: EpochTable, cfg: dict[str, Any]) -> tuple[EpochTable, EpochTable, EpochTable]:
    from mindmove.config import protocol_name

    protocol = protocol_name(cfg)
    window = float(cfg["preprocessing"]["epoch_end"]) - float(cfg["preprocessing"]["epoch_start"])
    if protocol == "held_out_run":
        subjects = cfg["dataset"]["subjects"]
        train = table.select(subjects=subjects, runs=cfg["dataset"]["train_runs"])
        val = table.select(subjects=subjects, runs=cfg["dataset"]["val_runs"])
        test = table.select(subjects=subjects, runs=cfg["dataset"]["test_runs"])
    elif protocol == "unseen_subjects":
        evaluation = cfg["evaluation"]
        runs = evaluation["runs"]
        train = table.select(subjects=evaluation["train_subjects"], runs=runs)
        val = table.select(subjects=evaluation["val_subjects"], runs=runs)
        test = table.select(subjects=evaluation["test_subjects"], runs=runs)
    else:
        raise ValueError(f"Unknown evaluation protocol '{protocol}'.")
    assert_split_integrity(train, val, test, protocol, window)
    return train, val, test


def preprocessing_fingerprint(cfg: dict[str, Any]) -> str:
    import hashlib

    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "preprocessing": cfg["preprocessing"],
        "channels": list(CANONICAL_CHANNELS),
        "label_map": {"T1": 0, "T2": 1, "T0": None},
        "imagery_runs": [4, 8, 12],
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_dataset(cfg: dict[str, Any], force: bool = False) -> tuple[EpochTable, dict[str, Any]]:
    """Download what is missing, preprocess each recording, and return all epochs."""
    import mne

    mne.set_log_level("ERROR")
    dirs = ensure_artifact_dirs(cfg)
    fingerprint = preprocessing_fingerprint(cfg)
    _download(cfg)
    tables: list[EpochTable] = []
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for subject, run in iter_subject_runs(cfg):
        outcome = _load_or_process(cfg, subject, run, fingerprint, force)
        if outcome["excluded"]:
            exclusions.append(outcome["exclusion"])
            logger.warning(
                "Excluded S%03d run %d: %s",
                subject,
                run,
                outcome["exclusion"]["reason"],
            )
            continue
        tables.append(outcome["table"])
        records.append(outcome["record"])
        logger.info(
            "S%03d run %d: %d left, %d right, %d rest excluded",
            subject,
            run,
            outcome["record"]["n_left"],
            outcome["record"]["n_right"],
            outcome["record"]["n_rest"],
        )
    manifest = _manifest(cfg, fingerprint, records, exclusions, tables)
    _write_json(dirs["root"] / "manifest.json", manifest)
    if not tables:
        reasons = "; ".join(item["reason"] for item in exclusions[:6])
        raise RuntimeError(f"No usable recordings. Exclusions: {reasons}")
    table = concatenate_tables(tables)
    _require_unique_ids(table)
    manifest["class_distribution"] = table.distribution()
    manifest["n_epochs"] = len(table)
    _write_json(dirs["root"] / "manifest.json", manifest)
    return table, manifest


def load_cached_dataset(cfg: dict[str, Any]) -> tuple[EpochTable, dict[str, Any]]:
    manifest_path = artifacts_dir(cfg) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "No preprocessed dataset found. From the project root run: "
            "python -m mindmove.data --config configs/default.yaml"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = preprocessing_fingerprint(cfg)
    if manifest.get("fingerprint") != fingerprint:
        raise RuntimeError(
            "Cached epochs were built with different preprocessing than this config. "
            "Run python -m mindmove.data again."
        )
    excluded = {(int(item["subject"]), int(item["run"])) for item in manifest.get("exclusions", [])}
    tables: list[EpochTable] = []
    missing: list[str] = []
    for subject, run in iter_subject_runs(cfg):
        if (subject, run) in excluded:
            continue
        path = _npz_path(cfg, subject, run)
        if not path.is_file():
            missing.append(f"S{subject:03d}R{run:02d}")
            continue
        tables.append(_read_table(path, fingerprint))
    if missing:
        preview = ", ".join(missing[:8])
        raise FileNotFoundError(f"Missing cached recordings ({preview}). Run python -m mindmove.data.")
    if not tables:
        raise RuntimeError("Every recording in this config was excluded. See artifacts/manifest.json.")
    table = concatenate_tables(tables)
    _require_unique_ids(table)
    return table, manifest


def _download(cfg: dict[str, Any]) -> None:
    from mne.datasets import eegbci

    from mindmove.config import dataset_runs, dataset_subjects

    root = dataset_path(cfg)
    root.mkdir(parents=True, exist_ok=True)
    subjects = dataset_subjects(cfg)
    runs = dataset_runs(cfg)
    for subject in subjects:
        logger.info("Fetching subject %s, runs %s", subject, runs)
        try:
            eegbci.load_data(
                [subject],
                runs,
                path=str(root),
                update_path=False,
                force_update=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not download EEGBCI subject {subject} runs {runs} from PhysioNet. "
                "Real EEG files are required; results cannot be substituted."
            ) from exc


def _load_or_process(cfg, subject: int, run: int, fingerprint: str, force: bool) -> dict[str, Any]:
    npz_path = _npz_path(cfg, subject, run)
    meta_path = _meta_path(cfg, subject, run)
    rejected_path = _rejected_path(cfg, subject, run)
    if not force and npz_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            table = _read_table(npz_path, fingerprint)
            meta["n_left"] = int(np.sum(table.labels == 0))
            meta["n_right"] = int(np.sum(table.labels == 1))
            meta["n_epochs"] = len(table)
            return {"excluded": False, "table": table, "record": meta}
    if not force and rejected_path.is_file():
        rejected = json.loads(rejected_path.read_text(encoding="utf-8"))
        if rejected.get("fingerprint") == fingerprint:
            return {"excluded": True, "exclusion": rejected}
    try:
        prepared = _read_and_preprocess(cfg, subject, run)
    except RecordingRejected as exc:
        _remove(npz_path)
        _remove(meta_path)
        exclusion = {
            "subject": int(subject),
            "run": int(run),
            "reason": exc.reason,
            "fingerprint": fingerprint,
        }
        _write_json(rejected_path, exclusion)
        return {"excluded": True, "exclusion": exclusion}
    table = _table_from_prepared(prepared, subject, run)
    _write_recording(npz_path, meta_path, table, prepared, subject, run, fingerprint)
    _remove(rejected_path)
    record = json.loads(meta_path.read_text(encoding="utf-8"))
    return {"excluded": False, "table": table, "record": record}


def _read_and_preprocess(cfg, subject: int, run: int) -> dict[str, Any]:
    import mne
    from mne.datasets import eegbci

    paths = eegbci.load_data(
        [subject],
        [run],
        path=str(dataset_path(cfg)),
        update_path=False,
        force_update=False,
    )
    raw = mne.io.read_raw_edf(paths[0], preload=True, verbose="ERROR")
    return preprocess_recording(raw, subject=subject, run=run, cfg=cfg)


def _table_from_prepared(prepared: dict[str, Any], subject: int, run: int) -> EpochTable:
    n_epochs = int(prepared["labels"].shape[0])
    if len(set(int(item) for item in prepared["trials"])) != n_epochs:
        raise RecordingRejected("duplicate trial indices within a recording")
    return EpochTable(
        data=prepared["data"],
        labels=prepared["labels"],
        subjects=np.full(n_epochs, int(subject), dtype=np.int64),
        runs=np.full(n_epochs, int(run), dtype=np.int64),
        trials=prepared["trials"],
        onsets=prepared["onsets"],
        times=prepared["times"],
        ch_names=prepared["ch_names"],
        sfreq=prepared["sfreq"],
    )


def _write_recording(npz_path, meta_path, table: EpochTable, prepared, subject, run, fingerprint) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = npz_path.with_name(npz_path.name + ".partial.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            data=table.data,
            labels=table.labels,
            subjects=table.subjects,
            runs=table.runs,
            trials=table.trials,
            onsets=table.onsets,
            times=table.times,
            ch_names=np.array(table.ch_names),
            sfreq=np.float64(table.sfreq),
            fingerprint=np.array(fingerprint),
        )
    temporary.replace(npz_path)
    meta = {
        "fingerprint": fingerprint,
        "subject": int(subject),
        "run": int(run),
        "n_epochs": len(table),
        "n_left": int(np.sum(table.labels == 0)),
        "n_right": int(np.sum(table.labels == 1)),
        "n_rest": int(prepared["n_rest"]),
        "n_unknown": int(prepared["n_unknown"]),
        "unknown_descriptions": list(prepared["unknown_descriptions"]),
        "annotation_counts": prepared["annotation_counts"],
        "observed_sfreq": float(prepared["observed_sfreq"]),
        "resampled": bool(prepared["resampled"]),
        "n_dropped_nonfinite": int(prepared["n_dropped_nonfinite"]),
        "n_dropped_outside": int(prepared["n_dropped_outside"]),
        "n_duplicate_events": int(prepared["n_duplicate_events"]),
        "sfreq": float(table.sfreq),
        "n_samples": int(table.data.shape[-1]),
    }
    _write_json(meta_path, meta)


def _read_table(path: Path, fingerprint: str) -> EpochTable:
    with np.load(path, allow_pickle=False) as blob:
        stored = str(np.asarray(blob["fingerprint"]).item())
        if stored != fingerprint:
            raise RuntimeError(f"{path.name} does not match the current preprocessing. Rebuild the cache.")
        table = EpochTable(
            data=np.array(blob["data"], dtype=np.float32, copy=True),
            labels=np.array(blob["labels"], dtype=np.int64, copy=True),
            subjects=np.array(blob["subjects"], dtype=np.int64, copy=True),
            runs=np.array(blob["runs"], dtype=np.int64, copy=True),
            trials=np.array(blob["trials"], dtype=np.int64, copy=True),
            onsets=np.array(blob["onsets"], dtype=np.float64, copy=True),
            times=np.array(blob["times"], dtype=np.float64, copy=True),
            ch_names=[str(name) for name in np.array(blob["ch_names"]).tolist()],
            sfreq=float(np.asarray(blob["sfreq"]).item()),
        )
    if table.ch_names != list(CANONICAL_CHANNELS):
        raise RuntimeError(f"{path.name} has an unexpected channel set.")
    return table


def _manifest(cfg, fingerprint, records, exclusions, tables) -> dict[str, Any]:
    import mne

    observed = sorted({float(record["observed_sfreq"]) for record in records})
    return {
        "fingerprint": fingerprint,
        "pipeline_version": PIPELINE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mne_version": mne.__version__,
        "n_samples": epoch_window_samples(cfg["preprocessing"]),
        "recordings": records,
        "exclusions": exclusions,
        "n_epochs": int(sum(len(table) for table in tables)),
        "observed_sfreq_hz": observed,
        "resampled_any": any(bool(record["resampled"]) for record in records),
    }


def _require_unique_ids(table: EpochTable) -> None:
    identifiers = table.trial_ids()
    if len(set(identifiers.tolist())) != len(identifiers):
        raise RuntimeError("The cached dataset contains duplicate trial identifiers.")


def _npz_path(cfg, subject: int, run: int) -> Path:
    return cache_dir(cfg) / f"S{int(subject):03d}_R{int(run):02d}.npz"


def _meta_path(cfg, subject: int, run: int) -> Path:
    return cache_dir(cfg) / f"S{int(subject):03d}_R{int(run):02d}.json"


def _rejected_path(cfg, subject: int, run: int) -> Path:
    return cache_dir(cfg) / f"S{int(subject):03d}_R{int(run):02d}.rejected.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _remove(path: Path) -> None:
    if path.is_file():
        path.unlink()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Download and preprocess left/right motor-imagery EEG.")
    parser.add_argument("--force", action="store_true", help="Recompute cached epochs.")
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_config(args.config)
    table, manifest = build_dataset(cfg, force=args.force)
    logger.info("Cached epochs: %s", table.distribution())
    logger.info("Excluded recordings: %d", len(manifest["exclusions"]))
    logger.info("Observed sampling rates (Hz): %s", manifest["observed_sfreq_hz"])
    logger.info("Resampled any recording: %s", manifest["resampled_any"])
    logger.info("Manifest: %s", artifacts_dir(cfg) / "manifest.json")


if __name__ == "__main__":
    main()

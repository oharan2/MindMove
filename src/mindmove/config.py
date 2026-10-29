"""Load and check experiment configuration."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

IMAGERY_RUNS = (4, 8, 12)
N_EEG_SUBJECTS = 109


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML config. Run commands from the project root.",
    )
    return parser


def find_project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "app.py").is_file() and (candidate / "src" / "mindmove").is_dir():
            return candidate
    raise ValueError(f"Could not find the MindMove project root from {config_path}.")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} did not contain a YAML mapping.")
    loaded["_config_path"] = str(config_path)
    loaded["_root"] = str(find_project_root(config_path))
    validate_config(loaded)
    return loaded


def public_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in cfg.items() if not str(key).startswith("_")}


def protocol_name(cfg: dict[str, Any]) -> str:
    evaluation = cfg.get("evaluation") or {}
    return str(evaluation.get("protocol", "held_out_run"))


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(cfg["_root"]) / path


def artifacts_dir(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg, cfg["paths"]["artifacts"])


def cache_dir(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg, cfg["paths"]["cache"])


def dataset_path(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg, cfg["dataset"]["path"])


def dataset_subjects(cfg: dict[str, Any]) -> list[int]:
    if protocol_name(cfg) == "unseen_subjects":
        evaluation = cfg["evaluation"]
        ordered = (
            list(evaluation["train_subjects"])
            + list(evaluation["val_subjects"])
            + list(evaluation["test_subjects"])
        )
        return _unique_ints(ordered)
    return [int(subject) for subject in cfg["dataset"]["subjects"]]


def dataset_runs(cfg: dict[str, Any]) -> list[int]:
    if protocol_name(cfg) == "unseen_subjects":
        return [int(run) for run in cfg["evaluation"]["runs"]]
    dataset = cfg["dataset"]
    return _unique_ints(list(dataset["train_runs"]) + list(dataset["val_runs"]) + list(dataset["test_runs"]))


def iter_subject_runs(cfg: dict[str, Any]):
    for subject in dataset_subjects(cfg):
        for run in dataset_runs(cfg):
            yield int(subject), int(run)


def ensure_artifact_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    root = artifacts_dir(cfg)
    paths = {
        "root": root,
        "cache": cache_dir(cfg),
        "models": root / "models",
        "metrics": root / "metrics",
        "figures": root / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    used = root / "config_used.yaml"
    used.write_text(yaml.safe_dump(public_config(cfg), sort_keys=False), encoding="utf-8")
    return paths


def validate_config(cfg: dict[str, Any]) -> None:
    for section in ("dataset", "preprocessing", "training", "model", "paths"):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"Config is missing the '{section}' section.")
    dataset = cfg["dataset"]
    if "path" not in dataset:
        raise ValueError("dataset.path is required.")
    for key in ("artifacts", "cache"):
        if key not in cfg["paths"]:
            raise ValueError(f"paths.{key} is required.")

    pre = cfg["preprocessing"]
    sfreq = float(pre["sfreq"])
    low = float(pre["low_freq"])
    high = float(pre["high_freq"])
    start = float(pre["epoch_start"])
    end = float(pre["epoch_end"])
    if sfreq <= 0:
        raise ValueError("preprocessing.sfreq must be positive.")
    if not (0 < low < high < sfreq / 2):
        raise ValueError("Bandpass must satisfy 0 < low_freq < high_freq < sfreq/2.")
    if not start < end:
        raise ValueError("epoch_start must be earlier than epoch_end.")
    if str(pre["reference"]).lower() != "average":
        raise ValueError("Only an average reference is implemented.")

    training = cfg["training"]
    if int(training["epochs"]) < 1:
        raise ValueError("training.epochs must be positive.")
    if int(training["patience"]) < 1:
        raise ValueError("training.patience must be positive.")
    if int(training["batch_size"]) < 2:
        raise ValueError("training.batch_size must be at least 2 so BatchNorm can train.")
    if float(training["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be positive.")
    device = str(training["device"]).lower()
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("training.device must be auto, cpu, cuda, or mps.")

    model = cfg["model"]
    if str(model["type"]) != "cnn1d":
        raise ValueError("model.type must be cnn1d.")
    if int(model["num_classes"]) != 2:
        raise ValueError("This experiment is binary left versus right.")
    dropout = float(model["dropout"])
    if not 0 <= dropout < 1:
        raise ValueError("model.dropout must be in [0, 1).")
    if int(model["csp_components"]) < 2:
        raise ValueError("model.csp_components must be at least 2.")
    for key in ("conv1_channels", "conv1_kernel", "conv2_channels", "conv2_kernel", "pool_kernel"):
        if int(model[key]) < 1:
            raise ValueError(f"model.{key} must be positive.")

    protocol = protocol_name(cfg)
    if protocol == "held_out_run":
        for key in ("subjects", "train_runs", "val_runs", "test_runs"):
            if key not in dataset:
                raise ValueError(f"dataset.{key} is required for the held-out-run protocol.")
        subjects = _unique_ints(dataset["subjects"], reject_duplicates=True)
        _check_subjects(subjects)
        groups = (
            ("train_runs", _unique_ints(dataset["train_runs"], reject_duplicates=True)),
            ("val_runs", _unique_ints(dataset["val_runs"], reject_duplicates=True)),
            ("test_runs", _unique_ints(dataset["test_runs"], reject_duplicates=True)),
        )
        _ensure_disjoint(groups, "run")
        for _, runs in groups:
            _check_imagery_runs(runs)
            if not runs:
                raise ValueError("Each of train_runs, val_runs, and test_runs must be non-empty.")
    elif protocol == "unseen_subjects":
        evaluation = cfg.get("evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError("evaluation is required for the unseen-subjects protocol.")
        groups = (
            ("train_subjects", _unique_ints(evaluation["train_subjects"], reject_duplicates=True)),
            ("val_subjects", _unique_ints(evaluation["val_subjects"], reject_duplicates=True)),
            ("test_subjects", _unique_ints(evaluation["test_subjects"], reject_duplicates=True)),
        )
        for name, subjects in groups:
            if not subjects:
                raise ValueError(f"evaluation.{name} must be non-empty.")
            _check_subjects(subjects)
        _ensure_disjoint(groups, "subject")
        runs = _unique_ints(evaluation["runs"], reject_duplicates=True)
        if not runs:
            raise ValueError("evaluation.runs must be non-empty.")
        _check_imagery_runs(runs)
    else:
        raise ValueError(
            f"Unknown evaluation.protocol '{protocol}'. Use held_out_run or unseen_subjects."
        )


def _unique_ints(values, reject_duplicates: bool = False) -> list[int]:
    result = []
    seen = set()
    for value in values:
        number = int(value)
        if number in seen:
            if reject_duplicates:
                raise ValueError(f"Duplicate entry {number} in config.")
            continue
        seen.add(number)
        result.append(number)
    return result


def _check_subjects(subjects: list[int]) -> None:
    bad = [subject for subject in subjects if subject < 1 or subject > N_EEG_SUBJECTS]
    if bad:
        raise ValueError(f"Subjects must be between 1 and {N_EEG_SUBJECTS}. Got {bad}.")


def _check_imagery_runs(runs: list[int]) -> None:
    bad = [run for run in runs if run not in IMAGERY_RUNS]
    if bad:
        raise ValueError(
            f"Runs {bad} are not left/right hand imagery. "
            "T1 means imagined left and T2 means imagined right only in runs 4, 8, and 12. "
            "Other runs use different tasks and cannot be mixed into this experiment."
        )


def _ensure_disjoint(groups: tuple[tuple[str, list[int]], ...], noun: str) -> None:
    owner: dict[int, str] = {}
    for name, values in groups:
        for value in values:
            if value in owner:
                raise ValueError(
                    f"{noun.capitalize()} {value} is in both {owner[value]} and {name}; "
                    "the same recording must not appear in two splits."
                )
            owner[value] = name

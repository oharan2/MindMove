"""Score saved models on the held-out split and write figures.

This is the only command that computes test-set metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from mindmove.baseline import load_baseline
from mindmove.config import (
    artifacts_dir,
    build_parser,
    configure_logging,
    ensure_artifact_dirs,
    load_config,
    protocol_name,
)
from mindmove.data import build_dataset, split_table
from mindmove.inference import (
    assert_same_training_ids,
    load_cnn_checkpoint,
    predict_baseline,
    predict_cnn,
)
from mindmove.metrics import compute_metrics, majority_class
from mindmove.models import resolve_device
from mindmove.preprocessing import CLASS_NAMES, display_name
from mindmove.visualization import (
    choose_display_channels,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_per_subject,
    plot_training_curves,
    plot_waveform_examples,
    save_figure,
)

logger = logging.getLogger(__name__)


def evaluate_models(cfg: dict) -> dict:
    import mne

    mne.set_log_level("ERROR")
    dirs = ensure_artifact_dirs(cfg)
    table, manifest = build_dataset(cfg)
    train, val, test = split_table(table, cfg)
    device = resolve_device(cfg["training"]["device"])
    cnn_path = dirs["models"] / "cnn.pt"
    baseline_path = dirs["models"] / "csp_lda.joblib"
    if not cnn_path.is_file():
        raise FileNotFoundError(f"Missing {cnn_path}. Run python -m mindmove.train first.")
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Missing {baseline_path}. Run python -m mindmove.baseline first.")

    model, payload = load_cnn_checkpoint(cnn_path, device)
    baseline = load_baseline(baseline_path)
    assert_same_training_ids(payload["train_trial_ids"], train.trial_ids(), "CNN normalization")
    assert_same_training_ids(baseline["train_trial_ids"], train.trial_ids(), "CSP + LDA")
    norm_path = dirs["root"] / "norm_stats.npz"
    with np.load(norm_path) as stored:
        if not np.allclose(stored["mean"], np.asarray(payload["norm_mean"])):
            raise RuntimeError("norm_stats.npz does not match the CNN checkpoint.")
        if not np.allclose(stored["std"], np.asarray(payload["norm_std"])):
            raise RuntimeError("norm_stats.npz does not match the CNN checkpoint.")
        assert_same_training_ids(stored["train_trial_ids"], train.trial_ids(), "norm_stats.npz")
    if int(payload["spec"]["n_channels"]) != train.data.shape[1]:
        raise RuntimeError("The CNN channel count does not match the cached epochs.")
    if int(payload["spec"]["n_times"]) != train.data.shape[2]:
        raise RuntimeError("The CNN epoch length does not match the cached epochs.")

    splits = {"validation": val, "test": test}
    model_predictions = {}
    for split_name, split in splits.items():
        cnn_pred, cnn_proba = predict_cnn(
            model, split.data, payload["norm_mean"], payload["norm_std"], device
        )
        csp_pred, csp_proba = predict_baseline(baseline["pipeline"], split.data)
        majority = majority_class(train.labels)
        majority_pred = np.full(len(split), majority, dtype=np.int64)
        model_predictions[split_name] = {
            "majority": (majority_pred, None),
            "csp_lda": (csp_pred, csp_proba),
            "cnn": (cnn_pred, cnn_proba),
        }

    scored = {
        split_name: {
            name: compute_metrics(splits[split_name].labels, prediction, splits[split_name].subjects)
            for name, (prediction, _proba) in predictions.items()
        }
        for split_name, predictions in model_predictions.items()
    }
    counts = np.bincount(np.asarray(train.labels, dtype=int), minlength=2)
    majority_index = int(np.argmax(counts))
    summary = {
        "experiment": _experiment_block(cfg, payload, manifest),
        "counts": {
            "train": train.distribution(),
            "validation": val.distribution(),
            "test": test.distribution(),
        },
        "exclusions": manifest.get("exclusions", []),
        "sampling": {
            "target_hz": float(cfg["preprocessing"]["sfreq"]),
            "observed_hz": manifest.get("observed_sfreq_hz", []),
            "resampled_any": bool(manifest.get("resampled_any", False)),
        },
        "chance_accuracy": 0.5,
        "majority_class": {
            "class_index": majority_index,
            "class_name": CLASS_NAMES[majority_index],
            "train_counts": {"left": int(counts[0]), "right": int(counts[1])},
            "tie_break": "lowest class index",
        },
        "validation": scored["validation"],
        "test": scored["test"],
        "waveform_examples": {},
        "figures": [],
    }
    _write_figures(cfg, dirs, test, payload, summary)
    _write_tables(dirs, summary, test, model_predictions["test"])
    report = render_report(summary)
    report_path = dirs["metrics"] / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    _write_json(dirs["metrics"] / "summary.json", summary)
    logger.info("Wrote %s", dirs["metrics"] / "summary.json")
    print(report)
    return summary


def render_report(summary: dict) -> str:
    experiment = summary["experiment"]
    counts = summary["counts"]
    lines = [
        "MindMove evaluation",
        experiment["description"],
        "",
        "Trial counts",
        f"  train      {counts['train']}",
        f"  validation {counts['validation']}",
        f"  test       {counts['test']}",
        f"Observed sampling rates (Hz): {summary['sampling']['observed_hz']}",
        f"Resampled any recording: {summary['sampling']['resampled_any']}",
        f"Excluded recordings: {len(summary['exclusions'])}",
    ]
    for item in summary["exclusions"]:
        lines.append(f"  S{int(item['subject']):03d} run {int(item['run'])}: {item['reason']}")
    lines.append("")
    lines.append(
        "Nominal chance accuracy for a balanced two-class task is 0.500. "
        f"The training majority class is {summary['majority_class']['class_name']} "
        f"(tie break: {summary['majority_class']['tie_break']})."
    )
    lines.append("")
    for split_name in ("validation", "test"):
        lines.append(split_name.capitalize())
        for model_name in ("majority", "csp_lda", "cnn"):
            metrics = summary[split_name][model_name]
            lines.append(
                f"  {model_name:10s}  accuracy {metrics['accuracy']:.3f}  "
                f"balanced {metrics['balanced_accuracy']:.3f}  "
                f"macro-F1 {metrics['macro_f1']:.3f}"
            )
            for class_name, scores in metrics["per_class"].items():
                lines.append(
                    f"    {class_name:5s}  precision {scores['precision']:.3f}  "
                    f"recall {scores['recall']:.3f}  f1 {scores['f1']:.3f}  "
                    f"support {scores['support']}"
                )
        lines.append("")
    test_cnn = summary["test"]["cnn"]["per_subject"]
    if test_cnn:
        accuracies = [row["accuracy"] for row in test_cnn]
        lines.append(
            "Per-subject CNN test accuracy ranged from "
            f"{min(accuracies):.3f} to {max(accuracies):.3f}."
        )
    if summary["experiment"]["protocol"] == "unseen_subjects":
        caveat = (
            "These figures describe this sample of trials. Trials from the same person are not "
            "independent. The test subjects were absent from training; this is not the default score."
        )
    else:
        caveat = (
            "These figures describe this sample of trials. Trials from the same person are not "
            "independent, and this protocol does not estimate accuracy on unseen people."
        )
    lines.append(caveat)
    lines.append("")
    return "\n".join(lines)


def _experiment_block(cfg: dict, payload: dict, manifest: dict) -> dict:
    protocol = protocol_name(cfg)
    if protocol == "held_out_run":
        description = (
            "Training, validation, and test sets are different recordings "
            f"(runs {_fmt_runs(cfg['dataset']['train_runs'])}, "
            f"{_fmt_runs(cfg['dataset']['val_runs'])}, and "
            f"{_fmt_runs(cfg['dataset']['test_runs'])}) from the same subjects "
            f"({_fmt_subjects(cfg['dataset']['subjects'])}). "
            "This measures generalization to a new session, not to unseen people."
        )
        subjects = [int(subject) for subject in cfg["dataset"]["subjects"]]
        runs = {
            "train_runs": [int(run) for run in cfg["dataset"]["train_runs"]],
            "val_runs": [int(run) for run in cfg["dataset"]["val_runs"]],
            "test_runs": [int(run) for run in cfg["dataset"]["test_runs"]],
        }
    else:
        evaluation = cfg["evaluation"]
        description = (
            "Training, validation, and test sets use disjoint subjects "
            f"{evaluation['train_subjects']}, {evaluation['val_subjects']}, and "
            f"{evaluation['test_subjects']}, with imagery runs {evaluation['runs']}. "
            "This is a separate unseen-subject experiment, not the default held-out-run score."
        )
        subjects = [int(subject) for subject in evaluation["train_subjects"] + evaluation["val_subjects"] + evaluation["test_subjects"]]
        runs = {"runs": [int(run) for run in evaluation["runs"]]}
    return {
        "protocol": protocol,
        "description": description,
        "subjects": subjects,
        **runs,
        "sfreq": float(cfg["preprocessing"]["sfreq"]),
        "bandpass_hz": [float(cfg["preprocessing"]["low_freq"]), float(cfg["preprocessing"]["high_freq"])],
        "epoch_window_s": [float(cfg["preprocessing"]["epoch_start"]), float(cfg["preprocessing"]["epoch_end"])],
        "n_samples": int(manifest.get("n_samples", payload["spec"]["n_times"])),
        "reference": str(cfg["preprocessing"]["reference"]),
        "seed": int(cfg["training"]["seed"]),
        "device": str(payload.get("device", "")),
        "csp_components": int(cfg["model"]["csp_components"]),
        "csp_reg": cfg["model"]["csp_reg"],
        "normalization": (
            "CNN inputs use a per-channel z-score fit on training epochs only. "
            "CSP+LDA uses the band-pass filtered epochs directly."
        ),
        "label_map": {"T1": "left (0)", "T2": "right (1)", "T0": "excluded"},
        "cnn": {
            "dropout": float(cfg["model"]["dropout"]),
            "epochs_requested": int(cfg["training"]["epochs"]),
            "epochs_ran": int(payload.get("epochs_ran", 0)),
            "best_epoch": int(payload.get("best_epoch", 0)),
            "best_val_balanced_accuracy": float(payload.get("best_val_balanced_accuracy", float("nan"))),
            "seconds": float(payload.get("seconds", float("nan"))),
        },
        "versions": _versions(),
    }


def _write_figures(cfg, dirs, test, payload, summary) -> None:
    figures = dirs["figures"]
    history = payload.get("history") or []
    if history:
        save_figure(
            plot_training_curves(history, int(payload.get("best_epoch", 0))),
            figures / "cnn_loss.png",
        )
    test_metrics = summary["test"]
    save_figure(
        plot_model_comparison(
            [
                {"name": "Majority", **_scores(test_metrics["majority"])},
                {"name": "CSP + LDA", **_scores(test_metrics["csp_lda"])},
                {"name": "CNN", **_scores(test_metrics["cnn"])},
            ]
        ),
        figures / "accuracy_comparison.png",
    )
    save_figure(
        plot_confusion_matrices(
            {
                "CSP + LDA": test_metrics["csp_lda"]["confusion_matrix"],
                "CNN": test_metrics["cnn"]["confusion_matrix"],
            }
        ),
        figures / "confusion_matrices.png",
    )
    subjects = [row["subject"] for row in test_metrics["cnn"]["per_subject"]]
    save_figure(
        plot_per_subject(
            subjects,
            {
                "CSP + LDA": [row["accuracy"] for row in test_metrics["csp_lda"]["per_subject"]],
                "CNN": [row["accuracy"] for row in test_metrics["cnn"]["per_subject"]],
            },
        ),
        figures / "per_subject_accuracy.png",
    )
    channels = choose_display_channels(test.ch_names)
    indexes = [test.ch_names.index(name) for name in channels]
    left = int(np.flatnonzero(test.labels == 0)[0])
    right = int(np.flatnonzero(test.labels == 1)[0])
    ids = test.trial_ids()
    save_figure(
        plot_waveform_examples(
            test.times,
            test.data[left, indexes],
            test.data[right, indexes],
            channels,
            f"{ids[left]}  {display_name(0)}",
            f"{ids[right]}  {display_name(1)}",
        ),
        figures / "sample_waveforms.png",
    )
    summary["waveform_examples"] = {
        "channels": channels,
        "left": str(ids[left]),
        "right": str(ids[right]),
    }
    summary["figures"] = [
        "figures/cnn_loss.png",
        "figures/accuracy_comparison.png",
        "figures/confusion_matrices.png",
        "figures/per_subject_accuracy.png",
        "figures/sample_waveforms.png",
    ]


def _write_tables(dirs, summary, test, predictions) -> None:
    rows = []
    for split_name in ("validation", "test"):
        for model_name, metrics in summary[split_name].items():
            rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "n_trials": metrics["n_trials"],
                }
            )
    pd.DataFrame(rows).to_csv(dirs["metrics"] / "model_comparison.csv", index=False)
    per_subject = []
    by_model = {
        name: {row["subject"]: row for row in summary["test"][name]["per_subject"]}
        for name in ("majority", "csp_lda", "cnn")
    }
    for subject in by_model["cnn"]:
        per_subject.append(
            {
                "subject": subject,
                "n_trials": by_model["cnn"][subject]["n_trials"],
                "n_left": by_model["cnn"][subject]["n_left"],
                "n_right": by_model["cnn"][subject]["n_right"],
                "majority_accuracy": by_model["majority"][subject]["accuracy"],
                "csp_lda_accuracy": by_model["csp_lda"][subject]["accuracy"],
                "cnn_accuracy": by_model["cnn"][subject]["accuracy"],
                "csp_lda_balanced_accuracy": by_model["csp_lda"][subject]["balanced_accuracy"],
                "cnn_balanced_accuracy": by_model["cnn"][subject]["balanced_accuracy"],
            }
        )
    pd.DataFrame(per_subject).to_csv(dirs["metrics"] / "per_subject_test.csv", index=False)
    cnn_pred, cnn_proba = predictions["cnn"]
    csp_pred, csp_proba = predictions["csp_lda"]
    majority_pred, _unused = predictions["majority"]
    np.savez_compressed(
        dirs["metrics"] / "test_predictions.npz",
        trial_ids=test.trial_ids(),
        subjects=test.subjects,
        runs=test.runs,
        y_true=test.labels,
        majority_pred=majority_pred,
        csp_pred=csp_pred,
        csp_proba=csp_proba,
        cnn_pred=cnn_pred,
        cnn_proba=cnn_proba,
    )
    text_parts = []
    for name, prediction in (("csp_lda", csp_pred), ("cnn", cnn_pred), ("majority", majority_pred)):
        text_parts.append(name)
        text_parts.append(
            classification_report(
                test.labels,
                prediction,
                labels=[0, 1],
                target_names=list(CLASS_NAMES),
                digits=3,
                zero_division=0,
            )
        )
    (dirs["metrics"] / "classification_reports.txt").write_text("\n".join(text_parts), encoding="utf-8")


def _fmt_subjects(subjects) -> str:
    values = [int(subject) for subject in subjects]
    if len(values) == 1:
        return str(values[0])
    if values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]}–{values[-1]}"
    return ", ".join(str(value) for value in values)


def _fmt_runs(runs) -> str:
    return ", ".join(str(int(run)) for run in runs)


def _scores(metrics: dict) -> dict:
    return {"accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"]}


def _versions() -> dict[str, str]:
    import platform

    import mne
    import sklearn
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mne": mne.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot write {type(value).__name__} to the metrics file.")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Evaluate saved CNN and CSP+LDA models on the held-out recordings.")
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_config(args.config)
    evaluate_models(cfg)
    logger.info("Figures: %s", artifacts_dir(cfg) / "figures")


if __name__ == "__main__":
    main()

"""CSP + LDA baseline fit on training epochs only."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from mindmove.config import artifacts_dir, build_parser, configure_logging, ensure_artifact_dirs, load_config
from mindmove.data import build_dataset, split_table
from mindmove.inference import predict_baseline
from mindmove.metrics import compute_metrics

logger = logging.getLogger(__name__)


def build_csp_lda(cfg: dict) -> Pipeline:
    """Four-component CSP followed by LDA.

    ``transform_into='average_power'`` with ``log=True`` returns the log of
    each spatial filter's power, the usual log-variance CSP feature.
    Components alternate between the largest and smallest eigenvalues
    (Blankertz ordering). Covariance uses Ledoit-Wolf shrinkage because a
    64-channel covariance is poorly conditioned when a class has few trials.
    Trace normalization is left off, matching the later Blankertz form rather
    than the original Koles form. None of these choices is searched on the
    test set.
    """
    model = cfg["model"]
    csp = CSP(
        n_components=int(model["csp_components"]),
        reg=model["csp_reg"],
        log=True,
        cov_est="concat",
        transform_into="average_power",
        norm_trace=False,
        component_order="alternate",
    )
    lda = LinearDiscriminantAnalysis(solver="svd")
    return Pipeline([("csp", csp), ("lda", lda)])


def fit_baseline(train, val, cfg: dict, out_dir: Path) -> dict:
    """Fit on ``train`` and score ``val``. The test set is not an argument."""
    overlap = set(train.trial_ids().tolist()) & set(val.trial_ids().tolist())
    if overlap:
        raise ValueError(f"Training and validation trials overlap, for example {sorted(overlap)[:5]}.")
    pipeline = build_csp_lda(cfg)
    training_epochs = np.array(train.data, dtype=np.float64, copy=True)
    pipeline.fit(training_epochs, np.asarray(train.labels, dtype=np.int64))
    features = pipeline.named_steps["csp"].transform(training_epochs[:2])
    expected = int(cfg["model"]["csp_components"])
    if features.ndim != 2 or features.shape[1] != expected:
        raise RuntimeError(f"CSP features have shape {features.shape}, expected (*, {expected}).")
    prediction, _proba = predict_baseline(pipeline, val.data)
    metrics = compute_metrics(val.labels, prediction, val.subjects)
    out_dir = Path(out_dir)
    model_path = out_dir / "models" / "csp_lda.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "train_trial_ids": train.trial_ids().tolist(),
            "n_components": expected,
            "reg": cfg["model"]["csp_reg"],
            "class_names": ["left", "right"],
        },
        model_path,
    )
    return {"path": str(model_path), "validation": metrics}


def load_baseline(path: Path) -> dict:
    payload = joblib.load(path)
    if "pipeline" not in payload:
        raise RuntimeError(f"{path} is not a MindMove CSP+LDA file.")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Train the CSP + LDA baseline. The test run is not scored here.")
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_config(args.config)
    import mne

    mne.set_log_level("ERROR")
    dirs = ensure_artifact_dirs(cfg)
    table, _manifest = build_dataset(cfg)
    train, val, test = split_table(table, cfg)
    logger.info(
        "Train %s | validation %s | test trials held out: %d",
        train.distribution(),
        val.distribution(),
        len(test),
    )
    del test
    result = fit_baseline(train, val, cfg, dirs["root"])
    validation = result["validation"]
    logger.info(
        "Validation accuracy %.3f, balanced accuracy %.3f, macro-F1 %.3f",
        validation["accuracy"],
        validation["balanced_accuracy"],
        validation["macro_f1"],
    )
    logger.info("Saved %s", result["path"])
    logger.info("Test metrics are written by python -m mindmove.evaluate.")


if __name__ == "__main__":
    main()

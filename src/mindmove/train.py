"""Train the 1D CNN. The test set is not an argument of the fit function."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mindmove.config import artifacts_dir, build_parser, configure_logging, ensure_artifact_dirs, load_config
from mindmove.data import build_dataset, split_table
from mindmove.inference import save_cnn_checkpoint
from mindmove.models import EEGCNN, resolve_device, set_seed
from mindmove.preprocessing import apply_channel_norm, fit_channel_norm

logger = logging.getLogger(__name__)


def fit_cnn(train, val, cfg: dict, out_dir: Path) -> dict:
    """Train on ``train`` and stop using validation balanced accuracy.

    Normalization statistics are computed from ``train`` only. The returned
    checkpoint is the validation-best weights, not the final epoch.
    """
    overlap = set(train.trial_ids().tolist()) & set(val.trial_ids().tolist())
    if overlap:
        raise ValueError(f"Training and validation trials overlap, for example {sorted(overlap)[:5]}.")
    training = cfg["training"]
    seed = int(training["seed"])
    set_seed(seed)
    device = resolve_device(training["device"])
    mean, std = fit_channel_norm(train.data)
    train_x = apply_channel_norm(train.data, mean, std)
    val_x = apply_channel_norm(val.data, mean, std)
    if not np.isfinite(train_x).all() or not np.isfinite(val_x).all():
        raise RuntimeError("Normalized EEG contains non-finite values.")

    model_cfg = cfg["model"]
    model = EEGCNN(
        n_channels=int(train.data.shape[1]),
        n_times=int(train.data.shape[2]),
        n_classes=int(model_cfg["num_classes"]),
        dropout=float(model_cfg["dropout"]),
        conv1_channels=int(model_cfg["conv1_channels"]),
        conv1_kernel=int(model_cfg["conv1_kernel"]),
        conv2_channels=int(model_cfg["conv2_channels"]),
        conv2_kernel=int(model_cfg["conv2_kernel"]),
        pool_kernel=int(model_cfg["pool_kernel"]),
    ).to(device)
    # Weight decay stays at Adam's default of 0; it is not a tuned hyperparameter.
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]), weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()
    batch_size = int(training["batch_size"])
    drop_last = len(train) > batch_size and len(train) % batch_size == 1
    loader = _loader(train_x, train.labels, batch_size, shuffle=True, seed=seed, drop_last=drop_last)

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict] = []
    max_epochs = int(training["epochs"])
    patience = int(training["patience"])
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        train_loss = _train_epoch(model, loader, optimizer, loss_fn, device)
        val_loss, val_pred, _val_proba = _predict_array(model, val_x, val.labels, device, batch_size)
        val_accuracy = float(accuracy_score(val.labels, val_pred))
        val_balanced = float(balanced_accuracy_score(val.labels, val_pred))
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}.")
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_accuracy": val_accuracy,
                "val_balanced_accuracy": val_balanced,
            }
        )
        logger.info(
            "Epoch %d/%d  train loss %.4f  val loss %.4f  val balanced accuracy %.3f",
            epoch,
            max_epochs,
            train_loss,
            val_loss,
            val_balanced,
        )
        if val_balanced > best_score:
            best_score = val_balanced
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d. Best epoch is %d.", epoch, best_epoch)
                break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    elapsed = time.perf_counter() - started
    out_dir = Path(out_dir)
    checkpoint = out_dir / "models" / "cnn.pt"
    extra = {
        "norm_mean": mean,
        "norm_std": std,
        "train_trial_ids": train.trial_ids().tolist(),
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_balanced_accuracy": float(best_score),
        "epochs_ran": len(history),
        "device": str(device),
        "seed": seed,
        "seconds": float(elapsed),
        "class_names": ["left", "right"],
    }
    save_cnn_checkpoint(checkpoint, model, extra)
    norm_path = out_dir / "norm_stats.npz"
    np.savez(
        norm_path,
        mean=mean,
        std=std,
        train_trial_ids=np.array(extra["train_trial_ids"]),
    )
    history_path = out_dir / "metrics" / "cnn_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return {
        "path": str(checkpoint),
        "norm_path": str(norm_path),
        "best_epoch": int(best_epoch),
        "best_val_balanced_accuracy": float(best_score),
        "epochs_ran": len(history),
        "device": str(device),
        "seconds": float(elapsed),
        "history": history,
    }


def _loader(data, labels, batch_size: int, shuffle: bool, seed: int, drop_last: bool) -> DataLoader:
    tensors = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64)),
    )
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        tensors,
        batch_size=int(batch_size),
        shuffle=shuffle,
        generator=generator,
        drop_last=drop_last,
    )


def _train_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total = 0.0
    count = 0
    for features, labels in loader:
        if int(features.shape[0]) < 2:
            continue
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(features), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * int(features.shape[0])
        count += int(features.shape[0])
    if count == 0:
        raise RuntimeError("The training loader produced no usable batches.")
    return total / count


@torch.no_grad()
def _predict_array(model, data, labels, device, batch_size: int):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    features = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
    targets = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))
    losses = 0.0
    predictions = []
    probabilities = []
    for start in range(0, len(features), int(batch_size)):
        batch_x = features[start : start + int(batch_size)].to(device)
        batch_y = targets[start : start + int(batch_size)].to(device)
        logits = model(batch_x)
        losses += float(loss_fn(logits, batch_y).item())
        proba = torch.softmax(logits, dim=-1)
        predictions.append(proba.argmax(dim=-1).cpu().numpy())
        probabilities.append(proba.cpu().numpy())
    return losses / len(features), np.concatenate(predictions), np.concatenate(probabilities)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Train the 1D CNN. The test run is not used for training or early stopping.")
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
    result = fit_cnn(train, val, cfg, dirs["root"])
    logger.info(
        "Best validation balanced accuracy %.3f at epoch %d (%d epochs run, %.1f s, %s)",
        result["best_val_balanced_accuracy"],
        result["best_epoch"],
        result["epochs_ran"],
        result["seconds"],
        result["device"],
    )
    logger.info("Saved %s", result["path"])
    logger.info("Test metrics are written by python -m mindmove.evaluate.")


if __name__ == "__main__":
    main()

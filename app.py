"""MindMove dashboard. Replay of prerecorded EEG, not a live brain-computer interface."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import matplotlib

matplotlib.use("Agg")

if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from mindmove.baseline import load_baseline
from mindmove.config import artifacts_dir, load_config
from mindmove.data import load_cached_dataset, split_table
from mindmove.inference import (
    assert_same_training_ids,
    binary_cursor_offset,
    load_cnn_checkpoint,
    predict_baseline,
    predict_cnn,
)
from mindmove.models import resolve_device
from mindmove.preprocessing import DISPLAY_NAMES, display_name
from mindmove.visualization import (
    choose_display_channels,
    plot_spectrum,
    plot_trial_waveforms,
)

st.set_page_config(page_title="MindMove", layout="wide")
REPLAY_NOTE = (
    "This dashboard replays prerecorded EEG from the PhysioNet EEG Motor "
    "Movement/Imagery Dataset. It is not a live brain-computer interface."
)


def main() -> None:
    st.title("MindMove")
    st.info(REPLAY_NOTE)
    config_path = st.sidebar.text_input("Config", "configs/default.yaml")
    if st.sidebar.button("Reload artifacts"):
        st.cache_resource.clear()
        st.rerun()
    view = st.sidebar.radio(
        "View",
        [
            "EEG Explorer",
            "Model Performance",
            "Brain-Controlled Cursor",
            "Prediction Explorer",
        ],
    )
    st.sidebar.caption(
        "Train from the project root with python -m mindmove.data, "
        "baseline, train, and evaluate. Then reopen this app."
    )
    try:
        runtime = get_runtime(config_path)
    except Exception as exc:
        st.error(str(exc))
        _commands()
        st.caption("Data: PhysioNet EEG Motor Movement/Imagery Dataset (Schalk et al. 2004; Goldberger et al. 2000).")
        return
    pages = {
        "EEG Explorer": render_explorer,
        "Model Performance": render_performance,
        "Brain-Controlled Cursor": render_cursor,
        "Prediction Explorer": render_predictions,
    }
    pages[view](runtime)
    st.caption("Data: PhysioNet EEG Motor Movement/Imagery Dataset (Schalk et al. 2004; Goldberger et al. 2000).")


def get_runtime(config_path: str) -> dict:
    cfg = load_config(config_path)
    if not (artifacts_dir(cfg) / "manifest.json").is_file():
        return {"cfg": cfg, "ready": False}
    return _cached_runtime(str(Path(config_path).expanduser().resolve()))


@st.cache_resource
def _cached_runtime(config_path: str) -> dict:
    cfg = load_config(config_path)
    table, manifest = load_cached_dataset(cfg)
    train, val, test = split_table(table, cfg)
    device = resolve_device(cfg["training"]["device"])
    root = artifacts_dir(cfg)
    cnn = None
    cnn_path = root / "models" / "cnn.pt"
    if cnn_path.is_file():
        model, payload = load_cnn_checkpoint(cnn_path, device)
        assert_same_training_ids(payload["train_trial_ids"], train.trial_ids(), "CNN")
        pred, proba = predict_cnn(model, test.data, payload["norm_mean"], payload["norm_std"], device)
        cnn = {"model": model, "payload": payload, "pred": pred, "proba": proba}
    baseline = None
    baseline_path = root / "models" / "csp_lda.joblib"
    if baseline_path.is_file():
        stored = load_baseline(baseline_path)
        assert_same_training_ids(stored["train_trial_ids"], train.trial_ids(), "CSP + LDA")
        pred, proba = predict_baseline(stored["pipeline"], test.data)
        baseline = {"stored": stored, "pred": pred, "proba": proba}
    summary = None
    summary_path = root / "metrics" / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "cfg": cfg,
        "ready": True,
        "table": table,
        "splits": {"train": train, "validation": val, "test": test},
        "manifest": manifest,
        "cnn": cnn,
        "baseline": baseline,
        "summary": summary,
        "device": str(device),
    }


def render_explorer(runtime: dict) -> None:
    st.header("EEG Explorer")
    if not runtime["ready"]:
        _commands()
        return
    table = runtime["table"]
    subject = st.selectbox("Participant", sorted(int(item) for item in np.unique(table.subjects)))
    subject_runs = sorted(int(item) for item in np.unique(table.runs[table.subjects == subject]))
    run = st.selectbox("Run", subject_runs)
    indices = np.flatnonzero((table.subjects == subject) & (table.runs == run))
    labels = table.trial_ids()
    choice = st.selectbox(
        "Trial",
        range(len(indices)),
        format_func=lambda local: (
            f"{labels[int(indices[local])]}  {display_name(int(table.labels[int(indices[local])]))}"
        ),
    )
    index = int(indices[int(choice)])
    st.subheader(display_name(int(table.labels[index])))
    st.write(
        f"Trial `{labels[index]}`. Cue onset at {table.onsets[index]:.3f} s in the recording. "
        f"Window {table.times[0]:.3f}–{table.times[-1]:.3f} s after the cue. "
        "Rest (T0) trials are not included."
    )
    channels = _channel_picker(table.ch_names, "explorer_channels")
    if not channels:
        st.warning("Select at least one channel.")
        return
    picked = _slice_channels(table.data[index], table.ch_names, channels)
    st.dataframe(_amplitude_table(picked, channels), hide_index=True)
    waveform = plot_trial_waveforms(table.times, picked, channels, f"{labels[index]}  {display_name(int(table.labels[index]))}")
    st.pyplot(waveform)
    spectrum = plot_spectrum(
        picked,
        table.sfreq,
        channels,
        "Spectrum of this epoch (dashed lines mark the 8–30 Hz passband)",
    )
    st.pyplot(spectrum)


def render_performance(runtime: dict) -> None:
    st.header("Model Performance")
    summary = runtime.get("summary")
    if not summary:
        st.warning("Saved metrics were not found. Run python -m mindmove.evaluate. This page does not retrain.")
        _commands()
        return
    st.caption(summary["experiment"]["description"])
    counts = pd.DataFrame(summary["counts"]).T
    st.subheader("Trial counts")
    st.dataframe(counts)
    st.subheader("Held-out test")
    st.dataframe(_metric_frame(summary["test"]), hide_index=True)
    st.caption(
        f"Nominal chance accuracy is {summary['chance_accuracy']:.2f}. "
        f"Majority class on the training set: {summary['majority_class']['class_name']} "
        f"(tie break: {summary['majority_class']['tie_break']})."
    )
    root = artifacts_dir(runtime["cfg"])
    for name in (
        "cnn_loss.png",
        "accuracy_comparison.png",
        "confusion_matrices.png",
        "per_subject_accuracy.png",
        "sample_waveforms.png",
    ):
        path = root / "figures" / name
        if path.is_file():
            st.image(str(path))
    per_subject = root / "metrics" / "per_subject_test.csv"
    if per_subject.is_file():
        st.subheader("Per-subject test accuracy")
        st.dataframe(pd.read_csv(per_subject), hide_index=True)
    with st.expander("Validation scores used for CNN early stopping"):
        st.dataframe(_metric_frame(summary["validation"]), hide_index=True)
    exclusions = summary.get("exclusions") or []
    st.subheader("Excluded recordings")
    if exclusions:
        st.dataframe(pd.DataFrame(exclusions), hide_index=True)
    else:
        st.write("None.")
    sampling = summary["sampling"]
    st.write(
        f"Target sampling rate {sampling['target_hz']} Hz. "
        f"Observed {sampling['observed_hz']}. "
        f"Resampled any recording: {sampling['resampled_any']}."
    )
    with st.expander("Experiment configuration"):
        st.json(summary["experiment"])


def render_cursor(runtime: dict) -> None:
    st.header("Brain-controlled cursor")
    st.warning("Prerecorded replay. The cursor is not reading a live EEG headset.")
    if not runtime["ready"]:
        _commands()
        return
    test = runtime["splits"]["test"]
    available = _available_models(runtime)
    if not available:
        st.warning("Train a model before using the cursor.")
        _commands()
        return
    model_name = st.radio("Model", list(available), horizontal=True, format_func=_model_label)
    n_trials = len(test)
    n_times = int(test.data.shape[-1])
    if "cursor_pick" not in st.session_state:
        st.session_state.cursor_pick = 0
    if "playhead" not in st.session_state:
        st.session_state.playhead = None
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "speed" not in st.session_state:
        st.session_state.speed = 1.0

    columns = st.columns(4)
    columns[0].button("Play", on_click=_on_play)
    columns[1].button("Pause", on_click=_on_pause)
    columns[2].button("Next trial", on_click=_bump_trial, args=(1, n_trials))
    columns[3].button("Previous trial", on_click=_bump_trial, args=(-1, n_trials))
    st.slider("Playback speed", min_value=0.5, max_value=8.0, step=0.5, key="speed")
    st.selectbox(
        "Held-out test trial",
        list(range(n_trials)),
        format_func=lambda index: _trial_caption(test, int(index)),
        key="cursor_pick",
        on_change=_reset_playback,
    )
    index = int(st.session_state.cursor_pick) % n_trials
    playhead = st.session_state.playhead
    revealed = playhead is None
    prediction, proba = _prediction(runtime, model_name, index)
    offset = binary_cursor_offset(proba) if revealed else 0.0
    _cursor_track(offset)
    truth = int(test.labels[index])
    if revealed:
        correct = int(prediction) == truth
        st.write(
            f"Model: **{_model_label(model_name)}**. "
            f"Prediction: **{display_name(int(prediction))}** "
            f"(confidence {float(np.max(proba)):.2f}). "
            f"Ground truth: **{display_name(truth)}**. "
            f"{'Correct' if correct else 'Incorrect'}."
        )
        st.caption(
            "One prediction is made for the whole epoch. "
            "The cursor does not update from smaller pieces of this same epoch."
        )
        shown = test.data[index]
        shown_times = test.times
        progress = 1.0
    else:
        st.write(
            f"Replaying the epoch. Ground truth: **{display_name(truth)}**. "
            "The prediction is shown when the full epoch has been replayed."
        )
        stop = max(2, int(playhead))
        shown = test.data[index, :, :stop]
        shown_times = test.times[:stop]
        progress = float(playhead) / float(n_times)
    st.progress(min(progress, 1.0))
    channels = _channel_picker(test.ch_names, "cursor_channels")
    if channels:
        picked = _slice_channels(shown, test.ch_names, channels)
        figure = plot_trial_waveforms(shown_times, picked, channels, _trial_caption(test, index))
        st.pyplot(figure)
    if st.session_state.playing:
        step = max(1, int(float(st.session_state.speed) * float(test.sfreq) * 0.08))
        current = 0 if playhead is None else int(playhead)
        current += step
        if current >= n_times:
            st.session_state.playhead = None
            st.session_state.playing = False
            st.rerun()
        st.session_state.playhead = current
        time.sleep(0.08)
        st.rerun()


def render_predictions(runtime: dict) -> None:
    st.header("Prediction Explorer")
    if not runtime["ready"]:
        _commands()
        return
    test = runtime["splits"]["test"]
    available = _available_models(runtime)
    if not available:
        st.warning("Train a model before inspecting predictions.")
        _commands()
        return
    model_name = st.radio(
        "Model",
        list(available),
        horizontal=True,
        format_func=_model_label,
        key="prediction_model",
    )
    subjects = sorted(int(item) for item in np.unique(test.subjects))
    subject = st.selectbox("Participant", subjects, key="prediction_subject")
    indices = np.flatnonzero(test.subjects == subject)
    local = st.selectbox(
        "Held-out trial",
        range(len(indices)),
        format_func=lambda item: _trial_caption(test, int(indices[item])),
        key="prediction_trial",
    )
    index = int(indices[int(local)])
    prediction, proba = _prediction(runtime, model_name, index)
    truth = int(test.labels[index])
    correct = int(prediction) == truth
    st.write(f"Model: **{_model_label(model_name)}**")
    st.write(f"Ground truth: **{display_name(truth)}**")
    st.write(f"Prediction: **{display_name(int(prediction))}** ({'correct' if correct else 'incorrect'})")
    st.dataframe(
        pd.DataFrame({"class": list(DISPLAY_NAMES), "probability": proba.astype(float)}),
        hide_index=True,
    )
    st.caption("The ground-truth label is displayed for evaluation. It is not passed to the model.")
    channels = _channel_picker(test.ch_names, "prediction_channels")
    if channels:
        picked = _slice_channels(test.data[index], test.ch_names, channels)
        figure = plot_trial_waveforms(test.times, picked, channels, _trial_caption(test, index))
        st.pyplot(figure)


def _available_models(runtime: dict) -> list[str]:
    names = []
    if runtime.get("cnn") is not None:
        names.append("cnn")
    if runtime.get("baseline") is not None:
        names.append("csp_lda")
    return names


def _model_label(name: str) -> str:
    return {"cnn": "CNN", "csp_lda": "CSP + LDA"}.get(name, name)


def _prediction(runtime: dict, model_name: str, index: int):
    bundle = runtime["cnn"] if model_name == "cnn" else runtime["baseline"]
    return int(bundle["pred"][index]), np.asarray(bundle["proba"][index], dtype=float)


def _channel_picker(ch_names, key: str) -> list[str]:
    default = choose_display_channels(ch_names)
    return st.multiselect("Channels", list(ch_names), default=default, key=key)


def _slice_channels(data, ch_names, selected: list[str]) -> np.ndarray:
    indexes = [list(ch_names).index(name) for name in selected]
    return np.asarray(data)[indexes]


def _amplitude_table(data, ch_names) -> pd.DataFrame:
    microvolts = np.asarray(data, dtype=float) * 1e6
    rows = []
    for name, signal in zip(ch_names, microvolts):
        rows.append(
            {
                "Channel": name,
                "Std (µV)": float(signal.std()),
                "Peak-to-peak (µV)": float(signal.max() - signal.min()),
                "Min (µV)": float(signal.min()),
                "Max (µV)": float(signal.max()),
            }
        )
    return pd.DataFrame(rows)


def _trial_caption(test, index: int) -> str:
    return f"{test.trial_ids()[index]}  {display_name(int(test.labels[index]))}"


def _cursor_track(position: float) -> None:
    position = float(np.clip(position, -1.0, 1.0))
    percent = (position + 1.0) / 2.0 * 100.0
    st.markdown(
        f"""
        <div style="background:#f4f6f8;border-radius:12px;padding:16px 12px;margin:8px 0 16px 0;">
          <div style="display:flex;justify-content:space-between;font-weight:600;color:#222;">
            <span>Imagined left</span><span>Imagined right</span>
          </div>
          <div style="position:relative;height:36px;margin-top:10px;background:linear-gradient(90deg,#d9e8ff,#f7f7f7,#ffd9d9);border-radius:18px;">
            <div style="position:absolute;left:50%;top:0;bottom:0;width:2px;background:#888;"></div>
            <div style="position:absolute;left:{percent:.2f}%;top:4px;width:28px;height:28px;margin-left:-14px;background:#1f4b99;border-radius:50%;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _on_play() -> None:
    st.session_state.playing = True
    if st.session_state.get("playhead") is None:
        st.session_state.playhead = 0


def _on_pause() -> None:
    st.session_state.playing = False


def _reset_playback() -> None:
    st.session_state.playhead = None
    st.session_state.playing = False


def _bump_trial(delta: int, n_trials: int) -> None:
    current = int(st.session_state.get("cursor_pick", 0))
    st.session_state.cursor_pick = (current + int(delta)) % int(n_trials)
    _reset_playback()


def _commands() -> None:
    st.markdown(
        """
From the project root, with the virtual environment active:

```bash
python -m mindmove.data --config configs/default.yaml
python -m mindmove.baseline --config configs/default.yaml
python -m mindmove.train --config configs/default.yaml
python -m mindmove.evaluate --config configs/default.yaml
streamlit run app.py
```
"""
    )


def _metric_frame(block: dict) -> pd.DataFrame:
    rows = []
    labels = {"majority": "Majority class", "csp_lda": "CSP + LDA", "cnn": "CNN"}
    for key, label in labels.items():
        metrics = block[key]
        rows.append(
            {
                "Model": label,
                "Accuracy": metrics["accuracy"],
                "Balanced accuracy": metrics["balanced_accuracy"],
                "Macro-F1": metrics["macro_f1"],
                "Trials": metrics["n_trials"],
            }
        )
    return pd.DataFrame(rows)


main()

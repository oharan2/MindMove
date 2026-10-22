# MindMove

MindMove trains a small decoder to tell imagined left-hand movement from imagined right-hand movement in prerecorded EEG. It is a proof of concept on the PhysioNet EEG Motor Movement/Imagery Dataset. It does not read a live headset, and it is not a medical device.

## Dataset

Recordings come from the [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (109 subjects, 64 EEG channels, 160 Hz), loaded with `mne.datasets.eegbci`. PhysioNet asks anyone who uses these recordings to cite:

- Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., and Wolpaw, J.R. (2004). BCI2000: a general-purpose brain-computer interface (BCI) system. *IEEE Transactions on Biomedical Engineering* 51(6):1034–1043.
- Goldberger, A.L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation* 101(23):e215–e220.

Only motor-imagery runs **4, 8, and 12** are used. In those runs the annotations mean:

| Annotation | Meaning in runs 4, 8, and 12 |
| --- | --- |
| T1 | Imagined opening and closing of the **left** hand (class 0) |
| T2 | Imagined opening and closing of the **right** hand (class 1) |
| T0 | Rest. Excluded from training and scoring |

T1 and T2 mean different tasks in the other runs (actual movement, or both fists versus both feet). Those runs are rejected by the config check so they cannot be mixed in by accident.

## Experimental design

The default config (`configs/default.yaml`) uses subjects 1–20:

| Split | Recording | Role |
| --- | --- | --- |
| Train | Run 4 | Fit CSP, LDA, the CNN, and the CNN's normalization |
| Validation | Run 8 | Early stopping and the CNN checkpoint |
| Test | Run 12 | Scored once by `python -m mindmove.evaluate` |

These are different recordings from the **same people**. A good score means the decoder survived a new session, not a new person. An optional disjoint-subject config is described below and is stored separately.

Each imagery cue becomes one epoch from 1.0 s to 3.0 s after onset (half-open). At 160 Hz that is **320 samples**. MNE's epoch endpoint is inclusive, so the sample exactly at 3.0 s is dropped. There is one window per cue. Trial ids are `S{subject}_R{run}_T{index}`, and the train, validation, and test ids are required to be disjoint.

Preprocessing, per recording (not on a file that already concatenates splits):

1. Standardize the 64 EEG channel names and keep them in a fixed order.
2. Average reference.
3. Resample to 160 Hz only if the file is not already 160 Hz.
4. Zero-phase 8–30 Hz FIR bandpass.
5. Epoch, drop rest, drop a non-finite epoch if one appears.

No amplitude threshold is applied. The 8–30 Hz filter is zero-phase, so it uses samples from both directions inside that recording. That is appropriate for this offline analysis and is not an online decoder.

The CNN input is a per-channel z-score. Mean and standard deviation are computed on **training epochs only** (over trials and time, population standard deviation, floored at `1e-6`) and saved in `artifacts/norm_stats.npz` and in the checkpoint. CSP + LDA does **not** use that z-score. It sees the band-passed epochs, because CSP estimates spatial covariance itself. Its features are the log power of four spatial filters (`transform_into="average_power"`, `log=True`), ordered by alternating large and small eigenvalues. Covariance uses Ledoit–Wolf shrinkage. Trace normalization is off. Those choices are fixed in the config; they were not searched on the test set.

The CNN is the 1D network in the plan: Conv1d 64→32 (kernel 15), batch norm, ELU, average pool, Conv1d 32→64 (kernel 7), batch norm, ELU, adaptive average pool, dropout 0.4, linear 64→2. Training uses Adam at learning rate 0.001, cross-entropy, batch size 32, seed 42, at most 50 epochs, and patience 8 on **validation balanced accuracy**. The saved weights are the best validation epoch, not the last epoch. Weight decay is 0. The device is CPU unless CUDA or MPS is actually available.

`python -m mindmove.baseline` and `python -m mindmove.train` report validation only. Test labels are not an argument of either fit function. `python -m mindmove.evaluate` is the only command that scores the test recordings. It also reports a majority-class baseline fit on the training labels (ties resolve to left) and a nominal chance level of 0.50 for a balanced two-class task.

## Install

Use Python 3.11 or 3.12. The commands below were run with Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

`--no-deps` keeps the PyTorch build selected by `requirements.txt` from being replaced.

## Commands

Run these from the project root, in order. The first command downloads the EDF files into `data/eegbci/` (about 60 files for the default 20 subjects).

```bash
python -m mindmove.data --config configs/default.yaml
python -m mindmove.baseline --config configs/default.yaml
python -m mindmove.train --config configs/default.yaml
python -m mindmove.evaluate --config configs/default.yaml
streamlit run app.py
```

Re-running data is a no-op while the preprocessing config is unchanged. `python -m mindmove.data --force` rebuilds the cache. Models, metrics, and figures are written under `artifacts/`. A one-subject execution check lives in `configs/smoke.yaml` and writes to `artifacts/smoke/`; those numbers are not the result below.

## Dashboard

`streamlit run app.py` opens four views:

- **EEG Explorer** — one cached trial, with waveforms, amplitude, and a spectrum. Sensorimotor channels C3, Cz, and C4 are selected first.
- **Model Performance** — the saved test metrics, confusion matrices, loss curve, and per-subject scores. Opening the page does not train.
- **Brain-controlled cursor** — replays a held-out epoch and moves a cursor once from that epoch's prediction (left for imagined left hand, right for imagined right hand). It does not emit a new prediction on every time step of the same epoch.
- **Prediction Explorer** — ground truth, predicted class, and probabilities for one held-out trial, labeled with the model that produced them.

The ground-truth label is shown for evaluation. Inference functions do not take it as an input. Every page states that this is a replay of prerecorded EEG.

## Results

These numbers are the held-out-run evaluation of subjects 1–20 (`configs/default.yaml`), written by `python -m mindmove.evaluate` to `artifacts/metrics/summary.json` on 22 October 2026. Scores below are rounded to three decimals. The one-subject smoke config is not included.

All 60 recordings (subjects 1–20, runs 4, 8, and 12) were already sampled at 160 Hz, so none was resampled. No recording was dropped, and no epoch was dropped for non-finite values. Each recording has 15 rest (`T0`) events; all 900 were excluded. The remaining 900 imagery epochs are balanced as follows.

| Split | Run | Trials | Left | Right |
| --- | --- | --- | --- | --- |
| Train | 4 | 300 | 150 | 150 |
| Validation | 8 | 300 | 153 | 147 |
| Test | 12 | 300 | 152 | 148 |

Nominal chance accuracy for this balanced two-class task is 0.500. The training counts are tied, so the majority-class baseline predicts left.

Held-out test (run 12, 300 trials):

| Model | Accuracy | Balanced accuracy | Macro-F1 |
| --- | --- | --- | --- |
| Majority class | 0.507 | 0.500 | 0.336 |
| CSP + LDA | 0.497 | 0.498 | 0.494 |
| CNN | 0.597 | 0.597 | 0.596 |

Per-class test scores (precision / recall / F1), with support 152 left and 148 right:

| Model | Left | Right |
| --- | --- | --- |
| Majority class | 0.507 / 1.000 / 0.673 | 0.000 / 0.000 / 0.000 |
| CSP + LDA | 0.504 / 0.414 / 0.455 | 0.491 / 0.581 / 0.533 |
| CNN | 0.610 / 0.566 / 0.587 | 0.585 / 0.628 / 0.606 |

The CNN test confusion matrix, rows true left then true right, columns predicted left then predicted right, is `[[86, 66], [55, 93]]`. CSP + LDA is `[[63, 89], [62, 86]]`.

Per-subject CNN test accuracy, 15 trials each, ranged from 0.400 to 0.800.

The CNN ran on CPU for 1.6 seconds, completed 23 of 50 epochs, and kept the weights from epoch 15 (validation balanced accuracy 0.583). Validation scores, which chose that checkpoint and were not the test set, were majority balanced accuracy 0.500, CSP + LDA 0.501, and CNN 0.583.

CSP + LDA, fit once on all 20 subjects together, stayed at chance on the held-out run (balanced accuracy 0.498). The CNN was above the majority baseline and above 0.500 on the same trials. These figures describe this sample. Trials from the same person are not independent, and this protocol does not estimate accuracy on unseen people.

## Optional unseen-subject split

`configs/subject_holdout.yaml` trains on subjects 1–12, validates on 13–16, and tests on 17–20, using runs 4, 8, and 12 inside each group. It writes to `artifacts/subject_holdout/` and does not replace the default model.

```bash
python -m mindmove.baseline --config configs/subject_holdout.yaml
python -m mindmove.train --config configs/subject_holdout.yaml
python -m mindmove.evaluate --config configs/subject_holdout.yaml
```

The cache from the default download is reused. Run this only as a separate question. Do not treat it as the headline score, and do not retune the default model from it.

That split was run once, after the default models were saved, and written to `artifacts/subject_holdout/metrics/summary.json`. Training used subjects 1–12 (540 trials, 274 left and 266 right), validation used 13–16 (180 trials, 90 left and 90 right), and the test used 17–20 (180 trials, 91 left and 89 right). On those unseen subjects the test scores were:

| Model | Accuracy | Balanced accuracy | Macro-F1 |
| --- | --- | --- | --- |
| Majority class | 0.506 | 0.500 | 0.336 |
| CSP + LDA | 0.522 | 0.524 | 0.509 |
| CNN | 0.528 | 0.532 | 0.429 |

The CNN predicted right on almost every unseen-subject trial (right recall 0.955, left recall 0.110). Per-subject CNN accuracy on the four test subjects ranged from 0.489 to 0.556. Both models stayed near chance once the people changed.

## Tests

```bash
pytest
```

The unit tests use synthetic EEG and do not download the dataset. They check the T1/T2 mapping, rest exclusion, the 320-sample epoch, training-only normalization, disjoint splits, the CNN output shape, probabilities, checkpoint reload, and that prediction functions have no label argument.

## Limitations

- The default test is a new recording of people who were already in training. Accuracy on a new person is a different experiment.
- Trials from one person are not independent. The reported accuracy is a description of these trials, not a population claim, and no p-value is attached.
- Twenty subjects and a few dozen imagery cues per recording are a small sample. Session-to-session EEG is noisy; a score near chance is a valid outcome.
- There is no blink, muscle, or bad-channel rejection beyond dropping a non-finite or completely flat channel.
- The bandpass and the 1–3 s window are fixed literature-style choices, not the result of a test-set search. They throw away information outside 8–30 Hz.
- The CNN is small on purpose. It is not an EEGNet, and it was not compared against a wider architecture search.
- The filter is zero-phase and the decision uses the whole 2 s epoch. This is not a causal, streaming BCI.
- The cursor demo replays stored trials. It will not move if a person imagines a movement in front of the screen.

## Layout

```text
app.py                      Streamlit dashboard
configs/default.yaml        Subjects 1–20, runs 4 / 8 / 12
src/mindmove/data.py        Download, cache, and splits
src/mindmove/preprocessing.py
src/mindmove/baseline.py    CSP + LDA
src/mindmove/train.py       CNN
src/mindmove/evaluate.py    Test metrics and figures
src/mindmove/inference.py   Prediction without labels
notebooks/exploration.ipynb Cached-epoch plots
artifacts/                  Models, metrics, figures
```

# Pre-registered protocol for the final experiments (written BEFORE any final run; 2026-09-20)

## Why this exists
Two findings made the earlier results unreliable as evidence about the CNN-LSTM, so the experiments are being redone under a fixed protocol:
1. **Rainfall timing** (`completion/a_rain_lag.py`, all four regions): rainfall file `j` is the rain that FALLS during interval `[1800 j, 1800 (j+1)) s`
   of the simulation (domain-mean depth gain = 0.494-0.502 x the file value; correlation 1.00 at lag 0 in every region), i.e. the files are mm/h
   intensities. The original loader gave the frame at time `t` the file `floor(t/1800)` = the interval that CONTAINS `t`, so the last observed frame
   carried up to 30 minutes of rain that had not yet fallen. All earlier results used this "hybrid" input.
2. **Baselines** (`completion/b_baselines.py`): a per-pixel LINEAR model that sees only depth history scores skill +0.15 (480 m) to +0.35 (30 m) against
   persistence, and +0.22 .. +0.40 with rain and state. The original CNN-LSTM (`model.py`, v1) scored +0.05 .. +0.23. It could not use the depth history
   (flatten + one 128-number dense bottleneck; depth scaled by 5 m).

## What was decided from what data (design phase)
Design used ONLY held-out TILES of the training region (Australia 60 m, fold 0 of 3, hybrid regime, 200 training windows, 12 epochs). The never-seen test
regions (UK, Mozambique) were not used for any design decision (`--blind` runs did not touch them; the four heavy runs that did evaluate them were not
consulted). Held-out skill on the same 300 windows: linear S1 +0.328, S3 +0.397 (400 windows); v2 (ConvLSTM at 16x16 + temporal skips) +0.568; v2 +
augmentation +0.653; v2lin (v2 + local CNN path) +0.706; v2lin + augmentation +0.705; **local-only CNN (4,101 parameters, no recurrence) +0.704;
clstm (full-resolution ConvLSTM, 4,741 parameters) +0.711**. The top four are within noise, so the simplest recurrent model is the project's CNN-LSTM:
**`clstm`**; `local` is the no-recurrence control; `v1` is the original model; `v2lin` is the heavy comparator (run on fewer folds if time allows).

## Frozen protocol
* Architectures: `v1` (original), `local`, `clstm` (main), `v2lin` (heavy, limited). Zero-initialised persistence-anchored output for all.
* Recipe: Adam, peak lr 1e-3, 200-step warm-up then cosine decay, gradient-norm clip 1.0, loss = MSE / persistence-MSE of the training windows, batch 1,
  **300 training windows, 30 epochs**, no early stopping or best-epoch selection (the final epoch is scored), seeds `1000 + 10*fold + 3`.
* Tiles: K = 3 spatial folds of the training region's valid tiles (fold assignment fixed by seed 0); every tile is held out exactly once; up to 300
  held-out windows; every window of the never-seen region. A stricter `--blocked` variant also holds out a time block.
* Regimes: `causal` (rain that has already fallen), `hybrid` (the original input), `forced` (hybrid + the rainfall of the 5 forecast intervals).
* Experiments: E480 Pakistan -> Mozambique, E60 Australia -> UK, E30 Australia -> UK (all depth only, 30-min steps, 5 -> 5 frames, 128-px patches).
* Metrics: skill = 1 - sum SSE_model / sum SSE_persistence; per lead; RMSE; CSI at 1 mm and 0.1 m. Intervals: cluster bootstrap (tiles / start times).
* Test-time ablations on every run; training-time ablations (variants) with `local` (all folds) and `clstm` (hybrid only).

## Hypotheses and how each will be judged (stated before the results)
* H1 clstm and local beat persistence on the held-out tiles AND on the never-seen regions in every regime (95% intervals above 0).
* H2 they beat the best linear baseline (S3) on the never-seen regions.
* H3 depth history matters: skill without depth history (`nodepthhist`, trained without it) is lower than with it by more than the fold-to-fold sd.
* H4 recurrence matters: clstm - local difference. Judged EQUIVALENT if |difference| < 0.02 with an interval inside +-0.03; otherwise the larger wins.
* H5 the ordering causal <= hybrid <= forced holds.
* H6 the strict space+time hold-out (`--blocked`) gives skill within 0.05 of the tile-only hold-out (otherwise held-out-tile numbers are optimistic
  because held-out tiles share time steps with the training tiles).
No architecture or recipe change will be made after the never-seen results are seen. Any deviation will be listed with its reason. Runs that fail
(dead network guard, crash) are reported, not silently retried with different settings.

## What is confirmatory and what is exploratory
Confirmatory: everything in this file, evaluated on the never-seen regions (they were not used to design v2/clstm/local). Exploratory: the reverse
transfer directions (data-starved training regions), the severity/rule/Grad-CAM re-evaluations, anything not listed above.

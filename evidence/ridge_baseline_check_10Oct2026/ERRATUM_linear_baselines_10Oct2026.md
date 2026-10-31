# Erratum: linear baselines of the laptop study (10 October 2026)

**Found by Aman Sumeet Saxena, checked and re-run by Anhad on 10 October 2026.**

## What was wrong
In `completion/b_baselines.py`, the function `fit()` fits the linear baselines S1, S2 and S3 as a ridge regression with the penalty
`lam * N * I` on the raw, unscaled features. The depth-change features (last change, change over the window) are tiny numbers (millimetres to
centimetres, in metres), so the penalty pushed their weights almost to zero. The linear baselines therefore could not use the depth history,
and they looked far weaker than they are. The MLP (S4) standardises its inputs, the lookup table (S5) has no penalty, and the networks are
not affected.

## What was done
The linear baselines were re-fitted with standardised features, on the same folds, windows and scoring. The shipped code was first re-run
unchanged and reproduced the old numbers exactly, in all 9 combinations of experiment and rain setting. A near-unpenalised fit gives the same
result as the standardised ridge to within 0.001. Details and scripts: `ridge_baseline_check_10Oct2026/` (README_ridge_check.txt,
ridge_check.py, h2_recheck.py, h2_recheck_out.json). No file of the original project was changed.

## Corrected numbers (skill on the never-seen flood, hybrid rain, mean of 3 folds)

| Model | E60 | E30 | E480 |
|---|---|---|---|
| Linear, depth history (S1): old | 0.323 | 0.345 | 0.151 |
| Linear, depth history (S1): corrected | **0.818** | **0.798** | **0.790** |
| Linear, rain + state + trend (S3): old | 0.345 | 0.359 | 0.182 |
| Linear, rain + state + trend (S3): corrected | **0.820** | **0.797** | **0.793** |

For comparison: lookup table 0.786 / 0.763 / 0.737, per-pixel MLP 0.846 / 0.854 / 0.829, local CNN 0.841 / 0.830 / 0.807, small ConvLSTM 0.866 / 0.840 / 0.811.
S2 (rain + state, no trend features) was not affected (0.207 / 0.188 / 0.100).

## Effect on the conclusions
- A linear model of the recent depth history already reaches about 0.79 to 0.82. The local CNN and the small ConvLSTM add only about 0.01 to 0.05 on top of it.
  The statement "the forecast is a nonlinear function of the depth history" is not supported. "Most of it is linear" is.
- **H2** (the networks beat the best linear baseline S3c on the never-seen flood): 15 of 16 cells pass the planned rule (mean skill above the baseline), not 16 of 16.
  The failing cell is the ConvLSTM with forced rain on E480 (0.765 against 0.811). In 8 cells the gain is 0.02 or more, in 7 cells it is below 0.02 (within seed noise), in 1 cell it is negative.
- Not affected: the Colab results (Model A, Model B, trend forecast, water-balance rule, capsule network, Grad-CAM), H1, H3 to H6, the lookup table, the MLP, the networks,
  the ablations, the optimisation experiments and the failure analysis.

## Documents that still show the old linear numbers or the old H2 result
Found by searching for the old numbers and for the words "linear baseline" (not every file was read): `REPORT.md`, `PROJECT_LOG.md`, `HANDOFF_README.md`,
`PREREGISTRATION.md`, `STATE.md`, `Project_Explainer_End_to_End_v2.md`, the review files in `3_Reviews_and_Audits`, and possibly the slide deck v2 (slides on the baseline ladder were not searched).
Read them together with this note. The CA3 report (`CA3_Project_Report_FINAL_corrected`) already contains the corrected numbers.

## Code
`completion/b_baselines.py` itself has not been changed. To fix it, standardise the columns of `X` in `fit()` (except the column of ones) before the ridge solve and map
the coefficients back, as done in `ridge_check.py` (function `make_fit`).

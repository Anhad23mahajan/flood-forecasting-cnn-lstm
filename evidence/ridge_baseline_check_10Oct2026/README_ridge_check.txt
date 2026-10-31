RIDGE BASELINE CHECK, 10 October 2026
Question (raised by Aman): in completion/b_baselines.py the ridge penalty is applied to unscaled features, so the small depth-change
features are shrunk to almost zero. Is that true, and how much does it change the report?

ANSWER: true, and the effect is large.

What was done
- ridge_check.py and h2_recheck.py re-run ONLY the linear baselines S1 and S3 of b_baselines.py on the same folds, the same windows and the same
  scoring (engine.split_for, engine.evaluate), reading the existing .npy caches. No project file was changed.
- Step 1 (validation): the unchanged code ("orig") reproduces the shipped numbers exactly, in all 9 experiment x rain-setting combinations
  (for example E480 hybrid S1 0.151, S3 0.182, which are the Table 7 numbers).
- Step 2: the same ridge (lam = 1e-3) on STANDARDISED features ("std"). A near-unpenalised fit ("ols", lam = 1e-9) gives the same result to
  within 0.001, so the penalty size does not matter once the features are scaled.
- Two stand-ins were needed on this machine: scipy.ndimage.uniform_filter (re-implemented in numpy, edge padding) and rasterio (not needed
  because the data come from the cache).

Table 7 rows, hybrid rain, mean of 3 folds, skill on the never-seen flood
                                       E60     E30     E480
  Linear, depth history (S1) shipped   0.323   0.345   0.151
  Linear, depth history (S1) standardised 0.818 0.798   0.790
  Linear, rain+state+trend (S3) shipped 0.345  0.359   0.182
  Linear, rain+state+trend (S3) standardised 0.820 0.797 0.793
  (for comparison: lookup table 0.786 0.763 0.737, per-pixel MLP 0.846 0.854 0.829, local CNN 0.841 0.830 0.807, small ConvLSTM 0.866 0.840 0.811)

Hypothesis H2 (networks beat the best linear baseline S3c on the never-seen flood), planned rule: mean skill above the baseline
  exp   rain     S3c shipped  S3c standardised | local CNN  small ConvLSTM | local - lin   ConvLSTM - lin
  E60   causal      0.334         0.819         |  0.822      0.853         |  +0.003       +0.034
  E60   hybrid      0.345         0.820         |  0.841      0.866         |  +0.021       +0.045
  E60   forced      0.365         0.837         |  0.866      0.854         |  +0.029       +0.017
  E30   causal      0.347         0.797         |  0.805      (no run)      |  +0.008
  E30   hybrid      0.359         0.797         |  0.830      0.840         |  +0.033       +0.042
  E30   forced      0.379         0.815         |  0.851      (no run)      |  +0.037
  E480  causal      0.170         0.794         |  0.794      0.810         |  +0.000       +0.016
  E480  hybrid      0.182         0.793         |  0.807      0.811         |  +0.014       +0.017
  E480  forced      0.202         0.811         |  0.834      0.765         |  +0.023       -0.046
  Shipped baseline: 16 of 16 cells pass. Standardised baseline: 15 of 16 pass (E480 forced ConvLSTM fails).
  With the report's own rule that differences below 0.02 cannot be resolved: 8 cells are real wins, 7 are ties, 1 is a loss.

Files: ridge_check.py, h2_recheck.py (run with the system python from a folder that can import the project's completion/ folder),
h2_recheck_out.json (all numbers).

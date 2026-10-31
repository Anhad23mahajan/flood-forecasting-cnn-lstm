# Independent verification of headline flood-forecasting numbers

Date of run: 2026-09-20. Everything below was recomputed from the raw GeoTIFFs in
`data\extracted\FloodCastBench\` with my own code (rasterio + numpy + scipy only; the scratch scripts are in
`completion\review\scratch\`). I did NOT import `engine.py`, `real_data_loader.py`, `windows.py`, `model*.py`, `b_baselines.py`
(I did READ `engine.py` and `b_baselines.py` afterwards, only to learn definitions for claim E and to interpret one finding).
Single-threaded (`OMP_NUM_THREADS=1`), no training, no existing file edited, moved or deleted; the only files written are this report
and my own scratch scripts / small result files. (One 144 MB scratch array of Mozambique frames was created by me and removed again.)

## Overall verdict

| Claim | Verdict |
|---|---|
| A. Rain timing (Australia 30 m, Mozambique 480 m): lag-0 peak, r ~ 1.00, slope ~ 0.5 | MATCH (all 13 lags, slope, intercept, totals reproduced to <= 4e-7; one undocumented subset definition, see A.3) |
| B. Persistence RMSE and n_windows on test windows (Mozambique 480 m, UK 60 m) | MATCH (n identical, same window set, RMSE differs by 3e-8 mm and 2e-9 mm) |
| C. Pool integrity (Pakistan 480 train, Australia 60 train, UK 60 test) | MATCH (bit-exact equality) |
| D. Paper-protocol persistence numbers, Mozambique 480 m | MATCH (identical to ~1e-15, also for the "final 20%" and "final 2 days" blocks) |
| E. Two-feature lookup table inputs use history only | CONFIRMED (file-index arithmetic + equality with pool arrays + engine code path) |

No mismatch beyond rounding was found in any of the requested numbers. Points worth knowing about (none of them is a bug in the numbers)
are listed in "Suspicious / noteworthy" at the end.

---

## A. Rainfall timing (files `rain_lag_Australia_30.json`, `rain_lag_Mozambique_480.json`)

### Method
1. Depth frames: files are named by integer seconds. For each region I checked the names are exactly 0, 300, 600, ..., (n-1)*300 (contiguous, no gaps, no non-integer names).
   Domain-mean depth D[j] (mm) = mean over ALL pixels of the frame at t = 1800*j, for j = 0..N (N = 480 intervals for Australia, 288 for Mozambique). No NaN and no negative value in any of the 481 / 289 frames read.
2. Rainfall: files sorted by name (`YYYYMMDD-SHHMMSS.tif`); checked that consecutive timestamps are exactly 1800 s apart (Australia 481 files, Mozambique 289 files).
   Read only the depth-covered window of the 30 m base grid (Australia offset (0,0), 1073x1073; Mozambique offset (150,50), window 151*16 x 138*16 = 2416 x 2208), took the 16x16 (Moz) block mean, then the mean over the grid = R[j] (mm/h).
   No NoData / NaN inside the window (0 pixels). (Block-mean-then-mean equals the plain mean of the window exactly, difference 0.0, because all blocks have equal size.)
3. dV[j] = D[j+1] - D[j] (mm of depth gained over interval j). Lag k pairs dV[j] with R[j+k] (k > 0: rain from later files). Pearson r over all valid pairs; least squares dV = slope*R[j] + intercept at k = 0.

### A.1 Correlation dV vs rain at each lag (claimed vs recomputed)

Australia 30 m (recomputed on the 394 "active" intervals, dV > 0.001 mm; see A.3):

| lag | claimed | recomputed | abs diff | (recomputed on all 480 intervals, for reference) |
|---|---|---|---|---|
| -6 | 0.716820 | 0.716820 | 3.7e-07 | 0.734797 |
| -5 | 0.740233 | 0.740232 | 2.9e-07 | 0.757180 |
| -4 | 0.763054 | 0.763054 | 1.8e-07 | 0.778888 |
| -3 | 0.815178 | 0.815178 | 4.6e-08 | 0.827820 |
| -2 | 0.872499 | 0.872499 | 1.7e-07 | 0.881360 |
| **-1** | 0.919288 | 0.919287 | 2.7e-07 | 0.924981 |
| **0** | **0.999771** | **0.999771** | 2.8e-10 | 0.999787 |
| **+1** | 0.921062 | 0.921062 | 1.2e-07 | 0.926684 |
| +2 | 0.875113 | 0.875113 | 1.1e-07 | 0.884021 |
| +3 | 0.818579 | 0.818579 | 1.1e-07 | 0.831552 |
| +4 | 0.765972 | 0.765972 | 7.2e-08 | 0.782662 |
| +5 | 0.742582 | 0.742582 | 3.4e-07 | 0.760443 |
| +6 | 0.720181 | 0.720180 | 3.5e-07 | 0.738259 |

Mozambique 480 m (all 288 intervals are active; the two columns coincide):

| lag | claimed | recomputed | abs diff |
|---|---|---|---|
| -6 | 0.687549 | 0.687549 | 1.1e-07 |
| -5 | 0.708202 | 0.708202 | 1.6e-07 |
| -4 | 0.726677 | 0.726677 | 2.2e-07 |
| -3 | 0.794050 | 0.794050 | 1.4e-07 |
| -2 | 0.864294 | 0.864294 | 5.0e-08 |
| **-1** | 0.928524 | 0.928523 | 3.0e-07 |
| **0** | **0.999574** | **0.999574** | 3.8e-11 |
| **+1** | 0.928377 | 0.928377 | 1.8e-07 |
| +2 | 0.862537 | 0.862536 | 3.5e-07 |
| +3 | 0.791065 | 0.791065 | 2.0e-09 |
| +4 | 0.722581 | 0.722580 | 1.9e-07 |
| +5 | 0.702258 | 0.702257 | 1.6e-07 |
| +6 | 0.681014 | 0.681014 | 1.5e-08 |

Lag 0 is the unique peak in both regions; the lag +/-1 neighbours are 0.92 to 0.93 (the residual correlation at neighbouring lags simply reflects the autocorrelation of the rainfall series, not a timing ambiguity).
The sign convention also matches (the claimed +1 and -1 values are reproduced separately). Verdict: **MATCH**.

### A.2 Regression at lag 0 and totals

| quantity | region | claimed | recomputed | abs diff | verdict |
|---|---|---|---|---|---|
| slope (mm depth per mm/h) | Australia | 0.4970370 | 0.4970371 | 9e-08 | MATCH |
| intercept (mm) | Australia | 0.0046266 | 0.0046264 | 2e-07 | MATCH |
| total depth gain (mm) | Australia | 362.68740 | 362.68739 | 3e-06 | MATCH |
| sum of rain intensity, files 0..479 | Australia | 726.02701 | 726.02701 | 2e-09 | MATCH |
| sum of rain intensity, all 481 files | Australia | 726.46418 | 726.46418 | 2e-09 | MATCH |
| n_active / n_intervals | Australia | 394 / 480 | 394 / 480 | 0 | MATCH |
| slope | Mozambique | 0.5006620 | 0.5006620 | 2e-08 | MATCH |
| intercept (mm) | Mozambique | 0.0318423 | 0.0318424 | 9e-08 | MATCH |
| total depth gain (mm) | Mozambique | 217.23602 | 217.23606 | 4e-05 | MATCH |
| sum of rain intensity, files 0..287 | Mozambique | 415.58070 | 415.58070 | 6e-08 | MATCH |
| sum of rain intensity, all 289 files | Mozambique | 415.59869 | 415.59869 | 6e-08 | MATCH |
| best lag | both | 0 | 0 | - | MATCH |

I also compared the project's cached rainfall stacks with my raw-TIFF domain means: Mozambique all 289 files, max abs difference 6.5e-9 mm/h; Australia 25 files (every 20th), difference 0.0. So the cache is faithful to the raw rain rasters.

### A.3 One thing you would not guess from the JSON
For Australia the correlations and the slope/intercept are computed on the 394 "active" intervals only, and "active" turns out to be `dV > 0.001 mm` (a mask defined on the dependent variable; 314 intervals have any rain, so it is not "rain > 0"; the first 38 intervals have no rain and no gain).
I found this by trying masks: `dV > 0.001` reproduces the claimed numbers to 4e-7, no other mask I tried did. Using all 480 intervals instead gives lag-0 r = 0.999787 (slightly higher), slope 0.497215 (vs 0.497037) and intercept 0.003533 (vs 0.004627); off-lag correlations are also slightly higher. So the masking does not inflate the headline; but the JSON key `n_intervals: 480` next to `n_active: 394` does not say which one the correlations use, and the mask is not documented in the file. The totals (`k0_total_gain_mm`, `k0_total_rain_mm`) are over all 480 intervals, not over the active ones.

### A.4 Units caveat (labelling, not a numerical error)
`rain_total_mm` and `k0_total_rain_mm` are sums of mm/h intensities over the 30-min files, not millimetres of rain. Millimetres of rain = 0.5 h x that sum:
Australia 363.01 mm of rain vs 362.69 mm domain-mean depth gain (99.91%); Mozambique 207.79 mm of rain vs 217.24 mm gain (104.5%). The Mozambique excess (~9.4 mm) is the intercept (0.0318 mm/interval x 288 = 9.2 mm) plus the 0.0007 excess slope: a constant non-rain water source of about 0.03 mm per 30 min (boundary inflow or similar) exists in the Mozambique simulation.
This does not change the slope ~ 0.5 claim (0.5007), but "depth gain = 0.5 x rain" is exact only for the slope, not for the total, in Mozambique.

---

## B. Persistence baseline on the never-seen test windows

### Method
Depth-file index i has time 300*i (names verified contiguous). L = n_files - 1 - 6*(5+5-1) = 1674 (Mozambique, 1729 files) and 810 (UK 60 m, 865 files).
Starts s = unique(int(linspace(0, L, 60))) (60 unique starts in both cases, e.g. Moz 0, 28, 56, 85, 113, 141, ...). For each start and each tile: last history frame = file s+24 (pred), targets = files s+30, s+36, s+42, s+48, s+54.
Keep the window iff max(last-history patch, last-forecast patch) >= 0.05 m. Persistence RMSE (mm) = 1000 * sqrt( sum over kept windows, 5 leads, all real pixels of (target - pred)^2 / number of such pixels ).
Tiles: Mozambique 128x128 at rows {0,23}, cols {0,10}; UK 60 m 85x85 at row 0, cols {0,52} (real pixels only, 7225 per frame).

| quantity | claimed | recomputed | abs diff | verdict |
|---|---|---|---|---|
| Mozambique 480 m, n_windows | 238 | 238 | 0 | MATCH |
| Mozambique 480 m, persistence RMSE (mm) | 17.075699532 | 17.075699506 | 2.6e-08 | MATCH |
| UK 60 m, n_windows | 120 | 120 | 0 | MATCH |
| UK 60 m, persistence RMSE (mm) | 15.700446932 | 15.700446930 | 2.4e-09 | MATCH |

Additional agreement beyond the two headline numbers:
* My set of kept (tile_r, tile_c, start) triples is identical to `test_windows` in both run JSONs (0 windows only on either side). Moz: 240 candidates, 2 dropped as dry (both at start 0: tiles (0,0) and (0,10)); UK: 0 dropped.
* Per-window, per-lead persistence squared errors: my values vs stored `sse_p` agree to max abs 1.1e-5 m^2 (Moz) and 1.8e-6 m^2 (UK), max relative 2.1e-7, i.e. float32 accumulation noise. Stored `npx` = 16384 (Moz) and 7225 (UK) are correct.
* The JSON's own `sse_p`/`npx` re-aggregate to the same RMSE (17.075700 / 15.700447), so the JSON is internally consistent.
* Per-lead persistence RMSE (mm, mine): Moz 5.23, 10.41, 15.56, 20.62, 25.59; UK 4.82, 9.60, 14.32, 18.95, 23.52.
* No NaN in any frame used. Including the 2 dry Moz windows would change the RMSE to 17.005 mm (dry windows alone: 0.76 mm), so the wet filter is immaterial for this number.

---

## C. Pool integrity (`completion\cache\pool_*`)

Method: `meta` rows are (tile row, tile col, start); pool tile size from the `size` field (128, 128, 85). For each window I read raw TIFF frames start+6k (k = 0..4) and start+6*(5+k), cropped [r:r+size, c:c+size], and compared with `dh[i,k]`, `df[i,k]` using `np.array_equal` (exact float32 equality). I also checked, where possible, that the stored patch differs from the neighbouring frames (start+6k+-1), so the test is able to detect an off-by-one.

| pool | windows in pool | windows checked | patches compared | exactly equal | abs diff | verdict |
|---|---|---|---|---|---|---|
| Pakistan_480_train | 1440 | 10 random (seed 20260920), then 100 random (seed 777) | 100, then 1000 | all | 0 | MATCH |
| Australia_60_train | 1271 | 10 random, then 100 random | 100, then 1000 | all | 0 | MATCH |
| UK_60_test | 120 | 10 random, then ALL 120 | 100, then 1200 | all | 0 | MATCH |

Also for every meta row: tile fits inside the grid, and max(start)+54 <= n_files-1 (Pakistan 3978+54 = 4032 = last file; Australia 60 m 2826+54 = 2880 = last file; UK 810+54 = 864 = last file). No NaN in stored patches. (The stdout label "10 windows" in the second run is stale text from a copied script; the second run really used 100 windows per big pool and all 120 UK windows.)

---

## D. Paper-protocol persistence, Mozambique 480 m (`paper_protocol_persistence_Mozambique_480.json`)

### Method
All 1729 frames (151x138) loaded. Samples k = 0..85 (frames 20k..20k+19; 86 samples = the file's `n_samples`; the last 9 frames 1720..1728 are unused). In sample k the prediction is frame 20k (the "frame 1") for each of the 19 later frames (20k+1..20k+19), whole domain. All metrics are POOLED over samples x 19 steps x pixels (34,049,292 values), float64:
rel_sq_err = sum (y-p)^2 / sum y^2; rel_l2_err = sqrt of that; NSE = 1 - sum (y-p)^2 / sum (y - mean y)^2 (pooled mean); r = pooled Pearson correlation of y and p; CSI = hits/(hits+misses+false alarms) with hit = (y >= t) and (p >= t), t = 0.001 m and 0.01 m ("> t" gives identical values).

| metric | claimed | recomputed | abs diff | verdict |
|---|---|---|---|---|
| whole event: rel_sq_err | 0.00044170311684 | 0.00044170311684 | 7e-19 | MATCH |
| whole event: rel_l2_err | 0.02101673420969 | 0.02101673420969 | 2e-17 | MATCH |
| whole event: NSE | 0.99951292724483 | 0.99951292724483 | 0 | MATCH |
| whole event: r | 0.99977277595964 | 0.99977277595964 | 1e-15 | MATCH |
| whole event: CSI 1 mm | 0.98315725186 | 0.98315725186 | 0 | MATCH |
| whole event: CSI 1 cm | 0.97300834741 | 0.97300834741 | 0 | MATCH |
| n_samples | 86 | 86 | 0 | MATCH |

Bonus (the file's other blocks; the sub-window definitions are not stated in the JSON, I found them by trying variants): "final 20% of the event" = the last floor(0.2*86) = 17 samples, "final 2 days" = the last 28 samples (samples whose first frame is within the last 576 frames). With those, rel_sq_err, rel_l2, NSE, r, CSI 1 mm, CSI 1 cm all agree with the file to <= 2e-15 (e.g. final 20%: NSE 0.99996753, CSI 1 cm 0.99642077; final 2 days: NSE 0.99995780, CSI 1 cm 0.99387443). Verdict: **MATCH**.

Interpretation note: the pooled NSE/r are dominated by the static spatial pattern of depth (the denominator is variance around a single pooled mean over space and time), so ~0.9995 says "the spatial map barely changes over 5 to 95 min", it does not say persistence has forecasting skill in a temporal-anomaly sense.

---

## E. Two-feature lookup table inputs (history-only check)

Picked one Mozambique test window deterministically (window index 106 of `pool_Mozambique_480_test`: tile (r,c) = (0,0), start s = 766, phase s mod 6 = 4).
* History files: 766, 772, 778, 784, 790 (t = 229800 ... 237000 s); forecast files: 796, 802, 808, 814, 820.
* d_T = file 790 (last observed), d_{T-1} = file 784; gap 1800 s = 30 min = the "one-step" of the table's `T_EDGES` (m per 30 min). change = d_T - d_{T-1} uses only files <= 790, strictly before the first forecast file 796.
* Pool `dh[106]` and `df[106]` equal the raw TIFF crops exactly, and `dh[106,4] - dh[106,3]` equals my raw-TIFF change bit-for-bit.
* The "leaky" alternative (first forecast frame minus d_T) is NOT identical to the feature (`np.allclose` False). corr(feature, lead-1 target change) over wet pixels = 0.9954 (a leak would be exactly 1.0); that high value is genuine physical persistence of the trend, not leakage.
* Reading `completion\engine.py` `compose`/`build_pool` and `completion\b_baselines.py`: `x[..., 0]` rows 0..T_IN-1 (T_IN = 5) are `dh` (history); `features`/`table_predictor`/`fit_table` use only `x[T_IN-1, ..., 0]` and `x[T_IN-2, ..., 0]`; the training target is `df - x[T_IN-1]`. In the "forced" regime the depth channel is extended with copies of the last observation (not real future depth) and the table still only indexes rows T_IN-1 and T_IN-2. So the change is computed from history only. Verdict: **CONFIRMED**.
* Caveat on what I did: my "perturb future frames and features are unchanged" test is tautological (the feature is defined from H only); the real evidence is the index arithmetic, the equality with the stored arrays, and the code path above. I did not re-fit or evaluate the table (not needed and would need project code).

---

## Suspicious / noteworthy (none invalidates a checked number)

1. **Undocumented mask in the Australia rain-lag numbers** (A.3): correlations/slope use 394 of 480 intervals selected by `dV > 0.001 mm`. The choice is harmless (full-set numbers are the same or higher) but it selects on the dependent variable and is not written into the JSON.
2. **Mislabelled units** (A.4): `rain_total_mm`, `k0_total_rain_mm` are sums of mm/h intensities; and in Mozambique the domain-mean depth gain exceeds accumulated rain by ~4.5% (constant 0.032 mm per interval source).
3. **Test-window independence.** The 238 Mozambique windows are 60 start times x 4 corner tiles; the tiles overlap by ~76% in area (e.g. (0,0) vs (23,10) share 105x118 of 128x128 pixels), and consecutive starts (28 files apart) are closer than the 54-file window span, so windows overlap in time too. The 120 UK windows are 60 starts x 2 tiles overlapping in 33 of 85 columns. The effective number of independent samples is far below 238 / 120 (order of the number of start times, and those are serially correlated); any confidence interval or significance statement should be clustered by start time. The per-window skill values themselves are correct, as verified.
4. **Wet filter looks at the target.** Window retention depends on the last FORECAST frame (max(...) >= 0.05 m). By the stated definition this is correct, and for the tested sets it removes only 2 of 240 Mozambique windows (persistence RMSE changes 17.076 to 17.005 mm) and 0 of 120 UK windows, so it is immaterial here; but the same rule applied to a dataset with many dry windows would condition on the outcome.
5. **"hybrid" rainfall regime sees (partly) future rain.** Claim A shows rain file j drives the depth change over [1800 j, 1800 (j+1)]. In `engine.rain_intervals`, the hybrid regime feeds the last history step rain file `start//6 + 4`, i.e. the file whose interval contains the last observation time; for phase 0 that whole 30 min interval is still in the future of the last frame, for phase 5 only 5 min of it is. The 'causal' regime shifts by -1 file. This is by design (the regimes are named causal/hybrid/forced) and consistent with A, but "hybrid" results should not be described as purely causal forecasts. I did not test the models themselves.
6. **Spatial crop offsets could not be independently pinned down with the method I used.** Claims A to E only need domain means or depth-grid coordinates, so they are unaffected. I did try to check the Mozambique offset (150,50) spatially: (i) correlation of block-mean DEM vs log depth over an offset grid is essentially flat (peak within 2e-4 of the claimed offset, best about (dy,dx) = (-2, -4) base pixels); (ii) correlation of per-cell depth gain over the first 4 intervals vs block-mean rain is a broad hump with a maximum at about (+2, +16) (0.8927 vs 0.8807 at the claimed offset), i.e. one 480 m cell east. The two indicators disagree in direction, both surfaces are very flat because the rain and DEM at this scale are smooth and water is routed between cells, so I regard this as inconclusive, not as evidence of a misalignment. The dedicated georeferencing check (`e_georef_coarse.py`, `results\georef_*.png`) was not audited by me.
7. **Rain vs depth file counts.** Depth intervals vs rain files: Australia 480 vs 481, UK 144 vs 145, Mozambique 288 vs 289 (one closing file), but Pakistan 672 vs 672 (no closing file). `engine.compose` clamps the rain index to the last file (`min(j, nf-1)`), which only affects the last few "forced"-regime windows for Pakistan. Minor.
8. **File naming / data hygiene.** All six depth sets have contiguous integer-second names 0..(n-1)*300 with no stray files; no NaN or negative depth in any frame I read; rain rasters have no NoData inside the depth-covered window for Australia and Mozambique (NoData -3.4e38 exists only outside the crop, as stated). Depth rasters carry no nodata tag and no geotransform (rasterio warns "NotGeoreferencedWarning"), so all geography comes from the project's offsets, not from the files. The rain files are named by wall-clock timestamp and the depth files by seconds since start; the alignment between them (rain file 0 <-> t = 0) is not stated in the files, it is confirmed only empirically by the lag-0 peak in A.
9. **Australia 30 m depth has a thin water film everywhere** (e.g. min depth 3e-5 m at t = 100200 s over all 1073x1073 pixels), whereas Mozambique has exact zeros. Any threshold-based wet mask (0.01 m, 0.05 m) therefore matters more for Australia than an "is it zero" test. Not a problem, just a property of the data.

## Not verified
* `rain_lag_UK_30.json`, `rain_lag_Pakistan_480.json`, `corr_dWet` (wet-pixel-count series), UK / Pakistan paper-protocol files, the model skill numbers, CSI entries inside the run JSONs, the E60 / E480 model RMSEs, the baseline ladder results, and the georeferencing of the DEM/rain to depth grids beyond the weak checks in item 6.
* I did not check the E30 or R* experiments.

## Scratch files (all in `completion\review\scratch\`)
`vcommon.py` (my helpers), `verify_A.py` + `analyse_A.py` (claim A; `A_*.npz` hold the domain-mean series), `verify_B.py` (`B_result.json`), `verify_C.py` / `verify_C2.py`, `verify_D.py` (`D_result.json`), `verify_E.py`, `verify_extra.py` (naming, cache-vs-raw rain, dry-window RMSE), `verify_offset.py`, `verify_offset2.py`.
Run times were short (Australia rain-lag extraction ~2 min, Mozambique ~1.5 min, everything else seconds); all single-threaded.

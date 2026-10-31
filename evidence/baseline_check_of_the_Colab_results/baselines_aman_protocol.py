"""Read-only check of Aman's notebook protocol: whole-map, one step (30 min) ahead, the same splits as his notebook.
Computes the skill (1 - SSE/SSE_copy, all cells) of simple baselines so his CNN-LSTM numbers can be judged against them:
  copy        : depth[t-1]
  trend1      : max(0, 2 d[t-1] - d[t-2])              (linear extrapolation, no parameters)
  trend_beta  : max(0, d[t-1] + beta (d[t-1]-d[t-2]))  (one number, fitted on the TRAINING part of the training flood)
  rain_unif   : d[t-1] + 0.5*mean(rain[t-2])/1000      (= his water-balance rule applied to 'copy')
  trendb+rule : his apply_water_balance applied to trend_beta
Also: draining check, CSI zero-denominator check, wet-cell dilution of the 'typical error'."""
import glob
import json
import os
import re
import sys
import time

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def read_tif(path):
    return np.array(Image.open(path)).astype("float32")


ROOT = r"C:\Dev\flood-forecast-demo\data\extracted\FloodCastBench"
OUT = r"baselines_result.json"
WET, FLOOD = 0.01, 0.30
INPUT_STEPS = 4
FIRST_TARGET = INPUT_STEPS + 1

DS = {
    "Australia_30m": dict(region="Australia", folder="High-fidelity flood forecasting/30m/Australia", scale=1, off=(0, 0)),
    "UK_30m": dict(region="UK", folder="High-fidelity flood forecasting/30m/UK", scale=1, off=(5, 5)),
    "Pakistan_480m": dict(region="Pakistan", folder="Low-fidelity flood forecasting/480m/Pakistan", scale=16, off=(42, 178)),
    "Mozambique_480m": dict(region="Mozambique", folder="Low-fidelity flood forecasting/480m/Mozambique", scale=16, off=(150, 50)),
}
PAIRS = [("30m", "Australia_30m", "UK_30m"), ("480m", "Pakistan_480m", "Mozambique_480m")]


def log(*a):
    print(*a, flush=True)


def depth_files(name):
    folder = os.path.join(ROOT, DS[name]["folder"])
    files = [f for f in glob.glob(os.path.join(folder, "*.tif")) if int(os.path.basename(f)[:-4]) % 1800 == 0]
    return sorted(files, key=lambda f: int(os.path.basename(f)[:-4]))


def rain_files(name):
    folder = os.path.join(ROOT, "Relevant data", "Rainfall", DS[name]["region"] + " flood")
    return sorted(glob.glob(os.path.join(folder, "*.tif")))


class Region:
    def __init__(self, name):
        self.name = name
        self.dfiles = depth_files(name)
        self.rfiles = rain_files(name)
        self.h, self.w = read_tif(self.dfiles[0]).shape
        self.n = len(self.dfiles)
        self.train_end = int(0.70 * self.n)
        self.val_end = int(0.85 * self.n)
        log(name, "maps", self.n, "grid", self.h, self.w, "rain files", len(self.rfiles))

    def depth(self, k):
        return np.nan_to_num(read_tif(self.dfiles[k]))

    def rain_mean(self, k):
        sc, (r0, c0) = DS[self.name]["scale"], DS[self.name]["off"]
        a = read_tif(self.rfiles[k])
        win = a[r0:r0 + self.h * sc, c0:c0 + self.w * sc]
        assert win.shape == (self.h * sc, self.w * sc), (win.shape, self.h * sc, self.w * sc)
        assert float(win.min()) > -1e30, "NoData inside the depth-aligned window"
        return float(win.mean(dtype="float64"))


def apply_water_balance(predicted, last, past_rain_mean):
    """Identical to Aman's apply_water_balance (past_rain given as its map mean, which is all the rule uses)."""
    predicted_change = predicted - last
    allowed_rise = 0.5 * float(past_rain_mean) / 1000
    extra_water = (float(np.mean(predicted_change)) - allowed_rise) * predicted.size
    corrected = predicted.copy()
    if extra_water < 0:
        corrected = corrected - extra_water / predicted.size
    else:
        wet = corrected > WET
        if wet.sum() > 0:
            corrected[wet] = corrected[wet] - extra_water / wet.sum()
    return np.maximum(corrected, 0)


def fit_beta(reg):
    """OLS slope of the next change on the previous change, over every cell of every training step."""
    sxx = sxr = 0.0
    prev2 = reg.depth(FIRST_TARGET - 2)
    prev = reg.depth(FIRST_TARGET - 1)
    mean_change = {}
    for t in range(FIRST_TARGET, reg.train_end):
        cur = reg.depth(t)
        x = (prev - prev2).astype("float64")
        r = (cur - prev).astype("float64")
        sxx += float((x * x).sum())
        sxr += float((x * r).sum())
        mean_change[t] = float(r.mean())
        prev2, prev = prev, cur
    return sxr / sxx, mean_change


def evaluate(reg, first, last_t, beta, tag):
    acc = {k: 0.0 for k in ("copy", "trend1", "trend_beta", "rain_unif", "trendb_rule")}
    cells = 0
    wet_cells = 0
    wet_sse_copy = 0.0
    csi_zero_steps = 0
    steps = 0
    mean_change = {}
    prev2 = reg.depth(first - 2)
    prev = reg.depth(first - 1)
    t0 = time.time()
    for t in range(first, last_t):
        true = reg.depth(t)
        past_rain = reg.rain_mean(t - 2)
        c = (prev - prev2)
        preds = {
            "copy": prev,
            "trend1": np.maximum(prev + c, 0),
            "trend_beta": np.maximum(prev + beta * c, 0),
        }
        preds["rain_unif"] = apply_water_balance(prev, prev, past_rain)
        preds["trendb_rule"] = apply_water_balance(preds["trend_beta"], prev, past_rain)
        for k in acc:
            acc[k] += float(((preds[k] - true) ** 2).sum(dtype="float64"))
        cells += true.size
        wetm = (prev > WET) | (true > WET)
        wet_cells += int(wetm.sum())
        wet_sse_copy += float(((prev - true)[wetm] ** 2).sum(dtype="float64"))
        if (true > FLOOD).sum() == 0 and (prev > FLOOD).sum() == 0:
            csi_zero_steps += 1
        mean_change[t] = float((true - prev).mean(dtype="float64"))
        steps += 1
        prev2, prev = prev, true
        if steps % 25 == 0:
            log("  ", tag, "step", steps, "of", last_t - first, "(%.0fs)" % (time.time() - t0))
    res = {
        "steps": steps,
        "skill": {k: 1 - acc[k] / acc["copy"] for k in acc if k != "copy"},
        "copy_rmse_cm_whole_map": 100 * np.sqrt(acc["copy"] / cells),
        "copy_rmse_cm_wet_cells": 100 * np.sqrt(wet_sse_copy / max(wet_cells, 1)),
        "wet_share_of_cells": wet_cells / cells,
        "csi_zero_denominator_steps": csi_zero_steps,
        "mean_change_per_step_mm": {int(k): 1000 * v for k, v in mean_change.items()},
    }
    return res


results = {}
for model_name, train_name, test_name in PAIRS:
    train = Region(train_name)
    test = Region(test_name)
    log("fitting beta on", train_name, "targets", FIRST_TARGET, "to", train.train_end - 1)
    beta, train_change = fit_beta(train)
    log("  beta =", round(beta, 4))
    train_steps = len(train_change)
    draining_train = sum(1 for v in train_change.values() if v < 0)
    results[model_name] = {"beta": beta, "train_steps": train_steps, "train_steps_with_falling_mean_depth": draining_train}

    res_train_late = evaluate(train, train.val_end, train.n, beta, train_name + " last15%")
    res_test = evaluate(test, FIRST_TARGET, test.n, beta, test_name + " whole")
    results[model_name][train_name] = res_train_late
    results[model_name][test_name] = res_test
    json.dump(results, open(OUT, "w"), indent=1, default=float)
    log("=== done", model_name)
    for nm, r in ((train_name, res_train_late), (test_name, res_test)):
        log(nm, {k: round(v, 3) for k, v in r["skill"].items()}, "copy rmse cm whole/wet",
            round(r["copy_rmse_cm_whole_map"], 3), round(r["copy_rmse_cm_wet_cells"], 3),
            "wet share", round(r["wet_share_of_cells"], 3), "csi-zero steps", r["csi_zero_denominator_steps"], "of", r["steps"])
log("ALL DONE")

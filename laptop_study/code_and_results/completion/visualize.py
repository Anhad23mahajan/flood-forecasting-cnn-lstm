"""Figures for the corrected study (used by flood_forecasting_study_colab.ipynb; also runnable locally).

  plot_ladder()          skill of every rung (copy the last frame ... ConvLSTM) on the never-seen region, from completion/results/report.json
  plot_ablations()       what the trained models use (test-time removal of one input), from the run files
  plot_regimes()         causal / hybrid / forced
  plot_curves(...)       training curves of one run (train loss / persistence, validation skill per epoch)
  plot_forecast(...)     example forecasts on never-seen windows: last observed depth, true depth at the last lead, forecast, forecast error, persistence error
                         (needs the weights saved by `engine.py run ... --save-weights`)
  plot_gradcam(...)      Grad-CAM of the depth forecast at the ConvLSTM hidden map for a few never-seen windows (same weights)
Every function returns the matplotlib figure and saves a PNG under completion/results/figures/.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib
if "google.colab" not in sys.modules and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import engine as E  # noqa: E402  (chdirs to the project root)

FIG = "completion/results/figures"
NAMES = {"E60": "Australia -> UK, 60 m", "E30": "Australia -> UK, 30 m", "E480": "Pakistan -> Mozambique, 480 m"}
COL = ["#1B4F72", "#5DA9C9", "#8FB88F", "#C98B5D", "#7A6F9B", "#B03A2E"]


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=110, bbox_inches="tight")
    return fig


def _report():
    return json.load(open("completion/results/report.json"))


def plot_ladder(exps=("E60", "E30", "E480"), regime="hybrid"):
    rep = _report()
    h, b = rep["headline"], rep["baselines"]
    rows = [("copy the\nlast frame", lambda e: 0.0), ("straight-line\ntrend", lambda e: (b.get(f"{e}|{regime}|S1") or {}).get("test")),
            ("2-number\nlookup table", lambda e: (b.get(f"{e}|{regime}|S5") or {}).get("test")), ("per-pixel\nMLP", lambda e: (b.get(f"{e}|{regime}|S4c") or {}).get("test")),
            ("original\nCNN-LSTM (v1)", lambda e: (h.get(f"{e}|v1|{regime}") or {}).get("test_mean")), ("local CNN", lambda e: (h.get(f"{e}|local|{regime}") or {}).get("test_mean")),
            ("ConvLSTM\n(final)", lambda e: (h.get(f"{e}|clstm|{regime}") or {}).get("test_mean"))]
    rows = [(n, f) for n, f in rows if any(f(e) is not None for e in exps)]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.8 / len(exps)
    for k, e in enumerate(exps):
        vals = [f(e) if f(e) is not None else np.nan for _, f in rows]
        bars = ax.bar(np.arange(len(rows)) + k * w, vals, w, label=NAMES[e], color=COL[k])
        for x, v in zip(np.arange(len(rows)) + k * w, vals):
            if not np.isnan(v):
                ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(rows)) + w * (len(exps) - 1) / 2)
    ax.set_xticklabels([n for n, _ in rows], fontsize=9)
    ax.set_ylabel("skill vs 'copy the last frame'  (0 = same, 1 = perfect)")
    ax.set_title(f"Skill on the never-seen region ({regime} rain input, mean of 3 tile folds)")
    ax.axhline(0, color="grey", lw=0.8); ax.legend(fontsize=9); ax.set_ylim(-0.05, 1.0)
    fig.tight_layout()
    return _save(fig, "ladder.png")


def plot_ablations(exps=("E60", "E480"), archs=("clstm", "local"), regime="hybrid"):
    g = __import__("d_aggregate").load_runs()
    keys = [("none", "normal"), ("rain_removed", "rain\nremoved"), ("terrain_removed", "terrain\nremoved"), ("depth_hist_frozen", "depth history\nfrozen"), ("rain_shifted_earlier", "rain strictly\npast")]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    combos = [(e, a) for e in exps for a in archs]
    w = 0.8 / max(1, len(combos))
    for k, (e, a) in enumerate(combos):
        runs = g.get((e, a, regime, "full", False, False, E.__dict__.get("EPOCHS_F", 30) if False else int(os.environ.get("FINAL_EPOCHS", 30)), int(os.environ.get("FINAL_NTRAIN", 300))), [])
        vals = [np.mean([r["ablation_test"][key] for r in runs if r.get("ablation_test") and key in r["ablation_test"]]) if runs and all(key in (r.get("ablation_test") or {}) for r in runs) else np.nan for key, _ in keys]
        ax.bar(np.arange(len(keys)) + k * w, vals, w, label=f"{a} / {NAMES[e]}", color=COL[k % len(COL)])
    ax.set_xticks(np.arange(len(keys)) + w * (len(combos) - 1) / 2)
    ax.set_xticklabels([n for _, n in keys], fontsize=9)
    ax.set_ylim(-0.2, 1.0); ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("skill when that input is removed at test time")
    ax.set_title("What the trained models use: the depth history, almost nothing else")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "ablations.png")


def plot_regimes(exps=("E60", "E480"), archs=("local", "clstm")):
    rep = _report()["headline"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    combos = [(e, a) for e in exps for a in archs]
    w = 0.8 / len(combos)
    for k, (e, a) in enumerate(combos):
        vals = [(rep.get(f"{e}|{a}|{r}") or {}).get("test_mean", np.nan) for r in ("causal", "hybrid", "forced")]
        ax.bar(np.arange(3) + k * w, vals, w, label=f"{a} / {NAMES[e]}", color=COL[k % len(COL)])
    ax.set_xticks(np.arange(3) + w * (len(combos) - 1) / 2)
    ax.set_xticklabels(["causal\n(past rain only)", "hybrid\n(the original input)", "forced\n(future rain too)"])
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("skill on the never-seen region"); ax.set_title("Rain settings (axis starts at 0.5)"); ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "regimes.png")


def plot_curves(exp="E60", arch="clstm", regime="hybrid", fold=0):
    fs = sorted(glob.glob(f"completion/results/runs/{exp}_{arch}_{regime}_full_f{fold}of3_*.json"))
    fs = [f for f in fs if "_blk" not in f and "_b8" not in f and "_lr" not in f]
    if not fs:
        raise FileNotFoundError("no run file")
    r = json.load(open(fs[0]))
    ep, tl, vs = zip(*r["history"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(ep, tl); axes[0].axhline(1.0, color="grey", ls="--", label="= copying the last frame"); axes[0].set_title("training loss / persistence loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(ep, vs); axes[1].axhline(0, color="grey", ls="--"); axes[1].set_title("skill on 48 held-out windows (higher is better)"); axes[1].set_xlabel("epoch")
    fig.suptitle(f"{arch} - {NAMES[exp]} - {regime} - fold {fold}")
    fig.tight_layout()
    return _save(fig, f"curves_{exp}_{arch}_{regime}_f{fold}.png")


def _load_model(exp, arch, regime, fold, epochs, K=3, batch=1):
    seed = 1000 + 10 * fold + K
    rid = f"{exp}_{arch}_{regime}_full_f{fold}of{K}_s{seed}_ep{epochs}"
    wd = os.path.join(E.CACHE, "weights")
    ep, dp = os.path.join(wd, rid + "_enc.weights.h5"), os.path.join(wd, rid + "_dec.weights.h5")
    if not (os.path.exists(ep) and os.path.exists(dp)):
        raise FileNotFoundError(f"weights not found ({rid}); train with --save-weights")
    tf = E._tf()
    T = E.T_IN + (E.HORIZON if regime == "forced" else 0)
    enc, dec, dech, _ = E.build_nets(arch, T)
    enc.load_weights(ep); dec.load_weights(dp)
    return tf, enc, dec, dech


def plot_forecast(exp="E60", arch="clstm", regime="hybrid", fold=0, epochs=30, n=3):
    tf, enc, dec, dech = _load_model(exp, arch, regime, fold, epochs)
    tr_pool, te_pool, tr_idx, va_idx, te_idx, held, seed = E.split_for(exp, fold, 3, n_train=int(os.environ.get("FINAL_NTRAIN", 300)), log=lambda s: None)
    # pick wet windows with visible change: the largest persistence error among a spread of never-seen windows
    cand = te_idx[::max(1, len(te_idx) // 40)]
    err = []
    for i in cand:
        x, y, m = E.compose(te_pool, i, regime)
        err.append(float(((x[E.T_IN - 1, ..., 0][None] - y) ** 2)[:, m].sum()))
    pick = [cand[j] for j in np.argsort(err)[::-1][:n]]
    fig, axes = plt.subplots(n, 5, figsize=(16, 3.2 * n))
    axes = np.atleast_2d(axes)
    for r_, i in enumerate(pick):
        x, y, m = E.compose(te_pool, i, regime)
        p = dech(dec, enc(tf.constant(x[None]), training=False), tf.constant(x[None]), E.HORIZON, training=False).numpy()[0]
        last, k = x[E.T_IN - 1, ..., 0], E.HORIZON - 1
        vmax = max(float(y[k].max()), float(last.max()), 1e-3)
        emax = max(float(np.abs(p[k] - y[k]).max()), float(np.abs(last - y[k]).max()), 1e-4)
        panels = [(last, "last observed depth (m)", "Blues", 0, vmax), (y[k], "TRUE depth 2.5 h later (m)", "Blues", 0, vmax), (p[k], f"{arch} forecast 2.5 h (m)", "Blues", 0, vmax),
                  (p[k] - y[k], "forecast error (m)", "RdBu_r", -emax, emax), (last - y[k], "'copy the last frame' error (m)", "RdBu_r", -emax, emax)]
        for c, (img, title, cmap, lo, hi) in enumerate(panels):
            a_ = axes[r_, c]
            im = a_.imshow(np.where(m, img, np.nan), cmap=cmap, vmin=lo, vmax=hi)
            a_.set_title(title if r_ == 0 else "", fontsize=10); a_.axis("off")
            if c in (2, 4):
                fig.colorbar(im, ax=a_, fraction=0.046)
        axes[r_, 0].text(-0.05, 0.5, f"window {i}", transform=axes[r_, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle(f"Never-seen region ({NAMES[exp]}): forecasts of a {arch} model vs the truth and vs copying the last frame ({regime})", y=1.0)
    fig.tight_layout()
    return _save(fig, f"forecast_{exp}_{arch}_{regime}_f{fold}.png")


def plot_gradcam(exp="E60", arch="clstm", regime="hybrid", fold=0, epochs=30, n=3):
    tf, enc, dec, dech = _load_model(exp, arch, regime, fold, epochs)
    import model_v2 as m2
    tr_pool, te_pool, tr_idx, va_idx, te_idx, held, seed = E.split_for(exp, fold, 3, n_train=int(os.environ.get("FINAL_NTRAIN", 300)), log=lambda s: None)
    cand = te_idx[::max(1, len(te_idx) // 40)]
    err = [float(((lambda x, y, m: ((x[E.T_IN - 1, ..., 0][None] - y) ** 2)[:, m].sum())(*E.compose(te_pool, i, regime)))) for i in cand]
    pick = [cand[j] for j in np.argsort(err)[::-1][:n]]
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 8))
    for c, i in enumerate(pick):
        x, y, m = E.compose(te_pool, i, regime)
        xb = tf.constant(x[None], tf.float32)
        with tf.GradientTape() as tape:
            h = enc(xb, training=False)
            hh = h[0] if isinstance(h, (list, tuple)) else h
            tape.watch(hh)
            frames = m2.decode_horizon_local(dec, [hh], xb, E.HORIZON, training=False)
            net = tf.reduce_sum(frames[0]) - E.HORIZON * tf.reduce_sum(xb[0, E.T_IN - 1, ..., 0])
            target = tf.stop_gradient(tf.where(net >= 0, 1.0, -1.0)) * tf.reduce_sum(frames[0])
        g = tape.gradient(target, hh)
        cam = tf.nn.relu(tf.reduce_sum(hh[0] * tf.reduce_mean(g, axis=(1, 2))[0], axis=-1)).numpy()
        cam = cam / (cam.max() + 1e-9)
        last = x[E.T_IN - 1, ..., 0]
        axes[0, c].imshow(np.where(m, last, np.nan), cmap="Blues"); axes[0, c].set_title("last observed depth", fontsize=10); axes[0, c].axis("off")
        axes[1, c].imshow(np.where(m, last, np.nan), cmap="Greys", alpha=0.6)
        axes[1, c].imshow(np.where(m, cam, np.nan), cmap="jet", alpha=0.55, vmin=0, vmax=1); axes[1, c].set_title("Grad-CAM: where the forecast's evidence is", fontsize=10); axes[1, c].axis("off")
    fig.suptitle(f"Grad-CAM of the depth forecast ({arch}, {NAMES[exp]}); hot = evidence for the predicted change", y=1.0)
    fig.tight_layout()
    return _save(fig, f"gradcam_{exp}_{arch}_{regime}_f{fold}.png")

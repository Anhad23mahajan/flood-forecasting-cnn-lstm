"""Baseline ladder (S5 two-feature lookup table, S1-S3 linear, S1c-S3c the same clipped at zero depth, S4/S4c a per-pixel MLP): how much of the neural network's skill can simple per-pixel LINEAR models reproduce, and does depth HISTORY carry
any information at all?  All models are scored on exactly the windows the neural models use (engine.split_for), against persistence.

  S0 persistence            (skill 0 by definition)
  S1 trend only             depth state + depth trend (last-minus-previous, last-minus-first) + 15x15 local means      (no rainfall, no terrain)
  S2 rain + state           depth state + wetness + terrain + rainfall features (+ wet x rain) [+ future rain in the forced regime]
  S3 rain + state + trend   S2 + the depth trend features
Each is a ridge regression of (future depth - last depth) at each of the 5 lead times on the features of the same pixel (plus a
15x15 local mean of the depth state and trend).  No spatial deep model, no LSTM: if S3 ~ the CNN-LSTM then the deep model adds little;
if S3 >> S2 then depth history is informative and a good model should use it.

Run: python completion/b_baselines.py <EXP> <regime> <fold> [--folds 3]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E  # noqa: E402  (chdirs to the project root)

T_IN, HORIZON = E.T_IN, E.HORIZON


def features(x, regime, sets):
    """Feature stack (H*W, F) for one composed window, for feature-set name in `sets` (a list of names)."""
    d = x[:T_IN, ..., 0]
    rain = x[..., 2]
    dT, tr1, tr4 = d[-1], d[-1] - d[-2], d[-1] - d[0]
    dem = x[0, ..., 1]
    dem_rel = (dem - dem.mean()) / 50.0
    wet = (dT > 0.01).astype(np.float32)
    loc = lambda a: uniform_filter(a, size=15, mode="nearest")
    ones = np.ones_like(dT)
    R = [rain[T_IN - 1], rain[:T_IN].mean(0)]
    if regime == "forced":
        R += [rain[T_IN + m] for m in range(HORIZON)]
    cols = {
        "S1": [ones, dT, tr1, tr4, wet, loc(dT), loc(tr1)],
        "S2": [ones, dT, wet, dem_rel, loc(dT)] + R + [wet * r for r in R],
        "S3": [ones, dT, wet, dem_rel, loc(dT), tr1, tr4, loc(tr1)] + R + [wet * r for r in R],
        # S4 (per-pixel MLP): the S3 features plus all four one-step depth changes, in centimetres
        "S4": [dT, wet, dem_rel, loc(dT), loc(tr1)] + [100.0 * (d[k + 1] - d[k]) for k in range(T_IN - 1)] + R + [wet * r for r in R],
    }
    return {k: np.stack([c.reshape(-1) for c in cols[k]], axis=1).astype(np.float32) for k in sets}


def fit(pool, tr_idx, regime, sets, px_per_window=3000, lam=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    XS = {k: [] for k in sets}
    Y = []
    for i in tr_idx:
        x, y, m = E.compose(pool, i, regime)
        F = features(x, regime, sets)
        sel = rng.choice(x.shape[1] * x.shape[2], size=px_per_window, replace=False)
        for k in sets:
            XS[k].append(F[k][sel])
        Y.append((y - x[T_IN - 1, ..., 0][None]).reshape(HORIZON, -1)[:, sel].T)
    Y = np.concatenate(Y)
    coefs = {}
    for k in sets:
        X = np.concatenate(XS[k]).astype(np.float64)
        A = X.T @ X + lam * len(X) * np.eye(X.shape[1])
        coefs[k] = np.linalg.solve(A, X.T @ Y.astype(np.float64))
    return coefs


def predictor(coef, regime, name, clip=False):
    def predict(x):
        F = features(x, regime, [name])[name]
        delta = (F @ coef).T.reshape(HORIZON, x.shape[1], x.shape[2])
        out = x[T_IN - 1, ..., 0][None] + delta
        return np.maximum(out, 0.0) if clip else out              # depth cannot be negative
    return predict


def fit_mlp(pool, tr_idx, regime, px_per_window=2000, epochs=12, seed=0):
    """S4: a small per-pixel MLP (2 x 64 ReLU) on the S4 features -> the 5 lead-time changes (in cm); NO spatial convolution, NO recurrence."""
    tf = E._tf()
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for i in tr_idx:
        x, y, m = E.compose(pool, i, regime)
        F = features(x, regime, ["S4"])["S4"]
        sel = rng.choice(x.shape[1] * x.shape[2], size=px_per_window, replace=False)
        X.append(F[sel])
        Y.append((y - x[T_IN - 1, ..., 0][None]).reshape(HORIZON, -1)[:, sel].T * 100.0)
    X, Y = np.concatenate(X).astype("float32"), np.concatenate(Y).astype("float32")
    mu, sd = X.mean(0), X.std(0) + 1e-6
    tf.keras.utils.set_random_seed(seed)
    net = tf.keras.Sequential([tf.keras.layers.Input((X.shape[1],)), tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dense(64, activation="relu"),
                               tf.keras.layers.Dense(HORIZON)])
    net.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    net.fit((X - mu) / sd, Y, batch_size=2048, epochs=epochs, verbose=0)
    return net, mu, sd


def mlp_predictor(net, mu, sd, regime, clip=False):
    def predict(x):
        F = features(x, regime, ["S4"])["S4"]
        delta = (net.predict((F - mu) / sd, batch_size=16384, verbose=0) / 100.0).T.reshape(HORIZON, x.shape[1], x.shape[2])
        out = x[T_IN - 1, ..., 0][None] + delta
        return np.maximum(out, 0.0) if clip else out
    return predict


D_EDGES = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 1e9])
T_EDGES = np.array([-1e9, -0.20, -0.10, -0.05, -0.02, -0.01, -0.005, -0.002, -0.0005, 0.0005, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 1e9])   # metres per 30 min


def _cell(d_last, tr1):
    return np.clip(np.digitize(d_last, D_EDGES) - 1, 0, len(D_EDGES) - 2), np.clip(np.digitize(tr1, T_EDGES) - 1, 0, len(T_EDGES) - 2)


def fit_table(pool, tr_idx, regime, px_per_window=3000, seed=0):
    """S5: a TWO-FEATURE LOOKUP TABLE. Mean future change (each lead) in each (current depth bin) x (last one-step change bin) cell of the training
    pixels. No learning machinery, no spatial context, no rainfall, no terrain."""
    rng = np.random.default_rng(seed)
    nd, nt = len(D_EDGES) - 1, len(T_EDGES) - 1
    S, N = np.zeros((nd, nt, HORIZON)), np.zeros((nd, nt))
    for i in tr_idx:
        x, y, m = E.compose(pool, i, regime)
        sel = rng.choice(x.shape[1] * x.shape[2], size=px_per_window, replace=False)
        d_last, tr1 = x[T_IN - 1, ..., 0].reshape(-1)[sel], (x[T_IN - 1, ..., 0] - x[T_IN - 2, ..., 0]).reshape(-1)[sel]
        delta = (y - x[T_IN - 1, ..., 0][None]).reshape(HORIZON, -1)[:, sel].T
        a, b = _cell(d_last, tr1)
        np.add.at(S, (a, b), delta)
        np.add.at(N, (a, b), 1.0)
    M = np.where(N[..., None] > 0, S / np.maximum(N[..., None], 1), np.nan)
    marg = np.array([np.nansum(S[a], axis=0) / max(1.0, N[a].sum()) for a in range(nd)])       # depth-bin marginal for empty cells
    for a in range(nd):
        for b in range(nt):
            if N[a, b] < 5:
                M[a, b] = marg[a]
    return M


def table_predictor(M):
    def predict(x):
        d_last = x[T_IN - 1, ..., 0]
        a, b = _cell(d_last, d_last - x[T_IN - 2, ..., 0])
        return d_last[None] + np.moveaxis(M[a, b], -1, 0)
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp"); ap.add_argument("regime"); ap.add_argument("fold", type=int)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--blocked", action="store_true")
    a = ap.parse_args()
    out_p = f"completion/results/baselines_{a.exp}_{a.regime}_f{a.fold}of{a.folds}_n{a.n_train}{'_blk' if a.blocked else ''}.json"
    tr_pool, te_pool, tr_idx, va_idx, te_idx, held, seed = E.split_for(a.exp, a.fold, a.folds, n_train=a.n_train, log=lambda s: print(s, flush=True), blocked=a.blocked)
    sets = ["S1", "S2", "S3"]
    coefs = fit(tr_pool, tr_idx, a.regime, sets)
    out = {"exp": a.exp, "regime": a.regime, "fold": a.fold, "K": a.folds, "n_train": len(tr_idx), "blocked": a.blocked, "S0": {"heldout": 0.0, "test": 0.0}}
    for k in sets:
        for clip in (False, True):
            pr = predictor(coefs[k], a.regime, k, clip)
            key = k + ("c" if clip else "")
            out[key] = {"heldout": E.summarize(E.evaluate(pr, tr_pool, va_idx, a.regime))["skill"],
                        "test": E.summarize(E.evaluate(pr, te_pool, te_idx, a.regime))["skill"]}
            print(f"{a.exp} {a.regime} fold {a.fold}: {key} heldout {out[key]['heldout']:+.4f}  test {out[key]['test']:+.4f}", flush=True)
    pr = table_predictor(fit_table(tr_pool, tr_idx, a.regime))
    out["S5"] = {"heldout": E.summarize(E.evaluate(pr, tr_pool, va_idx, a.regime))["skill"], "test": E.summarize(E.evaluate(pr, te_pool, te_idx, a.regime))["skill"]}
    print(f"{a.exp} {a.regime} fold {a.fold}: S5 (2-feature lookup table) heldout {out['S5']['heldout']:+.4f}  test {out['S5']['test']:+.4f}", flush=True)
    net, mu, sd = fit_mlp(tr_pool, tr_idx, a.regime)
    for clip in (False, True):
        pr = mlp_predictor(net, mu, sd, a.regime, clip)
        key = "S4" + ("c" if clip else "")
        out[key] = {"heldout": E.summarize(E.evaluate(pr, tr_pool, va_idx, a.regime))["skill"],
                    "test": E.summarize(E.evaluate(pr, te_pool, te_idx, a.regime))["skill"]}
        print(f"{a.exp} {a.regime} fold {a.fold}: {key} heldout {out[key]['heldout']:+.4f}  test {out[key]['test']:+.4f}", flush=True)
    os.makedirs("completion/results", exist_ok=True)
    json.dump(out, open(out_p, "w"), indent=1)


if __name__ == "__main__":
    main()

"""Checks Aman's claim about the ridge baselines (completion/b_baselines.py) WITHOUT touching any project file.
(a) 'orig'  : B.fit exactly as shipped (lam=1e-3 on raw features)  -> must reproduce the existing baselines_*_n300.json numbers
(b) 'std'   : same ridge (lam=1e-3) but on standardised features
(c) 'ols'   : standardised features, almost no penalty (lam=1e-9)
Only the linear models S1, S2, S3 are re-fitted. Nothing is written into the project folder.
rasterio (blocked on this machine) and scipy (not installed) are replaced by small stand-ins: the data are read from the
existing .npy caches, and uniform_filter is re-implemented in numpy (edge padding = scipy mode='nearest')."""
import json
import os
import sys
import time
import types

import numpy as np

# ---- stand-ins -------------------------------------------------------------------------------------------------
ras = types.ModuleType("rasterio")
ras_enums = types.ModuleType("rasterio.enums")
ras_enums.Resampling = types.SimpleNamespace(average="average")
ras.enums = ras_enums
sys.modules["rasterio"] = ras
sys.modules["rasterio.enums"] = ras_enums


def uniform_filter(a, size=15, mode="nearest"):
    assert mode == "nearest" and size % 2 == 1
    a = np.asarray(a)
    h = size // 2
    p = np.pad(a.astype(np.float64), h, mode="edge")
    c = np.zeros((p.shape[0] + 1, p.shape[1] + 1))
    c[1:, 1:] = p.cumsum(0).cumsum(1)
    out = (c[size:, size:] - c[:-size, size:] - c[size:, :-size] + c[:-size, :-size]) / float(size * size)
    return out.astype(a.dtype)


sc = types.ModuleType("scipy")
nd = types.ModuleType("scipy.ndimage")
nd.uniform_filter = uniform_filter
sc.ndimage = nd
sys.modules["scipy"] = sc
sys.modules["scipy.ndimage"] = nd

PROJ = r"C:\Dev\flood-forecast-demo"
sys.path.insert(0, os.path.join(PROJ, "completion"))
import b_baselines as B  # noqa: E402
import engine as E  # noqa: E402

T_IN, HORIZON = E.T_IN, E.HORIZON
LAM = 1e-3


def make_fit(mode):
    def fit(pool, tr_idx, regime, sets, px_per_window=3000, lam=LAM, seed=0):
        rng = np.random.default_rng(seed)
        XS = {k: [] for k in sets}
        Y = []
        for i in tr_idx:
            x, y, m = E.compose(pool, i, regime)
            F = B.features(x, regime, sets)
            sel = rng.choice(x.shape[1] * x.shape[2], size=px_per_window, replace=False)
            for k in sets:
                XS[k].append(F[k][sel])
            Y.append((y - x[T_IN - 1, ..., 0][None]).reshape(HORIZON, -1)[:, sel].T)
        Y = np.concatenate(Y).astype(np.float64)
        coefs = {}
        for k in sets:
            X = np.concatenate(XS[k]).astype(np.float64)
            if mode == "orig":
                A = X.T @ X + lam * len(X) * np.eye(X.shape[1])
                coefs[k] = np.linalg.solve(A, X.T @ Y)
            else:
                l2 = lam if mode == "std" else 1e-9
                mu, sd = X.mean(0), X.std(0)
                const = sd < 1e-12                       # the column of ones
                sd = np.where(const, 1.0, sd)
                mu = np.where(const, 0.0, mu)
                Z = (X - mu) / sd
                A = Z.T @ Z + l2 * len(Z) * np.eye(Z.shape[1])
                cz = np.linalg.solve(A, Z.T @ Y)
                coef = cz / sd[:, None]                  # back to raw features
                # intercept goes to the constant column: b = b_z - sum(c_j * mu_j)
                shift = (cz * (mu / sd)[:, None]).sum(0)
                ci = int(np.argmax(const))
                coef[ci] = coef[ci] - shift
                coefs[k] = coef
        return coefs
    return fit


def run(exp, regime="hybrid", folds=3, n_train=300, modes=("orig", "std", "ols"), sets=("S1", "S2", "S3")):
    res = {m: {k: [] for k in sets} for m in modes}
    ref = {k: [] for k in sets}
    for fold in range(folds):
        t0 = time.time()
        tr_pool, te_pool, tr_idx, va_idx, te_idx, held, seed = E.split_for(exp, fold, folds, n_train=n_train, log=lambda s: None)
        jf = os.path.join(PROJ, "completion", "results", "baselines_%s_%s_f%dof%d_n%d.json" % (exp, regime, fold, folds, n_train))
        j = json.load(open(jf))
        for k in sets:
            ref[k].append(j[k]["test"])
        for m in modes:
            coefs = make_fit(m)(tr_pool, tr_idx, regime, list(sets))
            for k in sets:
                pr = B.predictor(coefs[k], regime, k, False)
                res[m][k].append(E.summarize(E.evaluate(pr, te_pool, te_idx, regime))["skill"])
        print("  %s fold %d done in %.0f s" % (exp, fold, time.time() - t0), flush=True)
    print("\n=== %s %s, mean of %d folds, skill on the never-seen flood" % (exp, regime, folds))
    print("%-5s %12s %12s %12s %12s" % ("set", "json (shipped)", "orig(rerun)", "std ridge", "~OLS"))
    for k in sets:
        print("%-5s %12.3f %12.3f %12.3f %12.3f" % (k, np.mean(ref[k]), np.mean(res["orig"][k]), np.mean(res["std"][k]), np.mean(res["ols"][k])), flush=True)
    return ref, res


if __name__ == "__main__":
    exps = sys.argv[1:] or ["E60"]
    out = {}
    for e in exps:
        ref, res = run(e)
        out[e] = {"ref": {k: [float(v) for v in vs] for k, vs in ref.items()}, "res": {m: {k: [float(v) for v in vs] for k, vs in d.items()} for m, d in res.items()}}
        json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ridge_check_out.json"), "w"), indent=1)

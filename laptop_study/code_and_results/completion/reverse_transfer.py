"""REVERSE-direction transfers (EXPLORATORY, added 20 September after the user asked that nothing be dropped).

The paper's transfers train on the big regions (Pakistan, Australia) and test on the small ones (Mozambique, UK). Here the roles are swapped:
  R480  Mozambique 480 m -> Pakistan 480 m      R60  UK 60 m -> Australia 60 m      R30  UK 30 m -> Australia 30 m
The small regions have no room for tile folds (Mozambique 1 valid 128-px tile, UK 30 m 2, UK 60 m 0 - its domain is smaller than a patch), so the
protocol differs from the forward experiments and is NOT comparable with them cell by cell:
  * training windows come from the small region's "test-role" pool (its four corner tiles, or the padded windows for UK 60 m; 60 start times);
  * the 3 'folds' are TIME blocks of that pool (fold k = the k-th third of the start times is held out for validation, training windows whose 54-step span
    touches the block are dropped = the same embargo as the strict split); there are NO held-out tiles - `heldout` in the result files means held-out TIME blocks;
  * the never-seen region is the big one (every valid tile x 60 start times);
  * test-time ablations use at most 400 evenly spaced windows of the never-seen region (the big pools have thousands).
Everything else (recipe, models, metrics, dead-network guard) is engine.py, which is NOT modified: this module only swaps `engine.split_for` and
`engine.ablation_skills` in memory, then calls engine.run / b_baselines.main.
Usage:  python completion/reverse_transfer.py run R480 hybrid 0 --arch local --epochs 30 --n-train 300 [--no-ablate]
        python completion/reverse_transfer.py baseline R480 hybrid 0 --n-train 300
        python completion/reverse_transfer.py check R480 0          (split sizes only)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E  # noqa: E402  (chdirs to the project root)

MAX_ABLATE_WINDOWS = 400


def time_split_for(exp, fold, K=3, seed=None, n_train=400, n_val=300, log=print, blocked=False):
    cfg = E.EXPS[exp]
    (tr_reg, tr_res), (te_reg, te_res) = cfg["train"], cfg["test"]
    seed = 1000 + 10 * fold + K if seed is None else seed
    tr_pool = E.build_pool(tr_reg, tr_res, "test", log)          # corner / padded windows of the small region (60 start times)
    te_pool = E.build_pool(te_reg, te_res, "test", log)          # every valid tile of the big region
    starts = tr_pool["meta"][:, 2]
    blocks = np.array_split(np.array(sorted(set(starts.tolist()))), K)
    blk = set(blocks[fold].tolist())
    b_lo, b_hi = min(blk), max(blk)
    tr_all = [i for i in range(len(starts)) if not (b_lo - 54 <= int(starts[i]) <= b_hi + 54)]
    va_all = [i for i in range(len(starts)) if int(starts[i]) in blk]
    rng = np.random.default_rng(seed)
    tr_idx = sorted(rng.choice(tr_all, size=min(n_train, len(tr_all)), replace=False).tolist())
    va_idx = sorted(rng.choice(va_all, size=min(n_val, len(va_all)), replace=False).tolist())
    te_idx = list(range(len(te_pool["meta"])))
    log(f"{tr_reg} {tr_res} m (time-block folds, no tile folds): fold {fold}/{K}: wet windows train pool {len(tr_all)}, held-out time-block pool {len(va_all)}; "
        f"using {len(tr_idx)} train / {len(va_idx)} time-held-out / {len(te_idx)} {te_reg} windows")
    return tr_pool, te_pool, tr_idx, va_idx, te_idx, set(), seed


_orig_ablation = E.ablation_skills


def _ablation_subsampled(predict, pool, idx, regime, variant):
    idx = list(idx)
    if len(idx) > MAX_ABLATE_WINDOWS:
        idx = [idx[k] for k in np.linspace(0, len(idx) - 1, MAX_ABLATE_WINDOWS).astype(int)]
    return _orig_ablation(predict, pool, idx, regime, variant)


def main():
    if "--tag" in sys.argv:                  # ignored marker argument (the queue runners count running jobs by looking for 'engine.py run' in the command line)
        i = sys.argv.index("--tag")
        del sys.argv[i:i + 2]
    mode = sys.argv[1]
    E.split_for = time_split_for
    E.ablation_skills = _ablation_subsampled
    if mode == "check":
        exp, fold = sys.argv[2], int(sys.argv[3])
        tr, te, tri, vai, tei, _, _ = time_split_for(exp, fold, 3, n_train=300)
        print("train pool", len(tr["meta"]), "size", tr["size"], "| test pool", len(te["meta"]), "size", te["size"], "tiles", int(np.load(os.path.join(E.CACHE, f'pool_{te["region"]}_{te["res"]}_test_meta.npz'))["n_tiles"]))
        return
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if mode == "run":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("exp"); ap.add_argument("regime"); ap.add_argument("fold", type=int)
        ap.add_argument("--arch", default="local"); ap.add_argument("--variant", default="full")
        ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--n-train", type=int, default=300)
        ap.add_argument("--seed", type=int, default=None); ap.add_argument("--no-ablate", action="store_true")
        a = ap.parse_args()
        E.run(a.exp, a.regime, a.fold, variant=a.variant, K=3, epochs=a.epochs, seed=a.seed, n_train=a.n_train, n_test_ablate=not a.no_ablate,
              log=lambda s: print(s, flush=True), arch=a.arch)
    elif mode == "baseline":
        import b_baselines
        b_baselines.main()
    else:
        raise SystemExit("mode must be run | baseline | check")


if __name__ == "__main__":
    main()

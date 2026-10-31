"""Aggregate completion/results/runs/*.json (per-fold neural runs) and completion/results/baselines_*.json (baseline ladder) into tables.

Only runs of the FROZEN recipe (batch 1, lr 1e-3, the default seed of each fold, epochs/n_train as set by FINAL_EPOCHS / FINAL_NTRAIN) enter the headline groups;
runs with another batch size, learning rate or seed (batches OPT and F) are kept apart (`load_runs(extra=True)`) so they can never merge into a headline cell.

Uncertainty (revised after the independent code review, finding F2): neighbouring test windows overlap strongly in time (a window spans 54 depth steps, start times are
14-50 steps apart) and every never-seen region is ONE flood event, so start times are NOT independent.  Intervals therefore use BLOCK bootstraps:
  * never-seen region: start times sorted in time and cut into non-overlapping blocks of ceil(54 / median spacing) + 1 starts; blocks are resampled;
  * held-out tiles: a TWO-WAY cluster bootstrap that resamples tiles AND time blocks (weight of a window = count of its tile x count of its block);
  * paired differences (e.g. clstm - local) resample the same time blocks for both models.
Even so, each region is one event, and E60 / E30 share the same Australia and UK events: there are two independent train -> test pairs, not nine.
Usage: python completion/d_aggregate.py [--md out.md]
"""
import argparse
import collections
import glob
import json
import math
import os

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB = 2000
EPOCHS_F = int(os.environ.get("FINAL_EPOCHS", 30))      # the frozen recipe (override only for reduced re-runs, e.g. the Colab notebook)
NTR_F = int(os.environ.get("FINAL_NTRAIN", 300))
SPAN = 54                                               # depth steps covered by one window (5 input + 5 forecast frames at stride 6, minus one)


def _default_seed(r):
    return r["seed"] == 1000 + 10 * r["fold"] + r["K"]


def load_runs(extra=False):
    groups = collections.defaultdict(list)
    for f in sorted(glob.glob("completion/results/runs/*.json")):
        r = json.load(open(f))
        small_pool = r["exp"].startswith("R") and r["n_train"] <= NTR_F                    # reverse transfers train on ALL windows of a small region's pool (fewer than 300)
        if r.get("test") is None or r["epochs"] != EPOCHS_F or (r["n_train"] != NTR_F and not small_pool):      # design-phase / blind runs are not part of the final tables
            continue
        frozen = int(r.get("batch", 1)) == 1 and float(r.get("lr", 1e-3)) == 1e-3 and _default_seed(r)
        if frozen == extra:
            continue
        key = (r["exp"], r.get("arch", "v1"), r["regime"], r["variant"], bool(r.get("augment", False)), bool(r.get("blocked", False)), r["epochs"], NTR_F if small_pool else r["n_train"])
        if extra:
            key += (int(r.get("batch", 1)), float(r.get("lr", 1e-3)), int(r["seed"]) if not _default_seed(r) else "default", bool(r.get("guard", True)))
        groups[key].append(r)
    return groups


def load_baselines():
    b = collections.defaultdict(list)
    for f in glob.glob(f"completion/results/baselines_*_n{NTR_F}*.json"):
        d = json.load(open(f))
        b[(d["exp"], d["regime"], bool(d.get("blocked", False)))].append(d)
    return b


def complete(runs):
    """True only if the group holds exactly the K folds 0..K-1, once each."""
    return bool(runs) and sorted(r["fold"] for r in runs) == list(range(runs[0]["K"]))


def pooled_skill(win_list):
    m = sum(np.sum(w["sse_m"]) for w in win_list)
    p = sum(np.sum(w["sse_p"]) for w in win_list)
    return 1 - m / p


def _blocks(starts):
    """start index -> time-block id, blocks of ceil(SPAN / median spacing) + 1 consecutive start times; also the number of blocks."""
    u = sorted(set(int(s) for s in starts))
    sp = float(np.median(np.diff(u))) if len(u) > 1 else float(SPAN)
    L = int(math.ceil(SPAN / max(sp, 1.0))) + 1
    return {s: i // L for i, s in enumerate(u)}, int(math.ceil(len(u) / L))


def tile_bootstrap(runs, n=NB):
    """Pooled held-out skill: two-way cluster bootstrap over tiles x time blocks (every tile is held out in exactly one fold)."""
    rng = np.random.default_rng(12345)
    rows = []                                           # (tile id, start, sse_m, sse_p)
    for r in runs:
        w = r["heldout_windows"]
        for tr, tc, st, m, p in zip(w["tile_r"], w["tile_c"], w["start"], np.sum(w["sse_m"], axis=1), np.sum(w["sse_p"], axis=1)):
            rows.append(((tr, tc), st, m, p))
    tiles = sorted({t for t, *_ in rows})
    if len(tiles) < 2:
        return None
    tid = {t: i for i, t in enumerate(tiles)}
    blk, nb = _blocks([st for _, st, _, _ in rows])
    T = np.array([tid[t] for t, *_ in rows]); B = np.array([blk[int(st)] for _, st, _, _ in rows])
    M = np.array([m for *_, m, _ in rows]); P = np.array([p for *_, p in rows])
    out = []
    for _ in range(n):
        ct = np.bincount(rng.integers(0, len(tiles), len(tiles)), minlength=len(tiles))
        cb = np.bincount(rng.integers(0, nb, nb), minlength=nb)
        w = ct[T] * cb[B]
        out.append(1 - (w * M).sum() / max((w * P).sum(), 1e-30))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def _per_fold_starts(runs):
    per = []
    for r in runs:
        w = r["test_windows"]
        d = collections.defaultdict(lambda: [0.0, 0.0])
        for s, m, p in zip(w["start"], np.sum(w["sse_m"], axis=1), np.sum(w["sse_p"], axis=1)):
            d[s][0] += m
            d[s][1] += p
        per.append(d)
    return per


def start_bootstrap(runs, n=NB):
    """Mean over folds of the never-seen-region skill; resamples TIME BLOCKS of start times (non-overlapping blocks)."""
    rng = np.random.default_rng(12345)
    per = _per_fold_starts(runs)
    starts = sorted(set(runs[0]["test_windows"]["start"]))
    blk, nb = _blocks(starts)
    B = np.array([blk[s] for s in starts])
    Mf = np.array([[d[s][0] for s in starts] for d in per]); Pf = np.array([[d[s][1] for s in starts] for d in per])
    out = []
    for _ in range(n):
        cb = np.bincount(rng.integers(0, nb, nb), minlength=nb)[B]
        out.append(np.mean(1 - (Mf * cb).sum(1) / np.maximum((Pf * cb).sum(1), 1e-30)))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def paired_start_bootstrap(runs_a, runs_b, n=NB):
    """(mean difference, [95% CI]) of the fold-averaged never-seen skill of group A minus group B, resampling the SAME time blocks for both."""
    rng = np.random.default_rng(12345)
    pa, pb = _per_fold_starts(runs_a), _per_fold_starts(runs_b)
    starts = sorted(set(runs_a[0]["test_windows"]["start"]) & set(runs_b[0]["test_windows"]["start"]))
    blk, nb = _blocks(starts)
    B = np.array([blk[s] for s in starts])

    def arr(per, i):
        return np.array([[d[s][i] for s in starts] for d in per])
    Ma, Pa, Mb, Pb = arr(pa, 0), arr(pa, 1), arr(pb, 0), arr(pb, 1)
    point = float(np.mean(1 - Ma.sum(1) / Pa.sum(1)) - np.mean(1 - Mb.sum(1) / Pb.sum(1)))
    out = []
    for _ in range(n):
        cb = np.bincount(rng.integers(0, nb, nb), minlength=nb)[B]
        out.append(np.mean(1 - (Ma * cb).sum(1) / np.maximum((Pa * cb).sum(1), 1e-30)) - np.mean(1 - (Mb * cb).sum(1) / np.maximum((Pb * cb).sum(1), 1e-30)))
    return point, [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="completion/results/summary.md")
    a = ap.parse_args()
    groups, base = load_runs(), load_baselines()
    lines = ["| exp | arch | regime | variant | aug | blocked | K | held-out skill per fold | held-out pooled [95% CI] | test skill per fold | test mean [95% CI] | S1 / S2 / S3 (held-out; test) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for key, runs in sorted(groups.items()):
        runs = sorted(runs, key=lambda r: r["fold"])
        K = runs[0]["K"]
        exp, arch, regime, variant, aug, blk, ep, ntr = key
        ho = [r["heldout"]["skill"] for r in runs]
        te = [r["test"]["skill"] for r in runs]
        ok = complete(runs)
        ci_h = tile_bootstrap(runs) if ok else None
        ci_t = start_bootstrap(runs) if ok else None
        b = base.get((exp, regime, blk), [])
        bs = " / ".join(f"{np.mean([x[k]['heldout'] for x in b if k in x]):+.3f};{np.mean([x[k]['test'] for x in b if k in x]):+.3f}" for k in ("S1", "S2", "S3")) if b else "-"
        fmt = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]" if c else "(incomplete)"
        lines.append(f"| {exp} | {arch} | {regime} | {variant} | {'y' if aug else 'n'} | {'y' if blk else 'n'} | {len(runs)}/{K} | {' '.join(f'{v:+.3f}' for v in ho)} | "
                     f"{pooled_skill([r['heldout_windows'] for r in runs]):+.3f} {fmt(ci_h)} | {' '.join(f'{v:+.3f}' for v in te)} | {np.mean(te):+.3f} {fmt(ci_t)} | {bs} |")
        summary["|".join(map(str, key))] = dict(heldout=ho, test=te, ci_heldout=ci_h, ci_test=ci_t,
                                                ablation_heldout={k: float(np.mean([r["ablation_heldout"][k] for r in runs if r.get("ablation_heldout") and k in r["ablation_heldout"]]))
                                                                  for k in (runs[0].get("ablation_heldout") or {})},
                                                ablation_test={k: float(np.mean([r["ablation_test"][k] for r in runs if r.get("ablation_test") and k in r["ablation_test"]]))
                                                               for k in (runs[0].get("ablation_test") or {})})
    md = "\n".join(lines)
    extra = load_runs(extra=True)
    if extra:
        md += "\n\n### Non-frozen runs (other batch size / learning rate / seed): reported separately, never merged\n\n| exp | arch | regime | batch | lr | seed | n runs | test skill per fold | held-out skill per fold |\n|---|---|---|---|---|---|---|---|---|\n"
        for key, runs in sorted(extra.items(), key=lambda kv: str(kv[0])):
            runs = sorted(runs, key=lambda r: r["fold"])
            tk = " ".join("%+.3f" % r["test"]["skill"] for r in runs)
            hk = " ".join("%+.3f" % r["heldout"]["skill"] for r in runs)
            md += f"| {key[0]} | {key[1]} | {key[2]} | {key[8]} | {key[9]} | {key[10]} | {len(runs)} | {tk} | {hk} |\n"
    open(a.md, "w", encoding="utf-8").write(md + "\n")
    json.dump(summary, open(a.md.replace(".md", ".json"), "w"), indent=1)
    print(md)


if __name__ == "__main__":
    main()

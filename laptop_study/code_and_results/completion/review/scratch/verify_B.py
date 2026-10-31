import json, sys
import numpy as np
from vcommon import *

RUNS = r"C:\Dev\flood-forecast-demo\completion\results\runs"

def run(key, tiles, size_h, size_w, run_json):
    secs, files, bad, diffs = depth_axis(key)
    n = len(secs)
    assert secs == [300 * i for i in range(n)], "file names are not 0,300,600,... contiguous"
    L = n - 1 - 6 * (5 + 5 - 1)
    starts = np.unique(np.linspace(0, L, 60).astype(int))
    print(key, "n files", n, "L", L, "n unique starts", len(starts), starts[:6], starts[-3:])
    cache = {}
    def fr(i):
        if i not in cache: cache[i] = read_tif(files[300 * i]).astype(np.float64)
        return cache[i]
    sse_tot = 0.0; npx_tot = 0; kept = []; dropped = []; per = {}
    nan_seen = 0
    for s in starts:
        last_h = fr(s + 24); fut = [fr(s + 30 + 6 * k) for k in range(5)]
        nan_seen += np.isnan(last_h).sum() + sum(np.isnan(f).sum() for f in fut)
        for (r, c) in tiles:
            ph = last_h[r:r + size_h, c:c + size_w]; pf_last = fut[4][r:r + size_h, c:c + size_w]
            assert ph.shape == (size_h, size_w)
            if max(ph.max(), pf_last.max()) < 0.05:
                dropped.append((r, c, int(s))); continue
            sse = np.array([((fut[k][r:r + size_h, c:c + size_w] - ph) ** 2).sum() for k in range(5)])
            per[(r, c, int(s))] = sse
            sse_tot += sse.sum(); npx_tot += 5 * size_h * size_w
            kept.append((r, c, int(s)))
        if len(cache) > 40:
            for kk in [k for k in cache if k < s]: cache.pop(kk)
    rmse = np.sqrt(sse_tot / npx_tot) * 1000
    print(" NaN seen", nan_seen, "kept windows", len(kept), "dropped", len(dropped), dropped[:5])
    print(" persistence RMSE mm: %.6f" % rmse)
    j = json.load(open(f"{RUNS}\{run_json}"))
    tw = j["test_windows"]; claimed_n = j["test"]["n_windows"]; claimed_rmse = j["test"]["rmse_persistence_mm"]
    jk = list(zip(tw["tile_r"], tw["tile_c"], tw["start"]))
    print(" claimed n", claimed_n, "claimed rmse %.6f" % claimed_rmse, " diff %.3e" % abs(rmse - claimed_rmse))
    print(" window set equal to project's stored (tile_r,tile_c,start) set:", set(jk) == set(kept),
          " only-mine", len(set(kept) - set(jk)), " only-theirs", len(set(jk) - set(kept)))
    # per-window comparison of sse_p (m^2 per lead) & npx
    maxrel = 0; maxabs = 0
    for i, key3 in enumerate(jk):
        if key3 in per:
            a = np.array(tw["sse_p"][i]); b = per[key3]
            maxabs = max(maxabs, np.abs(a - b).max()); maxrel = max(maxrel, np.max(np.abs(a - b) / np.maximum(b, 1e-12)))
            assert tw["npx"][i] == size_h * size_w, (tw["npx"][i])
    print(" per-window per-lead sse_p max abs diff (m^2) %.3e, max rel diff %.3e" % (maxabs, maxrel))
    # project's own aggregation of its stored sse_p/npx, for cross-check of the JSON internal consistency
    sp = np.array(tw["sse_p"]).sum(); npx = np.array(tw["npx"]).sum() * 5
    print(" rmse from stored sse_p/npx (json self-consistency): %.6f" % (np.sqrt(sp / npx) * 1000))
    # per-lead RMSE mine
    per_lead = np.sqrt(np.sum([per[k] for k in kept], axis=0) / (len(kept) * size_h * size_w)) * 1000
    print(" per-lead persistence rmse mm", np.round(per_lead, 3))
    return dict(n=len(kept), rmse=rmse, claimed_n=claimed_n, claimed_rmse=claimed_rmse, per_lead=per_lead.tolist(), dropped=dropped, maxabs=float(maxabs), maxrel=float(maxrel))

res = {}
res["Mozambique_480"] = run("Mozambique_480", [(r, c) for r in (0, 23) for c in (0, 10)], 128, 128, "E480_local_hybrid_full_f0of3_s1003_ep30.json")
res["UK_60"] = run("UK_60", [(0, 0), (0, 52)], 85, 85, "E60_local_hybrid_full_f0of3_s1003_ep30.json")
json.dump(res, open("B_result.json", "w"), indent=1)

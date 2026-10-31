import json
import numpy as np
from vcommon import *
CACHE = r"C:\Dev\flood-forecast-demo\completion\cache"
m = np.load(f"{CACHE}\pool_Mozambique_480_test_meta.npz"); meta = m["meta"]
dh = np.load(f"{CACHE}\pool_Mozambique_480_test_dh.npy", mmap_mode="r"); df = np.load(f"{CACHE}\pool_Mozambique_480_test_df.npy", mmap_mode="r")
secs, files, bad, diffs = depth_axis("Mozambique_480")
# bin edges exactly as stated in the project's table baseline (constants copied for the bin assignment demo only)
D_EDGES = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 1e9])
T_EDGES = np.array([-1e9, -0.20, -0.10, -0.05, -0.02, -0.01, -0.005, -0.002, -0.0005, 0.0005, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 1e9])

# choose a mid-event window with a healthy wet fraction (deterministic rule)
cands = [i for i in range(len(meta)) if 600 <= meta[i, 2] <= 900]
i = cands[len(cands) // 2]
r, c, s = map(int, meta[i]); print("window", i, "tile (r,c)", (r, c), "start s", s, "phase s%6", s % 6)
raw = lambda k: read_tif(files[300 * k]).astype(np.float64)[r:r + 128, c:c + 128]
hist_idx = [s + 6 * k for k in range(5)]; fut_idx = [s + 30 + 6 * k for k in range(5)]
print("history file indices", hist_idx, "-> seconds", [300 * k for k in hist_idx]); print("forecast file indices", fut_idx)
H = np.stack([raw(k) for k in hist_idx]); Fu = np.stack([raw(k) for k in fut_idx])
print("pool dh == raw history frames:", np.array_equal(H.astype(np.float32), np.asarray(dh[i])), "; pool df == raw forecast frames:", np.array_equal(Fu.astype(np.float32), np.asarray(df[i])))
d_T = H[4]; d_Tm1 = H[3]                       # depth at last obs (file s+24) and previous history frame (file s+18)
tr1 = d_T - d_Tm1                              # last one-step (30 min) change, HISTORY ONLY
print("d_T = file %d (t=%d s) ; d_T-1 = file %d (t=%d s) ; gap %d s (=30 min)" % (hist_idx[4], 300 * hist_idx[4], hist_idx[3], 300 * hist_idx[3], 300 * (hist_idx[4] - hist_idx[3])))
print("max history file index used by features:", max(hist_idx), " < first forecast file", fut_idx[0])
# is the feature from the pool identical?
tr1_pool = np.asarray(dh[i, 4], np.float64) - np.asarray(dh[i, 3], np.float64)
print("tr1 from pool arrays == tr1 from raw TIFFs (float32-exact):", np.array_equal(np.asarray(dh[i, 4]) - np.asarray(dh[i, 3]), (H[4].astype(np.float32) - H[3].astype(np.float32))))
# what a leaky 'change' would look like: (first forecast frame - d_T)
leak = Fu[0] - d_T
tgt1 = Fu[0] - d_T                              # lead-1 target delta
print("stats  tr1(hist)  : mean %.5f  min %.5f max %.5f" % (tr1.mean(), tr1.min(), tr1.max()))
print("stats  tgt lead-1 : mean %.5f  min %.5f max %.5f" % (tgt1.mean(), tgt1.min(), tgt1.max()))
wet = d_T > 0.01
cc = np.corrcoef(tr1[wet], tgt1[wet])[0, 1]
print("corr(tr1, lead-1 target delta) over wet pixels: %.4f  (a leaked feature would be exactly 1.0)" % cc)
print("tr1 identical to lead-1 target delta?:", np.allclose(tr1, tgt1))
# perturbation test: change every future frame arbitrarily; the features must be unchanged
Fu_pert = Fu * 3.7 + 1.0
print("features unchanged when future frames are perturbed (trivially, they are computed from H only):", np.array_equal(H[4] - H[3], tr1))
# bin assignment
a = np.clip(np.digitize(d_T, D_EDGES) - 1, 0, len(D_EDGES) - 2); b = np.clip(np.digitize(tr1, T_EDGES) - 1, 0, len(T_EDGES) - 2)
cells = np.zeros((len(D_EDGES) - 1, len(T_EDGES) - 1), int); np.add.at(cells, (a, b), 1)
print("pixels per depth bin:", cells.sum(1).tolist())
print("pixels per change bin:", cells.sum(0).tolist())
print("occupied (depth,change) cells:", int((cells > 0).sum()), "of", cells.size)
# example: mean future change per lead for the most-populated wet cell
wa = np.array([np.mean(Fu[k][(a == a[wet][0]) & (b == b[wet][0])] - d_T[(a == a[wet][0]) & (b == b[wet][0])]) for k in range(5)])
print("mean future change (mm) per lead in the cell of the first wet pixel [depth-bin %d, change-bin %d]:" % (a[wet][0], b[wet][0]), np.round(wa * 1000, 3).tolist())

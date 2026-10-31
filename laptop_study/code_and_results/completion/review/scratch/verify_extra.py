import json
import numpy as np
from vcommon import *
CACHE = r"C:\Dev\flood-forecast-demo\completion\cache"
# 1. file naming / contiguity for all six depth sets
for key in DEPTH_DIR:
    secs, files, bad, diffs = depth_axis(key)
    print("naming", key, "n", len(secs), "contiguous 0..(n-1)*300:", secs == [300 * i for i in range(len(secs))], "non-integer-named entries:", bad)
# 2. project's cached rainfall stack vs my raw-TIFF block-means
for key in ["Mozambique_480", "Australia_30"]:
    z = np.load(f"A_{key}.npz"); R = z["R"]
    cache = np.load(f"{CACHE}\{key}_rain.npy", mmap_mode="r")
    print(key, "cache rain shape", cache.shape, "my R len", len(R))
    ids = range(len(R)) if key.startswith("Moz") else range(0, len(R), 20)
    dm = [float(np.asarray(cache[j], np.float64).mean()) for j in ids]
    d = np.abs(np.array(dm) - R[list(ids)])
    print("  max |cache domain-mean - raw domain-mean| over", len(list(ids)), "files: %.3e (max rel %.3e)" % (d.max(), (d / np.maximum(R[list(ids)], 1e-9)).max()))
# 3. Moz persistence RMSE if the 2 dropped (dry) windows were included
secs, files, bad, diffs = depth_axis("Mozambique_480"); n = len(secs)
L = n - 1 - 54; starts = np.unique(np.linspace(0, L, 60).astype(int))
sse = 0; npx = 0; sse_d = 0; npx_d = 0
for s in starts:
    ph = read_tif(files[300 * (s + 24)]).astype(np.float64); fut = [read_tif(files[300 * (s + 30 + 6 * k)]).astype(np.float64) for k in range(5)]
    for r in (0, 23):
        for c in (0, 10):
            sl = (slice(r, r + 128), slice(c, c + 128))
            e = sum(((f[sl] - ph[sl]) ** 2).sum() for f in fut)
            if max(ph[sl].max(), fut[4][sl].max()) >= 0.05: sse += e; npx += 5 * 128 * 128
            else: sse_d += e; npx_d += 5 * 128 * 128
print("Moz persistence RMSE kept-only %.5f mm ; incl. dry windows %.5f mm ; dry-window-only RMSE %.5f mm (n dropped windows %d)" %
      (np.sqrt(sse / npx) * 1000, np.sqrt((sse + sse_d) / (npx + npx_d)) * 1000, np.sqrt(sse_d / npx_d) * 1000, npx_d // (5 * 128 * 128)))
# 4. Pakistan rain files vs depth intervals (info only)
secs, files, bad, diffs = depth_axis("Pakistan_480"); print("Pakistan depth intervals", (len(secs) - 1) * 300 // 1800, "rain files", len(rain_files("Pakistan")))
for reg, key in [("UK", "UK_30"), ("Australia", "Australia_30"), ("Mozambique", "Mozambique_480")]:
    secs, files, bad, diffs = depth_axis(key); print(reg, "depth intervals", (len(secs) - 1) * 300 // 1800, "rain files", len(rain_files(reg)))

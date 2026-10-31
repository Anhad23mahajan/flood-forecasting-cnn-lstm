import numpy as np
from vcommon import *
from scipy.stats import rankdata
key = "Mozambique_480"; secs, files, bad, diffs = depth_axis(key)
oy, ox = CROP[key]; b = 16; H, W = 151, 138; M = 32
dem = read_tif(DEM_PATH("Mozambique"), window=Window(ox - M, oy - M, W * b + 2 * M, H * b + 2 * M)).astype(np.float64)
print("dem window min/max", dem.min(), dem.max(), "nodata in window:", (dem < -1e30).sum())
dmax = np.zeros((H, W)); 
for i in range(0, len(secs), 24): dmax = np.maximum(dmax, read_tif(files[300 * i]).astype(np.float64))
d_last = read_tif(files[300 * (len(secs) - 1)]).astype(np.float64)
for name, tgt in [("time-max depth (every 2 h)", dmax), ("final depth", d_last)]:
    tg = np.log(tgt.ravel() + 1e-3)
    res = []
    for dy in range(-24, 25, 2):
        for dx in range(-24, 25, 2):
            sub = dem[M + dy:M + dy + H * b, M + dx:M + dx + W * b]
            blk = sub.reshape(H, b, W, b).mean(axis=(1, 3))
            res.append((np.corrcoef(blk.ravel(), tg)[0, 1], dy, dx))
    res.sort()   # most negative first (low terrain <-> deep water)
    print(name, "most negative corr(DEM, log depth) offsets:", [("%.4f" % a, dy, dx) for a, dy, dx in res[:5]])
    z = [t for t in res if t[1] == 0 and t[2] == 0][0]; print("   at claimed offset (0,0): %.4f rank %d of %d" % (z[0], res.index(z) + 1, len(res)))

import numpy as np
from vcommon import *
key = "Mozambique_480"; secs, files, bad, diffs = depth_axis(key); rfs = rain_files("Mozambique")
oy, ox = CROP[key]; b = 16; H, W = 151, 138; M = 32
NI = 4
# summed rain over first NI intervals, read with margin M around the claimed crop
acc = np.zeros((H * b + 2 * M, W * b + 2 * M))
for j in range(NI):
    acc += read_tif(rfs[j], window=Window(ox - M, oy - M, W * b + 2 * M, H * b + 2 * M)).astype(np.float64)
gain = (read_tif(files[1800 * NI]).astype(np.float64) - read_tif(files[0]).astype(np.float64)) * 1000   # mm over first NI intervals
best = []
for dy in range(-M, M + 1, 2):
    for dx in range(-M, M + 1, 2):
        sub = acc[M + dy:M + dy + H * b, M + dx:M + dx + W * b]
        blk = sub.reshape(H, b, W, b).mean(axis=(1, 3))
        best.append((np.corrcoef(blk.ravel(), gain.ravel())[0, 1], dy, dx))
best.sort(reverse=True)
print("corr(per-cell depth gain over first %d intervals, block-mean rain) best offsets (corr, dy, dx relative to claimed (150,50)):" % NI)
for t in best[:5]: print("  %.4f  dy=%+d dx=%+d" % t)
zero = [t for t in best if t[1] == 0 and t[2] == 0][0]; print("at claimed offset: %.4f" % zero[0], " rank", best.index(zero) + 1, "of", len(best))
print("profile dx (dy=0):", [(dx, round(c_, 4)) for c_, dy, dx in sorted([t for t in best if t[1] == 0], key=lambda t: t[2])][::2])
print("profile dy (dx=0):", [(dy, round(c_, 4)) for c_, dy, dx in sorted([t for t in best if t[2] == 0], key=lambda t: t[1])][::2])

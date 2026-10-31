"""Evidence figure for the recovered 480 m geometry: the high-passed flood footprint (max depth over the event) next to the
high-passed terrain resampled with the ADOPTED geometry, and next to the same terrain resampled with a deliberately WRONG offset.
If the geometry is right, the left and middle panels show the same local structure and the right one does not."""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, map_coordinates

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = "data/extracted/FloodCastBench"
GEOM = {"Mozambique": (150, 50), "Pakistan": (42, 178)}
REGIONS = sys.argv[1:] or ["Mozambique", "Pakistan"]
fig, axes = plt.subplots(len(REGIONS), 3, figsize=(15, 4.9 * len(REGIONS)), squeeze=False)
for row, region in enumerate(REGIONS):
    files = sorted(glob.glob(f"{ROOT}/Low-fidelity flood forecasting/480m/{region}/*.tif"), key=lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1)))
    mx = None
    for i in np.linspace(0, len(files) - 1, 24).astype(int):
        with rasterio.open(files[i]) as s:
            a = s.read(1).astype("float32")
        mx = a if mx is None else np.maximum(mx, a)
    H, W = mx.shape
    with rasterio.open(f"{ROOT}/Relevant data/DEM/{region}_DEM.tif") as s:
        dem = s.read(1).astype("float32")
    FH, FW = dem.shape
    dem[dem < -1e30] = np.nan
    Q = 1 if FH * FW < 8e6 else 4
    if Q > 1:
        h4, w4 = FH // Q * Q, FW // Q * Q
        dem = np.nanmean(dem[:h4, :w4].reshape(h4 // Q, Q, w4 // Q, Q), axis=(1, 3))
    dem_f = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    hp = lambda a: a - gaussian_filter(a, 6.0)

    def sample(oy, ox):
        n = 3
        yy = (oy + (np.arange(H * n) + 0.5) / n * 16.0) / Q - 0.5
        xx = (ox + (np.arange(W * n) + 0.5) / n * 16.0) / Q - 0.5
        Y, X = np.meshgrid(yy, xx, indexing="ij")
        return map_coordinates(dem_f, [Y, X], order=1, mode="nearest").reshape(H, n, W, n).mean(axis=(1, 3))

    z = lambda a: (a - a.mean()) / (a.std() + 1e-9)
    foot = z(hp(np.log1p(mx * 20)))
    oy, ox = GEOM[region]
    good = z(hp(sample(oy, ox)))
    wy, wx = oy + (56 if region == "Mozambique" else 0), ox + (56 if region == "Mozambique" else 0)
    wy = min(wy, FH - 16 * H); wx = min(wx, FW - 16 * W)
    if (wy, wx) == (oy, ox):
        wy, wx = max(0, oy - 24), max(0, ox - 96)
    bad = z(hp(sample(wy, wx)))
    r = lambda a, b: float((a * b).mean())
    vm = 2.5
    for col, (img, title) in enumerate(((foot, f"{region}: flood footprint (max depth, high-passed)"),
                                        (-good, f"terrain, ADOPTED geometry (offset {oy},{ox}); r = {r(foot, good):+.3f}"),
                                        (-bad, f"terrain, WRONG offset ({wy},{wx}); r = {r(foot, bad):+.3f}"))):
        ax = axes[row, col]
        ax.imshow(img, cmap="RdBu_r", vmin=-vm, vmax=vm)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
fig.suptitle("Recovered 480 m geometry (16 base pixels per cell + crop): local structure lines up only at the adopted offset\n"
             "(terrain sign flipped so low ground = red like flooded ground)", fontsize=11)
plt.tight_layout()
os.makedirs("verification_outputs/multires", exist_ok=True)
out = "verification_outputs/multires/alignment_evidence.png" if len(REGIONS) > 1 else f"verification_outputs/multires/alignment_evidence_{REGIONS[0]}.png"
plt.savefig(out, dpi=85)
print("saved", out)

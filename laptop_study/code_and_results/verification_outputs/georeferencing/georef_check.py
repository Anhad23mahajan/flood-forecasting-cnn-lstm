"""Run: python verification_outputs/georeferencing/georef_check.py (needs the local dataset at
data/extracted/FloodCastBench and internet for Esri tiles).

Does the flood-depth raster line up with the real world? Overlay the observed depth of the patch the
notebook maps (lat -28.9536..-28.9189, lon 153.3759..153.4154) on Esri satellite imagery, using exactly the
notebook's georeferencing (world file origin is a pixel CENTRE -> half-pixel shift to the corner)."""
import io, math, os, sys, urllib.request
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # project root
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from rasterio.warp import transform, transform_bounds
from rasterio.transform import Affine
from real_data_loader import RegionData

ROOT = "data/extracted/FloodCastBench"
rd = RegionData(root_dir=ROOT, region="Australia", fidelity_dir="High-fidelity flood forecasting/30m")

# notebook's georeferencing
with open(f"{ROOT}/Relevant data/Georeferenced files/Australia.tfw") as f:
    a, _d, _b, e, c, fy = (float(x) for x in f.read().split())
corner_x, corner_y = c - a / 2, fy - e / 2
EPSG = 32756

# find the patch whose bounds match the notebook's printed span (patches sit on a 128-pixel grid)
target = (-28.9536, -28.9189, 153.3759, 153.4154)
best = None
for row in range(0, rd.grid_height - 127, 128):
    for col in range(0, rd.grid_width - 127, 128):
        tr = Affine(a, 0, corner_x + col * a, 0, e, corner_y + row * e)
        w, s_, e_, n = (tr * (0, 128))[0], (tr * (0, 128))[1], (tr * (128, 0))[0], (tr * (128, 0))[1]
        lon_min, lat_min, lon_max, lat_max = transform_bounds(f"EPSG:{EPSG}", "EPSG:4326", w, s_, e_, n)
        err = abs(lat_min - target[0]) + abs(lat_max - target[1]) + abs(lon_min - target[2]) + abs(lon_max - target[3])
        if best is None or err < best[0]:
            best = (err, row, col, lon_min, lat_min, lon_max, lat_max)
err, row, col, lon_min, lat_min, lon_max, lat_max = best
print(f"matched patch row={row} col={col}; bounds lat {lat_min:.4f}..{lat_max:.4f} lon {lon_min:.4f}..{lon_max:.4f} (bounds error {err:.5f} deg)")

# observed depth for a mid-flood time
start = 1500
seq, _tgt, _sev, _mx = rd.load_patch_sequence(row, col, 128, start, 5, 5, 6)
depth = seq[-1, ..., 0]
print(f"observed depth: max {depth.max():.2f} m, wet (>0.1 m) fraction {(depth > 0.1).mean():.1%}")

# Esri tiles covering the patch
Z = 15
def tile_xy(lon, lat):
    n = 2 ** Z
    return (int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n))
def tile_nw(x, y):
    n = 2 ** Z
    return (x / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n)))))
x0, y0 = tile_xy(lon_min, lat_max); x1, y1 = tile_xy(lon_max, lat_min)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
canvas = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
for tx in range(x0, x1 + 1):
    for ty in range(y0, y1 + 1):
        u = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{ty}/{tx}"
        b = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": UA}), timeout=30).read()
        canvas.paste(Image.open(io.BytesIO(b)).convert("RGB"), ((tx - x0) * 256, (ty - y0) * 256))
w_lon, n_lat = tile_nw(x0, y0); e_lon, s_lat = tile_nw(x1 + 1, y1 + 1)
print(f"imagery: {x1 - x0 + 1}x{y1 - y0 + 1} tiles at zoom {Z}")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax, mode in zip(axes, ("imagery only", "observed flood depth on imagery")):
    ax.imshow(canvas, extent=[w_lon, e_lon, s_lat, n_lat])
    if mode != "imagery only":
        rgba = np.zeros((128, 128, 4), dtype="float32"); rgba[..., 2] = 1.0; rgba[..., 0] = 0.0
        rgba[..., 3] = np.where(depth > 0.05, np.clip(np.sqrt(depth / max(depth.max(), 1e-9)) * 1.2, 0.25, 1.0), 0.0)
        ax.imshow(rgba, extent=[lon_min, lon_max, lat_min, lat_max], interpolation="nearest")
        ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min], [lat_min, lat_min, lat_max, lat_max, lat_min], "r-", lw=1.5)
    ax.set_xlim(lon_min - 0.004, lon_max + 0.004); ax.set_ylim(lat_min - 0.004, lat_max + 0.004)
    ax.set_title(mode); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "georef_check.png")
plt.savefig(out, dpi=110)
print("saved", out)

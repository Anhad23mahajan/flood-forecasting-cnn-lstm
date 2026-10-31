"""Georeferencing of the COARSE (480 m) regions against a real basemap.
The 30 m check (verification_outputs/georeferencing/georef_check.py) only covered Australia.  For a 480 m depth grid the upper-left corner
of cell (r, c) sits at base-grid pixel (oy + 16 r, ox + 16 c) of the archive's world file (which describes the 30 m grid), i.e.
    x = corner_x + (ox + 16 c) * 30 m ,  y = corner_y - (oy + 16 r) * 30 m        (corner = world-file pixel centre -/+ half a pixel)
with the recovered crop offset (oy, ox) from real_data_loader.LAYER_GEOMETRY.  This overlays the flooded area (depth > 5 cm at the frame
of largest wet area, and the event's maximum depth) on OpenStreetMap tiles (Esri imagery returned HTTP 403 from this machine when this was run; the rivers and coast in OSM are what is compared).  Precision of this check: kilometres (the offset itself
was recovered to +-2 base pixels by the terrain probe); it verifies the coordinate frame (UTM zone, world-file origin, crop offset)
place the flood on the real floodplain.

Run: python completion/e_georef_coarse.py <Mozambique|Pakistan|UK|Australia> [res]
"""
import glob
import io
import math
import os
import sys
import urllib.request

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from PIL import Image
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

from real_data_loader import LAYER_GEOMETRY

ROOT = "data/extracted/FloodCastBench"
region = sys.argv[1]
RES = int(sys.argv[2]) if len(sys.argv) > 2 else 480
EPSG = {"Mozambique": 32736, "Pakistan": 32642, "UK": 32630, "Australia": 32756}[region]
Z = {("Mozambique", 480): 11, ("Pakistan", 480): 9, ("UK", 60): 12, ("UK", 30): 12, ("Australia", 60): 11}[(region, RES)]
FID = {30: "High-fidelity flood forecasting/30m", 60: "High-fidelity flood forecasting/60m", 480: "Low-fidelity flood forecasting/480m"}[RES]
scale, (oy, ox) = LAYER_GEOMETRY[(region, RES)]
scale = LAYER_GEOMETRY[(region, RES)][0]
with open(f"{ROOT}/Relevant data/Georeferenced files/{region}.tfw") as f:
    a, _d, _b, e, cx, fy = (float(v) for v in f.read().split())
corner_x, corner_y = cx - a / 2, fy - e / 2

files = sorted(glob.glob(f"{ROOT}/{FID}/{region}/*.tif"), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
mx, best, best_area = None, None, -1
for i in np.linspace(0, len(files) - 1, 60).astype(int):
    with rasterio.open(files[i]) as s:
        d = s.read(1)
    mx = d if mx is None else np.maximum(mx, d)
    if (d > 0.05).mean() > best_area:
        best_area, best = (d > 0.05).mean(), d
H, W = best.shape
tr = Affine(a * scale, 0, corner_x + ox * a, 0, e * scale, corner_y + oy * e)     # e < 0
w, s_, e_, n = (tr * (0, H))[0], (tr * (0, H))[1], (tr * (W, 0))[0], (tr * (W, 0))[1]
lon_min, lat_min, lon_max, lat_max = transform_bounds(f"EPSG:{EPSG}", "EPSG:4326", w, s_, e_, n)
print(f"{region}: depth grid {H}x{W}; footprint lon {lon_min:.3f}..{lon_max:.3f}, lat {lat_min:.3f}..{lat_max:.3f}; wet fraction of the chosen frame {best_area:.2f}")


def tile_xy(lon, lat):
    nn = 2 ** Z
    return (int((lon + 180) / 360 * nn), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * nn))


def tile_nw(x, y):
    nn = 2 ** Z
    return (x / nn * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / nn)))))


x0, y0 = tile_xy(lon_min - 0.05, lat_max + 0.05)
x1, y1 = tile_xy(lon_max + 0.05, lat_min - 0.05)
UA = "flood-forecast-georef-check/1.0 (student project; anhadagammahajan@gmail.com)"
canvas = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
for tx in range(x0, x1 + 1):
    for ty in range(y0, y1 + 1):
        u = f"https://tile.openstreetmap.org/{Z}/{tx}/{ty}.png"
        b = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": UA}), timeout=30).read()
        canvas.paste(Image.open(io.BytesIO(b)).convert("RGB"), ((tx - x0) * 256, (ty - y0) * 256))
w_lon, n_lat = tile_nw(x0, y0)
e_lon, s_lat = tile_nw(x1 + 1, y1 + 1)
print(f"imagery: {x1 - x0 + 1}x{y1 - y0 + 1} tiles at zoom {Z}")

fig, axes = plt.subplots(1, 2, figsize=(16, 8.5 if region == "Mozambique" else 7))
for ax, mode in zip(axes, ("imagery only", f"{region}: simulated flood (max depth over the event) on imagery")):
    ax.imshow(canvas, extent=[w_lon, e_lon, s_lat, n_lat])
    if mode != "imagery only":
        rgba = np.zeros((H, W, 4), dtype="float32")
        rgba[..., 2] = 1.0
        rgba[..., 3] = np.where(mx > 0.05, np.clip(np.sqrt(mx / max(mx.max(), 1e-9)) * 1.3, 0.3, 1.0), 0.0)
        ax.imshow(rgba, extent=[lon_min, lon_max, lat_min, lat_max], interpolation="nearest")
        ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min], [lat_min, lat_min, lat_max, lat_max, lat_min], "r-", lw=1.2)
    pad = {"Mozambique": 0.15, "Pakistan": 0.5}.get(region, 0.03)
    ax.set_xlim(lon_min - pad, lon_max + pad)
    ax.set_ylim(lat_min - pad, lat_max + pad)
    ax.set_title(mode)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
plt.tight_layout()
os.makedirs("completion/results", exist_ok=True)
out = f"completion/results/georef_{region}_{RES}m.png"
plt.savefig(out, dpi=90)
print("saved", out)

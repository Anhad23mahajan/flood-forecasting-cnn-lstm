"""Independent helpers (rasterio + numpy only). Deliberately does NOT import any project code."""
import os, re, glob, warnings
import numpy as np
import rasterio
from rasterio.windows import Window
warnings.filterwarnings("ignore")

ROOT = r"C:\Dev\flood-forecast-demo\data\extracted\FloodCastBench"
DEPTH_DIR = {
    "Australia_30": os.path.join(ROOT, "High-fidelity flood forecasting", "30m", "Australia"),
    "UK_30":        os.path.join(ROOT, "High-fidelity flood forecasting", "30m", "UK"),
    "Australia_60": os.path.join(ROOT, "High-fidelity flood forecasting", "60m", "Australia"),
    "UK_60":        os.path.join(ROOT, "High-fidelity flood forecasting", "60m", "UK"),
    "Mozambique_480": os.path.join(ROOT, "Low-fidelity flood forecasting", "480m", "Mozambique"),
    "Pakistan_480":   os.path.join(ROOT, "Low-fidelity flood forecasting", "480m", "Pakistan"),
}
RAIN_DIR = lambda region: os.path.join(ROOT, "Relevant data", "Rainfall", f"{region} flood")
DEM_PATH = lambda region: os.path.join(ROOT, "Relevant data", "DEM", f"{region}_DEM.tif")
# crop offsets of the depth grid inside the 30 m base grid, (row, col) in base pixels
CROP = {"UK_30": (5, 5), "UK_60": (5, 5), "Australia_30": (0, 0), "Australia_60": (0, 0),
        "Mozambique_480": (150, 50), "Pakistan_480": (42, 178)}
BLOCK = {"Australia_30": 1, "UK_30": 1, "UK_60": 2, "Australia_60": 2, "Mozambique_480": 16, "Pakistan_480": 16}

def depth_files(key):
    """Return dict {seconds:int -> path} from raw file names (integer seconds)."""
    d = DEPTH_DIR[key]
    out = {}
    bad = []
    for f in os.listdir(d):
        m = re.fullmatch(r"(\d+)\.tif", f)
        if m: out[int(m.group(1))] = os.path.join(d, f)
        else: bad.append(f)
    return out, bad

def depth_axis(key):
    """Sorted seconds list + checks on spacing; frame index i <-> file i in sorted-by-integer order."""
    files, bad = depth_files(key)
    secs = sorted(files)
    diffs = np.diff(secs)
    return secs, files, bad, diffs

def read_tif(path, window=None):
    with rasterio.open(path) as s:
        return s.read(1, window=window)

def rain_files(region):
    fs = sorted(glob.glob(os.path.join(RAIN_DIR(region), "*.tif")))
    return fs

def parse_rain_name(p):
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})-S(\d{2})(\d{2})(\d{2})\.tif", os.path.basename(p))
    import datetime
    y, mo, d, h, mi, s = map(int, m.groups())
    return datetime.datetime(y, mo, d, h, mi, s)

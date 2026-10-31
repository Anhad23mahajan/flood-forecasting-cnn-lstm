import sys, json, time
import numpy as np
from vcommon import *

def run(key, region):
    t0 = time.time()
    secs, files, bad, diffs = depth_axis(key)
    print(key, "n depth files", len(secs), "first", secs[0], "last", secs[-1],
          "unique diffs", np.unique(diffs), "non-int-named files", bad)
    n_int = (len(secs) - 1) * 300 // 1800     # number of 30-min intervals
    # domain-mean depth (mm) at t = 1800*j, j=0..n_int
    D = np.zeros(n_int + 1); nanc = 0; negc = 0; wet = np.zeros(n_int + 1)
    for j in range(n_int + 1):
        a = read_tif(files[1800 * j]).astype(np.float64)
        nanc += np.isnan(a).sum(); negc += (a < 0).sum()
        D[j] = np.nanmean(a) * 1000.0
        wet[j] = (a > 0.05).sum()
    print(" depth NaN count", nanc, "neg count", negc, "time", time.time() - t0)
    # rainfall
    rfs = rain_files(region)
    ts = [parse_rain_name(p) for p in rfs]
    dt = np.unique([(ts[i+1] - ts[i]).total_seconds() for i in range(len(ts)-1)])
    print(" n rain files", len(rfs), "rain dt unique", dt, ts[0], ts[-1])
    oy, ox = CROP[key]; b = BLOCK[key]
    with rasterio.open(depth_files(key)[0][0]) as s: H, W = s.shape
    R = np.zeros(len(rfs)); nodata_in_crop = 0; R_plain = np.zeros(len(rfs))
    for i, p in enumerate(rfs):
        w = Window(ox, oy, W * b, H * b)
        a = read_tif(p, window=w).astype(np.float64)
        nodata_in_crop += ((a < -1e30) | np.isnan(a)).sum()
        blk = a.reshape(H, b, W, b).mean(axis=(1, 3))       # block-mean onto depth grid
        R[i] = blk.mean()
        R_plain[i] = a.mean()
    print(" rain nodata/NaN in crop", nodata_in_crop, " max |block-mean vs plain mean|", np.abs(R - R_plain).max(),
          "time", time.time() - t0)
    np.savez(f"A_{key}.npz", D=D, R=R, wet=wet)
    return D, R

if __name__ == "__main__":
    key = sys.argv[1]; region = sys.argv[2]
    run(key, region)

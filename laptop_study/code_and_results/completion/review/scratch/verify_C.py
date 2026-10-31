import json
import numpy as np
from vcommon import *
CACHE = r"C:\Dev\flood-forecast-demo\completion\cache"
rng = np.random.default_rng(20260920)
out = {}
for pool, key in [("Pakistan_480_train", "Pakistan_480"), ("Australia_60_train", "Australia_60"), ("UK_60_test", "UK_60")]:
    m = np.load(f"{CACHE}\pool_{pool}_meta.npz", allow_pickle=True)
    meta = m["meta"]; size = int(m["size"]); ntl = int(m["n_tiles"])
    dh = np.load(f"{CACHE}\pool_{pool}_dh.npy", mmap_mode="r"); df = np.load(f"{CACHE}\pool_{pool}_df.npy", mmap_mode="r")
    secs, files, bad, diffs = depth_axis(key); n = len(secs)
    assert secs == [300 * i for i in range(n)]
    with rasterio.open(files[0]) as s: H, W = s.shape
    print(pool, "meta", meta.shape, "size", size, "n_tiles", ntl, "grid", (H, W), "n files", n, "dh", dh.shape, "df", df.shape)
    # sanity on ALL meta rows
    r, c, st = meta[:, 0], meta[:, 1], meta[:, 2]
    print("  meta ranges: rows", r.min(), r.max(), "cols", c.min(), c.max(), "start", st.min(), st.max(), "max start+54 =", (st + 54).max(), "<= n-1 =", n - 1)
    th, tw_ = dh.shape[2], dh.shape[3]
    assert (r + th <= H).all() and (c + tw_ <= W).all(), "tile beyond grid"
    idx = rng.choice(len(meta), 10, replace=False)
    ok_all = True; nan_any = False; disc_all = True
    cache = {}
    def fr(i):
        if i not in cache: cache[i] = read_tif(files[300 * i])
        return cache[i]
    for i in idx:
        rr, cc, s0 = map(int, meta[i])
        for k in range(5):
            a = fr(s0 + 6 * k)[rr:rr + th, cc:cc + tw_]; b = fr(s0 + 6 * (5 + k))[rr:rr + th, cc:cc + tw_]
            e1 = np.array_equal(np.asarray(dh[i, k]), a); e2 = np.array_equal(np.asarray(df[i, k]), b)
            nan_any |= bool(np.isnan(np.asarray(dh[i, k])).any() or np.isnan(np.asarray(df[i, k])).any())
            ok_all &= e1 and e2
            # discriminating check: must differ from frame +-1 unless constant field
            for d_ in (-1, 1):
                a2 = fr(s0 + 6 * k + d_)[rr:rr + th, cc:cc + tw_]
                if np.array_equal(np.asarray(dh[i, k]), a2) and not np.array_equal(a, a2): disc_all = False
        if not (e1 and e2): print("  MISMATCH window", i, meta[i])
    print("  10 windows idx", sorted(idx.tolist()), "-> all 100 (dh,df) frame patches exactly equal:", ok_all, " NaN in stored:", nan_any)
    # extra: how many windows have all-zero dh (dry)?
    out[pool] = dict(idx=sorted(idx.tolist()), exact=bool(ok_all), meta_start_max=int(st.max()), n_files=n, size=size, n_windows=int(len(meta)))
json.dump(out, open("C_result.json", "w"), indent=1)

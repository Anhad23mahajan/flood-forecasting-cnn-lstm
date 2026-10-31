import json
import numpy as np
from vcommon import *
secs, files, bad, diffs = depth_axis("Mozambique_480"); n = len(secs)
assert secs == [300 * i for i in range(n)]
F = np.stack([read_tif(files[300 * i]) for i in range(n)])          # (1729,151,138) float32
print("frames", F.shape, F.dtype, "NaN", np.isnan(F).sum(), "min", F.min(), "max", F.max())
S = n // 20; print("n samples", S, "leftover frames", n - 20 * S)
np.save("D_frames_moz.npy", F)

def metrics(sample_ids, gt=True):
    """pooled over all selected samples, all 19 steps, all pixels"""
    sse = 0.0; sy2 = 0.0; sy = 0.0; N = 0; sp = 0.0; syp = 0.0; sp2 = 0.0
    cm = {0.001: [0, 0, 0], 0.01: [0, 0, 0]}; cm_gt = {0.001: [0, 0, 0], 0.01: [0, 0, 0]}
    for k in sample_ids:
        p = F[20 * k].astype(np.float64); Y = F[20 * k + 1:20 * k + 20].astype(np.float64)   # 19 targets
        d = Y - p[None]
        sse += (d ** 2).sum(); sy2 += (Y ** 2).sum(); sy += Y.sum(); N += Y.size
        P = np.broadcast_to(p[None], Y.shape)
        sp += P.sum(); sp2 += (P ** 2).sum(); syp += (Y * P).sum()
        for t in cm:
            for store, cmp in ((cm, lambda a, t: a >= t), (cm_gt, lambda a, t: a > t)):
                yy = cmp(Y, t); pp = cmp(P, t)
                store[t][0] += int((yy & pp).sum()); store[t][1] += int((yy & ~pp).sum()); store[t][2] += int((~yy & pp).sum())
    ymean = sy / N
    sst = sy2 - N * ymean ** 2
    nse = 1 - sse / sst
    pmean = sp / N
    cov = syp / N - ymean * pmean; vy = sy2 / N - ymean ** 2; vp = sp2 / N - pmean ** 2
    r = cov / np.sqrt(vy * vp)
    csi = lambda c: c[0] / (c[0] + c[1] + c[2])
    return dict(rel_sq_err=sse / sy2, rel_l2_err=np.sqrt(sse / sy2), NSE=nse, r=r,
                CSI_1mm_ge=csi(cm[0.001]), CSI_1cm_ge=csi(cm[0.01]), CSI_1mm_gt=csi(cm_gt[0.001]), CSI_1cm_gt=csi(cm_gt[0.01]), n=N)

CL = json.load(open(r"C:\Dev\flood-forecast-demo\completion\results\paper_protocol_persistence_Mozambique_480.json"))
res = {}
res["whole event"] = metrics(range(S))
# candidates for the sub-windows
frames_per_day = 288
for name, ids in [("final 20% (last ceil(.2*S) samples)", range(S - int(np.ceil(0.2 * S)), S)),
                  ("final 20% (last floor(.2*S) samples)", range(S - int(np.floor(0.2 * S)), S)),
                  ("final 20% (samples with 20k >= 0.8*n)", [k for k in range(S) if 20 * k >= 0.8 * n]),
                  ("final 2 days (samples with 20k >= n-576)", [k for k in range(S) if 20 * k >= n - 2 * frames_per_day]),
                  ("final 2 days (last 28)", range(S - 28, S)), ("final 2 days (last 29)", range(S - 29, S)), ("final 2 days (last 30)", range(S - 30, S))]:
    res[name] = metrics(list(ids)); res[name]["n_samples"] = len(list(ids))
for k, v in res.items():
    print(k, {a: (round(b, 8) if isinstance(b, float) else b) for a, b in v.items()})
print("CLAIMED whole:", CL["whole event"])
print("CLAIMED final20:", CL["final 20% of the event"])
print("CLAIMED final2d:", CL["final 2 days"])
json.dump(res, open("D_result.json", "w"), indent=1, default=float)

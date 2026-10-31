"""What does a trivial baseline score under the FloodCastBench paper's OWN protocol and metrics?

Paper protocol (Xu et al. 2025, Sci Data 12:431, Technical Validation; read from the open-access full text via Europe PMC PMC11903882):
  * a sample = 20 consecutive depth frames at 300 s spacing (t = 1..20, 95 minutes); the model gets the depth at t=1 (and, for FNO+, the terrain and
    the rainfall series t=1..20) and predicts the depth at t=2..20 (19 steps), over the WHOLE domain;
  * metrics over all pixels: RMSE (the paper's name for a RELATIVE error, sum|y-p|^2 / |y|^2 - its exact normalisation is not fully specified,
    both the squared and the square-rooted forms are reported here), NSE = 1 - sum(y-p)^2 / sum(y-ybar)^2, Pearson r,
    CSI = TP / (TP + FP + FN) at wet thresholds 0.001 m and 0.01 m;
  * splits (Low-fidelity Pakistan): 145 training samples (first 10 days), 28 validation (days 11-12), 28 test (final 2 days); (High-fidelity Australia):
    116 training (first 8 days), 14 validation (day 9), 14 test (day 10).  The transfer tests are trained on Pakistan / Australia and evaluated on
    Mozambique / UK (which samples of them is not stated; both the WHOLE event and its final 20% are scored here).
  * baselines in the paper: U-Net, FNO, FNO+.  No persistence baseline is reported.
PERSISTENCE here = predict the depth of frame t=1 for every t=2..20 (needs no training, no rainfall).  The paper's Table 5 numbers (quoted for reference):
    Pakistan -> Mozambique (480 m):  U-Net RMSE 1.602850 NSE -72.350318 r 0.903450 CSI(1mm) 0.724442 CSI(1cm) 0.667776
                                     FNO   0.163300 0.453167 0.980383 0.892473 0.915509 ;  FNO+ 0.078633 0.955450 0.985521 0.934028 0.912712
    Australia -> UK (60 m):          U-Net 0.525467 0.690026 0.956584 0.314035 0.517440 ; FNO 0.028990 0.998575 0.999600 0.755766 0.925626 ;
                                     FNO+ 0.024771 0.998949 0.999645 0.859821 0.963278
    30 m zero-shot (UK):             U-Net 0.556424 0.660375 0.960544 0.282739 0.553059 ; FNO 0.028297 0.998706 0.999604 0.722694 0.927164 ;
                                     FNO+ 0.025190 0.998953 0.999639 0.810674 0.958525

Run: python completion/f_paper_protocol.py <Region> <res>
"""
import glob
import json
import os
import sys

import numpy as np
import rasterio

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = "data/extracted/FloodCastBench"
FID = {30: "High-fidelity flood forecasting/30m", 60: "High-fidelity flood forecasting/60m", 480: "Low-fidelity flood forecasting/480m"}
region, res = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(f"{ROOT}/{FID[res]}/{region}/*.tif"), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
n = len(files)
n_samples = (n - 1) // 19            # consecutive samples that share their end/start frame: frames [19k .. 19k+19] ... use 20-frame windows, stride 20
starts = list(range(0, n - 19, 20))  # non-overlapping 20-frame samples
print(f"{region} {res} m: {n} frames -> {len(starts)} samples of 20 frames", flush=True)


def new_acc():
    return dict(n=0, sse=0.0, sy2=0.0, sy=0.0, sp=0.0, sp2=0.0, syp=0.0, tp=[0, 0], fp=[0, 0], fn=[0, 0])


def update(acc, y, p):
    y = y.astype(np.float64).ravel(); p = p.astype(np.float64).ravel()
    acc["n"] += y.size; acc["sse"] += float(((y - p) ** 2).sum()); acc["sy2"] += float((y ** 2).sum()); acc["sy"] += float(y.sum())
    acc["sp"] += float(p.sum()); acc["sp2"] += float((p ** 2).sum()); acc["syp"] += float((y * p).sum())
    for k, thr in enumerate((0.001, 0.01)):
        pw, yw = p > thr, y > thr
        acc["tp"][k] += int((pw & yw).sum()); acc["fp"][k] += int((pw & ~yw).sum()); acc["fn"][k] += int((~pw & yw).sum())


def finish(acc):
    n = acc["n"]
    ybar = acc["sy"] / n
    var = acc["sy2"] - n * ybar ** 2
    cov = acc["syp"] - acc["sy"] * acc["sp"] / n
    vp = acc["sp2"] - acc["sp"] ** 2 / n
    return dict(rel_sq_err=acc["sse"] / acc["sy2"], rel_l2_err=float(np.sqrt(acc["sse"] / acc["sy2"])), NSE=1 - acc["sse"] / var,
                r=cov / np.sqrt(max(var, 1e-30) * max(vp, 1e-30)),
                CSI_1mm=acc["tp"][0] / max(1, acc["tp"][0] + acc["fp"][0] + acc["fn"][0]), CSI_1cm=acc["tp"][1] / max(1, acc["tp"][1] + acc["fp"][1] + acc["fn"][1]))


sets = {"whole event": lambda k: True, "final 20% of the event": lambda k: k >= 0.8 * len(starts), "final 2 days" : lambda k: starts[k] * 300 >= (n - 1) * 300 - 2 * 86400}
accs = {name: new_acc() for name in sets}
for k, s in enumerate(starts):
    with rasterio.open(files[s]) as f:
        p0 = f.read(1)
    for j in range(1, 20):
        with rasterio.open(files[s + j]) as f:
            y = f.read(1)
        for name, sel in sets.items():
            if sel(k):
                update(accs[name], y, p0)
    if k % 20 == 0:
        print(f"  sample {k}/{len(starts)}", flush=True)
out = {name: finish(a) for name, a in accs.items()}
out["n_samples"] = len(starts)
for name, m in out.items():
    if isinstance(m, dict):
        print(f"{region} {res} m persistence [{name}]: rel-L2 {m['rel_l2_err']:.4f} (squared {m['rel_sq_err']:.5f})  NSE {m['NSE']:.6f}  r {m['r']:.6f}  CSI(1mm) {m['CSI_1mm']:.4f}  CSI(1cm) {m['CSI_1cm']:.4f}")
os.makedirs("completion/results", exist_ok=True)
json.dump(out, open(f"completion/results/paper_protocol_persistence_{region}_{res}.json", "w"), indent=1, default=float)

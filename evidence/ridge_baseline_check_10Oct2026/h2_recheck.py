"""Re-checks hypothesis H2 ('local CNN and small ConvLSTM beat the best linear baseline S3c on the never-seen flood') with properly
standardised ridge baselines. Read-only with respect to the project folder; output goes to the scratchpad."""
import glob
import json
import os
import sys

import numpy as np

import ridge_check as R  # sets up the stand-ins and imports engine / b_baselines

E, B, PROJ = R.E, R.B, R.PROJ
HERE = os.path.dirname(os.path.abspath(__file__))
EXPS = ["E60", "E30", "E480"]
REGS = ["causal", "hybrid", "forced"]

# ---- neural-network cell means from the 310 run files (same filter as the report's tables)
nets = {}
for f in glob.glob(os.path.join(PROJ, "completion", "results", "runs", "*.json")):
    d = json.load(open(f))
    if not (d.get("variant") == "full" and d.get("epochs") == 30 and d.get("batch") in (None, 1) and (d.get("lr") is None or abs(d["lr"] - 1e-3) < 1e-12)
            and not d.get("blocked") and not d.get("augment") and d.get("n_train") == 300 and d["seed"] in (1003, 1013, 1023) and d["arch"] in ("local", "clstm") and d.get("test")):
        continue
    nets.setdefault((d["exp"], d["arch"], d["regime"]), {})[d["fold"]] = d["test"]["skill"]
net_mean = {k: float(np.mean([v[i] for i in sorted(v)])) for k, v in nets.items() if len(v) == 3}

rows = []
for exp in EXPS:
    for regime in REGS:
        acc = {m: {"S1": [], "S3": [], "S3c": []} for m in ("orig", "std")}
        ship = {"S1": [], "S3c": []}
        for fold in range(3):
            tr_pool, te_pool, tr_idx, va_idx, te_idx, held, seed = E.split_for(exp, fold, 3, n_train=300, log=lambda s: None)
            jf = os.path.join(PROJ, "completion", "results", "baselines_%s_%s_f%dof3_n300.json" % (exp, regime, fold))
            if os.path.exists(jf):
                j = json.load(open(jf))
                ship["S1"].append(j["S1"]["test"]); ship["S3c"].append(j["S3c"]["test"])
            for m in ("orig", "std"):
                coefs = R.make_fit(m)(tr_pool, tr_idx, regime, ["S1", "S3"])
                for k, clip, key in (("S1", False, "S1"), ("S3", False, "S3"), ("S3", True, "S3c")):
                    pr = B.predictor(coefs[k], regime, k, clip)
                    acc[m][key].append(E.summarize(E.evaluate(pr, te_pool, te_idx, regime))["skill"])
        row = dict(exp=exp, regime=regime,
                   ship_S1=float(np.mean(ship["S1"])) if len(ship["S1"]) == 3 else None, ship_S3c=float(np.mean(ship["S3c"])) if len(ship["S3c"]) == 3 else None,
                   orig_S1=float(np.mean(acc["orig"]["S1"])), orig_S3c=float(np.mean(acc["orig"]["S3c"])),
                   std_S1=float(np.mean(acc["std"]["S1"])), std_S3=float(np.mean(acc["std"]["S3"])), std_S3c=float(np.mean(acc["std"]["S3c"])),
                   local=net_mean.get((exp, "local", regime)), clstm=net_mean.get((exp, "clstm", regime)))
        rows.append(row)
        print(json.dumps(row), flush=True)
        json.dump(rows, open(os.path.join(HERE, "h2_recheck_out.json"), "w"), indent=1)

print("\nexp   regime  | shipped S3c  rerun S3c  STD S3c | local   clstm | local-STD  clstm-STD")
p_old = p_new = tot = 0
for r in rows:
    for arch in ("local", "clstm"):
        if r[arch] is None:
            continue
        tot += 1
        p_old += int(r[arch] > (r["ship_S3c"] if r["ship_S3c"] is not None else r["orig_S3c"]))
        p_new += int(r[arch] > r["std_S3c"])
    print("%-5s %-7s | %10s %10.3f %8.3f | %6s %6s | %+9.3f %+9.3f" % (
        r["exp"], r["regime"], "%.3f" % r["ship_S3c"] if r["ship_S3c"] is not None else "n/a", r["orig_S3c"], r["std_S3c"],
        "%.3f" % r["local"] if r["local"] is not None else "-", "%.3f" % r["clstm"] if r["clstm"] is not None else "-",
        (r["local"] - r["std_S3c"]) if r["local"] is not None else float("nan"), (r["clstm"] - r["std_S3c"]) if r["clstm"] is not None else float("nan")))
print("H2 cells passing with the shipped baseline: %d of %d | with standardised baseline: %d of %d" % (p_old, tot, p_new, tot))

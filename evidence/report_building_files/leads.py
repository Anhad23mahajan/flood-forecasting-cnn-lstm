"""Mean skill by lead time (30 min ... 2.5 h) on the never-seen region, frozen recipe, hybrid input, mean of the 3 folds."""
import glob
import json
import numpy as np

R = r"C:\Dev\flood-forecast-demo\completion\results\runs"
out = {}
for f in sorted(glob.glob(R + r"\*.json")):
    d = json.load(open(f))
    batch_ok = d.get("batch") in (None, 1)
    lr_ok = d.get("lr") is None or abs(d["lr"] - 0.001) < 1e-12
    if not (d.get("regime") == "hybrid" and d.get("variant") == "full" and d.get("epochs") == 30 and batch_ok and lr_ok
            and not d.get("blocked") and not d.get("augment") and d.get("n_train") == 300):
        continue
    if d["seed"] not in (1003, 1013, 1023):
        continue
    key = (d["exp"], d["arch"])
    out.setdefault(key, {})[d["fold"]] = d["test"]["lead_skill"], d["test"]["skill"], d.get("code_md5")
for key in sorted(out):
    folds = out[key]
    leads = np.mean([folds[k][0] for k in sorted(folds)], axis=0)
    skill = np.mean([folds[k][1] for k in sorted(folds)])
    print(key, "folds", sorted(folds), "mean skill %.3f" % skill, "lead skill", np.round(leads, 3).tolist())

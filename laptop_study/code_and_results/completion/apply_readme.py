"""Insert / refresh the round-2 block at the top of HANDOFF_README.md (idempotent; markers <!-- ROUND2 START/END -->).
The '@@ROUND2_RESULTS@@' placeholder is replaced by a small table generated from completion/results/report.json."""
import json
import os
import re

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
block = open("completion/README_round2.md", encoding="utf-8").read()
rep = json.load(open("completion/results/report.json"))
h, b = rep["headline"], rep["baselines"]
rows = ["**Skill on the never-seen region (0 = copying the last frame), hybrid regime, mean over 3 tile folds** (full tables with intervals, other regimes and ablations: `completion/results/REPORT.md`):", "",
        "| model | Australia -> UK, 60 m | Australia -> UK, 30 m | Pakistan -> Mozambique, 480 m |", "|---|---|---|---|"]


def cell(e, key, kind):
    if kind == "b":
        v = b.get(f"{e}|hybrid|{key}")
        return f"{v['test']:+.3f}" if v else "-"
    v = h.get(f"{e}|{key}|hybrid")
    return f"{v['test_mean']:+.3f} [{v['test_ci'][0]:+.3f}, {v['test_ci'][1]:+.3f}]" if v else "-"


for label, key, kind in (("linear extrapolation of the depth history (S1)", "S1", "b"), ("linear with rain + state + trend (S3, clipped)", "S3c", "b"),
                         ("two-number lookup table (S5)", "S5", "b"), ("per-pixel MLP (S4, clipped)", "S4c", "b"), ("original CNN-LSTM (v1), same folds and recipe", "v1", "n"),
                         ("local CNN, kernel 1 (per pixel)", "local1", "n"), ("local CNN, 5x5 (4,101 parameters)", "local", "n"), ("**small ConvLSTM = the final CNN-LSTM (4,741 parameters)**", "clstm", "n")):
    r = [cell(e, key, kind) for e in ("E60", "E30", "E480")]
    if any(x != "-" for x in r):
        rows.append(f"| {label} | " + " | ".join(r) + " |")
block = block.replace("@@ROUND2_RESULTS@@", "\n".join(rows))
p = "HANDOFF_README.md"
s = open(p, encoding="utf-8").read()
wrapped = "<!-- ROUND2 START -->\n" + block + "<!-- ROUND2 END -->\n\n"
if "<!-- ROUND2 START -->" in s:
    s = re.sub(r"<!-- ROUND2 START -->.*?<!-- ROUND2 END -->\n\n", lambda m: wrapped, s, flags=re.S)
else:
    anchor = "**Read `PROJECT_LOG.md` first**"
    assert s.count(anchor) == 1
    s = s.replace(anchor, wrapped + anchor)
    old = "## Current state\n"
    assert s.count(old) == 1
    s = s.replace(old, "## Current state of the ORIGINAL model (v1) - kept as the record; see the round-2 block at the top for the corrected results\n")
open(p, "w", encoding="utf-8").write(s)
print("README updated")

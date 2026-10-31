"""Executes every code cell of flood_forecasting_colab.ipynb locally against the real
extracted dataset, with tiny sizes, so logic errors surface here instead of in a paid
Colab session. Skips only what can't run locally: pip installs, %%writefile (the modules
already exist here) and the Hugging Face download (points DATA_ROOT at the local copy)."""
import json
import os
import re
import sys
import time
import traceback

os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, ".")

NOTEBOOK = os.environ.get("DRYRUN_NOTEBOOK", "flood_forecasting_colab.ipynb")
MULTIRES = "multires" in NOTEBOOK   # the multi-resolution notebook has its own (tiny) defaults and config keys
STUDY = "study" in NOTEBOOK         # the corrected-study notebook: list-valued config, subprocess-driven
if STUDY:
    OVERRIDES = {"EPOCHS": 1, "N_TRAIN": 8, "THREADS": 2}
    RAW_OVERRIDES = {"EXPERIMENTS": '["E60"]', "REGIMES": '["hybrid", "forced"]', "MODELS": '["local", "clstm"]', "FOLDS": '[0]'}
elif MULTIRES:
    OVERRIDES = {"EPOCHS": 2, "N_TRAIN": 8, "N_VAL": 4, "N_STARTS": 8}
else:
    OVERRIDES = {
        "PER_CLASS_TRAIN": 3, "PER_CLASS_VAL": 2, "EPOCHS": 2, "N_START_INDICES": 12,
        "N_UNIFORM_AU": 6, "UK_START_STEP": 300, "CENSUS_START_STEP": 400,
    }
if len(sys.argv) > 1:  # optional JSON dict of overrides, for a longer local run
    OVERRIDES.update(json.loads(sys.argv[1]))
LOCAL_DATA_ROOT = "data/extracted/FloodCastBench"

with open(NOTEBOOK, encoding="utf-8") as f:
    nb = json.load(f)

ns = {"__name__": "__main__"}
ran = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    first = src.lstrip().splitlines()[0] if src.strip() else ""
    if first.startswith("%%writefile"):
        continue
    if "snapshot_download(" in src:
        src = f'DATA_ROOT = "{LOCAL_DATA_ROOT}"\nprint("local DATA_ROOT:", DATA_ROOT)'
    if STUDY and "# CONFIG (overridable)" in src:
        for k, v in OVERRIDES.items():
            src, n = re.subn(rf"^{k}\s*=\s*(\d+|True|False)", f"{k} = {v}", src, flags=re.M)
            assert n == 1, f"could not override {k}"
        for k, v in RAW_OVERRIDES.items():
            src, n = re.subn(rf"^{k}\s*=\s*\[.*?\]", f"{k} = {v}", src, flags=re.M)
            assert n == 1, f"could not override {k}"
    elif (not MULTIRES and not STUDY and "PER_CLASS_TRAIN" in src and "PATCH_SIZE = 128" in src) or (MULTIRES and "# CONFIG (overridable)" in src):
        for k, v in OVERRIDES.items():
            src, n = re.subn(rf"^{k}\s*=\s*(\d+|True|False)", f"{k} = {v}", src, flags=re.M)
            assert n == 1, f"could not override {k}"
    src = "\n".join(l for l in src.splitlines() if not l.strip().startswith("!"))
    label = (src.strip().splitlines() or ["<empty>"])[0][:70]
    print(f"\n--- cell {i}: {label}")
    t0 = time.perf_counter()
    try:
        exec(compile(src, f"<notebook cell {i}>", "exec"), ns)
    except Exception:
        traceback.print_exc()
        print(f"\nFAILED in cell {i} after {ran} cells")
        sys.exit(1)
    ran += 1
    print(f"    ok ({time.perf_counter() - t0:.1f}s)")
    if "folium.Map(" in src and os.environ.get("DRYRUN_MAP_HTML"):  # let a test open the interactive map
        ns["m"].save(os.environ["DRYRUN_MAP_HTML"])
        print("    saved interactive map to", os.environ["DRYRUN_MAP_HTML"])

print(f"\nALL {ran} EXECUTABLE CELLS RAN WITHOUT ERROR")

if os.environ.get("DRYRUN_EXTRA"):  # optional extra analysis, exec()d in the notebook's namespace
    with open(os.environ["DRYRUN_EXTRA"], encoding="utf-8") as f:
        exec(compile(f.read(), os.environ["DRYRUN_EXTRA"], "exec"), ns)

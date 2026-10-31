"""Build flood_forecasting_handoff.zip (the package for Aman): code, notebooks, documents, all result files, logs and review reports.
EXCLUDED on purpose: the raw dataset (data/, 46 GB), the window/rain caches (completion/cache, 13 GB; rebuilt by `python completion/engine.py cache <EXP>`), saved weights, the virtual environment,
__pycache__, the archive of earlier work (archive_earlier_work, 90 MB, not part of the study) and older zips.
Usage: python completion/make_handoff_zip.py   (writes ../flood_forecasting_handoff.zip next to PROJECT_LOG.md and prints what it contains)"""
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "flood_forecasting_handoff.zip")
TMP = OUT + ".tmp"
EXCLUDE_DIRS = {"venv", "data", "__pycache__", "cache", "weights", "archive_earlier_work", ".git", ".ipynb_checkpoints"}
EXCLUDE_TOP_FILES = {"flood_forecasting_handoff.zip", "study_results.zip"}
EXCLUDE_EXT = {".tif", ".tiff", ".h5", ".keras", ".pyc"}
INCLUDE_TOP_DIRS = {"completion", "verification_outputs", "archive_pre_redesign"}

n = size = 0
with zipfile.ZipFile(TMP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for name in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, name)
        if os.path.isfile(p):
            if name in EXCLUDE_TOP_FILES or os.path.splitext(name)[1].lower() in EXCLUDE_EXT or name.endswith(".tmp"):
                continue
            z.write(p, name); n += 1; size += os.path.getsize(p)
        elif name in INCLUDE_TOP_DIRS:
            for dp, dns, fns in os.walk(p):
                dns[:] = [d for d in dns if d not in EXCLUDE_DIRS]
                for fn in fns:
                    if os.path.splitext(fn)[1].lower() in EXCLUDE_EXT or fn.endswith(".tmp"):
                        continue
                    fp = os.path.join(dp, fn)
                    z.write(fp, os.path.relpath(fp, ROOT).replace(os.sep, "/")); n += 1; size += os.path.getsize(fp)
os.replace(TMP, OUT)
print(f"wrote {OUT}: {n} files, {size / 1e6:.1f} MB uncompressed, {os.path.getsize(OUT) / 1e6:.1f} MB zipped")

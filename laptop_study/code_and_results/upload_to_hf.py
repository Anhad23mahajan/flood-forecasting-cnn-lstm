from huggingface_hub import HfApi

REPO_ID = "mahajananhad/floodcastbench"
LOCAL_DIR = r"C:\Dev\flood-forecast-demo\data\extracted\FloodCastBench"

api = HfApi()

api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=False, exist_ok=True)
print(f"Repo ready: https://huggingface.co/datasets/{REPO_ID}")

README = """---
license: cc-by-4.0
---

# FloodCastBench (mirror)

Re-hosted for use in a CNN-LSTM spatiotemporal flood forecasting course project, to allow
selective per-region download from Google Colab (`snapshot_download(..., allow_patterns=...)`)
instead of a single non-resumable 21.6GB Zenodo archive.

Original dataset: Xu, Q., Shi, Y., Zhao, J. et al. "FloodCastBench: A Large-Scale Dataset and
Foundation Models for Flood Modeling and Forecasting." Scientific Data 12, 431 (2025).
https://doi.org/10.1038/s41597-025-04725-2

Original source: https://zenodo.org/records/14017092 (CC BY 4.0)

All credit for the dataset itself belongs to the original authors. This is an unmodified
re-upload of the same files for data-access convenience only.

## Structure

- `High-fidelity flood forecasting/<30m|60m>/<Australia|UK>/` - per-timestep depth TIFFs
- `Low-fidelity flood forecasting/480m/<Pakistan|Mozambique>/` - per-timestep depth TIFFs
- `Relevant data/` - DEM, land use, rainfall, georeferenced world files, initial conditions

Filenames are unpadded integers (simulation seconds) - sort numerically, not alphabetically.
"""

api.upload_file(
    path_or_fileobj=README.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("README uploaded.")

print(f"Starting upload of {LOCAL_DIR} ...")
api.upload_large_folder(
    repo_id=REPO_ID,
    repo_type="dataset",
    folder_path=LOCAL_DIR,
)
print("UPLOAD COMPLETE.")

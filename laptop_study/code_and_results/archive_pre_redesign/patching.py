"""
Patch extraction: the FloodCastBench team trained on whole scenes (batch size 1
on a 48GB A40 GPU). An 8GB laptop GPU needs much smaller inputs, so we crop
fixed-size patches from the full grid instead - this is dataset-structure-agnostic
(works on any stack of same-sized 2D arrays), verified here with synthetic data
before wiring in the real TIFF files.
"""

import numpy as np

PATCH_SIZE = 128


def extract_patches(grid_stack: np.ndarray, patch_size: int = PATCH_SIZE, stride: int | None = None):
    """grid_stack: (T, H, W) or (T, H, W, C). Returns a list of (row, col, patch) for every
    patch position that fits fully inside the grid (no padding - partial edge patches dropped,
    since padded/zero regions would corrupt depth/rainfall physical values)."""
    if stride is None:
        stride = patch_size  # non-overlapping by default

    if grid_stack.ndim == 3:
        T, H, W = grid_stack.shape
    else:
        T, H, W, C = grid_stack.shape

    if H < patch_size or W < patch_size:
        raise ValueError(f"Grid ({H}x{W}) is smaller than patch_size ({patch_size}) - cannot extract any patch.")

    patches = []
    for row in range(0, H - patch_size + 1, stride):
        for col in range(0, W - patch_size + 1, stride):
            patch = grid_stack[:, row:row + patch_size, col:col + patch_size, ...]
            patches.append((row, col, patch))
    return patches


def stack_channels(depth: np.ndarray, dem: np.ndarray, rainfall: np.ndarray) -> np.ndarray:
    """depth, rainfall: (T, H, W). dem: (H, W), static across time, broadcast to (T, H, W).
    Returns (T, H, W, 3) matching model.py's N_CHANNELS=3 input format."""
    T = depth.shape[0]
    dem_seq = np.broadcast_to(dem, (T,) + dem.shape)
    return np.stack([depth, dem_seq, rainfall], axis=-1)

"""
Real FloodCastBench data loader for one region (Stage 1 of the build plan).

Loads a spatial patch + temporal window of depth, DEM, and rainfall for a
single region, aligns rainfall's separate 30-minute calendar-date series to
depth's 5-minute simulation-second series, and derives a severity label from
future depth using the fixed thresholds decided in the project plan.

Verified against the real downloaded files:
  - depth: <region>/<t>.tif, t = unpadded seconds since sim start, step 300s
  - rainfall: Relevant data/Rainfall/<region> flood/<YYYYMMDD>-S<HHMMSS>.tif, step 1800s
  - Australia: depth, rainfall, DEM and land use all share the same 1073x1073 grid.
  - UK: depth is 170x275 but the other layers are 180x285 - a measured (5, 5) pixel
    offset is applied (see LAYER_GEOMETRY).
  - 60 m and 480 m depth rasters (Australia/UK at 60 m; Pakistan/Mozambique at 480 m) are the 30 m
    simulation resampled to 2x2 / 16x16 base-pixel cells and cropped, while DEM / land use / rainfall
    stay on the 30 m base grid. LAYER_GEOMETRY records each combination's scale and crop offset
    (recovered from the data and cross-checked, see its comment); DEM / rainfall are block-averaged
    to the depth cells. A combination with no registered geometry is refused, never guessed.
    (An earlier version of this loader claimed all layers matched - checked only on Australia - and
    later refused Pakistan/Mozambique as "not alignable" - a failure of the first search, not a
    property of the data.)

CONFIRMED PHYSICALLY (completion/a_rain_lag.py, round 2): rainfall file j is the rain that FALLS DURING [1800 j, 1800 (j+1)) s of the simulation; the values are mm/h
intensities (depth gain over an interval = 0.494-0.502 x the file value; correlation 1.00 at lag 0 in all four regions). The archive itself gives no clock time for
t = 0 (no TIFF tag or readme ties it to a calendar date); none is used. Original reasoning: rainfall's first file corresponds to depth's t=0. Evidence for this: for all four events the rainfall series starts on the
paper's stated event start date, and its duration equals the depth duration to the second
(e.g. Australia: 481 files at 1800 s = 10 days = 864000 s of depth at 300 s). Date-level
agreement was checked first; the 30-minute alignment is now measured (see above).
"""
import os
import re
import glob
from datetime import datetime

import numpy as np
import rasterio

DEPTH_STEP_SECONDS = 300
RAINFALL_STEP_SECONDS = 1800

# Pixel offset (rows, cols) of the depth grid inside the DEM / land-use / rainfall grid,
# for regions where those layers are larger than the depth raster. Measured, not assumed:
# UK depth is 170x275 but its DEM/land-use/rainfall are 180x285. Both an independent
# depth-vs-DEM correlation search (best at dy=5,dx=5, r=-0.27 vs a max of +0.03 over all
# 121 candidate offsets) and a wet-pixels-on-baseline-water enrichment search (~5x at
# dy=5..6,dx=5..6) peak there - consistent with a centered 5px border crop on each side.
# Australia's layers all match the depth grid exactly (verified) so it needs no entry.
#
# Geometry of EVERY (region, resolution) relative to the 30 m base grid that the DEM / land use / rainfall live on:
#   (scale, (row_offset, col_offset))  = base pixels per depth cell, and the base-pixel position of depth cell (0, 0).
# The 60 m and 480 m depth rasters are the 30 m results resampled to 2x2 and 16x16 base-pixel cells and CROPPED (the paper:
# "resampled ... using bilinear interpolation"). Each geometry below was recovered from the data with a probe that was first
# validated on the known cases (Australia 30 m -> (0,0), UK 30 m -> (5,5)), and cross-checked by independent signals:
#   Mozambique 480 m  (16, (150, 50))  terrain-structure r=-0.27 (5.2 robust-sd), DEM NoData all outside the crop
#   Pakistan   480 m  (16, (42, 178))  terrain-structure r=-0.09 (5.5 robust-sd) AND baseline-water r=+0.10 agree; NoData outside
#   UK          60 m  (2,  (5, 5))     terrain-structure r=-0.44 at (5..6, 5..6)
#   Australia   60 m  (2,  (0, 0))     terrain-structure r=-0.26 at (0..1, 0..1)
# Precision is about +-2 base pixels (60 m), i.e. < 6% of a 480 m cell. Scripts and logs: verification_outputs/audit/.
# (An earlier version of this loader refused Pakistan and Mozambique as "not alignable"; that was a failure of the first
# search - it did not remove the large-scale terrain trend - not a property of the data.)
LAYER_GEOMETRY = {
    ("Australia", 30): (1, (0, 0)), ("UK", 30): (1, (5, 5)),
    ("Australia", 60): (2, (0, 0)), ("UK", 60): (2, (5, 5)),
    ("Pakistan", 480): (16, (42, 178)), ("Mozambique", 480): (16, (150, 50)),
}
LAYER_OFFSETS = {"UK": (5, 5)}  # kept for older callers: 30 m offsets only

# The DEM / rainfall rasters carry the float32 NoData sentinel (-3.4028235e38): the UK layers have a padding ring of it
# outside the depth-aligned crop. A window that touched it would feed ~1e38 into the network (and the flat-DEM filter
# in windows.py would count such a patch as "valid", its DEM std being enormous). None of the windows this project
# uses touch it (verification_outputs/audit/investigate_uk_nodata.py), but the loader now refuses instead of trusting that.
NODATA_BELOW = -1e30


def has_nodata(arr: np.ndarray) -> bool:
    """True if `arr` holds the NoData sentinel or any non-finite value."""
    return bool((arr < NODATA_BELOW).any() or not np.isfinite(arr).all())

# Fixed absolute depth thresholds (meters), decided in the project plan.
SEVERITY_THRESHOLDS = [0.3, 1.0, 2.0]  # -> none / minor / moderate / severe
SEVERITY_CLASSES = ["none", "minor", "moderate", "severe"]


def depth_to_severity(max_depth_m: float) -> int:
    """Bin a max-depth value (meters) into one of the 4 fixed severity classes."""
    idx = 0
    for t in SEVERITY_THRESHOLDS:
        if max_depth_m >= t:
            idx += 1
    return idx  # 0=none, 1=minor, 2=moderate, 3=severe


class RegionData:
    """Indexes one region's depth/rainfall/DEM files without loading pixel data."""

    def __init__(self, root_dir: str, region: str, fidelity_dir: str):
        """
        root_dir: path to the extracted FloodCastBench folder
        region: e.g. "Australia"
        fidelity_dir: e.g. "High-fidelity flood forecasting/30m"
        """
        self.root_dir = root_dir
        self.region = region

        depth_dir = os.path.join(root_dir, fidelity_dir, region)
        depth_files = glob.glob(os.path.join(depth_dir, "*.tif"))
        if not depth_files:
            raise FileNotFoundError(f"No depth TIFFs found in {depth_dir}")

        def depth_seconds(path):
            name = os.path.splitext(os.path.basename(path))[0]
            m = re.match(r"^(\d+)$", name)
            if not m:
                raise ValueError(f"Expected purely numeric depth filename, got: {path}")
            return int(m.group(1))

        self.depth_files = sorted(depth_files, key=depth_seconds)
        self.depth_seconds = [depth_seconds(p) for p in self.depth_files]

        step = self.depth_seconds[1] - self.depth_seconds[0]
        for i in range(1, len(self.depth_seconds)):
            gap = self.depth_seconds[i] - self.depth_seconds[i - 1]
            if gap != step:
                raise ValueError(
                    f"Depth sequence is not evenly spaced at index {i}: "
                    f"gap={gap}s, expected {step}s"
                )
        self.depth_step_seconds = step

        rain_dir = os.path.join(root_dir, "Relevant data", "Rainfall", f"{region} flood")
        rain_files = glob.glob(os.path.join(rain_dir, "*.tif"))
        if not rain_files:
            raise FileNotFoundError(f"No rainfall TIFFs found in {rain_dir}")

        def rain_datetime(path):
            name = os.path.splitext(os.path.basename(path))[0]
            m = re.match(r"^(\d{8})-S(\d{6})$", name)
            if not m:
                raise ValueError(f"Unexpected rainfall filename format: {path}")
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")

        rain_with_dt = sorted(((rain_datetime(p), p) for p in rain_files), key=lambda x: x[0])
        self.rain_files = [p for _, p in rain_with_dt]
        self.rain_start = rain_with_dt[0][0]
        rain_step = (rain_with_dt[1][0] - rain_with_dt[0][0]).total_seconds()
        for i in range(1, len(rain_with_dt)):
            gap = (rain_with_dt[i][0] - rain_with_dt[i - 1][0]).total_seconds()
            if gap != rain_step:
                raise ValueError(f"Rainfall sequence is not evenly spaced at index {i}")
        self.rain_step_seconds = rain_step

        self.dem_path = os.path.join(root_dir, "Relevant data", "DEM", f"{region}_DEM.tif")
        if not os.path.isfile(self.dem_path):
            raise FileNotFoundError(self.dem_path)

        self.landuse_path = os.path.join(root_dir, "Relevant data", "Land use and land cover", f"{region}.tif")
        if not os.path.isfile(self.landuse_path):
            raise FileNotFoundError(self.landuse_path)

        with rasterio.open(self.depth_files[0]) as src:
            self.grid_height, self.grid_width = src.height, src.width

        m = re.search(r"(\d+)m$", fidelity_dir.rstrip("/"))
        if not m:
            raise ValueError(
                f"Cannot parse pixel size from fidelity_dir={fidelity_dir!r} "
                f"(expected it to end in e.g. '30m')"
            )
        self.pixel_size_m = float(m.group(1))

        key = (region, int(round(self.pixel_size_m)))
        if key not in LAYER_GEOMETRY:
            raise ValueError(
                f"{region} at {self.pixel_size_m:.0f} m has no registered layer geometry in LAYER_GEOMETRY - windows read "
                f"at the same (row, col) would cover different geography. Refusing to load rather than return misaligned data."
            )
        self.layer_scale, self.layer_offset = LAYER_GEOMETRY[key]
        dy, dx = self.layer_offset
        s = self.layer_scale
        for label, path in (("DEM", self.dem_path), ("land use", self.landuse_path),
                            ("rainfall", self.rain_files[0])):
            with rasterio.open(path) as src:
                layer_h, layer_w = src.height, src.width
            if dy + s * self.grid_height > layer_h or dx + s * self.grid_width > layer_w:
                raise ValueError(
                    f"{region}: {label} grid is {layer_h}x{layer_w}, too small for the registered geometry "
                    f"(scale {s}, offset {self.layer_offset}) of a {self.grid_height}x{self.grid_width} depth grid."
                )

    def _layer_window(self, row: int, col: int, patch_size: int):
        """Window (in 30 m base pixels) into the DEM / land-use / rainfall grids for a depth-grid (row, col) patch."""
        dy, dx = self.layer_offset
        s = self.layer_scale
        return ((dy + s * row, dy + s * (row + patch_size)), (dx + s * col, dx + s * (col + patch_size)))

    def _read_layer(self, path: str, row: int, col: int, patch_size: int, nearest: bool = False) -> np.ndarray:
        """Read a base-grid layer for a depth-grid patch, reduced to (patch, patch): block-mean over the cell's
        scale x scale base pixels (nearest = the cell-centre pixel, for categorical land use)."""
        s = self.layer_scale
        with rasterio.open(path) as src:
            a = src.read(1, window=self._layer_window(row, col, patch_size), boundless=True, fill_value=0)
        if s == 1:
            return a
        a = a.reshape(patch_size, s, patch_size, s)
        if nearest:
            return a[:, s // 2, :, s // 2]
        return a.astype(np.float64).mean(axis=(1, 3)).astype(np.float32)

    def load_dem_patch(self, row: int, col: int, patch_size: int) -> np.ndarray:
        return self._read_layer(self.dem_path, row, col, patch_size).astype(np.float32)

    def rainfall_index_for_depth_second(self, depth_second: int) -> int:
        """Map a depth timestep (seconds since sim start) to its rainfall file index,
        holding each 30-min rainfall value constant across the depth steps inside it.
        NOTE (round 2): file j covers [1800 j, 1800 (j+1)) s, so this returns rain that has NOT YET FALLEN for a frame inside the interval (the 'hybrid' regime: partly
        future rain for the last observed frame). For past-only input use file floor(t/1800) - 1 ('causal'; see completion/engine.py)."""
        idx = int(depth_second // self.rain_step_seconds)
        return min(idx, len(self.rain_files) - 1)

    def window_frame_indices(self, start_idx: int, t_in: int, horizon: int, stride: int = 1):
        """Depth-file indices for one window: (history_indices, future_indices).

        History is start_idx, start_idx+stride, ...; the forecast continues at the same
        stride. stride=1 is the native 5-minute step; stride=6 is 30 minutes, which is
        rainfall's native resolution (at stride 1, five history frames span only 25
        minutes, so the rainfall channel is essentially constant across the sequence).
        The severity scan in windows.py uses this same method so scan and loader can
        never disagree about which frames a window covers."""
        history = [start_idx + stride * k for k in range(t_in)]
        future = [start_idx + stride * (t_in + k) for k in range(horizon)]
        if future[-1] >= len(self.depth_files):
            raise ValueError(
                f"start_idx={start_idx} t_in={t_in} horizon={horizon} stride={stride} runs past "
                f"the end of the depth sequence ({len(self.depth_files)} files)"
            )
        return history, future

    def max_start_index(self, t_in: int, horizon: int, stride: int = 1) -> int:
        """Largest start_idx for which a whole window still fits."""
        return len(self.depth_files) - 1 - stride * (t_in + horizon - 1)

    def load_patch_sequence(self, row: int, col: int, patch_size: int,
                             start_idx: int, t_in: int, horizon: int, stride: int = 1):
        """Read one (row, col) patch across the window described by window_frame_indices():
        the history frames are the model input, the future frames are both the regression
        targets and the source of the severity label.

        Returns: input_sequence (t_in, H, W, 3) [depth, DEM, rainfall (mm/h; for the last frame this includes up to 30 min of not-yet-fallen rain)],
                 depth_targets (horizon, H, W),
                 severity_label (int, 0-3), max_future_depth (float)
        """
        window = ((row, row + patch_size), (col, col + patch_size))
        history_idx, future_idx = self.window_frame_indices(start_idx, t_in, horizon, stride)

        dem_patch = self.load_dem_patch(row, col, patch_size)

        depth_frames = []
        rainfall_frames = []
        for i in history_idx:
            with rasterio.open(self.depth_files[i]) as src:
                depth_frames.append(src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32))
            rain_idx = self.rainfall_index_for_depth_second(self.depth_seconds[i])
            rainfall_frames.append(self._read_layer(self.rain_files[rain_idx], row, col, patch_size).astype(np.float32))

        depth_hist = np.stack(depth_frames)          # (t_in, H, W)
        rainfall_hist = np.stack(rainfall_frames)     # (t_in, H, W)
        dem_seq = np.broadcast_to(dem_patch, depth_hist.shape)  # (t_in, H, W)
        input_sequence = np.stack([depth_hist, dem_seq, rainfall_hist], axis=-1)  # (t_in, H, W, 3)

        depth_targets = []
        for i in future_idx:
            with rasterio.open(self.depth_files[i]) as src:
                depth_targets.append(src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32))
        depth_targets = np.stack(depth_targets)  # (horizon, H, W)

        for what, arr in (("depth history", depth_hist), ("rainfall", rainfall_hist),
                          ("DEM", dem_patch), ("depth targets", depth_targets)):
            if has_nodata(arr):
                raise ValueError(
                    f"{self.region}: {what} window at depth-grid (row={row}, col={col}) contains NoData/NaN pixels "
                    f"(the layers have a NoData padding ring outside the aligned crop). Refusing to return it."
                )

        max_future_depth = float(depth_targets.max())
        severity_label = depth_to_severity(max_future_depth)

        return input_sequence, depth_targets, severity_label, max_future_depth

    def load_landuse_patch(self, row: int, col: int, patch_size: int) -> np.ndarray:
        """Real Esri/Impact Observatory Sentinel-2 land-cover codes (verified against
        this exact product's published legend): 1=Water, 2=Trees, 4=Flooded Vegetation,
        5=Crops, 7=Built Area, 8=Bare Ground, 9=Snow/Ice, 10=Clouds, 11=Rangeland.
        At 60 m / 480 m the class of each cell's centre pixel is returned (nearest neighbour)."""
        return self._read_layer(self.landuse_path, row, col, patch_size, nearest=True).astype(np.int32)

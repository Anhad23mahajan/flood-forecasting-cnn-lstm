"""
Stage 5b: predict_severity_with_reasoning() - the actual usable inference path,
combining the trained Capsule Network (Stage 3) with the Symbolic Rule Layer
(Stage 5) into one call. This is what a demo, or Stage 6's evaluation, calls -
not the individual pieces tested in isolation.
"""
import warnings

import numpy as np
from scipy import ndimage
import tensorflow as tf

from model import decode_horizon
from real_data_loader import RegionData, SEVERITY_CLASSES
from symbolic_rule_layer import apply_symbolic_rules

# The rule thresholds in symbolic_rule_layer.py were derived from Australia at 30m only.
RULE_CALIBRATION_PIXEL_SIZE_M = 30.0


def _compute_slope_pct(dem_patch: np.ndarray, pixel_size_m: float) -> float:
    gy, gx = np.gradient(dem_patch, pixel_size_m)
    return float(100 * np.mean(np.sqrt(gx ** 2 + gy ** 2)))


def _compute_drainage_proximity_m(landuse_patch: np.ndarray, pixel_size_m: float) -> float:
    water_mask = (landuse_patch == 1)  # verified against the real Esri/Impact Observatory legend
    if not water_mask.any():
        return float("nan")  # no baseline water anywhere in this patch - proximity undefined, not zero
    dist_px = ndimage.distance_transform_edt(~water_mask)
    return float(np.mean(dist_px) * pixel_size_m)


def predict_severity_with_reasoning(encoder, decoder, capsule_head, rd: RegionData,
                                     row: int, col: int, patch_size: int, start_idx: int,
                                     t_in: int, horizon: int, stride: int = 1):
    """Runs the real forward pass + rule layer on one real patch/time window.
    Returns a dict with the raw capsule prediction, the reasoned output, the
    justification, and the underlying feature values (for transparency/debugging).
    `stride` is the window's time stride in depth steps (6 = 30 min); depth-rate is
    computed per hour of real time, so it stays comparable across strides."""
    if rd.pixel_size_m != RULE_CALIBRATION_PIXEL_SIZE_M:
        warnings.warn(
            f"{rd.region} is at {rd.pixel_size_m:.0f}m but symbolic_rule_layer.py's thresholds were "
            f"derived at {RULE_CALIBRATION_PIXEL_SIZE_M:.0f}m (Australia) - a 128px patch covers a very "
            f"different physical area, so the rules are not calibrated for this region."
        )

    input_seq, depth_targets, true_severity, max_future_depth = rd.load_patch_sequence(
        row, col, patch_size, start_idx, t_in, horizon, stride
    )

    x_batch = tf.constant(input_seq[np.newaxis, ...], dtype=tf.float32)
    enc_out = encoder(x_batch, training=False)

    severity_scores = capsule_head(enc_out[2][:, -1], training=False)  # probs or lengths; argmax is the same
    raw_severity_idx = int(tf.argmax(severity_scores[0]))

    pred_depth_frames = decode_horizon(
        decoder, enc_out, x_batch, horizon, training=False
    ).numpy()[0]  # (horizon, H, W) - model's OWN depth prediction,
    # used for depth-rate, not the ground-truth depth_targets - this is what the reasoning layer
    # would actually have access to at real inference time (no ground truth available then)

    # Depth-rate = the model's PREDICTED change in patch-mean depth, from the last observed frame
    # to the last forecast frame, per hour of real time. (An earlier version differenced the
    # first and last FORECAST frames. With the original from-scratch decoder those frames were
    # essentially noise, so the rate that triggered the escalation rule was an artifact; with the
    # persistence-anchored decoder that difference is ~0 by construction. Measuring from the
    # observed frame uses the observation plus the model's predicted trend, which is the quantity
    # the rule thresholds were derived from - the real patch-mean change per hour.)
    last_observed_mean = float(input_seq[-1, ..., 0].mean())
    hours = horizon * stride * rd.depth_step_seconds / 3600.0
    depth_rate_m_per_hr = float((pred_depth_frames[-1].mean() - last_observed_mean) / hours)

    rainfall_mm = float(input_seq[-1, ..., 2].mean())   # mm/h; the hybrid input includes rain of the interval containing the last frame (partly not yet fallen)

    dem_patch = rd.load_dem_patch(row, col, patch_size)
    slope_pct = _compute_slope_pct(dem_patch, rd.pixel_size_m)

    landuse_patch = rd.load_landuse_patch(row, col, patch_size)
    drainage_proximity_m = _compute_drainage_proximity_m(landuse_patch, rd.pixel_size_m)

    reasoned_idx, reasons = apply_symbolic_rules(
        raw_severity_idx, depth_rate_m_per_hr, rainfall_mm, slope_pct,
        drainage_proximity_m if not np.isnan(drainage_proximity_m) else 0.0,
    )

    return {
        "true_severity": SEVERITY_CLASSES[true_severity],
        "raw_severity": SEVERITY_CLASSES[raw_severity_idx],
        "reasoned_severity": SEVERITY_CLASSES[reasoned_idx],
        "justification": "; ".join(reasons),
        "features": {
            "depth_rate_m_per_hr": depth_rate_m_per_hr,
            "rainfall_mm": rainfall_mm,
            "slope_pct": slope_pct,
            "drainage_proximity_m": drainage_proximity_m,
        },
        "pred_depth_frames": pred_depth_frames,
        "depth_targets": depth_targets,
    }

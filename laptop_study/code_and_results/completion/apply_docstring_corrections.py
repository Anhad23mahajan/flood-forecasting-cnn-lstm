"""Docstring / comment corrections in the original modules (stale_claims rows RL-01..PS-01). No behaviour changes except the wording of one rule-layer message. Idempotent.
Usage: python completion/apply_docstring_corrections.py"""
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EDITS = {
    "real_data_loader.py": [
        ("ASSUMPTION (not confirmed by any embedded metadata - no TIFF tag or readme ties\ndepth's t=0 to a real calendar date): rainfall's first file corresponds to\ndepth's t=0.",
         "CONFIRMED PHYSICALLY (completion/a_rain_lag.py, round 2): rainfall file j is the rain that FALLS DURING [1800 j, 1800 (j+1)) s of the simulation; the values are mm/h\nintensities (depth gain over an interval = 0.494-0.502 x the file value; correlation 1.00 at lag 0 in all four regions). The archive itself gives no clock time for\nt = 0 (no TIFF tag or readme ties it to a calendar date); none is used. Original reasoning: rainfall's first file corresponds to depth's t=0."),
        ("Date-level\nagreement is confirmed; the hour-level start is inferred, not verified.",
         "Date-level\nagreement was checked first; the 30-minute alignment is now measured (see above)."),
        ("holding each 30-min rainfall value constant across the depth steps inside it.\"\"\"",
         "holding each 30-min rainfall value constant across the depth steps inside it.\n        NOTE (round 2): file j covers [1800 j, 1800 (j+1)) s, so this returns rain that has NOT YET FALLEN for a frame inside the interval (the 'hybrid' regime: partly\n        future rain for the last observed frame). For past-only input use file floor(t/1800) - 1 ('causal'; see completion/engine.py).\"\"\""),
        ("        Returns: input_sequence (t_in, H, W, 3) [depth, DEM, rainfall],",
         "        Returns: input_sequence (t_in, H, W, 3) [depth, DEM, rainfall (mm/h; for the last frame this includes up to 30 min of not-yet-fallen rain)],"),
    ],
    "model.py": [
        ("depth[t], static DEM, rainfall[t] per historical timestep.",
         "depth[t], static DEM, rainfall[t] per historical timestep (rainfall = intensity in mm/h of the interval containing t: partly future for the last frame, the 'hybrid' input)."),
        ("DEM values (tens to hundreds of metres) swamp rainfall (0-9 mm)", "DEM values (tens to hundreds of metres) swamp rainfall (0-9 mm/h)"),
        ("RAIN_SCALE_MM = 5.0\n", "RAIN_SCALE_MM = 5.0     # in mm/h (the rainfall files hold intensities)\n"),
        ("  CO5 - encoder_lstm/decoder_lstm: the temporal encoder-decoder generating the forecast sequence\n",
         "  CO5 - encoder_lstm/decoder_lstm: the temporal encoder-decoder generating the forecast sequence\n        (round 2: in the trained ORIGINAL model (v1) the depth history was not exploited - see PROJECT_LOG 'Completion, round 2'; the final ConvLSTM is model_v2.py)\n"),
    ],
    "depth_experiment.py": [
        ("NOT comparable to the paper's tables: those forecast 19 steps from the initial state over the whole domain.",
         "NOT comparable to the paper's tables: those forecast 19 steps at 300 s from the true depth at t = 1 over the whole domain and report a relative error (not metres);\nplain persistence under that protocol is comparable to the paper's best model (completion/f_paper_protocol.py)."),
        ("30 m audit found the depth skill was driven by rainfall and not by the depth history.",
         "30 m audit of the ORIGINAL model (v1) found its skill was driven by rainfall and not by the depth history (a limitation of v1, not of the flood; see PROJECT_LOG 'Completion, round 2')."),
    ],
    "symbolic_rule_layer.py": [
        ("rainfall (patch-mean, mm):      none=", "rainfall (patch-mean, mm/h):    none="),
        ("RAINFALL_CORROBORATE_THRESHOLD = 3.0     # mm, patch-mean", "RAINFALL_CORROBORATE_THRESHOLD = 3.0     # mm/h, patch-mean"),
        ("f\"rainfall ({rainfall_mm:.2f} mm) is actively corroborating the rising depth trend\"",
         "f\"rainfall intensity ({rainfall_mm:.2f} mm/h, current 30-minute interval) accompanies the rising depth trend\""),
    ],
    "predict_severity.py": [
        ("    rainfall_mm = float(input_seq[-1, ..., 2].mean())",
         "    rainfall_mm = float(input_seq[-1, ..., 2].mean())   # mm/h; the hybrid input includes rain of the interval containing the last frame (partly not yet fallen)"),
    ],
}
for fn, edits in EDITS.items():
    s = open(fn, encoding="utf-8").read()
    n_done = 0
    for old, new in edits:
        if new in s:
            continue
        assert s.count(old) == 1, (fn, old[:60], s.count(old))
        s = s.replace(old, new)
        n_done += 1
    open(fn, "w", encoding="utf-8").write(s)
    print(fn, "edits applied:", n_done)

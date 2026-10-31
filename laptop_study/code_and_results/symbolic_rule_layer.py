"""
Stage 5: Symbolic Rule Layer (NeuroSymbolic) - post-hoc, deterministic rule logic
applied AFTER the trained model produces its outputs. No weights, no training,
no gradients through this module - it is a plain function, which is exactly
what makes the "NeuroSymbolic" label accurate (see project explainer doc).

All four thresholds below were derived from real Australia (30m, high-fidelity)
data spanning all 4 severity classes (10 examples each, from
train_stage3_balanced.py's class-balanced scan), NOT guessed:

  depth-rate (patch-mean, m/hr):  none=0.0000  minor=0.0002  moderate=0.0028  severe=0.0035
  rainfall (patch-mean, mm/h):    none=0.000   minor=0.189   moderate=2.055   severe=4.056
  slope (% grade):                none=4.85    minor=4.53    moderate=5.79    severe=8.22
  drainage-proximity (m to       none=911.9   minor=774.4   moderate=1030.1  severe=1260.6
  nearest baseline water px):

Two of these ran counter to the naive assumption before checking real data:
higher drainage-proximity and higher slope both correlate with MORE severe
outcomes here, not less. The physical reading: severe flooding means water
reaching the floodplain, farther from the baseline channel than routine minor
flooding - and often on higher-energy, steeper terrain, not just pooling in
flat low spots. The rules below follow the real data's direction, not the
initial (wrong) assumption.
"""
from real_data_loader import SEVERITY_CLASSES

DEPTH_RATE_ESCALATE_THRESHOLD = 0.0030   # m/hr, patch-mean - near the moderate/severe boundary
DEPTH_RATE_RECEDE_THRESHOLD = -0.0001    # m/hr, patch-mean - mildly negative = receding
RAINFALL_CORROBORATE_THRESHOLD = 3.0     # mm/h, patch-mean - near the moderate/severe boundary
SLOPE_HIGH_ENERGY_THRESHOLD = 7.0        # % grade - near the moderate/severe boundary
DRAINAGE_FLOODPLAIN_THRESHOLD = 1000.0   # meters - near the moderate/severe boundary


def apply_symbolic_rules(raw_severity_idx: int, depth_rate_m_per_hr: float, rainfall_mm: float,
                          slope_pct: float, drainage_proximity_m: float):
    """Returns (reasoned_severity_idx, list_of_justification_strings).
    raw_severity_idx: the Capsule Network's own prediction (0=none..3=severe) - the
    starting point these rules adjust, not replace outright."""
    severity_idx = raw_severity_idx
    reasons = []

    if depth_rate_m_per_hr > DEPTH_RATE_ESCALATE_THRESHOLD and drainage_proximity_m > DRAINAGE_FLOODPLAIN_THRESHOLD:
        if severity_idx < len(SEVERITY_CLASSES) - 1:
            severity_idx += 1
        reasons.append(
            f"escalated: depth rising {depth_rate_m_per_hr:.4f} m/hr while {drainage_proximity_m:.0f} m "
            f"from the baseline water extent - a floodplain-inundation pattern, not just channel rise"
        )

    if depth_rate_m_per_hr < DEPTH_RATE_RECEDE_THRESHOLD:
        if severity_idx > 0:
            severity_idx -= 1
        reasons.append(f"de-escalated: depth trend is receding ({depth_rate_m_per_hr:.4f} m/hr)")

    if rainfall_mm > RAINFALL_CORROBORATE_THRESHOLD and depth_rate_m_per_hr > 0:
        reasons.append(
            f"rainfall intensity ({rainfall_mm:.2f} mm/h, current 30-minute interval) accompanies the rising depth trend"
        )

    if slope_pct > SLOPE_HIGH_ENERGY_THRESHOLD and depth_rate_m_per_hr > 0:
        reasons.append(
            f"steep terrain ({slope_pct:.1f}% grade) suggests high-energy flow, not simple pooling"
        )

    if not reasons:
        reasons.append("no rule triggered an adjustment; raw model classification retained")

    return severity_idx, reasons

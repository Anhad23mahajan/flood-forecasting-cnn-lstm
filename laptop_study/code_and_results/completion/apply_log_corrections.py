"""Append 'Correction, round 2' notes to the historical sections of PROJECT_LOG.md (idempotent; only touches text BEFORE the round-2 section).
Each edit is (needle, mode, text, line_offset): the single line containing `needle` (must be unique in the historical part) is changed:
  append  = note added at the end of the line (or of the line `line_offset` lines below);  replace = `needle` replaced by `text`;  cell = note inserted before the closing pipe of a table row.
Source of the list: completion/review/stale_claims.md (rows PL-01 ... PL-31).
Usage: python completion/apply_log_corrections.py"""
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MARK = "[Correction, round 2"
RAIN = ("**[Correction, round 2: rainfall file j is the rain that FALLS DURING [1800 j, 1800 (j+1)) s and the values are mm/h intensities (measured in all four regions; "
        "'Completion, round 2', section 1). The original input gave the last observed frame the interval that CONTAINS it, i.e. up to 30 minutes of rain that had not yet fallen.]**")
V1 = ("**[Correction, round 2: this describes the ORIGINAL model (v1) only. v1 could not use the depth history (each frame is squeezed through one 128-number dense layer and depth is divided by 5 m) "
      "and its last-frame rain channel contained look-ahead rain; later baselines showed the depth history IS informative and carries the skill of the final models ('Completion, round 2', sections 2, 7 and 8).]**")
LOW = ("**[Correction, round 2: true for v1, but the bar is low: a per-pixel linear model of the depth history alone scores +0.15 (480 m) to +0.35 (30 m); the final models score +0.81 to +0.87 on never-seen regions, "
       "and a two-feature lookup table +0.74 to +0.79 ('Completion, round 2', sections 2, 7 and 8).]**")

EDITS = [
    ("Every number quoted was produced by code in this handoff.", "append",
     "\n\n**Round 2 (20 September 2026).** A second audit corrected several statements of the earlier sections (rain timing and 'no future information'; 'the skill is rainfall-driven / the history is not used'; "
     "'RMSE in metres' for the paper's relative error; georeferencing scope; the headline depth skill; the severity/rules/Grad-CAM results). The old text is kept; each stale statement is followed by a "
     "**[Correction, round 2 ...]** note. The current state of the project is the section **'Completion, round 2'** (sections 1-10) and **'What is and is not established'** at the end. "
     "Severity, symbolic-rule and Grad-CAM numbers in the sections before 'Completion, round 2' belong to the original (v1) joint model; their re-evaluation is section 9 of the round-2 section.", 0),
    ("independently confirmed, so the loader flags it as an assumption.", "append",
     " **[Correction, round 2: later confirmed physically: rainfall file j is the rain falling during [1800 j, 1800 (j+1)) s (domain-mean depth gain = 0.494-0.502 x the file value; correlation 1.00 at lag 0 in all four regions). "
     "The calendar clock time of t = 0 is still not given by the archive; no experiment uses it.]**", 0),
    ("| rainfall (mm) |", "replace", "| rainfall (mm/h; hybrid channel = the interval containing the last frame, partly future) |", 0),
    ("and tracks rainfall (see below).", "append", " **[Correction, round 2: v1 tracking a rainfall channel that contained up to 30 minutes of rain that had not yet fallen.]**", 0),
    ("Georeferencing with them is valid for Australia/UK only.", "append",
     " **[Correction, round 2: the world files give the 30 m grid directly; the recovered scale and offset were applied to the 60 m and 480 m grids and checked against OpenStreetMap for Mozambique, Pakistan and the UK at 60 m.]**", 0),
    ("with 4-6 mm of rainfall - the model picking up rainfall-driven change", "replace",
     "with 4-6 mm/h of (hybrid-channel) rainfall - the model responding to its rainfall input [Correction, round 2: for the last frame that input includes rain that had not yet fallen; this shows input sensitivity, not learned flood dynamics]", 0),
    ("depth skill is rainfall-driven and the depth history adds < 0.01.", "append",
     " " + V1 + " New mistakes found in round 2 are listed in section 10 of the round-2 section.", 0),
    ("(+0.189 vs +0.190)", "append", " " + LOW, 0),
    ("nothing here covers the", "append",
     " **[Correction, round 2: this check covers Australia at 30 m only; the coarse regions were checked separately against OpenStreetMap ('Completion, round 2', section 4).]**", 1),
    ("**No future information enters an input.**", "replace",
     "**No future DEPTH enters an input.** [Correction, round 2: rainfall did - each frame's rain channel is the interval that contains it, so the last observed frame carried up to 30 minutes of rain that had not yet fallen; this audit checked file indices, not what each rain file covers in time.]", 0),
    ("storm-phase clock.", "append", " " + V1, 0),
    ("up to 26.3 mm per 30 min,", "replace", "up to 26.3 mm/h [round-2 correction: mm/h intensities, about 13 mm per 30 min],", 0),
    ("Pakistan up to 56.8 mm)", "replace", "Pakistan up to 56.8 mm/h)", 0),
    ("the loader's floor mapping is correct for both.", "append",
     " **[Correction, round 2: alignment is also confirmed physically at 30-minute resolution (lag 0). The floor mapping pairs each frame with the interval that CONTAINS it: a correct alignment, but for the last observed frame that interval is partly future.]**", 0),
    ("Its Table 5 (RMSE m / NSE)", "replace", "Its Table 5 (relative error - unitless, NOT metres - / NSE)", 0),
    ("and the depth branch does not use terrain", "replace", "and the depth branch does not use terrain [round 2: v1's ablation; removing terrain also changes the skill of the newer models by less than 0.01]", 0),
    ("the target is depth at t=1..20 (19 steps) over the whole domain", "replace",
     "the model receives the true depth at t=1 of a 20-frame sample (300 s spacing) and predicts t=2..20 (19 steps) over the whole domain (FNO+ also receives terrain and the rainfall series)", 0),
    ("rain 26 mm, depth 15 m", "replace", "rain intensity 26 mm/h, depth 15 m", 0),
    ("in all four evaluations**", "append", " " + LOW, 0),
    ("Rainfall drives the skill at every resolution.**", "append", " " + V1, 0),
    ("FNO+ RMSE 0.0786 m Pakistan -> Mozambique, 0.0248 m Australia -> UK at 60 m", "replace",
     "FNO+ relative error 0.0786 Pakistan -> Mozambique, 0.0248 Australia -> UK at 60 m, unitless and not metres; plain persistence scores as well or better on most of the paper's own measures, see 'Completion, round 2', section 3", 0),
    ("worse than persistence in every evaluation**", "append", " **[Correction, round 2: a statement about v1.]**", 0),
    ("eight trained-model evaluations (4 evaluations x 2 seeds):**", "append", " " + V1, 0),
    ("gives each step its own rainfall", "append", " [round 2: the interval that contains the frame's time]", 0),
]

s = open("PROJECT_LOG.md", encoding="utf-8").read()
cut = s.find("<!-- ROUND2 START -->")
head, tail = (s[:cut], s[cut:]) if cut >= 0 else (s, "")
# the 'established' section at the end is rewritten by build_log.py, so stop before it
est = head.find("## What is and is not established")
if est >= 0:
    head, tail = head[:est], head[est:] + tail
L = head.split("\n")
done = skipped = 0
for needle, mode, text, off in EDITS:
    hits = [i for i, l in enumerate(L) if needle in l]
    if not hits and mode == "replace" and text in head:
        skipped += 1
        continue
    assert len(hits) == 1, (needle, len(hits))
    i = hits[0]
    j = i + off
    if MARK in L[j] or (mode == "replace" and text in L[i]) or (mode == "append" and text.strip()[:60] in head):
        skipped += 1
        continue
    if mode == "append":
        L[j] = L[j] + text
    elif mode == "replace":
        L[i] = L[i].replace(needle, text)
    done += 1
open("PROJECT_LOG.md", "w", encoding="utf-8").write("\n".join(L) + tail)
print("applied", done, "skipped (already there)", skipped)

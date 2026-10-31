"""Append/replace 'round 2' correction notes in the OLD ('Current state') part of HANDOFF_README.md (idempotent; rows HR-01 ... HR-12 of completion/review/stale_claims.md).
Usage: python completion/apply_readme_corrections.py"""
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = ("**[Round 2: this describes the ORIGINAL model (v1) only. v1 could not use the depth history and its last-frame rain channel held up to 30 minutes of not-yet-fallen rain; "
      "the final models' numbers are in the block at the top of this file and in PROJECT_LOG 'Completion, round 2'.]**")
EDITS = [
    ("**Read `PROJECT_LOG.md` first**, especially \"Pre-flight before Colab\".", "replace",
     "**Read `PROJECT_LOG.md` first**: \"Completion, round 2\" and the final \"What is and is not established\" are the CURRENT state; \"Pre-flight before Colab\" is the design story of the original (v1) model."),
    ("**The paper's three same-resolution transfer tests**: Australia -> UK at 30 m and 60 m, Pakistan -> Mozambique at 480 m", "replace",
     "**The paper's transfer tests**: Pakistan -> Mozambique at 480 m and Australia -> UK at 60 m are same-resolution (the paper's 30 m UK test is a zero-shot downscaling); we run those two plus our own Australia -> UK at 30 m"),
    ("Depth beats \"copy the last frame\" in a region it never saw.", "replace",
     "[Round 2: v1 statements; a per-pixel linear model of the depth history alone scores +0.15 to +0.35, the final models +0.81 to +0.87, so beating persistence is a low bar.] Depth beats \"copy the last frame\" in a region it never saw."),
    ("Skill +0.19 is ~10% lower RMSE, i.e. ~1-2 mm", "replace", "Skill +0.19 (v1) is ~10% lower RMSE, i.e. ~1-2 mm"),
    ("**Most important finding: the depth skill is rainfall-driven.**", "replace",
     "**Finding for the ORIGINAL model (v1): its depth skill was rainfall-driven** [Round 2: this described v1's weakness, not the flood - see the block at the top; the audit also did not test rain timing]."),
    ("UK 60 m: 2 padded windows", "replace", "UK 60 m: 2 padded tile positions x 60 start times = 120 windows"),
    ("the mechanism is the same as at 30 m (rainfall drives it; terrain and depth history do not matter)", "replace",
     "the v1 behaviour is the same as at 30 m (removing rainfall took v1's skill to about zero; a v1 property, see the top of this file)"),
]
s = open("HANDOFF_README.md", encoding="utf-8").read()
cut = s.find("<!-- ROUND2 END -->")
pre, body = (s[:cut + len("<!-- ROUND2 END -->")], s[cut + len("<!-- ROUND2 END -->"):]) if cut >= 0 else ("", s)
done = skipped = 0
for needle, mode, text in EDITS:
    n = body.count(needle)
    if text in body:
        skipped += 1
        continue
    assert n == 1, (needle[:50], n)
    body = body.replace(needle, text)
    done += 1
open("HANDOFF_README.md", "w", encoding="utf-8").write(pre + body)
print("applied", done, "skipped", skipped)

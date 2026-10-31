# Round 2 — the corrected and completed study (20–21 September 2026)

**Everything in this folder is NEW work.** The two ORIGINAL documents (the 15 September deck `CNN-LSTM_Based_Spatiotemporal_Flood_Evolution_Forecasting.pptx` and `Project_Explainer_End_to_End.md`) are one folder up, in `DL Project`, and were not changed.

## What is in here (read in this order)

| Folder / file | What it is |
|---|---|
| `1_Presentation_and_Explainers/Project_Story_for_Aman.md` | **Start here.** A plain-language guide to the whole project, from the beginning to the end (what went wrong, what we did, what we found, what runs where). Written so it can be explained to anyone. |
| `1_Presentation_and_Explainers/CNN-LSTM_Based_Spatiotemporal_Flood_Evolution_Forecasting_v2.pptx` | The updated 19-slide deck (new results slides, corrected claims, a new "Robustness and Extra Checks" slide). |
| `1_Presentation_and_Explainers/Project_Explainer_End_to_End_v2.md` | The full technical explainer with all result tables (section 10) and what is / is not established (section 12). |
| `2_Project_Record/PROJECT_LOG.md` | The complete record, including every mistake. The section "Completion, round 2" is the current state. |
| `2_Project_Record/REPORT.md`, `SEVERITY_REPORT.md` | All result tables, generated from the result files by script. |
| `2_Project_Record/PREREGISTRATION.md` | The test plan written down before the final runs. |
| `2_Project_Record/HANDOFF_README.md` | How to run everything (top block = round 2). |
| `3_Reviews_and_Audits/` | The independent checks: raw-data verification (two rounds), code reviews (two rounds), the stale-claims sweep and two document audits. |
| `4_Notebooks/` | The three notebooks. `flood_forecasting_study_colab.ipynb` is the current (smaller demo) one; the other two are the superseded original-model notebooks. |
| `5_Figures/` | Result figures and the final architecture diagram (`architecture_v3.png`, the one used in the deck). |
| `6_Code_and_Results_Package/flood_forecasting_handoff.zip` | All code, all result files (about 310 training runs), logs and reviews — 54 MB. Not included: the 46 GB dataset and the 13 GB caches (rebuildable). |

## The findings in five lines

1. The corrected final models score **+0.81 to +0.87 on floods they never saw**; the original model scored +0.06 to +0.19 on the same tests.
2. Most of that skill is simple: **continue each spot's recent depth trend** (a two-number lookup table already gets about 90%).
3. **The LSTM (recurrence) adds nothing measurable** when trained as long as a plain CNN; neighbouring pixels add nothing; a 150× larger model is no better.
4. Models fail on **the very first moments of a flood** they never saw in training; the capsule head, rulebook and severity heat map do not pay off (details in the story).
5. Two independent train→test pairs only (one flood per region); development-stage results, not a forecasting system.

## Corrections you should know about
Several earlier statements were wrong and are corrected in these documents: rain files contain up to 30 minutes of future rain (not "no future information"); the paper's "RMSE" is a relative error, not metres; "rainfall drives the skill" was true of the original model only; my safety check wrongly aborted 8 runs (I first called them "collapses"). The log lists every mistake.

*Not verified independently: the trained networks' own forecast errors (the raw-data numbers, baselines and splits were). The pptx validator of this environment is broken, so structural checks were done by other means. The Colab notebook has not yet been run on Colab (optional).*

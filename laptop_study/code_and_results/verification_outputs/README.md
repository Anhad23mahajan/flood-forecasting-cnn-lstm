# Verification outputs

**Top level (`stage1_*`, `stage2*`, `stage3*`, `stage4*`, `stage5*`)** - outputs from the *original*
local smoke tests. Stage 1 (the data loader) is still valid. Stages 2-5 were produced by the
pre-redesign architecture and are kept as a record; their conclusions were revised - see
`PROJECT_LOG.md`, "Pre-flight before Colab".

**`preflight/`** - the evidence behind the redesign. Each log is one local execution of the
Colab notebook's actual code against the real data (`dryrun_notebook.py`), at small scale:

| Log | Config | What it showed |
|---|---|---|
| `run1_residual_only_tiny_12win_2ep.log` | persistence-anchored decoder only, 12 windows, 2 epochs | mechanics; every `depth_rate` was 0.0000 |
| `run2_residual_only_100win_25ep_PLATEAU.log` | same, 100 windows, 25 epochs | depth skill flat at exactly +0.0205 from epoch 2 (and +0.0207 on the unseen UK); `depth_rate` 0.0007 for every window; severity predicted "moderate" for 38/40 |
| `run3_skip_plus_margin_100win_25ep.log` | + decoder skip connection, + margin loss | skill varies and reaches +0.09; `depth_rate` tracks rainfall; severity 25% -> 60% |
| `run4_cosine_lr_clip_100win_25ep.log` | + cosine LR decay, gradient clipping | skill climbs smoothly to +0.24 and holds; UK +0.16; severity 50% (overfit, train margin loss 0.085) |
| `run5_final_code_tiny_regression.log` | final code, 12 windows, 2 epochs | all 17 notebook cells execute (run4's Grad-CAM was blank - fixed afterwards, see log) |

All of these are ~100 training windows - smoke tests of whether the design can learn, not results.

# Archived: scripts from before the pre-Colab redesign

These are the local smoke-test scripts from Stages 2-5. They were written against the
**original** architecture (a decoder that rebuilt the depth field from scratch, no skip
connection, softmax cross-entropy on capsule lengths, raw un-normalized inputs, random-window
sampling) and **will not run** against the current `model.py`, whose decoder API changed.

They are kept as the record of what was run and reported at the time (their outputs are in
`../verification_outputs/`, top level). They were superseded, not deleted, because the reasons
they were superseded are the most important thing in this project so far:

- their sampler was first-found and their split random over windows from the same patches (leaky);
- they had no baselines, so "loss decreased" and "83% accuracy" went unchallenged;
- the decoder could never reach a persistence-level forecast, which is why Stage 2's predictions
  were blurry - not only "too little data", as first reported.

The current, working equivalents are `../dryrun_notebook.py` (executes the whole pipeline locally,
including Grad-CAM and the symbolic layer) and `../flood_forecasting_colab.ipynb`. See
`../PROJECT_LOG.md`, section "Pre-flight before Colab".

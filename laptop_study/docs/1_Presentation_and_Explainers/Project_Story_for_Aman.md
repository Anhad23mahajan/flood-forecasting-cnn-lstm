# The Whole Project, Start to Finish — a plain-language guide for Aman

*Written 20–21 September 2026 so that you (Aman) can follow everything that happened and explain it to anyone. Read the 5-minute version first; the rest is the detail. Every number here comes from the result files in `completion/` (section 9 says where things are).*

---

## 1. The 5-minute version

- **The project.** We teach a computer program to look at the last 5 "pictures" of a flood (water depth, terrain, rain) and draw the next 5 pictures, up to 2.5 hours ahead. The pictures come from a physics simulator (the dataset **FloodCastBench**: four simulated floods in Australia, the UK, Pakistan and Mozambique). We also added: a **severity guess** (none / minor / moderate / severe) using a *capsule network*, a **heat map** of where the program "looked" (*Grad-CAM*), and a small **rulebook** that adds a written reason to the severity guess.
- **How we grade it.** Skill = how much better than the lazy strategy "copy the last picture". **0 = no better than copying, 1 = perfect, negative = worse.**
- **What we did first.** We built the whole pipeline, trained on Australia, tested on the UK (a flood the program had never seen). It scored about +0.19 (roughly 10% less error than copying). We were pleased.
- **What was wrong.** Over several audits (checking our own work) we found the first results were partly wrong or misleading: we had used only half the dataset; the rain data secretly contained up to 30 minutes of the future; our program was weak and we didn't notice because we never compared it with easy guesses; and one of our "findings" (rain drives the forecast) was just a symptom of that weak program.
- **What we did about it.** We redid the study fairly: rules written down before running, many easy baselines, small new models, three rain settings, tests on map tiles and on floods never seen, checks by separate computer agents (they re-derived the raw-data numbers, the baselines and the train/test splits from the raw files without our code, and reviewed the code; the trained networks were not re-run), and about a dozen extra robustness checks.
- **What we found.** Most of the skill is simple: "the water keeps going the way it was going". A table with two numbers per spot (how deep now, how fast it just changed) already gets about 90% of the best score. Our small ConvLSTM (the final CNN-LSTM) and a small plain CNN both score about **+0.81 to +0.87 on floods they never saw** (the original model scored +0.06 to +0.19). **Looking at neighbouring pixels and adding memory (the LSTM) adds nothing measurable once both models are trained equally long.** The capsule head beat its plain control only without augmentation (with augmentation the gap vanishes, consistent with a small-sample effect); both lose to a one-line rule; the rulebook makes answers worse; the heat map for the depth forecast passes a fair test.
- **What we can and cannot claim.** Each never-seen flood is **one** event, so we have two independent train-and-test pairs, not many. Our models fail badly on the very first moments of a flood if they never saw such moments in training. These are careful development-stage results, not a forecasting system.

---

## 2. What the project is, from the very beginning

**The problem.** Physics simulators can predict how a flood spreads, but they are slow (the Pakistan event took about two weeks of computing). A trained neural network gives a forecast in seconds. The question: can a network learn to continue a flood simulation?

**The data.** FloodCastBench (Xu et al. 2025) has four simulated floods. For each: depth maps every 5 minutes, terrain (DEM), rainfall every 30 minutes, land use. Australia and the UK come at fine resolution (30 m and 60 m); Pakistan and Mozambique at coarse resolution (480 m). The coarse and 60 m maps are re-sampled, cropped copies of the 30 m simulation, so lining them up with terrain and rain needed a careful search (done, and checked against real maps).

**The task we set.** Cut each map into 128×128-pixel squares ("tiles"). Give the model 5 frames 30 minutes apart; ask for the next 5 frames. 30 minutes because rainfall is recorded every 30 minutes (at 5-minute steps the water barely changes and copying the last frame is already perfect to about 2 mm).

**The ideas we wanted to test** (from our project plan): a **CNN-LSTM** for the depth maps; a **Capsule Network** for the severity class; **Grad-CAM** as a visual explanation; a **symbolic rule layer** ("NeuroSymbolic") that adds a written reason; and a **cross-region test**: train on one flood, test on another.

**Two words you must know.**
- *Persistence* = the lazy forecast "the future equals the last observed picture". Every score is measured against it.
- *Skill* = 1 − (model's squared error ÷ persistence's squared error).

---

## 3. What we built first, and the first results

Steps (all recorded in `PROJECT_LOG.md`): put the dataset on Hugging Face so Colab can read it; write a loader that lines up depth, terrain and rain; build the model (a CNN encoder → LSTM → decoder that outputs "last frame + a learned change"); build the capsule head, Grad-CAM and the rule layer; run everything on Colab (T4 GPU) and locally.

Along the way we fixed several early problems (all in the log): a decoder that couldn't even reproduce the last frame; softmax on capsule lengths collapsed to one class; a broken first notebook and sampler; 5-minute steps made the task trivial.

**First results (the "original model", called v1):** trained on Australia, tested on the UK: depth skill about **+0.19** (seven runs ranged +0.04 to +0.22 on the UK). Severity: the capsule head scored 55–70%, while the trivial rule "threshold the deepest water in the last frame" scored about 93%. The rulebook changed almost nothing. We wrote "the forecast is last frame + current rainfall" and "the LSTM's history is not used" — which is true **of that original model only**, as section 4 explains.

---

## 4. What went wrong — in the order we found it

### 4.1 Half the dataset was never used (found on 20 September, by Anhad)
The pipeline used only Australia and the UK at 30 m — 13.5 of 26.9 GB. Pakistan, Mozambique and the 60 m data had been dropped after **one failed attempt** to line up their maps, and the project had been called "100% complete" while the audit had only covered the half in use. The failed attempt used a method that could not succeed (it forgot to remove the big terrain trend). A better method, tested first on cases where the answer was known, recovered all the alignments; we then ran the two extra transfer tests. **Lesson:** a failed first try is not proof that something is impossible, and an audit must cover everything.

### 4.2 The second audit (20 September) — deeper problems
1. **The rain data peeked into the future.** Each rainfall file is the rain that falls *during the 30 minutes after* its time stamp (we proved this in all four floods: the water gained by the domain matches the file, correlation 1.00). The original input gave the last observed frame the rain of the half-hour it sits in — up to 30 minutes of rain that had not yet fallen. I had told you "no future information enters any input"; that was wrong for rain. (It turned out to be worth only about +0.01 to +0.025, but the claim was false.)
2. **The original model was weak and we never noticed.** We only ever compared with "copy the last frame". When I finally tried easy guesses, a *straight-line* guess for each pixel scored about **twice** the original CNN-LSTM. The reason: the old design squeezed each frame through one 128-number layer and scaled depth by 5 m, so tiny changes (millimetres) were invisible to it. (We re-ran the original under the same folds: it scores +0.18 to +0.19 on the UK, +0.06 on Mozambique, uses only the rain, and cannot read the depth history.)
3. **An earlier "finding" was an artefact.** "Rainfall drives the forecast and the depth history is not used" described our weak model, not floods.
4. **The paper's score.** The paper's "RMSE" is a *relative* error, not metres (I had written it as metres). And on the paper's own measures, simply copying the initial picture matches or beats its best model (for Mozambique: NSE 0.9995 vs 0.955; CSI at 1 mm 0.98 vs 0.93). The paper has no such baseline. This does not make its models bad as surrogates; it shows the benchmark's scores mostly reward "the water is already there".

### 4.3 Smaller things found and corrected along the way
- A water-surface probe I tried to double-check the map alignment **failed its own test** (kept as a negative result, not used).
- A code review found an aggregation bug that would have silently mixed different runs, confidence intervals that were too narrow, two hypothesis tests coded looser than I had written them down, a Grad-CAM test that measured the wrong thing, and an input layout that handicaps the recurrent model in one setting — all fixed or disclosed.
- **A safety check of mine raised false alarms (21 September).** I had a check that stops a training run if the network's output stops varying ("dead network"). It looked at the 12 *earliest* held-out windows (tiles of the training flood), when the flood has barely started, so even healthy networks look "dead" there. It stopped eight runs. I first told Anhad the original model had "collapsed" — wrong: re-run without the check it trained normally (+0.18 to +0.195), and so did the first side-channel run it had stopped (+0.885; the second, a retry with the next seed, was not re-run). The check now samples windows across the whole held-out period.
- **A second review (21 September)** — one agent re-derived the new numbers from raw files (everything it could check matched: the stored copy-the-last-frame errors of about 190,000 window scores, the lookup-table baseline to 1e-8 or better, no training/test overlap in any of 56 blocked or reverse splits; it could not re-run the neural models), another reviewed my new code (no critical error; three major points, all fixed: an experiment that mixed "more data" with "more training", guard-disabled runs that were not marked in the tables, and a runner that could start every job at once).
- **The laptop hung once** because I briefly ran five training jobs instead of four on a 15.5 GB machine; every finished result was intact, only jobs in progress were repeated.

---

## 5. What we did in the corrected study

- **Rules written down first.** Before running anything final, we saved the protocol (recipe, folds, models, hypotheses H1–H6) in `completion/PREREGISTRATION.md` with a checksum. Model *design* used only held-out tiles of the training flood; the never-seen floods were not used for any design decision (four early design runs of the discarded heavy family did print their UK score; they were not consulted, and the two chosen models were designed with runs that never touched the UK).
- **Three experiments:** Australia → UK at 60 m (**E60**), the same at 30 m (**E30**), Pakistan → Mozambique at 480 m (**E480**). Depth only (severity only makes sense for Australia at 30 m).
- **Three ways to feed rain** (the "regimes"): *causal* = only rain that already fell; *hybrid* = the original input (partly future); *forced* = also the rain of the next 2.5 hours (a rain-forced surrogate, like the paper's FNO+).
- **A ladder of players, easy to fancy:** copy the last frame; straight-line trend; a **lookup table** keyed on two numbers (depth now, change just before); a **per-pixel MLP** (no map structure at all); our **local CNN** (4,101 parameters, no memory); our **small ConvLSTM** (4,741 parameters, the final CNN-LSTM); a **heavy** ConvLSTM (653k parameters); and the original v1.
- **Fair testing.** Every tile of the training flood is held out once (3 folds); the never-seen flood is the strong test. Confidence intervals use *block bootstraps* (neighbouring windows overlap in time and each flood is one event, so ordinary intervals would be too narrow). A stricter test also hides a block of time.
- **Turning things off** ("ablations"): remove rain, terrain or the depth history at test time; or train models without one ingredient.
- **Extra robustness checks (added after the main results):** five different random seeds per fold; training 30 vs 90 epochs; a learning curve (25 to 600 training windows); a different training recipe (bigger batches, higher learning rate); the strict "hidden time block" test repeated with other seeds; and the reverse direction (train on the small UK or Mozambique flood, test on Australia or Pakistan).
- **Independent checks:** separate agents re-derived the raw-data numbers, the baselines and the splits from the raw files without using our code (all matched; the trained networks were not re-run), and reviewed the code twice.

---

## 6. The results, in plain words

*Skill on the never-seen flood, hybrid input, average of 3 folds (0 = copying):*

| | Australia → UK, 60 m | Australia → UK, 30 m | Pakistan → Mozambique, 480 m |
|---|---|---|---|
| Straight-line trend | +0.32 | +0.34 | +0.15 |
| **Original CNN-LSTM (v1), same folds** | **+0.19** | **+0.19** | **+0.06** |
| Two-number lookup table | +0.79 | +0.76 | +0.74 |
| Per-pixel MLP | +0.85 | +0.85 | +0.83 |
| Local CNN | +0.84 | +0.83 | +0.81 |
| Small ConvLSTM (final CNN-LSTM) | +0.87 | +0.84 | +0.81 |

1. **Most of the skill is simple.** The lookup table alone gets about 90% of the best score. The forecast is essentially "continue the recent trajectory of the water at each spot".
2. **The models use the depth history and almost nothing else.** Freezing the history takes the skill to about zero; scrambling its order makes it strongly negative (for every model except the per-pixel CNN on the UK and the original model); removing rain costs only about 0.01–0.02; removing the terrain, nothing. The same picture appears when we *train* models without one ingredient (UK 60 m / Mozambique; the full local CNN scores +0.84 / +0.81): without the depth history +0.17 / +0.09; without the rain history +0.84 / +0.80; without any rain +0.84 / +0.80; without terrain +0.84 / +0.81. The depth history is worth about 0.7 of skill; rain about 0.01; terrain nothing.
3. **The original model is the odd one out.** It uses only the rain and cannot read the depth history (freeze or scramble the history: no change; remove the rain: skill drops to zero). That is why our early "finding" happened.
4. **Looking at neighbours does not help.** A per-pixel network, a 5×5 and an 11×11 CNN score within 0.01 of each other on the UK; on Mozambique the 11×11 CNN is 0.025 below the per-pixel one, so looking at neighbours never helps. Trained long enough (90 epochs) the plain CNN is level with the per-pixel network.
5. **Does the LSTM (recurrence) help? Not measurably.** With our standard 30-epoch recipe the ConvLSTM was ahead of the plain CNN by +0.025 on the UK at 60 m. But the plain CNN was slightly under-trained: trained for 90 epochs both (a fairness check we added), the difference is +0.004 (UK) and −0.014 (Mozambique), inside the noise. **So the honest claim is "the ConvLSTM matches the plain CNN", not "the LSTM captures temporal context better".**
6. **The rain leak was small.** For the local CNN, strictly past rain → the original input adds about +0.01 to +0.025 and knowing the future rain adds about +0.02 to +0.03 more (for the ConvLSTM the first step is +0.013 on the UK at 60 m and +0.000 on Mozambique). The ConvLSTM dipped in the "future rain" layout because we gave it the future rain as five extra time steps; giving it as extra input channels instead lifts it from +0.77 to +0.82 (Mozambique) and +0.85 to +0.88 (UK) — the dip was our input layout.
7. **Severity, capsules, rules, heat maps (Australia 30 m).** Without augmentation the capsule head is right 67.8% of the time on realistic windows and a plain dense head 60.2% (capsule better by 7.6 points). **With data augmentation both reach about 80% and the gap vanishes (+0.5 points, interval includes 0): the capsule's earlier edge is consistent with a small-sample effect (one seed per fold).** Both lose badly to the one-line rule (97.4%): the label is almost fixed by the last frame. The rulebook **lowers** accuracy with and without augmentation (67.8% → 63.1%; 80.3% → 74.1%); its stated reasons about the depth trend are true about 84% of the time. Grad-CAM for the depth forecast passes a fair deletion test (top-heat pixels matter more than two control masks in 84–92% of windows); for the severity class it is weak for the capsule head (top beats both controls in 55-56% of windows) and moderate for the dense head (73-75%). The same holds when the input contains no look-ahead rain.
8. **A weakness the strict test exposed: the very start of a flood.** If the model never saw the *first moments* of a flood in training, it fails badly on them. On Pakistan, 6 of the 480 hidden windows (the ones at the very start of the flood) carry 98% of the plain CNN's error; without them its score is +0.84 (the ConvLSTM on the same test scores +0.74). We checked it wasn't one unlucky run: five different seeds and a different training recipe all fail the same way. The same shows up when training on tiny Mozambique and testing on Pakistan (−0.88, but +0.83 without the first windows). The two-number lookup table (+0.55) and the ConvLSTM (+0.44) stay sensible there; the plain CNN and the per-pixel network do not. In the other reverse directions (UK → Australia) the plain CNN works: +0.66 to +0.74 (the ConvLSTM +0.58 to +0.70).
9. **How steady are the numbers?** With five seeds per fold the score on a never-seen flood varies by about ±0.01, so differences below 0.02 mean nothing. A much bigger ConvLSTM (653k parameters, 150× larger) does not do better (+0.84 on the UK, +0.80 on Mozambique): size is not the limit. A learning-curve check with the number of training steps held equal shows that doubling the training windows changes the plain CNN's score by only -0.002 to +0.008 (inside seed noise): its gain from more windows is consistent with the extra training steps, so what limits it looks like training time, not data.
10. **A different training recipe** (bigger batches, higher learning rate) gives about the same never-seen score (0.01 lower) and, in the one run made with it, the plain CNN's scores on hidden Pakistan tiles are steady (+0.77 against +0.43), which suggests those wobbles were training noise. The ConvLSTM behaves the same way (a little lower on Mozambique, +0.785 against +0.811).

---

## 7. What it all means — the honest conclusions

- **Good news:** the fixed pipeline works; skill on never-seen floods is high (+0.81 to +0.87 against +0.06 to +0.19 for the original), the numbers are stable to about ±0.01, the physics checks (rain timing, maps on real rivers, independent recomputation from raw files) hold up, and two independent reviews found no critical error.
- **Sobering news:** on this benchmark the difficult-looking machinery (LSTM memory, neighbour context, capsules, rulebook) adds nothing measurable beyond very simple rules. The right claim is not "the CNN-LSTM captures spatiotemporal dynamics" but "**most of the forecast skill on FloodCastBench, at 30-minute steps, is per-pixel trajectory extrapolation; a small ConvLSTM reaches the same skill as a plain CNN and a lookup table gets most of it**". That is a legitimate, useful finding.
- **Limits you must say out loud:** each never-seen flood is one event, so there are two independent train→test pairs; the never-seen skill is high partly because the benchmark's floods evolve smoothly; models fail on the start of a flood they never saw; the reverse-direction tests have no spatial hold-out (only hidden time blocks); numbers are development-stage results from one pre-registered study run on one laptop; a domain expert has not reviewed the rule reasons; and our numbers are not comparable with the paper's tables (different task).
- **What we left out on purpose** (analysed and dropped as redundant, in the log): ConvLSTM training-time ablations, and 30 m variants of some ConvLSTM checks (30 m is the same storm pair as 60 m, not new evidence).

---

## 8. What runs where (laptop vs Colab) — and how to reproduce

- **The full study** (about 310 training runs, about 52 hours of training in total on four parallel jobs, roughly 21 hours from the first to the last result) ran on Anhad's **laptop** (12-core CPU, 15.5 GB RAM). It is the source of every number in the documents.
- **The Colab study notebook** (`flood_forecasting_study_colab.ipynb`) is the **same code**, but its default is a **smaller demonstration**: the 60 m experiment, two models (local CNN and ConvLSTM), three rain settings, three folds. Reasons: the full data are about 27 GB (60 m alone about 3.5 GB) and a full Colab run would take many hours in one fragile session. Run it with a GPU (Runtime → Change runtime type → GPU → Run all, roughly an hour), then **File → Download .ipynb** — the download contains all plots (skill ladder, ablations, rain settings, training curves, forecast maps against the truth, Grad-CAM). To run more, edit the first settings cell (`EXPERIMENTS`, `MODELS`, `REGIMES`). The extra analyses (training-time ablations, strict hold-out, side-channel variant, seed repeats, learning curve, longer training, reverse transfers, original-model comparison, severity augmentation and causal regime) are only in `completion/run_queue.py`, which runs locally.
- **The two older notebooks** (`flood_forecasting_colab.ipynb`, `flood_forecasting_multires_colab.ipynb`) contain the **original model (v1)** and are marked "superseded". GPU and CPU runs are not bit-identical: in the first study the same configuration moved by about 0.01 to 0.03 skill between machines and thread counts, and a cross-hardware check of the final models was not done.

---

## 9. Where everything is

- **DL Project folder:** the deck (`..._v2.pptx`, 19 slides), the explainer (`Project_Explainer_End_to_End_v2.md`), this guide, `architecture_diagram/`. The original deck and explainer were left untouched.
- **Code handoff** (`flood_forecasting_handoff.zip`): `PROJECT_LOG.md` (the full record, including every mistake), `HANDOFF_README.md`, the notebooks, `model_v2.py`, `completion/` (engine, baselines, severity study, report and figure scripts, pre-registration, the review reports in `completion/review/`, every result file and log). `completion/results/REPORT.md` has all the tables.

---

## 10. How to explain it in three minutes (a suggested talk)

1. *The problem:* flood simulators are slow; can a small network continue them?  
2. *What we built:* a CNN-LSTM on flood "flipbooks", plus severity, heat maps and rules; judged against "copy the last picture".  
3. *What we found wrong in our own first results* (rain peeking into the future, a weak model, half the data unused) — and how we fixed it (pre-registered protocol, baseline ladder, independent checks).  
4. *What the corrected results say:* skill about +0.8–0.87 on unseen floods; most of it is simple trajectory extrapolation; the ConvLSTM matches a plain CNN when both train equally long; capsules, rules and heat maps: honest mixed results.  
5. *Limits:* one event per flood; not comparable with the paper's tables; the very start of an unseen flood fails.

**Likely questions.** *Why not just use persistence?* Persistence is the reference (skill 0); we beat it by a lot. *Why is a two-number table so good?* At 30-minute steps a flood evolves smoothly. *Is the LSTM needed?* No: with equal training a plain CNN does as well. *Why did you re-run things so many times?* Because the first results had mistakes we found by checking; the log lists all of them. *Is it a real forecasting system?* No: it's a benchmark study on simulated floods.

---

## 11. Glossary

**Persistence** = copy the last frame. **Skill** = improvement over persistence (0 = same, 1 = perfect). **Tile** = a 128×128 square of the map. **Fold** = one of 3 splits where a third of the tiles is hidden from training. **Held-out tiles** = tiles of the training flood never used for training. **Never-seen region** = a different flood in a different place (UK, Mozambique). **Regime** = how rain is fed to the model (causal / hybrid / forced). **Baseline ladder** = simple players from "copy" to "per-pixel network". **Ablation** = switch one ingredient off to see what matters. **CNN-LSTM / ConvLSTM** = a network that looks at pictures and remembers their order. **Capsule** = a network unit that outputs a small vector; here used for the severity class. **Grad-CAM** = a heat map of where a model's evidence came from. **Pre-registration** = writing the test plan down before running it. **Bootstrap** = re-sampling the results to see how uncertain they are. **Seed** = the random starting point of a training run; different seeds give slightly different scores. **Epoch** = one pass over the training windows. **Learning curve** = skill as a function of how much training data is used. **Onset windows** = windows at the very beginning of a flood simulation. **Guard** = the automatic check that stops a run whose network has stopped producing varied output.

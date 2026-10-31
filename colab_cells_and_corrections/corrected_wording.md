# Corrected wording for the PDF and the notebook text

All numbers below come from the notebook outputs (first version and version 2) and from the baseline check. "Trend" means: the water keeps changing the way it changed in the last 30 minutes (one number, 0.921 at 30 m and 0.948 at 480 m).

## A. Change these (a professor could catch them)

**1. PDF section 6, "The main result" box**
- Old: "So the model has learned something real about how floods move, and that knowledge carries over to new places."
- New: "On the two floods it has never seen, the first version beat 'copy the last map' (skill +0.31 in the UK and +0.15 in Mozambique). But a simple trend forecast, with no network at all, scores +0.95 and +0.94. Version 2, which starts from the trend forecast, matches it (+0.95 and +0.94). So at 30 minutes ahead, continuing the trend does almost all the work."

**2. PDF section 6, "The one failure: late Australia"**
- Old: "The model learned only from the rising, rainy part of the flood and never saw water draining away, so its small mistakes cost more than they gain."
- New: "The first version added a small amount of water over most of the map and missed the local rises and falls. The training part of the flood did contain falling water (in 73 of its 331 steps the average depth fell), and the trend forecast fitted on it scores +0.99 on this last period, so the failure was not caused by missing examples of draining water. Version 2 fixes it (+0.989)."

**3. PDF section 7, the water-balance check**
- Old: "All dots sit on the red line, in all four floods. The data conserves water exactly as the law says."
- New: "The dots sit very close to the red line in all four floods. Over the whole flood, the water that was added differs from the rain by up to about 5% (0.95 in the UK to 1.05 in Mozambique)."
- Old: "Nothing soaks into the ground and nothing flows out of the map."
- New: "Almost all of the rain stays on the ground. According to the dataset paper, the Australian, UK and Mozambique simulations let water leave freely at the edge of the map and the Pakistan simulation has a river flowing in, so the law is only an approximation (the totals above differ from the rain by up to about 5%)."

**4. PDF section 7, the result of the rule**
- Old: "The rule improves every flood, and halves Australia's problem. This is a good example of NeuroSymbolic AI..."
- New: "The rule helped the first version, because that version added extra water. On version 2 it makes almost no difference (between -0.007 and +0.001), so we keep it as a physics check and not as an improvement. The network finds where the water goes, and the rule checks that the total is sensible."

**5. PDF section 10, Grad-CAM**
- Old: "What the heat maps show (this is a strong point)" and "The model is looking at the right things, for physically sensible reasons."
- New: "What the heat maps show" and "The bright areas are places where the water is changing, which is sensible. These pictures are illustrations, not proof. In an earlier test, the heat maps of the capsule network changed from one training run to another."

**6. PDF section 14, questions and answers**
- Why add an LSTM? New: "The architecture we set out to build is a CNN with an LSTM, because a flood is a sequence of maps. We then tested it fairly. For a 30-minute forecast, continuing the trend already gets 91% to 99% of the possible improvement over copying, and the LSTM version adds only about 0.001 on top. Memory should matter more for longer forecasts, so that is what we test next."
- How do you know the model is actually good? New: "We compare it with two simple forecasts, 'copy the last map' and 'continue the trend'. The first version beat copying but not the trend. Version 2 matches the trend. So it is as good as the strongest simple forecast, and not yet better."
- Why not just use a bigger model? New: "We have not tested bigger models. We only have four floods, so we kept the model small. At 30 minutes ahead there is also little left to gain, at most about 1% to 9% of the copy error."

## B. Add these

- **Section 4 (the main model), a third design choice:** "Version 2 starts from a simple trend forecast and the network learns only a correction on top of it. Its last layer starts at zero, so before training it is exactly the trend forecast, and training can only add to it."
- **Section 5 (training):** say that, judging from the training figure, the first 30 m model kept its epoch-2 weights because the check error got worse after that, and that all results are from a single run.
- **Section 6:** add the sentence "The 'typical error' is averaged over the whole map, including dry squares, so for the copy forecast it is about 20% to 45% lower than the error over wet squares only."
- **Section 8 (capsule network):** add "In the Pakistan test 698 of the 1,000 tiles are Severe, so always answering Severe would already be right 69.8% of the time and 87% accuracy is less impressive than it looks. 'Severe' also means something different at 30 m (a 2 km tile) and at 480 m (a 30 km tile), so accuracies of the two models should not be compared. We have not yet compared the capsule network with a plain output layer."
- **Section 9 (rules):** add "All three rules use the flooded share of the map now or in the forecast, which is almost always a good guide to the rating. Rules 1 and 2 did most of the work and rule 3 changed the rating in only 21, 21, 10 and 0 tiles, so the gain says more about the rules than about the capsule network."
- **Section 12 (weaknesses):** add "A simple trend forecast beat the first version in all four tests (see the comparison chart). All results are single runs."

## C. Small fixes in the notebook

- The CSI function counts a step with no flooded squares as 0. Easiest fix: do not show the CSI columns on slides. (Mozambique: 18 of 284 steps; its copy CSI of 0.9065 would be about 0.97 without them.)
- Rule 2 in the code uses `>=` ("at least as much flooding"), but the table in the text says "more than". Make the text say "at least as much".
- The comment "Skip most of the calm tiles" in `make_training_tiles` is out of date, because all tiles are now kept.
- The training-progress figure appears only when a model is trained (TRAIN_NEW = True). With saved weights the cell prints nothing, so keep a saved copy of that figure for the slides.

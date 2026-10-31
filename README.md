# Can a neural network forecast a flood?

**CNN-LSTM Based Spatiotemporal Flood Evolution Forecasting** is our Deep Learning CA3 project at Symbiosis Institute of Technology, Pune (October 2026). This repository holds the whole project: the report, the presentation, the notebooks, the code and results of a larger study run on a laptop, and the guides we wrote to explain it.

A scroll-through website for the project is in Aman's repository: https://github.com/AmanS-06/FloodCastBench-Project (live at https://amans-06.github.io/FloodCastBench-Project/).

## The idea

A flood can be drawn as a series of water-depth maps. We give a neural network the last four maps, the rain that has already fallen and the height of the ground, and ask it to draw the map for 30 minutes later.

The catch is that in 30 minutes the water in most places hardly changes, so just copying the last map is already a very good forecast. We therefore compare every model with simple forecasts and ask one honest question: is the neural network really better than very simple rules?

We worked on **FloodCastBench**, a public dataset of four floods re-created with a physics simulator: Australia (Feb 2022), the UK (Dec 2015), Pakistan (Aug 2022) and Mozambique (Mar 2019). The models learn on Australia and Pakistan and are tested on the UK and Mozambique, which they never see during training.

## What we built

- A forecaster made of a convolutional neural network followed by a ConvLSTM layer (62,833 parameters, TensorFlow and Keras).
- Two versions of it: Model A starts from the last map, and Model B starts from a simple trend forecast and only learns a small correction.
- A capsule network that rates how serious the flooding will be (none, minor, moderate or severe), with written rules that check it and give a reason for every decision.
- Grad-CAM heat maps that show where the models look.
- A water-balance rule that checks the total amount of water in a forecast against the rain.

## What we found

- At 30 minutes ahead, a simple trend forecast (the last map plus a fraction of the last change, no network) scored 0.91 to 0.99. Our first model, Model A, scored only 0.15 to 0.35 where it beat copying, and it was worse than copying on the last part of the Australia flood.
- Model B, which starts from the trend, matched the trend in every test but did not beat it.
- In a larger laptop study of about 310 training runs, forecasting up to 2.5 hours ahead, a very simple linear model of each square's recent depths scored 0.79 to 0.82. The best networks scored 0.81 to 0.87.
- Trained for the same length, the ConvLSTM and a plain CNN scored the same.
- The capsule network did not clearly beat a plain output layer once rotated and flipped copies of the data were added.

The biggest lesson: **always compare a model with simple forecasts.** Our first model looked good until we did.

Skill is `1 - (error of the forecast / error of copying the last map)`. A skill of 0 means no better than copying, and 1 means perfect.

## Limits

- Every region has one flood, so there are only two independent train-and-test pairs.
- The floods are simulated, so the models learn to imitate a simulator and not real floods.
- The Colab results come from single runs.
- The capsule network and Grad-CAM were built on Model A and not rebuilt on Model B.

## What is in this repository

| Folder | What it holds |
|---|---|
| `report/` | The final CA3 report (Word and PDF) and its figures |
| `presentation/` | The final presentation (PowerPoint and PDF) and the first version from 15 September |
| `notebooks/` | The Colab notebooks: data preparation, data exploration, the model, and the final merged notebook |
| `colab_cells_and_corrections/` | Cells added to the notebooks, the corrected wording and the corrected explanation PDF |
| `evidence/` | The checks behind the numbers: the ridge-penalty check, the baseline check of the Colab results, and the report-building files |
| `laptop_study/docs/` | The record of the laptop study: project log, preregistration, reviews and audits, figures and the updated deck |
| `laptop_study/code_and_results/` | All code, the result files of about 310 training runs, logs and verification outputs of the laptop study |
| `beginner_guide/` | A guide that explains the project from the beginning (PDFs and their LaTeX source) |

Start with `report/`, then `laptop_study/docs/00_READ_ME_FIRST.md`. To run the laptop study, see `laptop_study/code_and_results/HANDOFF_README.md`.

## A correction to read about

On 10 October 2026 Aman found that the linear baselines had been fitted with an unscaled ridge penalty. Re-running them with standardised inputs moved the linear scores from 0.15 to 0.36 up to about 0.79 to 0.82. The report was corrected. `laptop_study/docs/ERRATUM_linear_baselines_10Oct2026.md` lists which older documents still show the old numbers.

## Not included

- The FloodCastBench data (about 46 GB) and the rebuildable caches. Get the data from the dataset authors.
- The saved model weights, which are on a shared Google Drive.

## Authors

Anhad Mahajan (PRN 24070126031) and Aman Sumeet Saxena (PRN 24070126030), AIML A2, Symbiosis Institute of Technology, Pune. Deep Learning CA3, October 2026.

## Credits

- **Data:** Q. Xu, Y. Shi, J. Zhao and X. X. Zhu, *FloodCastBench: a large-scale dataset and foundation models for flood modeling and forecasting*, Scientific Data, 2025.
- **Methods:** ConvLSTM (Shi et al., 2015), capsule networks (Sabour et al., 2017) and Grad-CAM (Selvaraju et al., 2017).

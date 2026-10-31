# Outputs the user pasted from their Colab runs (4 Oct 2026), copied verbatim from the chat

## Version-2 cell 1 (safety check)
Trend numbers: {'30m': 0.921, '480m': 0.948}
Parameters: 62833 (the first version has 62,833)
Largest difference between the untrained model and the trend forecast: 0.0
OK: the check passed

## Version-2 cell 2 (training), the lines that matter
30m: Validation error (cm), trend forecast: 0.1885 | after training: 0.1868   (stopped after epoch 9; best val_loss 0.0349 at epoch 5)
480m: Validation error (cm), trend forecast: 0.0243 | after training: 0.023    (stopped after epoch 8; best val_loss 5.2839e-04 at epoch 4)

## Version-2 cell 3 (whole-map skill, 30 min ahead)
| Region | Tested on | Skill: trend | Skill: CNN-LSTM (first version) | Skill: CNN-LSTM version 2 | Skill: version 2 + rule |
| Australia (30 m) | unseen time (last 15%) | 0.990 | -1.449 | 0.989 | 0.989 |
| UK (30 m) | never-seen region | 0.947 | 0.305 | 0.948 | 0.949 |
| Pakistan (480 m) | unseen time (last 15%) | 0.908 | 0.345 | 0.908 | 0.901 |
| Mozambique (480 m) | never-seen region | 0.942 | 0.150 | 0.943 | 0.941 |

## Baseline cell (screenshot table the user sent)
| Region | Tested on | Trend number | Skill: trend | Skill: trend + rule | Skill: copy + rule | Skill: CNN-LSTM | Skill: CNN-LSTM + rule |
| Australia (30 m) | unseen time (last 15%) | 0.921 | 0.990 | 0.990 | 0.000 | -1.449 | -0.710 |
| UK (30 m) | never-seen region | 0.921 | 0.947 | 0.947 | 0.053 | 0.305 | 0.320 |
| Pakistan (480 m) | unseen time (last 15%) | 0.948 | 0.908 | 0.905 | 0.142 | 0.345 | 0.404 |
| Mozambique (480 m) | never-seen region | 0.948 | 0.942 | 0.942 | 0.051 | 0.150 | 0.172 |

## Where the other numbers of the Colab notebook come from
The notebook outputs (copy/CNN-LSTM errors and skills, water-balance table, severity tables, rule counts) are in
model_aman.txt  (text extract of Aman's notebook "Flood_model_updated (2).ipynb")
My independent baseline script output: ...\08bc3062-701e-4be9-9b7d-f2389f9d26c1\scratchpad\baselines_log.txt and baselines_result.json (same folder).
The figures the user sent (skill bars, error over time, error maps) are in ...\107f3ba2-0656-4848-98d0-d997556bd414\scratchpad\report\diagrams\new_*.png (they are screenshots of the Colab figures).

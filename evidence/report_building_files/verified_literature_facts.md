# Facts about the literature, verified on 7 Oct 2026 by a web-checking agent (Crossref / PMC full text of the FloodCastBench paper)

## FloodCastBench (Xu, Shi, Zhao, Zhu; Scientific Data 12, article 431, 2025; DOI 10.1038/s41597-025-04725-2)
- Four events: Pakistan 2022 (18 to 31 Aug, 14 days), UK 2015 (4 to 7 Dec, 3 days), Australia 2022 (20 Feb to 2 Mar, 10 days), Mozambique 2019 (14 to 20 Mar, 6 days).
- Depth maps saved every 300 s (5 min). Simulated at 30 m. Released versions: 480 m (Pakistan, Mozambique, "low-fidelity"), 60 m and 30 m (Australia, UK, "high-fidelity").
- Terrain: FABDEM (30 m). Land cover: Esri Sentinel-2 land use / land cover. Rainfall: GPM-IMERG Final (0.1 degree, half-hourly), resampled bilinearly to 5 min and 30 m.
- Simulator: 2-D depth-averaged shallow water equations with the convective acceleration neglected (local-inertial form), finite differences on a staggered grid, own Python implementation on an NVIDIA A6000 GPU. Infiltration ignored when saturated.
- Pakistan's 480 m maps: the text says terrain and rainfall were down-sampled to 480 m (bilinear) and fed straight to the hydrodynamic model (a re-run). "Other flood events": results generated at 30 m then resampled (bilinear) to 480 m and 60 m. Mozambique is NOT named in that passage, so the text does not say explicitly how its 480 m maps were made. The paper is internally inconsistent on this point.
- Runtimes reported: Pakistan about one to two weeks, Mozambique about two days on the GPU.
- Benchmark: U-Net, FNO, FNO+ only (FNO+ = FNO plus DEM and rainfall as extra inputs). NO recurrent model (LSTM or ConvLSTM). Task: 20-step samples at 300 s (input = coordinates, time and the initial depth; output = depth for t = 2 to 20, 19 steps). Whether the whole domain is used is not stated. Metrics: "relative mean square error" (called RMSE, dimensionless), NSE, Pearson r, CSI at 0.001 m and 0.01 m. All models Adam, 100 epochs, batch size 1, cosine learning rate from 0.001.
- Transfer tests (zero-shot): Pakistan to Mozambique at 480 m, Australia to UK at 60 m and 30 m. The paper reports no persistence ("copy the last map") baseline (checked in our earlier study).

## FloodCast (Xu, Shi, Bamber, Ouyang, Zhu; Water Research 264, 122162, 2024)
- A framework with two modules, "multi-satellite observation and hydrodynamic modeling"; its forecasting part uses a physics-informed neural solver built on Fourier neural operators (resolution-invariant).

## Other references (bibliographic details verified; content statements used in the report are generic)
- ConvLSTM (Shi et al. 2015): convolutional LSTM for precipitation nowcasting (radar echo maps).
- Capsule networks (Sabour et al. 2017): dynamic routing between capsules (routing by agreement). Grad-CAM (Selvaraju et al. 2017): gradient-weighted class activation maps.
- Xiao et al. 2025 (Int. J. Disaster Risk Science): ConvLSTM for urban flood prediction under dynamic rainfall patterns and exploration of extrapolation capability (a single urban catchment per an earlier note). Xiang et al. 2025 (J. Hydrology): dual-attention ConvLSTM encoder-decoder for probabilistic flood forecasting at river-basin scale.
- Kratzert et al. 2018: LSTM networks for rainfall-runoff modelling. Mosavi et al. 2018: review of machine learning flood prediction. Bentivoglio et al. 2022: review of deep learning for flood mapping.
- Murphy 1988: skill scores based on mean square error. Hyndman and Koehler 2006: measures of forecast accuracy (naive benchmark). Garcez and Lamb (Artificial Intelligence Review 2023): neurosymbolic AI. Karpatne et al. 2017: theory-guided data science.

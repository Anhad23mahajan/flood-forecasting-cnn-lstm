"""
Stage 4 verification: train the depth branch briefly on real Australia data
(same smoke-test scale as Stage 2), then apply Grad-CAM and check the heatmap
actually highlights where the model's prediction is coming from - specifically,
that it lands on real flood-relevant structure (the river channel) rather than
being uniform noise or an all-zero map.
"""
import sys
import time

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from real_data_loader import RegionData
import model as flood_model
from train import CompiledTrainer
from grad_cam import grad_cam

PATCH_SIZE = flood_model.PATCH_SIZE
T_IN, HORIZON = 5, 5


def main():
    rd = RegionData(
        root_dir="data/extracted/FloodCastBench",
        region="Australia",
        fidelity_dir="High-fidelity flood forecasting/30m",
    )

    # reuse the same wet/flood-relevant patch location established in Stage 1's
    # sanity check (row=0, col=600 - clear river channel, confirmed flooded)
    row, col = 0, 600
    start_idx = 2016

    input_seq, depth_targets, severity, max_future = rd.load_patch_sequence(
        row, col, PATCH_SIZE, start_idx, T_IN, HORIZON
    )

    encoder, decoder = flood_model.build_models(dropout=0.3)
    optimizer = flood_model.OPTIMIZERS["adam"](learning_rate=1e-3)
    trainer = CompiledTrainer(encoder, decoder, optimizer, horizon=HORIZON)

    # a handful of real examples around the same wet region/time so the model
    # actually learns this area's real structure before we explain a prediction on it
    print("gathering a few nearby real training examples...")
    train_examples = []
    for dr in (-64, 0, 64):
        for dc in (-64, 0, 64):
            r, c = max(0, row + dr), max(0, min(col + dc, rd.grid_width - PATCH_SIZE))
            try:
                seq, tgt, sev, mx = rd.load_patch_sequence(r, c, PATCH_SIZE, start_idx, T_IN, HORIZON)
                train_examples.append((seq, tgt))
            except Exception:
                continue
    print(f"  {len(train_examples)} examples")

    print("training briefly...")
    for epoch in range(1, 9):
        t0 = time.perf_counter()
        losses = [trainer.run(x, dt, training=True) for x, dt in train_examples]
        print(f"  epoch {epoch}/8  loss={np.mean(losses):.5f}  ({time.perf_counter()-t0:.1f}s)")

    print("\ncomputing Grad-CAM on the held prediction...")
    heatmap, target = grad_cam(encoder, decoder, input_seq, horizon_step=0)
    print(f"target (sum predicted depth): {target:.3f}")
    print(f"heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]  std={heatmap.std():.4f}")

    # visualize: last input depth frame, Grad-CAM heatmap, overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    last_depth = input_seq[-1, ..., 0]
    axes[0].imshow(last_depth, cmap="Blues")
    axes[0].set_title("last observed depth frame")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    axes[2].imshow(last_depth, cmap="Blues")
    axes[2].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[2].set_title("overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("stage4_gradcam_check.png", dpi=120)
    print("saved stage4_gradcam_check.png")

    # quantitative check the plan calls for: does the heatmap correlate with
    # where depth is actually present, more than with a random baseline?
    depth_mask = (last_depth > last_depth.mean()).astype(np.float32)
    heatmap_on_wet = heatmap[depth_mask > 0].mean()
    heatmap_on_dry = heatmap[depth_mask == 0].mean()
    print(f"\nmean heatmap value on above-average-depth pixels: {heatmap_on_wet:.4f}")
    print(f"mean heatmap value on below-average-depth pixels:  {heatmap_on_dry:.4f}")


if __name__ == "__main__":
    main()

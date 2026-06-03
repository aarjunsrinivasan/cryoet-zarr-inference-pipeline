#!/usr/bin/env python3
"""Visualise what the model 'sees' as ribosomes in the in-situ chunk vs synthetic data.

Approach
--------
Rather than re-running the full sliding-window inference (which takes hours on
CPU and is already cached), we:

1. Load the model and apply it to individual 96³ patches extracted at:
   - GT ribosome positions (what does the model output at a known ribosome?)
   - Random background positions (what does background look like?)
   - Both in the in-situ chunk AND in the synthetic volume, for direct comparison.

2. For each patch we record the full 7-class softmax probability map (Z, Y, X)
   and plot the mid-slice of the ribosome channel.

3. We also generate a max-Z-projection heatmap of the ribosome channel from the
   already-cached full-volume predictions (1 209 point detections, all conf < 0.001)
   to see whether the low-confidence signal is spatially structured or noise.

Outputs (written to --output-dir):
  ribosome_patch_comparison.png   — 96³ patch softmax at GT / background, chunk vs synthetic
  ribosome_confidence_map.png     — max-projection heatmap of cached low-conf detections

Usage:
    python visualize_model_output_2c.py
    python visualize_model_output_2c.py --n-gt-patches 6 --n-bg-patches 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import monai.transforms
import numpy as np
import pandas as pd
import torch
import zarr

RIBO_CHANNEL = 3          # channel index in the 6-class model (0-indexed, background is 6)
PATCH_SIZE   = 96         # voxels — matches training crop size
N_CLASSES    = 7          # 6 particles + 1 background

CLASS_NAMES = [
    "ferritin-complex", "beta-amylase", "beta-galactosidase",
    "cytosolic-ribosome", "thyroglobulin", "virus-like-capsid", "background"
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_volume(zarr_path: Path) -> np.ndarray:
    return zarr.open(str(zarr_path), "r")["0"][:]


def load_gt_pts(json_path: Path) -> np.ndarray:
    with open(json_path) as f:
        pts = json.load(f)["points"]
    return np.array([[p["location"]["x"], p["location"]["y"], p["location"]["z"]]
                     for p in pts], dtype=np.float32)


def find_gt_picks(picks_dir: Path, object_name: str) -> Path:
    """Find the GT picks JSON for object_name, tolerating any session ID in the filename."""
    matches = sorted(picks_dir.glob(f"*_{object_name}.json"))
    if not matches:
        raise FileNotFoundError(f"No picks file for '{object_name}' in {picks_dir}")
    return matches[0]


def normalize_volume(vol_zyx: np.ndarray) -> np.ndarray:
    """Z-score normalize a full volume — matches the inference pipeline transform."""
    tf = monai.transforms.Compose([
        monai.transforms.EnsureChannelFirstd(keys=["input"], channel_dim="no_channel"),
        monai.transforms.NormalizeIntensityd(keys=["input"]),
    ])
    result = tf({"input": vol_zyx.astype(np.float32)})["input"]  # (1, Z, Y, X)
    return result[0].numpy()  # back to (Z, Y, X)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_dir: Path) -> torch.nn.Module:
    from topcup.model import SegNet
    pattern = "topcup_phantom_24_tomograms.ckpt"
    model = SegNet.ensemble_from_checkpoints(str(checkpoint_dir), pattern=pattern)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Single-patch inference
# ---------------------------------------------------------------------------

def extract_patch(vol_zyx: np.ndarray, centre_xyz: np.ndarray,
                  pixelsize: float, patch_vox: int = PATCH_SIZE) -> np.ndarray | None:
    """Return a (patch_vox, patch_vox, patch_vox) patch centred at centre_xyz (Å).
    Returns None if the centre is too close to a boundary."""
    Z, Y, X = vol_zyx.shape
    half = patch_vox // 2
    xi = int(round(centre_xyz[0] / pixelsize))
    yi = int(round(centre_xyz[1] / pixelsize))
    zi = int(round(centre_xyz[2] / pixelsize))
    if (xi - half < 0 or xi + half >= X or
            yi - half < 0 or yi + half >= Y or
            zi - half < 0 or zi + half >= Z):
        return None
    return vol_zyx[zi - half:zi + half,
                   yi - half:yi + half,
                   xi - half:xi + half].copy()


def run_patch(model, patch_zyx: np.ndarray) -> np.ndarray:
    """Return per-class softmax probability map (N_CLASSES, Z, Y, X) for one patch.
    patch_zyx must already be full-volume z-score normalised."""
    t   = torch.from_numpy(patch_zyx.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(t.to(device))["logits"]  # (1, C, Z, Y, X)
    probs = torch.softmax(logits[0], dim=0)      # (C, Z, Y, X)
    return probs.cpu().numpy()


# ---------------------------------------------------------------------------
# Figure 1: patch gallery — ribosome channel softmax at GT vs background
# ---------------------------------------------------------------------------

def fig_patch_gallery(model,
                      chunk: np.ndarray, gt_chunk_pts: np.ndarray,
                      syn: np.ndarray,   gt_syn_pts: np.ndarray,
                      pixelsize_chunk: float, pixelsize_syn: float,
                      n_gt: int, n_bg: int, out_path: Path):

    rng   = np.random.default_rng(42)
    Z_c, Y_c, X_c = chunk.shape
    Z_s, Y_s, X_s = syn.shape
    half  = PATCH_SIZE // 2

    # --- collect GT patches (in-situ) ---
    chunk_gt_patches, chunk_bg_patches = [], []
    idx_gt = rng.choice(len(gt_chunk_pts), min(n_gt * 3, len(gt_chunk_pts)), replace=False)
    for i in idx_gt:
        if len(chunk_gt_patches) >= n_gt:
            break
        p = extract_patch(chunk, gt_chunk_pts[i], pixelsize_chunk)
        if p is not None:
            chunk_gt_patches.append((p, gt_chunk_pts[i]))

    # --- collect background patches (in-situ) — positions away from GT ---
    attempts = 0
    while len(chunk_bg_patches) < n_bg and attempts < 500:
        attempts += 1
        xi = rng.integers(half, X_c - half)
        yi = rng.integers(half, Y_c - half)
        zi = rng.integers(half, Z_c - half)
        ctr = np.array([xi * pixelsize_chunk, yi * pixelsize_chunk, zi * pixelsize_chunk])
        if len(gt_chunk_pts) == 0 or np.min(np.linalg.norm(gt_chunk_pts - ctr, axis=1)) > 300:
            p = extract_patch(chunk, ctr, pixelsize_chunk)
            if p is not None:
                chunk_bg_patches.append((p, ctr))

    # --- collect GT patches (synthetic) ---
    syn_gt_patches = []
    idx_gt_s = rng.choice(len(gt_syn_pts), min(n_gt * 3, len(gt_syn_pts)), replace=False)
    for i in idx_gt_s:
        if len(syn_gt_patches) >= n_gt:
            break
        p = extract_patch(syn, gt_syn_pts[i], pixelsize_syn)
        if p is not None:
            syn_gt_patches.append((p, gt_syn_pts[i]))

    n_rows   = max(len(chunk_gt_patches), len(chunk_bg_patches), len(syn_gt_patches))
    n_cols   = 3   # chunk-GT | chunk-BG | synthetic-GT
    fig_w    = n_cols * 3.5
    fig_h    = max(n_rows, 1) * 3.5 + 1.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.5, wspace=0.3)

    col_titles = [
        f"In-situ chunk\nAT GT ribosome position",
        f"In-situ chunk\nAT BACKGROUND position",
        f"Synthetic 16463\nAT GT ribosome position",
    ]

    patch_groups = [chunk_gt_patches, chunk_bg_patches, syn_gt_patches]

    for col, (patches, col_title) in enumerate(zip(patch_groups, col_titles)):
        for row, (patch, centre) in enumerate(patches):
            probs = run_patch(model, patch)          # (C, Z, Y, X)
            ribo  = probs[RIBO_CHANNEL]              # (Z, Y, X)
            mid_z = ribo.shape[0] // 2
            img   = ribo[mid_z]                      # (Y, X) mid-slice

            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(img, cmap="hot", vmin=0, vmax=max(img.max(), 1e-6),
                           origin="lower")
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

            max_p = ribo.max()
            ax.set_title(f"max={max_p:.4f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

            if row == 0:
                ax.set_xlabel(col_title, fontsize=8, labelpad=6)

    fig.suptitle(
        "Ribosome-channel softmax probability — mid-slice of 96³ patches\n"
        "Top value = max probability in the patch",
        fontsize=11, y=1.01
    )
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[fig] Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: max-projection heatmap of low-conf detections
# ---------------------------------------------------------------------------

def fig_confidence_projection(pred_df: pd.DataFrame, gt_chunk_pts: np.ndarray,
                               chunk_shape: tuple, pixelsize: float, out_path: Path):
    Z, Y, X = chunk_shape
    ribo = pred_df[pred_df["particle_type"] == "cytosolic-ribosome"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # max-Z projection of all (even low-conf) ribosome predictions
    conf_map = np.zeros((Y, X), dtype=np.float32)
    if len(ribo):
        xi = (ribo.x / pixelsize).round().astype(int).clip(0, X - 1)
        yi = (ribo.y / pixelsize).round().astype(int).clip(0, Y - 1)
        np.maximum.at(conf_map, (yi.values, xi.values), ribo.conf.values)

    ax = axes[0]
    im = ax.imshow(conf_map, cmap="hot", origin="lower",
                   extent=[0, X * pixelsize, 0, Y * pixelsize])
    plt.colorbar(im, ax=ax, label="Max confidence score")
    if len(gt_chunk_pts):
        ax.scatter(gt_chunk_pts[:, 0], gt_chunk_pts[:, 1],
                   s=20, c="cyan", marker="+", linewidths=1.2,
                   label=f"GT ribosomes ({len(gt_chunk_pts)})")
        ax.legend(fontsize=8)
    max_conf_val = float(ribo.conf.max()) if len(ribo) else 0.0
    ax.set_title(f"Ribosome confidence max-projection (XY)\n"
                 f"n_detections={len(ribo)}, max_conf={max_conf_val:.4f}")
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")

    # confidence distribution
    ax = axes[1]
    if len(ribo):
        ax.hist(ribo.conf.values, bins=50, color="steelblue", alpha=0.8)
        ax.axvline(0.05,  ls=":", color="gray",  lw=1.2, label="Threshold 0.05")
        ax.axvline(0.19,  ls="--", color="black", lw=1.5, label="Threshold 0.19")
        ax.set_xlabel("Confidence score")
        ax.set_ylabel("Count")
        ax.set_title(f"Ribosome prediction confidence distribution\n"
                     f"(all {len(ribo)} detections, no threshold applied)")
        ax.legend(fontsize=9)
    ax.set_yscale("log")

    plt.suptitle("Where does the model place ribosome predictions in the in-situ chunk?",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[fig] Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",    default="output/inference_2c")
    p.add_argument("--data-dir",      default="data")
    p.add_argument("--checkpoint-dir",default="checkpoints")
    p.add_argument("--n-gt-patches",  type=int, default=6,
                   help="GT patches per dataset for the gallery (default 6)")
    p.add_argument("--n-bg-patches",  type=int, default=6,
                   help="Background patches for the gallery (default 6)")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.output_dir)
    data = Path(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("[load] Loading volumes …")
    chunk = load_volume(
        data / "ExperimentRuns/14070_chunk/VoxelSpacing7.840/wbp-denoised-ctfdeconv.zarr")
    syn   = load_volume(
        data / "ExperimentRuns/16463/VoxelSpacing10.012/wbp-denoised-denoiset-ctfdeconv.zarr")

    # Pre-normalise the full volumes exactly as the inference pipeline does:
    # NormalizeIntensityd applied to the whole tomogram before sliding-window.
    # Patches are extracted from these normalised volumes so the model sees the
    # same intensity distribution it saw during full-volume inference.
    print("[load] Pre-normalising volumes (full-volume z-score) …")
    chunk = normalize_volume(chunk)
    syn   = normalize_volume(syn)

    gt_chunk_pts = load_gt_pts(find_gt_picks(
        data / "ExperimentRuns/14070_chunk/Picks", "cytosolic-ribosome"))
    gt_syn_pts   = load_gt_pts(find_gt_picks(
        data / "ExperimentRuns/16463/Picks", "cytosolic-ribosome"))

    pred_df = pd.read_csv(out / "val_pred_df_seed.csv")

    # ── Load model ─────────────────────────────────────────────────────────
    print("[model] Loading checkpoint …")
    model = load_model(Path(args.checkpoint_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    print(f"[model] Running on {device}")

    # ── Figure 1: confidence max-projection (no re-inference needed) ───────
    print("\n[fig 1] Confidence projection map …")
    fig_confidence_projection(
        pred_df, gt_chunk_pts, chunk.shape, 7.84,
        out / "ribosome_confidence_map.png")

    # ── Figure 2: patch gallery (runs model on ~20 patches) ───────────────
    print("\n[fig 2] Patch gallery …")
    fig_patch_gallery(
        model,
        chunk, gt_chunk_pts,
        syn,   gt_syn_pts,
        7.84, 10.012,
        args.n_gt_patches, args.n_bg_patches,
        out / "ribosome_patch_comparison.png")

    print(f"\nDone. Figures written to {out}")


if __name__ == "__main__":
    main()

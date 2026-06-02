#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from pathlib import Path

import copick
import numpy as np
import pandas as pd

from evaluation import (
    BETA,
    evaluate_at_threshold,
    precision_recall_curve,
    average_precision,
)

# ---------------------------------------------------------------------------
# Checkpoint URL (from the official QuickStart notebook)
# ---------------------------------------------------------------------------
CHECKPOINT_URL = (
    "https://huggingface.co/kevinzhao/TopCUP/resolve/main"
    "/topcup_weights/topcup_phantom_24_tomograms.ckpt"
)
CHECKPOINT_FILENAME = "topcup_phantom_24_tomograms.ckpt"

# Per-class evaluation radii (Å) and default score thresholds — taken from
# the phantom training config in the official QuickStart notebook.
PARTICLE_RADII = {
    "ferritin-complex":    60.0,
    "beta-amylase":        65.0,
    "beta-galactosidase":  90.0,
    "cytosolic-ribosome": 150.0,
    "thyroglobulin":      130.0,
    "virus-like-capsid":  135.0,
}
DEFAULT_THRESHOLDS = {
    "ferritin-complex":    0.16,
    "beta-amylase":        0.25,
    "beta-galactosidase":  0.13,
    "cytosolic-ribosome":  0.19,
    "thyroglobulin":       0.18,
    "virus-like-capsid":   0.50,
}
EVAL_RADIUS_MULT = 0.5   # competition: matching radius = 0.5 × particle_radius


# ---------------------------------------------------------------------------
# Checkpoint download
# ---------------------------------------------------------------------------

def ensure_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dest = checkpoint_dir / CHECKPOINT_FILENAME
    if dest.exists():
        print(f"[checkpoint] Already present: {dest}")
        return dest
    print(f"[checkpoint] Downloading {CHECKPOINT_FILENAME} from Hugging Face ...")
    urllib.request.urlretrieve(CHECKPOINT_URL, dest)
    print(f"[checkpoint] Saved to {dest}")
    return dest


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _zero_threshold_config(src: Path) -> Path:
    """Return path to a temp copick config with all score_thresholds set to 0.0.
    Caller is responsible for deleting the file when done."""
    data = json.loads(src.read_text())
    for obj in data.get("pickable_objects", []):
        if isinstance(obj.get("metadata"), dict):
            obj["metadata"]["score_threshold"] = 0.0
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    return Path(path)


def run_inference(
    copick_config: Path,
    checkpoint: Path,
    output_dir: Path,
    run_id: str,
    pixelsize: float,
    tomo_type: str,
    batch_size: int = 1,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = output_dir / "val_pred_df_seed.csv"

    if raw_csv.exists():
        print(f"[inference] Predictions already cached at {raw_csv}, skipping.")
        return raw_csv

    from topcup.cli.cli import cli
    tmp_config = _zero_threshold_config(copick_config)
    try:
        cli.main(args=[
            "inference",
            "-c", str(tmp_config),
            "-ts", run_id,
            "-p", str(checkpoint.parent),
            "--pixelsize", str(pixelsize),
            "-tt", tomo_type,
            "-o", str(output_dir),
            "-pa", checkpoint.name,
            "-bs", str(batch_size),
        ], standalone_mode=False)
    finally:
        tmp_config.unlink(missing_ok=True)

    return raw_csv


# ---------------------------------------------------------------------------
# Ground-truth loading (via copick API — no hardcoded directory paths)
# ---------------------------------------------------------------------------

def load_gt(copick_config: Path, run_id: str, gt_user_id: str) -> pd.DataFrame:
    root = copick.from_file(str(copick_config))
    run  = root.get_run(run_id)
    rows = []
    # Track (user_id, session_id, object_name) to avoid loading the same picks
    # file twice when overlay_root == static_root (copick exposes picks from both).
    seen_sessions: set[tuple] = set()
    for picks in run.picks:
        if picks.user_id != gt_user_id:
            continue
        key = (picks.user_id, picks.session_id, picks.pickable_object_name)
        if key in seen_sessions:
            continue
        seen_sessions.add(key)
        pts = picks.load()
        if pts is None:
            continue
        obj = picks.pickable_object_name
        for p in pts.points:
            rows.append({"experiment": run_id, "particle_type": obj,
                         "x": p.location.x, "y": p.location.y, "z": p.location.z})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Task 2b: TopCUP inference + quality metrics")
    p.add_argument("--copick-config",  required=True,
                   help="Path to the filesystem copick config JSON")
    p.add_argument("--output-dir",     default="./output/inference_2b",
                   help="Directory for predictions and metric CSVs (default: ./output/inference_2b)")
    p.add_argument("--checkpoint-dir", default="./checkpoints",
                   help="Directory to store/find the .ckpt file (default: ./checkpoints)")
    p.add_argument("--run-id",         default="16463",
                   help="Copick run name to run inference on (default: 16463)")
    p.add_argument("--pixelsize",      type=float, default=10.012,
                   help="Voxel size in Å (default: 10.012)")
    p.add_argument("--tomo-type",      default="wbp-denoised-denoiset-ctfdeconv",
                   help="Tomogram type string in the copick config")
    p.add_argument("--gt-user-id",     default="data-portal",
                   help="user_id of the ground-truth picks (default: data-portal)")
    p.add_argument("--batch-size",     type=int, default=1,
                   help="Sliding-window batch size passed to the data loader (default: 1). "
                        "Higher values use more memory but speed up inference on GPU.")
    return p.parse_args()


def main():
    args = parse_args()

    copick_config  = Path(args.copick_config)
    output_dir     = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    run_id         = args.run_id
    pixelsize      = args.pixelsize
    tomo_type      = args.tomo_type
    gt_user_id     = args.gt_user_id
    batch_size     = args.batch_size

    # ── Step 1: ensure checkpoint ──────────────────────────────────────────
    print("=" * 65)
    print("Step 1: Checkpoint")
    print("=" * 65)
    checkpoint = ensure_checkpoint(checkpoint_dir)

    # ── Step 2: run inference ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"Step 2: Inference on run {run_id}")
    print("=" * 65)
    raw_csv = run_inference(copick_config, checkpoint, output_dir,
                            run_id, pixelsize, tomo_type, batch_size)

    # ── Step 3: load data ──────────────────────────────────────────────────
    pred_df = pd.read_csv(raw_csv)
    gt_df   = load_gt(copick_config, run_id, gt_user_id)

    print(f"\nRaw predictions: {len(pred_df)} (all confidence levels)")
    print(f"GT annotations:  {len(gt_df)}")
    print("\nGT counts per class:")
    print(gt_df["particle_type"].value_counts().to_string())

    # ── Step 4: per-class evaluation at default threshold ─────────────────
    print("\n" + "=" * 65)
    print("Step 3: Quality assessment at default thresholds")
    print("=" * 65)

    results = []
    for ptype, radius in PARTICLE_RADII.items():
        th = DEFAULT_THRESHOLDS[ptype]
        results.append(evaluate_at_threshold(pred_df, gt_df, th, ptype, radius * EVAL_RADIUS_MULT))

    res_df = pd.DataFrame(results).set_index("particle_type")
    cols = ["threshold","n_gt","n_pred","tp","fp","fn","precision","recall",
            f"f{BETA}","avg_tp_dist_A"]
    print(res_df[cols].to_string(float_format=lambda x: f"{x:.3f}"))

    # ── Step 5: ribosome P/R sweep ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 4: Ribosome — precision/recall sweep")
    print("=" * 65)

    thresholds  = np.arange(0.05, 0.70, 0.01)
    ribo_radius = PARTICLE_RADII["cytosolic-ribosome"] * EVAL_RADIUS_MULT
    pr_df       = precision_recall_curve(pred_df, gt_df, "cytosolic-ribosome",
                                         ribo_radius, thresholds)
    ap          = average_precision(pr_df)
    best        = pr_df.loc[pr_df[f"f{BETA}"].idxmax()]

    print(f"\nBest threshold:   {best['threshold']:.2f}")
    print(f"  Precision:      {best['precision']:.3f}")
    print(f"  Recall:         {best['recall']:.3f}")
    print(f"  F{BETA}:          {best[f'f{BETA}']:.3f}")
    print(f"  TP dist (mean): {best['avg_tp_dist_A']:.1f} Å")
    print(f"  TP/FP/FN:       {int(best['tp'])}/{int(best['fp'])}/{int(best['fn'])}")
    print(f"\nAverage Precision (AP): {ap:.3f}")

    # ── Step 6: all-class AP summary ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 5: Per-class AP summary")
    print("=" * 65)
    for ptype, radius in PARTICLE_RADII.items():
        curve  = precision_recall_curve(pred_df, gt_df, ptype,
                                        radius * EVAL_RADIUS_MULT, thresholds)
        ap_i   = average_precision(curve)
        best_i = curve.loc[curve[f"f{BETA}"].idxmax()]
        print(f"  {ptype:<22s}  AP={ap_i:.3f}  "
              f"best-F{BETA}={best_i[f'f{BETA}']:.3f} @ th={best_i['threshold']:.2f}  "
              f"avg_dist={best_i['avg_tp_dist_A']:.1f} Å")

    # ── Save ───────────────────────────────────────────────────────────────
    res_df.to_csv(output_dir / "results_default_threshold.csv")
    pr_df.to_csv(output_dir / "ribosome_pr_curve.csv", index=False)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()

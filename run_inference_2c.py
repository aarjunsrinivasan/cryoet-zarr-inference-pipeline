#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request
import warnings
from pathlib import Path

import copick
import numpy as np
import pandas as pd
import zarr

from evaluation import (
    BETA,
    evaluate_at_threshold,
    precision_recall_curve,
    average_precision,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_URL = (
    "https://huggingface.co/kevinzhao/TopCUP/resolve/main"
    "/topcup_weights/topcup_phantom_24_tomograms.ckpt"
)
CHECKPOINT_FILENAME = "topcup_phantom_24_tomograms.ckpt"

# Chunk definition (voxel coordinates in the full run-14070 zarr)
CHUNK_X_OFFSET = 160    # voxels
CHUNK_Y_OFFSET = 752    # voxels
CHUNK_SIZE_XY  = 256    # voxels

# Particle metadata — must match the phantom training order (channels 0-5).
# These values come from the official QuickStart example config.
PICKABLE_OBJECTS = [
    {"name": "ferritin-complex",   "label": 1, "radius": 60.0,  "is_particle": True, "color": [0,255,0,255],     "metadata": {"score_weight": 1, "score_threshold": 0.0, "class_loss_weight": 256}},
    {"name": "beta-amylase",       "label": 2, "radius": 65.0,  "is_particle": True, "color": [233,1,56,255],    "metadata": {"score_weight": 0, "score_threshold": 0.0, "class_loss_weight": 256}},
    {"name": "beta-galactosidase", "label": 3, "radius": 90.0,  "is_particle": True, "color": [94,6,164,255],    "metadata": {"score_weight": 2, "score_threshold": 0.0, "class_loss_weight": 256}},
    {"name": "cytosolic-ribosome", "label": 4, "radius": 150.0, "is_particle": True, "color": [127,191,127,255], "metadata": {"score_weight": 1, "score_threshold": 0.0, "class_loss_weight": 256}},
    {"name": "thyroglobulin",      "label": 5, "radius": 130.0, "is_particle": True, "color": [255,0,255,255],   "metadata": {"score_weight": 2, "score_threshold": 0.0, "class_loss_weight": 256}},
    {"name": "virus-like-capsid",  "label": 6, "radius": 135.0, "is_particle": True, "color": [255,127,0,255],   "metadata": {"score_weight": 1, "score_threshold": 0.0, "class_loss_weight": 256}},
    {"name": "nucleosome",         "label": 7, "radius": 50.0,  "is_particle": True, "color": [0,150,255,255],   "metadata": {"score_weight": 0, "score_threshold": 0.0, "class_loss_weight": 256}},
]

PARTICLE_RADII = {o["name"]: o["radius"] for o in PICKABLE_OBJECTS}
# Same thresholds as Task 2b so the side-by-side comparison is at the same operating point.
# (PICKABLE_OBJECTS has score_threshold=0.0 so topcup outputs all predictions unfiltered;
#  these thresholds are applied here in Python during evaluation.)
DEFAULT_THRESHOLDS = {
    "ferritin-complex":    0.16,
    "beta-amylase":        0.25,
    "beta-galactosidase":  0.13,
    "cytosolic-ribosome":  0.19,
    "thyroglobulin":       0.18,
    "virus-like-capsid":   0.50,
    "nucleosome":          0.50,
}
EVAL_RADIUS_MULT = 0.5


# ---------------------------------------------------------------------------
# Checkpoint
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
# Chunk extraction and copick setup
# ---------------------------------------------------------------------------

def extract_chunk_zarr(
    src_zarr: Path,
    dst_zarr: Path,
    x_offset: int,
    y_offset: int,
    size_xy: int,
) -> np.ndarray:
    """Slice (z_full, y_offset:y_offset+size_xy, x_offset:x_offset+size_xy) from src zarr,
    save as a new zarr at dst_zarr, and return the numpy array (Z, Y, X)."""
    if dst_zarr.exists():
        print(f"[chunk] Zarr already extracted at {dst_zarr}")
        store = zarr.open(str(dst_zarr), mode="r")
        return store["0"][:]

    print(f"[chunk] Extracting chunk from {src_zarr} …")
    src = zarr.open(str(src_zarr), mode="r")["0"]          # (Z, Y, X)
    chunk = src[:, y_offset:y_offset + size_xy, x_offset:x_offset + size_xy]  # (Z, 256, 256)
    chunk_np = chunk[:]

    dst_zarr.mkdir(parents=True, exist_ok=True)
    # Mirror the OME-Zarr structure expected by copick (array at group "0")
    dst = zarr.open_group(str(dst_zarr), mode="w")
    dst.create_dataset("0", data=chunk_np, chunks=(64, 64, 64), dtype=chunk_np.dtype)
    print(f"[chunk] Saved chunk shape {chunk_np.shape} → {dst_zarr}")
    return chunk_np


def build_chunk_copick_config(
    data_root: Path,
    chunk_run_id: str,
    pixelsize: float,
    tomo_type: str,
) -> Path:
    """Write a minimal filesystem copick config for the chunk run."""
    config_path = data_root / "copick_config_chunk.json"
    if config_path.exists():
        return config_path

    voxel_dir = f"{pixelsize:.3f}"
    config = {
        "name": "Chlamydomonas chunk",
        "description": f"Run 14070 chunk x={CHUNK_X_OFFSET} y={CHUNK_Y_OFFSET} size={CHUNK_SIZE_XY}",
        "version": "1.0.0",
        "pickable_objects": PICKABLE_OBJECTS,
        "config_type": "filesystem",
        "overlay_root": f"local://{data_root.resolve()}",
        "static_root":  f"local://{data_root.resolve()}",
        "overlay_fs_args": {"auto_mkdir": True},
    }
    config_path.write_text(json.dumps(config, indent=4))
    print(f"[copick] Chunk config written to {config_path}")
    return config_path


def save_chunk_gt_picks(
    full_run_picks_dir: Path,
    chunk_data_root: Path,
    chunk_run_id: str,
    pixelsize: float,
    x_offset: int,
    y_offset: int,
    size_xy: int,
    gt_user_id: str,
) -> int:
    """Filter GT ribosome picks to those inside the chunk; store in chunk-local coords."""
    out_dir = chunk_data_root / "ExperimentRuns" / chunk_run_id / "Picks"
    out_dir.mkdir(parents=True, exist_ok=True)

    x_min = x_offset * pixelsize
    x_max = (x_offset + size_xy) * pixelsize
    y_min = y_offset * pixelsize
    y_max = (y_offset + size_xy) * pixelsize
    x_off_A = x_min
    y_off_A = y_min

    total_in_chunk = 0
    picks_files = list(full_run_picks_dir.glob(f"{gt_user_id}_*_*.json"))
    for jf in picks_files:
        raw = json.loads(jf.read_text())
        obj = raw.get("pickable_object_name", jf.stem.split("_", 2)[-1])

        in_chunk = []
        for pt in raw.get("points", []):
            loc = pt["location"]
            if x_min <= loc["x"] <= x_max and y_min <= loc["y"] <= y_max:
                # Convert to chunk-local coordinates
                in_chunk.append({**pt, "location": {
                    "x": loc["x"] - x_off_A,
                    "y": loc["y"] - y_off_A,
                    "z": loc["z"],
                }})

        if not in_chunk:
            continue
        total_in_chunk += len(in_chunk)
        out_raw = {**raw, "points": in_chunk, "run_name": chunk_run_id}
        out_path = out_dir / jf.name
        if not out_path.exists():
            out_path.write_text(json.dumps(out_raw, indent=2))
            print(f"[picks] {obj}: {len(in_chunk)} picks in chunk → {out_path.name}")
        else:
            print(f"[picks] {obj}: {len(in_chunk)} picks already saved")

    return total_in_chunk


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    copick_config: Path,
    checkpoint: Path,
    output_dir: Path,
    run_id: str,
    pixelsize: float,
    tomo_type: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = output_dir / "val_pred_df_seed.csv"

    if raw_csv.exists():
        print(f"[inference] Predictions already cached at {raw_csv}, skipping.")
        return raw_csv

    from topcup.cli.cli import cli
    cli.main(args=[
        "inference",
        "-c", str(copick_config),
        "-ts", run_id,
        "-p", str(checkpoint.parent),
        "--pixelsize", str(pixelsize),
        "-tt", tomo_type,
        "-o", str(output_dir),
        "-pa", checkpoint.name,
    ], standalone_mode=False)

    return raw_csv




def load_gt_from_copick(config_path: Path, run_id: str, gt_user_id: str) -> pd.DataFrame:
    root = copick.from_file(str(config_path))
    run  = root.get_run(run_id)
    rows, seen = [], set()
    for picks in run.picks:
        if picks.user_id != gt_user_id:
            continue
        key = (picks.user_id, picks.session_id, picks.pickable_object_name)
        if key in seen:
            continue
        seen.add(key)
        pts = picks.load()
        if pts is None:
            continue
        for p in pts.points:
            rows.append({"experiment": run_id, "particle_type": picks.pickable_object_name,
                         "x": p.location.x, "y": p.location.y, "z": p.location.z})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Task 2c: Chlamydomonas chunk inference + failure analysis")
    p.add_argument("--copick-config",  required=True,
                   help="Copick config pointing to run-14070 data (has the full-res tomogram)")
    p.add_argument("--output-dir",     default="./output/inference_2c")
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--run-id",         default="14070")
    p.add_argument("--pixelsize",      type=float, default=7.84)
    p.add_argument("--tomo-type",      default="wbp-denoised-ctfdeconv")
    p.add_argument("--gt-user-id",     default="data-portal")
    p.add_argument("--results-2b",     default="./output/inference_2b/results_default_threshold.csv",
                   help="Path to Task 2b per-class metrics CSV for comparison")
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
    results_2b_csv = Path(args.results_2b)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive paths from the main copick config location
    data_root = copick_config.parent

    CHUNK_RUN_ID  = f"{run_id}_chunk"
    voxel_dir_str = f"{pixelsize:.3f}"

    src_zarr = (data_root / "ExperimentRuns" / run_id
                / f"VoxelSpacing{voxel_dir_str}" / f"{tomo_type}.zarr")
    chunk_zarr = (data_root / "ExperimentRuns" / CHUNK_RUN_ID
                  / f"VoxelSpacing{voxel_dir_str}" / f"{tomo_type}.zarr")
    chunk_picks_src = data_root / "ExperimentRuns" / run_id / "Picks"

    # ── Step 1: extract chunk ──────────────────────────────────────────────
    print("=" * 65)
    print("Step 1: Extract tomogram chunk")
    print("=" * 65)
    chunk_np = extract_chunk_zarr(src_zarr, chunk_zarr,
                                  CHUNK_X_OFFSET, CHUNK_Y_OFFSET, CHUNK_SIZE_XY)
    print(f"  Chunk shape (Z, Y, X): {chunk_np.shape}")

    # ── Step 2: build chunk copick config + GT picks ───────────────────────
    print("\n" + "=" * 65)
    print("Step 2: Copick config and GT picks for chunk")
    print("=" * 65)
    chunk_config = build_chunk_copick_config(data_root, CHUNK_RUN_ID, pixelsize, tomo_type)
    n_gt_ribo = save_chunk_gt_picks(
        chunk_picks_src, data_root, CHUNK_RUN_ID,
        pixelsize, CHUNK_X_OFFSET, CHUNK_Y_OFFSET, CHUNK_SIZE_XY, gt_user_id
    )
    print(f"  GT ribosomes in chunk: {n_gt_ribo}")

    # ── Step 3: checkpoint ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 3: Checkpoint")
    print("=" * 65)
    checkpoint = ensure_checkpoint(checkpoint_dir)

    # ── Step 4: inference ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"Step 4: Inference on chunk (pixelsize={pixelsize} Å)")
    print("=" * 65)
    raw_csv = run_inference(chunk_config, checkpoint, output_dir,
                            CHUNK_RUN_ID, pixelsize, tomo_type)

    # ── Step 5: load predictions + GT ─────────────────────────────────────
    pred_df = pd.read_csv(raw_csv)
    # Rename run_id in predictions from CHUNK_RUN_ID to match GT lookup
    pred_df["experiment"] = CHUNK_RUN_ID
    gt_df = load_gt_from_copick(chunk_config, CHUNK_RUN_ID, gt_user_id)

    print(f"\nRaw predictions: {len(pred_df)}")
    print(f"GT annotations:  {len(gt_df)}")
    print("\nGT counts:\n" + gt_df["particle_type"].value_counts().to_string())

    # ── Step 6: ribosome metrics ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 5: Quality assessment — cytosolic-ribosome")
    print("=" * 65)

    thresholds  = np.arange(0.05, 0.70, 0.01)
    ribo_radius = PARTICLE_RADII["cytosolic-ribosome"] * EVAL_RADIUS_MULT

    ribo_default = evaluate_at_threshold(
        pred_df, gt_df, DEFAULT_THRESHOLDS["cytosolic-ribosome"],
        "cytosolic-ribosome", ribo_radius
    )
    pr_df   = precision_recall_curve(pred_df, gt_df, "cytosolic-ribosome", ribo_radius, thresholds)
    ap      = average_precision(pr_df)
    best    = pr_df.loc[pr_df[f"f{BETA}"].idxmax()]

    print(f"\nAt default threshold ({DEFAULT_THRESHOLDS['cytosolic-ribosome']}):")
    print(f"  Precision={ribo_default['precision']:.3f}  "
          f"Recall={ribo_default['recall']:.3f}  "
          f"F{BETA}={ribo_default[f'f{BETA}']:.3f}  "
          f"avg_dist={ribo_default['avg_tp_dist_A']:.1f} Å")
    print(f"\nBest threshold: {best['threshold']:.2f}")
    print(f"  Precision={best['precision']:.3f}  Recall={best['recall']:.3f}  "
          f"F{BETA}={best[f'f{BETA}']:.3f}  avg_dist={best['avg_tp_dist_A']:.1f} Å")
    print(f"  TP/FP/FN: {int(best.tp)}/{int(best.fp)}/{int(best.fn)}")
    print(f"\nAUC-PR: {ap:.3f}")

    # ── Step 7: side-by-side comparison with Task 2b ───────────────────────
    print("\n" + "=" * 65)
    print("Step 6: Comparison — synthetic (2b) vs in-situ (2c)")
    print("=" * 65)

    ribo_2c = {
        "dataset": "in-situ (14070 chunk)",
        "n_gt": ribo_default["n_gt"],
        f"f{BETA}_default": ribo_default[f"f{BETA}"],
        "precision_default": ribo_default["precision"],
        "recall_default": ribo_default["recall"],
        "avg_dist_default": ribo_default["avg_tp_dist_A"],
        f"best_f{BETA}": best[f"f{BETA}"],
        "best_threshold": best["threshold"],
        "AUC-PR": ap,
    }

    if results_2b_csv.exists():
        df2b = pd.read_csv(results_2b_csv, index_col=0)
        if "cytosolic-ribosome" in df2b.index:
            r2b = df2b.loc["cytosolic-ribosome"]
            ribo_2b = {
                "dataset": "synthetic (16463)",
                "n_gt": int(r2b.n_gt),
                f"f{BETA}_default": r2b[f"f{BETA}"],
                "precision_default": r2b.precision,
                "recall_default": r2b.recall,
                "avg_dist_default": r2b.avg_tp_dist_A,
                f"best_f{BETA}": "—",
                "best_threshold": r2b.threshold,
                "AUC-PR": "—",
            }
            print(pd.DataFrame([ribo_2b, ribo_2c]).set_index("dataset").to_string())
        else:
            print("(Task 2b results file found but cytosolic-ribosome row missing)")
            print(pd.DataFrame([ribo_2c]).set_index("dataset").to_string())
    else:
        print(f"(Task 2b results not found at {results_2b_csv} — showing 2c only)")
        print(pd.DataFrame([ribo_2c]).set_index("dataset").to_string())

    # Save metric CSVs
    pr_df.to_csv(output_dir / "ribosome_pr_curve.csv", index=False)
    pd.DataFrame([ribo_default]).to_csv(output_dir / "results_default_threshold.csv", index=False)
    print(f"\nOutputs saved to {output_dir}/")


if __name__ == "__main__":
    main()

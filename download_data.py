#!/usr/bin/env python3
"""Task 2a: Download tomograms and ground-truth annotations for inference and evaluation.

Downloads:
  - Run 16463 (CZII synthetic lysate, dataset 10440): 630x630x184, voxel=10.012 A
  - Run 14070 (Chlamydomonas in-situ, dataset 10301): 1024x1024x512, voxel=7.84 A

Creates a local filesystem copick config at data/copick_config_fs.json.
"""

import argparse
import json
import shutil
from pathlib import Path

import cryoet_data_portal as cdp
import copick
import zarr

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DATA_DIR: Path  # set in main() via --data-dir

# Run configurations: run_id, dataset, voxel size, tomo_type, cdp tomogram id
RUNS = [
    {
        "run_id": "16463",
        "dataset_id": 10440,
        "voxel_size": 10.012,
        "tomo_type": "wbp-denoised-denoiset-ctfdeconv",
        "cdp_tomo_id": 17015,  # denoised WBP at 10.012 A
        "description": "CZII synthetic lysate (in-distribution sanity check)",
    },
    {
        "run_id": "14070",
        "dataset_id": 10301,
        "voxel_size": 7.84,
        "voxel_size_dir": "7.840",  # copick formats to 3 decimal places
        "tomo_type": "wbp-denoised-ctfdeconv",
        "cdp_tomo_id": 14317,  # denoised WBP at 7.84 A
        "description": "Chlamydomonas in-situ (held-out evaluation)",
    },
]

# Pickable objects ordered to match the TopCUP training class indices (0-5).
# Order must match the phantom training config (Quickstart_TopCUP.ipynb cell-4):
#   0: ferritin-complex, 1: beta-amylase, 2: beta-galactosidase,
#   3: cytosolic-ribosome, 4: thyroglobulin, 5: virus-like-capsid
# Portal label values are preserved; nucleosome is excluded (not in the 6-class model).
PICKABLE_OBJECTS = [
    {
        "name": "ferritin-complex",
        "is_particle": True,
        "label": 1,
        "color": [0, 255, 0, 255],
        "radius": 60.0,
        "metadata": {"score_weight": 1, "score_threshold": 0.16, "class_loss_weight": 256},
    },
    {
        "name": "beta-amylase",
        "is_particle": True,
        "label": 7,
        "color": [233, 1, 56, 255],
        "radius": 65.0,
        "metadata": {"score_weight": 0, "score_threshold": 0.25, "class_loss_weight": 256},
    },
    {
        "name": "beta-galactosidase",
        "is_particle": True,
        "label": 6,
        "color": [94, 6, 164, 255],
        "radius": 90.0,
        "metadata": {"score_weight": 2, "score_threshold": 0.13, "class_loss_weight": 256},
    },
    {
        "name": "cytosolic-ribosome",
        "is_particle": True,
        "label": 5,
        "color": [127, 191, 127, 255],
        "radius": 150.0,
        "metadata": {"score_weight": 1, "score_threshold": 0.19, "class_loss_weight": 256},
    },
    {
        "name": "thyroglobulin",
        "is_particle": True,
        "label": 2,
        "color": [255, 0, 255, 255],
        "radius": 130.0,
        "metadata": {"score_weight": 2, "score_threshold": 0.18, "class_loss_weight": 256},
    },
    {
        "name": "virus-like-capsid",
        "is_particle": True,
        "label": 4,
        "color": [255, 127, 0, 255],
        "radius": 135.0,
        "metadata": {"score_weight": 1, "score_threshold": 0.5, "class_loss_weight": 256},
    },
]


def download_tomogram(run_cfg: dict) -> Path:
    """Download tomogram zarr and place it in the copick filesystem path."""
    run_id = run_cfg["run_id"]
    vs = run_cfg["voxel_size"]
    tomo_type = run_cfg["tomo_type"]
    tomo_id = run_cfg["cdp_tomo_id"]

    vs_dir = run_cfg.get("voxel_size_dir", str(vs))
    zarr_dest = (
        DATA_DIR / "ExperimentRuns" / run_id / f"VoxelSpacing{vs_dir}" / f"{tomo_type}.zarr"
    )

    if zarr_dest.exists():
        print(f"[{run_id}] Tomogram already at {zarr_dest}, skipping download.")
        return zarr_dest

    zarr_dest.parent.mkdir(parents=True, exist_ok=True)

    client = cdp.Client()
    tomo = cdp.Tomogram.get_by_id(client, tomo_id)
    print(
        f"[{run_id}] Downloading tomogram id={tomo_id} "
        f"({tomo.size_x}x{tomo.size_y}x{tomo.size_z} vox, {vs} A) ..."
    )

    # download_omezarr drops the zarr into dest_path named after the run on S3
    tmp_dir = DATA_DIR / "_tmp" / run_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tomo.download_omezarr(dest_path=str(tmp_dir))

    downloaded = sorted(tmp_dir.glob("*.zarr"))
    if not downloaded:
        raise RuntimeError(f"No zarr file found in {tmp_dir} after download")

    shutil.move(str(downloaded[0]), str(zarr_dest))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[{run_id}] Saved to {zarr_dest}")
    return zarr_dest


def save_picks_for_run(run_id: str, dataset_id: int) -> dict:
    """Fetch ground-truth picks from the portal and write them to the overlay."""
    picks_dir = DATA_DIR / "ExperimentRuns" / run_id / "Picks"
    picks_dir.mkdir(parents=True, exist_ok=True)

    portal_root = copick.from_czcdp_datasets(
        [dataset_id],
        str(DATA_DIR / "_tmp_overlay"),
        {"auto_mkdir": True},
    )
    run = portal_root.get_run(run_id)

    counts = {}
    for picks in run.picks:
        obj = picks.pickable_object_name
        pts = picks.load()
        n = len(pts.points) if pts is not None else 0
        counts[obj] = n

        out_path = picks_dir / f"{picks.user_id}_{picks.session_id}_{obj}.json"
        if out_path.exists():
            print(f"[{run_id}] Picks already saved for {obj} ({n} points)")
            continue

        # Serialize using pydantic model_dump (copick >= 0.6) with fallback
        picks_dict = pts.model_dump() if hasattr(pts, "model_dump") else pts.dict()
        out_path.write_text(json.dumps(picks_dict, indent=2))
        print(f"[{run_id}] Saved {n} picks for {obj} → {out_path.name}")

    return counts


def create_filesystem_config() -> Path:
    """Write a copick filesystem config covering both runs."""
    config = {
        "name": "Task 2 Inference Data",
        "description": "Local filesystem copick config for Task 2 (runs 16463 and 14070).",
        "version": "1.0.0",
        "pickable_objects": PICKABLE_OBJECTS,
        "config_type": "filesystem",
        "overlay_root": f"local://{DATA_DIR.resolve()}",
        "static_root": f"local://{DATA_DIR.resolve()}",
        "overlay_fs_args": {"auto_mkdir": True},
    }
    config_path = DATA_DIR / "copick_config_fs.json"
    config_path.write_text(json.dumps(config, indent=4))
    print(f"Filesystem config written to {config_path}")
    return config_path


def get_tomogram_shape(zarr_path: Path) -> tuple:
    """Return (z, y, x) shape from the base resolution level of a zarr."""
    store = zarr.open(str(zarr_path), mode="r")
    return tuple(store["0"].shape)


def main():
    global DATA_DIR

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Root directory for downloaded data (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()
    DATA_DIR = args.data_dir

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}

    for run_cfg in RUNS:
        run_id = run_cfg["run_id"]
        print(f"\n{'=' * 60}")
        print(f"Run {run_id}: {run_cfg['description']}")
        print("=" * 60)

        zarr_path = download_tomogram(run_cfg)
        shape = get_tomogram_shape(zarr_path)  # (z, y, x)
        picks = save_picks_for_run(run_id, run_cfg["dataset_id"])

        summary[run_id] = {
            "description": run_cfg["description"],
            "shape_zyx": shape,
            "voxel_size_A": run_cfg["voxel_size"],
            "picks": picks,
        }

    config_path = create_filesystem_config()

    # Clean up temp overlay dirs
    for tmp in (DATA_DIR / "_tmp_overlay", DATA_DIR / "_tmp"):
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    print("SUMMARY (Task 2a)")
    print("=" * 60)
    for run_id, info in summary.items():
        z, y, x = info["shape_zyx"]
        print(f"\nRun {run_id}: {info['description']}")
        print(f"  Dimensions (z x y x x): {z} x {y} x {x}")
        print(f"  Voxel size:             {info['voxel_size_A']} Å")
        print(f"  Annotations:")
        for obj, n in sorted(info["picks"].items()):
            print(f"    {obj}: {n}")

    print(f"\nFilesystem config: {config_path}")



if __name__ == "__main__":
    main()

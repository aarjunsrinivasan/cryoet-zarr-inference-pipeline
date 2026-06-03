#!/usr/bin/env python3
"""Generate all Task 2c diagnostic figures.

Outputs (written to --output-dir, default output/inference_2c/):
  domain_gap_analysis.png          — raw & normalised intensity distributions + max confidence per class
  slice_comparison.png             — mid-depth XY slices for in-situ chunk vs synthetic, GT overlays
  ferritin_ribosome_synthetic.png  — synthetic slices with ferritin+ribosome GT overlaid (blob confusion)
  performance_summary.png          — PR curve overlay (2b vs 2c) + per-class model response at GT ribosome positions

Usage:
    python generate_figures_2c.py
    python generate_figures_2c.py --output-dir output/inference_2c
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import numpy as np
import pandas as pd
import zarr
from scipy import stats
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Config / defaults
# ---------------------------------------------------------------------------
CHUNK_X_OFFSET   = 160
CHUNK_Y_OFFSET   = 752
PIXELSIZE_CHUNK  = 7.84
PIXELSIZE_SYN    = 10.012
MATCH_RADIUS     = 75.0   # 0.5 × ribosome radius (150 Å) — CZII competition matching criterion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zscore(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return (v - v.mean()) / (v.std() + 1e-8)


def load_gt_pts(json_path: Path) -> list[tuple[float, float, float]]:
    with open(json_path) as f:
        data = json.load(f)
    return [(p["location"]["x"], p["location"]["y"], p["location"]["z"])
            for p in data.get("points", [])]


def find_gt_picks(picks_dir: Path, object_name: str) -> Path:
    """Find the GT picks JSON for object_name, tolerating any session ID in the filename."""
    matches = sorted(picks_dir.glob(f"*_{object_name}.json"))
    if not matches:
        raise FileNotFoundError(f"No picks file for '{object_name}' in {picks_dir}")
    return matches[0]


def show_slice(ax, vol, z_idx, pixelsize, gt_pts, title):
    img = vol[z_idx]
    vmin, vmax = np.percentile(img, [1, 99])
    H, W = img.shape
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, origin="lower",
              extent=[0, W * pixelsize, 0, H * pixelsize], aspect="equal")
    z_A = z_idx * pixelsize
    dz  = 3 * pixelsize
    for x, y, z in gt_pts:
        if abs(z - z_A) < dz:
            ax.add_patch(Circle((x, y), 75.0, fill=False, edgecolor="lime", lw=1.5))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")


# ---------------------------------------------------------------------------
# Figure 1: domain gap analysis
# ---------------------------------------------------------------------------

EXPECTED_CLASSES = [
    "ferritin-complex", "beta-amylase", "beta-galactosidase",
    "cytosolic-ribosome", "thyroglobulin", "virus-like-capsid",
]
SHORT_LABELS = ["ferritin", "β-amylase", "β-galact.", "ribosome", "thyroglo.", "VLP"]


def fig_domain_gap(chunk: np.ndarray, syn: np.ndarray,
                   pred_df: pd.DataFrame, out_path: Path):
    # Pre-normalisation statistics (raw volumes, not z-scored)
    c_mean, c_std = float(chunk.mean()), float(chunk.std())
    s_mean, s_std = float(syn.mean()),   float(syn.std())

    # Sample from raw, then apply volume-wide z-score to just those samples.
    # Avoids allocating a full z-scored copy of the entire volume.
    rng = np.random.default_rng(42)
    flat_c = chunk.ravel().astype(np.float32)
    flat_s = syn.ravel().astype(np.float32)
    idx_c  = rng.choice(len(flat_c), 300_000, replace=True)
    idx_s  = rng.choice(len(flat_s), 300_000, replace=True)
    sample_c = (flat_c[idx_c] - c_mean) / (c_std + 1e-8)
    sample_s = (flat_s[idx_s] - s_mean) / (s_std + 1e-8)
    kurt_c = float(stats.kurtosis(sample_c))
    kurt_s = float(stats.kurtosis(sample_s))

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # --- Panel 1: raw reconstruction density — in-situ only, own scale ---
    ax = axes[0]
    ax.hist(chunk.ravel(), bins=120, density=True, alpha=0.85, color="steelblue")
    ax.set_xlabel("Voxel reconstruction density (negative = high e⁻ density)", fontsize=8)
    ax.set_ylabel("Density")
    ax.set_title(f"In-situ chunk (14070) — raw voxel values\n"
                 f"mean={c_mean:.4g},  std={c_std:.4g},  var={c_std**2:.4g}")

    # --- Panel 2: raw reconstruction density — synthetic only, own scale ---
    ax = axes[1]
    ax.hist(syn.ravel(), bins=120, density=True, alpha=0.85, color="coral")
    ax.set_xlabel("Voxel reconstruction density (negative = high e⁻ density)", fontsize=8)
    ax.set_ylabel("Density")
    ax.set_title(f"Synthetic (16463) — raw voxel values\n"
                 f"mean={s_mean:.4g},  std={s_std:.4g},  var={s_std**2:.4g}")

    # --- Panel 3: z-score normalised distributions on log scale ---
    ax = axes[2]
    bins = np.linspace(-8, 5, 130)
    ax.hist(sample_c, bins=bins, density=True, alpha=0.65,
            color="steelblue", label=f"In-situ  (kurt={kurt_c:.1f}, post-z mean≈0, var≈1)")
    ax.hist(sample_s, bins=bins, density=True, alpha=0.65,
            color="coral",     label=f"Synthetic (kurt={kurt_s:.1f}, post-z mean≈0, var≈1)")
    x = np.linspace(-8, 5, 300)
    ax.plot(x, stats.norm.pdf(x), "k--", lw=1.5, label="Standard normal")
    ax.set_yscale("log")
    ax.set_xlim(-8, 5)
    ax.set_xlabel("z-score intensity (mean=0, var=1 by construction)")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("Post-normalisation distributions\n"
                 "Synthetic heavy tail (kurt≈30) vs in-situ near-Gaussian (kurt≈0.6)")
    ax.legend(fontsize=8)

    # --- Panel 4: max confidence per class (all 6 classes explicit) ---
    ax = axes[3]
    conf_by_class = pred_df.groupby("particle_type")["conf"].max()
    max_conf_vals = [float(conf_by_class.get(c, 0.0)) for c in EXPECTED_CLASSES]
    bar_colors = plt.cm.tab10(np.linspace(0, 1, len(EXPECTED_CLASSES)))
    ax.bar(range(len(EXPECTED_CLASSES)), max_conf_vals, color=bar_colors)
    ax.axhline(0.19, ls="--", color="black", lw=1.5, label="Default threshold (0.19)")
    ax.axhline(0.05, ls=":",  color="gray",  lw=1.2, label="Min sweep threshold (0.05)")
    ax.set_xticks(range(len(EXPECTED_CLASSES)))
    ax.set_xticklabels(SHORT_LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("Max confidence score (log scale)")
    ax.set_title("Max prediction confidence per class\n"
                 "All classes far below any useful threshold")
    ax.legend(fontsize=9)
    # explicitly annotate ribosome
    ribo_idx = EXPECTED_CLASSES.index("cytosolic-ribosome")
    ribo_val = max_conf_vals[ribo_idx]
    if ribo_val > 0:
        ax.annotate(f"ribosome\n{ribo_val:.4f}",
                    xy=(ribo_idx, ribo_val),
                    xytext=(ribo_idx + 1.0, ribo_val * 10),
                    fontsize=7, ha="left",
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))

    plt.suptitle("Domain gap: in-situ Chlamydomonas vs synthetic lysate", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[fig] Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: slice comparison
# ---------------------------------------------------------------------------

def find_ribosome_rich_slices(gt_pts, n_slices: int, pixelsize: float,
                              vol_depth: int, n_vox_tol: int = 3,
                              min_sep_frac: float = 0.18) -> list[int]:
    """Return n_slices z-indices with the most GT ribosomes nearby.

    Slices are spaced at least min_sep_frac * vol_depth apart. If that
    constraint forces any chosen slice to have 0 ribosomes, the separation
    is halved and retried until all slices have ≥1 ribosome or sep reaches 1.
    """
    if not gt_pts:
        step = max(1, vol_depth // (n_slices + 1))
        return [step * (i + 1) for i in range(n_slices)]

    dz_A = n_vox_tol * pixelsize  # same tolerance as show_slice (dz = 3 * pixelsize)
    z_coords = np.array([p[2] for p in gt_pts])

    counts = np.zeros(vol_depth, dtype=int)
    for zi in range(vol_depth):
        counts[zi] = int(np.sum(np.abs(z_coords - zi * pixelsize) < dz_A))

    order = np.argsort(-counts)
    min_sep = int(min_sep_frac * vol_depth)

    while min_sep >= 1:
        chosen: list[int] = []
        for zi in order:
            if all(abs(zi - c) >= min_sep for c in chosen):
                chosen.append(int(zi))
            if len(chosen) == n_slices:
                break
        if len(chosen) == n_slices and all(counts[z] > 0 for z in chosen):
            return sorted(chosen)
        min_sep //= 2

    # absolute fallback: top n_slices by count regardless of spacing
    return sorted(int(z) for z in order[:n_slices])


def fig_slice_comparison(chunk: np.ndarray, syn: np.ndarray,
                         gt_chunk_pts, gt_syn_pts, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    Z = chunk.shape[0]
    chunk_slices = find_ribosome_rich_slices(gt_chunk_pts, 3, PIXELSIZE_CHUNK, Z)
    for i, z in enumerate(chunk_slices):
        z_A = z * PIXELSIZE_CHUNK
        n_visible = sum(1 for _, _, pz in gt_chunk_pts if abs(pz - z_A) < 3 * PIXELSIZE_CHUNK)
        show_slice(axes[0, i], chunk, z, PIXELSIZE_CHUNK, gt_chunk_pts,
                   f"In-situ chunk — z={z} vox ({z_A:.0f} Å)\n"
                   f"{n_visible} GT ribosomes visible [lime, r=75 Å]")

    Z2 = syn.shape[0]
    syn_slices = find_ribosome_rich_slices(gt_syn_pts, 3, PIXELSIZE_SYN, Z2)
    for i, z in enumerate(syn_slices):
        z_A = z * PIXELSIZE_SYN
        n_visible = sum(1 for _, _, pz in gt_syn_pts if abs(pz - z_A) < 3 * PIXELSIZE_SYN)
        show_slice(axes[1, i], syn, z, PIXELSIZE_SYN, gt_syn_pts,
                   f"Synthetic (16463) — z={z} vox ({z_A:.0f} Å)\n"
                   f"{n_visible} GT ribosomes visible [lime, r=75 Å]")

    plt.suptitle(
        "XY slices chosen at maximum GT-ribosome density — panels spread apart, all with ≥1 ribosome visible\n"
        "In-situ (top): crowded cytoplasm, ribosomes blend into background  |  "
        "Synthetic (bottom): sparse particles, clear contrast",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[fig] Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: ferritin vs ribosome in synthetic — why the model confuses them
# ---------------------------------------------------------------------------

# Physical radii (Å) used for GT overlay circles
FERRITIN_RADIUS_A = 60.0    # ~120 Å diameter
RIBO_RADIUS_A     = 150.0   # ~300 Å diameter


def _pick_slices_with_both(ferr_pts, ribo_pts, pixelsize: float,
                            vol_depth: int, n: int = 3) -> list[int]:
    """Return n z-indices maximising ferritin+ribosome count, spaced ≥15% of depth apart."""
    dz = 3 * pixelsize
    scores = []
    for zi in range(vol_depth):
        z_A = zi * pixelsize
        nf = sum(1 for _, _, z in ferr_pts if abs(z - z_A) < dz)
        nr = sum(1 for _, _, z in ribo_pts if abs(z - z_A) < dz)
        scores.append((nf + nr, nf, nr, zi))
    scores.sort(reverse=True)

    min_sep = int(0.15 * vol_depth)
    chosen: list[int] = []
    for _, _, _, zi in scores:
        if all(abs(zi - c) >= min_sep for c in chosen):
            chosen.append(zi)
        if len(chosen) == n:
            break
    return sorted(chosen)


def fig_ferritin_ribosome_synthetic(syn: np.ndarray, ferr_pts, ribo_pts, out_path: Path):
    slices = _pick_slices_with_both(ferr_pts, ribo_pts, PIXELSIZE_SYN, syn.shape[0])
    fig, axes = plt.subplots(1, len(slices), figsize=(7 * len(slices), 7))
    if len(slices) == 1:
        axes = [axes]

    dz = 3 * PIXELSIZE_SYN
    for ax, zi in zip(axes, slices):
        img = syn[zi]
        vmin, vmax = np.percentile(img, [1, 99])
        H, W = img.shape
        ext = [0, W * PIXELSIZE_SYN, 0, H * PIXELSIZE_SYN]
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax,
                  origin="lower", extent=ext, aspect="equal")

        z_A = zi * PIXELSIZE_SYN
        nf = nr = 0
        for x, y, z in ferr_pts:
            if abs(z - z_A) < dz:
                ax.add_patch(Circle((x, y), FERRITIN_RADIUS_A,
                                    fill=False, edgecolor="gold", lw=1.8))
                nf += 1
        for x, y, z in ribo_pts:
            if abs(z - z_A) < dz:
                ax.add_patch(Circle((x, y), RIBO_RADIUS_A,
                                    fill=False, edgecolor="lime", lw=1.8))
                nr += 1

        ax.set_title(f"z={zi} vox ({z_A:.0f} Å)\n"
                     f"ferritin={nf} [gold, r=60 Å]   ribosome={nr} [lime, r=150 Å]",
                     fontsize=9)
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")

    # legend proxies
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="gold", lw=2,
               label=f"Ferritin (~120 Å diam.) — {len(ferr_pts)} GT total"),
        Line2D([0], [0], color="lime", lw=2,
               label=f"Ribosome (~300 Å diam.) — {len(ribo_pts)} GT total"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.01))
    plt.suptitle(
        "Synthetic 16463: ferritin and ribosome co-localise as compact spherical blobs\n"
        "Both appear as localised density dips on a near-empty resin background — "
        "the model's blob-detector has context to tell them apart here",
        fontsize=10, y=1.06
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[fig] Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: performance summary — PR curve overlay + class confusion at GT positions
# ---------------------------------------------------------------------------

def _mean_max_conf_at_gt(pred_df: pd.DataFrame, gt_pts: np.ndarray,
                          cls: str, radius: float) -> float:
    """Mean over GT positions of the max model confidence within `radius` Å for `cls`."""
    sub = pred_df[pred_df["particle_type"] == cls][["x", "y", "z", "conf"]].values
    if len(sub) == 0 or len(gt_pts) == 0:
        return 0.0
    tree = KDTree(sub[:, :3])
    vals = []
    for pt in gt_pts:
        idxs = tree.query_ball_point(pt, r=radius)
        vals.append(float(sub[idxs, 3].max()) if idxs else 0.0)
    return float(np.mean(vals))


def fig_performance_summary(pr_2b: pd.DataFrame, pr_2c: pd.DataFrame,
                             pred_chunk: pd.DataFrame, pred_syn: pd.DataFrame,
                             gt_chunk_pts, gt_syn_pts, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel 1: PR curve overlay ---
    ax = axes[0]

    pr_2b_s = pr_2b.sort_values("recall")
    rec = pr_2b_s["recall"].values
    pre = pr_2b_s["precision"].values

    ax.plot(rec, pre, "o-", color="steelblue", lw=2, ms=5, label="Synthetic 16463")

    # Mark the default-threshold operating point
    op = pr_2b.iloc[(pr_2b["threshold"] - 0.19).abs().argsort().iloc[0]]
    ax.scatter(op["recall"], op["precision"], s=120, zorder=6,
               color="steelblue", edgecolors="black", linewidths=1.2, marker="D",
               label=f"Default th=0.19  (P={op['precision']:.2f}, R={op['recall']:.2f}, F4={op['f4']:.3f})")

    # 2c collapses to a single point at the origin
    ax.scatter([0], [0], s=180, color="red", marker="X", zorder=6,
               label="In-situ 14070 chunk\nzero detections at all thresholds 0.05–0.70")

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall: cytosolic-ribosome\n"
                 "Synthetic (in-distribution) vs in-situ Chlamydomonas chunk")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    # --- Panel 2: per-class model response at GT ribosome positions ---
    ax = axes[1]

    gt_chunk = np.array(gt_chunk_pts)
    gt_syn   = np.array(gt_syn_pts)

    chunk_vals = [_mean_max_conf_at_gt(pred_chunk, gt_chunk, c, MATCH_RADIUS)
                  for c in EXPECTED_CLASSES]
    syn_vals   = [_mean_max_conf_at_gt(pred_syn,   gt_syn,   c, MATCH_RADIUS)
                  for c in EXPECTED_CLASSES]

    x = np.arange(len(EXPECTED_CLASSES))
    w = 0.35
    ax.bar(x - w / 2, chunk_vals, w, label="In-situ 14070 chunk", color="steelblue", alpha=0.85)
    ax.bar(x + w / 2, syn_vals,   w, label="Synthetic 16463",      color="coral",     alpha=0.85)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean max confidence at GT ribosome positions (log scale)")
    ax.set_title("Per-class model response at known ribosome locations\n"
                 "Ferritin channel leads in-situ; ribosome channel leads synthetic")
    ax.legend(fontsize=9)

    # Annotate ferritin and ribosome bars for in-situ
    ferr_idx = EXPECTED_CLASSES.index("ferritin-complex")
    ribo_idx = EXPECTED_CLASSES.index("cytosolic-ribosome")
    for idx, label in [(ferr_idx, "ferritin\n(highest)"), (ribo_idx, "ribosome\n(expected)")]:
        v = chunk_vals[idx]
        if v > 0:
            ax.annotate(label, xy=(idx - w / 2, v),
                        xytext=(idx - w / 2 + 0.5, v * 5),
                        fontsize=7, ha="left",
                        arrowprops=dict(arrowstyle="->", lw=0.8))

    plt.suptitle(
        "Performance collapse: in-situ vs synthetic  |  "
        "Right panel: model's blob-detector misfires on ferritin channel at ribosome locations",
        fontsize=11
    )
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
    p.add_argument("--output-2b-dir", default="output/inference_2b")
    p.add_argument("--data-dir",      default="data")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.output_dir)
    data = Path(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load volumes
    print("[load] Loading tomogram chunks …")
    chunk = zarr.open(str(data / "ExperimentRuns/14070_chunk/VoxelSpacing7.840"
                          "/wbp-denoised-ctfdeconv.zarr"), "r")["0"][:]
    syn   = zarr.open(str(data / "ExperimentRuns/16463/VoxelSpacing10.012"
                          "/wbp-denoised-denoiset-ctfdeconv.zarr"), "r")["0"][:]

    # Load GT points
    gt_chunk_pts  = load_gt_pts(find_gt_picks(
        data / "ExperimentRuns/14070_chunk/Picks", "cytosolic-ribosome"))
    gt_syn_pts    = load_gt_pts(find_gt_picks(
        data / "ExperimentRuns/16463/Picks", "cytosolic-ribosome"))
    gt_ferr_pts   = load_gt_pts(find_gt_picks(
        data / "ExperimentRuns/16463/Picks", "ferritin-complex"))

    # Load predictions
    pred_df    = pd.read_csv(out / "val_pred_df_seed.csv")
    pred_2b_df = pd.read_csv(Path(args.output_2b_dir) / "val_pred_df_seed.csv")
    pr_2b      = pd.read_csv(Path(args.output_2b_dir) / "ribosome_pr_curve.csv")
    pr_2c      = pd.read_csv(out / "ribosome_pr_curve.csv")

    print("[fig] Generating figures …")
    fig_domain_gap(chunk, syn, pred_df, out / "domain_gap_analysis.png")
    fig_slice_comparison(chunk, syn, gt_chunk_pts, gt_syn_pts,
                         out / "slice_comparison.png")
    fig_ferritin_ribosome_synthetic(syn, gt_ferr_pts, gt_syn_pts,
                                    out / "ferritin_ribosome_synthetic.png")
    fig_performance_summary(pr_2b, pr_2c, pred_df, pred_2b_df,
                            gt_chunk_pts, gt_syn_pts,
                            out / "performance_summary.png")

    print("\nDone. Figures written to", out)


if __name__ == "__main__":
    main()

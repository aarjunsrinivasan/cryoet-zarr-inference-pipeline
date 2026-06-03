# Setup

## 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

## 2. Create and activate a virtual environment

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
```

## 3. Install TopCUP

The original TopCUP repo hard-codes CUDA in two places. Use the fork below,
which fixes that:

```bash
uv pip install "git+https://github.com/aarjunsrinivasan/czii_cryoet_mlchallenge_winning_models.git"
```

To use the official GPU build instead:

```bash
uv pip install topcup
```

---

## Task 2a — Data Download

### Run IDs and data summary

| Role | Run ID | Dataset | Voxel size | Shape (Z × Y × X) | Ribosome annotations |
|---|---|---|---|---|---|
| In-distribution sanity check | **16463** (TS_5_4) | 10440 / dep. 10310 — CZII synthetic lysate | 10.012 Å | 184 × 630 × 630 | **31** |
| Held-out evaluation | **14070** (01122021_BrnoKrios…) | 10301 / dep. 10300 — Chlamydomonas in-situ | 7.84 Å | 512 × 1024 × 1024 | **202** |

Total download size: ~2.5 GB.

### How to reproduce

```bash
# Use default (./data/)
python download_data.py

# Use a custom location
python download_data.py --data-dir /path/to/my/data
```

This script:
1. Downloads each tomogram as OME-Zarr via `cryoet_data_portal.Tomogram.download_omezarr()`.
2. Fetches all ground-truth picks via `copick.from_czcdp_datasets()` and serialises them as JSON.
3. Writes a filesystem copick config at `data/copick_config_fs.json` that points at the local files, so subsequent inference runs read from disk rather than streaming from S3.

The copick config lists particles in the same positional order as the phantom
training dataset, which is what determines the channel mapping in the pretrained
checkpoint (channel assignment is positional, not name-based). The names used
here follow the evaluation dataset's (10301) convention and differ from the
phantom training names in three cases:

| Position (channel) | Phantom training name | Evaluation dataset name (used here) |
|---|---|---|
| 0 | `apo-ferritin` | `ferritin-complex` |
| 1 | `beta-amylase` | `beta-amylase` |
| 2 | `beta-galactosidase` | `beta-galactosidase` |
| 3 | `ribosome` | `cytosolic-ribosome` |
| 4 | `thyroglobulin` | `thyroglobulin` |
| 5 | `virus-like-particle` | `virus-like-capsid` |

This ordering was confirmed empirically via cross-evaluation during Task 2b development.

---

## Task 2b — Inference and Quality Assessment on Synthetic Data

### How to run

```bash
python run_inference_2b.py \
  --copick-config  data/copick_config_fs.json \
  --output-dir     output/inference_2b_batch \
  --checkpoint-dir checkpoints \
  --run-id         16463 \
  --pixelsize      10.012 \
  --tomo-type      wbp-denoised-denoiset-ctfdeconv \
  --batch-size     4        # increase for faster GPU inference; default 1
```

The checkpoint (`topcup_phantom_24_tomograms.ckpt`) is downloaded automatically
from Hugging Face on first run.

### Inference-time hyperparameters

**Confidence threshold — 0.19 for `cytosolic-ribosome` (default); sweep-best is 0.05**
The default of 0.19 comes from the phantom training config (grid-searched by
the original authors on their held-out validation set). On run 16463 the
threshold sweep found **0.05** as the strict F4 maximum (F4 = 0.820, 27 TP,
37 FP, 4 FN), vs 0.814 at 0.19 (26 TP, 21 FP, 5 FN). The gap is marginal
(ΔF4 = 0.006) and the lower threshold adds 16 extra false positives to gain
one true positive. **0.19 is used as the default**; 0.05 is reported as the
sweep-best for completeness.

**NMS radius — `int(0.5 × particle_radius / pixelsize)` → 7 voxels for ribosome**
`int(0.5 × 150 Å / 10.012 Å/vox) = int(7.49) = 7`.
Suppresses duplicate detections of the same physical particle. Half-radius
spacing is the tightest physically meaningful separation: two ribosomes cannot
overlap by more than their radius.

**Sliding window — 96³ voxels, 21 % overlap (MONAI default)**
96 vox ≈ 961 Å ≈ 3× ribosome diameter, giving the U-Net enough context around
each tile. The 21 % overlap averages out tile-edge artefacts.

**Test-time augmentation — original + full (z, y, x) flip, averaged**
Cheap symmetry augmentation; the model is approximately permutation-symmetric
in x/y/z, so flipping costs one extra forward pass but reduces variance.

**Evaluation matching radius — 0.5 × particle_radius (Å)**
Follows the CZII competition protocol. A predicted centre must fall within half
a particle radius of the GT annotation to count as a true positive.

### Quality metrics

**Detection: Precision, Recall, F_β (β = 4)**
β = 4 heavily weights recall over precision, matching the biological priority:
missing a particle is worse than one false positive in a discovery setting.
Reported at both the default threshold and the F4-maximising threshold from
the P/R sweep.

**Localization: mean Euclidean distance of TP matches (Å)**
Captures centering accuracy independently of the TP/FP/FN binary decision.
A model can score high F4 while still placing detections systematically off-
centre; this metric surfaces that.

**P/R curve + Average Precision (AP)**
Sweeps thresholds 0.05–0.70 to give a threshold-independent summary of the
precision–recall trade-off. AP is computed per class so classes with zero
ground-truth annotations (β-amylase is absent from run 16463) do not inflate
the aggregate.

### Results — run 16463 (synthetic, in-distribution)

#### Per-class metrics at default thresholds

| Class | Threshold | n_GT | TP | FP | FN | Precision | Recall | F4 | Mean TP dist (Å) |
|---|---|---|---|---|---|---|---|---|---|
| ferritin-complex | 0.16 | 46 | 41 | 244 | 5 | 0.144 | 0.891 | 0.683 | 18.9 |
| beta-amylase | 0.25 | 8 | 3 | 61 | 5 | 0.047 | 0.375 | 0.266 | 26.2 |
| beta-galactosidase | 0.13 | 10 | 5 | 71 | 5 | 0.066 | 0.500 | 0.360 | 18.1 |
| **cytosolic-ribosome** | **0.19** | **31** | **26** | **21** | **5** | **0.553** | **0.839** | **0.814** | **24.9** |
| thyroglobulin | 0.18 | 30 | 24 | 48 | 6 | 0.333 | 0.800 | 0.739 | 27.8 |
| virus-like-capsid | 0.50 | 11 | 11 | 0 | 0 | 1.000 | 1.000 | 1.000 | 20.4 |

#### Ribosome: default threshold vs sweep-best

| | Threshold | TP | FP | FN | Precision | Recall | F4 | Mean TP dist (Å) |
|---|---|---|---|---|---|---|---|---|
| Default | 0.19 | 26 | 21 | 5 | 0.553 | 0.839 | 0.814 | 24.9 |
| Sweep-best (max-F4) | 0.05 | 27 | 37 | 4 | 0.422 | 0.871 | 0.820 | 25.1 |

The sweep-best threshold gains one TP at the cost of 16 extra FPs (ΔF4 = +0.006); the default is preferred.

#### Per-class AP + best-F4 (threshold sweep 0.05–0.70)

AP is the area under the precision–recall curve (trapezoid rule). Values are low because precision stays moderate even at low thresholds (many false positives), compressing the curve area; F4 at the optimal threshold is the more informative operating-point metric.

| Class | n_GT | AP | Best F4 | @ threshold | Mean TP dist (Å) |
|---|---|---|---|---|---|
| ferritin-complex | 46 | 0.000 | 0.856 | 0.69 | 18.9 |
| beta-amylase | 8 | 0.007 | 0.313 | 0.52 | 26.2 |
| beta-galactosidase | 10 | 0.013 | 0.462 | 0.68 | 18.1 |
| **cytosolic-ribosome** | **31** | **0.036** | **0.820** | **0.05** | **25.1** |
| thyroglobulin | 30 | 0.106 | 0.751 | 0.26 | 27.8 |
| virus-like-capsid | 11 | 0.000 | 1.000 | 0.05 | 20.4 |

### Output files

| File | Contents |
|---|---|
| `output/inference_2b/val_pred_df_seed.csv` | Raw per-particle predictions with confidence scores |
| `output/inference_2b/results_default_threshold.csv` | Per-class P/R/F4/dist at default thresholds |
| `output/inference_2b/ribosome_pr_curve.csv` | Full P/R curve for `cytosolic-ribosome` |



## Task 2c — Held-out Inference and Failure Characterisation

### How to run

```bash
# Step 1: inference + metrics
python run_inference_2c.py \
  --copick-config  data/copick_config_fs.json \
  --output-dir     output/inference_2c \
  --checkpoint-dir checkpoints \
  --run-id         14070 \
  --pixelsize      7.84 \
  --tomo-type      wbp-denoised-ctfdeconv \
  --gt-user-id     data-portal \
  --results-2b     output/inference_2b/results_default_threshold.csv

# Step 2: domain gap + spatial failure figures
python generate_figures_2c.py
```

`run_inference_2c.py` extracts the chunk (x=160, y=752, 256×256, full z), builds a chunk-local copick config with coordinate-shifted GT picks, runs inference with the same checkpoint as Task 2b, and saves the metric CSVs. The two figure scripts read from the saved outputs and do not re-run inference.

### Results — chunk of run 14070 (in-situ Chlamydomonas)

#### Direct comparison: cytosolic-ribosome

| Dataset | n_GT | Threshold | TP | FP | FN | Precision | Recall | F4 | Mean TP dist (Å) | AP |
|---|---|---|---|---|---|---|---|---|---|---|
| Synthetic 16463 (2b) | 31 | 0.19 | 26 | 21 | 5 | 0.553 | 0.839 | 0.814 | 24.9 | 0.036 |
| **In-situ 14070 chunk (2c)** | **56** | **0.19** | **0** | **0** | **56** | **0.000** | **0.000** | **0.000** | — | **0.000** |

The model produces **zero ribosome detections** across the full threshold sweep 0.05–0.70. The maximum ribosome confidence score anywhere in the chunk is **0.0004** — two orders of magnitude below the lowest threshold tested.

### Failure analysis

#### 1. Visual inspection — what ribosomes look like in each setting

The most direct evidence of the domain gap: the z-slices below are picked at the **highest GT ribosome density**, spread across the volume (separation relaxed progressively until all three panels have ≥1 ribosome visible). Each lime circle marks a GT ribosome centred within ±3 voxels of the shown slice; the voxel z-coordinate and Ångström depth are printed in each panel title.

![Slice comparison](output/inference_2c/slice_comparison.png)

*Top row (in-situ chunk, run 14070): the three panels are the ribosome-richest sections; z-coordinates are printed in each title. Ribosomes are embedded in dense, continuous cytoplasm — their density signature is indistinguishable from the surrounding material at a glance. Bottom row (synthetic, run 16463): the same selection criterion applied to the synthetic volume. Particles are sparse, well-separated, and sit on a nearly empty resin background — the visual contrast that the model was trained to detect.*

This difference already explains the failure qualitatively: in-situ ribosomes do not produce the localised, high-contrast density blobs the model learned to trigger on.

#### 2. Statistical distribution mismatch

Both datasets are z-score normalised before the model (`NormalizeIntensityd`). The normalization is **per whole-volume**: MONAI computes mean and std from all voxels in the input tomogram (or chunk) and divides by those — it does not use fixed training-set statistics, and it does not normalize per-patch. Sliding-window inference extracts 96³ patches from the already-normalized volume. After whole-volume normalization the volume-level mean = 0 and variance = 1 by construction for any input — those first two moments are trivially equal and tell us nothing. What differs is the *shape* of the distribution (kurtosis, skewness), which is invariant to z-scoring.

**Scope of the comparison:** the in-situ statistics are computed from the **256×256×512 chunk** (the exact volume fed to the model at inference), not the full 512×1024×1024 14070 tomogram. The synthetic statistics are from the full 16463 tomogram. This is the correct comparison for explaining inference failure — it reflects what the model literally receives. The full in-situ tomogram would likely have kurtosis even closer to Gaussian (denser cytoplasm), so the domain gap reported here is a lower bound.

| Property | In-situ chunk (14070) | Synthetic (16463) |
|---|---|---|
| **Raw mean** | 4.882 | −4.59 × 10⁻⁷ |
| **Raw std (var)** | 18.12 (328.3) | 5.93 × 10⁻⁶ (3.51 × 10⁻¹¹) |
| **Post z-score mean** (whole-volume) | ≈ 0 (by construction) | ≈ 0 (by construction) |
| **Post z-score variance** (whole-volume) | ≈ 1 (by construction) | ≈ 1 (by construction) |
| **Post z-score kurtosis (excess)** | **0.6** | **30.4** |
| **Post z-score skewness** | −0.13 | −3.00 |

Synthetic training data has kurtosis ≈ 30: mostly empty resin background with a few dense, well-separated particles creating extreme negative-tail outliers. The model learned to detect those outlier signatures. The in-situ chunk has kurtosis ≈ 0.6 (essentially Gaussian): crowded cytoplasm fills the volume uniformly, erasing the outlier peaks the model relies on. Every voxel looks like background to the model.

![Domain gap analysis](output/inference_2c/domain_gap_analysis.png)

*Panels left to right: (1) In-situ raw reconstruction density — negative values are physically expected in CTF-corrected WBP tomograms (high electron density ↔ negative voxel values), mean/std/var shown. (2) Synthetic raw reconstruction density — separate panel on its own scale (~6 orders of magnitude different from in-situ). (3) z-score normalised distributions on log scale — synthetic heavy tail extends to −7σ, in-situ is compact and near-Gaussian. (4) Max prediction confidence per class — all six classes explicitly shown with short labels; ribosome bar annotated with its value (~0.0004), two orders of magnitude below the lowest threshold.*


### Differences between synthetic and in-situ settings

| Factor | Synthetic lysate (training) | In-situ Chlamydomonas (test) |
|---|---|---|
| Cellular context | Sparse, isolated particles in resin | Crowded cytoplasm; ribosomes in polysomes and on ER membranes |
| Intensity distribution | Sparse background + extreme outlier peaks; kurtosis ≈ 30 | Dense, near-Gaussian; kurtosis ≈ 0.6 |
| Noise | Simulation noise model | Real CTF artefacts, beam-induced motion, sample-specific background |
| Voxel size | 10.012 Å | 7.84 Å — model rescales internally, but this range may be outside training augmentation |
| Ribosome appearance | Isolated in solution; clean density blob | In polysome/ER context; neighbouring densities overlap the ribosome signal |
| Background | Simple resin | Membranes, organelles, cytoskeletal elements |

### Output files

| File | Contents |
|---|---|
| `output/inference_2c/val_pred_df_seed.csv` | Raw predictions (all classes, all confidence levels) |
| `output/inference_2c/results_default_threshold.csv` | Ribosome metrics at default threshold (0.19) |
| `output/inference_2c/ribosome_pr_curve.csv` | Full P/R curve sweep 0.05–0.70 |
| `output/inference_2c/domain_gap_analysis.png` | 4-panel: in-situ raw intensity (own scale, mean/std in title), synthetic raw intensity (own scale), z-score normalised overlay (log), per-class max confidence with ribosome annotated |
| `output/inference_2c/slice_comparison.png` | XY slices at ribosome-richest z-depths (all panels guaranteed ≥1 ribosome visible) with GT overlays — in-situ vs synthetic |
#!/usr/bin/env bash
# CPU-only environment test. Hides all GPUs via CUDA_VISIBLE_DEVICES so PyTorch
# falls back to CPU. Uses cached prediction CSVs so GPU inference is skipped.
# visualize_model_output_2c.py runs on CPU (~10-20 min for 20 patch forward passes).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=""   # hide all GPUs → torch.cuda.is_available() == False

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Setup ────────────────────────────────────────────────────────────────────

if ! command -v uv &>/dev/null; then
  echo "[setup] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source ~/.bashrc
else
  echo "[setup] uv already installed, skipping."
fi

if [ ! -d ".venv" ]; then
  echo "[setup] Creating virtual environment..."
  uv venv .venv --python 3.11
else
  echo "[setup] .venv already exists, skipping."
fi

source .venv/bin/activate

if ! python -c "import topcup" &>/dev/null 2>&1; then
  echo "[setup] Installing TopCUP fork..."
  uv pip install "git+https://github.com/aarjunsrinivasan/czii_cryoet_mlchallenge_winning_models.git"
else
  echo "[setup] TopCUP already installed, skipping."
fi

if ! python -c "import matplotlib" &>/dev/null 2>&1; then
  echo "[setup] Installing matplotlib..."
  uv pip install matplotlib
else
  echo "[setup] matplotlib already installed, skipping."
fi

# ── Import smoke test ────────────────────────────────────────────────────────

echo ""
echo "=== Import smoke test ==="
python - <<'EOF'
import copick, zarr, cryoet_data_portal
import numpy, pandas, scipy, matplotlib
import torch, monai
import topcup
print(f"  torch      {torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"  monai      {monai.__version__}")
print(f"  topcup     ok")
print(f"  copick     {copick.__version__}")
print("  All imports OK")
EOF

# ── Data check ───────────────────────────────────────────────────────────────

echo ""
if [ ! -f "data/copick_config_fs.json" ]; then
  echo "=== Task 2a: Data Download ==="
  echo "  No local data found. Running download (~2.5 GB)..."
  python download_data.py
else
  echo "=== Task 2a: Data already present, skipping download. ==="
fi

# ── Task 2b ──────────────────────────────────────────────────────────────────

echo ""
echo "=== Task 2b: Metrics from cached predictions (no GPU inference) ==="
# val_pred_df_seed.csv exists → inference step is skipped automatically
python run_inference_2b.py \
  --copick-config  data/copick_config_fs.json \
  --output-dir     output/inference_2b \
  --checkpoint-dir checkpoints \
  --run-id         16463 \
  --pixelsize      10.012 \
  --tomo-type      wbp-denoised-denoiset-ctfdeconv

# ── Task 2c ──────────────────────────────────────────────────────────────────

echo ""
echo "=== Task 2c: Metrics from cached predictions (no GPU inference) ==="
python run_inference_2c.py \
  --copick-config  data/copick_config_fs.json \
  --output-dir     output/inference_2c \
  --checkpoint-dir checkpoints \
  --run-id         14070 \
  --pixelsize      7.84 \
  --tomo-type      wbp-denoised-ctfdeconv \
  --gt-user-id     data-portal \
  --results-2b     output/inference_2b/results_default_threshold.csv

echo ""
echo "=== Task 2c: Diagnostic figures ==="
python generate_figures_2c.py

echo ""
echo "=== Task 2c: Model-output figures (CPU — may take ~10-20 min) ==="
python visualize_model_output_2c.py

# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "CPU test complete."

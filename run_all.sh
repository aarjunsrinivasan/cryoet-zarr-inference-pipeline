#!/usr/bin/env bash
# Run all task steps in order. Setup is skipped if already done.
set -euo pipefail

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

# ── Task 2a: Data Download ────────────────────────────────────────────────────

echo ""
echo "=== Task 2a: Data Download ==="
if [ -f "data/copick_config_fs.json" ]; then
  echo "[2a] Data already downloaded (copick_config_fs.json exists), skipping."
else
  python download_data.py
fi

# ── Task 2b: Inference on Synthetic Data ─────────────────────────────────────

echo ""
echo "=== Task 2b: Inference on Synthetic Data ==="
python run_inference_2b.py \
  --copick-config  data/copick_config_fs.json \
  --output-dir     output/inference_2b \
  --checkpoint-dir checkpoints \
  --run-id         16463 \
  --pixelsize      10.012 \
  --tomo-type      wbp-denoised-denoiset-ctfdeconv \
  --batch-size     1

# ── Task 2c: Held-out Inference + Figures ────────────────────────────────────

echo ""
echo "=== Task 2c: Held-out Inference ==="
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
echo "=== Task 2c: Diagnostic Figures ==="
python generate_figures_2c.py

echo ""
echo "=== Task 2c: Model-output Evidence Figures ==="
python visualize_model_output_2c.py

# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "All tasks complete."
echo "Outputs:"
echo "  output/inference_2b/  — Task 2b results"
echo "  output/inference_2c/  — Task 2c results and figures"

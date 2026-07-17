#!/usr/bin/env bash
# End-to-end pipeline. Run on a GPU box with internet (downloads model + data from HF).
set -euo pipefail
cd "$(dirname "$0")"

pip install -r requirements.txt
python smoke.py       # ~10s pre-flight: template/shape/grad sanity (also warms the model cache)
python extract.py     # -> cache/data.pt   (frozen forward passes, cache all-layer hidden states)
python train.py       # -> results/metrics.json, results/roc.png, results/layer_sweep.png
python analyze.py     # -> results/analysis.json, results/pca.png, results/length_hist.png (CPU-only)
echo "Done. See results/*.json and results/*.png"

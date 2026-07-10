#!/usr/bin/env bash
# Step 3 (on the server): generate all configs and run the full sweep.
# 7 languages x 4 wired models (resnet18, resnet50, septr, ast) x 3 runs.
# (wav2vec2 and Whisper+MLP are NOT wired into the repo yet — separate task.)
#
# Usage:  bash 3_run_full_sweep.sh <REPO_ROOT> <DATA_ROOT>
set -e
REPO="${1:?give repo root, e.g. ~/xmad-bench}"
DATA="${2:?give data root, e.g. ~/xmad_data}"
STEPB="$REPO/repro_tools/stepB"     # helper scripts travel inside the repo (see guide, Part 1)

source venv/bin/activate

# Restore the paper's per-model batch sizes for a GPU (make_configs already sets these:
# resnet18=200, resnet50=120, septr=10, ast=10). num_workers can be higher on a server:
sed -i 's/num_workers=2/num_workers=8/g' "$REPO/detection/data/data_manager.py" || true

# 1) generate configs pointing at the server data root
python "$STEPB/make_configs.py" "$DATA"

# 2) run everything, 3 seeds each, collecting metrics to results.csv
python "$STEPB/run_stepB.py" --repo "$REPO" --configs "$STEPB/configs" \
       --results "$REPO/results_full_sweep.csv" --repeats 3

echo "Sweep done. Metrics -> $REPO/results_full_sweep.csv ; logs/checkpoints under detection/experiments/."

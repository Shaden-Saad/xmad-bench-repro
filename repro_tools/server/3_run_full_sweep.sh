#!/usr/bin/env bash
# Step 3: run the sweep for a chosen set of languages (default: all 7).
# Models: the 4 wired ones (resnet18, resnet50, septr, ast); becomes 6 if the
# SSL models were added (add_ssl_models.sh). 3 runs per config.
#
# Usage:
#   bash 3_run_full_sweep.sh <REPO_ROOT> <DATA_ROOT> ["langs"] [tag]
# Examples:
#   node 1:  CUDA_VISIBLE_DEVICES=0 bash 3_run_full_sweep.sh ~/xmad-bench ~/xmad_data "ar en de zh" node1
#   node 2:  CUDA_VISIBLE_DEVICES=0 bash 3_run_full_sweep.sh ~/xmad-bench ~/xmad_data "ro ru es"    node2
# To put EACH language on its OWN GPU in parallel, use run_per_language_gpu.sh instead.
set -e
REPO="${1:?give repo root, e.g. ~/xmad-bench}"
DATA="${2:?give data root, e.g. ~/xmad_data}"
LANGS="${3:-ar en de zh ro ru es}"                 # subset for this call; default all 7
TAG="${4:-$(echo "$LANGS" | tr ' ,' '--')}"        # names the configs dir + results file
STEPB="$REPO/repro_tools/stepB"

# Activate your Python env (edit if your path differs):
for A in venv/bin/activate "$REPO/venv/bin/activate" "$HOME/xmad-env/bin/activate"; do
  [ -f "$A" ] && { source "$A"; break; }
done

# One-time, idempotent runtime patches (safe to re-run):
#  (a) more DataLoader workers on a server
(
  flock -x 200
  sed -i 's/num_workers=2/num_workers=8/g' "$REPO/detection/data/data_manager.py" 2>/dev/null || true
  #  (b) make main.py read $CONFIG_PATH so parallel runs never share config.json
  grep -q "CONFIG_PATH" "$REPO/detection/main.py" || \
    sed -i "s|json.load(open('./config.json'))|json.load(open(os.environ.get('CONFIG_PATH','./config.json')))|" "$REPO/detection/main.py"

) 200>"$REPO/.patch.lock"

# Use 6 models only if the SSL code has been added; otherwise the 4 wired ones.
if [ -f "$REPO/detection/ssl_models.py" ]; then
  MODELS="resnet18,resnet50,septr,ast,wav2vec2,whisper"
else
  MODELS="resnet18,resnet50,septr,ast"
fi

CFGDIR="$STEPB/configs_$TAG"
echo ">> Generating configs for: $LANGS  (models: $MODELS, tag=$TAG, GPU=${CUDA_VISIBLE_DEVICES:-default})"
python "$STEPB/make_configs.py" "$DATA" "$CFGDIR" "$LANGS" "$MODELS"

python "$STEPB/run_stepB.py" --repo "$REPO" --configs "$CFGDIR" \
       --results "$REPO/results_$TAG.csv" --repeats 3

echo "Done ($TAG). Metrics -> $REPO/results_$TAG.csv ; logs/checkpoints under detection/experiments/."

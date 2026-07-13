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

# ---------------------------------------------------------------------------
# Activate the Python environment. FAIL LOUDLY if it is missing.
# (Running with the wrong Python causes hard crashes - SIGILL, returncode -4 -
#  with no traceback. Never let the sweep proceed without a verified env.)
# Set XMAD_ENV to the env ROOT (the folder containing bin/activate), e.g.
#   export XMAD_ENV=/project/home/p2xxxxx/you/xmad-env
# ---------------------------------------------------------------------------
ACT=""
for A in "${XMAD_ENV:+$XMAD_ENV/bin/activate}" \
         "$REPO/venv/bin/activate" \
         "$HOME/xmad-env/bin/activate" \
         "$HOME/venv/bin/activate"; do
  [ -n "$A" ] && [ -f "$A" ] && { ACT="$A"; break; }
done
if [ -z "$ACT" ]; then
  echo "ERROR: no Python environment found."
  echo "  Create one:  bash $REPO/repro_tools/server/1_setup_cuda_env.sh"
  echo "  Then set an ABSOLUTE path, e.g.:  export XMAD_ENV=\$HOME/xmad-env"
  echo "  (A relative 'source venv/bin/activate' does NOT work under SLURM.)"
  exit 1
fi
source "$ACT"
echo ">> activated env: $ACT  ($(python -V 2>&1))"

# Verify the environment actually works BEFORE launching dozens of jobs.
python - <<'PYCHK'
import sys
try:
    import torch, pandas, soundfile, librosa, pedalboard, sklearn, transformers
except Exception as e:
    print(f"ERROR: dependency import failed: {e}"); sys.exit(1)
print(f">> torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
PYCHK
if [ $? -ne 0 ]; then
  echo "ERROR: the Python env at $ACT is not usable (imports failed or crashed)."
  echo "  If this crashed with 'Illegal instruction', a wheel is incompatible with"
  echo "  this CPU - rebuild it, e.g.:  pip install --force-reinstall --no-binary :all: pedalboard"
  exit 1
fi

# One-time, idempotent runtime patches (safe to re-run):
#  (a) more DataLoader workers on a server
sed -i 's/num_workers=2/num_workers=8/g' "$REPO/detection/data/data_manager.py" 2>/dev/null || true
#  (b) make main.py read $CONFIG_PATH so parallel runs never share config.json
grep -q "CONFIG_PATH" "$REPO/detection/main.py" || \
  sed -i "s|json.load(open('./config.json'))|json.load(open(os.environ.get('CONFIG_PATH','./config.json')))|" "$REPO/detection/main.py"

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

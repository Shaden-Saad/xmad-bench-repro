#!/usr/bin/env bash
# Run EACH language on its OWN GPU, in parallel, on this node.
# Assigns language 1 -> GPU 0, language 2 -> GPU 1, etc. (one GPU per language).
#
# Usage:  bash run_per_language_gpu.sh <REPO_ROOT> <DATA_ROOT> <lang1> <lang2> ...
# Node 1 (4 GPUs):  bash run_per_language_gpu.sh ~/xmad-bench ~/xmad_data ar en de zh
# Node 2 (3 GPUs):  bash run_per_language_gpu.sh ~/xmad-bench ~/xmad_data ro ru es
#
# Requires at least as many visible GPUs as languages.
#
# Per language (one GPU) it runs:
#   * 4 models: resnet18, resnet50, septr, ast   (6 if the SSL models were added)
#   * 3 repetitions per model                     -> 12 runs per language
# and writes results_<lang>.csv + log_<lang>.txt.
set -e
REPO="${1:?give repo root}"; DATA="${2:?give data root}"; shift 2
LANGS="$@"
[ -z "$LANGS" ] && { echo "Give languages, e.g.:  $0 ~/xmad-bench ~/xmad_data ar en de zh"; exit 1; }

# Activate env once
for A in venv/bin/activate "$REPO/venv/bin/activate" "$HOME/xmad-env/bin/activate"; do
  [ -f "$A" ] && { source "$A"; break; }
done

# Apply the runtime patches ONCE up front (avoids a race if done inside parallel jobs)
sed -i 's/num_workers=2/num_workers=8/g' "$REPO/detection/data/data_manager.py" 2>/dev/null || true
grep -q "CONFIG_PATH" "$REPO/detection/main.py" || \
  sed -i "s|json.load(open('./config.json'))|json.load(open(os.environ.get('CONFIG_PATH','./config.json')))|" "$REPO/detection/main.py"

ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
nlang=$(echo $LANGS | wc -w)
echo "Languages: $LANGS  ($nlang) | visible GPUs: $ngpu"
[ "$ngpu" -lt "$nlang" ] && echo "WARNING: fewer GPUs ($ngpu) than languages ($nlang); some will share a GPU."

gpu=0
for lang in $LANGS; do
  echo ">> launching $lang on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu bash "$REPO/repro_tools/server/3_run_full_sweep.sh" \
       "$REPO" "$DATA" "$lang" "$lang" > "$REPO/log_$lang.txt" 2>&1 &
  gpu=$((gpu+1))
done
echo "All languages launched in parallel. Waiting for them to finish..."
wait
echo "DONE. Per-language results: $REPO/results_<lang>.csv  | logs: $REPO/log_<lang>.txt"
echo "Combine with:  awk 'FNR==1 && NR!=1{next} {print}' $REPO/results_*.csv > $REPO/results_full_sweep.csv"

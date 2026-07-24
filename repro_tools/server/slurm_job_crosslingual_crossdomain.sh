#!/bin/bash
# =============================================================================
# Cross-lingual, CROSS-DOMAIN test — inference only, NO training.
#
# The cross-lingual runs were tested on the in-domain source (commonvoice-en/zh)
# and scored ~99% cross-ACC, far above the paper's 72-88%. The authors' own
# detection/config.json evaluates a multilingual setup on the CROSS-DOMAIN source
# (mailabs-en). This re-scores the SAME 12 cross-lingual checkpoints on the
# cross-domain test sets (mailabs-en + aishell3) to test whether that explains the
# gap.
#
# INFERENCE ONLY: best_model.pkl is fixed (selected on the in-domain validation
# split of the TRAINING languages, independent of the test set); only test_datasets
# changes. eval_inference.py calls the repo's own Trainer.test_out_of_domain(), so
# the metric matches the published '### Out of domain' numbers exactly. Nothing is
# retrained; each checkpoint is one forward pass.
#
# Requires the 12 cross-lingual checkpoints on disk:
#     detection/experiments/crosslingual_<model>_r0<N>/best_model.pkl
#
# SUBMIT
#   sbatch repro_tools/server/slurm_job_crosslingual_crossdomain.sh
# =============================================================================
#SBATCH --job-name=xling_xdomain
#SBATCH --time=06:00:00
#SBATCH --account=p200635
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output job_xling_xdomain_%j.out

set -e

REPO="${REPO:-/project/home/p201284/guo/xmad-bench-repro}"
DATA="${DATA:-/project/home/p201284/guo/xmad_data/}"
STEPB="$REPO/repro_tools/stepB"
TEST_SETS="mailabs-en,aishell3"      # cross-domain english + mandarin
MODELS=(resnet18 resnet50 septr ast)

cd "$REPO"

# --- environment (absolute path; relative activate fails under SLURM) ------------
export XMAD_ENV=${XMAD_ENV:-$REPO/venv}
[ -f "$XMAD_ENV/bin/activate" ] || { echo "ERROR: no env at $XMAD_ENV/bin/activate"; exit 1; }
source "$XMAD_ENV/bin/activate"
echo ">> env: $XMAD_ENV  ($(python -V 2>&1))"
python -c "import torch; print('>> torch', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1

# --- runtime patches needed for the TEST loader ----------------------------------
#  meta separator: the cross-domain test set includes mailabs-en (English), which the
#  released code mis-reads without the sniff patch; and the at-most cap protects the
#  extract_samples draw. main.py CONFIG_PATH patch is not needed (we import directly).
grep -q "_read_meta" "$REPO/detection/data/base_dataset_test.py" || \
  bash "$REPO/repro_tools/fix_meta_separator.sh" || true
grep -q "at most" "$REPO/detection/data/base_dataset_test.py" || \
  python "$REPO/repro_tools/fix_at_most_sampling.py" "$REPO" || true

# --- ensure the cross-lingual configs exist (deterministic regenerate) -----------
CFG_ALL="$STEPB/configs_all_xling"
CFG_XL="$STEPB/configs_crosslingual"
mkdir -p "$CFG_XL"
if [ ! -f "$CFG_XL/config_crosslingual_ast.json" ]; then
    rm -rf "$CFG_ALL"
    python "$STEPB/make_configs.py" "$DATA" "$CFG_ALL" "ar en de zh ro ru es"
    mv "$CFG_ALL"/config_crosslingual_*.json "$CFG_XL"/
fi

OUT="$REPO/results_crosslingual_crossdomain.csv"
rm -f "$OUT"

# --- score every completed checkpoint on the cross-domain test set ---------------
missing=0
for m in "${MODELS[@]}"; do
    for r in 01 02 03; do
        exp="crosslingual_${m}_r${r}"
        ck="$REPO/detection/experiments/$exp/best_model.pkl"
        if [ ! -f "$ck" ]; then
            echo ">> SKIP $exp: no best_model.pkl on disk"; missing=$((missing+1)); continue
        fi
        echo ">> === $exp on $TEST_SETS ==="
        PYTHONPATH="$REPO/detection" python -u "$STEPB/eval_inference.py" \
            --repo "$REPO" \
            --config "$CFG_XL/config_crosslingual_${m}.json" \
            --exp-name "$exp" \
            --data-root "$DATA" \
            --test-datasets "$TEST_SETS" \
            --num-samples 3000 \
            --results "$OUT"
    done
done

echo
echo "Done. Cross-domain cross-lingual scores -> $OUT"
echo "Rows recorded (expect 12): $(($(wc -l < "$OUT") - 1))"
[ "$missing" -gt 0 ] && echo "NOTE: $missing checkpoint(s) were missing and skipped."
echo "Push it:  git add -A && git commit -m 'crosslingual cross-domain eval' && git push"

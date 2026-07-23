#!/bin/bash
# =============================================================================
# Finish the ONE cross-lingual run that the 24 h limit cut short.
#
# STATE (2026-07-23): 11 of the 12 cross-lingual runs completed. Only
# crosslingual_ast_r03 was killed — AST is the slowest detector (batch 10) and a
# full 20-epoch AST run does not fit in 24 h. Its checkpoint is on disk at epoch 16:
#     detection/experiments/crosslingual_ast_r03/latest_checkpoint.pkl
#
# This RESUMES that run: fix_resume_training.py makes train() start at
# checkpoint['epoch'] + 1, so it trains epochs 17-20 (the 4 remaining) and stops —
# exactly Yuejun's suggestion. run_stepB.py detects latest_checkpoint.pkl on its own
# and sets resume_training=True; no flag needed. Only ~4 epochs remain, so this fits
# in 24 h with wide margin.
#
# It touches ONLY crosslingual_ast_r03:
#   - --repeat-index 3  restricts the run to r03,
#   - --results appends, so the existing r01/r02 rows are untouched,
#   - resnet18/50 and septr are not referenced at all.
#
# NOTE on faithfulness: a resumed run is a valid 20-epoch training but not
# bit-identical to an uninterrupted one (RNG for shuffling/augmentation is not
# restored). This is immaterial here because the repo sets NO seed — every run is
# already an independent draw, which is why the paper reports a 3-run mean +/- SD.
# Flag crosslingual_ast_r03 as "resumed" in the record.
#
# SUBMIT
#   sbatch repro_tools/server/slurm_job_crosslingual_ast_r03.sh
# =============================================================================
#SBATCH --job-name=xling_ast_r03
#SBATCH --time=24:00:00
#SBATCH --account=p200635
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output job_xling_ast_r03_%j.out

set -e

REPO="${REPO:-/project/home/p201284/guo/xmad-bench-repro}"
DATA="${DATA:-/project/home/p201284/guo/xmad_data/}"
STEPB="$REPO/repro_tools/stepB"

cd "$REPO"

# --- environment (absolute path; a relative activate fails under SLURM) ----------
export XMAD_ENV=${XMAD_ENV:-$REPO/venv}
[ -f "$XMAD_ENV/bin/activate" ] || { echo "ERROR: no env at $XMAD_ENV/bin/activate"; exit 1; }
source "$XMAD_ENV/bin/activate"
echo ">> env: $XMAD_ENV  ($(python -V 2>&1))"
python -c "import torch; print('>> torch', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1

# --- runtime patches, idempotent (same set the full sweep applies) ---------------
grep -q "_read_meta" "$REPO/detection/data/base_dataset.py" || \
  bash "$REPO/repro_tools/fix_meta_separator.sh" || true
grep -q "CONFIG_PATH" "$REPO/detection/main.py" || \
  sed -i "s|json.load(open('./config.json'))|json.load(open(os.environ.get('CONFIG_PATH','./config.json')))|" "$REPO/detection/main.py"
python "$REPO/repro_tools/fix_resume_training.py" "$REPO" || { echo "ERROR: resume patch failed"; exit 1; }
python "$REPO/repro_tools/fix_at_most_sampling.py"  "$REPO" || { echo "ERROR: at-most patch failed"; exit 1; }

# --- ensure the AST cross-lingual config exists (regenerate; deterministic) -------
CFG_ALL="$STEPB/configs_all_xling"
CFG_XL="$STEPB/configs_crosslingual"
mkdir -p "$CFG_XL"
if [ ! -f "$CFG_XL/config_crosslingual_ast.json" ]; then
    echo ">> regenerating configs (all 7 languages in one call, for the cross-lingual branch)"
    rm -rf "$CFG_ALL"
    python "$STEPB/make_configs.py" "$DATA" "$CFG_ALL" "ar en de zh ro ru es"
    mv "$CFG_ALL"/config_crosslingual_ast.json "$CFG_XL"/
fi
[ -f "$CFG_XL/config_crosslingual_ast.json" ] || { echo "ERROR: AST config missing"; exit 1; }

# --- confirm the checkpoint we intend to resume from ------------------------------
CK="$REPO/detection/experiments/crosslingual_ast_r03/latest_checkpoint.pkl"
if [ -f "$CK" ]; then
    python - "$CK" <<'EOF'
import torch, sys
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(f">> resuming crosslingual_ast_r03 from epoch {c['epoch']} -> will train epochs "
      f"{c['epoch']+1}..20 ({20 - c['epoch']} remaining)")
EOF
else
    echo ">> no checkpoint found; crosslingual_ast_r03 will train from scratch (20 epochs)."
    echo "   NOTE: a full AST run may exceed 24 h — raise --time if it is killed again."
fi

# --- run ONLY crosslingual_ast_r03 (resume auto-detected) -------------------------
echo ">> running crosslingual_ast_r03 only"
CUDA_VISIBLE_DEVICES=0 python -u "$STEPB/run_stepB.py" \
    --repo "$REPO" \
    --configs "$CFG_XL/config_crosslingual_ast.json" \
    --results "$REPO/results_crosslingual_ast.csv" \
    --repeats 3 --repeat-index 3 \
    2>&1 | tee "$REPO/job_crosslingual_ast_r03.log"

echo
echo "Done. Row appended to results_crosslingual_ast.csv:"
grep "recorded:" "$REPO/job_crosslingual_ast_r03.log" | tail -1

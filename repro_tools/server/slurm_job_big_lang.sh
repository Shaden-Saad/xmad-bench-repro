#!/bin/bash
# =============================================================================
# One GPU per (model, repeat) — for the LARGE languages: ar, en, ru (and ro).
#
# WHY THIS EXISTS
# ---------------
# The old layout gave each LANGUAGE one GPU, which ran all 12 runs
# (4 models x 3 repeats) sequentially. Measured training rows:
#
#     de  14,934   -> 12 runs *just* fit in 48h   (completed)
#     es  12,694   -> 12 runs fit in 48h          (completed)
#     zh  13,776   -> ~11/12 done when killed
#     ro  30,820   -> 2.1x German
#     ru  67,444   -> 4.5x German
#     ar  76,602   -> 5.1x German   <-- 1 run done in 48h
#     en  96,368   -> 6.5x German
#
# So Arabic needs ~5x German's wall-clock for the same 12 runs. Splitting only by
# MODEL (4 GPUs, 3 repeats each) still leaves ~3 x 20h = 60h on one GPU -> it would
# hit the 48h wall a SECOND time. Splitting by (model, repeat) gives each GPU
# exactly ONE run (~20h for Arabic), which fits with room to spare.
#
# LAYOUT
# ------
# SLURM array of 3 tasks (one per repeat). Each array task = 1 node x 4 GPUs,
# running the 4 models in parallel, one model per GPU:
#
#     array task 1  -> node, GPU0..3 = resnet18/resnet50/septr/ast, repeat r01
#     array task 2  -> node, GPU0..3 = resnet18/resnet50/septr/ast, repeat r02
#     array task 3  -> node, GPU0..3 = resnet18/resnet50/septr/ast, repeat r03
#
# = 12 GPUs, 12 runs, all in parallel. The repeat index is baked into exp_name
# (ar_ast_r02), so checkpoints and configs never collide across nodes.
#
# SUBMIT
# ------
#     sbatch --job-name=ar slurm_job_big_lang.sh          # Arabic (default)
#     XMAD_LANG=en sbatch --job-name=en slurm_job_big_lang.sh
#     XMAD_LANG=ru sbatch --job-name=ru slurm_job_big_lang.sh
#
# Completed runs are skipped automatically (run_stepB reads results_*.csv), so
# resubmitting after a failure only trains what is missing.
# =============================================================================
#SBATCH --time=48:00:00
#SBATCH --account=p200635
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-3
#SBATCH --output job_%x_r%a_%A.out

REPO=/project/home/p201284/guo/xmad-bench-repro
DATA=/project/home/p201284/guo/xmad_data/
SWEEP="$REPO/repro_tools/server/3_run_full_sweep.sh"

LANG="${XMAD_LANG:-ar}"                 # ar by default; en / ru via XMAD_LANG
REPEAT="${SLURM_ARRAY_TASK_ID:-1}"      # this task runs exactly this repeat
MODELS=(resnet18 resnet50 septr ast)    # one per GPU

# ---------------------------------------------------------------------------
# Python environment — ABSOLUTE path (a relative `source venv/bin/activate`
# silently fails under SLURM and every run dies with SIGILL / returncode -4).
# ---------------------------------------------------------------------------
export XMAD_ENV=${XMAD_ENV:-$REPO/venv}

if [ ! -f "$XMAD_ENV/bin/activate" ]; then
    echo "ERROR: no Python environment at $XMAD_ENV/bin/activate"
    echo "Create it once:  bash $REPO/repro_tools/server/1_setup_cuda_env.sh"
    exit 1
fi
source "$XMAD_ENV/bin/activate"
echo ">> env: $XMAD_ENV  ($(python -V 2>&1))"
echo ">> language=$LANG  repeat=r0$REPEAT  (array task $REPEAT of 3)"

# Fail fast rather than launching 4 doomed runs.
python -c "import torch, pandas, soundfile, librosa, pedalboard, sklearn, transformers; \
print('>> torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), \
'| GPUs visible:', torch.cuda.device_count())" || {
    echo "ERROR: dependencies not usable in $XMAD_ENV."
    echo "If this crashed with 'Illegal instruction', a wheel is wrong for this CPU."
    echo "  pedalboard must be 0.9.13 on MeluXina:  pip install pedalboard==0.9.13"
    exit 1; }

# ---------------------------------------------------------------------------
# Launch: one model per GPU, all four in parallel, this repeat only.
# XMAD_MODELS  -> restricts the sweep to a single model
# XMAD_REPEAT  -> restricts it to a single repeat (--repeat-index)
# tag          -> names the results file: results_<lang>_<model>_r0<N>.csv
# ---------------------------------------------------------------------------
for i in "${!MODELS[@]}"; do
    MODEL=${MODELS[$i]}
    TAG="${LANG}_${MODEL}_r0${REPEAT}"
    echo ">> GPU $i : $LANG / $MODEL / r0$REPEAT  -> results_$TAG.csv"
    CUDA_VISIBLE_DEVICES=$i XMAD_MODELS=$MODEL XMAD_REPEAT=$REPEAT \
        bash "$SWEEP" "$REPO" "$DATA" "$LANG" "$TAG" \
        > "job_${TAG}.log" 2>&1 &
done

wait
echo "Finished: $LANG repeat r0$REPEAT (all 4 models)."
echo "Metrics -> $REPO/results_${LANG}_*_r0${REPEAT}.csv"

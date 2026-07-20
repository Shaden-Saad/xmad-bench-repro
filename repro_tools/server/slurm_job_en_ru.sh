#!/bin/bash
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

REPEAT="${SLURM_ARRAY_TASK_ID:-1}"      # this task runs exactly this repeat

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
echo ">> repeat=r0$REPEAT  (array task $REPEAT of 3)"

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
#
# NOTE on RUN_LANG: renamed from LANG because `LANG` is a reserved shell/
# system locale variable (controls encoding/sorting for every subprocess).
# Reusing it here would silently change locale behavior for python/etc.
#
# XMAD_MODELS -> restricts the sweep to a single model
# XMAD_REPEAT -> restricts it to a single repeat (--repeat-index)
# tag         -> names the results file: results_<lang>_<model>_r0<N>.csv
# ---------------------------------------------------------------------------

# GPU 0 : ru / septr
RUN_LANG=ru
MODEL=septr
TAG="${RUN_LANG}_${MODEL}_r0${REPEAT}"
echo ">> GPU 0 : $RUN_LANG / $MODEL / r0$REPEAT  -> results_$TAG.csv"
CUDA_VISIBLE_DEVICES=0 XMAD_MODELS=$MODEL XMAD_REPEAT=$REPEAT \
bash "$SWEEP" "$REPO" "$DATA" "$RUN_LANG" "$TAG" \
    > "job_${TAG}.log" 2>&1 &

# GPU 1 : en / resnet18
RUN_LANG=en
MODEL=resnet18
TAG="${RUN_LANG}_${MODEL}_r0${REPEAT}"
echo ">> GPU 1 : $RUN_LANG / $MODEL / r0$REPEAT  -> results_$TAG.csv"
CUDA_VISIBLE_DEVICES=1 XMAD_MODELS=$MODEL XMAD_REPEAT=$REPEAT \
bash "$SWEEP" "$REPO" "$DATA" "$RUN_LANG" "$TAG" \
    > "job_${TAG}.log" 2>&1 &

# GPU 2 : en / resnet50
RUN_LANG=en
MODEL=resnet50
TAG="${RUN_LANG}_${MODEL}_r0${REPEAT}"
echo ">> GPU 2 : $RUN_LANG / $MODEL / r0$REPEAT  -> results_$TAG.csv"
CUDA_VISIBLE_DEVICES=2 XMAD_MODELS=$MODEL XMAD_REPEAT=$REPEAT \
bash "$SWEEP" "$REPO" "$DATA" "$RUN_LANG" "$TAG" \
    > "job_${TAG}.log" 2>&1 &

# GPU 3 : en / septr
RUN_LANG=en
MODEL=septr
TAG="${RUN_LANG}_${MODEL}_r0${REPEAT}"
echo ">> GPU 3 : $RUN_LANG / $MODEL / r0$REPEAT  -> results_$TAG.csv"
CUDA_VISIBLE_DEVICES=3 XMAD_MODELS=$MODEL XMAD_REPEAT=$REPEAT \
bash "$SWEEP" "$REPO" "$DATA" "$RUN_LANG" "$TAG" \
    > "job_${TAG}.log" 2>&1 &

wait
echo "Finished repeat r0$REPEAT (all 4 model/lang combos)."
echo "Metrics -> $REPO/results_*_r0${REPEAT}.csv"
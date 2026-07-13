#!/bin/bash
#SBATCH --time=10:30:00
#SBATCH --account=p200635
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --output job_%x_%j.out

REPO=/project/home/p201284/guo/xmad-bench-repro
DATA=/project/home/p201284/guo/xmad_data
SWEEP="$REPO/repro_tools/server/3_run_full_sweep.sh"

# --------------------------------------------------------------------------
# Python environment  (THIS is what broke the previous run)
#
# A RELATIVE path — `source venv/bin/activate` — does NOT work under SLURM:
# the job starts in a different working directory, the file isn't found, the
# env is never activated, and every run then dies instantly with
# returncode -4 (SIGILL) and no traceback.  Always use an ABSOLUTE path.
#
# Create the environment ONCE (on a node with a GPU visible):
#     bash $REPO/repro_tools/server/1_setup_cuda_env.sh
# If MeluXina needs modules first, load them before that, e.g.:
#     module load Python
#     module load CUDA
# --------------------------------------------------------------------------
# export XMAD_ENV=${XMAD_ENV:-$HOME/xmad-env}     # <-- absolute path to the env root

# if [ ! -f "$XMAD_ENV/bin/activate" ]; then
#     echo "ERROR: no Python environment at $XMAD_ENV/bin/activate"
#     echo "Create it once:  bash $REPO/repro_tools/server/1_setup_cuda_env.sh"
#     echo "Or export XMAD_ENV=/absolute/path/to/your/env before submitting."
#     exit 1
# fi
# source "$XMAD_ENV/bin/activate"
# echo ">> env: $XMAD_ENV  ($(python -V 2>&1))"

# # Fail fast if the environment is unusable, instead of launching doomed jobs.
# python -c "import torch, pandas, soundfile, librosa, pedalboard, sklearn, transformers; \
# print('>> torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())" || {
#     echo "ERROR: dependencies not usable in $XMAD_ENV."
#     echo "If this crashed with 'Illegal instruction', a wheel is incompatible with this CPU:"
#     echo "  pip install --force-reinstall --no-binary :all: pedalboard"
#     exit 1; }

source ../../venv/bin/activate

LANGS=(ru de en es)

for i in "${!LANGS[@]}"; do
    LANG=${LANGS[$i]}
    GPU=$i
    echo ">> Launching $LANG on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU bash "$SWEEP" "$REPO" "$DATA" "$LANG" "node2_$LANG" \
        > "job_node2_${LANG}.log" 2>&1 &
done

wait
echo "All node2 language sweeps finished."

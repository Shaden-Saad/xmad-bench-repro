#!/usr/bin/env bash
#SBATCH --job-name=xmad-sweep
#SBATCH --partition=gpu            # ASK the LIST admin for the correct GPU partition name
#SBATCH --gres=gpu:1               # 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00            # raise if the full sweep needs longer
#SBATCH --output=xmad_sweep_%j.log
#
# Submit with:  sbatch slurm_job.sh
# Only needed if LIST uses the SLURM scheduler. If it is a plain SSH box, use tmux instead
# (see the migration guide) and just run 3_run_full_sweep.sh directly.

# module load cuda/12.1          # uncomment/adjust if the server uses environment modules
source ~/xmad-env/bin/activate

REPO=$HOME/xmad-bench
DATA=$HOME/xmad_data
bash "$REPO/repro_tools/server/3_run_full_sweep.sh" "$REPO" "$DATA"

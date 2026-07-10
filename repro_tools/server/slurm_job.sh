#!/bin/bash
#SBATCH --time=20:30:00
#SBATCH --account=p200635
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --output job_%x_%j.out


source venv/bin/activate

REPO=/project/home/p201284/guo/xmad-bench-repro
DATA=/project/home/p201284/guo/xmad_data
bash "$REPO/repro_tools/server/3_run_full_sweep.sh" "$REPO" "$DATA"

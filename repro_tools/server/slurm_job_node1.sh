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
SWEEP="$REPO/repro_tools/server/3_run_full_sweep.sh"

LANGS=(ar en ru)

for i in "${!LANGS[@]}"; do
    LANG=${LANGS[$i]}
    GPU=$i
    echo ">> Launching $LANG on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU bash "$SWEEP" "$REPO" "$DATA" "$LANG" "node1_$LANG" \
        > "job_node1_${LANG}.log" 2>&1 &
done

wait
echo "All node1 language sweeps finished."
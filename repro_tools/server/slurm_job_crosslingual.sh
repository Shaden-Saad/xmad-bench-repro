#!/bin/bash
# =============================================================================
# Cross-lingual experiment (XMAD-Bench §4.2) — the last outstanding cell of Table 3.
#
#   train on : commonvoice-{ar, de, ro, ru, es}
#   test on  : commonvoice-{en, zh}
#   models   : resnet18, resnet50, septr, ast   (one per GPU, in parallel)
#   repeats  : 3 each  ->  12 runs total
#
# WHY THIS EXPERIMENT WAS MISSING
# -------------------------------
# make_configs.py emits the cross-lingual configs only when ALL SEVEN languages are
# passed in a single call (cross_lingual = set(langs) == set(LANGS.keys())). The
# original sweep was split by language group to fit the 48 h queue limit, so that
# branch never fired. Nothing in the released repository prevented it.
#
# SAMPLE BUDGET
# -------------
# The paper says "at most 3,000 samples per language" without stating per-class or
# total. Confirmed 2026-07-21 (Yuejun Guo) as 3,000 in TOTAL, and verified against
# the loader: base_dataset.py draws num_samples//2 names from the REAL rows, then
# re-filters on those names; XMAD pairs real and fake under a shared sample_name, so
# both come back. make_configs.py therefore sets num_samples = 3000
# (~1,500 real + 1,500 fake per language; ~15,000 training utterances over 5 languages).
#
# SIZING
# ------
# ~15,000 training rows is close to German (14,934), whose 12 runs only just fit 48 h
# on ONE GPU. Splitting by model gives 3 runs per GPU, so 24 h is ample.
#
# SUBMIT
#   sbatch repro_tools/server/slurm_job_crosslingual.sh
# =============================================================================
#SBATCH --job-name=xling
#SBATCH --time=24:00:00
#SBATCH --account=p200635
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --output job_crosslingual_%j.out

set -e

# Defaults match the MeluXina layout; override from the environment if paths differ,
# e.g.  REPO=/other/path sbatch repro_tools/server/slurm_job_crosslingual.sh
REPO="${REPO:-/project/home/p201284/guo/xmad-bench-repro}"
DATA="${DATA:-/project/home/p201284/guo/xmad_data/}"
STEPB="$REPO/repro_tools/stepB"
MODELS=(resnet18 resnet50 septr ast)     # one per GPU

cd "$REPO"

# ---------------------------------------------------------------------------
# Python environment — ABSOLUTE path. A relative `source venv/bin/activate` fails
# silently under SLURM and every run then dies with SIGILL / returncode -4.
# ---------------------------------------------------------------------------
export XMAD_ENV=${XMAD_ENV:-$REPO/venv}

if [ ! -f "$XMAD_ENV/bin/activate" ]; then
    echo "ERROR: no Python environment at $XMAD_ENV/bin/activate"
    exit 1
fi
source "$XMAD_ENV/bin/activate"
echo ">> env: $XMAD_ENV  ($(python -V 2>&1))"

python -c "import torch, pandas, soundfile, librosa, pedalboard, sklearn, transformers; \
print('>> torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), \
'| GPUs:', torch.cuda.device_count())" || {
    echo "ERROR: dependencies not usable in $XMAD_ENV."
    echo "  (pedalboard must be 0.9.13 on MeluXina — newer wheels SIGILL on this CPU)"
    exit 1; }

# ---------------------------------------------------------------------------
# Runtime patches, idempotent — the same ones the per-language sweep applied.
#  (a) meta.csv separator: the released code hard-codes tab for en/ru but every
#      released meta.csv is comma-separated. REQUIRED HERE: the cross-lingual test
#      split is English + Mandarin, so without this the run cannot load its test set.
#  (b) main.py reads $CONFIG_PATH so parallel runs never share config.json.
#  (c) resumable training, so a run killed by the time limit continues rather than
#      restarting (the repo's own resume_training re-runs the full epoch loop).
# ---------------------------------------------------------------------------
grep -q "_read_meta" "$REPO/detection/data/base_dataset.py" || \
  bash "$REPO/repro_tools/fix_meta_separator.sh" || true
grep -q "CONFIG_PATH" "$REPO/detection/main.py" || \
  sed -i "s|json.load(open('./config.json'))|json.load(open(os.environ.get('CONFIG_PATH','./config.json')))|" "$REPO/detection/main.py"
python "$REPO/repro_tools/fix_resume_training.py" "$REPO" || {
  echo "ERROR: could not apply the resumable-training patch."; exit 1; }

# ---------------------------------------------------------------------------
# 1. Generate configs. ALL SEVEN languages must be in one call or the cross-lingual
#    configs are not written at all. This also (re)writes the 28 per-language
#    configs into the same directory — which is exactly why step 2 exists.
# ---------------------------------------------------------------------------
CFG_ALL="$STEPB/configs_all_xling"
CFG_XL="$STEPB/configs_crosslingual"
rm -rf "$CFG_ALL" "$CFG_XL"
mkdir -p "$CFG_XL"

echo ">> generating configs (all 7 languages in ONE call)"
python "$STEPB/make_configs.py" "$DATA" "$CFG_ALL" "ar en de zh ro ru es"

# ---------------------------------------------------------------------------
# 2. Isolate the 4 cross-lingual configs.
#    CRITICAL: run_stepB.py globs *.json from a directory, so pointing it at
#    $CFG_ALL would enqueue 32 configs x 3 repeats = 96 runs and re-execute the
#    entire finished sweep. A fresh --results file means nothing would be skipped.
# ---------------------------------------------------------------------------
mv "$CFG_ALL"/config_crosslingual_*.json "$CFG_XL"/
n=$(ls "$CFG_XL"/config_crosslingual_*.json 2>/dev/null | wc -l)
if [ "$n" -ne 4 ]; then
    echo "ERROR: expected 4 cross-lingual configs, found $n."
    echo "  make_configs.py only emits them when all 7 languages are in a single call."
    exit 1
fi
echo ">> isolated $n cross-lingual configs in $CFG_XL"
# $CFG_XL is passed as argv so the script has no duplicated absolute paths.
python - "$CFG_XL" <<'EOF'
import json, glob, os, sys
f = sorted(glob.glob(os.path.join(sys.argv[1], "config_crosslingual_*.json")))[0]
d = json.load(open(f))["dataset"]
print(f">> sanity: train={d['train_datasets']}")
print(f">>         test ={d['test_datasets']}")
print(f">>         extract_samples={d['extract_samples']}  num_samples={d['num_samples']}"
      f"   (expect 3000 => ~1500 real + 1500 fake per language)")
EOF

# ---------------------------------------------------------------------------
# 3. Run: one model per GPU, 3 repeats each, all four in parallel.
#    run_stepB.py accepts a single config file as well as a directory, so each GPU
#    is given exactly one model. Separate --results files avoid concurrent writes.
# ---------------------------------------------------------------------------
for i in "${!MODELS[@]}"; do
    M=${MODELS[$i]}
    echo ">> GPU $i : crosslingual / $M / 3 repeats -> results_crosslingual_$M.csv"
    CUDA_VISIBLE_DEVICES=$i python -u "$STEPB/run_stepB.py" \
        --repo "$REPO" \
        --configs "$CFG_XL/config_crosslingual_${M}.json" \
        --results "$REPO/results_crosslingual_${M}.csv" \
        --repeats 3 \
        > "$REPO/job_crosslingual_${M}.log" 2>&1 &
done

wait

echo
echo "All cross-lingual runs finished."
echo "Metrics -> $REPO/results_crosslingual_*.csv"
echo "Logs    -> $REPO/job_crosslingual_*.log"
grep -h "recorded:" "$REPO"/job_crosslingual_*.log | wc -l | xargs echo "Completed runs (expect 12):"

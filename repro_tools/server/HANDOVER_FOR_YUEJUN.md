# XMAD-Bench reproduction — run instructions (LIST GPU server)

Self-contained guide for running the full detection sweep on the LIST GPU server.
The code and helper scripts are already prepared and on GitHub; you only need to
run it and return one results file. Estimated compute: ~50-110 GPU-hours total.

Repository:  https://github.com/Shaden-Saad/xmad-bench-repro   (branch: reproduction)

## What this runs
Trains and evaluates audio-deepfake detectors on XMAD-Bench, reproducing the
paper's in-domain vs. cross-domain results: 7 languages x 4 models
(resnet18, resnet50, septr, ast) x 3 runs. Metrics (ACC/AUC/EER) are written to a
CSV. (Two further models, wav2vec2 and Whisper+MLP, are not wired into the code
yet, so they are out of scope for this run.)

## Prerequisites
- Access to a GPU node on LIST (1 NVIDIA GPU is enough).
- The dataset (~80 GB, seven language ZIPs). Source: a Google Drive folder shared
  with Shaden — **please coordinate with Shaden to get access or a copy** before
  starting. The folder contains ar/de/en/es/ro/ru/zh-cn ZIPs and an EULA
  (CC BY-NC-SA 4.0, non-commercial research use).

## Steps

**1. Is there a SLURM scheduler?** SSH into LIST and run:
```
command -v sbatch >/dev/null && echo "SLURM present" || echo "No SLURM (plain box)"
```

**2. Get a GPU.**
- SLURM:  `srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=8:00:00 --pty bash`
  (partition name may differ; `sinfo` lists them.)
- Plain box: just confirm `nvidia-smi` shows a GPU.

**3. Clone the repo:**
```
git clone https://github.com/Shaden-Saad/xmad-bench-repro.git ~/xmad-bench
cd ~/xmad-bench && git checkout reproduction
```

**4. Build the environment** (CUDA PyTorch + libraries):
```
bash repro_tools/server/1_setup_cuda_env.sh
```
If `nvidia-smi` shows a CUDA version other than 12.1, edit the `cu121` in that
script (e.g. `cu118`, `cu124`) before running. Success prints `CUDA available: True`.

**4b. Download the dataset (~80 GB) — directly onto the server.**
The data is publicly linked from the paper's GitHub README, so it can be pulled
straight from Google Drive to the server (no need to transfer via anyone's laptop).
```
pip install gdown
mkdir -p ~/xmad_data/_zips && cd ~/xmad_data/_zips
gdown --folder https://drive.google.com/drive/folders/1PjboiIGjNWU6UeuIHrZu3ofF70o0A5-X
```
This downloads all the language ZIPs (ar, de, en, es, ro, ru, zh-cn) plus the EULA.

Fallbacks if Google throttles a large file ("quota exceeded" / "too many users"):
- Sign in / use the folder shared to your account, then retry (avoids anonymous limits).
- Download a single big file by its Drive ID (right-click file -> Share -> copy link
  -> the ID is the long string in the URL):
  `gdown --id <FILE_ID> -O en.zip`
- Or `pip install "gdown>=5"` and retry; newer gdown handles Drive confirmations better.

Licence note: the data is CC BY-NC-SA 4.0 (non-commercial research) with an EULA
(EULA_form_S.pdf in the folder) — keep it on LIST storage only, don't redistribute.

**5. Stage the data into ONE folder:**
```
bash repro_tools/server/2_prepare_data.sh ~/xmad_data
```
It guides you to download the ZIPs (via `gdown`, or copy them from Shaden), unzips,
and flattens everything so all dataset folders sit under `~/xmad_data`. **Then it
prints the folder names** — please check they read like:
`commonvoice-de, mailabs-de, commonvoice-ar, masc, commonvoice-en, mailabs-en,
commonvoice-zh, aishell-3, commonvoice-ro, voxpopuli, commonvoice-ru, mailabs-ru,
commonvoice-es, mailabs-es`. If any differ, edit the mapping at the top of
`repro_tools/stepB/make_configs.py` to match, and let Shaden know.

**6. Run the sweep.** The scripts restore the paper's per-model batch sizes and
auto-detect the models (4 wired; 6 if the SSL models were added). Pick the option
that matches how you want to use the GPUs:

*Option A — everything on one GPU (simplest):*
```
tmux new -s sweep
bash repro_tools/server/3_run_full_sweep.sh ~/xmad-bench ~/xmad_data
# detach: Ctrl-b then d ; reattach: tmux attach -t sweep
```

*Option B — split the 7 languages across two nodes (your plan: 4 + 3).*
Give each node a subset of languages (3rd argument) and a tag (4th argument):
```
# on node 1:
CUDA_VISIBLE_DEVICES=0 bash repro_tools/server/3_run_full_sweep.sh ~/xmad-bench ~/xmad_data "ar en de zh" node1
# on node 2:
CUDA_VISIBLE_DEVICES=0 bash repro_tools/server/3_run_full_sweep.sh ~/xmad-bench ~/xmad_data "ro ru es" node2
```
Each writes its own `results_<tag>.csv`.

*Option C — one language per GPU, in parallel on a node.*
Assigns language 1 -> GPU 0, language 2 -> GPU 1, etc.:
```
# node 1 (4 GPUs):
bash repro_tools/server/run_per_language_gpu.sh ~/xmad-bench ~/xmad_data ar en de zh
# node 2 (3 GPUs):
bash repro_tools/server/run_per_language_gpu.sh ~/xmad-bench ~/xmad_data ro ru es
```
Each language writes `results_<lang>.csv` + `log_<lang>.txt`.

*(SLURM alternative:* edit partition/time in `repro_tools/server/slurm_job.sh`,
then `sbatch repro_tools/server/slurm_job.sh` — by default it runs the whole sweep;
change the languages/tag inside it to submit per-node jobs.)

Parallel runs are safe: each run uses its own config file and unique experiment
name, so nothing is shared or overwritten between GPUs.

## What to send back
- The per-run results CSV(s): `results_full_sweep.csv` (Option A) or the
  `results_<tag>.csv` / `results_<lang>.csv` files (Options B/C).
  Columns: exp_name, model, train, test, in/cross ACC/AUC/EER (as fractions).
- Combine per-language/-node files with:
  `awk 'FNR==1 && NR!=1{next} {print}' results_*.csv > results_full_sweep.csv`
- Optionally the per-run logs/checkpoints under `detection/experiments/`.
Send the results CSV(s) to Shaden; that's all that's needed to fill the tables.

## If something errors
Common issues and fixes are documented in `repro_tools/stepB/CODE_FIXES_APPLIED.md`.
The most likely snags: a CUDA-version mismatch in step 4 (edit the `cu121` wheel),
a SLURM partition name in step 6, or a dataset folder name in step 5. Send Shaden
the error text and the printed folder list.

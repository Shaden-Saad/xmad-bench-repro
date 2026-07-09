# Moving the reproduction to the LIST GPU server (via your own GitHub repo)

Goal: put your fixed code + helper scripts on GitHub, clone it on the LIST GPU
server, and run the full sweep (7 languages x 4 models x 3 runs) there with the
paper's batch sizes.

Fill in the two placeholders as you go:
- `Shaden-Saad`  = your GitHub username
- `<LIST_USER>@<LIST_HOST>` = your login on the LIST server

Note: only 4 detectors are wired into the code (resnet18, resnet50, septr, ast).
wav2vec2 and Whisper+MLP need extra code and are a separate task.

---

## PART 1 — Put your fixed code on GitHub (do this on your Mac)

**1.1 Bundle the helper scripts into the repo** so everything travels together:
```
cp -R "$HOME/Library/CloudStorage/OneDrive-UniversityofAberdeen/PhD_Study/06_Writing/Paper-02_Reproducibility/XMAD-Bench_repro" ~/xmad-repro/xmad-bench/repro_tools
```

**1.2 Make sure large data is NOT committed.** Add a line to .gitignore:
```
cd ~/xmad-repro/xmad-bench
printf "\n*.zip\nexperiments/\nrepro_tools/**/*.xlsx\n" >> .gitignore
```

**1.3 Turn your current state into a branch and commit** (you are on a detached
HEAD from the pinned commit, with the fixes applied):
```
git checkout -b reproduction
git add -A
git commit -m "Compatibility fixes + reproduction scripts (German verified)"
```

**1.4 Create your GitHub repo and push.**
Easiest: on github.com, click **New repository**, name it e.g. `xmad-bench-repro`,
leave it empty (no README), Create. Then:
```
git remote add myrepo https://github.com/Shaden-Saad/xmad-bench-repro.git
git push -u myrepo reproduction
```
GitHub will ask for a password — use a **Personal Access Token** (GitHub → Settings →
Developer settings → Personal access tokens), not your account password.

(Attribution: the original code is Apache-2.0 by ristea et al.; keep their LICENSE
file in the repo. Your repo is a modified copy for reproduction — fine to host.)

---

## PART 2 — On the LIST GPU server

**2.1 First, find out whether LIST uses SLURM.** SSH in and run:
```
command -v sbatch >/dev/null && echo "SLURM present -> use the sbatch/srun route" || echo "No SLURM -> plain box, use tmux"
```
Then get a GPU:
- If SLURM:  `srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=8:00:00 --pty bash`
  (partition name may differ — `sinfo` lists partitions; ask the admin if unsure.)
- If plain box: just confirm `nvidia-smi` shows a GPU. If none, ask the admin how GPUs are allocated.

**2.2 Clone your repo:**
```
git clone https://github.com/Shaden-Saad/xmad-bench-repro.git ~/xmad-bench
cd ~/xmad-bench
git checkout reproduction
```

**2.3 Build the CUDA environment:**
```
bash repro_tools/server/1_setup_cuda_env.sh
```
Edit the `cu121` in that script if `nvidia-smi` shows a different CUDA version.
Success prints `CUDA available: True`.

**2.4 Get the data into one folder:**
```
bash repro_tools/server/2_prepare_data.sh ~/xmad_data
```
Follow its prompts to download the language zips (via `gdown`, or `rsync` from your
Mac). It unzips and flattens everything so all dataset folders sit directly under
`~/xmad_data`. **Then check the printed folder names** match the mapping in
`repro_tools/stepB/make_configs.py` (commonvoice-de, mailabs-de, masc, aishell-3,
voxpopuli, ...). Edit that mapping if any names differ.

**2.5 Run the full sweep.**
- SLURM: `sbatch repro_tools/server/slurm_job.sh`  (edit partition/time first)
- Plain box (run in tmux so it survives disconnects):
```
tmux new -s sweep
source ~/xmad-env/bin/activate
bash repro_tools/server/3_run_full_sweep.sh ~/xmad-bench ~/xmad_data
# detach with Ctrl-b then d ; reattach later with: tmux attach -t sweep
```
This restores the paper's per-model batch sizes (resnet18=200, resnet50=120,
septr=10, ast=10), generates all configs, and runs every language x model x 3 seeds.

---

## PART 3 — Collect results

- Parsed metrics accumulate in `~/xmad-bench/results_full_sweep.csv`
  (columns: exp_name, model, train, test, in/cross ACC/AUC/EER — as fractions).
- Full logs + checkpoints are under `detection/experiments/<exp_name>/`.
- Copy the CSV back to your Mac/OneDrive and send it to me; I'll aggregate the
  3 seeds per cell to mean +/- SD, convert to percentages, and fill the whole
  tracker spreadsheet against the paper's Table 3.

---

## Quick checklist
- [ ] repro_tools copied into repo, data git-ignored
- [ ] branch committed and pushed to your GitHub
- [ ] GPU obtained on LIST (nvidia-smi works)
- [ ] repo cloned, CUDA env built (CUDA available: True)
- [ ] data downloaded + flattened; folder names verified vs make_configs.py
- [ ] sweep launched (sbatch or tmux)
- [ ] results_full_sweep.csv returned for tabulation

## Tell me to tailor this
Two things let me make the commands exact: (1) your GitHub username, and (2) whether
LIST uses SLURM or is a plain SSH box. Send those and I'll finalise the scripts.

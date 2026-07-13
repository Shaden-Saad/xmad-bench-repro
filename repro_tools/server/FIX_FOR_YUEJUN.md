# Fix: why every run failed (returncode -4) — and what to change

## Diagnosis (from your logs — thank you, they were exactly what was needed)

**The data is fine.** Your own log proves it:
```
OK  commonvoice-de   rows=14934   sep=','
OK  mailabs-de       rows=3100    sep=','
PRE-FLIGHT PASSED — data layout looks correct for this config.
```
So this was NOT a data-processing problem.

**The real cause** is in the SLURM output:
```
/var/spool/parastation/jobs/4827190: line 12: venv/bin/activate: No such file or directory
```
`slurm_job_node1.sh` line 12 uses a RELATIVE path (`source venv/bin/activate`). SLURM runs
the job from a different working directory, so it isn't found and the environment is never
activated. The sweep script then fell back to whatever `python` was on PATH — an
incompatible one — so every run died instantly with **returncode -4 = SIGILL (illegal
instruction)** and an empty stderr (a hard crash produces no Python traceback).

That silent fallback was a flaw in `3_run_full_sweep.sh` — it has now been fixed to
**fail loudly** and to verify the environment before launching any jobs.

## What to do (3 steps)

**1. Create the Python environment on MeluXina** (if you haven't already), using an
absolute location:
```
cd /project/home/p201284/guo/xmad-bench-repro
bash repro_tools/server/1_setup_cuda_env.sh          # creates ~/xmad-env
```
If MeluXina needs modules first, load them before running it, e.g.:
```
module load Python
module load CUDA
```
Adjust the CUDA wheel inside `1_setup_cuda_env.sh` to match `nvidia-smi`.

**2. Point the job scripts at it with an ABSOLUTE path.**
In `slurm_job_node1.sh` and `slurm_job_node2.sh`, replace line 12:
```bash
source venv/bin/activate                    # <-- WRONG (relative; fails under SLURM)
```
with:
```bash
export XMAD_ENV=$HOME/xmad-env              # absolute path to the env root
source $XMAD_ENV/bin/activate
```
(Use whatever absolute path your env actually lives at.)

**3. Pull the updated sweep script** (it now fails loudly instead of silently running
with the wrong Python, and it checks that torch/CUDA work before starting):
```
git pull
```

## Verify before submitting the full job
Run this on a GPU node — it should print a torch version and `CUDA available: True`:
```
source $XMAD_ENV/bin/activate
python -c "import torch, pandas, soundfile, librosa, pedalboard, sklearn, transformers; \
print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
```
If that line crashes with **"Illegal instruction"**, one of the wheels is incompatible with
MeluXina's CPU. Find the culprit by importing them one at a time, then rebuild it from
source, e.g.:
```
pip install --force-reinstall --no-binary :all: pedalboard
```
(`pedalboard` — used for the audio augmentations — is the most likely candidate, as it
ships precompiled binaries.)

Once that check passes, resubmit:
```
sbatch repro_tools/server/slurm_job_node1.sh
sbatch repro_tools/server/slurm_job_node2.sh
```

## Notes on your setup (all good)
- Node split (node1: ar, zh, ro | node2: de, en, es, ru) and one-GPU-per-language: correct.
- 4 models x 3 repetitions per language: correct.
- The `flock` guard you added around the patch step: nice, that prevents a race.

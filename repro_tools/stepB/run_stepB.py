#!/usr/bin/env python3
"""Step B/C sweep runner for XMAD-Bench.

For each config file, this:
  1. runs preflight_check,
  2. copies it to <repo>/detection/config.json (main.py reads './config.json'),
  3. launches training with the correct PYTHONPATH,
  4. parses stdout for the best in-domain eval and the cross-domain result,
  5. appends a row to results.csv.

Run 3 seeds by passing --repeats 3 (exp_name gets a _rNN suffix so checkpoints
don't collide). NOTE: the repo sets no random seed, so repeats vary naturally;
that is how the paper's mean +/- std over 3 runs is produced.

SURVIVING A KILLED JOB (SLURM time limit)
-----------------------------------------
Before launching each run, this checks what that run left on disk and picks one of:
  1. in the results CSV                  -> SKIP (nothing to do)
  2. cross-domain result in
     experiments/<exp>/__hystoryEval__.txt -> RECOVER the row, no retraining.
     (trainer.py writes that file open/write/close, so it survives a kill even
      though the runner's stdout buffer does not.)
  3. experiments/<exp>/latest_checkpoint.pkl present -> RESUME from the last
     completed epoch (needs repro_tools/fix_resume_training.py, applied by the
     run scripts; without it the repo would train train_epochs AGAIN on top).
  4. nothing on disk                     -> fresh run.
So resubmitting a killed job costs at most one partial epoch, not the whole sweep.

Usage:
  python run_stepB.py --repo /path/to/xmad-bench --configs configs \
                      --results results.csv --repeats 3
  python run_stepB.py --repo ... --configs configs/config_ro_ast.json   # single
"""
import argparse, csv, json, os, re, shutil, subprocess, sys, glob

# Same folder as this script; used to recover runs interrupted by a job kill.
from recover_results_from_experiments import parse_history

EVAL_RE = re.compile(r"ACC = ([0-9.]+), AUC = ([0-9.]+), EER = ([0-9.]+)")


def make_row(cfg, best_in, cross):
    """Build a results row from the run's own config (always available here)."""
    ds = cfg["dataset"]
    return {
        "exp_name": cfg["exp_name"], "model": cfg["model_type"],
        "train": ",".join(ds.get("train_datasets", [ds["dataset_train"]])),
        "test": ",".join(ds.get("test_datasets", [ds["dataset_test"]])),
        "in_acc": best_in and best_in["acc"], "in_auc": best_in and best_in["auc"],
        "in_eer": best_in and best_in["eer"],
        "cross_acc": cross and cross["acc"], "cross_auc": cross and cross["auc"],
        "cross_eer": cross and cross["eer"],
    }

def parse_metrics(stdout):
    """Return (best_indomain, crossdomain) dicts of acc/auc/eer."""
    best_in = None
    cross = None
    for line in stdout.splitlines():
        m = EVAL_RE.search(line)
        if not m:
            continue
        acc, auc, eer = map(float, m.groups())
        rec = {"acc": acc, "auc": auc, "eer": eer}
        if line.startswith("### EVAL"):
            if best_in is None or acc > best_in["acc"]:
                best_in = rec
        elif line.startswith("### Out of domain"):
            cross = rec
    return best_in, cross

def already_done(results_path):
    """exp_names already recorded in the results CSV (so we can skip them).

    run_stepB flushes after every completed run, so a job killed by the SLURM time
    limit still leaves its finished runs on disk. Re-running those wastes GPU time.
    """
    done = set()
    if results_path and os.path.exists(results_path):
        try:
            with open(results_path, newline="") as f:
                for row in csv.DictReader(f):
                    n = (row.get("exp_name") or "").strip()
                    # only count rows that actually carry a metric
                    if n and (row.get("cross_acc") or "").strip():
                        done.add(n)
        except OSError:
            pass
    return done


def run_one(repo, cfg_path, repeat, results_writer, preflight, done=frozenset()):
    cfg = json.load(open(cfg_path))
    base_name = cfg["exp_name"]
    cfg["exp_name"] = f"{base_name}_r{repeat:02d}"
    if cfg["exp_name"] in done:
        print(f"SKIP {cfg['exp_name']}: already completed (present in results CSV)", flush=True)
        return
    det = os.path.join(repo, "detection")

    # ---------------------------------------------------------------------
    # Decide what to do with any state this run left behind after a kill.
    # exp_path is relative to detection/ (main.py runs with cwd=detection).
    # ---------------------------------------------------------------------
    exp_dir = os.path.join(det, cfg.get("exp_path", "./experiments"), cfg["exp_name"])
    hist = os.path.join(exp_dir, "__hystoryEval__.txt")
    ckpt = os.path.join(exp_dir, "latest_checkpoint.pkl")

    # (a) Finished on disk but missing from the CSV (the job was killed before the
    #     runner could flush its stdout). Recover the row instead of retraining.
    if os.path.exists(hist):
        best_in, cross = parse_history(hist)
        if cross is not None:
            row = make_row(cfg, best_in, cross)
            results_writer.writerow(row)
            print(f"RECOVERED {cfg['exp_name']}: complete on disk, no retraining needed", flush=True)
            print("recorded:", row, flush=True)
            return

    # (b) Partially trained: continue from the last completed epoch rather than
    #     throwing the work away. Needs the resumable-training patch
    #     (repro_tools/fix_resume_training.py), which the run scripts apply.
    if os.path.exists(ckpt):
        cfg["resume_training"] = True
        print(f"RESUME {cfg['exp_name']}: found latest_checkpoint.pkl, continuing "
              f"from the last completed epoch", flush=True)
    else:
        cfg["resume_training"] = False

    if preflight:
        pf = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "preflight_check.py"), cfg_path])
        if pf.returncode != 0:
            print(f"SKIP {cfg_path}: preflight failed")
            return

    # Parallel-safe: each run gets its OWN config file (never the shared config.json),
    # so several languages can run at once on different GPUs without clobbering it.
    # main.py reads $CONFIG_PATH (falls back to ./config.json). Requires the one-line
    # CONFIG_PATH patch applied to main.py (the run scripts apply it automatically).
    cfg_dir = os.path.join(det, "_run_configs")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_file = os.path.join(cfg_dir, cfg["exp_name"] + ".json")
    with open(cfg_file, "w") as f:
        json.dump(cfg, f, indent=2)

    # detection-only on PYTHONPATH (code is self-contained after the import fix;
    # putting the repo root here would re-trigger the root utils.py collision).
    # CUDA_VISIBLE_DEVICES is inherited from the environment (set it per GPU).
    # PYTHONUNBUFFERED so the child streams instead of block-buffering into the pipe.
    env = dict(os.environ, PYTHONPATH=det, CONFIG_PATH=cfg_file, PYTHONUNBUFFERED="1")
    print(f"\n=== RUN {cfg['exp_name']} (model={cfg['model_type']}) ===", flush=True)

    # Stream the child's output LIVE, line by line, while keeping a copy for parsing.
    # (Previously this used capture_output=True, which hid ALL output until the run
    #  finished — so a multi-hour job looked frozen and progress was invisible.
    #  stderr is merged into stdout so errors appear in the log too.)
    proc = subprocess.Popen([sys.executable, "-u", "main.py"], cwd=det, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    captured = []
    for line in proc.stdout:
        captured.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()          # so `tail -f job_*.log` shows progress in real time
    proc.wait()
    out = "".join(captured)
    if proc.returncode != 0:
        print(f"FAILED: {cfg['exp_name']} (returncode={proc.returncode})", flush=True)
        return
    best_in, cross = parse_metrics(out)
    # A resumed run only prints the epochs it ran this time, so the best in-domain
    # eval from BEFORE the kill would be missed. The history file has every epoch.
    if os.path.exists(hist):
        hist_in, hist_cross = parse_history(hist)
        if hist_in and (best_in is None or hist_in["acc"] > best_in["acc"]):
            best_in = hist_in
        if cross is None:
            cross = hist_cross
    row = make_row(cfg, best_in, cross)
    results_writer.writerow(row)
    print("recorded:", row, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to cloned xmad-bench")
    ap.add_argument("--configs", required=True, help="configs dir OR a single config.json")
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--repeat-index", type=int, default=None,
                    help="run ONLY this repeat (e.g. 2 -> exp_name *_r02). Lets each "
                         "(model, repeat) go to its own GPU for the big languages "
                         "(ar/en/ru), instead of 3 repeats queued on one GPU.")
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--redo-all", action="store_true",
                    help="re-run everything, even runs already present in the results CSV "
                         "(default: completed runs are skipped)")
    a = ap.parse_args()

    cfgs = sorted(glob.glob(os.path.join(a.configs, "*.json"))) if os.path.isdir(a.configs) else [a.configs]
    fields = ["exp_name","model","train","test","in_acc","in_auc","in_eer","cross_acc","cross_auc","cross_eer"]

    # Skip runs already completed (e.g. after a job was killed by the time limit).
    done = set() if a.redo_all else already_done(a.results)
    if done:
        print(f">> {len(done)} run(s) already completed in {a.results} — these will be SKIPPED.", flush=True)
        print(f"   (use --redo-all to force re-running them)", flush=True)

    # Which repeats does THIS process run? Normally 1..repeats; with --repeat-index,
    # exactly one (so 3 processes on 3 GPUs cover r01/r02/r03 without colliding —
    # the repeat number is baked into exp_name, so checkpoints stay separate).
    repeats = [a.repeat_index] if a.repeat_index else list(range(1, a.repeats + 1))

    new = not os.path.exists(a.results)
    with open(a.results, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new: w.writeheader()
        for cfg in cfgs:
            for r in repeats:
                run_one(a.repo, cfg, r, w, preflight=not a.no_preflight, done=done)
                fh.flush()
    print(f"\nDone. Results -> {a.results}")

if __name__ == "__main__":
    main()

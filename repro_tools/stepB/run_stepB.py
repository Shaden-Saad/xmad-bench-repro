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

Usage:
  python run_stepB.py --repo /path/to/xmad-bench --configs configs \
                      --results results.csv --repeats 3
  python run_stepB.py --repo ... --configs configs/config_ro_ast.json   # single
"""
import argparse, csv, json, os, re, shutil, subprocess, sys, glob

EVAL_RE = re.compile(r"ACC = ([0-9.]+), AUC = ([0-9.]+), EER = ([0-9.]+)")

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

def run_one(repo, cfg_path, repeat, results_writer, preflight):
    cfg = json.load(open(cfg_path))
    base_name = cfg["exp_name"]
    cfg["exp_name"] = f"{base_name}_r{repeat:02d}"
    det = os.path.join(repo, "detection")

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
    env = dict(os.environ, PYTHONPATH=det, CONFIG_PATH=cfg_file)
    print(f"\n=== RUN {cfg['exp_name']} (model={cfg['model_type']}) ===")
    proc = subprocess.run([sys.executable, "main.py"], cwd=det, env=env,
                          capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-2000:])
    
    print(f"[DEBUG] returncode={proc.returncode}")
    print(f"[DEBUG] stderr_len={len(proc.stderr)}")

    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:] or "(stderr is empty)")
        print(f"FAILED: {cfg['exp_name']}")
        return
    best_in, cross = parse_metrics(proc.stdout)
    row = {
        "exp_name": cfg["exp_name"], "model": cfg["model_type"],
        "train": ",".join(cfg["dataset"].get("train_datasets", [cfg["dataset"]["dataset_train"]])),
        "test": ",".join(cfg["dataset"].get("test_datasets", [cfg["dataset"]["dataset_test"]])),
        "in_acc": best_in and best_in["acc"], "in_auc": best_in and best_in["auc"], "in_eer": best_in and best_in["eer"],
        "cross_acc": cross and cross["acc"], "cross_auc": cross and cross["auc"], "cross_eer": cross and cross["eer"],
    }
    results_writer.writerow(row)
    print("recorded:", row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to cloned xmad-bench")
    ap.add_argument("--configs", required=True, help="configs dir OR a single config.json")
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-preflight", action="store_true")
    a = ap.parse_args()

    cfgs = sorted(glob.glob(os.path.join(a.configs, "*.json"))) if os.path.isdir(a.configs) else [a.configs]
    fields = ["exp_name","model","train","test","in_acc","in_auc","in_eer","cross_acc","cross_auc","cross_eer"]
    new = not os.path.exists(a.results)
    with open(a.results, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new: w.writeheader()
        for cfg in cfgs:
            for r in range(1, a.repeats + 1):
                run_one(a.repo, cfg, r, w, preflight=not a.no_preflight)
                fh.flush()
    print(f"\nDone. Results -> {a.results}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rebuild results_<tag>.csv from the per-experiment history files on disk.

WHY THIS EXISTS
---------------
When a SLURM job is killed by the time limit, the runner's stdout buffer is lost,
so job_<tag>.log contains no "recorded:" lines and results_<tag>.csv is never
written. It LOOKS like the completed runs vanished.

They did not. detection/trainer.py calls save_logs_eval() after every evaluation,
which does open(..., "a") / write / close on

    <exp_path>/<exp_name>/__hystoryEval__.txt

i.e. it is flushed to disk immediately, independent of stdout. Every
"### EVAL::" line and the final "### Out of domain::" line survive the kill.

This script walks those files and rebuilds the results CSV, so completed runs are
recovered WITHOUT retraining and are then skipped by run_stepB's --skip-done logic.

A run counts as COMPLETE only if it has an "### Out of domain::" line (the
cross-domain test is the last thing a run does). Runs that were mid-training when
the job died are reported as incomplete and must be retrained.

Usage
-----
  # rebuild one language's results file
  python recover_results_from_experiments.py --repo ~/xmad-bench-repro \
         --results ~/xmad-bench-repro/results_node1_zh.csv --filter zh_

  # see what is on disk without writing anything
  python recover_results_from_experiments.py --repo ~/xmad-bench-repro --dry-run
"""
import argparse, csv, json, os, re, sys

# Matches both:
#   ### EVAL:: Epoch 7:: Loss = 0.03 :: ACC = 0.99, AUC = 0.99, EER = 0.01
#   ### Out of domain:: Best epoch 7 :: ACC = 0.92, AUC = 0.98, EER = 0.06
METRIC_RE = re.compile(r"ACC = ([0-9.eE+-]+), AUC = ([0-9.eE+-]+), EER = ([0-9.eE+-]+)")

FIELDS = ["exp_name", "model", "train", "test",
          "in_acc", "in_auc", "in_eer", "cross_acc", "cross_auc", "cross_eer"]

# exp_name is "<lang>_<model>_r<NN>" (see make_configs.py + run_stepB.py)
EXP_RE = re.compile(r"^(?P<lang>[a-z]+|crosslingual)_(?P<model>resnet18|resnet50|septr|ast|wav2vec2|whisper)_r(?P<rep>\d+)$")


def parse_history(path):
    """Return (best_indomain, crossdomain) from a __hystoryEval__.txt file."""
    best_in, cross = None, None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = METRIC_RE.search(line)
            if not m:
                continue
            acc, auc, eer = map(float, m.groups())
            rec = {"acc": acc, "auc": auc, "eer": eer}
            if line.startswith("### EVAL"):
                # the trainer saves best_model.pkl on best ACC, so "best" = max ACC
                if best_in is None or acc > best_in["acc"]:
                    best_in = rec
            elif line.startswith("### Out of domain"):
                cross = rec  # last one wins
    return best_in, cross


def config_for(repo, exp_name):
    """run_stepB writes each run's config to detection/_run_configs/<exp_name>.json."""
    p = os.path.join(repo, "detection", "_run_configs", exp_name + ".json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return None


def row_for(repo, exp_name, best_in, cross):
    cfg = config_for(repo, exp_name)
    if cfg:
        ds = cfg.get("dataset", {})
        model = cfg.get("model_type", "")
        train = ",".join(ds.get("train_datasets", [ds.get("dataset_train", "")]))
        test = ",".join(ds.get("test_datasets", [ds.get("dataset_test", "")]))
    else:
        # fall back to the name if the config file is gone
        m = EXP_RE.match(exp_name)
        model = m.group("model") if m else ""
        train = test = ""
    return {
        "exp_name": exp_name, "model": model, "train": train, "test": test,
        "in_acc": best_in and best_in["acc"], "in_auc": best_in and best_in["auc"],
        "in_eer": best_in and best_in["eer"],
        "cross_acc": cross["acc"], "cross_auc": cross["auc"], "cross_eer": cross["eer"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to the xmad-bench-repro checkout")
    ap.add_argument("--exp-path", default=None,
                    help="experiments dir (default: <repo>/detection/experiments)")
    ap.add_argument("--results", default=None,
                    help="CSV to write/merge into (omit with --dry-run to just look)")
    ap.add_argument("--filter", default="",
                    help="only recover runs whose exp_name starts with this (e.g. 'zh_')")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    a = ap.parse_args()

    exp_path = a.exp_path or os.path.join(a.repo, "detection", "experiments")
    if not os.path.isdir(exp_path):
        sys.exit(f"No experiments dir at {exp_path}")

    complete, incomplete = [], []
    for name in sorted(os.listdir(exp_path)):
        if not name.startswith(a.filter):
            continue
        hist = os.path.join(exp_path, name, "__hystoryEval__.txt")
        if not os.path.isfile(hist):
            incomplete.append((name, "no __hystoryEval__.txt (died before first eval)"))
            continue
        best_in, cross = parse_history(hist)
        if cross is None:
            n = "no eval yet" if best_in is None else "trained but no cross-domain test"
            incomplete.append((name, n))
            continue
        complete.append(row_for(a.repo, name, best_in, cross))

    print(f"Scanned {exp_path}")
    print(f"  COMPLETE   : {len(complete)}  (recoverable — no retraining needed)")
    print(f"  INCOMPLETE : {len(incomplete)}  (must be retrained)")
    for n, why in incomplete:
        print(f"      - {n}: {why}")

    if a.dry_run or not a.results:
        for r in complete:
            print(f"  {r['exp_name']:24s} in_acc={r['in_acc']}  cross_acc={r['cross_acc']}")
        if not a.results:
            print("\n(no --results given: nothing written)")
        return

    # Merge, never clobber: keep rows already in the CSV, add only what's missing.
    existing, seen = [], set()
    if os.path.exists(a.results):
        with open(a.results, newline="") as f:
            for row in csv.DictReader(f):
                existing.append(row)
                if (row.get("exp_name") or "").strip():
                    seen.add(row["exp_name"].strip())

    added = [r for r in complete if r["exp_name"] not in seen]
    with open(a.results, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in existing:
            w.writerow({k: row.get(k, "") for k in FIELDS})
        for row in added:
            w.writerow(row)

    print(f"\n{a.results}: {len(existing)} kept + {len(added)} recovered = {len(existing) + len(added)} rows")
    for r in added:
        print(f"  + {r['exp_name']}")
    print("\nThese runs will now be SKIPPED on the next sbatch (run_stepB reads this CSV).")


if __name__ == "__main__":
    main()

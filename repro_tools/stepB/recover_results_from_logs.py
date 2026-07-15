#!/usr/bin/env python3
"""Rebuild results_*.csv from the job logs.

WHY THIS EXISTS
---------------
run_stepB.py APPENDS each completed run to results_<tag>.csv. But those CSVs were
tracked in git as 0-byte files, so any `git pull` / `git checkout` reset them to the
committed empty version and silently wiped the rows. (This is why results_node2_es.csv
was empty even though all 12 Spanish runs succeeded.)

Nothing is lost when this happens: run_stepB.py also PRINTS every completed run to the
log as a line beginning with "recorded: {...}". This script parses those lines back
into a proper CSV.

The underlying cause is now fixed (results_*.csv are git-ignored and untracked), but
this script remains useful for recovering any run whose CSV was lost, and as a
belt-and-braces check that the CSV matches the log.

USAGE
-----
  # rebuild every results_<tag>.csv from every job_<tag>.log in a folder
  python recover_results_from_logs.py <logs-dir> <output-dir>

  # example (on the HPC, from the repo root)
  python repro_tools/stepB/recover_results_from_logs.py repro_tools/server .

Log file  job_node2_es.log  ->  results_node2_es.csv
"""
import argparse, ast, csv, glob, os, re

FIELDS = ["exp_name", "model", "train", "test",
          "in_acc", "in_auc", "in_eer", "cross_acc", "cross_auc", "cross_eer"]


def parse_log(path):
    """Extract the 'recorded: {...}' dicts from one job log."""
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("recorded:"):
                continue
            try:
                d = ast.literal_eval(s.split("recorded:", 1)[1].strip())
                if isinstance(d, dict) and "exp_name" in d:
                    rows.append(d)
            except (ValueError, SyntaxError):
                pass  # truncated/interleaved line - skip it
    return rows


def tag_from_logname(name):
    """job_node2_es.log -> node2_es"""
    b = os.path.basename(name)
    m = re.match(r"job_(.+)\.log$", b)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs_dir", help="folder containing job_<tag>.log files")
    ap.add_argument("out_dir", nargs="?", default=".", help="where to write results_<tag>.csv")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite an existing results CSV even if it already has rows")
    a = ap.parse_args()

    logs = sorted(glob.glob(os.path.join(a.logs_dir, "job_*.log")))
    if not logs:
        print(f"No job_*.log files found in {a.logs_dir}")
        return

    total = 0
    for log in logs:
        tag = tag_from_logname(log)
        if not tag or tag.startswith("slurm"):     # skip the SLURM .out wrappers
            continue
        rows = parse_log(log)
        if not rows:
            print(f"  {os.path.basename(log):28s} -> no completed runs yet (still training?)")
            continue
        out = os.path.join(a.out_dir, f"results_{tag}.csv")
        # don't clobber a CSV that already holds MORE rows than the log
        if os.path.exists(out) and not a.overwrite:
            try:
                existing = sum(1 for _ in open(out)) - 1
            except OSError:
                existing = 0
            if existing >= len(rows) > 0:
                print(f"  {os.path.basename(log):28s} -> CSV already has {existing} rows (>= {len(rows)}), leaving it")
                continue
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in FIELDS})
        models = sorted({r.get("model") for r in rows})
        print(f"  {os.path.basename(log):28s} -> {out}  ({len(rows)} runs; models: {', '.join(models)})")
        total += len(rows)

    print(f"\nRecovered {total} run(s) in total.")
    print("Next: python repro_tools/stepB/aggregate_results.py <out_dir> "
          "repro_tools/stepB/XMAD-Bench_Reported_vs_Reproduced.xlsx")


if __name__ == "__main__":
    main()

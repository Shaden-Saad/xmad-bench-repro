#!/usr/bin/env python3
"""Pre-flight check for an XMAD-Bench config before a full run.

Validates that the dataset folders referenced by a config exist and are laid
out the way the dataset code expects, WITHOUT launching training. Catches the
common failures (wrong folder name, missing real/fake dir, bad meta.csv columns,
wrong CSV separator) up front.

Usage:  python preflight_check.py path/to/config.json
"""
import json, os, sys

REQUIRED_COLS = {"sample_name", "is_fake", "split"}          # split needed for train sets
TEST_ONLY_COLS = {"sample_name", "is_fake"}                  # cross-domain test sets: split not filtered
TAB_SETS = {"commonvoice-ru", "commonvoice-en", "mailabs-ru", "mailabs-en"}

def check_dataset(root, name, expect_split):
    problems = []
    dpath = os.path.join(root, name)
    if not os.path.isdir(dpath):
        return [f"[{name}] folder missing: {dpath}"]
    meta = os.path.join(dpath, "meta.csv")
    if not os.path.isfile(meta):
        problems.append(f"[{name}] meta.csv missing")
        return problems
    sep = "\t" if name in TAB_SETS else ","
    try:
        import pandas as pd
        df = pd.read_csv(meta, sep=sep)
    except Exception as e:
        return [f"[{name}] meta.csv unreadable with sep={sep!r}: {e}"]
    cols = set(df.columns)
    need = REQUIRED_COLS if expect_split else TEST_ONLY_COLS
    missing = need - cols
    if missing:
        problems.append(f"[{name}] meta.csv missing columns {missing} (sep={sep!r}, got {sorted(cols)})")
    for sub in ("real", "fake"):
        if not os.path.isdir(os.path.join(dpath, sub)):
            problems.append(f"[{name}] missing '{sub}/' directory")
    if expect_split and "split" in cols:
        vals = set(map(str, df["split"].unique()))
        if not ({"train", "val"} & vals):
            problems.append(f"[{name}] no train/val rows in 'split' (found {vals})")
    if not problems:
        n = len(df)
        print(f"  OK  {name:20s} rows={n:<8d} sep={sep!r}")
    return problems

def main(cfg_path):
    cfg = json.load(open(cfg_path))["dataset"]
    root = cfg["root_path"]
    print(f"root_path: {root}")
    print(f"multilingual: {cfg['multilingual']}  model check for config: {cfg_path}\n")
    all_problems = []
    if cfg["multilingual"]:
        for ds in cfg["train_datasets"]:
            all_problems += check_dataset(root, ds, expect_split=True)
        for ds in cfg["test_datasets"]:
            all_problems += check_dataset(root, ds, expect_split=False)
    else:
        all_problems += check_dataset(root, cfg["dataset_train"], expect_split=True)
        all_problems += check_dataset(root, cfg["dataset_test"], expect_split=False)
    print()
    if all_problems:
        print("PRE-FLIGHT FAILED:")
        for p in all_problems:
            print("  -", p)
        sys.exit(1)
    print("PRE-FLIGHT PASSED — data layout looks correct for this config.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.json")

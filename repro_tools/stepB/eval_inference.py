#!/usr/bin/env python3
"""Score an already-trained XMAD-Bench checkpoint on a chosen test set. NO training.

WHY
---
The cross-lingual runs were evaluated on the in-domain source (commonvoice-en/zh).
The authors' own detection/config.json evaluates a multilingual setup on the
CROSS-DOMAIN source (mailabs-en). To settle which the paper's Table-3 cross-lingual
row matches, we re-score the SAME 12 checkpoints on the cross-domain test sets
(mailabs-en + aishell3). This is inference only:

  - the trained weights are fixed (best_model.pkl, selected on the in-domain
    validation split of the TRAINING languages — independent of the test set),
  - only test_datasets changes,
  - so no retraining is needed or performed.

It reuses the repo's OWN evaluation, Trainer.test_out_of_domain(), so the metric is
identical to the one that produced the published '### Out of domain' numbers (the
as-released logit-ranked AUC/EER; this script does not 'correct' anything).

Run from the repo root with detection/ on PYTHONPATH, e.g.:
  PYTHONPATH=detection python repro_tools/stepB/eval_inference.py \
      --repo . --config repro_tools/stepB/configs_crosslingual/config_crosslingual_ast.json \
      --exp-name crosslingual_ast_r03 \
      --data-root /path/to/xmad_data \
      --test-datasets mailabs-en,aishell3 \
      --num-samples 3000 --results results_crosslingual_crossdomain_ast.csv
"""
import argparse, csv, json, os, sys

import torch
import torch.optim as optim

# these import from detection/ (must be on PYTHONPATH), exactly like main.py
from data.data_manager import DataManager
from trainer import Trainer
from models_util import (get_resnet18_model, get_resnet50_model,
                         get_septr_model, get_ast_model)

BUILD = {"resnet18": get_resnet18_model, "resnet50": get_resnet50_model,
         "septr": get_septr_model, "ast": get_ast_model}


def parse_out_of_domain(text):
    """Pull ACC/AUC/EER from the '### Out of domain' line the trainer prints."""
    import re
    m = re.search(r"Out of domain.*ACC = ([0-9.eE+-]+), AUC = ([0-9.eE+-]+), EER = ([0-9.eE+-]+)", text)
    return tuple(map(float, m.groups())) if m else (None, None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True, help="the cross-lingual config for this model")
    ap.add_argument("--exp-name", required=True,
                    help="experiment folder holding best_model.pkl, e.g. crosslingual_ast_r03")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--test-datasets", required=True,
                    help="comma-separated cross-domain test folders, e.g. mailabs-en,aishell3")
    ap.add_argument("--num-samples", type=int, default=3000,
                    help="total per test language (matches the cross-lingual run); "
                         "capped at availability by the at-most patch")
    ap.add_argument("--results", required=True)
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    cfg["device"] = ("cuda" if torch.cuda.is_available()
                     else "mps" if torch.backends.mps.is_available() else "cpu")

    # Point the evaluator at the EXISTING checkpoint and the NEW test set. Training
    # fields are left as-is but never used: we call test_out_of_domain(), not train().
    d = cfg["dataset"]
    d["root_path"] = a.data_root
    d["test_datasets"] = a.test_datasets.split(",")
    d["extract_samples"] = True
    d["num_samples"] = a.num_samples
    cfg["exp_name"] = a.exp_name                    # where best_model.pkl lives
    cfg["exp_path"] = os.path.join(a.repo, "detection", "experiments")

    model_type = cfg["model_type"]
    ck = os.path.join(cfg["exp_path"], a.exp_name, "best_model.pkl")
    if not os.path.isfile(ck):
        sys.exit(f"ERROR: no checkpoint at {ck} — run this only for completed runs.")

    ast_proc = (model_type == "ast")
    model = BUILD[model_type](cfg)

    # Build ONLY the test loader (no train/val loaders => no data is read for training).
    dm = DataManager(cfg)
    test_loader = dm.get_dataloader_test(ast_proc)

    # Dummy optimiser/scheduler: Trainer requires them, test_out_of_domain ignores them.
    crit = torch.nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sch = torch.optim.lr_scheduler.StepLR(opt, cfg["lr_sch_step"], gamma=cfg["lr_sch_gamma"])

    class _NullWriter:
        def add_scalar(self, *a, **k): pass

    trainer = Trainer(model, None, None, crit, opt, sch, _NullWriter(), cfg)

    print(f">> scoring {a.exp_name} ({model_type}) on {d['test_datasets']} "
          f"[inference only, no training]", flush=True)

    # capture the '### Out of domain' line the method prints
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        trainer.test_out_of_domain(test_loader)
    out = buf.getvalue()
    sys.stdout.write(out); sys.stdout.flush()

    acc, auc, eer = parse_out_of_domain(out)
    row = {"exp_name": a.exp_name, "model": model_type,
           "train": ",".join(d.get("train_datasets", [])),
           "test": ",".join(d["test_datasets"]),
           "cross_acc": acc, "cross_auc": auc, "cross_eer": eer}
    fields = ["exp_name", "model", "train", "test", "cross_acc", "cross_auc", "cross_eer"]
    new = not os.path.exists(a.results)
    with open(a.results, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new: w.writeheader()
        w.writerow(row)
    print("recorded:", row, flush=True)


if __name__ == "__main__":
    main()

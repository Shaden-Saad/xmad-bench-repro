# Code fixes required to run XMAD-Bench detection (documented for reproducibility)

The released code does not run as-is on a current environment. Three changes were
needed. All are compatibility fixes — none change the models, data, or metrics.
Verified end-to-end on synthetic German-layout data (train + in-domain eval +
cross-domain test all complete).

## Fix 1 — import collision (blocks all runs)
The repo has a root-level `utils.py` (which imports `torchaudio`) AND a
`detection/utils/` folder. Because `detection/utils/` has no `__init__.py`, the
root `utils.py` always shadows it, so `from utils.stats_manager import ...` in
`trainer.py` fails ("No module named 'torchaudio'" / "utils is not a package").

Fix: make the detection folder self-contained by removing the `detection.`
prefix from two files, then run from inside `detection/` with only that folder
on PYTHONPATH (so the root `utils.py` is never on the path).
- `detection/data/data_manager.py`:
  `from detection.data.base_dataset import BaseDataset`  ->  `from data.base_dataset import BaseDataset`
  `from detection.data.base_dataset_test import BaseDatasetTest` -> `from data.base_dataset_test import BaseDatasetTest`
- `detection/models_util.py`:
  `from detection.septr import SeparableTr`  ->  `from septr import SeparableTr`
  `from detection.ast_model import ASTModelLocal`  ->  `from ast_model import ASTModelLocal`

Consequence: `torchaudio` is NOT needed for the detection experiments at all
(it is only used by the root-level generation/util scripts).

## Fix 2 — torch.load default changed (blocks cross-domain test + resume)
PyTorch >= 2.6 defaults `torch.load(..., weights_only=True)`, which rejects the
authors' checkpoints (dicts with optimizer state). `test_out_of_domain()` and
resume both crash with `UnpicklingError: Weights only load failed`.

Fix: add `weights_only=False` to both `torch.load` calls in `detection/trainer.py`.

## Fix 3 — DataLoader workers (laptop convenience, optional)
`detection/data/data_manager.py` hard-codes `num_workers=10`. Fine on a server;
lowered to 2 for a laptop to avoid worker warnings/slow start. Not a correctness issue.

## Fix 4 — Apple GPU (MPS) support (Mac only; optional accelerator)
The code only selects CUDA or CPU, so on a Mac it defaults to CPU. Two changes
let it use the Apple GPU (MPS), which is much faster for local runs:
- `detection/main.py`: device selection extended to
  `'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')`.
- `detection/trainer.py`: cast inputs to float32 BEFORE moving to device
  (`inputs.float().to(device)` instead of `inputs.to(device).float()`), because
  MPS does not support float64 (audio loads as float64). This reorder is also
  harmless on CPU/GPU.
These are environment accommodations for the hardware, not changes to the method.
NOTE: batch_size was set to 64 (laptop memory) vs the paper's 200 for ResNet-18 —
a disclosed deviation; restore 200 on a GPU server for a faithful run.

## Fix 7 — hard-coded TAB separator blocks English, Russian and cross-lingual
### (VERIFIED 2026-07 — the most consequential finding so far)

The code hard-codes the meta.csv separator by dataset name:
- `detection/data/base_dataset.py`: `sep="\t"` for `commonvoice-en`, `commonvoice-ru`
- `detection/data/base_dataset_test.py`: `sep="\t"` for `mailabs-en`, `mailabs-ru`
- comma for every other dataset.

But **every meta.csv in the released data is COMMA-separated** — verified across
all 14 downloaded files (7 languages, in-domain + cross-domain). Reading a comma
file with `sep="\t"` loads the whole row as ONE column; the next line
(`meta_aux['is_fake']`) then raises:

    KeyError: 'is_fake'

Impact: **English, Russian and the cross-lingual experiment cannot run as
released — 18 of the 48 Table-3 cells (38%)**. German, Spanish, Romanian,
Arabic and Mandarin are unaffected because they take the comma path, which is
why the German reproduction succeeded.

Fix — detect the separator instead of assuming it. In both files, replace the
name-based if/else with a sniff:

```python
def _sep(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = f.readline()
    return "\t" if head.count("\t") > head.count(",") else ","

meta_aux = pd.read_csv(p, sep=_sep(p))
```

(Equivalently, since all released files are comma-separated, simply deleting the
`sep="\t"` branches also works — but sniffing is robust if the authors later fix
the data instead of the code.)

Note: this is a genuine defect in the released artifact, not an environment
issue. It means the authors' published code could not have produced their own
published English/Russian/cross-lingual numbers from the data as distributed —
the data and the code disagree. Worth raising with the authors.

## FINDING (not applied as a fix) — AUC/EER are computed from the wrong quantity

`detection/utils/stats_manager.py` does:

    # Get probability of positive class
    positive_scores = predictions[:, 1]
    roc_auc = roc_auc_score(labels, positive_scores)
    fpr, tpr, thresholds = roc_curve(labels, positive_scores)

The comment says "probability", but the models end in `Linear(..., 2)` with **no
softmax** (see models_util.py / ast_model.py), so `predictions` are **raw logits**.
For a 2-class model, P(fake) = sigmoid(logit1 - logit0). Ranking by `logit1` ALONE
is only equivalent if `logit0` is constant across samples, which it is not.
=> Every AUC and EER value produced by this code (i.e. all of Table 3) is computed
   from a mis-specified score.

Demonstration (verified): a model whose argmax is 100% correct can score AUC = 0.00
under this code.
    class 1 -> logits (0, 1)   argmax = 1 (correct), logit1 = 1
    class 0 -> logits (5, 3)   argmax = 0 (correct), logit1 = 3  (HIGHER)
  accuracy = 1.00 ; AUC via logit1 = 0.00 ; AUC via (logit1 - logit0) = 1.00

Why it went unnoticed: for a confident, well-separated model (the ResNets, ~100%)
`logit1` happens to rank correctly, so the bug is invisible. It only bites weak or
unstable models — which is exactly where we saw it (German SepTr run 2: ACC 83%,
AUC 36%).

Consistent with the paper's own numbers: Table 3 reports implausible error bars
*specifically for SepTr* — Russian SepTr AUC 82.29 +/- 22.36 and EER 20.83 +/- 20.38;
cross-lingual SepTr AUC 82.76 +/- 15.66. Std devs of +/-20 are the signature of an
unstable metric, and they appear on the weakest model.

Correct fix (NOT applied, to keep the reproduction faithful to the released code):
    positive_scores = predictions[:, 1] - predictions[:, 0]
    # or: torch.softmax(predictions, -1)[:, 1]
Recommend reporting both (as-released and corrected) and raising with the authors.

## Fix 8 — `resume_training` retrains from epoch 1 (makes killed jobs unrecoverable)
### (VERIFIED 2026-07 — found while recovering from a SLURM time-limit kill)

`detection/trainer.py` has a `resume_training` flag, but it does not resume. It
loads the checkpoint and then runs the FULL epoch loop again:

```python
if self.config['resume_training'] is True:
    checkpoint = torch.load(... 'latest_checkpoint.pkl' ...)
    self.network.load_state_dict(checkpoint['model_weights'])
    self.optimizer.load_state_dict(checkpoint['optimizer'])

for i in range(1, self.config['train_epochs'] + 1):   # <-- always 1..20
```

So resuming a run killed at epoch 14 trains 14 + 20 = **34 epochs**, not 20. The
result is not the paper's 20-epoch model. Two further defects:
- the **LR scheduler is never restored**, so the resumed run uses the wrong LR;
- **`best_metric` is never restored**, so it restarts at 0.0 and `best_model.pkl`
  is overwritten by the first post-resume epoch **even if that epoch is worse**.
  This silently corrupts the checkpoint the cross-domain test then loads.

Impact: with a 48h queue limit and Arabic/English needing ~5-6.5x German's
wall-clock, a killed job could not be continued at all — the only faithful option
was to retrain from scratch. Applied fix (`repro_tools/fix_resume_training.py`):
start at `checkpoint['epoch'] + 1`; restore `lr_scheduler` and `best_metric`
(taking `max` with `best_model.pkl`'s stored `stats`, which also repairs
old-format checkpoints); and write the `latest` checkpoint at the END of the
epoch so "checkpoint epoch N" means N is fully complete. Verified by simulation:
kill at epoch 6 + resume => exactly 20 total epochs (old code: 26); best_model.pkl
preserved against 17 worse post-resume epochs; resumed run ends on the same LR as
an uninterrupted one.

DEVIATION TO DISCLOSE: a resumed run is a valid 20-epoch training but is not
bit-identical to an uninterrupted one, because RNG state (shuffling, augmentation)
is not restored. This is minor here precisely because the repo sets no seed at all
— every run is already an independent random draw, which is why the paper reports
mean ± std over 3 runs. Resumed runs remain a legitimate sample from that
distribution; runs used in the final table should note whether they were resumed.

## Related finding — metrics survive a kill, but only because of a side effect
The runner's stdout is block-buffered, so a job killed by the time limit leaves an
apparently EMPTY log and no results CSV. The metrics are nonetheless on disk:
`trainer.py` calls `save_logs_eval()`, which does `open(...,'a')` / `write` /
`close` per line into `experiments/<exp_name>/__hystoryEval__.txt`. That flush is
incidental — the authors get crash-safe metrics by accident, not by design. We
rely on it deliberately (`recover_results_from_experiments.py`) to rebuild results
CSVs without retraining.

## Note on non-determinism
The repo sets no random seed, so results vary run-to-run. Reproduce the paper's
3-run mean +/- std, not exact numbers.

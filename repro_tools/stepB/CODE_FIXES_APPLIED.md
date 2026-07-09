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

## Note on non-determinism
The repo sets no random seed, so results vary run-to-run. Reproduce the paper's
3-run mean +/- std, not exact numbers.

# Step B — In-domain reproduction (+ Step C cross-domain)

Everything here runs on your GPU machine after Step A's environment is set up
and the dataset is downloaded. One `main.py` run produces **both** the
in-domain metrics (per-epoch eval → `best_model.pkl`) **and** the cross-domain
metrics (`test_out_of_domain`, run automatically at the end). So Steps B and C
are covered by the same command.

## 0. One-time
```bash
# activate the Step A env, then confirm the dataset root layout:
python make_configs.py /ABS/PATH/TO/xmad_data   # regenerates configs/ with your root_path
```
Edit `make_configs.py`'s `LANGS` mapping first if the downloaded **folder names**
differ from the inferred ones (see the warning at the top of that file).

## 1. Pre-flight (catches layout problems before wasting GPU time)
```bash
python preflight_check.py configs/config_ro_ast.json
```
Checks the folders exist, `meta.csv` is readable with the correct separator
(TAB for commonvoice-en/ru & mailabs-en/ru, comma otherwise), the
`real/` and `fake/` dirs are present, and the `split` column has train/val rows.

## 2. Run one experiment (single language, single model)
```bash
python run_stepB.py --repo /ABS/PATH/TO/xmad-bench \
                    --configs configs/config_ro_ast.json \
                    --results results.csv --repeats 3
```
`run_stepB.py` copies the config to `detection/config.json`, launches training
with the correct `PYTHONPATH`, parses the best in-domain eval and the
cross-domain result from stdout, and appends rows to `results.csv`.

## 3. Full sweep (all languages × the 4 wired models × 3 seeds)
```bash
python run_stepB.py --repo /ABS/PATH/TO/xmad-bench --configs configs \
                    --results results.csv --repeats 3
```
This is a large job. Recommended: start with Romanian + Spanish (both have a
clear reported drop) on AST and ResNet-50 to validate the pipeline end-to-end
before committing to the whole grid.

## 4. Fill the tracker
Aggregate the 3 seed rows per config (mean) in `results.csv`, multiply by 100,
and paste into the yellow cells of `XMAD-Bench_Reported_vs_Reproduced.xlsx`.
Δ columns compute automatically.

## What to compare against Table 3
- **In-domain (Step B):** ACC should be ~99–100% for most models. If you cannot
  reach ~100% in-domain, something is wrong with data/labels/env — fix before Step C.
- **Cross-domain (Step C):** should drop sharply, for several languages toward
  chance. Recovering that in-domain→cross-domain gap is the paper's core claim.

## Gotchas baked into these tools (so results are interpretable)
1. **No random seed** anywhere in the repo → runs are non-deterministic. Reproduce
   the 3-run mean ± std, not exact numbers. `--repeats 3` handles this.
2. **`loss_function` in config is ignored** — `main.py` hardcodes
   `CrossEntropyLoss` (the `bce_loss` in config.json has no effect).
3. **Metric units:** code outputs fractions [0,1]; the paper reports percentages.
   Multiply by 100 before entering into the tracker.
4. **Import-path bug:** never run `python detection/main.py` directly — the runner
   sets `PYTHONPATH=<repo>:<repo>/detection` for you.
5. **`num_workers=10`** is hard-coded in `data_manager.py`; lower it if you see
   DataLoader worker warnings on smaller nodes.
6. **HF/torch-hub downloads at runtime** (AST weights, ResNet weights) — ensure
   network on the GPU node or pre-cache `HF_HOME` / torch hub.
7. **wav2vec2.0 and Whisper+MLP** are in Table 3 but NOT wired into
   `models_util.py`/config. Reproducing those two rows needs extra code (separate task).

## Files
- `make_configs.py` — generates per-language + cross-lingual configs.
- `configs/` — 7 languages × 4 models + 4 cross-lingual configs (edit root_path/names).
- `preflight_check.py` — data-layout validator.
- `run_stepB.py` — sweep runner → results.csv.
- `XMAD-Bench_Reported_vs_Reproduced.xlsx` — tracker pre-filled with Table 3.

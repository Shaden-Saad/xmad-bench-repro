# Completed results — XMAD-Bench reproduction (Step 1)

Frozen, completed sweep output. These files are **archived on purpose** and are
tracked in git, unlike the live `results_*.csv` at the repo root (which are
git-ignored: a `git pull` used to reset them to the committed version and wipe
rows that had just been written on the HPC).

| File | Contents |
|---|---|
| `results_node2_de.csv` | German, 12 runs (4 models x 3 repeats), raw fractions |
| `results_node2_es.csv` | Spanish, 12 runs (4 models x 3 repeats), raw fractions |
| `summary_mean_sd.csv` | mean +/- SD per (language, model), converted to percent |

Provenance: produced by Yuejun Guo on **MeluXina (EuroHPC)**, 4x A100-SXM4-40GB,
account p200635, from commit history of `Shaden-Saad/xmad-bench-repro` (branch
`reproduction`). Received 2026-07-16. Independently cross-checked against the job
logs (`repro_tools/server/job_node2_de.log`, `job_node2_es.log`), which contain
12 `recorded:` lines each — the CSVs and the logs agree.

Configuration: **the paper's own batch sizes** (resnet18=200, resnet50=120,
septr=10, ast=10), 20 epochs, lr 5e-4, weight_decay 0, augment_chance 0.5,
3 runs per cell. No random seed is set by the repo, so the 3 runs vary naturally
— which is exactly how the paper's mean +/- std is produced.

## German — reproduced vs paper (Table 3)

| Model | in-ACC (paper / ours) | cross-ACC (paper / ours) | cross-EER (paper / ours) |
|---|---|---|---|
| ResNet-18 | 100.00 / **99.99 ± 0.02** | 90.31 / **95.63 ± 1.97** | 3.65 / **2.47 ± 0.65** |
| ResNet-50 | 100.00 / **100.00 ± 0.00** | 96.96 / **95.95 ± 3.55** | 2.16 / **3.35 ± 3.83** |
| SepTr | 99.95 / **94.69 ± 1.05** | 79.62 / **76.10 ± 4.30** | 19.95 / **29.01 ± 1.19** |
| AST | 99.65 / **99.64 ± 0.17** | 91.31 / **91.24 ± 1.95** | 8.68 / **8.80 ± 2.43** |

## Spanish — reproduced vs paper (Table 3)

| Model | in-ACC (paper / ours) | cross-ACC (paper / ours) | cross-EER (paper / ours) |
|---|---|---|---|
| ResNet-18 | 100.00 / **99.93 ± 0.05** | 90.21 / **95.78 ± 0.31** | 4.90 / **3.17 ± 1.56** |
| ResNet-50 | 100.00 / **99.96 ± 0.04** | 89.72 / **91.89 ± 4.85** | 4.07 / **4.10 ± 2.88** |
| SepTr | 96.47 / **94.70 ± 1.54** | 75.22 / **78.45 ± 3.44** | 24.52 / **21.43 ± 0.98** |
| AST | 98.80 / **96.58 ± 3.94** | 93.77 / **84.36 ± 9.33** | 5.73 / **6.62 ± 3.08** |

## Reading of these results

**The paper's central claim reproduces.** Near-perfect in-domain accuracy with a
clear drop cross-domain holds for both languages and all four released detectors.
The ResNets and AST land close to the reported values, and several of our
cross-domain numbers are *better* than the paper's (German ResNet-18 +5.32,
Spanish ResNet-18 +5.57) — the gap the paper describes is real, and if anything
slightly smaller here.

**Two cells diverge and are reported honestly:**

1. **German SepTr cross-EER: 29.01 vs 19.95 reported** (+9.06, far outside the
   paper's ±1.32 band). In-domain also falls short: 94.69 vs 99.95. The identical
   pipeline reproduced the other three detectors on the same data, which isolates
   the disagreement to SepTr rather than to our setup.
2. **Spanish AST cross-ACC: 84.36 ± 9.33 vs 93.77** (−9.41). Driven by one bad
   run (`es_ast_r03`: in-ACC 92.03, cross-ACC 73.62) against two runs near 90;
   the ±9.33 SD is the signature of an unstable cell, not a settled result.

**These are consistent with a code-level defect we found** (see
`../stepB/CODE_FIXES_APPLIED.md`, "AUC/EER are computed from the wrong
quantity"): AUC and EER are ranked by the raw class-1 logit rather than a
probability, which is only harmless for confident, well-separated models. It bites
weak or unstable ones — precisely SepTr and the bad AST run. The paper's own
Table 3 shows the same signature, reporting ±20-point standard deviations
specifically for SepTr (Russian SepTr AUC 82.29 ± 22.36, EER 20.83 ± 20.38).

## Status of the remaining languages

Complete: **German, Spanish**. Outstanding: Arabic, English, Mandarin, Romanian,
Russian, plus the cross-lingual cell. The first sweep was killed by the 48h queue
limit; see `../server/slurm_job_big_lang.sh` for the one-GPU-per-(model, repeat)
layout that the large languages need — Arabic has 5.1x and English 6.5x German's
training rows, and German's 12 runs only just fit in 48h.

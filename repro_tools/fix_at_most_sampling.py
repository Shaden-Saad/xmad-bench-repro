#!/usr/bin/env python3
"""Make the sample budget mean "AT MOST n", as the paper states. No behaviour change
when enough samples exist; prevents a crash when they do not.

THE DEFECT
----------
XMAD-Bench 4.2 specifies the cross-lingual sample budget as:

    "we randomly select at most 3,000 samples per language"

"At most" is min(requested, available). The released code cannot express that:

    detection/data/base_dataset.py:75
    detection/data/base_dataset_test.py:57
        sample_names = meta_real["sample_name"].sample(self.config["num_samples"] // 2)

pandas' .sample(n) defaults to replace=False and RAISES when n exceeds the
population, rather than capping:

    ValueError: Cannot take a larger sample than population when 'replace=False'

Critically, the sampling happens AFTER the split filter (train / val), so the
population is the real rows of ONE split of ONE language, not the whole corpus.
The validation split of the smaller languages holds fewer than num_samples // 2
real utterances, so the cross-lingual experiment aborts while building its
validation loader -- before a single epoch runs.

OBSERVED 2026-07-21 on MeluXina: all 12 cross-lingual runs (4 detectors x 3 repeats)
failed identically at data_manager.get_dataloaders -> BaseDataset(mode="test").

This is independent of the value chosen for num_samples: a LARGER budget fails
harder. num_samples = 6000 requests 3000 real per split and fails sooner than
3000 (which requests 1500). Lowering the budget to the confirmed 3,000 total made
the run strictly more likely to succeed, not less.

THE FIX
-------
Cap the draw at the available population, which is precisely what "at most" means:

    n_avail = len(meta_real)
    n_want  = self.config["num_samples"] // 2
    sample_names = meta_real["sample_name"].sample(min(n_want, n_avail))

Semantics when the population is sufficient are IDENTICAL to the released code --
the same number of names is drawn from the same pool. The cap only engages where
the released code would have raised, so no already-completed result is affected.
The per-language runs never reach this branch at all (extract_samples = False).

It also prints a one-line notice whenever the cap engages, so a reduced budget is
recorded in the job log rather than silently applied.

Idempotent. Usage:
    python fix_at_most_sampling.py /path/to/xmad-bench-repro
"""
import ast
import os
import sys

MARKER = "# [repro] 'at most' sampling"

OLD = '''                meta_real = meta_aux[meta_aux['is_fake']==0]
                sample_names = meta_real["sample_name"].sample(self.config["num_samples"] // 2)'''

NEW = '''                meta_real = meta_aux[meta_aux['is_fake']==0]
                {marker}: the paper says "at most N samples per language" (4.2), and
                # "at most" is min(requested, available). pandas .sample(n) raises
                # instead of capping, and the pool here is one SPLIT of one language,
                # which for val is smaller than the budget. Cap it.
                _n_want = self.config["num_samples"] // 2
                _n_avail = len(meta_real)
                if _n_want > _n_avail:
                    print(f"  [at-most] {{dataset}}: requested {{_n_want}} real samples, "
                          f"only {{_n_avail}} available in this split - using {{_n_avail}}",
                          flush=True)
                sample_names = meta_real["sample_name"].sample(min(_n_want, _n_avail))'''.format(marker=MARKER)

TARGETS = ["detection/data/base_dataset.py", "detection/data/base_dataset_test.py"]


def patch(path):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print(f"  already patched: {path}")
        return True
    if OLD not in src:
        print(f"  ERROR: expected sampling block not found in {path}")
        print("         Not patching - inspect the file by hand.")
        return False

    src = src.replace(OLD, NEW, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)

    with open(path, encoding="utf-8") as f:
        ast.parse(f.read())          # must still be valid Python
    print(f"  patched: {path}")
    return True


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    print("Applying 'at most' sampling patch...")
    ok = True
    for rel in TARGETS:
        p = os.path.join(repo, rel)
        if not os.path.isfile(p):
            print(f"  ERROR: {p} not found")
            ok = False
            continue
        ok &= patch(p)
    if not ok:
        sys.exit(1)
    print("Done. Sample budgets are now capped at the available population.")


if __name__ == "__main__":
    main()

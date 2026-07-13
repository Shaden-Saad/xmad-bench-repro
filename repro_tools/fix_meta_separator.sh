#!/usr/bin/env bash
# Fix: meta.csv separator is hard-coded in the released code, but the released
# DATA is inconsistent (commonvoice-ru is tab-separated; commonvoice-en is
# comma-separated), so the original code cannot load the English data.
#
# This replaces the hard-coded tab special-cases with auto-detection of the
# separator (sniffs the header line), which is correct for either format.
#
# Run ONCE from the repo root:  bash repro_tools/fix_meta_separator.sh
# Idempotent: exits if already applied.
set -e
DET="detection"
[ -d "$DET" ] || { echo "Run from the repo root (no detection/ folder here)."; exit 1; }

if grep -q "_read_meta" "$DET/data/base_dataset.py"; then
  echo "Already applied (_read_meta present). Nothing to do."; exit 0
fi

python3 - "$DET" <<'PY'
import sys, os, re
DET = sys.argv[1]

HELPER = '''

def _read_meta(path):
    """Read a meta.csv, auto-detecting comma vs tab separation.

    The released XMAD-Bench data is inconsistent: some meta.csv files are
    tab-separated (e.g. commonvoice-ru) and others comma-separated (e.g.
    commonvoice-en), while the original code hard-coded a tab separator for
    exactly commonvoice-en/ru and mailabs-en/ru. That mismatch makes the
    released code unable to load its own English data. Sniffing the header
    is correct for either format.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as _f:
        _head = _f.readline()
    _sep = "\\t" if _head.count("\\t") > _head.count(",") else ","
    return pd.read_csv(path, sep=_sep)
'''

def patch(path, pairs, add_helper_after):
    s = open(path).read()
    for old, new in pairs:
        assert old in s, f"PATTERN NOT FOUND in {path}:\n{old}"
        s = s.replace(old, new)
    # insert helper after the last import line
    idx = s.index(add_helper_after) + len(add_helper_after)
    s = s[:idx] + HELPER + s[idx:]
    open(path, "w").write(s)
    print("patched", path)

# ---------------- base_dataset.py ----------------
p = os.path.join(DET, "data", "base_dataset.py")
s = open(p).read()
anchor = "from transformers import ASTFeatureExtractor"
pairs = [
# multilingual: drop the hard-coded tab special-case
("""            if dataset not in ["commonvoice-ru", "commonvoice-en"]:
                meta_aux = pd.read_csv(os.path.join(self.config['root_path'], dataset, "meta.csv"))
            else:
                meta_aux = pd.read_csv(os.path.join(self.config['root_path'], dataset, "meta.csv"), sep="\\t")""",
 """            meta_aux = _read_meta(os.path.join(self.config['root_path'], dataset, "meta.csv"))"""),
# single-language: use the same auto-detecting reader
("""            meta = pd.read_csv(os.path.join(self.config['root_path'], dataset, "meta.csv"))
            meta = meta[meta['split'] == "train"]""",
 """            meta = _read_meta(os.path.join(self.config['root_path'], dataset, "meta.csv"))
            meta = meta[meta['split'] == "train"]"""),
("""            meta = pd.read_csv(os.path.join(self.config['root_path'], dataset, "meta.csv"))
            meta = meta[meta['split'] == "val"]""",
 """            meta = _read_meta(os.path.join(self.config['root_path'], dataset, "meta.csv"))
            meta = meta[meta['split'] == "val"]"""),
]
patch(p, pairs, anchor)

# ---------------- base_dataset_test.py ----------------
p = os.path.join(DET, "data", "base_dataset_test.py")
pairs = [
("""            if dataset not in ["mailabs-ru", "mailabs-en"]:
                meta_aux = pd.read_csv(os.path.join(self.config['root_path'], dataset, "meta.csv"))
            else:
                meta_aux = pd.read_csv(os.path.join(self.config['root_path'], dataset, "meta.csv"), sep="\\t")""",
 """            meta_aux = _read_meta(os.path.join(self.config['root_path'], dataset, "meta.csv"))"""),
("""        meta = pd.read_csv(os.path.join(self.config['root_path'], dataset, "meta.csv"))""",
 """        meta = _read_meta(os.path.join(self.config['root_path'], dataset, "meta.csv"))"""),
]
patch(p, pairs, anchor)
print("ALL PATCHES APPLIED")
PY

echo "Syntax check:"
for f in data/base_dataset.py data/base_dataset_test.py; do
  python3 -m py_compile "$DET/$f" && echo "  OK  $f" || echo "  FAIL $f"
done
echo "Done. meta.csv separator is now auto-detected (comma or tab)."

# Data-step troubleshooting (for the office session with Yuejun)

Yuejun reports "errors related to the data processing process". This is the step we
predicted as most fragile. Below: what to run, the likely causes, and the fixes.

## Run this first — one command that gathers everything
```
bash repro_tools/server/diagnose_data.sh ~/xmad_data ~/xmad-bench
```
It prints the folder names, meta.csv headers/separators, file counts, and runs the
pre-flight check. **Send Shaden/Claude that whole output** — it pinpoints the cause.

If you'd rather do it by hand, these three are the core:
```
ls ~/xmad_data                                  # actual dataset folder names
head -2 ~/xmad_data/<some-dataset>/meta.csv     # columns + comma vs tab
python repro_tools/stepB/preflight_check.py <a config file>   # names the exact problem
```

## Likely cause #1 (most probable): dataset folder names don't match the mapping
The folder names in `repro_tools/stepB/make_configs.py` were **inferred**, not verified,
for every language except German. German unzipped cleanly to `commonvoice-de` + `mailabs-de`,
but the other zips had wrapper folders (`ro-007`, `ar-005`, ...), and the cross-domain
names (`masc`, `aishell-3`, `voxpopuli`, `mailabs-*`) were an educated guess.

**Symptom:** configs point at folders that do not exist -> load/KeyError/FileNotFound.
**Fix:** edit the `LANGS` mapping at the top of `repro_tools/stepB/make_configs.py` so each
entry matches the REAL folder names printed by `ls ~/xmad_data`, then re-run the sweep.
```python
LANGS = {
    "ar": ("commonvoice-ar", "masc"),        # <- confirm both names
    "en": ("commonvoice-en", "mailabs-en"),
    "de": ("commonvoice-de", "mailabs-de"),  # verified correct
    "zh": ("commonvoice-zh", "aishell-3"),
    "ro": ("commonvoice-ro", "voxpopuli"),
    "ru": ("commonvoice-ru", "mailabs-ru"),
    "es": ("commonvoice-es", "mailabs-es"),
}
```

## Likely cause #2: the flatten step missed folders
`2_prepare_data.sh` moves folders matching `commonvoice-*`, `mailabs-*`, `masc*`,
`aishell*`, `voxpopuli*` up into one root. If a real folder is named differently, it
stays buried inside its wrapper (e.g. `~/xmad_data/ar-005/commonvoice-ar`).
**Symptom:** `ls ~/xmad_data` shows wrappers (`ar-005`, `ro-007`) instead of dataset folders.
**Fix:** move the inner folders up manually:
```
cd ~/xmad_data
for w in */ ; do
  for d in "$w"*/ ; do [ -d "$d" ] && [ -f "$d/meta.csv" ] && mv "$d" . ; done
done
ls ~/xmad_data     # should now list dataset folders directly
```

## Likely cause #3: meta.csv separator (tab vs comma)
The code reads `meta.csv` as **TAB-separated** for `commonvoice-en`, `commonvoice-ru`,
`mailabs-en`, `mailabs-ru`, and **comma** for everything else.
**Symptom:** a dataset loads with one giant column, or KeyError on 'sample_name'/'is_fake'.
**Fix:** check with `head -2 <dataset>/meta.csv`. If a set's real separator differs from
what the code assumes, tell Claude which one — it's a one-line change in
`detection/data/base_dataset.py` / `base_dataset_test.py`.

## Likely cause #4: incomplete download / unzip
Google Drive throttles large files; `en.zip` (~20 GB) and `ru.zip` (~17 GB) are the
usual casualties.
**Symptom:** missing folders, truncated zips, or far fewer audio files than expected.
**Fix:** re-download the affected language (`gdown --id <FILE_ID> -O en.zip`), verify the
zip opens, and re-run the flatten. Compare file counts against the paper's Table 1.

## What to send back to Claude
1. The full output of `diagnose_data.sh` (or at minimum `ls ~/xmad_data`).
2. The **exact error text / traceback** — not a summary. The traceback names the file
   and line, which usually identifies the cause immediately.

Most of these are 5-minute fixes once the real folder names are known.

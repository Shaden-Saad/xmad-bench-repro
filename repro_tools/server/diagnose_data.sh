#!/usr/bin/env bash
# One-shot data diagnostic. Prints everything needed to identify a data-step failure.
# Usage:  bash repro_tools/server/diagnose_data.sh <DATA_ROOT> [REPO_ROOT]
# Then send the WHOLE output back to Shaden.
DATA="${1:-$HOME/xmad_data}"
REPO="${2:-$HOME/xmad-bench}"

echo "=================================================="
echo " XMAD data diagnostic"
echo " DATA_ROOT = $DATA"
echo " REPO_ROOT = $REPO"
echo "=================================================="

echo
echo "---------- 1. Top level of DATA_ROOT ----------"
ls -1 "$DATA" 2>&1

echo
echo "---------- 2. Any wrapper folders still nested? ----------"
echo "(a dataset folder is one that CONTAINS meta.csv)"
find "$DATA" -maxdepth 3 -name meta.csv 2>/dev/null | sed "s|$DATA/||; s|/meta.csv||" | sort

echo
echo "---------- 3. Per-dataset detail ----------"
for d in $(find "$DATA" -maxdepth 3 -name meta.csv 2>/dev/null | xargs -n1 dirname | sort); do
  name=$(echo "$d" | sed "s|$DATA/||")
  nreal=$(ls "$d/real" 2>/dev/null | wc -l)
  nfake=$(ls "$d/fake" 2>/dev/null | wc -l)
  nrows=$(( $(wc -l < "$d/meta.csv") - 1 ))
  echo ""
  echo ">>> $name"
  echo "    real files: $nreal | fake files: $nfake | meta rows: $nrows"
  echo "    meta.csv header:"
  head -1 "$d/meta.csv" | sed 's/^/      /'
  echo "    first data row:"
  sed -n '2p' "$d/meta.csv" | sed 's/^/      /'
  # separator guess
  h=$(head -1 "$d/meta.csv")
  if echo "$h" | grep -q $'\t'; then echo "    separator: TAB"; else echo "    separator: COMMA"; fi
  # sanity: does real/ and fake/ exist?
  [ -d "$d/real" ] || echo "    !! MISSING real/ directory"
  [ -d "$d/fake" ] || echo "    !! MISSING fake/ directory"
done

echo
echo "---------- 4. What the configs EXPECT (mapping in make_configs.py) ----------"
grep -A 12 "^LANGS = {" "$REPO/repro_tools/stepB/make_configs.py" 2>/dev/null

echo
echo "---------- 5. Pre-flight check on a generated config (if any) ----------"
CFG=$(ls "$REPO"/repro_tools/stepB/configs*/*.json 2>/dev/null | head -1)
if [ -n "$CFG" ]; then
  echo "using config: $CFG"
  python "$REPO/repro_tools/stepB/preflight_check.py" "$CFG" 2>&1
else
  echo "no configs generated yet (run make_configs.py first) — skipping"
fi

echo
echo "---------- 6. Disk space ----------"
df -h "$DATA" 2>/dev/null | tail -2

echo
echo "=================================================="
echo " Done. Send this ENTIRE output back to Shaden."
echo "=================================================="

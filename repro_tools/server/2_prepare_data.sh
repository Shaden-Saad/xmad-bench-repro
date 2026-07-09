#!/usr/bin/env bash
# Step 2 (on the server): download the dataset and flatten it into ONE root folder.
# The sweep needs every dataset folder (commonvoice-*, mailabs-*, masc, aishell*, voxpopuli)
# directly under a single DATA_ROOT, because the code loads all languages from one root_path.
#
# The zips live in the authors' Google Drive. Easiest is to download them onto the server
# with `gdown` (installs via pip). You need each file's Drive ID, OR make your own copy.
set -e

DATA_ROOT="${1:-$HOME/xmad_data}"          # pass a path or default to ~/xmad_data
mkdir -p "$DATA_ROOT/_zips" "$DATA_ROOT"
cd "$DATA_ROOT/_zips"

echo "== Option A: download with gdown (fill in the Drive file IDs) =="
echo "   pip install gdown"
echo "   gdown --id <DE_ZIP_ID> -O de.zip   # repeat per language, or:"
echo "   gdown --folder https://drive.google.com/drive/folders/1PjboiIGjNWU6UeuIHrZu3ofF70o0A5-X"
echo "== Option B: upload the zips you already downloaded on the Mac via scp/rsync =="
echo "   (run on the Mac):  rsync -avP ~/Downloads/*.zip user@list-server:$DATA_ROOT/_zips/"
echo
read -p "Press Enter once the language zips are in $DATA_ROOT/_zips ..."

# Unzip everything
for z in *.zip; do echo "unzip $z"; unzip -q -o "$z"; done

# Flatten: move each real dataset folder up into DATA_ROOT (handles wrapper folders like de/, ar-005/)
find . -maxdepth 3 -type d \( -name 'commonvoice-*' -o -name 'mailabs-*' -o -iname 'masc*' \
   -o -iname 'aishell*' -o -iname 'voxpopuli*' \) -exec mv -v {} "$DATA_ROOT"/ \; 2>/dev/null || true

echo
echo "== Dataset folders now under $DATA_ROOT: =="
ls -1 "$DATA_ROOT" | grep -v _zips
echo
echo ">> Verify these names match make_configs.py's mapping (commonvoice-de, mailabs-de, masc, aishell-3, voxpopuli, ...)."
echo ">> If any differ (e.g. 'masc-xxx'), edit the LANGS mapping in stepB/make_configs.py accordingly."

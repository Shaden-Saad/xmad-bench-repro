# Adding the two missing detectors: wav2vec2 and Whisper+MLP

The released XMAD-Bench code contains only 4 of the paper's 6 detectors
(ResNet-18/50, SepTr, AST). This adds the other two, implemented faithfully to
the paper's description, so the sweep can cover the full Table 3.

## What was implemented
- **wav2vec 2.0** — a pretrained wav2vec2 encoder + a linear classifier, fine-tuned
  end-to-end on the raw 16 kHz waveform (paper: "consumes raw audio waveforms...
  requiring no handcrafted spectral preprocessing"). Temporal mean-pooling of the
  encoder outputs before the classifier.
- **Whisper+MLP** — the Whisper encoder is **frozen**; its outputs are aggregated
  by **Global Average Pooling over time** into one embedding, which trains a shallow
  **MLP** (paper 4.1). Only the MLP is trainable.

New file `detection/ssl_models.py` holds both model classes. `models_util.py`,
`main.py`, `data_manager.py`, `base_dataset.py`, `base_dataset_test.py` were patched
to add a raw-waveform / Whisper-feature data path and the two model builders.

## Verified
Both models were tested end-to-end in a sandbox with small checkpoints
(wav2vec2-base, whisper-tiny): forward + backward produce (B, 2) logits, the
wav2vec2 classifier receives gradients, the Whisper encoder is confirmed frozen
with gradients flowing only to the MLP, and the dataset returns the right shapes
(wav2vec2: raw waveform 80000; whisper: log-mel 80/128 x 3000). All files compile.

## Checkpoint assumption (documented reproducibility caveat)
The paper says these come "from Hugging Face" but does **not name the exact
checkpoints**. We use sensible, documented defaults:
- wav2vec2 -> `facebook/wav2vec2-xls-r-300m` (multilingual, matches XMAD-Bench's 7 languages)
- whisper  -> `openai/whisper-large-v3` (the encoder the paper names)
These are configurable via `wav2vec2_checkpoint` / `whisper_checkpoint` in config.json
(make_configs.py sets them). If a different checkpoint is intended, change it there
and note it as a deviation.

## How to apply and push (on your Mac)
1. Refresh repro_tools in your repo so it includes this script + updated make_configs:
   ```
   cd ~/xmad-repro/xmad-bench
   rm -rf repro_tools
   cp -R "$HOME/Library/CloudStorage/OneDrive-UniversityofAberdeen/PhD_Study/06_Writing/Paper-02_Reproducibility/XMAD-Bench_repro" repro_tools
   ```
2. Apply the model code to detection/:
   ```
   bash repro_tools/add_ssl_models.sh
   ```
   (Idempotent; prints "OK" for each file after a syntax check.)
3. (Recommended) quick smoke test on your Mac with small checkpoints, ~5 min:
   ```
   source ~/xmad-env/bin/activate
   cd detection
   python - <<'PY'
import json; c=json.load(open('config.json'))
c['exp_name']='de_w2v_smoke'; c['model_type']='wav2vec2'
c['wav2vec2_checkpoint']='facebook/wav2vec2-base'   # small, for the test only
c['dataset']['extract_samples']=True; c['dataset']['num_samples']=200
c['batch_size']=4; c['train_epochs']=1
json.dump(c,open('config.json','w'),indent=2)
PY
   PYTHONPATH="$HOME/xmad-repro/xmad-bench/detection" python main.py
   ```
   Expect training-loss lines, an `### EVAL::` line, and an `### Out of domain::` line.
4. Commit and push:
   ```
   cd ~/xmad-repro/xmad-bench
   git add -A
   git commit -m "Add wav2vec2 and Whisper+MLP detectors (all 6 models)"
   git push
   ```

After the push, the server sweep (make_configs + run_stepB) will automatically
include wav2vec2 and whisper -> 7 languages x 6 models x 3 runs = the full Table 3.
On the GPU server the SSL checkpoints download from Hugging Face on first use
(xls-r-300m ~1.2 GB, whisper-large-v3 ~3 GB) — ensure network access on the node.

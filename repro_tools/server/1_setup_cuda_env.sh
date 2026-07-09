#!/usr/bin/env bash
# Step 1 (on the LIST GPU server): build the Python environment with CUDA PyTorch.
# Run from the cloned repo root. Adjust the CUDA wheel (cu121) to match the server.
set -e

# If the server uses modules, you may first need e.g.:  module load cuda/12.1  (ask the admin)
nvidia-smi || { echo "No GPU visible — are you on a GPU node / did you request one?"; exit 1; }

python -m venv ~/xmad-env
source ~/xmad-env/bin/activate
pip install --upgrade pip

# CUDA PyTorch — change cu121 to match 'nvidia-smi' CUDA version (e.g. cu118, cu124)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Detection dependencies (torchaudio NOT needed)
pip install transformers einops pedalboard soundfile librosa pandas scikit-learn tensorboard tqdm

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| torch', torch.__version__)"
echo "Env ready. Activate later with: source ~/xmad-env/bin/activate"

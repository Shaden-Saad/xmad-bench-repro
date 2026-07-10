#!/usr/bin/env bash
# Adds the wav2vec2 and Whisper+MLP detectors to the XMAD-Bench detection code.
# Run ONCE from the repo root (the folder containing detection/):
#     bash repro_tools/add_ssl_models.sh
# Idempotent: exits if already applied. Tested end-to-end (forward+backward) for both models.
set -e
DET="detection"
[ -d "$DET" ] || { echo "Run this from the repo root (no detection/ folder here)."; exit 1; }

if grep -q "get_wav2vec2_model" "$DET/models_util.py"; then
  echo "Already applied (get_wav2vec2_model present). Nothing to do."; exit 0
fi

# 1) new model definitions
cat > "$DET/ssl_models.py" <<'PY'
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, WhisperModel


class Wav2Vec2Classifier(nn.Module):
    """wav2vec 2.0 fine-tuned end-to-end on raw waveforms (paper: HuggingFace wav2vec2)."""
    def __init__(self, checkpoint="facebook/wav2vec2-xls-r-300m", num_classes=2):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(checkpoint)
        self.fc = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T) normalised 16 kHz waveform
        hidden = self.encoder(x).last_hidden_state      # (B, L, H)
        pooled = hidden.mean(dim=1)                      # temporal mean-pool
        return self.fc(pooled)


class WhisperMLP(nn.Module):
    """Frozen Whisper-Large-v3 encoder + global average pooling + trainable MLP (paper 4.1)."""
    def __init__(self, checkpoint="openai/whisper-large-v3", hidden_mlp=256, num_classes=2):
        super().__init__()
        self.encoder = WhisperModel.from_pretrained(checkpoint).encoder
        for p in self.encoder.parameters():
            p.requires_grad = False                      # encoder stays frozen
        d = self.encoder.config.d_model
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden_mlp), nn.ReLU(), nn.Linear(hidden_mlp, num_classes)
        )

    def forward(self, x):
        # x: (B, n_mels, 3000) Whisper log-mel features
        with torch.no_grad():
            enc = self.encoder(x).last_hidden_state      # (B, L, d)
        pooled = enc.mean(dim=1)                          # Global Average Pooling over time
        return self.mlp(pooled)
PY
echo "wrote $DET/ssl_models.py"

# 2) patch the framework files (exact string replacements)
python3 - "$DET" <<'PY'
import sys, os
DET=sys.argv[1]
def patch(path, pairs):
    s=open(path).read()
    for old,new in pairs:
        assert old in s, f"PATTERN NOT FOUND in {path} (has the code changed?):\n---\n{old}\n---"
        s=s.replace(old,new,1)
    open(path,"w").write(s); print("patched",path)

mu=os.path.join(DET,"models_util.py"); s=open(mu).read()
s += '''

def get_wav2vec2_model(config):
    from ssl_models import Wav2Vec2Classifier
    ckpt = config.get("wav2vec2_checkpoint", "facebook/wav2vec2-xls-r-300m")
    return Wav2Vec2Classifier(ckpt).to(config['device'])


def get_whisper_mlp_model(config):
    from ssl_models import WhisperMLP
    ckpt = config.get("whisper_checkpoint", "openai/whisper-large-v3")
    return WhisperMLP(ckpt).to(config['device'])
'''
open(mu,"w").write(s); print("patched",mu)

patch(os.path.join(DET,"main.py"), [
("""    model_type = config.get("model_type", "resnet18")
    ast_proc = False
    if model_type == "resnet18":
        model = get_resnet18_model(config)
    elif model_type == "resnet50":
        model = get_resnet50_model(config)
    elif model_type == "septr":
        model = get_septr_model(config)
    elif model_type == "ast":
        model = get_ast_model(config)
        ast_proc = True""",
"""    model_type = config.get("model_type", "resnet18")
    proc = "spec"
    if model_type == "resnet18":
        model = get_resnet18_model(config)
    elif model_type == "resnet50":
        model = get_resnet50_model(config)
    elif model_type == "septr":
        model = get_septr_model(config)
    elif model_type == "ast":
        model = get_ast_model(config)
        proc = "ast"
    elif model_type == "wav2vec2":
        model = get_wav2vec2_model(config)
        proc = "wav2vec2"
    elif model_type == "whisper":
        model = get_whisper_mlp_model(config)
        proc = "whisper\""""),
("""    train_loader, validation_loader = data_manager.get_dataloaders(ast_proc)
    test_loader = data_manager.get_dataloader_test(ast_proc)""",
"""    train_loader, validation_loader = data_manager.get_dataloaders(proc)
    test_loader = data_manager.get_dataloader_test(proc)"""),
])

patch(os.path.join(DET,"data","data_manager.py"), [
("""    def get_dataloaders(self, ast_proc=False):
        train = BaseDataset(config=self.config, mode="train", ast_proc=ast_proc)
        test = BaseDataset(config=self.config, mode="test", ast_proc=ast_proc)""",
"""    def get_dataloaders(self, proc="spec"):
        train = BaseDataset(config=self.config, mode="train", proc=proc)
        test = BaseDataset(config=self.config, mode="test", proc=proc)"""),
("""    def get_dataloader_test(self, ast_proc=False):
        train = BaseDatasetTest(config=self.config, ast_proc=ast_proc)""",
"""    def get_dataloader_test(self, proc="spec"):
        train = BaseDatasetTest(config=self.config, proc=proc)"""),
])

init_old='''    def __init__(self, config, mode="train", ast_proc=False):
        self.ast_processor = None if ast_proc is False else ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

        self.config = config['dataset']'''
init_new='''    def __init__(self, config, mode="train", proc="spec"):
        self.proc = proc
        self.ast_processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593") if proc == "ast" else None
        self.w2v_processor = None
        self.whisper_processor = None
        if proc == "wav2vec2":
            from transformers import Wav2Vec2FeatureExtractor
            self.w2v_processor = Wav2Vec2FeatureExtractor.from_pretrained(config.get("wav2vec2_checkpoint", "facebook/wav2vec2-xls-r-300m"))
        elif proc == "whisper":
            from transformers import WhisperFeatureExtractor
            self.whisper_processor = WhisperFeatureExtractor.from_pretrained(config.get("whisper_checkpoint", "openai/whisper-large-v3"))

        self.config = config['dataset']'''
getitem_old='''        if self.ast_processor is not None:
            freq_domain = self.ast_processor(time_domain, sampling_rate=fs, return_tensors="np").data['input_values']
        else:
            freq_domain = self._get_freq_features(time_domain)[0]
        return freq_domain, label'''
getitem_new='''        if self.proc == "ast":
            freq_domain = self.ast_processor(time_domain, sampling_rate=fs, return_tensors="np").data['input_values']
        elif self.proc == "wav2vec2":
            freq_domain = self.w2v_processor(time_domain, sampling_rate=fs, return_tensors="np").input_values[0].astype(np.float32)
        elif self.proc == "whisper":
            freq_domain = self.whisper_processor(time_domain, sampling_rate=fs, return_tensors="np").input_features[0].astype(np.float32)
        else:
            freq_domain = self._get_freq_features(time_domain)[0]
        return freq_domain, label'''
patch(os.path.join(DET,"data","base_dataset.py"), [(init_old,init_new),(getitem_old,getitem_new)])

init_old2='''    def __init__(self, config, ast_proc=False):
        self.ast_processor = None if ast_proc is False else ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.config = config['dataset']'''
init_new2='''    def __init__(self, config, proc="spec"):
        self.proc = proc
        self.ast_processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593") if proc == "ast" else None
        self.w2v_processor = None
        self.whisper_processor = None
        if proc == "wav2vec2":
            from transformers import Wav2Vec2FeatureExtractor
            self.w2v_processor = Wav2Vec2FeatureExtractor.from_pretrained(config.get("wav2vec2_checkpoint", "facebook/wav2vec2-xls-r-300m"))
        elif proc == "whisper":
            from transformers import WhisperFeatureExtractor
            self.whisper_processor = WhisperFeatureExtractor.from_pretrained(config.get("whisper_checkpoint", "openai/whisper-large-v3"))
        self.config = config['dataset']'''
patch(os.path.join(DET,"data","base_dataset_test.py"), [(init_old2,init_new2),(getitem_old,getitem_new)])
print("ALL PATCHES APPLIED")
PY

echo "Syntax check:"
for f in ssl_models.py models_util.py main.py data/data_manager.py data/base_dataset.py data/base_dataset_test.py; do
  python3 -m py_compile "$DET/$f" && echo "  OK  $f" || echo "  FAIL $f"
done
echo "Done. wav2vec2 and whisper are now available as model_type values."

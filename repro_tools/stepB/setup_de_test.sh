#!/usr/bin/env bash
# One-shot: apply the 3 code fixes + write a German test config, then you run main.py.
# Assumes: local repo at ~/xmad-repro/xmad-bench  and German data at ~/Downloads/de
# Uses macOS (BSD) sed syntax.  Edit REPO / DATA below if your paths differ.
set -e
REPO="$HOME/xmad-repro/xmad-bench"
DATA="$HOME/Downloads/de"

echo "Applying code fixes..."
# Fix 1: make detection self-contained
sed -i '' 's/from detection\.data\.base_dataset import BaseDataset/from data.base_dataset import BaseDataset/' "$REPO/detection/data/data_manager.py"
sed -i '' 's/from detection\.data\.base_dataset_test import BaseDatasetTest/from data.base_dataset_test import BaseDatasetTest/' "$REPO/detection/data/data_manager.py"
sed -i '' 's/from detection\.septr import SeparableTr/from septr import SeparableTr/' "$REPO/detection/models_util.py"
sed -i '' 's/from detection\.ast_model import ASTModelLocal/from ast_model import ASTModelLocal/' "$REPO/detection/models_util.py"
# Fix 2: torch.load weights_only=False
sed -i '' "s/map_location=self.config\['device'\])/map_location=self.config['device'], weights_only=False)/g" "$REPO/detection/trainer.py"
# Fix 3: fewer DataLoader workers on a laptop
sed -i '' 's/num_workers=10/num_workers=2/g' "$REPO/detection/data/data_manager.py"

echo "Writing German test config..."
cat > "$REPO/detection/config.json" <<EOF
{
  "exp_name": "de_test01", "exp_path": "./experiments", "about_experiment": "German first reproduction test",
  "dataset": {
    "root_path": "$DATA",
    "dataset_train": "commonvoice-de", "dataset_test": "mailabs-de",
    "multilingual": true, "train_datasets": ["commonvoice-de"], "test_datasets": ["mailabs-de"],
    "extract_samples": true, "num_samples": 1000,
    "augment_chance": 0.5,
    "noise_augment": true, "noise_augment_chance": 0.0, "gaussian_noise_augment_chance": 0.15,
    "time_augment": true, "time_augment_chance": 0.2, "spec_augm": false, "spec_chance": 0.05,
    "mixup_augment": false, "mixup_augment_chance": 0.2, "speed_augment": true, "speed_augment_chance": 0.15,
    "volume_augment": true, "volume_augment_chance": 0.2, "clipping": true, "clipping_augment_chance": 0.2,
    "reverb": true, "reverb_augment_chance": 0.2, "spectral_shift": true,
    "spectral_shift_augment_chance_HP": 0.15, "spectral_shift_augment_chance_LP": 0.15,
    "spectral_shift_augment_chance_peak": 0.15, "pitch": true, "pitch_augment_chance": 0.2
  },
  "model_type": "resnet18", "lr": 0.0005, "lr_sch_step": 50, "lr_sch_gamma": 0.5,
  "weight_decay": 0.0, "batch_size": 16, "loss_function": "bce_loss",
  "train_epochs": 3, "eval_net_epoch": 1, "save_net_epochs": 2, "print_loss": 20, "resume_training": false
}
EOF

echo "Done. Now run:"
echo "  cd $REPO/detection"
echo "  PYTHONPATH=\"$REPO/detection\" python main.py"

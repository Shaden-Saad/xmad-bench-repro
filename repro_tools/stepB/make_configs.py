#!/usr/bin/env python3
"""Generate ready-to-run config.json files for XMAD-Bench Step B.

Per-language reproduction uses MULTILINGUAL mode with single-element lists.
Why: the multilingual code path applies the correct CSV separator (TAB for
commonvoice-en/ru and mailabs-en/ru, comma otherwise). The single-language
path does NOT handle the tab separator, so it silently misreads those sets.

IMPORTANT: the dataset FOLDER NAMES below are inferred from the paper (Table 1)
and the repo's default config. Confirm them against the actual Google Drive
download and edit the mapping if they differ.
"""
import json, os, copy

# language -> (in-domain CommonVoice train set, cross-domain test set)
LANGS = {
    "ar": ("commonvoice-ar", "masc"),
    "en": ("commonvoice-en", "mailabs-en"),
    "de": ("commonvoice-de", "mailabs-de"),
    "zh": ("commonvoice-zh", "aishell-3"),
    "ro": ("commonvoice-ro", "voxpopuli"),
    "ru": ("commonvoice-ru", "mailabs-ru"),
    "es": ("commonvoice-es", "mailabs-es"),
}
MODELS = ["resnet18", "resnet50", "septr", "ast"]  # wav2vec2 / whisper need extra wiring

BASE = {
    "exp_name": "TO_SET", "exp_path": "./experiments", "about_experiment": "XMAD-Bench Step B repro",
    "dataset": {
        "root_path": "/path/to/data",
        "dataset_train": "commonvoice-ro", "dataset_test": "voxpopuli",
        "multilingual": True, "train_datasets": ["commonvoice-ro"], "test_datasets": ["voxpopuli"],
        "extract_samples": False, "num_samples": 4000,
        "augment_chance": 0.5,
        "noise_augment": True, "noise_augment_chance": 0.00, "gaussian_noise_augment_chance": 0.15,
        "time_augment": True, "time_augment_chance": 0.20,
        "spec_augm": False, "spec_chance": 0.05,
        "mixup_augment": False, "mixup_augment_chance": 0.20,
        "speed_augment": True, "speed_augment_chance": 0.15,
        "volume_augment": True, "volume_augment_chance": 0.20,
        "clipping": True, "clipping_augment_chance": 0.20,
        "reverb": True, "reverb_augment_chance": 0.20,
        "spectral_shift": True, "spectral_shift_augment_chance_HP": 0.15,
        "spectral_shift_augment_chance_LP": 0.15, "spectral_shift_augment_chance_peak": 0.15,
        "pitch": True, "pitch_augment_chance": 0.20,
    },
    "model_type": "ast", "lr": 0.0005, "lr_sch_step": 50, "lr_sch_gamma": 0.5,
    "weight_decay": 0.0, "batch_size": 10, "loss_function": "bce_loss",
    "train_epochs": 20, "eval_net_epoch": 1, "save_net_epochs": 2,
    "print_loss": 20, "resume_training": False,
}
# paper's per-model batch sizes
BATCH = {"resnet18": 200, "resnet50": 120, "septr": 10, "ast": 10}  # wav2vec2: 16

def main(root_path="/path/to/data", out_dir="configs"):
    os.makedirs(out_dir, exist_ok=True)
    for lang, (train_ds, test_ds) in LANGS.items():
        for model in MODELS:
            c = copy.deepcopy(BASE)
            c["dataset"]["root_path"] = root_path
            c["dataset"]["dataset_train"] = train_ds
            c["dataset"]["dataset_test"] = test_ds
            c["dataset"]["train_datasets"] = [train_ds]
            c["dataset"]["test_datasets"] = [test_ds]
            c["model_type"] = model
            c["batch_size"] = BATCH[model]
            c["exp_name"] = f"{lang}_{model}"
            with open(os.path.join(out_dir, f"config_{lang}_{model}.json"), "w") as f:
                json.dump(c, f, indent=2)
    # cross-lingual: train on ar,de,ro,ru,es ; test on en,zh (paper 4.2)
    for model in MODELS:
        c = copy.deepcopy(BASE)
        c["dataset"]["root_path"] = root_path
        c["dataset"]["multilingual"] = True
        c["dataset"]["train_datasets"] = ["commonvoice-ar","commonvoice-de","commonvoice-ro","commonvoice-ru","commonvoice-es"]
        c["dataset"]["test_datasets"] = ["commonvoice-en","commonvoice-zh"]
        c["dataset"]["extract_samples"] = True
        c["dataset"]["num_samples"] = 6000   # ~3000 real + 3000 fake per lang; confirm vs paper
        c["model_type"] = model
        c["batch_size"] = BATCH[model]
        c["exp_name"] = f"crosslingual_{model}"
        with open(os.path.join(out_dir, f"config_crosslingual_{model}.json"), "w") as f:
            json.dump(c, f, indent=2)
    print(f"Wrote {len(LANGS)*len(MODELS)+len(MODELS)} configs to {out_dir}/")

if __name__ == "__main__":
    import sys
    main(root_path=sys.argv[1] if len(sys.argv) > 1 else "/path/to/data")

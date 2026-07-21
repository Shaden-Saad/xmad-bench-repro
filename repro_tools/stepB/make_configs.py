#!/usr/bin/env python3
"""Generate ready-to-run config.json files for XMAD-Bench Step B.

Per-language reproduction uses MULTILINGUAL mode with single-element lists
(it is the mode the repo's default config uses, and it applies the split filter
consistently).

CSV SEPARATOR — VERIFIED 2026-07, IMPORTANT:
every meta.csv in the released data is COMMA-separated. The multilingual code
path nevertheless hard-codes sep='\t' for commonvoice-en / commonvoice-ru
(base_dataset.py) and mailabs-en / mailabs-ru (base_dataset_test.py). Those
files then load as a single column and the next line raises KeyError:'is_fake'.
=> English, Russian and the cross-lingual experiment CANNOT RUN as released
   (18 of the 48 Table-3 cells). Apply the separator fix (see
   CODE_FIXES_APPLIED.md, fix 7) before running those languages.
German/Spanish/Romanian/Arabic/Mandarin are unaffected (comma path).

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
    # VERIFIED against the data on MeluXina: the folder is "aishell3" (no hyphen).
    # An earlier local edit had "aishell-3", which fails preflight. The 12 completed
    # Mandarin runs record 'test': 'aishell3'.
    "zh": ("commonvoice-zh", "aishell3"),
    "ro": ("commonvoice-ro", "voxpopuli"),
    "ru": ("commonvoice-ru", "mailabs-ru"),
    "es": ("commonvoice-es", "mailabs-es"),
}
CORE_MODELS = ["resnet18", "resnet50", "septr", "ast"]   # always available
SSL_MODELS = ["wav2vec2", "whisper"]                     # only after add_ssl_models.sh
MODELS = CORE_MODELS  # default; override per call with the `models` argument

# SSL checkpoints (paper does not name exact checkpoints; documented assumptions):
#   wav2vec2 -> multilingual XLS-R 300M ; whisper -> Whisper-Large-v3 (as the paper states)
WAV2VEC2_CKPT = "facebook/wav2vec2-xls-r-300m"
WHISPER_CKPT = "openai/whisper-large-v3"

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
    "wav2vec2_checkpoint": WAV2VEC2_CKPT, "whisper_checkpoint": WHISPER_CKPT,
    "model_type": "ast", "lr": 0.0005, "lr_sch_step": 50, "lr_sch_gamma": 0.5,
    "weight_decay": 0.0, "batch_size": 10, "loss_function": "bce_loss",
    "train_epochs": 20, "eval_net_epoch": 1, "save_net_epochs": 2,
    "print_loss": 20, "resume_training": False,
}
# paper's per-model batch sizes
BATCH = {"resnet18": 200, "resnet50": 120, "septr": 10, "ast": 10, "wav2vec2": 16, "whisper": 16}
# (paper gives 16 for wav2vec2; Whisper+MLP unspecified -> 16 used, safe as its encoder is frozen)

def main(root_path="/path/to/data", out_dir="configs", langs=None, cross_lingual=None, models=None):
    # langs: optional subset of language codes to generate (default: all 7).
    #        e.g. ["ar","en","de","zh"] for node 1, ["ro","ru","es"] for node 2.
    # models: optional list (default CORE_MODELS = the 4 wired detectors).
    #        pass CORE_MODELS + SSL_MODELS once add_ssl_models.sh has been applied.
    models = models or MODELS
    langs = list(LANGS.keys()) if not langs else [l for l in langs if l in LANGS]
    # cross-lingual only when the full set is generated, unless forced
    if cross_lingual is None:
        cross_lingual = set(langs) == set(LANGS.keys())
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for lang in langs:
        train_ds, test_ds = LANGS[lang]
        for model in models:
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
            n += 1
    # cross-lingual: train on ar,de,ro,ru,es ; test on en,zh (paper 4.2)
    for model in ([] if not cross_lingual else models):
        c = copy.deepcopy(BASE)
        c["dataset"]["root_path"] = root_path
        c["dataset"]["multilingual"] = True
        c["dataset"]["train_datasets"] = ["commonvoice-ar","commonvoice-de","commonvoice-ro","commonvoice-ru","commonvoice-es"]
        c["dataset"]["test_datasets"] = ["commonvoice-en","commonvoice-zh"]
        c["dataset"]["extract_samples"] = True
        # Paper 4.2: "we randomly select at most 3,000 samples per language" — ambiguous
        # as to per-class or total. CONFIRMED 2026-07-21 (Yuejun Guo) as 3,000 in TOTAL,
        # and verified against the loader: base_dataset.py samples num_samples//2 names
        # from the REAL rows only, then re-filters on those names; because XMAD pairs
        # real and fake under a shared sample_name, both members come back. So
        # num_samples is already a total => 3000 gives ~1500 real + 1500 fake.
        # (Only read when extract_samples is True, i.e. the cross-lingual configs.)
        c["dataset"]["num_samples"] = 3000
        c["model_type"] = model
        c["batch_size"] = BATCH[model]
        c["exp_name"] = f"crosslingual_{model}"
        with open(os.path.join(out_dir, f"config_crosslingual_{model}.json"), "w") as f:
            json.dump(c, f, indent=2)
        n += 1
    print(f"Wrote {n} configs to {out_dir}/  (languages: {', '.join(langs)}"
          f"{'; + cross-lingual' if cross_lingual else ''})")

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "/path/to/data"
    out  = sys.argv[2] if len(sys.argv) > 2 else "configs"
    langs = sys.argv[3].replace(",", " ").split() if len(sys.argv) > 3 and sys.argv[3] else None
    models = sys.argv[4].replace(",", " ").split() if len(sys.argv) > 4 and sys.argv[4] else None
    main(root_path=root, out_dir=out, langs=langs, models=models)

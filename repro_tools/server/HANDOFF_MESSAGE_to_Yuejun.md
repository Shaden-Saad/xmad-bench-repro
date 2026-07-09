# Message to send Yuejun

Subject: XMAD-Bench reproduction — run on LIST GPU

Hi Yuejun,

Thanks for helping run this on the LIST GPU server. Everything is prepared — code
and scripts are on GitHub:

  https://github.com/Shaden-Saad/xmad-bench-repro   (branch: reproduction)

Please follow the step-by-step guide in the repo at:
  repro_tools/server/HANDOVER_FOR_YUEJUN.md

In short:
  1. Clone the repo and check out the `reproduction` branch.
  2. Run the three numbered scripts in repro_tools/server/ (environment -> data -> sweep).
  3. Send me back the file it produces: results_full_sweep.csv

Scope of this run: 7 languages x 4 models (ResNet-18, ResNet-50, SepTr, AST) x 3 runs.
(Two more models — wav2vec2 and Whisper+MLP — are not in the released code; I'll add
those in a second round, so they're out of scope for now.)
Estimated compute: roughly 50-110 GPU-hours in total.

Two things to sort at the start:
  - Dataset (~80 GB): you'll need the seven language ZIPs. I can share the Google
    Drive folder or send the zips — tell me which you prefer.
  - Two spots adapt to LIST: the CUDA version (step 4) and, if LIST uses SLURM, the
    partition name (step 6). The guide explains exactly where.

If anything errors, send me the message and the folder-name list the data script
prints — common fixes are noted in repro_tools/stepB/CODE_FIXES_APPLIED.md.

Thanks so much!
Shaden

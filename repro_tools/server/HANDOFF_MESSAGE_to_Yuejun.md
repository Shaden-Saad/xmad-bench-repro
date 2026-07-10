# Message to send Yuejun

Subject: XMAD-Bench reproduction — run on the GPU server

Hi Yuejun,

Thanks for helping run this on the GPU server we've been granted access to.
Everything is prepared — code and scripts are on GitHub (public repo):

  https://github.com/Shaden-Saad/xmad-bench-repro   (branch: reproduction)

Please follow the step-by-step guide in the repo at:
  repro_tools/server/HANDOVER_FOR_YUEJUN.md

In short:
  1. Clone the repo and check out the `reproduction` branch.
  2. Run the three numbered scripts in repro_tools/server/ (environment -> data -> sweep).
  3. Send me back the file it produces: results_full_sweep.csv

Scope: 7 languages x 4 models (ResNet-18, ResNet-50, SepTr, AST) x 3 runs
(~50-110 GPU-hours). I'm adding two more models (wav2vec2, Whisper+MLP) shortly —
I'll let you know if they're ready before you start; otherwise we'll do them in a
second round.

The dataset (~80 GB) downloads directly onto the server from this link — the guide
has the exact command:
  https://drive.google.com/drive/folders/1PjboiIGjNWU6UeuIHrZu3ofF70o0A5-X

Two spots adapt to the server (the CUDA version, and — if the server uses SLURM —
the partition name); the guide explains where. Any errors, just send them my way.
Thanks so much!

Shaden

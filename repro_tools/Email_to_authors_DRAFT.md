# Draft email to XMAD-Bench authors

To:   raducu.ionescu@gmail.com  (corresponding author; consider cc'ing the first authors)
Subject: Request for wav2vec 2.0 and Whisper+MLP baseline code — XMAD-Bench reproduction

---

Dear Dr. Ionescu and colleagues,

I am a PhD student at the University of Aberdeen carrying out a reproducibility
study of XMAD-Bench (arXiv:2506.00462). First, thank you for releasing the
benchmark, dataset, and detection code so openly — it made this work possible.
We have already reproduced your German results for ResNet-18 and ResNet-50
closely (in-domain ~100%, cross-domain ~97%), consistent with Table 3.

We are now extending the reproduction to the remaining detectors. The released
`detection/` framework implements ResNet-18/50, SepTr, and AST, but we were
unable to find the model/training code for two baselines reported in Table 3:

  1. wav2vec 2.0 (Baevski et al., 2020; Tak et al., 2022)
  2. Whisper-Large-v3 + MLP (frozen encoder + shallow MLP)

Would you be able to share the code for these two baselines (for example, a
branch or commit, or the relevant model files)? To reproduce them faithfully,
it would also help to confirm a few details:

  - The exact pretrained checkpoint used for wav2vec 2.0 — e.g. a multilingual
    XLS-R model, or the wav2vec2 front-end + AASIST variant from Tak et al. 2022.
  - For Whisper+MLP: the pooling and MLP configuration, and confirmation that the
    Whisper encoder is fully frozen during fine-tuning.
  - (Minor) the per-language sample budget for the cross-lingual experiment
    (Section 4.2 mentions "at most 3,000 samples per language").

We are glad to acknowledge your help in any resulting write-up, and to share our
reproduction notes if useful. Thank you again for the open release and for any
pointers you can provide.

Best regards,
Shaden Saad
PhD Student, University of Aberdeen
r02sa25@abdn.ac.uk

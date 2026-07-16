#!/usr/bin/env python3
"""Make training RESUMABLE, so a job killed by the SLURM time limit loses at most
one epoch instead of the whole run.

THE PROBLEM WITH THE RELEASED CODE
----------------------------------
detection/trainer.py already has a `resume_training` flag, but it is broken for
reproduction. It loads the checkpoint and then still runs a FULL train_epochs loop:

    if self.config['resume_training'] is True:
        checkpoint = torch.load(... 'latest_checkpoint.pkl' ...)
        self.network.load_state_dict(checkpoint['model_weights'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    for i in range(1, self.config['train_epochs'] + 1):     # <-- always 1..20

So resuming a run killed at epoch 14 trains 14 + 20 = 34 epochs, not 20. The
resulting model is NOT the paper's 20-epoch model. It also:
  - never restores the LR scheduler, and
  - never restores best_metric, so best_model.pkl can be OVERWRITTEN by a worse
    epoch after the resume (best_metric restarts at 0.0).

WHAT THIS PATCH CHANGES (3 edits, all in detection/trainer.py)
-------------------------------------------------------------
1. train() starts at checkpoint['epoch'] + 1 and stops at train_epochs, so the
   total is always exactly train_epochs. If training already finished, the loop
   is skipped and it goes straight to the cross-domain test.
2. The resume restores lr_scheduler state and best_metric (taking the max of the
   checkpoint value and best_model.pkl's stored 'stats', which also repairs
   checkpoints written by the old code).
3. The 'latest' checkpoint is saved at the END of the epoch (after eval and
   lr_scheduler.step()) instead of in the middle, so a checkpoint always means
   "epoch N is completely finished". The training maths for an uninterrupted run
   is unchanged - only the moment the checkpoint file is written moves.

DEVIATION TO DISCLOSE
---------------------
A resumed run is a valid train_epochs-epoch training, but it is not bit-identical
to an uninterrupted one: RNG state for shuffling and augmentation is not restored.
This matters little here because the repo sets NO random seed, so every run is
already a fresh random draw - which is exactly why the paper reports mean +/- std
over 3 runs. Resumed runs stay a legitimate sample.

Idempotent: safe to run repeatedly. Usage:
    python fix_resume_training.py /path/to/xmad-bench-repro
"""
import os
import re
import sys

MARKER = "# [repro] resumable training"

OLD_TRAIN = """    def train(self):
        if self.config['resume_training'] is True:
            checkpoint = torch.load(os.path.join(self.config['exp_path'],
                                                 self.config['exp_name'],
                                                 'latest_checkpoint.pkl'),
                                    map_location=self.config['device'], weights_only=False)
            self.network.load_state_dict(checkpoint['model_weights'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        for i in range(1, self.config['train_epochs'] + 1):
            print('Training on epoch ' + str(i))
            self.train_epoch(i)
            self.save_net_state(i, latest=True)

            if i % self.config['eval_net_epoch'] == 0:
                self.eval_net(i)

            if i % self.config['save_net_epochs'] == 0:
                self.save_net_state(i)

            self.lr_scheduler.step()
"""

NEW_TRAIN = '''    def train(self):
        {marker}
        # Resume where the last checkpoint left off, so the TOTAL number of epochs is
        # always train_epochs. (The released code re-ran the full 1..train_epochs loop
        # on top of the loaded weights, which trained ~2x the intended epochs.)
        start_epoch = 1
        if self.config.get('resume_training') is True:
            ckpt_path = os.path.join(self.config['exp_path'], self.config['exp_name'],
                                     'latest_checkpoint.pkl')
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=self.config['device'],
                                        weights_only=False)
                self.network.load_state_dict(checkpoint['model_weights'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if 'lr_scheduler' in checkpoint:
                    self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                # Restore the best score, otherwise best_model.pkl would be overwritten
                # by the first post-resume epoch even if it is worse. best_model.pkl's
                # 'stats' is authoritative (written exactly when best_metric improves),
                # so take the max - this also repairs old-format checkpoints.
                self.best_metric = checkpoint.get('best_metric', 0.0)
                best_path = os.path.join(self.config['exp_path'], self.config['exp_name'],
                                         'best_model.pkl')
                if os.path.exists(best_path):
                    prev = torch.load(best_path, map_location='cpu', weights_only=False)
                    self.best_metric = max(self.best_metric, prev.get('stats', 0.0))
                start_epoch = checkpoint['epoch'] + 1
                msg = (f"### RESUME:: checkpoint at epoch {{checkpoint['epoch']}}; continuing at "
                       f"epoch {{start_epoch}} of {{self.config['train_epochs']}} "
                       f"(best ACC so far = {{self.best_metric}})")
                print(msg, flush=True)
                save_logs_eval(os.path.join(self.config['exp_path'], self.config['exp_name']), msg)
            else:
                print('### RESUME:: resume_training=True but no latest_checkpoint.pkl found; '
                      'training from scratch', flush=True)

        if start_epoch > self.config['train_epochs']:
            print(f"### RESUME:: all {{self.config['train_epochs']}} epochs already done; "
                  f"skipping training, going straight to the cross-domain test", flush=True)

        for i in range(start_epoch, self.config['train_epochs'] + 1):
            print('Training on epoch ' + str(i))
            self.train_epoch(i)

            if i % self.config['eval_net_epoch'] == 0:
                self.eval_net(i)

            if i % self.config['save_net_epochs'] == 0:
                self.save_net_state(i)

            self.lr_scheduler.step()

            # Saved LAST, so "checkpoint epoch i" means epoch i is fully complete
            # (trained + evaluated + scheduler stepped). Resuming at i+1 is then exact.
            self.save_net_state(i, latest=True)
'''.format(marker=MARKER)

OLD_SAVE = """            to_save = {
                'epoch': epoch,
                'model_weights': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }"""

NEW_SAVE = """            to_save = {
                'epoch': epoch,
                'model_weights': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                # needed for an exact resume (see fix_resume_training.py)
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'best_metric': self.best_metric,
            }"""


def norm(s):
    """Compare ignoring trailing whitespace differences."""
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


def patch(path):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print(f"  already patched: {path}")
        return True

    out = src
    ok = True

    # --- edit 1+3: the train() method -------------------------------------
    if norm(OLD_TRAIN) in norm(out):
        # operate on the real text, tolerant of trailing spaces
        pat = re.compile(re.escape(OLD_TRAIN.rstrip()).replace(r"\ \n", r"\s*\n"))
        new_out, n = pat.subn(NEW_TRAIN.rstrip(), out, count=1)
        if n == 1:
            out = new_out
        else:
            out = out.replace(OLD_TRAIN.rstrip(), NEW_TRAIN.rstrip(), 1)
    elif OLD_TRAIN.rstrip() in out:
        out = out.replace(OLD_TRAIN.rstrip(), NEW_TRAIN.rstrip(), 1)
    else:
        print("  ERROR: train() does not match the expected released code.")
        print("         Not patching - inspect detection/trainer.py by hand.")
        ok = False

    # --- edit 2: save lr_scheduler + best_metric in the latest checkpoint --
    if ok:
        if OLD_SAVE in out:
            out = out.replace(OLD_SAVE, NEW_SAVE, 1)
        elif "'lr_scheduler': self.lr_scheduler.state_dict()" in out:
            pass  # already there
        else:
            print("  ERROR: save_net_state() 'latest' branch does not match.")
            ok = False

    if not ok:
        return False

    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"  patched: {path}")
    return True


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    trainer = os.path.join(repo, "detection", "trainer.py")
    if not os.path.isfile(trainer):
        sys.exit(f"trainer.py not found at {trainer}")

    print("Applying resumable-training patch...")
    if not patch(trainer):
        sys.exit(1)

    # sanity: the file must still be valid Python
    import ast
    with open(trainer, encoding="utf-8") as f:
        ast.parse(f.read())
    print("  trainer.py is valid Python.")
    print("Done. Killed runs now continue from their last completed epoch.")


if __name__ == "__main__":
    main()

"""
trainer.py  ─  Training loop, evaluation, and checkpointing
═══════════════════════════════════════════════════════════════════════════════
Fixes vs original
──────────────────
  ① CRITICAL — LR scheduler was stepped once per epoch but WARMUP_STEPS was
    400.  With 100 epochs the model never completed warmup; LR peaked at only
    100/400 = 25 % of the target value.  Scheduler now steps once per BATCH
    (gradient update step), so warmup completes in the correct number of
    actual update steps regardless of epoch length.

  ② AMP (Automatic Mixed Precision) support for CUDA — roughly 2× throughput
    on modern GPUs with no accuracy cost.

  ③ Full checkpoint resume: last_checkpoint.pt saves model, optimiser, and
    scheduler state so training can be interrupted and resumed exactly.

  ④ Word accuracy added alongside CER.  An exact-match metric is far more
    interpretable: "87 % of words transliterated perfectly."

  ⑤ MAX_EVAL_SAMPLES = 0 → evaluate on ALL validation pairs (no cap).

  ⑥ Validation CER is computed over the full validation set, not a random
    sub-sample of 200, giving a stable, reproducible metric.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset import CharTokenizer
from model import Seq2SeqTransformer, greedy_decode


# ─── String-distance metrics ─────────────────────────────────────────────────

def levenshtein(a: str, b: str) -> int:
    """Standard dynamic-programming Levenshtein distance."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def cer(pred: str, target: str) -> float:
    """Character Error Rate ∈ [0, 1]."""
    if not target:
        return 0.0 if not pred else 1.0
    return min(levenshtein(pred, target) / len(target), 1.0)


# ─── One training epoch ───────────────────────────────────────────────────────

def train_epoch(
    model:     Seq2SeqTransformer,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    criterion: nn.Module,
    device:    torch.device,
    scaler:    Optional[GradScaler] = None,
    grad_clip: float = 1.0,
) -> tuple[float, int]:
    """
    Run one full pass over the training set.

    Returns
    -------
    avg_loss
        Per-token cross-entropy (excluding PAD).
    total_steps
        Number of optimiser steps taken (= number of batches).

    FIX: scheduler.step() is called INSIDE the batch loop (once per
    gradient update) so that WARMUP_STEPS refers to real update steps,
    not epochs.  The original code stepped once per epoch, making warmup
    400× too slow.
    """
    model.train()
    total_loss   = 0.0
    total_tokens = 0
    steps        = 0

    for src, tgt, src_pad, tgt_pad in loader:
        src     = src.to(device, non_blocking=True)
        tgt     = tgt.to(device, non_blocking=True)
        src_pad = src_pad.to(device, non_blocking=True)
        tgt_pad = tgt_pad.to(device, non_blocking=True)

        # Teacher forcing: feed SOS…cn, predict c1…EOS
        tgt_in     = tgt[:, :-1]           # (B, S-1) — decoder input
        tgt_target = tgt[:, 1:]            # (B, S-1) — labels
        tgt_in_pad = tgt_pad[:, :-1]

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # AMP: forward in fp16/bf16, loss scaling to avoid underflow
            with autocast():
                logits = model(src, tgt_in,
                               src_pad_mask=src_pad,
                               tgt_pad_mask=tgt_in_pad)
                B, S, V = logits.shape
                loss = criterion(logits.reshape(B * S, V), tgt_target.reshape(B * S))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(src, tgt_in,
                           src_pad_mask=src_pad,
                           tgt_pad_mask=tgt_in_pad)
            B, S, V = logits.shape
            loss = criterion(logits.reshape(B * S, V), tgt_target.reshape(B * S))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # FIX: step the LR scheduler per batch, not per epoch
        scheduler.step()
        steps += 1

        n_tok = tgt_target.ne(0).sum().item()
        total_loss   += loss.item() * n_tok
        total_tokens += n_tok

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, steps


# ─── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:       Seq2SeqTransformer,
    val_pairs:   list[tuple[str, str]],
    src_tok:     CharTokenizer,
    tgt_tok:     CharTokenizer,
    device:      torch.device,
    max_samples: int = 0,           # 0 = use ALL validation pairs
    n_examples:  int = 5,           # number of printed examples
) -> tuple[float, float, list[tuple[str, str, str]]]:
    """
    Evaluate on validation pairs using fast greedy decoding.

    Returns
    -------
    corpus_cer  : float in [0, 1]  — lower is better
    word_acc    : float in [0, 1]  — higher is better (exact word match)
    examples    : list of (roman, predicted, target) for display
    """
    model.eval()

    # Use all pairs unless a cap is specified
    sample_pool = val_pairs if max_samples <= 0 else val_pairs[:max_samples]

    sos = tgt_tok.ch2id["<SOS>"]
    eos = tgt_tok.ch2id["<EOS>"]

    total_edit = 0
    total_len  = 0
    exact_hits = 0
    examples: list[tuple[str, str, str]] = []

    for roman, target in sample_pool:
        src_ids = src_tok.encode(roman, add_sos=False, add_eos=True)
        out_ids = greedy_decode(model, src_ids, sos, eos, device)
        pred    = tgt_tok.decode(out_ids)

        dist = levenshtein(pred, target)
        total_edit += dist
        total_len  += max(len(target), 1)
        if pred == target:
            exact_hits += 1

        if len(examples) < n_examples:
            examples.append((roman, pred, target))

    corpus_cer = total_edit / max(total_len, 1)
    word_acc   = exact_hits / max(len(sample_pool), 1)
    return corpus_cer, word_acc, examples


# ─── Trainer ─────────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages the full training lifecycle:
      • one optimiser (AdamW) with warmup + cosine LR decay
      • optional AMP (CUDA only)
      • best model checkpointing (by validation CER)
      • periodic full checkpoints (model + optimiser + scheduler)
      • early stopping
      • JSON training log
    """

    def __init__(
        self,
        model:           Seq2SeqTransformer,
        train_loader:    DataLoader,
        val_pairs:       list[tuple[str, str]],
        src_tok:         CharTokenizer,
        tgt_tok:         CharTokenizer,
        device:          torch.device,
        model_dir:       str   = "models",
        lr:              float = 3e-4,
        warmup_steps:    int   = 2000,   # gradient-update steps, NOT epochs
        patience:        int   = 20,     # validation rounds, NOT epochs
        label_smoothing: float = 0.1,
        grad_clip:       float = 1.0,
        weight_decay:    float = 1e-4,
        use_amp:         bool  = True,
        eval_every:      int   = 5,
        save_every:      int   = 10,
        max_eval_samples:int   = 0,
    ) -> None:

        self.model        = model
        self.train_loader = train_loader
        self.val_pairs    = val_pairs
        self.src_tok      = src_tok
        self.tgt_tok      = tgt_tok
        self.device       = device
        self.model_dir    = model_dir
        self.patience     = patience
        self.grad_clip    = grad_clip
        self.eval_every   = eval_every
        self.save_every   = save_every
        self.max_eval_samples = max_eval_samples

        os.makedirs(model_dir, exist_ok=True)

        # ── Loss ─────────────────────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,
            label_smoothing=label_smoothing,
        )

        # ── Optimiser ────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=weight_decay,
        )

        # ── LR schedule: linear warmup → cosine decay ─────────────────
        # FIX: warmup_steps is now in BATCH steps.
        # The cosine tail decays over 10 × warmup_steps more steps,
        # then floors at 10 % of peak LR to prevent collapse.
        cosine_steps = max(1, warmup_steps * 10)

        def _lr_lambda(step: int) -> float:
            step = max(step, 1)
            if step < warmup_steps:
                return step / warmup_steps
            t = (step - warmup_steps) / cosine_steps
            return max(0.10, 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0))))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=_lr_lambda
        )

        # ── AMP ──────────────────────────────────────────────────────────
        self._use_amp = use_amp and (device.type == "cuda")
        self.scaler: Optional[GradScaler] = GradScaler() if self._use_amp else None

        # ── State ─────────────────────────────────────────────────────────
        self.history:    list[dict] = []
        self.best_cer:   float      = float("inf")
        self.no_improve: int        = 0
        self.global_step: int       = 0     # total gradient-update steps

    # ── Paths ─────────────────────────────────────────────────────────────────

    def _best_path(self) -> str:
        return os.path.join(self.model_dir, "best_model.pt")

    def _last_path(self) -> str:
        return os.path.join(self.model_dir, "last_checkpoint.pt")

    def _periodic_path(self, epoch: int) -> str:
        return os.path.join(self.model_dir, f"checkpoint_epoch{epoch:04d}.pt")

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def _save_best(self) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "best_cer":    self.best_cer,
                "global_step": self.global_step,
            },
            self._best_path(),
        )

    def _save_full(self, epoch: int, path: str) -> None:
        """Save everything needed to resume training exactly."""
        torch.save(
            {
                "epoch":           epoch,
                "global_step":     self.global_step,
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state":    self.scaler.state_dict() if self.scaler else None,
                "best_cer":        self.best_cer,
                "no_improve":      self.no_improve,
                "history":         self.history,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> int:
        """
        Load a full checkpoint.  Returns the epoch to resume FROM
        (i.e., the next epoch to run).
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if self.scaler and ckpt.get("scaler_state"):
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.best_cer    = ckpt["best_cer"]
        self.no_improve  = ckpt["no_improve"]
        self.global_step = ckpt["global_step"]
        self.history     = ckpt.get("history", [])
        resume_epoch     = ckpt["epoch"] + 1
        print(f"  Resumed from epoch {ckpt['epoch']}  "
              f"(best CER {self.best_cer:.4f}, step {self.global_step})")
        return resume_epoch

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, epochs: int = 100, start_epoch: int = 1) -> None:
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("\n" + "═" * 70)
        print("  TRAINING")
        print("═" * 70)
        print(f"  Device       : {self.device}")
        print(f"  AMP          : {self._use_amp}")
        print(f"  Parameters   : {n_params:,}")
        print(f"  Max epochs   : {epochs}")
        print(f"  Eval every   : {self.eval_every} epochs")
        print(f"  Patience     : {self.patience} eval rounds")
        print(f"  Warmup steps : {self.scheduler.base_lrs}")
        print("═" * 70 + "\n")

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()

            train_loss, batch_steps = train_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.scheduler,
                self.criterion,
                self.device,
                scaler=self.scaler,
                grad_clip=self.grad_clip,
            )
            self.global_step += batch_steps

            # ── Periodic full checkpoint ───────────────────────────────────
            if epoch % self.save_every == 0:
                self._save_full(epoch, self._periodic_path(epoch))
            self._save_full(epoch, self._last_path())   # always overwrite last

            # ── Validation ────────────────────────────────────────────────
            if epoch % self.eval_every == 0 or epoch == 1:
                val_cer, word_acc, examples = evaluate(
                    self.model,
                    self.val_pairs,
                    self.src_tok,
                    self.tgt_tok,
                    self.device,
                    max_samples=self.max_eval_samples,
                )

                elapsed = time.time() - t0
                lr_now  = self.optimizer.param_groups[0]["lr"]
                improved = val_cer < self.best_cer - 1e-4
                tag = " ◀ BEST" if improved else ""

                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"loss {train_loss:.4f} | "
                    f"CER {val_cer:.4f} | "
                    f"WordAcc {word_acc*100:.1f}%{tag} | "
                    f"lr {lr_now:.2e} | "
                    f"step {self.global_step:,} | "
                    f"{elapsed:.1f}s"
                )

                if improved:
                    self.best_cer   = val_cer
                    self.no_improve = 0
                    self._save_best()
                else:
                    self.no_improve += 1

                # Print sample predictions every 25 epochs
                if epoch % 25 == 0 or epoch == 1:
                    print()
                    for roman, pred, target in examples:
                        ok = "✓" if pred == target else "✗"
                        print(f"    {ok} {roman:30s} → {pred:20s}  (target: {target})")
                    print()

                self.history.append({
                    "epoch":       epoch,
                    "global_step": self.global_step,
                    "train_loss":  round(train_loss, 5),
                    "val_cer":     round(val_cer, 5),
                    "word_acc":    round(word_acc, 5),
                    "lr":          lr_now,
                })

                # ── Early stopping ─────────────────────────────────────────
                if self.no_improve >= self.patience:
                    print(
                        f"\n  Early stopping at epoch {epoch} "
                        f"({self.no_improve} eval rounds without improvement)"
                    )
                    break

        # ── Final summary ─────────────────────────────────────────────────
        print(
            f"\n  Best Val CER  : {self.best_cer:.4f}  "
            f"(word accuracy ≈ {(1 - self.best_cer) * 100:.1f}%)\n"
            f"  Best model    : {self._best_path()}\n"
            f"  Global steps  : {self.global_step:,}"
        )

        # Save training log
        with open(os.path.join(self.model_dir, "training_log.json"), "w") as f:
            json.dump(self.history, f, indent=2)
"""
trainer.py — Training loop, evaluation, and checkpointing.

Exports
───────
  TrainConfig   — dataclass of all training hyperparameters
  select_device — auto-picks cuda / mps / cpu
  set_seed      — seeds python, numpy, torch
  Trainer       — full training lifecycle

API compatibility
─────────────────
Trainer.__init__ accepts BOTH call patterns:

  Pattern 1 — Kaggle notebook style (config object):
      Trainer(model, train_loader, val_loader=val_loader,
              val_pairs=val_pairs, src_tok=src_tok, tgt_tok=tgt_tok,
              device=device, config=cfg)
      trainer.train()

  Pattern 2 — train.py style (keyword args):
      Trainer(model, train_loader, val_pairs=val_pairs,
              src_tok=src_tok, tgt_tok=tgt_tok, device=device,
              model_dir=..., lr=..., warmup_steps=..., ...)
      trainer.train(epochs=N, start_epoch=1)
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, *a, **k):  # type: ignore
        return x

from dataset import CharTokenizer, EOW_IDX, PAD_IDX
from model import Seq2SeqTransformer, greedy_decode

logger = logging.getLogger("trainer")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ── Utilities ────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    """Seed python / numpy / torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(prefer: Optional[str] = None) -> torch.device:
    """Pick the best available device (cuda > mps > cpu)."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(prev[j]+1, curr[j-1]+1, prev[j-1]+cost)
        prev = curr
    return prev[n]


# ── TrainConfig ──────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """All training hyperparameters. Pass as `config=cfg` to Trainer."""
    epochs:           int   = 150
    lr:               float = 3e-4
    weight_decay:     float = 1e-4
    warmup_steps:     int   = 2000       # gradient-update steps, NOT epochs
    label_smoothing:  float = 0.1
    grad_clip:        float = 1.0
    betas:            tuple = (0.9, 0.98)
    eps:              float = 1e-9
    cosine_mult:      int   = 10
    lr_floor:         float = 0.10
    patience:         int   = 25         # validation rounds without improvement
    eval_every:       int   = 1          # epochs between evaluations
    save_every:       int   = 1          # epochs between periodic checkpoints
    max_eval_samples: int   = 0          # 0 = use ALL val pairs
    n_examples:       int   = 8
    use_amp:          bool  = True       # CUDA only
    seed:             int   = 42
    model_dir:        str   = "models"
    repetition_penalty: float = 1.3      # >1 stops last-char repetition loops
    device:           Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── Causal mask helper ───────────────────────────────────────────────────────
def _causal(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1
    )


# ── Single training epoch ────────────────────────────────────────────────────
def train_epoch(
    model:     Seq2SeqTransformer,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    criterion: nn.Module,
    device:    torch.device,
    scaler=None,
    grad_clip: float = 1.0,
    epoch:     int = 0,
) -> tuple[float, int]:
    """One pass over training data. Returns (avg_loss, n_steps)."""
    model.train()
    total_loss = total_tok = steps = 0

    for src, tgt, src_pad, tgt_pad in tqdm(loader, desc=f"train e{epoch}", leave=False):
        src     = src.to(device, non_blocking=True)
        tgt     = tgt.to(device, non_blocking=True)
        src_pad = src_pad.to(device, non_blocking=True)
        tgt_pad = tgt_pad.to(device, non_blocking=True)

        tgt_in     = tgt[:, :-1]
        tgt_target = tgt[:, 1:]
        tgt_in_pad = tgt_pad[:, :-1]
        causal     = _causal(tgt_in.size(1), device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(src, tgt_in, src_pad, tgt_in_pad, tgt_mask=causal)
                B, S, V = logits.shape
                loss = criterion(logits.reshape(B*S, V), tgt_target.reshape(B*S))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(src, tgt_in, src_pad, tgt_in_pad, tgt_mask=causal)
            B, S, V = logits.shape
            loss = criterion(logits.reshape(B*S, V), tgt_target.reshape(B*S))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()   # ← per gradient-update step, NOT per epoch
        steps += 1

        n_tok = tgt_target.ne(PAD_IDX).sum().item()
        total_loss += loss.item() * n_tok
        total_tok  += n_tok

    return total_loss / max(total_tok, 1), steps


# ── Teacher-forced validation (fast, uses train-time forward) ────────────────
@torch.no_grad()
def teacher_forced_eval(
    model:     Seq2SeqTransformer,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, float]:
    """Returns (avg_loss, char_accuracy, eow_accuracy)."""
    model.eval()
    total_loss = total_tok = correct = eow_correct = eow_total = 0.0

    for src, tgt, src_pad, tgt_pad in loader:
        src     = src.to(device); tgt     = tgt.to(device)
        src_pad = src_pad.to(device); tgt_pad = tgt_pad.to(device)

        tgt_in     = tgt[:, :-1]
        tgt_target = tgt[:, 1:]
        tgt_in_pad = tgt_pad[:, :-1]
        causal     = _causal(tgt_in.size(1), device)

        logits = model(src, tgt_in, src_pad, tgt_in_pad, tgt_mask=causal)
        B, S, V = logits.shape
        loss = criterion(logits.reshape(B*S, V), tgt_target.reshape(B*S))

        pred    = logits.argmax(-1)
        non_pad = tgt_target.ne(PAD_IDX)
        n_tok   = non_pad.sum().item()

        correct   += ((pred == tgt_target) & non_pad).sum().item()
        eow_mask   = tgt_target.eq(EOW_IDX)
        eow_total  += eow_mask.sum().item()
        eow_correct += ((pred == tgt_target) & eow_mask).sum().item()

        total_loss += loss.item() * n_tok
        total_tok  += n_tok

    char_acc = correct    / max(total_tok,   1)
    eow_acc  = eow_correct / max(eow_total,  1)
    return total_loss / max(total_tok, 1), char_acc, eow_acc


# ── Autoregressive validation (true metric) ──────────────────────────────────
@torch.no_grad()
def evaluate(
    model:       Seq2SeqTransformer,
    val_pairs:   list[tuple[str, str]],
    src_tok:     CharTokenizer,
    tgt_tok:     CharTokenizer,
    device:      torch.device,
    max_samples: int   = 0,
    n_examples:  int   = 8,
    repetition_penalty: float = 1.3,
) -> tuple[float, float, list[tuple[str, str, str]]]:
    """
    Greedy-decode validation pairs.
    Returns (corpus_CER, word_accuracy, examples).
    word_accuracy = exact-match fraction — the key transliteration metric.
    """
    model.eval()
    pool = val_pairs if max_samples <= 0 else val_pairs[:max_samples]

    sos = tgt_tok.ch2id["<SOS>"]
    eos = tgt_tok.ch2id["<EOS>"]

    total_edit = total_len = exact = 0
    examples: list[tuple[str, str, str]] = []

    for roman, target in tqdm(pool, desc="val-decode", leave=False):
        src_ids = src_tok.encode(roman, add_sos=False, add_eos=True)
        out_ids = greedy_decode(
            model, src_ids, sos, eos, device,
            repetition_penalty=repetition_penalty,
        )
        pred = tgt_tok.decode(out_ids)

        total_edit += levenshtein(pred, target)
        total_len  += max(len(target), 1)
        if pred == target:
            exact += 1
        if len(examples) < n_examples:
            examples.append((roman, pred, target))

    cer      = total_edit / max(total_len, 1)
    word_acc = exact / max(len(pool), 1)
    return cer, word_acc, examples


# ── Trainer ──────────────────────────────────────────────────────────────────
class Trainer:
    """
    Full training lifecycle: optimiser, LR schedule, AMP, checkpointing,
    early stopping.

    Accepts two call patterns — see module docstring.
    Best model selected by validation WORD ACCURACY (exact match).
    """

    def __init__(
        self,
        model:        Seq2SeqTransformer,
        train_loader: DataLoader,
        # ── val data ── (one of val_loader or val_pairs must be supplied)
        val_loader:   Optional[DataLoader]             = None,
        val_pairs:    Optional[list[tuple[str, str]]]  = None,
        src_tok:      Optional[CharTokenizer]          = None,
        tgt_tok:      Optional[CharTokenizer]          = None,
        device:       Optional[torch.device]           = None,
        # ── Pattern 1: pass a TrainConfig object ──
        config:       Optional[TrainConfig]            = None,
        # ── Pattern 2: pass individual kwargs (backward-compat) ──
        model_dir:    str   = "models",
        lr:           float = 3e-4,
        warmup_steps: int   = 2000,
        patience:     int   = 25,
        label_smoothing: float = 0.1,
        grad_clip:    float = 1.0,
        weight_decay: float = 1e-4,
        use_amp:      bool  = True,
        eval_every:   int   = 1,
        save_every:   int   = 1,
        max_eval_samples: int = 0,
    ) -> None:

        # ── Resolve config ────────────────────────────────────────────────
        if config is not None:
            cfg = config
        else:
            cfg = TrainConfig(
                lr=lr, weight_decay=weight_decay, warmup_steps=warmup_steps,
                label_smoothing=label_smoothing, grad_clip=grad_clip,
                patience=patience, use_amp=use_amp, eval_every=eval_every,
                save_every=save_every, max_eval_samples=max_eval_samples,
                model_dir=model_dir,
            )
        self.cfg = cfg

        if device is None:
            device = select_device(cfg.device)

        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.val_pairs    = val_pairs or []
        self.src_tok      = src_tok
        self.tgt_tok      = tgt_tok
        self.device       = device

        os.makedirs(cfg.model_dir, exist_ok=True)

        # Loss
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=PAD_IDX,
            label_smoothing=cfg.label_smoothing,
        )

        # Optimiser
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )

        # LR schedule — stepped per gradient update
        warmup     = max(1, cfg.warmup_steps)
        cosine_end = max(1, warmup * cfg.cosine_mult)
        floor      = cfg.lr_floor

        def _lr(step: int) -> float:
            step = max(step, 1)
            if step < warmup:
                return step / warmup
            t = (step - warmup) / cosine_end
            return max(floor, 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0))))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr)

        # AMP — CUDA only (MPS autocast is unstable)
        self._use_amp = cfg.use_amp and (device.type == "cuda")
        self.scaler   = torch.cuda.amp.GradScaler() if self._use_amp else None

        # State
        self.history:       list[dict] = []
        self.best_word_acc: float      = 0.0
        self.best_cer:      float      = float("inf")
        self.no_improve:    int        = 0
        self.global_step:   int        = 0

    # ── Path helpers ──────────────────────────────────────────────────────────
    def _best_path(self)          -> str: return os.path.join(self.cfg.model_dir, "best_model.pt")
    def _last_path(self)          -> str: return os.path.join(self.cfg.model_dir, "last_checkpoint.pt")
    def _periodic_path(self, ep)  -> str: return os.path.join(self.cfg.model_dir, f"checkpoint_epoch{ep:04d}.pt")

    # ── Checkpoint I/O ────────────────────────────────────────────────────────
    def _save_best(self) -> None:
        torch.save({
            "model_state":   self.model.state_dict(),
            "best_word_acc": self.best_word_acc,
            "best_cer":      self.best_cer,
            "global_step":   self.global_step,
            "config":        self.cfg.to_dict(),
        }, self._best_path())

    def _save_full(self, epoch: int, path: str) -> None:
        torch.save({
            "epoch":           epoch,
            "global_step":     self.global_step,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state":    self.scaler.state_dict() if self.scaler else None,
            "best_word_acc":   self.best_word_acc,
            "best_cer":        self.best_cer,
            "no_improve":      self.no_improve,
            "history":         self.history,
            "config":          self.cfg.to_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """Load full checkpoint; returns next epoch to run."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if self.scaler and ckpt.get("scaler_state"):
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.best_word_acc = ckpt.get("best_word_acc", 0.0)
        self.best_cer      = ckpt.get("best_cer", float("inf"))
        self.no_improve    = ckpt.get("no_improve", 0)
        self.global_step   = ckpt.get("global_step", 0)
        self.history       = ckpt.get("history", [])
        next_ep = ckpt["epoch"] + 1
        logger.info("Resumed from epoch %d (best WordAcc=%.1f%%, step=%d)",
                    ckpt["epoch"], self.best_word_acc * 100, self.global_step)
        return next_ep

    # ── Training loop ─────────────────────────────────────────────────────────
    def train(self, epochs: int = 0, start_epoch: int = 1) -> None:
        """
        Run the training loop.

        epochs      : overrides cfg.epochs when > 0
        start_epoch : resume from this epoch (1 = fresh start)
        """
        cfg      = self.cfg
        n_epochs = epochs if epochs > 0 else cfg.epochs
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info("═" * 66)
        logger.info("TRAINING   device=%s   AMP=%s   params=%s",
                    self.device, self._use_amp, f"{n_params:,}")
        logger.info("epochs=%d  warmup_steps=%d  patience=%d  eval_every=%d",
                    n_epochs, cfg.warmup_steps, cfg.patience, cfg.eval_every)
        logger.info("repetition_penalty=%.2f  label_smoothing=%.2f",
                    cfg.repetition_penalty, cfg.label_smoothing)
        logger.info("═" * 66)

        needs_autoregressive = bool(self.val_pairs and self.src_tok and self.tgt_tok)
        if not needs_autoregressive:
            logger.warning(
                "val_pairs / src_tok / tgt_tok not supplied — "
                "word accuracy and CER will NOT be computed."
            )

        try:
            for epoch in range(start_epoch, n_epochs + 1):
                t0 = time.time()

                train_loss, batch_steps = train_epoch(
                    self.model, self.train_loader,
                    self.optimizer, self.scheduler,
                    self.criterion, self.device,
                    scaler=self.scaler,
                    grad_clip=cfg.grad_clip,
                    epoch=epoch,
                )
                self.global_step += batch_steps

                # Always save last checkpoint (enables resume after every epoch)
                self._save_full(epoch, self._last_path())
                if epoch % cfg.save_every == 0:
                    self._save_full(epoch, self._periodic_path(epoch))

                # ── Validation ─────────────────────────────────────────────
                if epoch % cfg.eval_every == 0 or epoch == 1:
                    elapsed = time.time() - t0
                    lr_now  = self.optimizer.param_groups[0]["lr"]
                    row: dict = {
                        "epoch": epoch, "global_step": self.global_step,
                        "train_loss": round(train_loss, 5), "lr": lr_now,
                    }

                    # Fast teacher-forced metrics
                    if self.val_loader is not None:
                        val_loss, char_acc, eow_acc = teacher_forced_eval(
                            self.model, self.val_loader, self.criterion, self.device
                        )
                        row.update(val_loss=round(val_loss, 5),
                                   char_acc=round(char_acc, 5),
                                   eow_acc=round(eow_acc, 5))

                    # Autoregressive word accuracy + CER
                    word_acc = cer = 0.0
                    examples: list = []
                    if needs_autoregressive:
                        cer, word_acc, examples = evaluate(
                            self.model, self.val_pairs,
                            self.src_tok, self.tgt_tok,   # type: ignore
                            self.device,
                            max_samples=cfg.max_eval_samples,
                            n_examples=cfg.n_examples,
                            repetition_penalty=cfg.repetition_penalty,
                        )
                        row.update(val_cer=round(cer, 5),
                                   word_acc=round(word_acc, 5))

                    improved = word_acc > self.best_word_acc + 1e-4 if needs_autoregressive else train_loss < self.best_cer
                    tag = " ◀ BEST" if improved else ""

                    # ── Log line ──────────────────────────────────────────
                    parts = [f"Epoch {epoch:4d}/{n_epochs}"]
                    parts.append(f"loss {train_loss:.4f}")
                    if "val_loss" in row:
                        parts.append(f"vloss {row['val_loss']:.4f}")
                        parts.append(f"charAcc {row['char_acc']*100:.1f}%")
                        parts.append(f"EOW {row['eow_acc']*100:.1f}%")
                    if needs_autoregressive:
                        parts.append(f"CER {cer:.4f}")
                        parts.append(f"WordAcc {word_acc*100:.1f}%{tag}")
                    parts.append(f"lr {lr_now:.2e}")
                    parts.append(f"step {self.global_step:,}")
                    parts.append(f"{elapsed:.1f}s")
                    logger.info(" | ".join(parts))

                    if improved:
                        self.best_word_acc = word_acc
                        self.best_cer      = min(self.best_cer, cer)
                        self.no_improve    = 0
                        self._save_best()
                    else:
                        self.no_improve += 1

                    # Print predictions every 10 epochs (or epoch 1)
                    if (epoch % 10 == 0 or epoch == 1) and examples:
                        logger.info("  Sample predictions:")
                        for roman, pred, target in examples:
                            ok = "✓" if pred == target else "✗"
                            logger.info("    %s %-28s → %-20s  (target: %s)",
                                        ok, roman, pred, target)

                    self.history.append(row)

                    if self.no_improve >= cfg.patience:
                        logger.info(
                            "Early stop at epoch %d (%d rounds without improvement)",
                            epoch, self.no_improve,
                        )
                        break

        except KeyboardInterrupt:
            logger.warning("Interrupted — last checkpoint saved.")
            raise
        finally:
            with open(os.path.join(cfg.model_dir, "training_log.json"), "w") as f:
                json.dump(self.history, f, indent=2)

        logger.info(
            "Best WordAcc %.1f%%  CER %.4f  → %s",
            self.best_word_acc * 100, self.best_cer, self._best_path()
        )
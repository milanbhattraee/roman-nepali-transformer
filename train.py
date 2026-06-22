#!/usr/bin/env python3
"""
train.py — Entry point: Roman Nepali → Devanagari Transformer

Usage
─────
  python train.py
  python train.py --csv data/roman_nepali_clean.csv
  python train.py --epochs 150 --lr 3e-4
  python train.py --resume          # continue from last_checkpoint.pt
  python train.py --no-augment
  python train.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import CharTokenizer, build_dataloaders, load_word_pairs
from model import Seq2SeqTransformer, count_parameters
from trainer import Trainer, TrainConfig, set_seed, select_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Roman Nepali → Devanagari Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--csv",       default=Config.DATA_PATH)
    p.add_argument("--model-dir", default=Config.MODEL_DIR)
    # Architecture
    p.add_argument("--d-model",    type=int,   default=Config.D_MODEL)
    p.add_argument("--n-heads",    type=int,   default=Config.N_HEADS)
    p.add_argument("--enc-layers", type=int,   default=Config.NUM_ENCODER_LAYERS)
    p.add_argument("--dec-layers", type=int,   default=Config.NUM_DECODER_LAYERS)
    p.add_argument("--ff-dim",     type=int,   default=Config.FF_DIM)
    p.add_argument("--dropout",    type=float, default=Config.DROPOUT)
    # Training
    p.add_argument("--epochs",      type=int,   default=Config.EPOCHS)
    p.add_argument("--batch-size",  type=int,   default=Config.BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=Config.LEARNING_RATE)
    p.add_argument("--warmup",      type=int,   default=Config.WARMUP_STEPS)
    p.add_argument("--patience",    type=int,   default=Config.PATIENCE)
    p.add_argument("--grad-clip",   type=float, default=Config.GRAD_CLIP)
    p.add_argument("--weight-decay",type=float, default=Config.WEIGHT_DECAY)
    p.add_argument("--label-smooth",type=float, default=Config.LABEL_SMOOTHING)
    p.add_argument("--rep-penalty", type=float, default=Config.REPETITION_PENALTY)
    # Augmentation
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--noise-prob", type=float, default=Config.NOISE_PROB)
    # Misc
    p.add_argument("--no-amp",     action="store_true")
    p.add_argument("--workers",    type=int, default=Config.NUM_WORKERS)
    p.add_argument("--eval-every", type=int, default=Config.EVAL_EVERY)
    p.add_argument("--save-every", type=int, default=Config.SAVE_EVERY)
    p.add_argument("--seed",       type=int, default=Config.RANDOM_SEED)
    p.add_argument("--resume",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Apply overrides ───────────────────────────────────────────────────────
    Config.DATA_PATH          = args.csv
    Config.MODEL_DIR          = args.model_dir
    Config.D_MODEL            = args.d_model
    Config.N_HEADS            = args.n_heads
    Config.NUM_ENCODER_LAYERS = args.enc_layers
    Config.NUM_DECODER_LAYERS = args.dec_layers
    Config.FF_DIM             = args.ff_dim
    Config.DROPOUT            = args.dropout
    Config.EPOCHS             = args.epochs
    Config.BATCH_SIZE         = args.batch_size
    Config.LEARNING_RATE      = args.lr
    Config.WARMUP_STEPS       = args.warmup
    Config.PATIENCE           = args.patience
    Config.GRAD_CLIP          = args.grad_clip
    Config.WEIGHT_DECAY       = args.weight_decay
    Config.LABEL_SMOOTHING    = args.label_smooth
    Config.REPETITION_PENALTY = args.rep_penalty
    Config.AUGMENT            = not args.no_augment
    Config.NOISE_PROB         = args.noise_prob
    Config.USE_AMP            = not args.no_amp
    Config.NUM_WORKERS        = args.workers
    Config.EVAL_EVERY         = args.eval_every
    Config.SAVE_EVERY         = args.save_every
    Config.RANDOM_SEED        = args.seed
    Config.RESUME             = args.resume

    try:
        Config.validate()
    except AssertionError as e:
        print(f"\nConfig error: {e}\n")
        sys.exit(1)

    print("=" * 70)
    print("  ROMAN NEPALI → DEVANAGARI  |  Transformer Trainer")
    print("=" * 70)
    for k, v in [
        ("CSV",         Config.DATA_PATH),
        ("Model dir",   Config.MODEL_DIR),
        ("Epochs",      Config.EPOCHS),
        ("Batch size",  Config.BATCH_SIZE),
        ("LR",          Config.LEARNING_RATE),
        ("Warmup steps",Config.WARMUP_STEPS),
        ("Augment",     f"{Config.AUGMENT}  noise_prob={Config.NOISE_PROB}"),
        ("Rep penalty", Config.REPETITION_PENALTY),
        ("AMP",         Config.USE_AMP),
        ("Resume",      Config.RESUME),
        ("Seed",        Config.RANDOM_SEED),
    ]:
        print(f"  {k:<14}: {v}")
    print()

    if not os.path.isfile(Config.DATA_PATH):
        print(f"ERROR: CSV not found: {Config.DATA_PATH}")
        sys.exit(1)

    set_seed(Config.RANDOM_SEED)
    device = select_device()
    print(f"  Device: {device}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading dataset...")
    pairs, word_dict = load_word_pairs(Config.DATA_PATH)
    print(f"  Unique word pairs: {len(pairs):,}\n")

    # ── Tokenizers ────────────────────────────────────────────────────────────
    src_tok = CharTokenizer().build([r for r, _ in pairs])
    tgt_tok = CharTokenizer().build([d for _, d in pairs])
    print(f"  Source vocab : {src_tok.vocab_size}")
    print(f"  Target vocab : {tgt_tok.vocab_size}\n")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print("Building data loaders...")
    train_loader, val_loader, train_pairs, val_pairs = build_dataloaders(
        pairs=pairs, src_tok=src_tok, tgt_tok=tgt_tok,
        val_split=Config.VAL_SPLIT, batch_size=Config.BATCH_SIZE,
        augment=Config.AUGMENT, noise_prob=Config.NOISE_PROB,
        seed=Config.RANDOM_SEED, num_workers=Config.NUM_WORKERS,
    )
    print(f"  Train: {len(train_pairs):,}  |  Val: {len(val_pairs):,}\n")

    # ── Save tokenizers + word dict ───────────────────────────────────────────
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    with open(Config.tokenizer_path(), "w", encoding="utf-8") as f:
        json.dump({
            "src": {"ch2id": src_tok.ch2id, "id2ch": {str(k): v for k, v in src_tok.id2ch.items()}},
            "tgt": {"ch2id": tgt_tok.ch2id, "id2ch": {str(k): v for k, v in tgt_tok.id2ch.items()}},
        }, f, ensure_ascii=False, indent=2)

    with open(Config.word_dict_path(), "w", encoding="utf-8") as f:
        json.dump(word_dict, f, ensure_ascii=False, indent=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Seq2SeqTransformer(
        src_vocab_size=src_tok.vocab_size, tgt_vocab_size=tgt_tok.vocab_size,
        d_model=Config.D_MODEL, nhead=Config.N_HEADS,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS,
        dim_feedforward=Config.FF_DIM, dropout=Config.DROPOUT,
        max_seq_len=Config.MAX_SEQ_LEN,
    ).to(device)

    # Save model architecture config (needed by predict.py)
    with open(Config.model_config_path(), "w") as f:
        json.dump({
            "src_vocab_size": src_tok.vocab_size,
            "tgt_vocab_size": tgt_tok.vocab_size,
            "d_model":            Config.D_MODEL,
            "nhead":              Config.N_HEADS,
            "num_encoder_layers": Config.NUM_ENCODER_LAYERS,
            "num_decoder_layers": Config.NUM_DECODER_LAYERS,
            "dim_feedforward":    Config.FF_DIM,
            "dropout":            Config.DROPOUT,
            "max_seq_len":        Config.MAX_SEQ_LEN,
        }, f, indent=2)

    print(f"  Model parameters: {count_parameters(model):,}\n")

    # ── Trainer ───────────────────────────────────────────────────────────────
    cfg = TrainConfig(
        epochs=Config.EPOCHS, lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY, warmup_steps=Config.WARMUP_STEPS,
        label_smoothing=Config.LABEL_SMOOTHING, grad_clip=Config.GRAD_CLIP,
        patience=Config.PATIENCE, use_amp=Config.USE_AMP,
        eval_every=Config.EVAL_EVERY, save_every=Config.SAVE_EVERY,
        max_eval_samples=Config.MAX_EVAL_SAMPLES,
        repetition_penalty=Config.REPETITION_PENALTY,
        model_dir=Config.MODEL_DIR, seed=Config.RANDOM_SEED,
    )

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        val_pairs=val_pairs, src_tok=src_tok, tgt_tok=tgt_tok,
        device=device, config=cfg,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    last_ckpt   = Config.last_checkpoint_path()
    if Config.RESUME and os.path.isfile(last_ckpt):
        start_epoch = trainer.load_checkpoint(last_ckpt)
        print(f"  Resumed from epoch {start_epoch - 1}\n")
    elif Config.RESUME:
        print(f"  --resume set but no checkpoint at {last_ckpt}; starting fresh.\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.train(epochs=Config.EPOCHS, start_epoch=start_epoch)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    for rel in ("best_model.pt", "last_checkpoint.pt", "tokenizers.json",
                "word_dictionary.json", "model_config.json", "training_log.json"):
        path = os.path.join(Config.MODEL_DIR, rel)
        size = f"{os.path.getsize(path)/1024:.0f} KB" if os.path.isfile(path) else "—"
        print(f"  {path:<50}  {size}")
    print()
    print("  Next: python predict.py \"k xa khabar tapai\"")
    print()


if __name__ == "__main__":
    main()
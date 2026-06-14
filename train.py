#!/usr/bin/env python3
"""
train.py  ─  Entry point: Roman Nepali → Devanagari Transformer
═══════════════════════════════════════════════════════════════════════════════
Usage
─────
  python train.py                          # use all Config defaults
  python train.py --csv data/my.csv        # custom dataset
  python train.py --epochs 200 --lr 1e-4  # override hyper-parameters
  python train.py --resume                 # continue from last_checkpoint.pt
  python train.py --no-augment             # disable data augmentation

Full list: python train.py --help
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

# ── Allow running from the project root without installing the package ────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import CharTokenizer, build_dataloaders, load_word_pairs
from model import Seq2SeqTransformer, count_parameters
from trainer import Trainer


# ─── Reproducibility ──────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    """Fix all relevant RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (may slow things down slightly on CUDA)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Roman Nepali → Devanagari Transformer Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--csv",        type=str,   default=Config.DATA_PATH,     help="Dataset CSV file")
    p.add_argument("--model-dir",  type=str,   default=Config.MODEL_DIR,     help="Output directory for checkpoints")

    # Architecture
    p.add_argument("--d-model",    type=int,   default=Config.D_MODEL,       help="Embedding / hidden dimension")
    p.add_argument("--n-heads",    type=int,   default=Config.N_HEADS,       help="Attention heads")
    p.add_argument("--enc-layers", type=int,   default=Config.NUM_ENCODER_LAYERS, help="Encoder depth")
    p.add_argument("--dec-layers", type=int,   default=Config.NUM_DECODER_LAYERS, help="Decoder depth")
    p.add_argument("--ff-dim",     type=int,   default=Config.FF_DIM,        help="Feed-forward inner dimension")
    p.add_argument("--dropout",    type=float, default=Config.DROPOUT,       help="Dropout rate")

    # Training
    p.add_argument("--epochs",     type=int,   default=Config.EPOCHS,        help="Max training epochs")
    p.add_argument("--batch-size", type=int,   default=Config.BATCH_SIZE,    help="Batch size")
    p.add_argument("--lr",         type=float, default=Config.LEARNING_RATE, help="Peak learning rate")
    p.add_argument("--warmup",     type=int,   default=Config.WARMUP_STEPS,  help="LR warmup steps (batch updates, NOT epochs)")
    p.add_argument("--patience",   type=int,   default=Config.PATIENCE,      help="Early-stopping patience (eval rounds)")
    p.add_argument("--grad-clip",  type=float, default=Config.GRAD_CLIP,     help="Gradient clipping norm")
    p.add_argument("--weight-decay", type=float, default=Config.WEIGHT_DECAY, help="AdamW weight decay")
    p.add_argument("--label-smooth", type=float, default=Config.LABEL_SMOOTHING, help="Label smoothing ε")

    # Augmentation
    p.add_argument("--no-augment",  action="store_true", help="Disable data augmentation")
    p.add_argument("--noise-prob",  type=float, default=Config.NOISE_PROB,   help="Fraction of pairs to augment [0,1]")

    # AMP / hardware
    p.add_argument("--no-amp",     action="store_true",  help="Disable Automatic Mixed Precision (CUDA only)")
    p.add_argument("--workers",    type=int,   default=Config.NUM_WORKERS,   help="DataLoader worker processes")

    # Checkpointing / eval
    p.add_argument("--resume",     action="store_true",  help="Resume from last_checkpoint.pt if it exists")
    p.add_argument("--eval-every", type=int,   default=Config.EVAL_EVERY,    help="Validate every N epochs")
    p.add_argument("--save-every", type=int,   default=Config.SAVE_EVERY,    help="Save periodic checkpoint every N epochs")

    # Misc
    p.add_argument("--seed",       type=int,   default=Config.RANDOM_SEED,   help="Global random seed")

    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Apply CLI overrides to Config ─────────────────────────────────────
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
    Config.AUGMENT            = not args.no_augment
    Config.NOISE_PROB         = args.noise_prob
    Config.USE_AMP            = not args.no_amp
    Config.NUM_WORKERS        = args.workers
    Config.RESUME             = args.resume
    Config.EVAL_EVERY         = args.eval_every
    Config.SAVE_EVERY         = args.save_every
    Config.RANDOM_SEED        = args.seed

    # ── Validate config before doing any work ─────────────────────────────
    try:
        Config.validate()
    except AssertionError as exc:
        print(f"\nConfig error: {exc}")
        sys.exit(1)

    # ── Header ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  ROMAN NEPALI → DEVANAGARI TRANSFORMER")
    print("=" * 70)
    print(f"  Dataset       : {Config.DATA_PATH}")
    print(f"  Model dir     : {Config.MODEL_DIR}")
    print(f"  Epochs        : {Config.EPOCHS}")
    print(f"  Batch size    : {Config.BATCH_SIZE}")
    print(f"  Learning rate : {Config.LEARNING_RATE}")
    print(f"  Warmup steps  : {Config.WARMUP_STEPS}  (batch updates)")
    print(f"  Augment       : {Config.AUGMENT}  (noise_prob={Config.NOISE_PROB})")
    print(f"  AMP           : {Config.USE_AMP}")
    print(f"  Resume        : {Config.RESUME}")
    print(f"  Seed          : {Config.RANDOM_SEED}")
    print()

    # ── Validate file exists ───────────────────────────────────────────────
    if not os.path.isfile(Config.DATA_PATH):
        print(f"ERROR: Dataset not found: {Config.DATA_PATH}")
        sys.exit(1)

    # ── Seed everything ───────────────────────────────────────────────────
    seed_everything(Config.RANDOM_SEED)

    device = get_device()
    print(f"  Device: {device}\n")

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading dataset...")
    pairs, word_dict = load_word_pairs(Config.DATA_PATH)

    if not pairs:
        print("ERROR: No training pairs extracted. Check CSV columns (Romanized, Devanagari).")
        sys.exit(1)

    print(f"\n  Total unique word pairs: {len(pairs):,}\n")

    # ── Build tokenisers ──────────────────────────────────────────────────
    src_tok = CharTokenizer().build([roman for roman, _ in pairs])
    tgt_tok = CharTokenizer().build([devan  for _, devan  in pairs])

    print(f"  Source vocab size : {src_tok.vocab_size}")
    print(f"  Target vocab size : {tgt_tok.vocab_size}\n")

    # ── DataLoaders ───────────────────────────────────────────────────────
    print("Building data loaders...")
    train_loader, val_loader, train_pairs, val_pairs = build_dataloaders(
        pairs=pairs,
        src_tok=src_tok,
        tgt_tok=tgt_tok,
        val_split=Config.VAL_SPLIT,
        batch_size=Config.BATCH_SIZE,
        augment=Config.AUGMENT,
        noise_prob=Config.NOISE_PROB,
        seed=Config.RANDOM_SEED,
        num_workers=Config.NUM_WORKERS,
    )
    print()

    # ── Persist tokenisers + word dict ────────────────────────────────────
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    tok_path = Config.tokenizer_path()
    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "src": {"ch2id": src_tok.ch2id,
                        "id2ch": {str(k): v for k, v in src_tok.id2ch.items()}},
                "tgt": {"ch2id": tgt_tok.ch2id,
                        "id2ch": {str(k): v for k, v in tgt_tok.id2ch.items()}},
            },
            f, ensure_ascii=False, indent=2,
        )

    wd_path = Config.word_dict_path()
    with open(wd_path, "w", encoding="utf-8") as f:
        json.dump(word_dict, f, ensure_ascii=False, indent=2)

    # ── Model ─────────────────────────────────────────────────────────────
    model = Seq2SeqTransformer(
        src_vocab_size     = src_tok.vocab_size,
        tgt_vocab_size     = tgt_tok.vocab_size,
        d_model            = Config.D_MODEL,
        nhead              = Config.N_HEADS,
        num_encoder_layers = Config.NUM_ENCODER_LAYERS,
        num_decoder_layers = Config.NUM_DECODER_LAYERS,
        dim_feedforward    = Config.FF_DIM,
        dropout            = Config.DROPOUT,
        max_seq_len        = Config.MAX_SEQ_LEN,
    ).to(device)

    print(f"  Model parameters : {count_parameters(model):,}\n")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model            = model,
        train_loader     = train_loader,
        val_pairs        = val_pairs,
        src_tok          = src_tok,
        tgt_tok          = tgt_tok,
        device           = device,
        model_dir        = Config.MODEL_DIR,
        lr               = Config.LEARNING_RATE,
        warmup_steps     = Config.WARMUP_STEPS,
        patience         = Config.PATIENCE,
        label_smoothing  = Config.LABEL_SMOOTHING,
        grad_clip        = Config.GRAD_CLIP,
        weight_decay     = Config.WEIGHT_DECAY,
        use_amp          = Config.USE_AMP,
        eval_every       = Config.EVAL_EVERY,
        save_every       = Config.SAVE_EVERY,
        max_eval_samples = Config.MAX_EVAL_SAMPLES,
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    last_ckpt   = Config.last_checkpoint_path()
    if Config.RESUME:
        if os.path.isfile(last_ckpt):
            start_epoch = trainer.load_checkpoint(last_ckpt)
        else:
            print(f"  WARNING: --resume set but no checkpoint found at {last_ckpt}; "
                  "starting from scratch.\n")

    # ── Train ─────────────────────────────────────────────────────────────
    trainer.train(epochs=Config.EPOCHS, start_epoch=start_epoch)

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print()
    print("  Files written:")
    for rel in (
        "best_model.pt",
        "last_checkpoint.pt",
        "tokenizers.json",
        "word_dictionary.json",
        "training_log.json",
    ):
        path = os.path.join(Config.MODEL_DIR, rel)
        size = f"{os.path.getsize(path) / 1024:.0f} KB" if os.path.isfile(path) else "—"
        print(f"    {path:<50s}  {size}")
    print()


if __name__ == "__main__":
    main()
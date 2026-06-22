"""
config.py — Centralized, validated training configuration.
All hyper-parameters in one place. Import and modify directly or via CLI.
"""
from __future__ import annotations
import os


class Config:
    # ── Data ─────────────────────────────────────────────────────────────────
    DATA_PATH   : str  = "/kaggle/input/roman-nepali-transliteration-data/roman_nepali_clean.csv"
    MODEL_DIR   : str  = "models"
    NUM_WORKERS : int  = 0

    # ── Architecture ─────────────────────────────────────────────────────────
    D_MODEL            : int   = 256
    N_HEADS            : int   = 4      # D_MODEL must be divisible by N_HEADS
    NUM_ENCODER_LAYERS : int   = 4
    NUM_DECODER_LAYERS : int   = 4
    FF_DIM             : int   = 512
    DROPOUT            : float = 0.1
    MAX_SEQ_LEN        : int   = 60

    # ── Training ─────────────────────────────────────────────────────────────
    EPOCHS          : int   = 150
    BATCH_SIZE      : int   = 64
    LEARNING_RATE   : float = 3e-4
    # WARMUP_STEPS = gradient-update steps, NOT epochs.
    # ~80-280 batches/epoch × 2000 ≈ 7-25 epochs of warmup.
    WARMUP_STEPS    : int   = 2000
    PATIENCE        : int   = 25       # validation rounds without improvement
    GRAD_CLIP       : float = 1.0
    LABEL_SMOOTHING : float = 0.1
    WEIGHT_DECAY    : float = 1e-4
    USE_AMP         : bool  = True     # CUDA only; ignored on MPS/CPU

    # ── Data / Augmentation ──────────────────────────────────────────────────
    VAL_SPLIT   : float = 0.10
    RANDOM_SEED : int   = 42
    AUGMENT     : bool  = True
    NOISE_PROB  : float = 1.0

    # ── Evaluation ───────────────────────────────────────────────────────────
    EVAL_EVERY       : int = 1     # run validation every N epochs
    MAX_EVAL_SAMPLES : int = 0     # 0 = use ALL validation pairs

    # ── Checkpointing ────────────────────────────────────────────────────────
    RESUME     : bool = True
    SAVE_EVERY : int  = 1

    # ── Inference ────────────────────────────────────────────────────────────
    BEAM_WIDTH         : int   = 5
    MAX_OUT_LEN        : int   = 60
    REPETITION_PENALTY : float = 1.3   # >1 penalises repeated tokens

    # ── Derived path helpers ──────────────────────────────────────────────────
    @classmethod
    def best_model_path(cls) -> str:
        return os.path.join(cls.MODEL_DIR, "best_model.pt")

    @classmethod
    def last_checkpoint_path(cls) -> str:
        return os.path.join(cls.MODEL_DIR, "last_checkpoint.pt")

    @classmethod
    def tokenizer_path(cls) -> str:
        return os.path.join(cls.MODEL_DIR, "tokenizers.json")

    @classmethod
    def word_dict_path(cls) -> str:
        return os.path.join(cls.MODEL_DIR, "word_dictionary.json")

    @classmethod
    def model_config_path(cls) -> str:
        return os.path.join(cls.MODEL_DIR, "model_config.json")

    @classmethod
    def training_log_path(cls) -> str:
        return os.path.join(cls.MODEL_DIR, "training_log.json")

    @classmethod
    def validate(cls) -> None:
        assert cls.D_MODEL % cls.N_HEADS == 0, \
            f"D_MODEL ({cls.D_MODEL}) must be divisible by N_HEADS ({cls.N_HEADS})"
        assert 0.0 < cls.VAL_SPLIT < 1.0,       "VAL_SPLIT must be in (0, 1)"
        assert cls.BATCH_SIZE > 0,               "BATCH_SIZE must be > 0"
        assert cls.LEARNING_RATE > 0,            "LEARNING_RATE must be > 0"
        assert cls.GRAD_CLIP > 0,                "GRAD_CLIP must be > 0"
        assert 0.0 <= cls.LABEL_SMOOTHING < 1.0, "LABEL_SMOOTHING in [0, 1)"
        assert cls.PATIENCE > 0,                 "PATIENCE must be > 0"
        assert cls.WARMUP_STEPS >= 0,            "WARMUP_STEPS must be >= 0"
        assert cls.MAX_SEQ_LEN > 0,              "MAX_SEQ_LEN must be > 0"
        assert 0.0 <= cls.NOISE_PROB <= 1.0,     "NOISE_PROB in [0, 1]"
        assert cls.REPETITION_PENALTY >= 1.0,    "REPETITION_PENALTY must be >= 1.0"
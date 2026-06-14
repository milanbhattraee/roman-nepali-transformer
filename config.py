"""
config.py ── Centralized, validated training configuration.

All hyper-parameters live here.  Import and modify Config attributes directly
or override them via CLI arguments in train.py.
"""
from __future__ import annotations
import os


class Config:
    # ── Data ─────────────────────────────────────────────────────────────────
    DATA_PATH   : str  = "data/roman_nepali_clean.csv"
    MODEL_DIR   : str  = "models"
    NUM_WORKERS : int  = 0      # DataLoader workers (0 = main process, safest on all OS)

    # ── Architecture ─────────────────────────────────────────────────────────
    D_MODEL            : int   = 256   # embedding & hidden dimension
    N_HEADS            : int   = 4     # attention heads  (D_MODEL % N_HEADS == 0)
    NUM_ENCODER_LAYERS : int   = 4     # encoder depth
    NUM_DECODER_LAYERS : int   = 4     # decoder depth
    FF_DIM             : int   = 512   # feed-forward inner dimension
    DROPOUT            : float = 0.1
    MAX_SEQ_LEN        : int   = 60    # max character sequence length

    # ── Training ─────────────────────────────────────────────────────────────
    EPOCHS          : int   = 100
    BATCH_SIZE      : int   = 64
    LEARNING_RATE   : float = 3e-4
    # ⚠ WARMUP_STEPS is counted in gradient-update steps (batches), NOT epochs.
    # With ~80–280 batches/epoch, 2000 steps ≈ 7–25 epochs of warmup — correct.
    WARMUP_STEPS    : int   = 2000
    PATIENCE        : int   = 20        # validation rounds without improvement
    GRAD_CLIP       : float = 1.0       # max gradient L2 norm
    LABEL_SMOOTHING : float = 0.1       # cross-entropy ε
    WEIGHT_DECAY    : float = 1e-4      # AdamW L2 penalty
    USE_AMP         : bool  = True      # Automatic Mixed Precision (CUDA only)

    # ── Data / Augmentation ──────────────────────────────────────────────────
    VAL_SPLIT  : float = 0.10
    RANDOM_SEED: int   = 42
    AUGMENT    : bool  = True
    NOISE_PROB : float = 1.0   # fraction of eligible train pairs to augment
                                # 1.0 ≈ doubles train set (same as original behaviour)

    # ── Evaluation ───────────────────────────────────────────────────────────
    EVAL_EVERY       : int = 5    # run validation every N epochs
    MAX_EVAL_SAMPLES : int = 500  # max val pairs per eval run (0 = all)

    # ── Checkpointing ────────────────────────────────────────────────────────
    RESUME     : bool = True   # auto-resume from last_checkpoint.pt if found
    SAVE_EVERY : int  = 10      # save periodic checkpoint every N epochs

    # ── Inference ────────────────────────────────────────────────────────────
    BEAM_WIDTH  : int = 5
    MAX_OUT_LEN : int = 60

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
    def training_log_path(cls) -> str:
        return os.path.join(cls.MODEL_DIR, "training_log.json")

    @classmethod
    def validate(cls) -> None:
        """Raise AssertionError early if any setting is invalid."""
        assert cls.D_MODEL % cls.N_HEADS == 0, (
            f"D_MODEL ({cls.D_MODEL}) must be divisible by N_HEADS ({cls.N_HEADS})"
        )
        assert 0.0 < cls.VAL_SPLIT < 1.0,      "VAL_SPLIT must be in (0, 1)"
        assert cls.BATCH_SIZE      > 0,          "BATCH_SIZE must be > 0"
        assert cls.LEARNING_RATE   > 0,          "LEARNING_RATE must be > 0"
        assert cls.GRAD_CLIP       > 0,          "GRAD_CLIP must be > 0"
        assert 0.0 <= cls.LABEL_SMOOTHING < 1.0, "LABEL_SMOOTHING in [0, 1)"
        assert cls.PATIENCE        > 0,          "PATIENCE must be > 0"
        assert cls.WARMUP_STEPS    >= 0,         "WARMUP_STEPS must be >= 0"
        assert cls.MAX_SEQ_LEN     > 0,          "MAX_SEQ_LEN must be > 0"
        assert 0.0 <= cls.NOISE_PROB <= 1.0,     "NOISE_PROB in [0, 1]"
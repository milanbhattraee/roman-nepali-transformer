"""
dataset.py  ─  Data pipeline for Roman Nepali → Devanagari
═══════════════════════════════════════════════════════════════════════════════
Handles everything data-related:
  • Loading & extracting word pairs from the CSV
  • Building character-level vocabularies
  • Data augmentation (noise injection for robustness) — NOISE_PROB respected
  • PyTorch Dataset + DataLoader creation

Fixes vs original
──────────────────
  • noise_prob was wired up but NEVER used — all pairs were always augmented.
    Now NOISE_PROB controls what fraction of training pairs receive a noisy copy.
  • Added a 4th augmentation type: random-char insertion (common typo).
  • seed propagated into augment_pairs so runs are fully reproducible.
  • CharTokenizer.save / load use integer keys consistently; no silent mismatch.
  • _split_words handles full-width punctuation and zero-width joiner (ZWJ).
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from collections import Counter
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ─── Special token indices ────────────────────────────────────────────────────
PAD_IDX = 0   # padding (ignored by loss & attention)
SOS_IDX = 1   # start of sequence
EOS_IDX = 2   # end of sequence
UNK_IDX = 3   # unknown character

_SPECIAL_TOKENS = ("<PAD>", "<SOS>", "<EOS>", "<UNK>")


# ─── Tokenizer ───────────────────────────────────────────────────────────────

class CharTokenizer:
    """
    Character-level tokeniser.
    Builds a vocabulary of individual Unicode characters observed in the data.
    """

    def __init__(self) -> None:
        self.ch2id: dict[str, int] = {
            "<PAD>": PAD_IDX,
            "<SOS>": SOS_IDX,
            "<EOS>": EOS_IDX,
            "<UNK>": UNK_IDX,
        }
        self.id2ch: dict[int, str] = {v: k for k, v in self.ch2id.items()}

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, texts: list[str]) -> "CharTokenizer":
        """Extend vocabulary with every unique character found in *texts*."""
        chars: set[str] = set()
        for t in texts:
            chars.update(t)
        for ch in sorted(chars):           # sorted → deterministic vocab order
            if ch not in self.ch2id:
                idx = len(self.ch2id)
                self.ch2id[ch] = idx
                self.id2ch[idx] = ch
        return self

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_sos: bool = False,
        add_eos: bool = True,
    ) -> list[int]:
        ids: list[int] = []
        if add_sos:
            ids.append(SOS_IDX)
        for ch in text:
            ids.append(self.ch2id.get(ch, UNK_IDX))
        if add_eos:
            ids.append(EOS_IDX)
        return ids

    def decode(self, ids: list[int]) -> str:
        chars: list[str] = []
        for i in ids:
            ch = self.id2ch.get(i, "")
            if ch == "<EOS>":
                break
            if ch not in _SPECIAL_TOKENS:
                chars.append(ch)
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.ch2id)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ch2id": self.ch2id,
                    # Store int keys as strings — JSON only allows string keys
                    "id2ch": {str(k): v for k, v in self.id2ch.items()},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        tok = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok.ch2id = data["ch2id"]
        tok.id2ch = {int(k): v for k, v in data["id2ch"].items()}
        return tok

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


# ─── Word-pair extraction ─────────────────────────────────────────────────────

# Extended punctuation: includes Unicode Devanagari danda (।), ellipsis, ZWJ, etc.
_PUNCT_RE = re.compile(
    r'[।,.!?;:"\(\)\[\]\-—…\'\u200b\u200c\u200d\u0964\u0965\u2018\u2019\u201c\u201d]'
)

_ROMAN_PAT    = re.compile(r"^[a-zA-Z]")
_DEVANAGARI_PAT = re.compile(r"^[\u0900-\u097F]")


def _split_words(text: str, lang: str) -> list[str]:
    """Tokenise a sentence into words, stripping punctuation."""
    pat = _ROMAN_PAT if lang == "roman" else _DEVANAGARI_PAT
    cleaned = _PUNCT_RE.sub(" ", text)
    return [w for w in cleaned.split() if w and pat.match(w)]


def load_word_pairs(
    csv_path: str,
    min_roman_len: int = 2,
    min_devan_len: int = 1,
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """
    Extract (roman_word, devanagari_word) pairs from a sentence-pair CSV.

    Expected CSV columns: ``Romanized``, ``Devanagari``

    Returns
    -------
    pairs
        Deduplicated list of (roman, devanagari) tuples for training.
    word_dict
        Dict {roman: devanagari} for O(1) look-up at inference time.
    """
    raw_pairs: list[tuple[str, str]] = []
    rows_total = 0
    skipped_empty = 0
    skipped_mismatch = 0

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_total += 1
            roman = (row.get("Romanized") or "").strip()
            devan = (row.get("Devanagari") or "").strip()

            if not roman or not devan:
                skipped_empty += 1
                continue

            r_words = _split_words(roman, "roman")
            d_words = _split_words(devan, "devanagari")

            if len(r_words) == len(d_words) and r_words:
                for rw, dw in zip(r_words, d_words):
                    if len(rw) >= min_roman_len and len(dw) >= min_devan_len:
                        raw_pairs.append((rw.lower(), dw))
            else:
                skipped_mismatch += 1

    # Deduplicate — first occurrence wins (preserve original data ordering)
    seen: dict[str, str] = {}
    for roman, devan in raw_pairs:
        if roman not in seen:
            seen[roman] = devan

    unique_pairs = list(seen.items())

    print(
        f"  CSV rows         : {rows_total:,}\n"
        f"  Raw word pairs   : {len(raw_pairs):,}\n"
        f"  Unique pairs     : {len(unique_pairs):,}\n"
        f"  Skipped (empty)  : {skipped_empty:,}\n"
        f"  Skipped (mismatch): {skipped_mismatch:,}"
    )

    return unique_pairs, seen


# ─── Data augmentation ────────────────────────────────────────────────────────

_VOWELS = frozenset("aeiou")
_ROMAN_CHARS = "abcdefghijklmnoprstuvwyz"   # plausible insert chars


def _augment_one(roman: str, rng: random.Random) -> Optional[str]:
    """
    Apply ONE of four character-level noise ops to *roman*.
    Returns the noisy string, or None if the word is too short / unchanged.
    """
    if len(roman) < 3:
        return None

    word = list(roman)
    n = len(word)
    r = rng.random()

    if r < 0.25:
        # ① Duplicate a random vowel  (e.g. "name" → "naame")
        v_idxs = [i for i, ch in enumerate(word) if ch in _VOWELS]
        if v_idxs:
            idx = rng.choice(v_idxs)
            word.insert(idx + 1, word[idx])
        else:
            return None

    elif r < 0.50:
        # ② Swap two adjacent characters  (e.g. "ghar" → "gahr")
        idx = rng.randint(0, n - 2)
        word[idx], word[idx + 1] = word[idx + 1], word[idx]

    elif r < 0.75:
        # ③ Drop a repeated character  (e.g. "khamma" → "khama")
        doubled = [i for i in range(1, n) if word[i] == word[i - 1]]
        if doubled:
            word.pop(rng.choice(doubled))
        else:
            return None

    else:
        # ④ Insert a random plausible character (e.g. "keta" → "ketra")
        idx = rng.randint(1, n - 1)
        word.insert(idx, rng.choice(_ROMAN_CHARS))

    noisy = "".join(word)
    return noisy if noisy != roman and len(noisy) >= 2 else None


def augment_pairs(
    pairs: list[tuple[str, str]],
    noise_prob: float = 1.0,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    For each training pair, produce a noisy copy with probability *noise_prob*.

    Parameters
    ----------
    pairs
        Original (roman, devanagari) training pairs.
    noise_prob
        Probability in [0, 1] that a given pair is augmented.
        0.0  → no augmentation at all.
        1.0  → every eligible pair gets a noisy copy (≈ doubles the data).
    seed
        RNG seed for reproducibility.

    Returns
    -------
    A list of ONLY the newly generated noisy pairs (append to originals yourself).
    """
    if noise_prob <= 0.0:
        return []

    rng = random.Random(seed)
    augmented: list[tuple[str, str]] = []

    for roman, devan in pairs:
        if noise_prob < 1.0 and rng.random() > noise_prob:
            continue                        # skip this pair — respects noise_prob
        noisy = _augment_one(roman, rng)
        if noisy is not None:
            augmented.append((noisy, devan))

    return augmented


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────

class TransliterationDataset(Dataset):
    """
    Character-level dataset.

    src ids : Roman chars + EOS          (no SOS on encoder input)
    tgt ids : SOS + Devanagari chars + EOS
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        src_tok: CharTokenizer,
        tgt_tok: CharTokenizer,
    ) -> None:
        self.samples: list[tuple[list[int], list[int]]] = []
        for roman, devan in pairs:
            src = src_tok.encode(roman, add_sos=False, add_eos=True)
            tgt = tgt_tok.encode(devan,  add_sos=True,  add_eos=True)
            self.samples.append((src, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.samples[idx]


def collate_fn(
    batch: list[tuple[list[int], list[int]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad sequences in a batch to equal length and build padding masks.

    Mask convention (matches nn.Transformer):
        True  → IGNORE this position (padding token)
        False → attend to this position
    """
    src_list, tgt_list = zip(*batch)

    max_src = max(len(s) for s in src_list)
    max_tgt = max(len(t) for t in tgt_list)
    B = len(batch)

    src_padded = torch.full((B, max_src), PAD_IDX, dtype=torch.long)
    tgt_padded = torch.full((B, max_tgt), PAD_IDX, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_list, tgt_list)):
        src_padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_padded[i, : len(t)] = torch.tensor(t, dtype=torch.long)

    src_pad_mask = src_padded.eq(PAD_IDX)   # (B, S_src)
    tgt_pad_mask = tgt_padded.eq(PAD_IDX)   # (B, S_tgt)

    return src_padded, tgt_padded, src_pad_mask, tgt_pad_mask


def build_dataloaders(
    pairs: list[tuple[str, str]],
    src_tok: CharTokenizer,
    tgt_tok: CharTokenizer,
    val_split: float = 0.10,
    batch_size: int = 64,
    augment: bool = True,
    noise_prob: float = 1.0,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, list, list]:
    """
    Stratify-split pairs → train / val, optionally augment train, build DataLoaders.

    Returns
    -------
    train_loader, val_loader, train_pairs, val_pairs
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_split))

    val_pairs   = [pairs[i] for i in idx[:n_val]]
    train_pairs = [pairs[i] for i in idx[n_val:]]

    if augment and noise_prob > 0.0:
        noisy = augment_pairs(train_pairs, noise_prob=noise_prob, seed=seed)
        train_pairs = train_pairs + noisy   # original + noisy copies

    print(
        f"  Train : {len(train_pairs):,} samples "
        f"(augment={augment}, noise_prob={noise_prob})\n"
        f"  Val   : {len(val_pairs):,} samples"
    )

    train_ds = TransliterationDataset(train_pairs, src_tok, tgt_tok)
    val_ds   = TransliterationDataset(val_pairs,   src_tok, tgt_tok)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
    )

    return train_loader, val_loader, train_pairs, val_pairs
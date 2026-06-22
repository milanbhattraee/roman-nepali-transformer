"""
dataset.py — Character-level Roman→Devanagari data pipeline.

Handles TWO CSV formats automatically:

  Format A (sentence-level, header row):
      Romanized,Devanagari
      Ma bajaarma kehi..., म बजारमा केही...

  Format B (word-level, no header):
      काहुँडाँडालगायतका,kahundandalgayatka
      प्रह्रीको,prahriko

Detection logic:
  1. Read first non-empty line.
  2. If it looks like a header (contains ASCII 'Romanized'/'Devanagari') → Format A.
  3. Else check which column contains Devanagari Unicode → determines column order.

Output is always a list of (roman_str, devanagari_str) word-level pairs.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ── Special token indices (must match model.py & trainer.py) ─────────────────
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
EOW_IDX = 4   # End-of-Word — hard boundary that stops repetition loops

_SPECIAL = {"<PAD>", "<SOS>", "<EOS>", "<UNK>", "<EOW>"}


# ── Tokenizer ─────────────────────────────────────────────────────────────────
class CharTokenizer:
    """Character-level tokenizer with explicit special tokens."""

    def __init__(self) -> None:
        self.ch2id: dict[str, int] = {
            "<PAD>": PAD_IDX,
            "<SOS>": SOS_IDX,
            "<EOS>": EOS_IDX,
            "<UNK>": UNK_IDX,
            "<EOW>": EOW_IDX,
        }
        self.id2ch: dict[int, str] = {v: k for k, v in self.ch2id.items()}

    def build(self, texts: list[str]) -> "CharTokenizer":
        chars: set[str] = set()
        for t in texts:
            chars.update(t)
        for ch in sorted(chars):
            if ch not in self.ch2id:
                idx = len(self.ch2id)
                self.ch2id[ch] = idx
                self.id2ch[idx] = ch
        return self

    def encode(self, text: str, add_sos: bool = False, add_eos: bool = True,
               add_eow: bool = False) -> list[int]:
        ids: list[int] = []
        if add_sos:
            ids.append(SOS_IDX)
        for ch in text:
            ids.append(self.ch2id.get(ch, UNK_IDX))
        if add_eow:
            ids.append(EOW_IDX)
        if add_eos:
            ids.append(EOS_IDX)
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for i in ids:
            ch = self.id2ch.get(i, "")
            if ch == "<EOS>":
                break
            if ch not in _SPECIAL:
                out.append(ch)
        return "".join(out)

    @property
    def vocab_size(self) -> int:
        return len(self.ch2id)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"ch2id": self.ch2id,
                 "id2ch": {str(k): v for k, v in self.id2ch.items()}},
                f, ensure_ascii=False, indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        tok = cls()
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        tok.ch2id = d["ch2id"]
        tok.id2ch = {int(k): v for k, v in d["id2ch"].items()}
        return tok

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


# ── CSV format detection ───────────────────────────────────────────────────────
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_ROMAN_RE      = re.compile(r"^[a-zA-Z]")
_PUNCT_RE      = re.compile(
    r'[।,.!?;:"\(\)\[\]\-—…\'\u200b\u200c\u200d\u0964\u0965]'
)
_WS_RE = re.compile(r"\s+")


def _is_devanagari(s: str) -> bool:
    return bool(_DEVANAGARI_RE.search(s))


def _detect_format(csv_path: str) -> tuple[str, int, int]:
    """
    Returns (format, roman_col, devan_col).
    format is 'sentence' (has header, sentence-level) or 'word' (no header, word-level).
    roman_col / devan_col are 0-based column indices.
    """
    with open(csv_path, encoding="utf-8") as f:
        first = ""
        for line in f:
            line = line.strip()
            if line:
                first = line
                break

    if not first:
        raise ValueError("CSV file appears empty.")

    cols = first.split(",", 1)
    if len(cols) < 2:
        raise ValueError(f"CSV must have at least 2 columns. Got: {first!r}")

    c0, c1 = cols[0].strip(), cols[1].strip()

    # Header row detection
    if "romanized" in c0.lower() or "roman" in c0.lower():
        return "sentence", 0, 1   # Roman first, Devanagari second
    if "devanagari" in c0.lower() or "nepali" in c0.lower():
        return "sentence", 1, 0   # Devanagari first, Roman second
    if "romanized" in c1.lower() or "roman" in c1.lower():
        return "sentence", 1, 0

    # No header — detect by content
    if _is_devanagari(c0) and not _is_devanagari(c1):
        return "word", 1, 0   # Devanagari first, Roman second
    if _is_devanagari(c1) and not _is_devanagari(c0):
        return "word", 0, 1   # Roman first, Devanagari second

    # Default: assume Roman first
    return "word", 0, 1


# ── Word-pair extraction ──────────────────────────────────────────────────────
def _split_sentence_words(roman: str, devan: str) -> list[tuple[str, str]]:
    """Extract aligned (roman, devanagari) word pairs from a sentence pair."""
    r_words = [w.lower() for w in _PUNCT_RE.sub(" ", roman).split()
               if w and _ROMAN_RE.match(w)]
    d_words = [w for w in _PUNCT_RE.sub(" ", devan).split()
               if w and _DEVANAGARI_RE.match(w)]

    if len(r_words) != len(d_words) or not r_words:
        return []

    return [(r, d) for r, d in zip(r_words, d_words)
            if len(r) >= 2 and len(d) >= 1]


def load_word_pairs(
    csv_path: str,
    max_len_ratio: float = 4.0,
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """
    Load (roman, devanagari) word pairs from a CSV file.
    Auto-detects whether it is sentence-level or word-level, and
    which column is Roman vs Devanagari.

    Parameters
    ----------
    csv_path      : path to CSV file
    max_len_ratio : drop pairs where len(devan) > ratio * len(roman)
                    (removes noise that causes repetition loops)

    Returns
    -------
    pairs     : deduplicated list of (roman, devanagari)
    word_dict : {roman: devanagari} for O(1) lookup at inference
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    fmt, roman_col, devan_col = _detect_format(csv_path)
    logger.info("CSV format detected: %s  roman_col=%d  devan_col=%d",
                fmt, roman_col, devan_col)

    raw_pairs: list[tuple[str, str]] = []
    skipped = 0

    with open(csv_path, encoding="utf-8") as f:
        # Skip header row if format is 'sentence'
        content = f.read().strip().splitlines()

    for line in content:
        line = line.strip()
        if not line:
            continue

        parts = line.split(",", max(roman_col, devan_col))
        if len(parts) <= max(roman_col, devan_col):
            skipped += 1
            continue

        roman_raw = parts[roman_col].strip()
        devan_raw = parts[devan_col].strip()

        # Skip header line (detected retroactively)
        if not _is_devanagari(devan_raw) or not roman_raw:
            skipped += 1
            continue

        if fmt == "sentence":
            # Sentence-level: extract word pairs
            pairs_from_line = _split_sentence_words(roman_raw, devan_raw)
            raw_pairs.extend(pairs_from_line)
        else:
            # Word-level: use directly
            roman_clean = roman_raw.lower().strip()
            devan_clean = devan_raw.strip()

            # Length ratio filter — kills the main repetition cause
            if len(devan_clean) > max_len_ratio * max(len(roman_clean), 1):
                skipped += 1
                continue
            if len(roman_clean) >= 2 and len(devan_clean) >= 1:
                raw_pairs.append((roman_clean, devan_clean))

    # Deduplicate: first occurrence wins
    seen: dict[str, str] = {}
    for roman, devan in raw_pairs:
        if roman not in seen:
            seen[roman] = devan

    unique_pairs = list(seen.items())

    logger.info(
        "Loaded %d raw pairs → %d unique (skipped %d)",
        len(raw_pairs), len(unique_pairs), skipped,
    )
    if len(unique_pairs) == 0:
        raise ValueError(
            "No valid pairs extracted. Check column order and encoding."
        )

    return unique_pairs, seen


# ── Augmentation ──────────────────────────────────────────────────────────────
_VOWELS      = frozenset("aeiou")
_ROMAN_CHARS = "abcdefghijklmnoprstuvwyz"


def _augment_one(roman: str, rng: random.Random) -> Optional[str]:
    if len(roman) < 3:
        return None
    word = list(roman)
    n    = len(word)
    r    = rng.random()

    if r < 0.25:
        vi = [i for i, c in enumerate(word) if c in _VOWELS]
        if vi:
            i = rng.choice(vi)
            word.insert(i + 1, word[i])
        else:
            return None
    elif r < 0.50:
        i = rng.randint(0, n - 2)
        word[i], word[i+1] = word[i+1], word[i]
    elif r < 0.75:
        dd = [i for i in range(1, n) if word[i] == word[i-1]]
        if dd:
            word.pop(rng.choice(dd))
        else:
            return None
    else:
        word.insert(rng.randint(1, n-1), rng.choice(_ROMAN_CHARS))

    noisy = "".join(word)
    return noisy if noisy != roman and len(noisy) >= 2 else None


def augment_pairs(
    pairs: list[tuple[str, str]],
    noise_prob: float = 1.0,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Return noisy copies of pairs (same target, perturbed source)."""
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    for roman, devan in pairs:
        if noise_prob < 1.0 and rng.random() > noise_prob:
            continue
        noisy = _augment_one(roman, rng)
        if noisy:
            out.append((noisy, devan))
    return out


# ── Dataset ───────────────────────────────────────────────────────────────────
class TransliterationDataset(Dataset):
    """
    src : roman chars + EOS          (no SOS — encoder reads full input)
    tgt : SOS + devanagari + EOW + EOS
    EOW gives the decoder a hard word-boundary signal that prevents
    repetition loops common in open-vocabulary seq2seq.
    """

    def __init__(self, pairs: list[tuple[str, str]],
                 src_tok: CharTokenizer, tgt_tok: CharTokenizer,
                 add_eow: bool = True) -> None:
        self.data: list[tuple[list[int], list[int]]] = []
        for roman, devan in pairs:
            src = src_tok.encode(roman, add_sos=False, add_eos=True)
            tgt = tgt_tok.encode(devan,  add_sos=True,  add_eos=True,
                                  add_eow=add_eow)
            self.data.append((src, tgt))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def collate_fn(batch):
    src_list, tgt_list = zip(*batch)
    B = len(batch)
    max_src = max(len(s) for s in src_list)
    max_tgt = max(len(t) for t in tgt_list)

    src_pad = torch.full((B, max_src), PAD_IDX, dtype=torch.long)
    tgt_pad = torch.full((B, max_tgt), PAD_IDX, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_list, tgt_list)):
        src_pad[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_pad[i, :len(t)] = torch.tensor(t, dtype=torch.long)

    return src_pad, tgt_pad, src_pad.eq(PAD_IDX), tgt_pad.eq(PAD_IDX)


def build_dataloaders(
    pairs: list[tuple[str, str]],
    src_tok: CharTokenizer,
    tgt_tok: CharTokenizer,
    val_split: float = 0.10,
    batch_size: int = 64,
    augment: bool = True,
    noise_prob: float = 1.0,
    add_eow: bool = True,
    seed: int = 42,
    num_workers: int = 0,
):
    if not pairs:
        raise ValueError("pairs is empty")

    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_split))

    val_p   = [pairs[i] for i in idx[:n_val]]
    train_p = [pairs[i] for i in idx[n_val:]]

    if augment and noise_prob > 0:
        noisy   = augment_pairs(train_p, noise_prob=noise_prob, seed=seed)
        train_p = train_p + noisy

    logger.info("DataLoaders: train=%d val=%d add_eow=%s", len(train_p), len(val_p), add_eow)

    kw = dict(collate_fn=collate_fn, num_workers=num_workers,
              pin_memory=(num_workers > 0))

    train_ds = TransliterationDataset(train_p, src_tok, tgt_tok, add_eow=add_eow)
    val_ds   = TransliterationDataset(val_p,   src_tok, tgt_tok, add_eow=add_eow)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)

    return train_loader, val_loader, train_p, val_p
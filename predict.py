#!/usr/bin/env python3
"""
predict.py — Production inference for Roman Nepali → Devanagari

Loads the trained model from models/ and transliterates Roman Nepali text
using three layers in order:

  Layer 1 — Dictionary  : exact match from training data  (100% accurate)
  Layer 2 — Spell fix   : edit-distance correction for typos
  Layer 3 — Neural model: beam search for unseen words

Usage
─────
  python predict.py "k xa khabar"
  python predict.py "hajurko hajurlai hajurle" --verbose
  python predict.py --interactive
  python predict.py --evaluate --csv data/roman_nepali_clean.csv
  python predict.py --beam-width 5

  # In Python:
  from predict import Predictor
  p = Predictor.load()
  print(p.translate("k xa khabar tapai"))
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CharTokenizer
from model import Seq2SeqTransformer, beam_search, greedy_decode


# ── Levenshtein (for spell correction) ──────────────────────────────────────
def _lev(a: str, b: str) -> int:
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


# ── Chat shortcuts (highest-priority overrides) ───────────────────────────────
CHAT_SHORTCUTS: dict[str, str] = {
    "k":       "के",
    "xa":      "छ",
    "x":       "छ",
    "xaina":   "छैन",
    "xainan":  "छैनन्",
    "xainana": "छैनन्",
    "xau":     "छौ",
    "xan":     "छन्",
    "ho":      "हो",
    "hoina":   "होइन",
}

# ── Punctuation map ──────────────────────────────────────────────────────────
_PUNCT = {".": "।", "?": "?", "!": "!", ",": ",", ";": ";", ":": ":"}
_DIGIT = str.maketrans("0123456789", "०१२३४५६७८९")


class Predictor:
    """
    Production transliterator.  Combine dictionary + spell correction + model.

    Load once, call translate() many times.
    """

    def __init__(
        self,
        model:      Seq2SeqTransformer,
        src_tok:    CharTokenizer,
        tgt_tok:    CharTokenizer,
        word_dict:  dict[str, str],
        device:     torch.device,
        beam_width: int   = 5,
        repetition_penalty: float = 1.3,
        spell_ratio: float = 0.00,
    ) -> None:
        self.model    = model
        self.src_tok  = src_tok
        self.tgt_tok  = tgt_tok
        self.device   = device
        self.beam_width = beam_width
        self.rep_pen  = repetition_penalty
        self.spell_ratio = spell_ratio

        # Merge word dict with chat shortcuts (shortcuts win)
        self.word_dict: dict[str, str] = {k.lower(): v for k, v in word_dict.items()}
        for k, v in CHAT_SHORTCUTS.items():
            self.word_dict[k.lower()] = v

        # Sorted vocabulary for fast spell-correction candidate search
        self._vocab_by_len: dict[int, list[str]] = {}
        for w in self.word_dict:
            self._vocab_by_len.setdefault(len(w), []).append(w)

        self._sos = tgt_tok.ch2id.get("<SOS>", 1)
        self._eos = tgt_tok.ch2id.get("<EOS>", 2)

    # ── Factory ──────────────────────────────────────────────────────────────
    @classmethod
    def load(
        cls,
        model_dir:  str   = "models",
        beam_width: int   = 5,
        repetition_penalty: float = 1.3,
        spell_ratio: float = 0.0,
        device:     str | None = None,
    ) -> "Predictor":
        """Load everything from models/ and return a ready Predictor."""

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Model directory '{model_dir}' not found. "
                "Run train.py first."
            )

        # ── Device ───────────────────────────────────────────────────────
        if device:
            dev = torch.device(device)
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")

        # ── Tokenizers ────────────────────────────────────────────────────
        src_tok, tgt_tok = cls._load_tokenizers(model_dir)

        # ── Model architecture ────────────────────────────────────────────
        mcfg = cls._load_model_config(model_dir, src_tok, tgt_tok)

        model = Seq2SeqTransformer(**mcfg).to(dev)

        # ── Weights ───────────────────────────────────────────────────────
        best_pt = os.path.join(model_dir, "best_model.pt")
        last_pt = os.path.join(model_dir, "last_checkpoint.pt")

        ckpt_path = best_pt if os.path.isfile(best_pt) else last_pt
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"No checkpoint found in {model_dir}. "
                "Run train.py first."
            )

        ckpt = torch.load(ckpt_path, map_location=dev)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        model.eval()

        best_acc = ckpt.get("best_word_acc", ckpt.get("best_cer", "?"))
        print(f"[Predictor] Loaded from '{ckpt_path}'")
        print(f"[Predictor] Device={dev}  src_vocab={src_tok.vocab_size}"
              f"  tgt_vocab={tgt_tok.vocab_size}  best_metric={best_acc}")

        # ── Word dictionary ───────────────────────────────────────────────
        wd_path = os.path.join(model_dir, "word_dictionary.json")
        word_dict: dict[str, str] = {}
        if os.path.isfile(wd_path):
            with open(wd_path, encoding="utf-8") as f:
                word_dict = json.load(f)
            print(f"[Predictor] Dictionary: {len(word_dict):,} entries")
        else:
            print("[Predictor] WARNING: word_dictionary.json not found; "
                  "dict lookup disabled.")

        return cls(model, src_tok, tgt_tok, word_dict, dev,
                   beam_width=beam_width,
                   repetition_penalty=repetition_penalty,
                   spell_ratio=spell_ratio)

    # ── Tokenizer loading (handles both save formats) ─────────────────────────
    @staticmethod
    def _load_tokenizers(model_dir: str) -> tuple[CharTokenizer, CharTokenizer]:
        """
        Handles two JSON layouts:
          Layout A (train.py output):
              {"src": {"ch2id": ..., "id2ch": ...}, "tgt": {...}}
          Layout B (older / alternative):
              {"ch2id": ..., "id2ch": ...}   (single tokenizer)
        """
        tok_path = os.path.join(model_dir, "tokenizers.json")
        if not os.path.isfile(tok_path):
            raise FileNotFoundError(f"tokenizers.json not found in {model_dir}")

        with open(tok_path, encoding="utf-8") as f:
            raw = json.load(f)

        def _make(d: dict) -> CharTokenizer:
            t = CharTokenizer()
            t.ch2id = {k: int(v) for k, v in d["ch2id"].items()}
            t.id2ch = {int(k): v for k, v in d["id2ch"].items()}
            return t

        if "src" in raw and "tgt" in raw:
            # Layout A — standard
            src_tok = _make(raw["src"])
            tgt_tok = _make(raw["tgt"])
        elif "ch2id" in raw:
            # Layout B — single tokenizer (used for both sides)
            tok = _make(raw)
            src_tok = tok
            tgt_tok = tok
        else:
            raise ValueError(
                "Unrecognised tokenizers.json format. "
                "Expected {'src': ..., 'tgt': ...} or {'ch2id': ...}"
            )

        return src_tok, tgt_tok

    # ── Model config loading ──────────────────────────────────────────────────
    @staticmethod
    @staticmethod
    def _load_model_config(
        model_dir: str,
        src_tok:   CharTokenizer,
        tgt_tok:   CharTokenizer,
    ) -> dict:
        """
        Try model_config.json first; fall back to inferring from checkpoint
        tensor shapes; final fallback: use safe defaults.
        """
        cfg_path = os.path.join(model_dir, "model_config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            # Ensure vocab sizes match loaded tokenizers
            cfg["src_vocab_size"] = src_tok.vocab_size
            cfg["tgt_vocab_size"] = tgt_tok.vocab_size
            return cfg

        # Try to infer from checkpoint
        for fname in ("best_model.pt", "last_checkpoint.pt"):
            pt = os.path.join(model_dir, fname)
            if not os.path.isfile(pt):
                continue
            try:
                ckpt  = torch.load(pt, map_location="cpu")
                state = ckpt.get("model_state", ckpt)
                
                # d_model from src embedding shape
                d_model = state["src_embed.weight"].shape[1]
                
                # dim_feedforward: from linear1 weight
                ff_dim  = state.get(
                    "transformer.encoder.layers.0.linear1.weight",
                    torch.zeros(512, d_model)
                ).shape[0]
                
                # Correctly find unique layer indices instead of counting all keys
                enc_layers = set()
                dec_layers = set()
                for k in state:
                    enc_match = re.match(r"transformer\.encoder\.layers\.(\d+)\.", k)
                    if enc_match:
                        enc_layers.add(int(enc_match.group(1)))
                    dec_match = re.match(r"transformer\.decoder\.layers\.(\d+)\.", k)
                    if dec_match:
                        dec_layers.add(int(dec_match.group(1)))
                
                n_enc = len(enc_layers)
                n_dec = len(dec_layers)
                nhead = 4 # default to 4 if can't determine from structure

                print(f"[Predictor] Inferred arch from {fname}: "
                      f"d_model={d_model} ff={ff_dim} enc={n_enc} dec={n_dec}")
                return {
                    "src_vocab_size": src_tok.vocab_size,
                    "tgt_vocab_size": tgt_tok.vocab_size,
                    "d_model":            d_model,
                    "nhead":              nhead,
                    "num_encoder_layers": max(n_enc, 1),
                    "num_decoder_layers": max(n_dec, 1),
                    "dim_feedforward":    ff_dim,
                    "dropout":            0.0,   # no dropout at inference
                    "max_seq_len":        60,
                }
            except Exception as e:
                print(f"[Predictor] Could not infer arch from {fname}: {e}")
                continue

        # Last resort: safe defaults (matches Config defaults)
        print("[Predictor] WARNING: using default architecture. "
              "If this fails to load weights, check d_model / n_heads.")
        return {
            "src_vocab_size": src_tok.vocab_size,
            "tgt_vocab_size": tgt_tok.vocab_size,
            "d_model": 256, "nhead": 4,
            "num_encoder_layers": 4, "num_decoder_layers": 4,
            "dim_feedforward": 512, "dropout": 0.0, "max_seq_len": 60,
        }
    # ── Spell correction ─────────────────────────────────────────────────────
    def _spell_correct(self, word: str) -> str | None:
        """Return closest dictionary word within edit ratio, or None."""
        if self.spell_ratio <= 0.0:
            return None
        max_dist = max(1, round(len(word) * self.spell_ratio))
        best_w, best_d = None, max_dist + 1

        for L in range(max(1, len(word) - max_dist),
                       len(word) + max_dist + 1):
            for cand in self._vocab_by_len.get(L, []):
                d = _lev(word, cand)
                if d < best_d:
                    best_d, best_w = d, cand
                if best_d == 0:
                    return best_w
        return best_w if best_w is not None else None

    # ── Single-word transliteration ──────────────────────────────────────────
    def _word(self, roman: str, use_beam: bool = True,
              verbose: bool = False) -> str:
        w = roman.lower().strip()
        if not w:
            return roman

        # Layer 1: exact dict lookup
        if w in self.word_dict:
            if verbose:
                print(f"  {roman:<28} → {self.word_dict[w]:<20} [DICT]")
            return self.word_dict[w]

        # Layer 2: spell correction
        corrected = self._spell_correct(w)
        if corrected and corrected in self.word_dict:
            result = self.word_dict[corrected]
            if verbose:
                print(f"  {roman:<28} → {result:<20} [SPELL→{corrected}]")
            return result

        # Layer 3: neural model
        src_ids = self.src_tok.encode(w, add_sos=True, add_eos=True)
        if use_beam:
            out_ids = beam_search(
                self.model, src_ids, self._sos, self._eos, self.device,
                beam_width=self.beam_width,
                repetition_penalty=self.rep_pen,
            )
        else:
            out_ids = greedy_decode(
                self.model, src_ids, self._sos, self._eos, self.device,
                repetition_penalty=self.rep_pen,
            )
        result = self.tgt_tok.decode(out_ids)
        if verbose:
            print(f"  {roman:<28} → {result:<20} [MODEL]")
        return result

    # ── Sentence transliteration ─────────────────────────────────────────────
    def translate(self, text: str, use_beam: bool = True,
                  verbose: bool = False) -> str:
        """Transliterate a Roman Nepali sentence to Devanagari."""
        if not text or not text.strip():
            return text

        tokens = self._tokenise(text)
        if verbose:
            print(f"\nInput : {text}")
            print("─" * 55)
        out: list[str] = []
        for kind, val in tokens:
            if kind == "word":
                out.append(self._word(val, use_beam=use_beam, verbose=verbose))
            elif kind == "number":
                out.append(val.translate(_DIGIT))
            elif kind == "space":
                out.append(" ")
            elif kind == "punct":
                out.append(_PUNCT.get(val, val))
            else:
                out.append(val)
        result = "".join(out).strip()
        if verbose:
            print("─" * 55)
            print(f"Output: {result}\n")
        return result

    def translate_batch(self, texts: list[str], use_beam: bool = True) -> list[str]:
        return [self.translate(t, use_beam=use_beam) for t in texts]

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    @staticmethod
    def _tokenise(text: str) -> list[tuple[str, str]]:
        tokens: list[tuple[str, str]] = []
        buf: list[str] = []

        def flush():
            if buf:
                tokens.append(("word", "".join(buf)))
                buf.clear()

        for ch in text:
            if ch.isalpha() or ch == "'":
                buf.append(ch)
            else:
                flush()
                if ch.isdigit():
                    tokens.append(("number", ch))
                elif ch == " ":
                    tokens.append(("space", " "))
                elif ch in ".!?,;:":
                    tokens.append(("punct", ch))
                else:
                    tokens.append(("other", ch))
        flush()

        # Merge consecutive digits
        merged: list[list] = []
        for k, v in tokens:
            if merged and merged[-1][0] == "number" and k == "number":
                merged[-1][1] += v
            else:
                merged.append([k, v])
        return [(k, v) for k, v in merged]


# ── Evaluation helper ─────────────────────────────────────────────────────────
def _run_evaluate(predictor: Predictor, csv_path: str) -> None:
    import csv as _csv
    _PUNCT_RE = re.compile(r'[।,.!?;:"\(\)\[\]\-—…\'\u200b\u0964\u0965]')
    _DEV_RE   = re.compile(r"[\u0900-\u097F]")
    _ROM_RE   = re.compile(r"^[a-zA-Z]")

    total = exact = 0
    total_edit = total_len = 0
    wrong: list[tuple] = []

    # auto-detect format (same logic as dataset.py)
    from dataset import _detect_format
    _, roman_col, devan_col = _detect_format(csv_path)

    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", max(roman_col, devan_col))
            if len(parts) <= max(roman_col, devan_col):
                continue
            roman_raw = parts[roman_col].strip()
            devan_raw = parts[devan_col].strip()
            if not _DEV_RE.search(devan_raw) or not roman_raw:
                continue

            r_words = [w.lower() for w in _PUNCT_RE.sub(" ", roman_raw).split()
                       if w and _ROM_RE.match(w)]
            d_words = [w for w in _PUNCT_RE.sub(" ", devan_raw).split()
                       if w and _DEV_RE.match(w)]

            if len(r_words) != len(d_words):
                continue

            for rw, dw in zip(r_words, d_words):
                pred = predictor._word(rw, use_beam=True)
                total += 1
                if pred == dw:
                    exact += 1
                else:
                    total_edit += _lev(pred, dw)
                    total_len  += max(len(dw), 1)
                    if len(wrong) < 20:
                        wrong.append((rw, pred, dw))

    cer = total_edit / max(total_len, 1)
    print("\n" + "═" * 65)
    print("  EVALUATION")
    print("═" * 65)
    print(f"  Word pairs  : {total:,}")
    print(f"  Exact match : {exact:,} ({100*exact/max(total,1):.1f}%)")
    print(f"  CER         : {cer:.4f}  (char accuracy {(1-cer)*100:.1f}%)")
    if wrong:
        print("\n  Mismatches (first 20):")
        for rw, pred, dw in wrong:
            print(f"    {rw:25s} → {pred:20s}  [expected: {dw}]")
    print("═" * 65)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Roman Nepali → Devanagari predictor")
    ap.add_argument("text",          nargs="?", default=None)
    ap.add_argument("--model-dir",   default="models")
    ap.add_argument("--beam-width",  type=int,   default=5)
    ap.add_argument("--rep-penalty", type=float, default=1.3)
    ap.add_argument("--greedy",      action="store_true",
                    help="Use greedy decoding (faster, slightly worse)")
    ap.add_argument("--verbose",  "-v", action="store_true")
    ap.add_argument("--interactive","-i", action="store_true")
    ap.add_argument("--evaluate", "-e", action="store_true")
    ap.add_argument("--csv",  default="data/roman_nepali_clean.csv")
    ap.add_argument("--spell-ratio", type=float, default=0.0, help="Spell check threshold")
    args = ap.parse_args()

    predictor = Predictor.load(
        model_dir=args.model_dir,
        beam_width=args.beam_width,
        repetition_penalty=args.rep_penalty,
        spell_ratio=args.spell_ratio,
    )
    use_beam = not args.greedy

    if args.evaluate:
        _run_evaluate(predictor, args.csv)
        return
    

    if args.interactive:
        print("\n" + "═" * 55)
        print("  Roman Nepali → Devanagari  (type 'exit' to quit)")
        print("═" * 55 + "\n")
        while True:
            try:
                line = input("Roman  : ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if line.lower() in ("exit", "quit", "q"):
                break
            if not line:
                continue
            print(f"Nepali : {predictor.translate(line, use_beam=use_beam)}\n")
        return

    if args.text:
        print(predictor.translate(args.text, use_beam=use_beam,
                                  verbose=args.verbose))
        return

    # No args — show demo
    # No args — show demo
    demos = [
        # Greetings & Checking In
        "k xa khabar",
        "sanchai hunuhunchha",
        "khana khanu vayo",
        "ghar ma sabai lai kasto chha",
        "namaste ramro din",
        "aama k gardai hunuhunchha",
        
        # Casual Conversation & Plans
        "aaja k garne bichar chha",
        "bholi bata naya kaam suru garne",
        "kata harako yati dherai din",
        "chiya kham na ta",
        "aba k garne ta",
        "bholi bhetumla hai ta",
        "kata jadaichhau ahile",
        
        # Student & Tech Life
        "project ko kaam kasto hudai cha",
        "bholi college jana parchha",
        "code run bhayo ta",
        "mero laptop ali slow chha yar",
        "assignment sakiyo ki baki chha",
        "database ma error aayo",
        
        # Market & Daily Operations
        "aaja market kasto chha",
        "share ko vau badhyo ki ghatyo",
        "paisako ali tension chha",
        "hisab kitab milouna parne chha",
        
        # Opinions, Expressions & Fillers
        "malai Nepal man parcha",
        "kasto ramro kura garnu vayo",
        "malai euta kura sodhna man thiyo",
        "hasayo yar timile tyo kura le",
        "k ma timro hoina ra",
        "ho ra maile ta thahai paina",
        "la la hunchha ni ta",
        "thikai chha ni aba yestai ho",
        "hait khatra lagyo malai ta"
    ]
    print("\n  Quick demo:")
    for d in demos:
        print(f"  {d:<35} → {predictor.translate(d, use_beam=use_beam)}")
    print()
    print('  Usage: python predict.py "k xa khabar tapai"')
    print('         python predict.py --interactive')
    print('         python predict.py --evaluate')
    print()


if __name__ == "__main__":
    main()
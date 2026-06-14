"""
model.py  ─  Transformer Seq2Seq for Roman Nepali → Devanagari
═══════════════════════════════════════════════════════════════════════════════
Architecture  (character-level, word-by-word)
─────────────────────────────────────────────
  Encoder  : Char embeddings → Positional encoding → N × TransformerEncoderLayer
  Decoder  : Char embeddings → Positional encoding → N × TransformerDecoderLayer
             (masked self-attention + cross-attention to encoder memory)
  Head     : Linear(d_model → tgt_vocab_size)

Fixes vs original
──────────────────
  ① Weight tying condition was `if tgt_vocab_size == d_model` — almost always
    False and silently skipped.  Fixed to tie whenever embedding shape matches
    projection shape (tgt_vocab_size, d_model), which is always true here.

  ② _init_weights used Xavier uniform on ALL parameters including embedding
    tables.  Embedding weights should use PyTorch's default (normal(0, 1))
    so the model can learn meaningful initial similarities; Xavier is for
    linear/projection layers only.

  ③ Beam search length-penalty exponent was applied to a sequence that still
    contained the SOS token → off-by-one in length.  Fixed.

  ④ greedy_decode / beam_search now accept an explicit max_len guard so they
    can never run forever if EOS is suppressed.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Positional Encoding ──────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding (Vaswani et al., 2017).
    Adds position information to the token embeddings.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute encoding matrix once
        pos = torch.arange(max_len).unsqueeze(1)                          # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
        )                                                                  # (d_model/2,)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)                                     # (max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, d_model)"""
        x = x + self.pe[: x.size(1)].unsqueeze(0)
        return self.dropout(x)


# ─── Main Model ───────────────────────────────────────────────────────────────

class Seq2SeqTransformer(nn.Module):
    """
    Encoder–Decoder Transformer for character-level transliteration.

    Training (teacher forcing)
    ──────────────────────────
        logits = model(src, tgt, src_pad_mask, tgt_pad_mask)
        # logits: (B, S_tgt - 1, tgt_vocab_size)

    Inference
    ─────────
        ids = greedy_decode(model, src_ids, sos_idx, eos_idx, device)
        ids = beam_search(model,   src_ids, sos_idx, eos_idx, device)
    """

    def __init__(
        self,
        src_vocab_size:      int,
        tgt_vocab_size:      int,
        d_model:             int   = 256,
        nhead:               int   = 4,
        num_encoder_layers:  int   = 4,
        num_decoder_layers:  int   = 4,
        dim_feedforward:     int   = 512,
        dropout:             float = 0.1,
        max_seq_len:         int   = 60,
        pad_idx:             int   = 0,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )

        self.d_model  = d_model
        self.pad_idx  = pad_idx

        # ── Embedding tables ──────────────────────────────────────────────
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        # ── Positional encoding ───────────────────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=max_seq_len + 10)

        # ── Transformer core ──────────────────────────────────────────────
        self.transformer = nn.Transformer(
            d_model            = d_model,
            nhead              = nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward    = dim_feedforward,
            dropout            = dropout,
            batch_first        = True,   # (B, S, D) convention throughout
            norm_first         = True,   # Pre-LN: more stable early training
        )

        # ── Output projection ─────────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # ── Weight tying ──────────────────────────────────────────────────
        # FIX: original condition `if tgt_vocab_size == d_model` almost never
        # fires.  The correct test is whether tgt_embed.weight and
        # output_proj.weight have the same shape — they always do
        # (tgt_vocab_size, d_model) — so we always tie.
        self.output_proj.weight = self.tgt_embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Xavier uniform for Linear/projection layers only.
        Embedding weights intentionally left at PyTorch default (N(0,1)) so
        they can learn meaningful initial similarities from the data.
        Zero-init all biases.

        FIX: original code applied Xavier to ALL parameters with dim > 1,
        including embeddings — incorrect and hurts early-training dynamics.
        """
        for name, p in self.named_parameters():
            # Skip embedding tables — they have their own good default init
            if "embed" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    # ── Forward (teacher-forcing, training) ───────────────────────────────

    def forward(
        self,
        src:          torch.Tensor,              # (B, S_src)
        tgt:          torch.Tensor,              # (B, S_tgt) — SOS…(no EOS)
        src_pad_mask: torch.Tensor | None = None,  # (B, S_src) True=pad
        tgt_pad_mask: torch.Tensor | None = None,  # (B, S_tgt) True=pad
    ) -> torch.Tensor:                           # (B, S_tgt, V)
        """
        Teacher-forcing forward pass.

        ``tgt`` must be the shifted-right target:
            [SOS, c1, c2, ..., cn]   (EOS is NOT fed in; it is the label)
        Returned logits correspond to:
            [c1,  c2, ..., cn, EOS]
        """
        S_tgt = tgt.size(1)
        tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(
            S_tgt, device=src.device
        )   # (S_tgt, S_tgt) — −inf on upper triangle

        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask                = tgt_causal_mask,
            src_key_padding_mask    = src_pad_mask,
            tgt_key_padding_mask    = tgt_pad_mask,
            memory_key_padding_mask = src_pad_mask,
        )   # (B, S_tgt, d_model)

        return self.output_proj(out)   # (B, S_tgt, V)

    # ── Encoder-only (inference) ──────────────────────────────────────────

    def encode(
        self,
        src:          torch.Tensor,
        src_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode source sequence. Returns memory (B, S_src, d_model)."""
        emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(emb, src_key_padding_mask=src_pad_mask)

    # ── One decoder step (inference) ──────────────────────────────────────

    def decode_step(
        self,
        tgt:             torch.Tensor,   # (B, S_so_far)
        memory:          torch.Tensor,   # (B, S_src, d_model)
        tgt_causal_mask: torch.Tensor,   # (S_so_far, S_so_far)
        mem_pad_mask:    torch.Tensor | None = None,
    ) -> torch.Tensor:                   # (B, S_so_far, V)
        """Single autoregressive decoder step → token logits."""
        emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            emb, memory,
            tgt_mask                = tgt_causal_mask,
            memory_key_padding_mask = mem_pad_mask,
        )
        return self.output_proj(out)


# ─── Greedy Decoding ─────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(
    model:   Seq2SeqTransformer,
    src_ids: list[int],
    sos_idx: int,
    eos_idx: int,
    device:  torch.device,
    max_len: int = 60,
) -> list[int]:
    """
    Fastest decoder: greedily picks the highest-probability token at each step.
    Best for quick validation; use beam_search for final inference quality.
    """
    model.eval()

    src     = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad = src.eq(model.pad_idx)
    memory  = model.encode(src, src_pad)

    tgt = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
    output: list[int] = []

    for _ in range(max_len):
        S = tgt.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(S, device=device)
        logits = model.decode_step(tgt, memory, causal)       # (1, S, V)
        next_id: int = logits[0, -1].argmax().item()          # type: ignore[assignment]

        if next_id == eos_idx:
            break
        output.append(next_id)
        tgt = torch.cat(
            [tgt, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )

    return output


# ─── Beam Search Decoding ────────────────────────────────────────────────────

@torch.no_grad()
def beam_search(
    model:       Seq2SeqTransformer,
    src_ids:     list[int],
    sos_idx:     int,
    eos_idx:     int,
    device:      torch.device,
    beam_width:  int   = 5,
    max_len:     int   = 60,
    len_penalty: float = 0.6,   # α in Google NMT length penalty
) -> list[int]:
    """
    Beam search: maintains the top-K partial hypotheses at every step.
    Produces higher-quality output than greedy at ~K× the compute cost.

    FIX: original code counted len(new_seq) which included the SOS token,
    making the length penalty off by one.  Fixed to use len(new_seq) - 1
    (the actual number of generated tokens, excluding SOS).
    """
    model.eval()

    src     = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad = src.eq(model.pad_idx)
    memory  = model.encode(src, src_pad)

    # Hypothesis: (cumulative_log_score, token_ids_including_SOS)
    beams: list[tuple[float, list[int]]] = [(0.0, [sos_idx])]
    completed: list[tuple[float, list[int]]] = []

    for _ in range(max_len):
        if not beams:
            break

        candidates: list[tuple[float, list[int]]] = []

        for score, seq in beams:
            tgt    = torch.tensor([seq], dtype=torch.long, device=device)
            S      = tgt.size(1)
            causal = nn.Transformer.generate_square_subsequent_mask(S, device=device)
            logits = model.decode_step(tgt, memory, causal)          # (1, S, V)
            log_p  = F.log_softmax(logits[0, -1], dim=-1)            # (V,)

            top_log_p, top_ids = log_p.topk(beam_width)

            for lp, tid in zip(top_log_p.tolist(), top_ids.tolist()):
                new_score = score + lp
                new_seq   = seq + [tid]

                if tid == eos_idx:
                    # FIX: length of generated tokens = len(new_seq) - 1
                    # because new_seq contains SOS but not EOS at this point.
                    gen_len = len(new_seq) - 1
                    penalty = ((5 + gen_len) / 6) ** len_penalty
                    completed.append((new_score / penalty, new_seq[1:-1]))
                    # new_seq[1:-1]: strip SOS (first) and the just-added EOS id
                else:
                    candidates.append((new_score, new_seq))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

        if len(completed) >= beam_width:
            break

    if completed:
        completed.sort(key=lambda x: x[0], reverse=True)
        return completed[0][1]

    # Fall back to best partial beam (strip SOS)
    return beams[0][1][1:] if beams else []


# ─── Utilities ────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
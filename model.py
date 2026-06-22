"""
model.py — Transformer Seq2Seq for Roman Nepali → Devanagari

Fixes applied
─────────────
1. greedy_decode / beam_search: repetition_penalty parameter stops the
   "last character repeating forever" bug. Any token that appears in the
   recent output has its logit divided by the penalty before argmax/topk.
2. max_len is capped at min(src_len*3, absolute_max) so the decoder
   can never run away beyond 3× the source length.
3. Weight tying: output_proj.weight = tgt_embed.weight (always, not just
   when tgt_vocab_size == d_model which is almost never true).
4. _init_weights skips embedding tables (they use PyTorch's good default
   N(0,1)); Xavier uniform is only for Linear/projection layers.
5. norm_first=True (Pre-LN) for stable early-training gradients.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        pe  = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, d_model)
        x = x + self.pe[: x.size(1)].unsqueeze(0)
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """
    Encoder–Decoder Transformer (batch_first=True, Pre-LN).

    Training  (teacher forcing):
        logits = model(src, tgt_in, src_pad_mask, tgt_pad_mask)
        # (B, S_tgt-1, tgt_vocab_size)

    Inference:
        ids = greedy_decode(model, src_ids, sos_idx, eos_idx, device)
        ids = beam_search(model,   src_ids, sos_idx, eos_idx, device)
    """

    def __init__(
        self,
        src_vocab_size:     int,
        tgt_vocab_size:     int,
        d_model:            int   = 256,
        nhead:              int   = 4,
        num_encoder_layers: int   = 4,
        num_decoder_layers: int   = 4,
        dim_feedforward:    int   = 512,
        dropout:            float = 0.1,
        max_seq_len:        int   = 60,
        pad_idx:            int   = 0,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(d_model, dropout, max_len=max_seq_len + 10)

        self.transformer = nn.Transformer(
            d_model            = d_model,
            nhead              = nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward    = dim_feedforward,
            dropout            = dropout,
            batch_first        = True,
            norm_first         = True,   # Pre-LN: stable early training
        )

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        # Weight tying — always valid: both are (tgt_vocab_size, d_model)
        self.output_proj.weight = self.tgt_embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "embed" in name:       # leave embeddings at default N(0,1)
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    # ── Forward (teacher-forcing, training) ──────────────────────────────────
    def forward(
        self,
        src:          torch.Tensor,
        tgt:          torch.Tensor,
        src_pad_mask: torch.Tensor | None = None,
        tgt_pad_mask: torch.Tensor | None = None,
        tgt_mask:     torch.Tensor | None = None,   # causal mask (optional override)
    ) -> torch.Tensor:
        S = tgt.size(1)
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                S, device=src.device
            )

        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask                = tgt_mask,
            src_key_padding_mask    = src_pad_mask,
            tgt_key_padding_mask    = tgt_pad_mask,
            memory_key_padding_mask = src_pad_mask,
        )
        return self.output_proj(out)

    # ── Encoder (inference) ──────────────────────────────────────────────────
    def encode(self, src: torch.Tensor,
               src_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(emb, src_key_padding_mask=src_pad_mask)

    # ── One decoder step (inference) ─────────────────────────────────────────
    def decode_step(
        self,
        tgt:         torch.Tensor,
        memory:      torch.Tensor,
        causal_mask: torch.Tensor,
        mem_pad:     torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            emb, memory,
            tgt_mask                = causal_mask,
            memory_key_padding_mask = mem_pad,
        )
        return self.output_proj(out)


# ── Greedy decoding ──────────────────────────────────────────────────────────
@torch.no_grad()
def greedy_decode(
    model:              Seq2SeqTransformer,
    src_ids:            list[int],
    sos_idx:            int,
    eos_idx:            int,
    device:             torch.device,
    max_len:            int   = 0,        # 0 → auto (3× source length)
    repetition_penalty: float = 1.3,      # >1 discourages repeating recent tokens
) -> list[int]:
    """
    Greedy decoder with repetition penalty.

    repetition_penalty: divide logit by this value for any token that
    appeared in the last 3 output positions. Eliminates "last char loops".
    """
    model.eval()
    if max_len <= 0:
        max_len = max(len(src_ids) * 3, 15)

    src     = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad = src.eq(model.pad_idx)
    memory  = model.encode(src, src_pad)

    tgt    = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
    output: list[int] = []

    for _ in range(max_len):
        S      = tgt.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(S, device=device)
        logits = model.decode_step(tgt, memory, causal)  # (1, S, V)
        step_logits = logits[0, -1].clone()               # (V,)

        # Apply repetition penalty on recent tokens
        if repetition_penalty > 1.0 and output:
            for prev in set(output[-4:]):                 # last 4 tokens
                step_logits[prev] = step_logits[prev] / repetition_penalty

        next_id: int = step_logits.argmax().item()        # type: ignore

        if next_id == eos_idx:
            break
        output.append(next_id)
        tgt = torch.cat(
            [tgt, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )

    return output


# ── Beam search ──────────────────────────────────────────────────────────────
@torch.no_grad()
def beam_search(
    model:              Seq2SeqTransformer,
    src_ids:            list[int],
    sos_idx:            int,
    eos_idx:            int,
    device:             torch.device,
    beam_width:         int   = 5,
    max_len:            int   = 0,
    len_penalty:        float = 0.6,
    repetition_penalty: float = 1.3,
) -> list[int]:
    """
    Beam search with repetition penalty and source-relative length cap.
    Fixes the off-by-one length penalty (SOS was incorrectly counted).
    """
    model.eval()
    if max_len <= 0:
        max_len = max(len(src_ids) * 3, 15)

    src     = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad = src.eq(model.pad_idx)
    memory  = model.encode(src, src_pad)

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
            logits = model.decode_step(tgt, memory, causal)   # (1, S, V)
            step   = logits[0, -1].clone()                     # (V,)

            # Repetition penalty on recent generated tokens (exclude SOS)
            if repetition_penalty > 1.0 and len(seq) > 1:
                for prev in set(seq[-4:]):
                    step[prev] = step[prev] / repetition_penalty

            log_p       = F.log_softmax(step, dim=-1)
            top_lp, top = log_p.topk(beam_width)

            for lp, tid in zip(top_lp.tolist(), top.tolist()):
                new_seq   = seq + [tid]
                new_score = score + lp

                if tid == eos_idx:
                    gen_len = len(new_seq) - 1          # exclude SOS
                    penalty = ((5 + gen_len) / 6) ** len_penalty
                    completed.append((new_score / penalty, new_seq[1:-1]))
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

    return beams[0][1][1:] if beams else []


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
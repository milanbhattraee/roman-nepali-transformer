import os
import sys
import json
import torch
import torch.nn.functional as F

from config import Config
from dataset import CharTokenizer, SOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX
from model import Seq2SeqTransformer, beam_search

class Predictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "best_model.pt" if os.path.exists("best_model.pt") else Config.best_model_path()
        self.tokenizer_path = "tokenizers.json" if os.path.exists("tokenizers.json") else Config.tokenizer_path()
        
        self._load_resources()

    def _load_resources(self):
        # Load Checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state', checkpoint)
        
        # Determine architecture from weights
        src_vocab_size = state_dict['src_embed.weight'].shape[0]
        tgt_vocab_size = state_dict['tgt_embed.weight'].shape[0]

        self.model = Seq2SeqTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=Config.D_MODEL,
            nhead=Config.N_HEADS,
            num_encoder_layers=Config.NUM_ENCODER_LAYERS,
            num_decoder_layers=Config.NUM_DECODER_LAYERS,
            dim_feedforward=Config.FF_DIM,
            dropout=Config.DROPOUT,
            max_seq_len=Config.MAX_SEQ_LEN
        ).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Load Tokenizers
        with open(self.tokenizer_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.src_tok = CharTokenizer()
            self.tgt_tok = CharTokenizer()
            self.src_tok.char2idx = data['src']['ch2id']
            self.tgt_tok.idx2char = {int(k): v for k, v in data['tgt']['id2ch'].items()}

    def predict(self, text: str) -> str:
        text = text.lower().strip()
        if not text: return ""

        src_ids = [self.src_tok.char2idx.get(c, UNK_IDX) for c in text]

        with torch.no_grad():
            # Use your native beam search
            indices = beam_search(
                self.model, src_ids, SOS_IDX, EOS_IDX, self.device, 
                beam_width=3, max_len=Config.MAX_SEQ_LEN
            )

        # --- SAFETY DECODING LAYER ---
        final_chars = []
        for i, idx in enumerate(indices):
            if idx in (SOS_IDX, EOS_IDX, PAD_IDX):
                continue
            
            # Repetition Penalty: Prevent triple-char loops (e.g., 'nnn')
            if len(final_chars) >= 2:
                char = self.tgt_tok.idx2char.get(idx, "")
                if char == final_chars[-1] == final_chars[-2]:
                    continue # Skip this char to break the loop
            
            final_chars.append(self.tgt_tok.idx2char.get(idx, ""))

        return "".join(final_chars)

if __name__ == "__main__":
    predictor = Predictor()
    while True:
        txt = input("Enter Roman Nepali word: ").strip()
        if txt.lower() in ['exit', 'quit']: break
        print(f"Devanagari: {predictor.predict(txt)}\n")
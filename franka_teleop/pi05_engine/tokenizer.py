# Adapted from OpenPI PaligemmaTokenizer (Apache-2.0), as included in the pinned
# AEGIS reference openpi/src/openpi/models/tokenizer.py. No JAX or framework imports.
"""Local official PaliGemma SentencePiece tokenizer, using OpenPI's BOS/padding rule."""
from pathlib import Path
import torch


class SentencePiecePromptTokenizer:
    def __init__(self, model_file):
        path = Path(model_file)
        if not path.is_file():
            raise FileNotFoundError(path)
        import sentencepiece
        self.processor = sentencepiece.SentencePieceProcessor(model_file=str(path))
        if self.processor.bos_id() < 0:
            raise ValueError("PaliGemma tokenizer must declare a BOS token")

    def __call__(self, prompts, *, max_length, truncation=True, padding="max_length",
                 padding_side="right", return_tensors="pt"):
        if padding != "max_length" or padding_side != "right" or return_tensors != "pt" or not truncation:
            raise ValueError("this tokenizer implements only the pinned pi05 right-padding recipe")
        tokens, masks = [], []
        for prompt in prompts:
            ids = self.processor.encode(prompt, add_bos=True)[:max_length]
            length = len(ids)
            tokens.append(ids + [0] * (max_length-length))
            masks.append([True]*length + [False]*(max_length-length))
        return {"input_ids":torch.tensor(tokens,dtype=torch.long),
                "attention_mask":torch.tensor(masks,dtype=torch.bool)}

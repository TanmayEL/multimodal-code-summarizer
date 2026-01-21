"""
Very simple embedding helper for RAG stuff.

Right now:
- just hashes tokens into a fixed size vector
- later I can swap this with real OpenAI / HF embeddings

"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class EmbedConfig:
    dim: int = 256


class SimpleEmbedder:
    """
    Idea: for each token, hash it into a random-but-deterministic vector,
    then average all token vectors. This is obviously not great, but it
    lets me build + test the whole RAG pipeline.
    """

    def __init__(self, cfg: EmbedConfig | None = None):
        self.cfg = cfg or EmbedConfig()
        self.dim = self.cfg.dim

    def _token_to_vec(self, tok: str) -> np.ndarray:
        #hash token to get a pseudo-random vector
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        #repeat / trim bytes to match dim
        raw = (h * ((self.dim // len(h)) + 1))[: self.dim]
        arr = np.frombuffer(raw, dtype=np.uint8).astype("float32")
        arr = (arr / 127.5) - 1.0
        return arr

    def embed_text(self, text: str) -> np.ndarray:
        """
        Turn a piece of text into a single vector.

        NOTE: later I probably want to:
        - use a real tokenizer
        - use OpenAI / HF embeddings
        """
        if not text:
            return np.zeros(self.dim, dtype="float32")

        toks: List[str] = text.split()
        vecs = [self._token_to_vec(t) for t in toks]
        mat = np.stack(vecs, axis=0)
        out = mat.mean(axis=0)
        norm = np.linalg.norm(out) + 1e-8
        return (out / norm).astype("float32")




"""
To:
- store chunk text + embeddings
- do cosine-sim search for top-k
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Chunk:
    id: str
    file_path: str
    start_line: int
    end_line: int
    text: str


class SimpleIndex:

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embs: np.ndarray | None = None

    def add(self, chunk: Chunk, emb: np.ndarray) -> None:
        if self.embs is None:
            self.embs = emb.reshape(1, -1)
        else:
            self.embs = np.vstack([self.embs, emb.reshape(1, -1)])
        self.chunks.append(chunk)

    def _cosine_sim(self, q: np.ndarray) -> np.ndarray:
        if self.embs is None or len(self.chunks) == 0:
            return np.zeros(0, dtype="float32")
        q = q.reshape(1, -1)
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        e_norm = self.embs / (np.linalg.norm(self.embs, axis=1, keepdims=True) + 1e-8)
        sims = (q_norm @ e_norm.T).squeeze(0)  # [N]
        return sims.astype("float32")

    def search(self, q_emb: np.ndarray, k: int = 5) -> List[Tuple[Chunk, float]]:
        #return top-k chunks + scores for a query embedding.
        sims = self._cosine_sim(q_emb)
        if sims.size == 0:
            return []
        k = min(k, sims.shape[0])
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        out: List[Tuple[Chunk, float]] = []
        for i in idx:
            out.append((self.chunks[int(i)], float(sims[int(i)])))
        return out

    def save(self, dir_path: str | Path) -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)

        meta = [asdict(c) for c in self.chunks]
        with open(d / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if self.embs is None:
            np.save(d / "embs.npy", np.zeros((0, 1), dtype="float32"))
        else:
            np.save(d / "embs.npy", self.embs.astype("float32"))

    @classmethod
    def load(cls, dir_path: str | Path) -> "SimpleIndex":
        d = Path(dir_path)
        inst = cls()

        chunks_file = d / "chunks.json"
        embs_file = d / "embs.npy"
        if not chunks_file.exists() or not embs_file.exists():
            return inst

        with open(chunks_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        inst.chunks = [Chunk(**c) for c in raw]
        inst.embs = np.load(embs_file).astype("float32")

        return inst




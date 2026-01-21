"""
Retriever that uses the simple index + embedder.

Given some PR-ish info it tries to find relevant chunks from the repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .embedder import SimpleEmbedder
from .index import Chunk, SimpleIndex


@dataclass
class PRInput:
    """
    For now I just use:
    - title
    - description
    - comments (list of strings)
    - diff_text (whole diff as text)
    """

    title: str
    description: str
    comments: List[str]
    diff_text: str


class PRRetriever:

    def __init__(self, index: SimpleIndex, embedder: SimpleEmbedder | None = None):
        self.index = index
        self.embedder = embedder or SimpleEmbedder()

    def _build_query_text(self, pr: PRInput) -> str:
        parts: List[str] = []
        if pr.title:
            parts.append(f"title: {pr.title}")
        if pr.description:
            parts.append(f"desc: {pr.description}")
        if pr.comments:
            parts.append(" ".join(pr.comments[:5]))
        if pr.diff_text:
            #only keep first ~80 lines of diff, rest is probably too much
            lines = pr.diff_text.split("\n")
            parts.append("\n".join(lines[:80]))
        return "\n".join(parts)

    def retrieve(self, pr: PRInput, k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Returns top-k chunks from the index that look relevant for this PR.
        """
        q_txt = self._build_query_text(pr)
        q_emb = self.embedder.embed_text(q_txt)
        return self.index.search(q_emb, k=k)




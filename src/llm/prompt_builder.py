"""
Helpers for building prompts for the LLM.
"""

from __future__ import annotations
from typing import List, Tuple
from src.rag.index import Chunk


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_pr_summary_prompt(
    title: str,
    description: str,
    comments: List[str],
    diff_text: str,
    retrieved_chunks: List[Tuple[Chunk, float]],
    max_ctx_chars: int = 4000,
) -> str:
    #Build a single big prompt string for the LLM,i just cap total characters, good enough for now.
    lines: List[str] = []

    lines.append(
        "You are a senior software engineer doing a code review. "
        "Given a pull request and some related code context, "
        "write a short, clear summary of what this PR changes and why."
    )
    lines.append("")
    lines.append("=== PR META ===")
    lines.append(f"Title: {title}")
    lines.append("")
    lines.append("Description:")
    lines.append(description or "(no description)")
    lines.append("")

    if comments:
        lines.append("Comments (partial):")
        for c in comments[:5]:
            lines.append(f"- {c}")
        lines.append("")

    if diff_text:
        #dont dump the whole diff, just first N lines
        diff_lines = diff_text.split("\n")[:120]
        lines.append("=== DIFF (truncated) ===")
        lines.extend(diff_lines)
        lines.append("")

    if retrieved_chunks:
        lines.append("=== RELATED CODE SNIPPETS (from repo) ===")
        for chunk, score in retrieved_chunks[:5]:
            lines.append(f"# file: {chunk.file_path}  (score={score:.3f})")
            lines.append(chunk.text)
            lines.append("")

    lines.append("=== TASK ===")
    lines.append(
        "Summarize this PR in 3-6 bullet points. "
        "Focus on behavior changes, important refactors, and any risks."
    )

    prompt = "\n".join(lines)
    #so I don't blow up the context window
    return _truncate(prompt, max_ctx_chars)




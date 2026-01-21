"""
Quickscript to build a repo index for RAG.

Usage (from project root, after installing deps):

    python scripts/build_repo_index.py --repo-root . --out-dir data/rag_index

For now this:
- walks the repo
- grabs small text chunks from *.py files (can expand later)
- builds fake embeddings with SimpleEmbedder
- saves a SimpleIndex to disc
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.rag.embedder import SimpleEmbedder
from src.rag.index import Chunk, SimpleIndex


def iter_python_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*.py"):
        #skip venv, tests etc for now if needed
        if "venv" in p.parts:
            continue
        files.append(p)
    return files


def chunk_file(path: Path, max_lines: int = 40) -> List[Chunk]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    lines = txt.split("\n")
    chunks: List[Chunk] = []

    if not lines:
        return chunks

    start = 0
    cid = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        piece = "\n".join(lines[start:end])
        chunk = Chunk(
            id=f"{path}:{cid}",
            file_path=str(path),
            start_line=start + 1,
            end_line=end,
            text=piece,
        )
        chunks.append(chunk)
        cid += 1
        start = end

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--out-dir", type=str, default="data/rag_index")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[index] scanning repo: {root}")
    py_files = iter_python_files(root)
    print(f"[index] found {len(py_files)} python files")

    embedder = SimpleEmbedder()
    index = SimpleIndex()

    for f in py_files:
        chunks = chunk_file(f)
        for ch in chunks:
            emb = embedder.embed_text(ch.text)
            index.add(ch, emb)

    index.save(out_dir)
    print(
        f"[index] saved index with {len(index.chunks)} chunks "
        f"to {out_dir}"
    )


if __name__ == "__main__":
    main()




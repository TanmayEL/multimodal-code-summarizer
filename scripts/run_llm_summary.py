"""
Small CLI to run the LLM-based PR summarizer.
the PR json should look roughly like:
{
  "title": "...",
  "description": "...",
  "comments": ["c1", "c2"],
  "diff_text": "git diff here"
}
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.llm.summarizer import LLMPRSummarizer, PRSummaryInput
from src.rag.embedder import SimpleEmbedder
from src.rag.index import SimpleIndex
from src.rag.retriever import PRRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=str, default="data/rag_index")
    parser.add_argument("--pr-json", type=str, required=True)
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    pr_path = Path(args.pr_json)

    if not pr_path.exists():
        raise FileNotFoundError(f"PR json not found: {pr_path}")

    print(f"[llm] loading index from {index_dir}")
    idx = SimpleIndex.load(index_dir)
    print(f"[llm] index has {len(idx.chunks)} chunks")

    retriever = PRRetriever(idx, SimpleEmbedder())
    summarizer = LLMPRSummarizer(retriever)

    raw = json.loads(pr_path.read_text(encoding="utf-8"))

    pr = PRSummaryInput(
        title=raw.get("title", ""),
        description=raw.get("description", ""),
        comments=raw.get("comments", []),
        diff_text=raw.get("diff_text", ""),
    )

    print("[llm] running summarizer...")
    out = summarizer.summarize(pr)

    print("\n=== SUMMARY ===")
    print(out["summary"])
    print("\n=== DEBUG INFO ===")
    print(f"retrieved chunks: {out['num_retrieved']}")
    print("files used:")
    for f in out["retrieved_files"]:
        print(f"- {f}")


if __name__ == "__main__":
    main()




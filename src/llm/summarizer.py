"""
High level PR summarizer that glues:
- retriever (RAG over repo)
- prompt builder
- LLM client
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any

from src.llm.client import LLMClient, LLMConfig
from src.llm.prompt_builder import build_pr_summary_prompt
from src.rag.retriever import PRInput, PRRetriever


@dataclass
class PRSummaryInput:

    title: str
    description: str
    comments: List[str]
    diff_text: str


class LLMPRSummarizer:
    #brings together the whole LLM-based summarization pipeline.

    def __init__(
        self,
        retriever: PRRetriever,
        llm_client: LLMClient | None = None,
    ):
        self.retriever = retriever
        self.llm = llm_client or LLMClient()

    def summarize(self, pr: PRSummaryInput) -> Dict[str, Any]:
        #runs - RAG retrieval -> prompt construction -> llm generation
        
        #1
        pr_rag = PRInput(
            title=pr.title,
            description=pr.description,
            comments=pr.comments,
            diff_text=pr.diff_text,
        )
        retrieved = self.retriever.retrieve(pr_rag, k=5)

        #2
        prompt = build_pr_summary_prompt(
            title=pr.title,
            description=pr.description,
            comments=pr.comments,
            diff_text=pr.diff_text,
            retrieved_chunks=retrieved,
        )

        #3
        summary = self.llm.generate(prompt)

        return {
            "summary": summary,
            "num_retrieved": len(retrieved),
            "retrieved_files": list({c.file_path for (c, _) in retrieved}),
        }




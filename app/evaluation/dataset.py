"""Evaluation dataset loading/saving.

Question schema::

    {
      "question": "What is RAG?",
      "expected_video": "VIDEO_ID" | ["VIDEO_ID", ...],   # optional
      "expected_keywords": ["retrieval", "generation"]      # optional
    }
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class EvalQuestion:
    question: str
    expected_video: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalQuestion":
        ev: Union[str, list, None] = d.get("expected_video")
        if isinstance(ev, str):
            videos = [ev]
        elif isinstance(ev, list):
            videos = [str(x) for x in ev]
        else:
            videos = []
        kws = [str(k).lower() for k in d.get("expected_keywords", [])]
        return cls(
            question=str(d["question"]),
            expected_video=videos,
            expected_keywords=kws,
        )


def load_questions(path: str) -> List[EvalQuestion]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Evaluation file must contain a JSON list of questions.")
    return [EvalQuestion.from_dict(d) for d in data]


def save_questions(path: str, questions: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(questions, fh, ensure_ascii=False, indent=2)


SAMPLE_QUESTIONS = [
    {
        "question": "What is retrieval augmented generation?",
        "expected_video": "",
        "expected_keywords": ["retrieval", "generation", "context"],
    },
    {
        "question": "Why are embeddings used in a RAG system?",
        "expected_video": "",
        "expected_keywords": ["embedding", "vector", "similarity"],
    },
    {
        "question": "What is a vector database used for?",
        "expected_video": "",
        "expected_keywords": ["vector", "database", "index"],
    },
    {
        "question": "How does reranking improve retrieval?",
        "expected_video": "",
        "expected_keywords": ["rerank", "relevance", "candidates"],
    },
    {
        "question": "What is chunking and why does it matter?",
        "expected_video": "",
        "expected_keywords": ["chunk", "overlap", "context"],
    },
]

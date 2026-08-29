"""Prompt templates for the RAG pipeline.

Every template is grounding-first: the model is told to use only the supplied
context, to cite the numbered sources, to preserve uncertainty, and to refuse
rather than invent. Context blocks are rendered with a stable ``[n]`` index that
maps 1:1 to the citation list the app builds separately.
"""
from __future__ import annotations

from typing import List

from app.retrieval.base import RetrievalResult
from app.utils import format_timestamp

INSUFFICIENT_EVIDENCE = (
    "I couldn't find enough information in the indexed videos to answer this "
    "confidently."
)

QA_SYSTEM = (
    "You are a precise research assistant answering questions strictly from "
    "transcript excerpts of YouTube videos.\n"
    "Rules:\n"
    "1. Use ONLY the numbered context below. Do not use outside knowledge.\n"
    "2. Cite every factual claim with its source marker, e.g. [1], [2].\n"
    "3. If the context does not contain the answer, reply exactly: "
    f'"{INSUFFICIENT_EVIDENCE}"\n'
    "4. Preserve uncertainty; never fabricate facts, numbers, or timestamps.\n"
    "5. Be concise and well-structured."
)


def render_context(results: List[RetrievalResult]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        ts = format_timestamp(r.start_time)
        lines.append(f"[{i}] (video: {r.title} @ {ts})\n{r.text}")
    return "\n\n".join(lines)


def build_qa_messages(
    query: str,
    results: List[RetrievalResult],
    history: List[dict] | None = None,
) -> List[dict]:
    context = render_context(results)
    messages: List[dict] = [{"role": "system", "content": QA_SYSTEM}]
    if history:
        # Include recent turns for conversational follow-ups.
        messages.extend(history[-6:])
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above, citing sources like [1], [2]."
    )
    messages.append({"role": "user", "content": user})
    return messages


def build_comparison_messages(query: str, grouped: dict) -> List[dict]:
    blocks = []
    idx = 1
    index_map = {}
    for title, results in grouped.items():
        block_lines = [f"### Video: {title}"]
        for r in results:
            ts = format_timestamp(r.start_time)
            block_lines.append(f"[{idx}] (@ {ts}) {r.text}")
            index_map[idx] = r
            idx += 1
        blocks.append("\n".join(block_lines))
    context = "\n\n".join(blocks)
    system = (
        "You are comparing how several YouTube videos treat a topic. "
        "Use ONLY the provided excerpts. Cite claims with [n]. "
        "Explicitly note agreements, differences, and any gaps. "
        f'If the excerpts are insufficient, say: "{INSUFFICIENT_EVIDENCE}"'
    )
    user = (
        f"Excerpts grouped by video:\n{context}\n\n"
        f"Task: {query}\n\n"
        "Produce a structured comparison, citing each claim with [n]."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_summary_messages(title: str, transcript_context: str) -> List[dict]:
    system = (
        "You summarise educational YouTube videos from their transcript. "
        "Use ONLY the transcript. Do not invent content."
    )
    user = (
        f"Video title: {title}\n\n"
        f"Transcript excerpts (with timestamps):\n{transcript_context}\n\n"
        "Produce:\n"
        "1. Overview (2-3 sentences)\n"
        "2. Key concepts (bullet list)\n"
        "3. Important takeaways (bullet list)\n"
        "Keep it grounded in the transcript."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_chapters_messages(title: str, timed_context: str) -> List[dict]:
    system = (
        "You create chapter markers for a YouTube video from timestamped "
        "transcript lines. Every chapter MUST use a timestamp that appears in "
        "the provided lines. Output one chapter per line as `SECONDS | Title`. "
        "Do not invent timestamps."
    )
    user = (
        f"Video title: {title}\n\n"
        f"Timestamped transcript lines:\n{timed_context}\n\n"
        "Return 5-10 chapters, each as `SECONDS | Short Title`, "
        "ordered by time, using only timestamps present above."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_learning_messages(title: str, transcript_context: str) -> List[dict]:
    system = (
        "You are a tutor creating study material from a video transcript. "
        "Use ONLY the transcript. Ground every point in it."
    )
    user = (
        f"Video title: {title}\n\n"
        f"Transcript excerpts (with timestamps):\n{transcript_context}\n\n"
        "Produce:\n"
        "1. Beginner-friendly explanation (short paragraph)\n"
        "2. Key concepts to remember (bullets)\n"
        "3. Most important points (bullets)\n"
        "Keep everything grounded in the transcript."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_quiz_messages(title: str, timed_context: str, n: int = 5) -> List[dict]:
    system = (
        "You write quiz questions grounded in a video transcript. "
        "Return STRICT JSON only: a list of objects with keys "
        '"question", "answer", and "source_seconds". '
        "source_seconds MUST be a timestamp (in seconds) that appears in the "
        "provided lines. Do not add prose, markdown, or code fences."
    )
    user = (
        f"Video title: {title}\n\n"
        f"Timestamped transcript lines:\n{timed_context}\n\n"
        f"Write {n} quiz questions as strict JSON as described."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

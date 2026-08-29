"""Prompt templates for the RAG pipeline."""
from __future__ import annotations

from typing import List

from app.retrieval.base import RetrievalResult
from app.utils import format_timestamp

INSUFFICIENT_EVIDENCE = (
    "I couldn't find enough information in the indexed videos to answer this "
    "confidently."
)

QA_SYSTEM = (
    "Answer from the transcript excerpts only.\n"
    "Use the numbered context and cite sources like [1], [2] when you make a factual claim.\n"
    "If the context does not support an answer, reply exactly: "
    f'"{INSUFFICIENT_EVIDENCE}"\n'
    "Keep the response short and clear."
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
        "Compare the excerpts from the selected videos. "
        "Use only the provided text and cite claims with [n]. "
        "Call out similarities, differences, and missing information. "
        f'If the excerpts are not enough, say: "{INSUFFICIENT_EVIDENCE}"'
    )
    user = (
        f"Excerpts grouped by video:\n{context}\n\n"
        f"Task: {query}\n\n"
        "Produce a structured comparison, citing each claim with [n]."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_summary_messages(title: str, transcript_context: str) -> List[dict]:
    system = (
        "Summarize the video using only the transcript. Do not invent details."
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
        "Create chapter markers from the transcript timestamps. "
        "Only use timestamps that appear in the provided transcript. "
        "Return one chapter per line as `SECONDS | Title`."
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
        "Turn the transcript into simple study notes. Keep every point grounded in the video text."
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
        "Write quiz questions from the transcript. Return JSON only with "
        '"question", "answer", and "source_seconds" fields. '
        "Use only timestamps that appear in the transcript."
    )
    user = (
        f"Video title: {title}\n\n"
        f"Timestamped transcript lines:\n{timed_context}\n\n"
        f"Write {n} quiz questions as strict JSON as described."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

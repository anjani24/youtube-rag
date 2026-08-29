"""Generate study notes and quiz questions from video transcript chunks."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.config import Config
from app.llm.provider import BaseLLM
from app.rag.prompts import build_learning_messages, build_quiz_messages
from app.summarization.summarizer import _ordered_chunks, _plain_context, _timed_context
from app.storage.vector_store import BaseVectorStore
from app.utils import format_timestamp, log, youtube_timestamp_url


@dataclass
class QuizQuestion:
    question: str
    answer: str
    source_seconds: float

    def render(self, video_id: str) -> str:
        return (
            f"Q: {self.question}\n"
            f"A: {self.answer}\n"
            f"Source: {format_timestamp(self.source_seconds)} "
            f"{youtube_timestamp_url(video_id, self.source_seconds)}"
        )


@dataclass
class LearningPack:
    video_id: str
    title: str
    notes: str
    quiz: List[QuizQuestion] = field(default_factory=list)

    def render(self) -> str:
        parts = [f"Learning Mode — {self.title}", "=" * 60, "", self.notes, ""]
        if self.quiz:
            parts.append("Quiz:")
            parts.append("")
            for i, q in enumerate(self.quiz, start=1):
                parts.append(f"{i}. {q.render(self.video_id)}")
                parts.append("")
        return "\n".join(parts)


def _extract_json(text: str) -> Optional[list]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _parse_quiz(text: str, valid_starts: List[float]) -> List[QuizQuestion]:
    data = _extract_json(text)
    if not data:
        return []
    questions: List[QuizQuestion] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if not q or not a:
            continue
        try:
            secs = float(item.get("source_seconds", 0))
        except (TypeError, ValueError):
            secs = 0.0
        snapped = min(valid_starts, key=lambda s: abs(s - secs)) if valid_starts else secs
        questions.append(QuizQuestion(question=q, answer=a, source_seconds=snapped))
    return questions


def learning_mode(
    video_id: str,
    config: Config,
    store: BaseVectorStore,
    llm: BaseLLM,
    n_questions: int = 5,
) -> Optional[LearningPack]:
    chunks = _ordered_chunks(store, video_id)
    if not chunks:
        return None
    title = chunks[0]["metadata"].get("title", video_id)
    valid_starts = [float(c["metadata"].get("start_time", 0)) for c in chunks]

    try:
        notes = llm.chat(build_learning_messages(title, _plain_context(chunks)))
    except Exception as exc:  # noqa: BLE001
        log.error("Learning notes LLM call failed: %s", exc)
        notes = "(Study notes unavailable — LLM call failed.)"

    quiz: List[QuizQuestion] = []
    try:
        raw = llm.chat(build_quiz_messages(title, _timed_context(chunks), n=n_questions))
        quiz = _parse_quiz(raw, valid_starts)
    except Exception as exc:  # noqa: BLE001
        log.warning("Quiz LLM call failed: %s", exc)

    return LearningPack(video_id=video_id, title=title, notes=notes, quiz=quiz)

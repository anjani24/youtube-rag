"""Shared retrieval result type."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RetrievalResult:
    id: str
    text: str
    metadata: dict
    score: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)

    @property
    def video_id(self) -> str:
        return self.metadata.get("video_id", "")

    @property
    def title(self) -> str:
        return self.metadata.get("title", self.video_id)

    @property
    def start_time(self) -> float:
        return float(self.metadata.get("start_time", 0.0))

    @property
    def end_time(self) -> float:
        return float(self.metadata.get("end_time", 0.0))

    @property
    def chunk_index(self) -> int:
        return int(self.metadata.get("chunk_index", 0))

    @classmethod
    def from_record(cls, rec: dict, score: float = 0.0) -> "RetrievalResult":
        return cls(
            id=rec["id"],
            text=rec["text"],
            metadata=rec.get("metadata", {}),
            score=score,
        )

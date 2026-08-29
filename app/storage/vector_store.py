"""Vector store abstraction.

A thin interface over the persistence layer so the rest of the app never talks
to Chroma directly. This is what makes "swap Chroma for pgvector later" a
localized change: implement the same interface and update the factory.

Records are plain dicts::

    {"id": str, "text": str, "metadata": {...}, "distance": float | None}
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from app.chunking.chunker import Chunk
from app.config import Config
from app.utils import log


class BaseVectorStore:
    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        raise NotImplementedError

    def query(
        self,
        embedding: List[float],
        n_results: int,
        video_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        raise NotImplementedError

    def get_chunks(self, video_ids: Optional[List[str]] = None) -> List[dict]:
        raise NotImplementedError

    def has_video(self, video_id: str) -> bool:
        raise NotImplementedError

    def list_videos(self) -> List[dict]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def delete_video(self, video_id: str) -> None:
        raise NotImplementedError


def _chunk_records(chunks: List[Chunk]):
    ids = [c.chunk_id for c in chunks]
    docs = [c.text for c in chunks]
    metas = [
        {
            "video_id": c.video_id,
            "title": c.title,
            "start_time": float(c.start_time),
            "end_time": float(c.end_time),
            "chunk_index": int(c.chunk_index),
        }
        for c in chunks
    ]
    return ids, docs, metas


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, config: Config) -> None:
        try:
            import chromadb  # type: ignore
            from chromadb.config import Settings  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "chromadb is not installed. Run `pip install chromadb`."
            ) from exc
        self._client = chromadb.PersistentClient(
            path=config.chroma_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        # We supply our own embeddings, so no embedding_function is needed.
        self._col = self._client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return
        ids, docs, metas = _chunk_records(chunks)
        self._col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    def query(self, embedding, n_results, video_ids=None) -> List[dict]:
        where = {"video_id": {"$in": video_ids}} if video_ids else None
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=max(1, n_results),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i in range(len(ids)):
            out.append(
                {
                    "id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i],
                    "distance": dists[i],
                }
            )
        return out

    def get_chunks(self, video_ids=None) -> List[dict]:
        where = {"video_id": {"$in": video_ids}} if video_ids else None
        res = self._col.get(where=where, include=["documents", "metadatas"])
        out = []
        ids = res.get("ids", [])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        for i in range(len(ids)):
            out.append({"id": ids[i], "text": docs[i], "metadata": metas[i]})
        return out

    def has_video(self, video_id: str) -> bool:
        res = self._col.get(where={"video_id": video_id}, limit=1)
        return len(res.get("ids", [])) > 0

    def list_videos(self) -> List[dict]:
        res = self._col.get(include=["metadatas"])
        return _aggregate_videos(res.get("metadatas", []))

    def count(self) -> int:
        return self._col.count()

    def delete_video(self, video_id: str) -> None:
        self._col.delete(where={"video_id": video_id})


class InMemoryVectorStore(BaseVectorStore):
    """Non-persistent store using cosine similarity. Handy for tests."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self._records: Dict[str, dict] = {}

    def add_chunks(self, chunks, embeddings) -> None:
        ids, docs, metas = _chunk_records(chunks)
        for i, cid in enumerate(ids):
            self._records[cid] = {
                "id": cid,
                "text": docs[i],
                "metadata": metas[i],
                "embedding": embeddings[i],
            }

    @staticmethod
    def _cosine(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    def query(self, embedding, n_results, video_ids=None) -> List[dict]:
        scored = []
        for rec in self._records.values():
            if video_ids and rec["metadata"]["video_id"] not in video_ids:
                continue
            sim = self._cosine(embedding, rec["embedding"])
            scored.append(
                {
                    "id": rec["id"],
                    "text": rec["text"],
                    "metadata": rec["metadata"],
                    "distance": 1.0 - sim,  # mimic cosine distance
                }
            )
        scored.sort(key=lambda r: r["distance"])
        return scored[:n_results]

    def get_chunks(self, video_ids=None) -> List[dict]:
        out = []
        for rec in self._records.values():
            if video_ids and rec["metadata"]["video_id"] not in video_ids:
                continue
            out.append(
                {"id": rec["id"], "text": rec["text"], "metadata": rec["metadata"]}
            )
        return out

    def has_video(self, video_id: str) -> bool:
        return any(
            r["metadata"]["video_id"] == video_id for r in self._records.values()
        )

    def list_videos(self) -> List[dict]:
        return _aggregate_videos([r["metadata"] for r in self._records.values()])

    def count(self) -> int:
        return len(self._records)

    def delete_video(self, video_id: str) -> None:
        for cid in [
            cid
            for cid, r in self._records.items()
            if r["metadata"]["video_id"] == video_id
        ]:
            del self._records[cid]


def _aggregate_videos(metadatas: List[dict]) -> List[dict]:
    videos: Dict[str, dict] = {}
    for m in metadatas:
        vid = m.get("video_id")
        if vid not in videos:
            videos[vid] = {"video_id": vid, "title": m.get("title", vid), "chunk_count": 0}
        videos[vid]["chunk_count"] += 1
    return sorted(videos.values(), key=lambda v: v["title"].lower())


def get_vector_store(config: Config) -> BaseVectorStore:
    if config.vector_store.lower() == "memory":
        return InMemoryVectorStore(config)
    return ChromaVectorStore(config)

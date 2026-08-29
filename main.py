"""Main CLI entry point for the project."""
from __future__ import annotations

import os
import sys
from typing import List, Optional

from app.config import Config
from app.evaluation.dataset import (
    SAMPLE_QUESTIONS,
    load_questions,
    save_questions,
)
from app.evaluation.evaluator import Evaluator, format_report
from app.ingestion.ingest import ingest_video
from app.learning.quiz import learning_mode
from app.rag.qa import RAGPipeline
from app.rag.topics import search_topic
from app.summarization.summarizer import summarize_video
from app.utils import setup_logging

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _RICH = True
    console = Console()
except Exception:  # pragma: no cover - rich optional
    _RICH = False
    console = None


def _print(msg: str = "") -> None:
    if _RICH:
        console.print(msg)
    else:
        print(msg)


def _rule(title: str = "") -> None:
    if _RICH:
        console.rule(title)
    else:
        print("-" * 60)
        if title:
            print(title)


def _status(msg: str):
    if _RICH:
        return console.status(f"[bold cyan]{msg}")

    class _Dummy:
        def __enter__(self):
            print(msg)
            return self

        def __exit__(self, *a):
            return False

    return _Dummy()


class App:
    def __init__(self) -> None:
        self.config = Config.from_env()
        setup_logging(self.config.log_level)
        self.pipeline: Optional[RAGPipeline] = None
        self.history: List[dict] = []

    def _ensure_pipeline(self) -> RAGPipeline:
        if self.pipeline is None:
            with _status("Loading models and vector store…"):
                self.pipeline = RAGPipeline(self.config)
        return self.pipeline

    # ------------------------------------------------------------------ #
    def header(self) -> None:
        pipe = self._ensure_pipeline()
        videos = pipe.store.list_videos()
        chunks = pipe.store.count()
        banner = (
            "╔══════════════════════════════════════════╗\n"
            "║            YouTube Video Search          ║\n"
            "╚══════════════════════════════════════════╝\n"
            f"\nIndexed videos: {len(videos)}   Chunks: {chunks}"
        )
        if _RICH:
            console.print(Panel.fit(banner, style="bold cyan"))
        else:
            print(banner)

    def menu(self) -> str:
        _print()
        _print("1. Add YouTube video")
        _print("2. List indexed videos")
        _print("3. Ask a question")
        _print("4. Search topics")
        _print("5. Summarize video")
        _print("6. Compare videos")
        _print("7. Learning mode")
        _print("8. Run evaluation")
        _print("9. Exit")
        return input("\nChoose an option: ").strip()

    # ------------------------------------------------------------------ #
    def add_video(self) -> None:
        url = input("Enter YouTube URL: ").strip()
        if not url:
            _print("No URL provided.")
            return
        pipe = self._ensure_pipeline()
        with _status("Fetching transcript, chunking, embedding…"):
            result = ingest_video(url, self.config, pipe.store, pipe.embedder)
        if result.ok and result.duplicate:
            _print(f"[yellow]{result.message}[/yellow]" if _RICH else result.message)
        elif result.ok:
            _print(f"[green]{result.message}[/green]" if _RICH else result.message)
        else:
            _print(f"[red]{result.message}[/red]" if _RICH else f"Error: {result.message}")

    def list_videos(self) -> None:
        pipe = self._ensure_pipeline()
        videos = pipe.store.list_videos()
        if not videos:
            _print("No videos indexed yet.")
            return
        if _RICH:
            table = Table(title="Indexed Videos")
            table.add_column("#", justify="right")
            table.add_column("Video ID")
            table.add_column("Title")
            table.add_column("Chunks", justify="right")
            for i, v in enumerate(videos, 1):
                table.add_row(str(i), v["video_id"], v["title"], str(v["chunk_count"]))
            console.print(table)
        else:
            for i, v in enumerate(videos, 1):
                print(f"{i}. {v['video_id']} | {v['title']} | {v['chunk_count']} chunks")

    def _select_videos(self, prompt: str, allow_all: bool = True) -> Optional[List[str]]:
        pipe = self._ensure_pipeline()
        videos = pipe.store.list_videos()
        if not videos:
            _print("No videos indexed yet.")
            return None
        self.list_videos()
        raw = input(prompt).strip()
        if not raw and allow_all:
            return None  # all videos
        ids = []
        for token in raw.replace(",", " ").split():
            if token.isdigit() and 1 <= int(token) <= len(videos):
                ids.append(videos[int(token) - 1]["video_id"])
            else:
                ids.append(token)  # allow raw ids
        return ids or None

    def ask(self) -> None:
        pipe = self._ensure_pipeline()
        video_ids = self._select_videos(
            "Restrict to which videos? (numbers/ids, blank = all): "
        )
        _print("Conversation mode — type 'exit' to return to the menu.")
        while True:
            q = input("\nYou: ").strip()
            if q.lower() in {"exit", "quit", "back"}:
                break
            if not q:
                continue
            with _status("Retrieving and reasoning…"):
                answer = pipe.answer_question(q, video_ids=video_ids, history=self.history)
            _print("\n" + answer.render())
            self.history.append({"role": "user", "content": q})
            self.history.append({"role": "assistant", "content": answer.text})

    def search_topics(self) -> None:
        pipe = self._ensure_pipeline()
        video_ids = self._select_videos(
            "Search which videos? (numbers/ids, blank = all): "
        )
        query = input("Topic to search: ").strip()
        if not query:
            return
        with _status("Searching…"):
            hits = search_topic(pipe, query, video_ids=video_ids, top_k=self.config.top_k)
        if not hits:
            _print("No matching moments found.")
            return
        _print(f"\nTopic: {query}\n")
        for i, hit in enumerate(hits, 1):
            _print(hit.render(i))
            _print("")

    def summarize(self) -> None:
        pipe = self._ensure_pipeline()
        ids = self._select_videos("Summarize which video? (number/id): ", allow_all=False)
        if not ids:
            return
        with _status("Summarizing…"):
            summary = summarize_video(ids[0], self.config, pipe.store, pipe.llm)
        if summary is None:
            _print("Video not found or has no chunks.")
            return
        _print("\n" + summary.render())

    def compare(self) -> None:
        pipe = self._ensure_pipeline()
        ids = self._select_videos(
            "Compare which videos? (numbers/ids, at least two): ", allow_all=False
        )
        if not ids or len(ids) < 2:
            _print("Select at least two videos.")
            return
        query = input("What should the comparison focus on? ").strip() or (
            "Compare how these videos explain the main topic."
        )
        with _status("Comparing…"):
            answer = pipe.compare_videos(query, ids)
        _print("\n" + answer.render())

    def learn(self) -> None:
        pipe = self._ensure_pipeline()
        ids = self._select_videos("Learning mode for which video? (number/id): ", allow_all=False)
        if not ids:
            return
        with _status("Building study material and quiz…"):
            pack = learning_mode(ids[0], self.config, pipe.store, pipe.llm)
        if pack is None:
            _print("Video not found or has no chunks.")
            return
        _print("\n" + pack.render())

    def evaluate(self) -> None:
        pipe = self._ensure_pipeline()
        path = os.path.join(self.config.evaluation_dir, "questions.json")
        if not os.path.exists(path):
            save_questions(path, SAMPLE_QUESTIONS)
            _print(f"Created sample evaluation file at {path}.")
        questions = load_questions(path)
        if pipe.store.count() == 0:
            _print("Index at least one video before running evaluation.")
            return
        with _status("Running evaluation across retrieval strategies…"):
            evaluator = Evaluator(pipe, self.config)
            report = evaluator.run(questions, k=self.config.top_k)
        _print("\n" + format_report(report))

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        while True:
            self.header()
            choice = self.menu()
            actions = {
                "1": self.add_video,
                "2": self.list_videos,
                "3": self.ask,
                "4": self.search_topics,
                "5": self.summarize,
                "6": self.compare,
                "7": self.learn,
                "8": self.evaluate,
            }
            if choice == "9":
                _print("Goodbye!")
                return
            action = actions.get(choice)
            if action:
                try:
                    action()
                except KeyboardInterrupt:
                    _print("\nCancelled.")
                except Exception as exc:  # noqa: BLE001 - keep the CLI alive
                    _print(f"[red]Error: {exc}[/red]" if _RICH else f"Error: {exc}")
            else:
                _print("Invalid option.")


def _run_evaluate_cli() -> int:
    app = App()
    app.evaluate()
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1].lower() == "evaluate":
        return _run_evaluate_cli()
    try:
        App().run()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

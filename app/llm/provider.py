"""LLM provider abstraction (chat completions).

Default is any OpenAI-compatible endpoint (set ``OPENAI_BASE_URL`` to point at
a compatible host). A ``FakeLLM`` is provided for offline tests and demos so the
whole pipeline runs without network access or keys.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from app.config import Config
from app.utils import with_retries

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


class BaseLLM:
    def chat(self, messages: List[Message], **kwargs) -> str:
        raise NotImplementedError


class OpenAILLM(BaseLLM):
    def __init__(self, config: Config) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai is not installed. Run `pip install openai`.") from exc
        config.require_llm_key()
        self._client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            timeout=config.request_timeout,
        )
        self._cfg = config

    def chat(self, messages, **kwargs) -> str:
        @with_retries(max_retries=self._cfg.max_retries, backoff=self._cfg.retry_backoff)
        def _call():
            return self._client.chat.completions.create(
                model=kwargs.get("model", self._cfg.llm_model),
                messages=messages,
                temperature=kwargs.get("temperature", self._cfg.llm_temperature),
                max_tokens=kwargs.get("max_tokens", self._cfg.llm_max_tokens),
            )

        resp = _call()
        return (resp.choices[0].message.content or "").strip()


class FakeLLM(BaseLLM):
    """Deterministic offline LLM.

    Accepts an optional ``responder`` callable for tests. By default it echoes a
    grounded-looking answer built from the supplied context so the end-to-end
    pipeline (including citation extraction) is exercisable without a real API.
    """

    def __init__(self, responder: Optional[Callable[[List[Message]], str]] = None) -> None:
        self._responder = responder

    def chat(self, messages, **kwargs) -> str:
        if self._responder is not None:
            return self._responder(messages)
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        # Return the first ~60 words of the provided context as a stand-in answer.
        snippet = " ".join(user.split()[:60])
        return f"[FAKE-LLM answer based on provided context] {snippet}"


def get_llm(config: Config) -> BaseLLM:
    if config.llm_provider.lower() == "fake":
        return FakeLLM()
    return OpenAILLM(config)

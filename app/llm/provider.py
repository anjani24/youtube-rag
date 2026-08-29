"""LLM provider abstraction (chat completions).

Supports multiple free and paid providers:
* ``GroqLLM``       -- free tier at groq.com, very fast (recommended)
* ``OllamaLLM``     -- local, fully offline, requires Ollama installation
* ``OpenAILLM``     -- paid, but can use OpenAI-compatible endpoints
* ``FakeLLM``       -- deterministic, for offline tests and demos

Default is Groq (free, fast, no credit card needed).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from app.config import Config
from app.utils import with_retries, log

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


class BaseLLM:
    def chat(self, messages: List[Message], **kwargs) -> str:
        raise NotImplementedError


class GroqLLM(BaseLLM):
    """Groq API LLM (free tier with generous limits)."""

    def __init__(self, config: Config) -> None:
        try:
            from groq import Groq  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("groq is not installed. Run `pip install groq`.") from exc
        config.require_llm_key()
        self._client = Groq(api_key=config.groq_api_key)
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


class OllamaLLM(BaseLLM):
    """Ollama LLM (local, fully offline)."""

    def __init__(self, config: Config) -> None:
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("requests is not installed. Run `pip install requests`.") from exc
        config.require_llm_key()
        self._base_url = config.ollama_base_url or "http://localhost:11434"
        self._cfg = config
        # Test connection
        try:
            requests.get(f"{self._base_url}/api/tags", timeout=5)
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            ) from e

    def chat(self, messages, **kwargs) -> str:
        import requests  # type: ignore

        @with_retries(max_retries=self._cfg.max_retries, backoff=self._cfg.retry_backoff)
        def _call():
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": kwargs.get("model", self._cfg.llm_model),
                    "messages": messages,
                    "stream": False,
                },
                timeout=self._cfg.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()

        result = _call()
        return (result.get("message", {}).get("content") or "").strip()


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
    provider = config.llm_provider.lower()
    if provider == "fake":
        return FakeLLM()
    elif provider == "groq":
        return GroqLLM(config)
    elif provider == "ollama":
        return OllamaLLM(config)
    elif provider == "openai":
        return OpenAILLM(config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Choose from: groq, ollama, openai, fake")

"""
Pluggable LLM interface.

Every loop pattern reasons by calling an `LLM.generate(prompt)` — never a
specific provider SDK directly — so swapping in a real provider later
(Anthropic, OpenAI, ...) doesn't touch loop logic. `MockLLM` and
`ScriptedLLM` below let the harness, tools, and all three loop patterns be
built and tested deterministically without a live API key; wiring a real
provider is a matter of adding one more `LLM` subclass here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


@dataclass
class LLMResponse:
    text: str
    total_tokens: int


def _estimate_tokens(*texts: str) -> int:
    """Whitespace-split word count as a cheap, provider-agnostic token
    proxy. Good enough for comparing relative cost across loop patterns
    on the same mock backend; swap for a real tokenizer once a real
    provider is wired in."""
    return sum(len(t.split()) for t in texts)


class LLM(ABC):
    """Base class every loop pattern calls through. A loop must never
    depend on a concrete LLM implementation — only on this interface —
    so the same loop code runs unchanged against a mock or a real
    provider."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        ...


class MockLLM(LLM):
    """Deterministic LLM stand-in driven by a user-supplied responder
    function: `prompt -> response text`. Use this to script exact
    reasoning/action/critique text for a given prompt shape in tests and
    structural eval runs."""

    def __init__(self, responder: Callable[[str], str]):
        self._responder = responder

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        text = self._responder(prompt)
        return LLMResponse(text=text, total_tokens=_estimate_tokens(prompt, text))


class ScriptedLLM(LLM):
    """Deterministic LLM stand-in that returns a fixed queue of responses
    in order, one per call, regardless of prompt content. Useful when a
    loop's prompt content is incidental and only the sequence of
    responses matters (e.g. testing an N-iteration control loop)."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._calls: list[str] = []

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        self._calls.append(prompt)
        if not self._responses:
            raise RuntimeError("ScriptedLLM exhausted: no more responses queued")
        text = self._responses.pop(0)
        return LLMResponse(text=text, total_tokens=_estimate_tokens(prompt, text))

    @property
    def call_count(self) -> int:
        return len(self._calls)

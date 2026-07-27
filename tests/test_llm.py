from __future__ import annotations

from dataclasses import dataclass

import pytest

from harness.llm import GroqLLM, MockLLM, ScriptedLLM


def test_mock_llm_calls_responder_with_prompt():
    seen = []

    def responder(prompt: str) -> str:
        seen.append(prompt)
        return "answer"

    llm = MockLLM(responder)
    result = llm.generate("what is 2+2?")

    assert result.text == "answer"
    assert seen == ["what is 2+2?"]
    assert result.total_tokens > 0


def test_scripted_llm_returns_responses_in_order():
    llm = ScriptedLLM(["first", "second"])

    r1 = llm.generate("prompt a")
    r2 = llm.generate("prompt b")

    assert r1.text == "first"
    assert r2.text == "second"
    assert llm.call_count == 2


def test_scripted_llm_raises_when_exhausted():
    llm = ScriptedLLM(["only"])
    llm.generate("prompt")

    with pytest.raises(RuntimeError):
        llm.generate("prompt again")


@dataclass
class _StubMessage:
    content: str


@dataclass
class _StubChoice:
    message: _StubMessage


@dataclass
class _StubUsage:
    total_tokens: int


@dataclass
class _StubResponse:
    choices: list[_StubChoice]
    usage: _StubUsage


class _StubCompletions:
    def __init__(self, response: _StubResponse):
        self._response = response
        self.last_call: dict | None = None

    def create(self, **kwargs):
        self.last_call = kwargs
        return self._response


class _StubChat:
    def __init__(self, completions: _StubCompletions):
        self.completions = completions


class _StubGroqClient:
    def __init__(self, response: _StubResponse):
        self.chat = _StubChat(_StubCompletions(response))


def test_groq_llm_generate_uses_injected_client():
    response = _StubResponse(
        choices=[_StubChoice(message=_StubMessage(content="the answer"))],
        usage=_StubUsage(total_tokens=42),
    )
    client = _StubGroqClient(response)
    llm = GroqLLM(model="llama-3.3-70b-versatile", client=client)

    result = llm.generate("what is 2+2?")

    assert result.text == "the answer"
    assert result.total_tokens == 42
    call = client.chat.completions.last_call
    assert call["model"] == "llama-3.3-70b-versatile"
    assert call["messages"] == [{"role": "user", "content": "what is 2+2?"}]


def test_groq_llm_falls_back_to_estimate_when_usage_missing():
    response = _StubResponse(
        choices=[_StubChoice(message=_StubMessage(content="short answer"))],
        usage=None,
    )
    llm = GroqLLM(client=_StubGroqClient(response))

    result = llm.generate("a prompt")

    assert result.text == "short answer"
    assert result.total_tokens > 0


def test_groq_llm_requires_api_key_without_client(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)

    with pytest.raises(RuntimeError):
        GroqLLM()

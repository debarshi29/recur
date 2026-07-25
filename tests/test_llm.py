from __future__ import annotations

import pytest

from harness.llm import MockLLM, ScriptedLLM


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

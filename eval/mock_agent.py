"""
Deterministic mock reasoning policy shared by all three loop patterns for
structural eval runs in the absence of a live LLM API key.

This is NOT a real reasoning agent -- it is a simple, stateless
(prompt-in, text-out) heuristic that issues one search per task and
returns a naive extraction from the top search result as its final
answer. It exists so eval/run_comparison.py can exercise every loop
pattern's control flow (tool calls, iteration counts, termination) end to
end and log real structural metrics (iterations, tool_calls, tokens,
wall-clock) to MLflow. Accuracy numbers produced against this policy are
NOT meaningful as a reasoning benchmark -- see docs/writeup.md for that
caveat once a real provider is wired into harness/llm.py.
"""
from __future__ import annotations

import json
import re

from harness.llm import MockLLM

_QUESTION_RE = re.compile(r"Question:\s*(.+)")


def _extract_question(prompt: str) -> str:
    m = _QUESTION_RE.search(prompt)
    return m.group(1).strip() if m else ""


def _has_observation(prompt: str) -> bool:
    return "Observation:" in prompt


def _last_observation(prompt: str) -> str:
    parts = prompt.split("Observation:")
    return parts[-1].split("\n\n")[0].strip() if len(parts) > 1 else ""


def _naive_answer(observation: str) -> str:
    first_sentence = observation.split(".")[0].strip()
    return first_sentence[:120]


def policy(prompt: str) -> str:
    """One search, then a naive one-sentence extraction as the final
    answer -- exercises the full act/observe/answer control flow without
    claiming to be a real reasoning policy. Reflection's critique prompts
    are always accepted (GOOD) so the mock never revises -- a mock critic
    can't meaningfully judge answer quality any better than the mock
    drafter produced it."""
    if "Candidate answer:" in prompt and "Critique the candidate answer" in prompt:
        return "GOOD"
    if not _has_observation(prompt):
        question = _extract_question(prompt)
        kwargs_json = json.dumps({"query": question})
        return f"ACTION: web_search | {kwargs_json}"
    observation = _last_observation(prompt)
    return f"FINAL_ANSWER: {_naive_answer(observation)}"


def make_mock_llm() -> MockLLM:
    return MockLLM(policy)

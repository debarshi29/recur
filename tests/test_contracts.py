from __future__ import annotations

import pytest

from harness.contracts import (
    LoopRunConfig,
    Task,
    Tool,
    exact_match_scorer,
)


class AlwaysOkTool(Tool):
    name = "always_ok"
    description = "Always succeeds."

    def _run(self, **kwargs):
        return "ok"


class AlwaysFailTool(Tool):
    name = "always_fail"
    description = "Always raises."

    def _run(self, **kwargs):
        raise RuntimeError("boom")


class FlakyTool(Tool):
    """Fails N times, then succeeds — used to test retry-until-success."""

    name = "flaky"
    description = "Fails a fixed number of times before succeeding."

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    def _run(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"flaky failure #{self.calls}")
        return "recovered"


class _DummyLoop:
    """Minimal concrete AgentLoop stand-in to exercise _call_tool without
    committing to any real loop pattern's reasoning logic."""

    def __init__(self, tools, config=None):
        from harness.contracts import AgentLoop

        class _Impl(AgentLoop):
            name = "dummy"

            def run(self, task, scorer=exact_match_scorer):
                raise NotImplementedError

        self._impl = _Impl(tools, config)

    def call_tool(self, name, **kwargs):
        return self._impl._call_tool(name, **kwargs)


# ---------------------------------------------------------------------------
# Tool wrapping
# ---------------------------------------------------------------------------

def test_tool_run_wraps_success():
    result = AlwaysOkTool().run()
    assert result.ok is True
    assert result.output == "ok"
    assert result.error is None
    assert result.latency_ms >= 0


def test_tool_run_wraps_failure_as_data_not_exception():
    result = AlwaysFailTool().run()
    assert result.ok is False
    assert result.output is None
    assert "RuntimeError" in result.error
    assert "boom" in result.error


# ---------------------------------------------------------------------------
# Retry / backoff via AgentLoop._call_tool
# ---------------------------------------------------------------------------

def test_call_tool_retries_until_success():
    flaky = FlakyTool(fail_times=2)
    loop = _DummyLoop(
        [flaky],
        LoopRunConfig(tool_max_retries=3, tool_backoff_base_s=0.0),
    )
    result = loop.call_tool("flaky")
    assert result.ok is True
    assert result.output == "recovered"
    assert flaky.calls == 3


def test_call_tool_gives_up_after_max_retries():
    fail_tool = AlwaysFailTool()
    loop = _DummyLoop(
        [fail_tool],
        LoopRunConfig(tool_max_retries=2, tool_backoff_base_s=0.0),
    )
    result = loop.call_tool("always_fail")
    assert result.ok is False
    assert "boom" in result.error


def test_call_tool_unknown_tool_returns_error_result_not_exception():
    loop = _DummyLoop([AlwaysOkTool()], LoopRunConfig(tool_backoff_base_s=0.0))
    result = loop.call_tool("does_not_exist")
    assert result.ok is False
    assert result.error == "Unknown tool: does_not_exist"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "predicted,gold,expected_ok",
    [
        ("Paris", "Paris", True),
        ("paris", "Paris", True),
        ("  Paris  ", "Paris", True),
        ("London", "Paris", False),
    ],
)
def test_exact_match_scorer(predicted, gold, expected_ok):
    ok, score = exact_match_scorer(predicted, gold)
    assert ok is expected_ok
    assert score == (1.0 if expected_ok else 0.0)


# ---------------------------------------------------------------------------
# Task defaults
# ---------------------------------------------------------------------------

def test_task_defaults():
    task = Task(id="t1", question="What is 2+2?", gold_answer="4")
    assert task.expected_hops == 1
    assert task.metadata == {}

from __future__ import annotations

import time

from harness.contracts import LoopRunConfig, Task, Tool
from harness.llm import MockLLM
from harness.reliability import CircuitBreaker, Timeout, run_graph
from harness.tools import CalculatorTool
from loops.react_loop import ReActLoop

# ---------------------------------------------------------------------------
# CircuitBreaker unit tests
# ---------------------------------------------------------------------------

def test_circuit_breaker_closed_initially():
    breaker = CircuitBreaker(threshold=3)
    assert breaker.is_open("some_tool") is False


def test_circuit_breaker_opens_after_threshold_consecutive_failures():
    breaker = CircuitBreaker(threshold=3)
    for _ in range(3):
        breaker.record_failure("flaky")
    assert breaker.is_open("flaky") is True


def test_circuit_breaker_resets_on_success():
    breaker = CircuitBreaker(threshold=3)
    breaker.record_failure("flaky")
    breaker.record_failure("flaky")
    breaker.record_success("flaky")
    breaker.record_failure("flaky")
    assert breaker.is_open("flaky") is False  # only 1 consecutive failure since the reset


# ---------------------------------------------------------------------------
# Timeout unit tests
# ---------------------------------------------------------------------------

def test_timeout_not_expired_immediately():
    timeout = Timeout(timeout_s=10.0)
    assert timeout.expired() is False


def test_timeout_expires_after_deadline():
    timeout = Timeout(timeout_s=0.01)
    time.sleep(0.02)
    assert timeout.expired() is True


# ---------------------------------------------------------------------------
# Circuit breaker integrated into AgentLoop._call_tool
# ---------------------------------------------------------------------------

class AlwaysFailTool(Tool):
    name = "always_fail"
    description = "Always raises."

    def __init__(self):
        self.calls = 0

    def _run(self, **kwargs):
        self.calls += 1
        raise RuntimeError("persistent failure")


def test_persistently_failing_tool_stops_being_retried_within_a_run():
    """A tool that always fails must not be retried forever across many
    loop iterations -- after circuit_breaker_threshold whole calls (each
    already exhausting tool_max_retries) have failed in a row, further
    calls short-circuit without touching the tool again."""
    tool = AlwaysFailTool()
    config = LoopRunConfig(tool_max_retries=1, tool_backoff_base_s=0.0, circuit_breaker_threshold=2)
    loop = ReActLoop([tool], MockLLM(lambda p: "ACTION: always_fail | {}"), config=config)

    # 2 whole calls to exhaust the breaker threshold (each attempts up to
    # tool_max_retries+1 = 2 real tool invocations).
    loop._call_tool("always_fail")
    loop._call_tool("always_fail")
    assert tool.calls == 4  # 2 calls x 2 attempts each

    result = loop._call_tool("always_fail")
    assert result.ok is False
    assert "Circuit breaker open" in result.error
    assert tool.calls == 4  # unchanged -- the tool itself was never invoked again


def test_react_loop_with_persistently_failing_tool_completes_without_hanging():
    tool = AlwaysFailTool()
    config = LoopRunConfig(
        max_iterations=20,
        tool_max_retries=0,
        tool_backoff_base_s=0.0,
        circuit_breaker_threshold=2,
    )
    llm = MockLLM(lambda p: "ACTION: always_fail | {}")
    loop = ReActLoop([tool], llm, config=config)
    task = Task(id="t1", question="q", gold_answer="a")

    result = loop.run(task)

    assert result.iterations == 20  # ran to max_iterations, never crashed or hung
    # the tool stopped being called well before max_iterations once the
    # breaker opened (threshold=2 whole calls, each 1 attempt)
    assert tool.calls == 2


# ---------------------------------------------------------------------------
# Timeout enforcement integrated into run_graph via a loop run
# ---------------------------------------------------------------------------

def test_react_loop_stops_early_on_timeout_instead_of_running_to_max_iterations():
    def slow_responder(prompt: str) -> str:
        time.sleep(0.05)
        return "ACTION: calculator | {\"expression\": \"1+1\"}"

    config = LoopRunConfig(max_iterations=1000, timeout_s=0.12, tool_backoff_base_s=0.0)
    loop = ReActLoop([CalculatorTool()], MockLLM(slow_responder), config=config)
    task = Task(id="t1", question="never resolves", gold_answer="anything")

    start = time.perf_counter()
    result = loop.run(task)
    elapsed = time.perf_counter() - start

    assert result.iterations < 1000  # timeout cut the run short, not max_iterations
    assert elapsed < 5.0  # generous bound; would be ~50s+ if it ran to max_iterations


# ---------------------------------------------------------------------------
# Checkpoint resume via run_graph directly on a toy graph
# ---------------------------------------------------------------------------

def test_run_graph_resumes_from_checkpoint_after_a_simulated_crash():
    from typing import TypedDict

    from langgraph.graph import END, StateGraph

    from harness.reliability import new_checkpointer

    class S(TypedDict):
        steps: list
        fail_once: bool

    calls = {"n": 0}

    def node_a(state: S) -> S:
        state["steps"].append("a")
        return state

    def node_b(state: S) -> S:
        calls["n"] += 1
        if state.get("fail_once") and calls["n"] == 1:
            raise RuntimeError("simulated crash")
        state["steps"].append("b")
        return state

    graph_builder = StateGraph(S)
    graph_builder.add_node("a", node_a)
    graph_builder.add_node("b", node_b)
    graph_builder.set_entry_point("a")
    graph_builder.add_edge("a", "b")
    graph_builder.add_edge("b", END)
    graph = graph_builder.compile(checkpointer=new_checkpointer())

    initial_state: S = {"steps": [], "fail_once": True}

    try:
        run_graph(graph, initial_state, thread_id="thread-1", timeout_s=10.0, recursion_limit=10)
        raised = False
    except RuntimeError:
        raised = True
    assert raised is True  # the simulated crash propagated, as a real crash would

    # "process restart": call run_graph again with a *fresh* initial_state
    # dict, exactly as a new run() invocation would build one -- it must
    # resume from the checkpoint (node_a not re-run) rather than restart.
    final_state = run_graph(
        graph,
        {"steps": [], "fail_once": True},
        thread_id="thread-1",
        timeout_s=10.0,
        recursion_limit=10,
    )

    assert final_state["steps"] == ["a", "b"]  # 'a' appears once, not twice
    assert calls["n"] == 2  # node_b: one failed attempt, one successful retry

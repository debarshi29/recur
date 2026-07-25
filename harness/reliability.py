"""
Shared reliability primitives every loop pattern inherits unchanged:
LangGraph checkpointing (resume a crashed run from its last completed
step instead of restarting), a per-tool circuit breaker (stop retrying a
tool that keeps failing within a single run), and an enforced loop-level
wall-clock timeout. These live here, not per-pattern, so every loop gets
identical production behavior -- see AgentLoop._call_tool in contracts.py
for the circuit breaker's integration point and run_graph below for
checkpointing/timeout.
"""
from __future__ import annotations

import time
from collections import defaultdict

from langgraph.checkpoint.memory import MemorySaver


def new_checkpointer() -> MemorySaver:
    """Fresh in-memory checkpointer, one per AgentLoop instance. Every
    task run made through that instance can be resumed by thread_id
    (task.id) for the instance's lifetime."""
    return MemorySaver()


class CircuitBreaker:
    """Per-tool-name consecutive-failure counter scoped to a single loop
    instance. Once a tool has failed `threshold` whole calls (each call
    already having exhausted its own retries) in a row, further calls to
    that tool short-circuit immediately without touching the tool or
    sleeping through backoff -- this is what stops a persistently-failing
    tool from being retried into a hang across many loop iterations."""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self._consecutive_failures: dict[str, int] = defaultdict(int)

    def record_success(self, tool_name: str) -> None:
        self._consecutive_failures[tool_name] = 0

    def record_failure(self, tool_name: str) -> None:
        self._consecutive_failures[tool_name] += 1

    def is_open(self, tool_name: str) -> bool:
        return self._consecutive_failures[tool_name] >= self.threshold


class Timeout:
    """Wall-clock budget checked between LangGraph steps. Not a
    preemptive interrupt -- a node already running finishes -- but it
    stops a slow-converging or stuck loop promptly at the next step
    boundary rather than always running to max_iterations."""

    def __init__(self, timeout_s: float):
        self._deadline = time.perf_counter() + timeout_s

    def expired(self) -> bool:
        return time.perf_counter() >= self._deadline


def run_graph(graph, initial_state: dict, thread_id: str, timeout_s: float, recursion_limit: int) -> dict:
    """Every loop pattern calls this instead of `self._graph.invoke(...)`
    directly, so checkpoint-resume and timeout enforcement are identical
    across patterns. Resumes from an existing checkpoint for `thread_id`
    (rather than restarting from `initial_state`) if one is pending --
    e.g. after a previous call raised mid-run -- and stops advancing the
    graph as soon as `timeout_s` elapses, returning whatever state was
    last reached."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    snapshot = graph.get_state(config)
    resuming = bool(snapshot.next)
    stream_input = None if resuming else initial_state

    timeout = Timeout(timeout_s)
    last_state = snapshot.values if resuming else initial_state
    for state in graph.stream(stream_input, config=config, stream_mode="values"):
        last_state = state
        if timeout.expired():
            break
    return last_state

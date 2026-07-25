"""
Shared contracts for the loop-engineering harness.

Every loop pattern (ReAct, Reflection, Plan-Execute, ...) is built against
these interfaces. This is the seam that makes head-to-head comparison
meaningful: same tools, same task definition, same scoring — only the
control-loop architecture differs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import time
import uuid


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    ok: bool
    output: Any
    error: str | None = None
    latency_ms: float = 0.0


class Tool(ABC):
    """Base class for any tool a loop can call (search, calculator, retrieval)."""

    name: str
    description: str

    @abstractmethod
    def _run(self, **kwargs) -> Any:
        ...

    def run(self, **kwargs) -> ToolResult:
        """Wraps _run with timing + error isolation so a bad tool call
        never crashes a loop — it becomes an observation the loop can
        reason about and recover from."""
        start = time.perf_counter()
        try:
            output = self._run(**kwargs)
            return ToolResult(
                ok=True,
                output=output,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:  # noqa: BLE001 — intentional: tool failures are data
            return ToolResult(
                ok=False,
                output=None,
                error=f"{type(e).__name__}: {e}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single multi-hop research/QA question the loops will be run against."""

    id: str
    question: str
    gold_answer: str
    expected_hops: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    loop_name: str
    predicted_answer: str
    correct: bool
    score: float  # 0-1, allows partial credit scoring functions
    iterations: int
    tool_calls: int
    total_tokens: int
    wall_clock_ms: float
    trace: list[dict] = field(default_factory=list)  # step-by-step record


ScoringFn = Callable[[str, str], tuple[bool, float]]  # (predicted, gold) -> (correct, score)


def exact_match_scorer(predicted: str, gold: str) -> tuple[bool, float]:
    ok = predicted.strip().lower() == gold.strip().lower()
    return ok, 1.0 if ok else 0.0


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

class LoopStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    MAX_ITERATIONS = "max_iterations"
    ERROR = "error"


@dataclass
class LoopRunConfig:
    max_iterations: int = 10
    timeout_s: float = 60.0
    tool_max_retries: int = 2
    tool_backoff_base_s: float = 0.5


class AgentLoop(ABC):
    """Every loop pattern (ReAct, Reflection, Plan-Execute) implements this.

    The contract is intentionally narrow: given a task and a set of tools,
    produce a TaskResult. Everything about *how* the loop reasons is internal
    to the subclass — that's the variable under test.
    """

    name: str

    def __init__(self, tools: list[Tool], config: LoopRunConfig | None = None):
        self.tools = {t.name: t for t in tools}
        self.config = config or LoopRunConfig()

    @abstractmethod
    def run(self, task: Task, scorer: ScoringFn = exact_match_scorer) -> TaskResult:
        ...

    def _call_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Retry-with-backoff wrapper every loop should route tool calls
        through, so reliability behavior is consistent across patterns."""
        tool = self.tools.get(tool_name)
        if tool is None:
            return ToolResult(ok=False, output=None, error=f"Unknown tool: {tool_name}")

        last_result: ToolResult | None = None
        for attempt in range(self.config.tool_max_retries + 1):
            last_result = tool.run(**kwargs)
            if last_result.ok:
                return last_result
            if attempt < self.config.tool_max_retries:
                time.sleep(self.config.tool_backoff_base_s * (2 ** attempt))
        return last_result  # last failure, after exhausting retries

    @staticmethod
    def new_run_id() -> str:
        return str(uuid.uuid4())

"""
Concrete tool implementations. Swap/extend these — the loops only depend
on the Tool interface in contracts.py, not on these implementations.
"""
from __future__ import annotations

from harness.contracts import Tool


class WebSearchTool(Tool):
    """Stub search tool. Wire this to your real search backend
    (Tavily, SerpAPI, a local index, etc.) for the actual eval run —
    keeping it swappable is the point."""

    name = "web_search"
    description = "Search for information relevant to a query. Returns top snippets."

    def __init__(self, backend=None):
        self._backend = backend

    def _run(self, query: str) -> str:
        if self._backend is None:
            raise NotImplementedError(
                "Wire a real search backend before running evals — "
                "this stub exists so the harness is testable without network calls."
            )
        return self._backend(query)


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate a basic arithmetic expression."

    def _run(self, expression: str) -> float:
        # Deliberately restricted eval — no builtins, digits/operators only.
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            raise ValueError(f"Disallowed characters in expression: {expression!r}")
        return eval(expression, {"__builtins__": {}}, {})  # noqa: S307 — sandboxed


class ScratchpadTool(Tool):
    """Lets a loop persist intermediate notes across iterations —
    useful for plan-execute and reflection patterns that need to
    track state beyond the raw conversation."""

    name = "scratchpad"
    description = "Write a note to the scratchpad, or read it back with action='read'."

    def __init__(self):
        self._notes: list[str] = []

    def _run(self, action: str, content: str | None = None) -> str:
        if action == "write":
            if content is None:
                raise ValueError("content required for write")
            self._notes.append(content)
            return "noted"
        elif action == "read":
            return "\n".join(self._notes) if self._notes else "(empty)"
        raise ValueError(f"Unknown action: {action}")
